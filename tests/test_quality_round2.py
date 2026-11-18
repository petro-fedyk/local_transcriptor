"""Quality defaults, prompt style, CPU cache, and live recovery tests."""
from types import SimpleNamespace
import json
import time

import numpy as np
import pytest

from voice_to_speech import engine
from voice_to_speech.config import Settings, SettingsStore
from voice_to_speech.engine import ModelHandle, transcribe_file, transcribe_live
from voice_to_speech.live import LiveSession
from voice_to_speech.textfilter import is_hallucination


UK_STYLE = 'Добре, домовились. Тоді зробимо так, як обговорювали, — і перевіримо результат.'
EN_STYLE = "Okay, sounds good. Let's do it the way we discussed, and then check the results."


@pytest.mark.parametrize(('language', 'prefix'), [('uk', UK_STYLE), ('en', EN_STYLE), (None, '')])
def test_prompt_style_precedes_glossary_for_live_and_file(language, prefix):
    calls = []

    class Model:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return iter([]), SimpleNamespace(language=language)

    handle = ModelHandle(Model(), 'test', 'cpu')
    transcribe_live(handle, np.zeros(16000, np.float32), language, 'STM32 Олена recent words')
    list(transcribe_file(handle, 'speech.wav', language, 'STM32 Олена', clip_timestamps=[12.5]))
    expected_live = f'{prefix} STM32 Олена recent words'.strip()
    expected_file = f'{prefix} STM32 Олена'.strip()
    assert calls[0]['initial_prompt'] == expected_live
    assert calls[1]['initial_prompt'] == expected_file
    assert calls[1]['clip_timestamps'] == [12.5]
    list(transcribe_file(handle, 'speech.wav', language, 'STM32'))
    assert 'clip_timestamps' not in calls[2]


def test_explicit_cpu_model_is_cached_separately(tmp_path, monkeypatch):
    folder = tmp_path / 'large-v3'
    folder.mkdir()
    (folder / 'model.bin').touch()
    calls = []

    def factory(path, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(engine, 'WhisperModel', factory)
    manager = engine.ModelManager(tmp_path, device='cuda')
    cuda = manager.get('large-v3')
    cpu = manager.get('large-v3', device='cpu')
    assert cuda is not cpu
    assert manager.get('large-v3') is cuda
    assert manager.get('large-v3', device='cpu') is cpu
    assert [call['device'] for call in calls] == ['cuda', 'cpu']
    assert calls[1]['compute_type'] == 'int8'
    assert manager.states()['large-v3@cpu']['device'] == 'cpu'
    manager.unload('large-v3', device='cpu')
    assert 'large-v3@cpu' not in manager.states()
    assert manager.get('large-v3') is cuda


def test_cpu_request_reuses_default_cpu_instance(tmp_path, monkeypatch):
    folder = tmp_path / 'large-v3'
    folder.mkdir()
    (folder / 'model.bin').touch()
    calls = []
    monkeypatch.setattr(engine, 'WhisperModel', lambda *args, **kwargs: calls.append(kwargs) or object())
    manager = engine.ModelManager(tmp_path, device='cpu')
    default = manager.get('large-v3')
    assert manager.get('large-v3', device='cpu') is default
    assert len(calls) == 1


def test_resumed_file_keeps_absolute_segment_offsets():
    segment = SimpleNamespace(start=12.5, end=13.0, text='Текст')

    class Model:
        def transcribe(self, audio, **kwargs):
            assert kwargs['clip_timestamps'] == [12.5]
            return iter([segment]), SimpleNamespace(duration=20)

    result = list(transcribe_file(ModelHandle(Model(), 'test', 'cpu'), 'speech.wav', 'uk', clip_timestamps=[12.5]))
    assert result[0][0] is segment
    assert (result[0][0].start, result[0][0].end) == (12.5, 13.0)


def test_uk_default_respects_existing_saved_language(tmp_path):
    path = tmp_path / 'settings.json'
    assert Settings().language == 'uk'
    assert SettingsStore(path).snapshot().language == 'uk'
    path.write_text(json.dumps({'language': 'auto'}), encoding='utf-8')
    assert SettingsStore(path).snapshot().language == 'auto'


def test_you_is_only_short_exact_hallucination():
    assert is_hallucination(' You! ', 5.99)
    assert not is_hallucination('You', 6)
    assert not is_hallucination('You are welcome.', 1)


class SettingsBox:
    def snapshot(self):
        return Settings()


class Capture:
    def __init__(self, source, device, on_block):
        self.source = source
        self.on_block = on_block

    def start(self):
        pass

    def stop(self):
        pass


class Segmenter:
    def push(self, samples, offset):
        return [SimpleNamespace(audio=samples, start_s=offset, end_s=offset + .5)]

    def flush(self):
        return []


@pytest.mark.parametrize('failures', [1, 2])
def test_live_retries_oom_once_then_continues(tmp_path, monkeypatch, failures):
    import voice_to_speech.live as live_module
    trims = []
    monkeypatch.setattr(live_module, 'trim_memory', lambda: trims.append(True))

    class Engine:
        def __init__(self):
            self.calls = 0

        def get(self, name):
            return SimpleNamespace(name=name, device='cuda')

        def transcribe_live(self, handle, audio, language, prompt):
            self.calls += 1
            if self.calls <= failures:
                raise RuntimeError('CUDA failed with error OUT OF MEMORY')
            return SimpleNamespace(text='Реальна розмова.', language='uk')

    events = []
    model = Engine()
    live = LiveSession(tmp_path, SettingsBox(), model, Capture, Segmenter, events.append)
    sid, _ = live.start(True, False)
    session = live._session
    capture = session['captures']['system']
    capture.on_block('system', np.zeros(8000, np.float32), session['mono_start'] + .5)
    if failures == 2:
        capture.on_block('system', np.zeros(8000, np.float32), session['mono_start'] + 1)
    deadline = time.monotonic() + 2
    while not any(event.get('type') == 'transcript' for event in events):
        assert time.monotonic() < deadline
        time.sleep(.01)
    live.stop()
    live.shutdown(2)
    assert model.calls == failures + 1
    assert trims
    assert any(event.get('level') == 'warning' for event in events)
    assert any(event.get('level') == 'error' for event in events) is (failures == 2)
    assert 'Реальна розмова.' in (tmp_path / sid / 'transcript.txt').read_text()

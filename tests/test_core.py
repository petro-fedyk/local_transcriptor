from types import SimpleNamespace
import threading
import numpy as np
import pytest
from voice_to_speech.segmenter import StreamingSegmenter
from voice_to_speech.textfilter import EchoDeduper, clean_text, is_hallucination
from voice_to_speech import engine
from voice_to_speech.engine import ModelHandle, transcribe_live, transcribe_file


def test_streaming_segmenter_emits_voice_after_silence():
    def vad(audio):
        voiced = np.flatnonzero(audio)
        return [{'start': int(voiced[0]), 'end': int(voiced[-1])+1}] if len(voiced) else []
    segmenter = StreamingSegmenter(sample_rate=1000, vad_fn=vad, min_silence_s=.2, pad_s=.1, min_speech_s=.1)
    assert not segmenter.push(np.ones(400, dtype=np.float32), 7)
    segments = segmenter.push(np.zeros(600, dtype=np.float32), 7.4)
    assert len(segments) == 1
    assert segments[0].start_s == 7
    assert segments[0].end_s == 7.5
    assert len(segments[0].audio) == 500
    assert not segmenter.flush()


def test_text_cleanup_hallucination_and_echo():
    assert clean_text(' Hello\n world ') == 'Hello world'
    assert clean_text('hello world hello world hello world hello world') == 'hello world'
    assert clean_text('...!?') == ''
    assert is_hallucination('Thanks for watching!', 5)
    assert not is_hallucination('Thanks for watching!', 6)
    assert is_hallucination('Visit amara.org please', 30)
    deduper = EchoDeduper(window_s=2)
    deduper.remember('Hello world', 5, 6)
    assert deduper.is_echo('hello world!', 7, 8)
    assert not deduper.is_echo('hello world', 10, 11)


def test_transcribe_live_filters_and_sets_options():
    good = SimpleNamespace(start=0., end=.5, text=' hi ')
    noise = SimpleNamespace(start=.5, end=1., text='noise', no_speech_prob=.9, avg_logprob=-2)
    class Model:
        def transcribe(self, audio, **kwargs):
            assert kwargs['beam_size'] == 5
            assert kwargs['condition_on_previous_text'] is False
            assert kwargs['without_timestamps'] is True
            return iter([good, noise]), SimpleNamespace(language='en')
    result = transcribe_live(ModelHandle(Model(), 'test', 'cpu'), np.zeros(10), 'en')
    assert result.text == 'hi'
    assert result.language == 'en'
    assert result.segments_info == [{'start':0., 'end':.5, 'text':'hi'}]


def test_file_releases_lock_between_yields():
    lock = threading.Lock()
    class Model:
        def transcribe(self, path, **kwargs):
            assert lock.locked()
            assert kwargs['vad_parameters']['speech_pad_ms'] == 400
            def segments():
                assert lock.locked()
                yield 'first'
                assert lock.locked()
                yield 'second'
            return segments(), 'info'
    output = transcribe_file(ModelHandle(Model(), 'test', 'cpu', lock), 'x.wav')
    assert next(output) == ('first', 'info')
    assert not lock.locked()
    assert next(output) == ('second', 'info')


def test_missing_model_never_invokes_whisper(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('must not download a missing model')
    monkeypatch.setattr(engine, 'WhisperModel', forbidden)
    manager = engine.ModelManager(tmp_path, device='cpu')
    with pytest.raises(engine.ModelMissingError, match='vts models'):
        manager.get('large-v3')
    assert manager.states()['large-v3']['state'] == 'error'


def test_local_model_options_and_cache(tmp_path, monkeypatch):
    folder = tmp_path / 'large-v3'
    folder.mkdir()
    (folder/'model.bin').touch()
    calls = []
    def factory(path, **kwargs):
        calls.append((path, kwargs))
        return object()
    monkeypatch.setattr(engine, 'WhisperModel', factory)
    manager = engine.ModelManager(tmp_path, device='cpu')
    assert manager.get('large-v3') is manager.get('large-v3')
    assert len(calls) == 1
    assert calls[0][0] == str(folder)
    assert calls[0][1]['local_files_only'] is True
    assert manager.states()['large-v3']['state'] == 'ready'


def test_capture_float32_blocks_with_injected_process():
    import io
    from voice_to_speech.audio import CaptureSource
    received = []
    calls = []
    class Process:
        def __init__(self):
            self.stdout = io.BytesIO(np.ones(8000, dtype='<f4').tobytes())
        def terminate(self):
            pass
        def wait(self, timeout):
            pass
    def factory(command, **kwargs):
        calls.append(command)
        return Process()
    capture = CaptureSource(source='mic', device='test-source', popen_factory=factory,
        callback=lambda source, samples, timestamp: (received.append((source, samples, timestamp)), capture._stop.set()))
    capture._run()
    assert '--device=test-source' in calls[0]
    assert len(received[0][1]) == 8000
    assert received[0][1].dtype == np.float32


def test_list_sources_separates_monitors_microphones_apps(monkeypatch):
    from voice_to_speech import audio
    def pactl(kind):
        if kind == 'sources':
            return [{'name':'sink.monitor', 'description':'System', 'monitor_of_sink':0},
                    {'name':'mic', 'description':'Mic', 'monitor_of_sink':4294967295}]
        return [{'index':4, 'properties':{'application.name':'Browser'}}]
    monkeypatch.setattr(audio, '_pactl', pactl)
    sources = audio.list_sources()
    assert sources['error'] is None
    assert sources['monitors'][0]['name'] == 'sink.monitor'
    assert sources['microphones'][0]['name'] == 'mic'
    assert sources['apps'][0]['index'] == 4


def test_segmenter_vad_cadence_hard_limit_and_silence_trim():
    calls = []
    def vad(audio):
        calls.append(len(audio))
        return [{'start':0, 'end':len(audio)}]
    segmenter = StreamingSegmenter(sample_rate=100, vad_fn=vad, max_segment_s=2, pad_s=0)
    assert not segmenter.push(np.ones(50), 0)
    assert calls == []
    assert not segmenter.push(np.ones(50), .5)
    assert calls == [100]
    segments = segmenter.push(np.ones(150), 1)
    assert len(segments) == 1
    assert len(segments[0].audio) == 200
    assert len(segmenter._audio) == 50
    silent = StreamingSegmenter(sample_rate=100, vad_fn=lambda audio: [])
    silent.push(np.zeros(400), 0)
    assert len(silent._audio) == 50
    assert silent._start == 3.5


def test_large_cuda_failure_prefers_turbo_cpu(tmp_path, monkeypatch):
    for name in ('large-v3', 'large-v3-turbo'):
        folder = tmp_path/name
        folder.mkdir()
        (folder/'model.bin').touch()
    def factory(path, **kwargs):
        if kwargs['device'] == 'cuda':
            raise RuntimeError('cuInit failed with 999')
        assert path.endswith('large-v3-turbo')
        return object()
    monkeypatch.setattr(engine, 'WhisperModel', factory)
    manager = engine.ModelManager(tmp_path, device='cuda')
    with pytest.warns(RuntimeWarning):
        handle = manager.get('large-v3')
    assert (handle.name, handle.device) == ('large-v3-turbo', 'cpu')
    assert 'sudo rmmod nvidia_uvm' in manager.cuda_hint


def _amplitude_vad(audio):
    """Return all nonzero runs for deterministic segmenter tests."""
    active = np.asarray(audio) != 0
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    return [{'start': int(start), 'end': int(end)}
            for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def test_segmenter_silence_flush_and_short_speech():
    silent = StreamingSegmenter(sample_rate=100, vad_fn=_amplitude_vad)
    assert silent.push(np.zeros(400, np.float32), 0) == []
    assert silent.flush() == []
    short = StreamingSegmenter(sample_rate=100, vad_fn=_amplitude_vad, min_speech_s=.4)
    assert short.push(np.r_[np.ones(20), np.zeros(100)].astype(np.float32), 4) == []
    assert short.flush() == []


def test_segmenter_flush_keeps_relative_offsets():
    segmenter = StreamingSegmenter(sample_rate=100, vad_fn=_amplitude_vad, pad_s=.1)
    audio = np.r_[np.zeros(20), np.ones(60)].astype(np.float32)
    assert segmenter.push(audio, 5) == []
    result = segmenter.flush()
    assert len(result) == 1
    assert result[0].start_s == pytest.approx(5.1)
    assert result[0].end_s == pytest.approx(5.8)


def test_segmenter_max_cut_at_gap_then_hard_cut():
    gap = StreamingSegmenter(sample_rate=100, vad_fn=_amplitude_vad,
                             max_segment_s=2, pad_s=0, min_silence_s=.8)
    audio = np.r_[np.ones(90), np.zeros(30), np.ones(90), np.zeros(40)].astype(np.float32)
    result = gap.push(audio, 0)
    assert len(result) == 1
    assert result[0].end_s == pytest.approx(.9)
    assert len(gap._audio) == 160
    hard = StreamingSegmenter(sample_rate=100, vad_fn=_amplitude_vad,
                              max_segment_s=2, pad_s=0)
    result = hard.push(np.ones(250, np.float32), 0)
    assert len(result) == 1
    assert result[0].end_s == pytest.approx(2)
    assert len(hard._audio) == 50

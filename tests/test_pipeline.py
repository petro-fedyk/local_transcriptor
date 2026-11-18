from pathlib import Path
from types import SimpleNamespace
import json
import threading
import time
import wave

import numpy as np

from voice_to_speech.live import LiveSession
from voice_to_speech.jobs import JobQueue
from voice_to_speech.config import Settings


class SettingsBox:
    def __init__(self, **kwargs):
        self.value = Settings(**kwargs)
    def snapshot(self):
        return self.value


class FakeEngine:
    def __init__(self):
        self.handle = SimpleNamespace(name="fake", device="cpu", lock=threading.RLock())
        self.entered = threading.Event()
        self.gate = None
        self.fail = False
        self.calls = []
    def get(self, name):
        return self.handle
    def transcribe_live(self, handle, audio, language, prompt):
        self.calls.append(threading.current_thread().name)
        return SimpleNamespace(text="Реальна розмова", language="uk")
    def transcribe_file(self, handle, path, language, glossary):
        self.entered.set()
        if self.gate:
            self.gate.wait(2)
        yield SimpleNamespace(start=.25, end=1.5, text=Path(path).stem), SimpleNamespace(language="uk", duration=2)
        if self.fail:
            raise RuntimeError("тестова помилка")
        yield SimpleNamespace(start=1.5, end=2, text="Завершення"), SimpleNamespace(language="uk", duration=2)


class FakeSegmenter:
    def push(self, samples, offset):
        return [SimpleNamespace(audio=samples.copy(), start_s=offset, end_s=offset + len(samples) / 16000)]
    def flush(self):
        return []


class FakeCapture:
    instances = []
    def __init__(self, source, device, on_block):
        self.source, self.device, self.on_block = source, device, on_block
        self.stopped = False
        self.instances.append(self)
    def start(self):
        self.on_block(self.source, np.ones(8000, np.float32) * .2, time.monotonic())
    def stop(self):
        self.stopped = True


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.01)


def wav_file(path, seconds=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * (seconds * 16000))


def test_live_creates_valid_wav_meta_and_worker_transcription(tmp_path):
    FakeCapture.instances = []
    events = []
    engine = FakeEngine()
    live = LiveSession(tmp_path, SettingsBox(), engine, FakeCapture, FakeSegmenter, events.append)
    assert not live.snapshot()["active"]
    sid, started = live.start(True, False)
    assert started
    assert live.start(True, False) == (sid, False)
    wait_for(lambda: any(event.get("type") == "transcript" for event in events))
    with wave.open(str(tmp_path / sid / "system.wav")) as wav:
        assert wav.getnframes() == 8000
    assert engine.calls == ["vts-live"]
    live.set_sources(False, True)
    assert FakeCapture.instances[0].stopped
    assert live.snapshot()["sources"] == {"system": False, "mic": True}
    assert live.stop() == (sid, True)
    live.shutdown(2)
    assert not live.snapshot()["finishing"]
    meta = json.loads((tmp_path / sid / "meta.json").read_text())
    assert meta["ended_at"] and meta["sources"] == {"system": True, "mic": True}
    assert "[Система] Реальна розмова" in (tmp_path / sid / "transcript.txt").read_text()
    assert "(uk)" not in (tmp_path / sid / "transcript.txt").read_text()


def test_live_preserves_time_alignment_across_pause(tmp_path):
    live = LiveSession(tmp_path, SettingsBox(), FakeEngine(), FakeCapture, FakeSegmenter)
    sid, _ = live.start(False, False)
    live.set_sources(True, False)
    capture = FakeCapture.instances[-1]
    start = live._session["mono_start"]
    capture.on_block("system", np.zeros(8000, np.float32), start + 3.5)
    live.stop()
    live.shutdown(2)
    with wave.open(str(tmp_path / sid / "system.wav")) as wav:
        assert wav.getnframes() == 56000


def test_upload_job_done_and_srt(tmp_path):
    path = tmp_path / "upload_1" / "original.wav"
    wav_file(path)
    queue = JobQueue(tmp_path, SettingsBox(), FakeEngine())
    job = queue.enqueue_upload("upload_1", path, "sample.wav")
    wait_for(lambda: queue.list()[0]["status"] == "done")
    queue.shutdown(2)
    result = queue.list()[0]
    assert result["progress"] == 1
    assert result["model_used"] == "fake"
    assert "original" in (path.parent / "transcript.txt").read_text()
    assert "00:00:00,250 --> 00:00:01,500" in (path.parent / "transcript.srt").read_text()
    assert json.loads((path.parent / "meta.json").read_text())["job"]["id"] == job["id"]


def test_job_error_keeps_partial_result(tmp_path):
    path = tmp_path / "upload_1" / "original.wav"
    wav_file(path)
    engine = FakeEngine()
    engine.fail = True
    queue = JobQueue(tmp_path, SettingsBox(), engine)
    queue.enqueue_upload("upload_1", path, "sample.wav")
    wait_for(lambda: queue.list()[0]["status"] == "error")
    queue.shutdown(2)
    text = (path.parent / "transcript.txt").read_text()
    assert "original" in text and "Помилка транскрипції: тестова помилка" in text


def test_cancel_and_shutdown_are_stoppable(tmp_path):
    engine = FakeEngine()
    engine.gate = threading.Event()
    path = tmp_path / "upload_1" / "original.wav"
    wav_file(path)
    queue = JobQueue(tmp_path, SettingsBox(), engine)
    first = queue.enqueue_upload("upload_1", path, "sample.wav")
    assert engine.entered.wait(1)
    second = queue.enqueue_upload("upload_2", path, "sample.wav")
    assert queue.cancel(second["id"])["status"] == "cancelled"
    queue.cancel(first["id"])
    engine.gate.set()
    queue.shutdown(2)
    assert not queue._worker.is_alive()
    assert all(job["status"] == "cancelled" for job in queue.list())
    assert "Скасовано" in (path.parent / "transcript.txt").read_text()


def test_hq_merges_both_sources_and_deduplicates(tmp_path):
    folder = tmp_path / "meeting_1"
    wav_file(folder / "system.wav")
    wav_file(folder / "microphone.wav")
    queue = JobQueue(tmp_path, SettingsBox(echo_dedupe=False), FakeEngine())
    queue.enqueue_session_hq("meeting_1")
    wait_for(lambda: queue.list()[0]["status"] == "done")
    queue.shutdown(2)
    text = (folder / "transcript_hq.txt").read_text()
    assert "[Система] system" in text and "[Мікрофон] microphone" in text
    assert queue.list()[0]["kind"] == "session_hq"

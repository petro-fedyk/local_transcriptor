"""Sequential HQ output and resumable file inference without CUDA hardware."""
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import wave

import pytest

from voice_to_speech.config import Settings
from voice_to_speech.jobs import JobQueue


class SettingsBox:
    def __init__(self, **values):
        self.value = Settings(**values)

    def snapshot(self):
        return self.value


def segment(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


def write_wav(path, seconds=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * (16000 * seconds))


def wait_done(queue, status="done"):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        jobs = queue.list()
        if jobs and jobs[0]["status"] in ("done", "error", "cancelled"):
            assert jobs[0]["status"] == status, jobs[0]
            return jobs[0]
        time.sleep(.01)
    pytest.fail("job did not finish")


class BaseEngine:
    def __init__(self, device="cpu"):
        self.handle = SimpleNamespace(name="large-v3", device=device)

    def get(self, name, device=None):
        if device:
            return SimpleNamespace(name=name, device=device)
        return self.handle


def test_hq_decodes_sequentially_then_sorts_and_dedupes(tmp_path):
    class Engine(BaseEngine):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.sources = []

        def transcribe_file(self, handle, path, language, glossary):
            self.sources.append(Path(path).stem)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if Path(path).stem == "system":
                    yield segment(2, 3, "Спільна фраза"), SimpleNamespace(duration=4)
                else:
                    yield segment(0, 1, "Окрема думка"), SimpleNamespace(duration=4)
                    yield segment(2, 3, "Спільна фраза"), SimpleNamespace(duration=4)
            finally:
                self.active -= 1

    folder = tmp_path / "meeting_1"
    write_wav(folder / "system.wav")
    write_wav(folder / "microphone.wav")
    engine = Engine()
    queue = JobQueue(tmp_path, SettingsBox(), engine)
    try:
        queue.enqueue_session_hq("meeting_1")
        wait_done(queue)
    finally:
        queue.shutdown(2)
    text = (folder / "transcript_hq.txt").read_text()
    assert engine.sources == ["system", "microphone"]
    assert engine.max_active == 1 and engine.active == 0
    assert text.index("[Мікрофон] Окрема думка") < text.index("[Система] Спільна фраза")
    assert text.count("Спільна фраза") == 1
    assert not (folder / "transcript_hq.partial.txt").exists()
    assert "00:00:00,000 --> 00:00:01,000" in (folder / "transcript_hq.srt").read_text()


def test_cuda_oom_resumes_absolute_offsets_once(tmp_path, monkeypatch):
    class Engine(BaseEngine):
        def __init__(self):
            super().__init__("cuda")
            self.calls = []

        def transcribe_file(self, handle, path, language, glossary, clip_timestamps=None):
            self.calls.append((handle.device, clip_timestamps))
            if len(self.calls) == 1:
                yield segment(0, 1, "Перша фраза"), SimpleNamespace(duration=4)
                raise RuntimeError("CUDA failed with error OUT OF MEMORY")
            assert clip_timestamps == [1.0]
            yield segment(0, 1, "Повтор, який треба пропустити"), SimpleNamespace(duration=4)
            yield segment(1, 4, "Друга фраза"), SimpleNamespace(duration=4)

    monkeypatch.setattr(JobQueue, "_wait_after_oom", staticmethod(lambda job: job["_cancel"].is_set()))
    path = tmp_path / "upload_1" / "original.wav"
    write_wav(path)
    engine = Engine()
    queue = JobQueue(tmp_path, SettingsBox(), engine)
    try:
        queue.enqueue_upload("upload_1", path, "sample.wav")
        result = wait_done(queue)
    finally:
        queue.shutdown(2)
    text = (path.parent / "transcript.txt").read_text()
    assert engine.calls == [("cuda", None), ("cuda", [1.0])]
    assert text.count("Перша фраза") == 1 and text.count("Друга фраза") == 1
    assert "Повтор" not in text
    assert "[0:00:01 - 0:00:04] Друга фраза" in text
    assert result["progress"] == 1


def test_repeated_cuda_oom_resumes_on_cpu(tmp_path, monkeypatch):
    class Engine(BaseEngine):
        def __init__(self):
            super().__init__("cuda")
            self.calls = []
            self.gets = []

        def get(self, name, device=None):
            self.gets.append((name, device))
            return super().get(name, device)

        def transcribe_file(self, handle, path, language, glossary, clip_timestamps=None):
            self.calls.append((handle.device, clip_timestamps))
            if handle.device == "cuda":
                if len(self.calls) == 1:
                    yield segment(0, 1, "До помилки"), SimpleNamespace(duration=4)
                raise RuntimeError("CUDA out of memory")
            yield segment(1, 4, "Після помилки"), SimpleNamespace(duration=4)

    monkeypatch.setattr(JobQueue, "_wait_after_oom", staticmethod(lambda job: job["_cancel"].is_set()))
    path = tmp_path / "upload_1" / "original.wav"
    write_wav(path)
    events = []
    engine = Engine()
    queue = JobQueue(tmp_path, SettingsBox(), engine, events.append)
    try:
        queue.enqueue_upload("upload_1", path, "sample.wav")
        wait_done(queue)
    finally:
        queue.shutdown(2)
    assert engine.calls == [("cuda", None), ("cuda", [1.0]), ("cuda", [1.0]), ("cpu", [1.0])]
    assert ("large-v3", "cpu") in engine.gets
    assert any(event.get("type") == "log" and "CPU" in event.get("text", "") for event in events)
    progress = [event["progress"] for event in events if event.get("type") == "job"]
    assert progress == sorted(progress)
    text = (path.parent / "transcript.txt").read_text()
    assert "До помилки" in text and "Після помилки" in text


def test_hq_error_preserves_partial_and_final_text(tmp_path):
    class Engine(BaseEngine):
        def transcribe_file(self, handle, path, language, glossary):
            yield segment(1, 2, "Текст перед помилкою"), SimpleNamespace(duration=4)
            raise RuntimeError("тестова помилка декодування")

    folder = tmp_path / "meeting_1"
    write_wav(folder / "system.wav")
    write_wav(folder / "microphone.wav")
    queue = JobQueue(tmp_path, SettingsBox(), Engine())
    try:
        queue.enqueue_session_hq("meeting_1")
        wait_done(queue, "error")
    finally:
        queue.shutdown(2)
    for name in ("transcript_hq.partial.txt", "transcript_hq.txt"):
        text = (folder / name).read_text()
        assert "Текст перед помилкою" in text and "тестова помилка декодування" in text


def test_hq_cancel_preserves_durable_partial(tmp_path):
    class Engine(BaseEngine):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def transcribe_file(self, handle, path, language, glossary):
            yield segment(1, 2, "Збережений фрагмент"), SimpleNamespace(duration=4)
            self.entered.set()
            self.release.wait(2)

    folder = tmp_path / "meeting_1"
    write_wav(folder / "system.wav")
    engine = Engine()
    queue = JobQueue(tmp_path, SettingsBox(), engine)
    try:
        job = queue.enqueue_session_hq("meeting_1")
        assert engine.entered.wait(1)
        assert "Збережений фрагмент" in (folder / "transcript_hq.partial.txt").read_text()
        queue.cancel(job["id"])
        engine.release.set()
        wait_done(queue, "cancelled")
    finally:
        engine.release.set()
        queue.shutdown(2)
    for name in ("transcript_hq.partial.txt", "transcript_hq.txt"):
        text = (folder / name).read_text()
        assert "Збережений фрагмент" in text and "Скасовано" in text


@pytest.mark.parametrize("hq", [False, True])
def test_empty_job_has_no_speech_note(tmp_path, hq):
    class Engine(BaseEngine):
        def transcribe_file(self, handle, path, language, glossary):
            yield segment(0, 1, "..."), SimpleNamespace(duration=4)

    folder = tmp_path / ("meeting_1" if hq else "upload_1")
    path = folder / ("system.wav" if hq else "original.wav")
    write_wav(path)
    queue = JobQueue(tmp_path, SettingsBox(), Engine())
    try:
        if hq:
            queue.enqueue_session_hq(folder.name)
        else:
            queue.enqueue_upload(folder.name, path, "sample.wav")
        wait_done(queue)
    finally:
        queue.shutdown(2)
    text = (folder / ("transcript_hq.txt" if hq else "transcript.txt")).read_text()
    assert "Мови не виявлено (лише музика, шум або тиша)." in text


def test_queued_hq_cancel_creates_partial_and_final_note(tmp_path):
    class Engine(BaseEngine):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def transcribe_file(self, handle, path, language, glossary):
            self.entered.set()
            self.release.wait(2)
            yield segment(0, 1, "Текст"), SimpleNamespace(duration=4)

    upload = tmp_path / "upload_1" / "original.wav"
    folder = tmp_path / "meeting_1"
    write_wav(upload)
    write_wav(folder / "system.wav")
    engine = Engine()
    queue = JobQueue(tmp_path, SettingsBox(), engine)
    try:
        queue.enqueue_upload("upload_1", upload, "sample.wav")
        assert engine.entered.wait(1)
        job = queue.enqueue_session_hq("meeting_1")
        assert queue.cancel(job["id"])["status"] == "cancelled"
        for name in ("transcript_hq.partial.txt", "transcript_hq.txt"):
            text = (folder / name).read_text()
            assert "Скасовано" in text and "Мови не виявлено" in text
    finally:
        engine.release.set()
        queue.shutdown(2)


def test_broken_cuda_instance_is_discarded_and_reloaded(tmp_path, monkeypatch):
    """Real failure: OOM, then the same CT2 instance raises 'invalid device ordinal' forever."""
    class Engine:
        def __init__(self):
            self.instances = 0
            self.handle = self._new()
            self.discarded, self.unloaded, self.calls = [], [], []

        def _new(self):
            self.instances += 1
            return SimpleNamespace(name="large-v3", device="cuda", broken=False, serial=self.instances)

        def get(self, name, device=None):
            return self.handle

        def peek(self, name, device=None):
            return object()

        def unload(self, name, device=None):
            self.unloaded.append(name)

        def discard(self, name, device=None, handle=None):
            self.discarded.append(handle.serial)

        def reload(self, name, device=None):
            self.handle = self._new()
            return self.handle

        def transcribe_file(self, handle, path, language, glossary, clip_timestamps=None):
            self.calls.append((handle.serial, clip_timestamps))
            if handle.broken:
                raise RuntimeError("parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal")
            if handle.serial == 1:
                yield segment(0, 1, "Перша фраза"), SimpleNamespace(duration=4)
                handle.broken = True
                raise RuntimeError("CUDA failed with error out of memory")
            yield segment(1, 4, "Друга фраза"), SimpleNamespace(duration=4)

    monkeypatch.setattr(JobQueue, "_wait_after_oom", staticmethod(lambda job: job["_cancel"].is_set()))
    path = tmp_path / "upload_1" / "original.wav"
    write_wav(path)
    engine = Engine()
    queue = JobQueue(tmp_path, SettingsBox(), engine)
    try:
        queue.enqueue_upload("upload_1", path, "sample.wav")
        wait_done(queue)
    finally:
        queue.shutdown(2)
    text = (path.parent / "transcript.txt").read_text()
    assert engine.calls == [(1, None), (2, [1.0])]  # never reuses the broken instance
    assert engine.discarded == [1]
    assert "large-v3-turbo" in engine.unloaded  # idle live model freed to make room
    assert text.count("Перша фраза") == 1 and text.count("Друга фраза") == 1


def test_cpu_fallback_returns_to_cuda_when_vram_is_free(tmp_path, monkeypatch):
    class Engine:
        def __init__(self):
            self.calls, self.unloaded = [], []
            self.cuda_ok = False

        def get(self, name, device=None):
            if device == "cuda" and not self.cuda_ok:
                raise RuntimeError("CUDA failed with error out of memory")
            return SimpleNamespace(name=name, device=device or "cuda")

        def unload(self, name, device=None):
            self.unloaded.append((name, device))

        def transcribe_file(self, handle, path, language, glossary, clip_timestamps=None):
            self.calls.append((handle.device, clip_timestamps))
            start = clip_timestamps[0] if clip_timestamps else 0
            if handle.device == "cuda" and not clip_timestamps:
                raise RuntimeError("CUDA failed with error out of memory")
            for second in range(int(start), 4):
                if handle.device == "cpu" and second == 2:
                    self.cuda_ok = True  # VRAM got freed while we were on CPU
                yield segment(second, second + 1, f"Фраза {second}"), SimpleNamespace(duration=4)

    monkeypatch.setattr(JobQueue, "_wait_after_oom", staticmethod(lambda job: job["_cancel"].is_set()))
    monkeypatch.setattr(JobQueue, "CUDA_RETRY_S", 0)
    path = tmp_path / "upload_1" / "original.wav"
    write_wav(path)
    engine = Engine()
    queue = JobQueue(tmp_path, SettingsBox(), engine)
    try:
        queue.enqueue_upload("upload_1", path, "sample.wav")
        wait_done(queue)
    finally:
        queue.shutdown(2)
    text = (path.parent / "transcript.txt").read_text()
    assert ("cuda", [3.0]) in engine.calls or any(c[0] == "cuda" and c[1] for c in engine.calls)
    assert engine.calls[-1][0] == "cuda"
    for second in range(4):
        assert text.count(f"Фраза {second}") == 1
    assert "повернуто на CUDA" in text

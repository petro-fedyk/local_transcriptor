"""Resume routes restore progress from files after a server restart."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import httpx

from voice_to_speech.config import Config
from voice_to_speech.server import create_app


class FakeEngine:
    cuda_hint = None

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.handle = SimpleNamespace(name="large-v3", device="cpu", lock=threading.RLock())

    def states(self):
        return {}

    def get(self, name, device=None):
        return self.handle

    def unload(self, name, device=None):
        pass

    def transcribe_file(self, handle, path, language, glossary, clip_timestamps=None):
        self.calls.append(clip_timestamps)
        info = SimpleNamespace(duration=2.0)
        if clip_timestamps is None:
            yield SimpleNamespace(start=0.0, end=1.0, text="Перша фраза"), info
            if self.fail:
                raise RuntimeError("Тестова помилка декодування")
        yield SimpleNamespace(start=1.0, end=2.0, text="Друга фраза"), info


def run_test(coro):
    async def pump():
        task = asyncio.create_task(coro)
        # Restricted test sandboxes may not wake on executor completions.
        while not task.done():
            await asyncio.sleep(.01)
        return await task
    return asyncio.run(pump())


async def wait_status(client, job_id, expected):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        jobs = (await client.get("/api/jobs")).json()["jobs"]
        match = next((job for job in jobs if job["id"] == job_id), None)
        if match and match["status"] == expected:
            return match
        await asyncio.sleep(.01)
    raise AssertionError(f"job {job_id} did not reach {expected}")


def test_resume_upload_from_files_after_fresh_app(tmp_path):
    async def run():
        static = Path(__file__).resolve().parent.parent / "static"
        config = Config(data_dir=tmp_path, port=8765, static_dir=static)
        first = create_app(config, engine=FakeEngine(fail=True), capture_factory=lambda **kwargs: None)
        async with first.router.lifespan_context(first):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=first), base_url="http://localhost:8765") as client:
                response = await client.post("/api/jobs/upload", files={"file": ("meeting.m4a", b"audio", "audio/mp4")})
                assert response.status_code == 202
                old_job = response.json()["job"]
                await wait_status(client, old_job["id"], "error")
        # A new app has no in-memory job; it must use the saved files and meta.
        folder = tmp_path / "recordings" / old_job["session_id"]
        engine = FakeEngine()
        second = create_app(config, engine=engine, capture_factory=lambda **kwargs: None)
        async with second.router.lifespan_context(second):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=second), base_url="http://localhost:8765") as client:
                assert (await client.post("/api/jobs/unknown/resume")).status_code == 404
                response = await client.post(f"/api/sessions/{old_job['session_id']}/resume")
                assert response.status_code == 202
                new_job = response.json()["job"]
                await wait_status(client, new_job["id"], "done")
        assert engine.calls == [[1.0]]
        text = (folder / "transcript.txt").read_text(encoding="utf-8")
        srt = (folder / "transcript.srt").read_text(encoding="utf-8")
        assert "Помилка транскрипції" in text
        assert "Продовжено з 0:00:01" in text
        assert text.count("Перша фраза") == 1
        assert text.count("Друга фраза") == 1
        assert srt.startswith("1\n") and "\n2\n00:00:01,000 --> 00:00:02,000" in srt
    run_test(run())

"""HTTP, WebSocket, and startup contract tests without hardware."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from voice_to_speech.config import Config
from voice_to_speech.security import LocalOnlyMiddleware
from voice_to_speech.server import create_app, socket_activation_fd


class FakeEngine:
    cuda_hint = None

    def __init__(self):
        self.handle = SimpleNamespace(name="large-v3-turbo", device="cpu", lock=threading.RLock())

    def states(self):
        return {"large-v3-turbo": {"state": "ready", "device": "cpu", "error": None}}

    def get(self, name):
        return self.handle

    def transcribe_live(self, handle, audio, language, prompt):
        return SimpleNamespace(text="Тестовий текст", language="uk", segments_info=[])

    def transcribe_file(self, handle, path, language, glossary):
        info = SimpleNamespace(duration=2.0, language="uk")
        yield SimpleNamespace(start=0.0, end=1.0, text="Перший фрагмент"), info
        yield SimpleNamespace(start=1.0, end=2.0, text="Другий фрагмент"), info

    def unload(self, name):
        pass


class FakeCapture:
    def __init__(self, source, device, on_block):
        self.source = source
        self.on_block = on_block
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


def make_app(tmp_path, engine=None):
    static = Path(__file__).resolve().parent.parent / "static"
    return create_app(Config(data_dir=tmp_path, port=8765, static_dir=static),
                      engine=engine or FakeEngine(), capture_factory=FakeCapture)


async def call(client, method, url, **kwargs):
    return await client.request(method, url, **kwargs)


async def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return
        await asyncio.sleep(.02)
    assert await predicate()


def run_test(coro):
    async def pump():
        task = asyncio.create_task(coro)
        # The restricted test sandbox does not wake its loop for executor
        # completions. Keep this workaround in tests, not in the server.
        while not task.done():
            await asyncio.sleep(.01)
        return await task
    return asyncio.run(pump())


def test_socket_activation_env():
    assert socket_activation_fd({"LISTEN_PID": "42", "LISTEN_FDS": "1"}, 42) == 3
    assert socket_activation_fd({"LISTEN_PID": "41", "LISTEN_FDS": "1"}, 42) is None
    assert socket_activation_fd({"LISTEN_PID": "bad", "LISTEN_FDS": "1"}, 42) is None
    assert socket_activation_fd({"LISTEN_PID": "42", "LISTEN_FDS": "0"}, 42) is None


def test_security_http_and_websocket(tmp_path):
    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8765") as client:
                assert (await call(client, "GET", "/api/status", headers={"host": "evil.test:8765"})).status_code == 403
                assert (await call(client, "POST", "/api/settings", headers={"origin": "http://evil.test:8765"}, json={})).status_code == 403
                assert (await call(client, "POST", "/api/settings", headers={"origin": "http://localhost:8765"}, json={})).status_code == 200
                assert (await call(client, "POST", "/api/settings", json={})).status_code == 200
        for origin, expected in (("http://evil.test:8765", "websocket.close"), ("http://localhost:8765", "websocket.accept")):
            sent = []
            async def accepted(scope, receive, send):
                await send({"type": "websocket.accept"})
            async def collect(event):
                sent.append(event)
            middleware = LocalOnlyMiddleware(accepted, port=8765)
            await middleware({"type": "websocket", "headers": [(b"host", b"localhost:8765"), (b"origin", origin.encode())]},
                             lambda: None, collect)
            assert sent[0]["type"] == expected
    run_test(run())


def test_settings_session_and_listing(tmp_path):
    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8765") as client:
                assert (await client.get("/api/status")).json()["session"]["active"] is False
                assert (await client.post("/api/settings", json={"language": "xx"})).status_code == 422
                assert (await client.post("/api/settings", json={"glossary": "Плата STM32"})).json()["glossary"] == "Плата STM32"
                response = await client.post("/api/session/start", json={"system": True, "mic": False})
                assert response.status_code == 200
                sid = response.json()["session_id"]
                assert sid.startswith("meeting_")
                assert (await client.post("/api/session/sources", json={"system": False, "mic": True})).status_code == 200
                assert (await client.post("/api/session/stop")).status_code == 200
                folder = tmp_path / "recordings" / sid
                async def meta_done():
                    return (folder / "meta.json").exists() and json.loads((folder / "meta.json").read_text()).get("ended_at") is not None
                await wait_for(meta_done)
                listing = (await client.get("/api/sessions")).json()["sessions"]
                assert listing[0]["id"] == sid
                assert not any(f["name"].endswith(".wav") for f in listing[0]["files"])
                assert (await client.delete("/api/sessions/bad")).status_code == 400
                assert (await client.delete("/api/sessions/meeting_../../etc")).status_code in (400, 404)
                assert (await client.delete(f"/api/sessions/{sid}")).status_code == 200
    run_test(run())


def test_upload_job_and_live_parallel(tmp_path):
    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8765") as client:
                sid = (await client.post("/api/session/start", json={"system": False, "mic": False})).json()["session_id"]
                response = await client.post("/api/jobs/upload", files={"file": ("speech.wav", b"RIFF" + b"0" * 50, "audio/wav")})
                assert response.status_code == 202
                job_id = response.json()["job"]["id"]
                async def job_done():
                    return next(j for j in (await client.get("/api/jobs")).json()["jobs"] if j["id"] == job_id)["status"] == "done"
                await wait_for(job_done)
                assert (await client.get("/api/status")).json()["session"]["active"] is True
                uploads = [p for p in (tmp_path / "recordings").iterdir() if p.name.startswith("upload_")]
                assert len(uploads) == 1
                assert "Перший фрагмент" in (uploads[0] / "transcript.txt").read_text()
                assert "-->" in (uploads[0] / "transcript.srt").read_text()
                assert (await client.post("/api/session/stop")).status_code == 200
    run_test(run())


def test_shutdown_rejects_active_jobs_without_force(tmp_path):
    class SlowEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def transcribe_file(self, handle, path, language, glossary):
            self.entered.set()
            self.release.wait(2)
            yield SimpleNamespace(start=0, end=1, text="Текст"), SimpleNamespace(duration=1)

    async def run():
        engine = SlowEngine()
        app = make_app(tmp_path, engine)
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8765") as client:
                    upload = await client.post("/api/jobs/upload", files={"file": ("audio.wav", b"RIFF" + b"0" * 50, "audio/wav")})
                    assert upload.status_code == 202
                    async def running():
                        return (await client.get("/api/status")).json()["jobs"]["running"] == 1
                    await wait_for(running)
                    assert (await client.post("/api/shutdown")).status_code == 409
                    engine.release.set()
        finally:
            engine.release.set()
    run_test(run())

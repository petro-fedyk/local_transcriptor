"""Crash protection tests without NVIDIA hardware or logind."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import httpx
import pytest

from voice_to_speech import engine, guard, summary
from voice_to_speech.config import Config
from voice_to_speech.server import create_app


def _inhibit_script(tmp_path, monkeypatch, *, refuse_lid=False):
    script = tmp_path / "systemd-inhibit"
    script.write_text("#!/usr/bin/env python3\n"
                      "import os, sys, time\n"
                      "with open(os.environ['GUARD_CALLS'], 'a') as out: out.write(' '.join(sys.argv[1:]) + '\\n')\n"
                      + ("if 'handle-lid-switch' in ' '.join(sys.argv): sys.exit(1)\n" if refuse_lid else "")
                      + "time.sleep(3600)\n")
    script.chmod(0o755)
    calls = tmp_path / "calls"
    monkeypatch.setenv("GUARD_CALLS", str(calls))
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    return calls


def test_sleep_guard_lifecycle(tmp_path, monkeypatch):
    calls = _inhibit_script(tmp_path, monkeypatch)
    sleeper = guard.SleepGuard()
    try:
        sleeper.update(True)
        first = sleeper._process
        assert sleeper.state()["active"] is True
        assert sleeper.state()["what"] == "sleep:idle:handle-lid-switch"
        sleeper.update(True)
        assert sleeper._process is first
        assert len(calls.read_text().splitlines()) == 1
        sleeper.update(False)
        assert first.poll() is not None
        assert sleeper.state()["active"] is False
        sleeper.update(True)
        second = sleeper._process
        sleeper.shutdown()
        assert second.poll() is not None
    finally:
        sleeper.shutdown()


def test_sleep_guard_missing_and_lid_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(guard.shutil, "which", lambda _: None)
    sleeper = guard.SleepGuard()
    sleeper.update(True)
    assert sleeper.state()["active"] is False
    assert "недоступний" in sleeper.state()["error"]
    sleeper.shutdown()

    monkeypatch.undo()
    calls = _inhibit_script(tmp_path, monkeypatch, refuse_lid=True)
    sleeper = guard.SleepGuard()
    try:
        sleeper.update(True)
        assert sleeper.state()["active"] is True
        assert sleeper.state()["what"] == "sleep:idle"
        lines = calls.read_text().splitlines()
        assert "handle-lid-switch" in lines[0]
        assert "--what=sleep:idle " in lines[1]
    finally:
        sleeper.shutdown()


def test_gpu_memory_parsing_and_failure(monkeypatch):
    guard._gpu_cached_at = 0
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="4096, 2700\n"))
    assert guard.gpu_memory() == {"total_mb": 4096, "free_mb": 2700}
    guard._gpu_cached_at = 0
    def fail(*a, **k):
        raise FileNotFoundError()
    monkeypatch.setattr(guard.subprocess, "run", fail)
    assert guard.gpu_memory() is None


@pytest.mark.parametrize("free,expected", [(1000, "cpu"), (4096, "cuda")])
def test_engine_headroom(tmp_path, monkeypatch, free, expected):
    model = tmp_path / "large-v3-turbo"
    model.mkdir()
    (model / "model.bin").touch()
    calls = []
    def factory(path, **kwargs):
        calls.append(kwargs["device"])
        return object()
    monkeypatch.setattr(engine, "WhisperModel", factory)
    monkeypatch.setattr(guard, "gpu_memory", lambda: {"total_mb": 4096, "free_mb": free})
    with pytest.warns(RuntimeWarning) if expected == "cpu" else __import__("contextlib").nullcontext():
        manager = engine.ModelManager(tmp_path, device="cuda")
        handle = manager.get("large-v3-turbo")
    assert handle.device == expected
    assert calls == [expected]
    if expected == "cpu":
        assert "робочого столу" in manager.cuda_hint


@pytest.mark.parametrize("free,expected", [(3900, 99), (2500, 20), (1500, 0)])
def test_llama_initial_ngl(tmp_path, monkeypatch, free, expected):
    monkeypatch.setattr(guard, "gpu_memory", lambda: {"total_mb": 4096, "free_mb": free})
    server = summary.LlamaServer(tmp_path / "llama", tmp_path / "model", tmp_path / "log")
    tried = []
    monkeypatch.setattr(server, "_start", lambda: tried.append(server.ngl))
    with server:
        assert tried == [expected]


def test_server_guard_status_and_session_transitions(tmp_path):
    class FakeGuard:
        def __init__(self):
            self.updates = []
        def update(self, busy):
            self.updates.append(busy)
        def state(self):
            return {"active": bool(self.updates and self.updates[-1]), "what": "sleep:idle", "error": None}
        def shutdown(self):
            pass
    class FakeEngine:
        cuda_hint = None
        def states(self):
            return {}
        def get(self, name):
            return SimpleNamespace(model=object(), name=name, device="cpu", lock=threading.RLock())
    class FakeCapture:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass
    fake = FakeGuard()
    static = Path(__file__).resolve().parent.parent / "static"
    app = create_app(Config(data_dir=tmp_path, port=8765, static_dir=static), engine=FakeEngine(),
                     capture_factory=FakeCapture, sleep_guard=fake)
    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8765") as client:
                assert "sleep_guard" in (await client.get("/api/status")).json()
                assert (await client.post("/api/session/start", json={"system": False, "mic": False})).status_code == 200
                assert True in fake.updates
                assert (await client.post("/api/session/stop")).status_code == 200
                assert fake.updates[-1] is False
    async def pump():
        task = asyncio.create_task(exercise())
        while not task.done():
            await asyncio.sleep(.01)
        return await task
    asyncio.run(pump())

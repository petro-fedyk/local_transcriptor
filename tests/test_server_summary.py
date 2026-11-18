"""Summary HTTP integration without a GPU or llama-server process."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from voice_to_speech.config import Config
from voice_to_speech.server import create_app
from test_server import FakeCapture, FakeEngine, run_test, wait_for


class FakeLlama:
    def __init__(self, server_path, model_path, log_path):
        self.model_used = "fake-local-model"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def stop(self):
        pass

    def chat(self, messages, max_tokens):
        if messages[-1]["content"].startswith("Створи підсумок"):
            return "# Підсумок зустрічі\n## Коротко\n- Обговорили план."
        return "### 00:00–00:01\n- Обговорили план."


def make_app(tmp_path):
    config = Config(data_dir=tmp_path, port=8765, static_dir=Path(__file__).resolve().parent.parent / "static")
    config.llm_server_path.parent.mkdir(parents=True, exist_ok=True)
    config.llm_server_path.touch()
    config.llm_model_path.parent.mkdir(parents=True, exist_ok=True)
    config.llm_model_path.touch()
    return create_app(config, engine=FakeEngine(), capture_factory=FakeCapture, summary_server_factory=FakeLlama)


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8765")


async def done(client, job_id):
    return next(job for job in (await client.get("/api/jobs")).json()["jobs"] if job["id"] == job_id)["status"] == "done"


def test_sources_and_sessions(tmp_path):
    async def run():
        root = tmp_path / "recordings"
        for sid, name in (("meeting_2026-01-01", "transcript.txt"),
                          ("upload_2026-01-02", "transcript_hq.srt"),
                          ("summary_2026-01-03", "summary_input.md"),
                          ("invalid_2026-01-04", "transcript.txt")):
            folder = root / sid
            folder.mkdir(parents=True)
            (folder / name).write_text("Текст", encoding="utf-8")
        (root / "summary_2026-01-03" / "summary.md").write_text("# Підсумок", encoding="utf-8")
        (root / "summary_2026-01-03" / "summary_source.txt").write_text("Текст", encoding="utf-8")
        (root / "summary_2026-01-03" / "meta.json").write_text(json.dumps({
            "created_at": "2026-01-03T12:00:00+02:00", "original_filename": "notes.md",
            "sources": {"system": True, "mic": False}}, ensure_ascii=False), encoding="utf-8")
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app), client_for(app) as client:
            page = await client.get("/summary")
            assert page.status_code == 200
            assert page.headers["content-type"].startswith("text/html")
            assert "Підсумки зустрічей" in page.text
            response = await client.get("/api/summary/sources")
            assert response.status_code == 200
            sources = response.json()["sources"]
            assert {(s["session_id"], s["file"]) for s in sources} == {
                ("meeting_2026-01-01", "transcript.txt"),
                ("upload_2026-01-02", "transcript_hq.srt"),
                ("summary_2026-01-03", "summary_input.md")}
            assert all(s["size"] > 0 and s["modified"] and s["label"] for s in sources)
            summary_source = next(s for s in sources if s["session_id"].startswith("summary_"))
            assert summary_source["has_summary"] and summary_source["has_summary_source"]
            assert summary_source["created_at"] == "2026-01-03T12:00:00+02:00"
            assert summary_source["original_filename"] == "notes.md"
            assert summary_source["sources"] == {"system": True, "mic": False}
            sessions = (await client.get("/api/sessions")).json()["sessions"]
            assert {s["kind"] for s in sessions} == {"meeting", "upload", "summary"}
            assert any(f["label"] == "Вхідний текст" for s in sessions for f in s["files"])
            assert any(f["label"] == "Підсумок зустрічі" for s in sessions for f in s["files"])
    run_test(run())


def test_enqueue_download_and_validation(tmp_path):
    async def run():
        app = make_app(tmp_path)
        sid = "meeting_2026-01-01"
        folder = tmp_path / "recordings" / sid
        folder.mkdir()
        (folder / "transcript.txt").write_text("[00:00:01] [Система] Обговорили план.", encoding="utf-8")
        async with app.router.lifespan_context(app), client_for(app) as client:
            for filename in ("../x", "meta.json"):
                assert (await client.post("/api/jobs/summary", json={"session_id": sid, "file": filename})).status_code == 400
            assert (await client.post("/api/jobs/summary", json={"session_id": "meeting_999", "file": "transcript.txt"})).status_code == 404
            assert (await client.post("/api/jobs/summary", json={"session_id": "../etc", "file": "transcript.txt"})).status_code == 400
            (tmp_path / "recordings" / "meeting_2026-01-02").symlink_to(folder, target_is_directory=True)
            assert (await client.post("/api/jobs/summary", json={"session_id": "meeting_2026-01-02", "file": "transcript.txt"})).status_code == 400
            response = await client.post("/api/jobs/summary", json={"session_id": sid, "file": "transcript.txt"})
            assert response.status_code == 202
            job_id = response.json()["job"]["id"]
            await wait_for(lambda: done(client, job_id))
            jobs = (await client.get("/api/jobs")).json()["jobs"]
            assert any(j["kind"] == "summary" and j["id"] == job_id for j in jobs)
            app.state.vts.jobs.list = lambda: [{"id": "whisper-example", "kind": "upload", "status": "done"}]
            assert {j["kind"] for j in (await client.get("/api/jobs")).json()["jobs"]} == {"upload", "summary"}
            result = await client.get(f"/recordings/{sid}/summary.md")
            assert result.status_code == 200
            assert result.headers["content-type"] == "text/markdown; charset=utf-8"
            assert "# Підсумок зустрічі" in result.text
            assert (folder / "summary_source.txt").is_file()
            listing = (await client.get("/api/sessions")).json()["sessions"]
            assert any(f["name"] == "summary.md" and f["label"] == "Підсумок зустрічі" for f in listing[0]["files"])
    run_test(run())


def test_upload_validation(tmp_path, monkeypatch):
    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app), client_for(app) as client:
            assert (await client.post("/api/jobs/summary/upload", files={"file": ("bad.pdf", b"text")})).status_code == 400
            monkeypatch.setattr("voice_to_speech.server.MAX_SUMMARY_UPLOAD", 5)
            assert (await client.post("/api/jobs/summary/upload", files={"file": ("big.txt", b"123456")})).status_code == 400
            monkeypatch.setattr("voice_to_speech.server.MAX_SUMMARY_UPLOAD", 20 * 1024 * 1024)
            assert (await client.post("/api/jobs/summary/upload", files={"file": ("bad.txt", b"\xff")})).status_code == 400
            assert not list((tmp_path / "recordings").glob("summary_*"))
            response = await client.post("/api/jobs/summary/upload", files={"file": ("notes.txt", "Обговорили план.".encode())})
            assert response.status_code == 202
            sid = response.json()["job"]["session_id"]
            assert sid.startswith("summary_")
            folder = tmp_path / "recordings" / sid
            assert (folder / "summary_input.txt").read_text(encoding="utf-8") == "Обговорили план."
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            assert meta["kind"] == "summary" and meta["original_filename"] == "notes.txt"
            assert (await client.get(f"/recordings/{sid}/summary_input.txt")).headers["content-type"] == "text/plain; charset=utf-8"
    run_test(run())


def test_cancel_resume_and_shutdown(tmp_path):
    async def run():
        app = make_app(tmp_path)
        sid = "meeting_2026-01-01"
        folder = tmp_path / "recordings" / sid
        folder.mkdir()
        (folder / "transcript.txt").write_text("Обговорили план.", encoding="utf-8")
        waiting = {"value": True}
        app.state.vts.summary.gpu_busy = lambda: "Очікує завершення запису" if waiting["value"] else None
        async with app.router.lifespan_context(app), client_for(app) as client:
                response = await client.post("/api/jobs/summary", json={"session_id": sid, "file": "transcript.txt"})
                job_id = response.json()["job"]["id"]
                async def is_waiting():
                    jobs = (await client.get("/api/jobs")).json()["jobs"]
                    return next(j for j in jobs if j["id"] == job_id)["stage"] == "Очікує завершення запису"
                await wait_for(is_waiting)
                assert (await client.get("/api/status")).json()["summary_jobs"]["running"] == 1
                assert (await client.post("/api/shutdown")).status_code == 409
                assert (await client.delete(f"/api/sessions/{sid}")).status_code == 409
                cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
                assert cancelled.status_code == 200
                waiting["value"] = False
                async def is_cancelled():
                    jobs = (await client.get("/api/jobs")).json()["jobs"]
                    return next(j for j in jobs if j["id"] == job_id)["status"] == "cancelled"
                await wait_for(is_cancelled)
                resumed = await client.post(f"/api/jobs/{job_id}/resume")
                assert resumed.status_code == 202
                assert resumed.json()["job"]["kind"] == "summary"
                await wait_for(lambda: done(client, resumed.json()["job"]["id"]))
    run_test(run())


def test_llm_status_missing_files(tmp_path):
    async def run():
        app = make_app(tmp_path)
        app.state.vts.config.llm_server_path.unlink()
        app.state.vts.config.llm_model_path.unlink()
        async with app.router.lifespan_context(app), client_for(app) as client:
            status = (await client.get("/api/llm/status")).json()
            assert status["server_found"] is False and status["model_found"] is False
            assert status["ready"] is False and status["hint"]
    run_test(run())

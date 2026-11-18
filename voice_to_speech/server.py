"""FastAPI application, lifecycle, local API, and WebSocket events."""

from __future__ import annotations

import asyncio
import codecs
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from functools import partial
import json
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import time
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Config, SettingsError, SettingsStore
from .security import LocalOnlyMiddleware

LOG = logging.getLogger(__name__)
SESSION_RE = re.compile(r"^(meeting|upload|summary)_[0-9_-]+$")
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".opus", ".flac", ".webm", ".aac", ".wma", ".mkv", ".mov"}
MAX_UPLOAD = 2 * 1024 * 1024 * 1024
MAX_SUMMARY_UPLOAD = 20 * 1024 * 1024
SUMMARY_FILES = {"transcript_hq.txt", "transcript.txt", "transcript_hq.srt", "summary_input.txt", "summary_input.srt", "summary_input.md"}
FILE_LABELS = {"transcript.txt": "Транскрипція", "transcript.srt": "Субтитри", "transcript_hq.txt": "Якісна транскрипція", "transcript_hq.partial.txt": "Часткова якісна транскрипція", "transcript_hq.srt": "Якісні субтитри", "system.wav": "Системний звук", "microphone.wav": "Мікрофон", "summary.md": "Підсумок зустрічі", "summary_source.txt": "Текст без дублів", "summary_input.txt": "Вхідний текст", "summary_input.srt": "Вхідний текст", "summary_input.md": "Вхідний текст"}


def socket_activation_fd(env: dict[str, str], pid: int) -> int | None:
    """Return the inherited socket fd only for this process."""
    try:
        if int(env.get("LISTEN_PID", "0")) == pid and int(env.get("LISTEN_FDS", "0")) >= 1:
            return 3
    except ValueError:
        pass
    return None


def power_status(root: Path = Path("/sys/class/power_supply")) -> dict:
    on_ac = None
    battery_percent = None
    try:
        for item in root.iterdir():
            kind = (item / "type").read_text().strip()
            if kind == "Mains":
                on_ac = (item / "online").read_text().strip() == "1"
            elif kind == "Battery":
                battery_percent = int((item / "capacity").read_text().strip())
    except (OSError, ValueError):
        pass
    return {"on_ac": on_ac, "battery_percent": battery_percent}


def safe_session_path(recordings: Path, session_id: str) -> Path:
    if not SESSION_RE.fullmatch(session_id):
        raise HTTPException(400, "Некоректний ідентифікатор сесії.")
    root = recordings.resolve()
    candidate = root / session_id
    if candidate.is_symlink():
        raise HTTPException(400, "Некоректний шлях сесії.")
    path = candidate.resolve()
    if path.parent != root:
        raise HTTPException(400, "Некоректний шлях сесії.")
    return path


def create_app(config: Config | None = None, engine=None, capture_factory=None, segmenter_factory=None, *, open_browser: bool = False, summary_server_factory=None, sleep_guard=None) -> FastAPI:
    config = config or Config()
    config.recordings.mkdir(parents=True, exist_ok=True)
    settings = SettingsStore(config.settings_path)
    if engine is None:
        from .engine import ModelManager
        engine = ModelManager(config.models)
    from .jobs import JobQueue
    from .live import LiveSession
    from .summary import SummaryManager
    from .engine import trim_memory
    from .guard import SleepGuard, gpu_memory, DESKTOP_RESERVE_MB
    from .audio import CaptureSource, list_sources

    class State:
        pass

    state = State()
    state.loop = None
    state.clients: set[WebSocket] = set()
    state.clients_lock = threading.Lock()
    state.started = time.monotonic()
    state.idle_since = state.started
    state.last_busy = False
    state.idle_task = None
    state.uvicorn_server = None
    state.exiting = False
    state.open_browser = open_browser
    state.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vts-api")
    state.settings = settings
    state.engine = engine
    state.config = config
    state.sleep_guard = sleep_guard if sleep_guard is not None else SleepGuard()

    async def offload(fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(state.executor, partial(fn, *args, **kwargs))

    async def send_event(event: dict):
        with state.clients_lock:
            clients = list(state.clients)
        stale = []
        for client in clients:
            try:
                await client.send_json(event)
            except Exception:
                stale.append(client)
        if stale:
            with state.clients_lock:
                state.clients.difference_update(stale)

    def broadcast(event: dict):
        loop = state.loop
        if loop is not None and loop.is_running():
            if event.get("type") in ("status", "job", "backlog"):
                loop.call_soon_threadsafe(update_sleep_guard)
            if event.get("type") == "status":
                event = {"type": "status", **status()}
            asyncio.run_coroutine_threadsafe(send_event(event), loop)
            if event.get("type") in ("job", "backlog"):
                asyncio.run_coroutine_threadsafe(send_event({"type": "status", **status()}), loop)

    state.jobs = JobQueue(config.recordings, settings, engine, broadcast=broadcast,
                          live_state=lambda: state.live.snapshot() if hasattr(state, "live") else
                          {"active": False, "finishing": False})
    state.live = LiveSession(config.recordings, settings, engine,
                             capture_factory=capture_factory or CaptureSource,
                             segmenter_factory=segmenter_factory,
                             broadcast=broadcast,
                             enqueue_hq=state.jobs.enqueue_session_hq)

    def gpu_busy() -> str | None:
        live = state.live.snapshot()
        if live.get("active") or live.get("finishing"):
            return "Очікує завершення запису"
        counts = state.jobs.counts()
        if counts.get("queued") or counts.get("running"):
            return "Очікує завершення транскрипції"
        return None

    def gpu_release():
        states = getattr(engine, "states", None)
        unload = getattr(engine, "unload", None)
        if callable(states) and callable(unload):
            for name in tuple(states()):
                unload(name)
        trim_memory()

    summary_options = {} if summary_server_factory is None else {"server_factory": summary_server_factory}
    state.summary = SummaryManager(config.recordings, config, broadcast, gpu_busy, gpu_release, **summary_options)

    def busy() -> bool:
        session = state.live.snapshot()
        counts = state.jobs.counts()
        return bool(session.get("active") or session.get("finishing") or counts.get("queued") or counts.get("running") or state.summary.busy())

    def update_sleep_guard():
        state.sleep_guard.update(busy())

    def touch():
        state.idle_since = time.monotonic()

    def status() -> dict:
        update_sleep_guard()
        current = settings.snapshot()
        busy_now = busy()
        elapsed = time.monotonic() - state.idle_since
        idle_in = None if current.idle_shutdown_minutes == 0 or busy_now else max(0, round(current.idle_shutdown_minutes * 60 - elapsed))
        model_states = engine.states() if hasattr(engine, "states") else {}
        model_states = dict(model_states)
        model_states["cuda_hint"] = getattr(engine, "cuda_hint", None)
        return {"server": {"version": __version__, "port": config.port, "pid": os.getpid(),
                           "uptime_s": round(time.monotonic() - state.started), "idle_shutdown_in_s": idle_in},
                "models": model_states, "session": state.live.snapshot(), "jobs": state.jobs.counts(),
                "summary_jobs": state.summary.counts(),
                "power": power_status(), "sleep_guard": state.sleep_guard.state()}

    async def publish_status():
        await send_event({"type": "status", **(await offload(status))})

    async def idle_loop():
        while not state.exiting:
            await asyncio.sleep(5)
            now_busy = busy()
            state.sleep_guard.update(now_busy)
            if state.last_busy and not now_busy:
                touch()
            state.last_busy = now_busy
            minutes = settings.snapshot().idle_shutdown_minutes
            if minutes and not now_busy and time.monotonic() - state.idle_since >= minutes * 60:
                state.exiting = True
                await send_event({"type": "shutdown", "reason": "idle"})
                if state.uvicorn_server is not None:
                    state.uvicorn_server.should_exit = True
                break

    def preload_live_model():
        name = settings.snapshot().live_model
        try:
            engine.get(name)
        except Exception as exc:
            LOG.error("Не вдалося завантажити модель: %s", exc)
            broadcast({"type": "log", "level": "error", "text": str(exc)})
        broadcast({"type": "status", **status()})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.loop = asyncio.get_running_loop()
        try:
            os.nice(10)
        except (OSError, AttributeError):
            pass
        threading.Thread(target=preload_live_model, name="vts-model-load", daemon=True).start()
        state.idle_task = asyncio.create_task(idle_loop())
        if state.open_browser:
            import webbrowser
            state.loop.call_later(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{config.port}/"))
        try:
            yield
        finally:
            state.exiting = True
            if state.idle_task:
                state.idle_task.cancel()
                try:
                    await state.idle_task
                except asyncio.CancelledError:
                    pass
            await offload(state.live.shutdown, 25)
            await offload(state.jobs.shutdown, 25)
            await offload(state.summary.shutdown, 25)
            state.sleep_guard.shutdown()
            state.executor.shutdown(wait=False, cancel_futures=True)
            state.loop = None

    app = FastAPI(lifespan=lifespan)
    app.state.vts = state
    app.add_middleware(LocalOnlyMiddleware, port=config.port)
    app.mount("/static", StaticFiles(directory=config.static_dir), name="static")
    app.mount("/recordings", StaticFiles(directory=config.recordings), name="recordings")

    @app.get("/")
    async def index():
        return FileResponse(config.static_dir / "index.html")

    @app.get("/summary")
    async def summary_page():
        return FileResponse(config.static_dir / "summary.html")

    @app.get("/api/status")
    async def get_status():
        return await offload(status)

    @app.get("/api/guard/status")
    async def guard_status():
        await offload(update_sleep_guard)
        return {"sleep_guard": state.sleep_guard.state(), "gpu_memory": await offload(gpu_memory),
                "desktop_reserve_mb": DESKTOP_RESERVE_MB}

    @app.get("/api/settings")
    async def get_settings():
        return asdict(settings.snapshot())

    @app.post("/api/settings")
    async def update_settings(request: Request):
        try:
            patch = await request.json()
            previous = settings.snapshot()
            value = await offload(settings.update, patch)
        except (SettingsError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        touch()
        if value.live_model != previous.live_model:
            threading.Thread(target=preload_live_model, daemon=True).start()
        if (value.system_source != previous.system_source or value.mic_source != previous.mic_source):
            current = state.live.snapshot()
            if current.get("active"):
                await offload(state.live.set_sources, current["sources"]["system"], current["sources"]["mic"])
        await publish_status()
        return asdict(value)

    @app.get("/api/sources")
    async def get_sources():
        return await offload(list_sources)

    @app.post("/api/session/start")
    async def start_session(request: Request):
        data = await request.json()
        system, mic = data.get("system", True), data.get("mic", False)
        if type(system) is not bool or type(mic) is not bool:
            raise HTTPException(422, "Джерела мають бути так або ні.")
        session_id, started = await offload(state.live.start, system, mic)
        touch()
        await publish_status()
        return {"session_id": session_id, "started": started}

    @app.post("/api/session/stop")
    async def stop_session():
        session_id, stopped = await offload(state.live.stop)
        touch()
        await publish_status()
        return {"session_id": session_id, "stopped": stopped}

    @app.post("/api/session/sources")
    async def set_sources(request: Request):
        data = await request.json()
        system, mic = data.get("system"), data.get("mic")
        if type(system) is not bool or type(mic) is not bool:
            raise HTTPException(422, "Вкажіть обидва джерела як так або ні.")
        result = await offload(state.live.set_sources, system, mic)
        touch()
        await publish_status()
        return result if isinstance(result, dict) else state.live.snapshot()

    @app.get("/api/sessions")
    async def get_sessions():
        def listing():
            items = []
            for folder in sorted(config.recordings.iterdir(), key=lambda p: p.name.split("_", 1)[-1], reverse=True):
                if not folder.is_dir() or not SESSION_RE.fullmatch(folder.name) or folder.is_symlink():
                    continue
                meta_path = folder / "meta.json"
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                except (OSError, ValueError):
                    meta = {}
                files = []
                for path in sorted(folder.iterdir()):
                    if not path.is_file() or path.is_symlink():
                        continue
                    if path.name not in FILE_LABELS and not path.name.startswith("original."):
                        continue
                    size = path.stat().st_size
                    if path.suffix == ".wav" and size <= 44:
                        continue
                    files.append({"name": path.name, "label": FILE_LABELS.get(path.name, "Початковий файл"),
                                  "url": f"/recordings/{quote(folder.name)}/{quote(path.name)}", "size": size})
                items.append({"id": folder.name, "kind": folder.name.split("_", 1)[0],
                              "meta": meta, "files": files})
            return items
        return {"sessions": await offload(listing)}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        path = safe_session_path(config.recordings, session_id)
        snap = state.live.snapshot()
        if snap.get("id") == session_id and (snap.get("active") or snap.get("finishing")):
            raise HTTPException(409, "Зупиніть сесію перед видаленням.")
        if any(j.get("session_id") == session_id and j.get("status") in ("queued", "running") for j in state.jobs.list() + state.summary.list()):
            raise HTTPException(409, "Дочекайтеся завершення задачі перед видаленням.")
        if not path.is_dir():
            raise HTTPException(404, "Сесію не знайдено.")
        await offload(shutil.rmtree, path)
        touch()
        await publish_status()
        return {"deleted": session_id}

    @app.post("/api/jobs/upload", status_code=202)
    async def upload(file: UploadFile):
        filename = Path(file.filename or "").name
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(422, "Непідтримуваний формат файлу.")
        session_id = f"upload_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}"
        folder = config.recordings / session_id
        await offload(folder.mkdir, parents=True, exist_ok=False)
        destination = folder / f"original{extension}"
        total = 0
        queued = False
        try:
            out = await offload(destination.open, "wb")
            try:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise HTTPException(413, "Файл перевищує ліміт 2 ГіБ.")
                    await offload(out.write, chunk)
            finally:
                await offload(out.close)
            meta = {"id": session_id, "kind": "upload", "original_filename": filename,
                    "created_at": datetime.now().isoformat()}
            await offload((folder / "meta.json").write_text, json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
            job = await offload(state.jobs.enqueue_upload, session_id, destination, filename)
            queued = True
            touch()
            await publish_status()
            return {"job": job}
        except Exception:
            if not queued:
                await offload(shutil.rmtree, folder, ignore_errors=True)
            raise
        finally:
            await file.close()

    @app.post("/api/jobs/session/{session_id}", status_code=202)
    async def hq_session(session_id: str):
        path = safe_session_path(config.recordings, session_id)
        if not session_id.startswith("meeting_") or not path.is_dir():
            raise HTTPException(404, "Сесію не знайдено.")
        job = await offload(state.jobs.enqueue_session_hq, session_id)
        touch()
        await publish_status()
        return {"job": job}

    @app.get("/api/summary/sources")
    async def summary_sources():
        def listing():
            sources = []
            for folder in config.recordings.iterdir():
                if not SESSION_RE.fullmatch(folder.name) or not folder.is_dir() or folder.is_symlink():
                    continue
                meta_path = folder / "meta.json"
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() and not meta_path.is_symlink() else {}
                    if not isinstance(meta, dict):
                        meta = {}
                except (OSError, ValueError):
                    meta = {}
                for name in SUMMARY_FILES:
                    if name.startswith("summary_input.") and not folder.name.startswith("summary_"):
                        continue
                    path = folder / name
                    if not path.is_file() or path.is_symlink():
                        continue
                    stat = path.stat()
                    sources.append({"session_id": folder.name, "file": name,
                                    "label": f"{folder.name} · {FILE_LABELS[name]}", "size": stat.st_size,
                                    "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                                    "has_summary": (folder / "summary.md").is_file(),
                                    "has_summary_source": (folder / "summary_source.txt").is_file(),
                                    "started_at": meta.get("started_at"), "created_at": meta.get("created_at"),
                                    "ended_at": meta.get("ended_at"), "sources": meta.get("sources"),
                                    "original_filename": meta.get("original_filename")})
            return sorted(sources, key=lambda item: item["modified"], reverse=True)
        return {"sources": await offload(listing)}

    @app.post("/api/jobs/summary", status_code=202)
    async def enqueue_summary(request: Request):
        try:
            data = await request.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(400, "Некоректні дані підсумку.") from exc
        if not isinstance(data, dict) or not isinstance(data.get("session_id"), str) or not isinstance(data.get("file"), str):
            raise HTTPException(400, "Вкажіть сесію та файл транскрипції.")
        session_id, filename = data["session_id"], data["file"]
        folder = safe_session_path(config.recordings, session_id)
        if filename not in SUMMARY_FILES or (filename.startswith("summary_input.") and not session_id.startswith("summary_")):
            raise HTTPException(400, "Некоректний файл підсумку.")
        if not folder.is_dir():
            raise HTTPException(404, "Сесію не знайдено.")
        source = folder / filename
        if source.is_symlink():
            raise HTTPException(400, "Некоректний файл підсумку.")
        try:
            job = await offload(state.summary.enqueue, session_id, filename)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        touch()
        await publish_status()
        return {"job": job}

    @app.post("/api/jobs/summary/upload", status_code=202)
    async def upload_summary(file: UploadFile):
        filename = Path(file.filename or "").name
        extension = Path(filename).suffix.lower()
        if extension not in {".txt", ".srt", ".md"}:
            raise HTTPException(400, "Непідтримуваний формат текстового файлу.")
        session_id = f"summary_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}"
        folder = config.recordings / session_id
        await offload(folder.mkdir, parents=True, exist_ok=False)
        destination = folder / f"summary_input{extension}"
        total = 0
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        queued = False
        try:
            out = await offload(destination.open, "wb")
            try:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_SUMMARY_UPLOAD:
                        raise HTTPException(400, "Текстовий файл перевищує ліміт 20 МіБ.")
                    try:
                        decoder.decode(chunk)
                    except UnicodeDecodeError as exc:
                        raise HTTPException(400, "Текстовий файл має бути у форматі UTF-8.") from exc
                    await offload(out.write, chunk)
                try:
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError as exc:
                    raise HTTPException(400, "Текстовий файл має бути у форматі UTF-8.") from exc
            finally:
                await offload(out.close)
            meta = {"kind": "summary", "original_filename": filename, "created_at": datetime.now().astimezone().isoformat()}
            await offload((folder / "meta.json").write_text, json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
            job = await offload(state.summary.enqueue, session_id, destination.name)
            queued = True
            touch()
            await publish_status()
            return {"job": job}
        except ValueError as exc:
            if not queued:
                await offload(shutil.rmtree, folder, ignore_errors=True)
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            if not queued:
                await offload(shutil.rmtree, folder, ignore_errors=True)
            raise HTTPException(404, str(exc)) from exc
        except Exception:
            if not queued:
                await offload(shutil.rmtree, folder, ignore_errors=True)
            raise
        finally:
            await file.close()

    @app.get("/api/llm/status")
    async def llm_status():
        return await offload(state.summary.status)

    @app.get("/api/jobs")
    async def list_jobs():
        return {"jobs": state.jobs.list() + state.summary.list()}

    @app.post("/api/jobs/{job_id}/resume", status_code=202)
    async def resume_job(job_id: str):
        try:
            job = await offload(state.summary.resume if state.summary.owns(job_id) else state.jobs.resume_job, job_id)
        except ValueError as exc:
            raise HTTPException(400 if state.summary.owns(job_id) else 422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        if job is None:
            raise HTTPException(404, "Задачу не знайдено.")
        touch()
        await publish_status()
        return {"job": job}

    @app.post("/api/sessions/{session_id}/resume", status_code=202)
    async def resume_session(session_id: str):
        path = safe_session_path(config.recordings, session_id)
        if not path.is_dir():
            raise HTTPException(404, "Сесію не знайдено.")
        try:
            job = await offload(state.jobs.resume_session, session_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        if job is None:
            raise HTTPException(404, "Немає задачі для продовження.")
        touch()
        await publish_status()
        return {"job": job}

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        job = await offload(state.summary.cancel if state.summary.owns(job_id) else state.jobs.cancel, job_id)
        if job is None:
            raise HTTPException(404, "Задачу не знайдено.")
        touch()
        await publish_status()
        return {"job": job}

    @app.post("/api/shutdown")
    async def shutdown(force: int = 0):
        counts = state.jobs.counts()
        if (counts.get("queued") or counts.get("running") or state.summary.busy()) and not force:
            raise HTTPException(409, "Є активні задачі. Підтвердьте примусове вимкнення.")
        touch()
        await send_event({"type": "shutdown", "reason": "user"})
        if force:
            for job in state.jobs.list():
                if job.get("status") in ("queued", "running"):
                    await offload(state.jobs.cancel, job["id"])
            for job in state.summary.list():
                if job.get("status") in ("queued", "running"):
                    await offload(state.summary.cancel, job["id"])
        await offload(state.live.stop)
        state.exiting = True
        if state.uvicorn_server is not None:
            state.loop.call_later(0.2, setattr, state.uvicorn_server, "should_exit", True)
        return {"stopping": True}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        with state.clients_lock:
            state.clients.add(ws)
        try:
            await ws.send_json({"type": "status", **(await offload(status))})
            while True:
                message = await ws.receive_text()
                if message == "ping":
                    await ws.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            with state.clients_lock:
                state.clients.discard(ws)

    return app

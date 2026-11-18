"""Capture sessions with one independent transcription worker per session."""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import queue
import threading
import time
import wave

import numpy as np

from .engine import is_cuda_failure, trim_memory
from .textfilter import EchoDeduper, clean_text, is_hallucination

LOG = logging.getLogger(__name__)
LABELS = {"system": "Система", "mic": "Мікрофон"}


def clock(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}"


class LiveSession:
    def __init__(self, recordings, settings_store, engine, capture_factory=None,
                 segmenter_factory=None, broadcast=lambda event: None, enqueue_hq=None):
        self.recordings = Path(recordings)
        self.settings_store = settings_store
        self.engine = engine
        if capture_factory is None:
            from .audio import CaptureSource
            capture_factory = CaptureSource
        if segmenter_factory is None:
            from .segmenter import StreamingSegmenter
            segmenter_factory = StreamingSegmenter
        self.capture_factory = capture_factory
        self.segmenter_factory = segmenter_factory
        self.broadcast = broadcast
        self.enqueue_hq = enqueue_hq
        self._lock = threading.RLock()
        self._session = None
        self._closed = False

    def _emit(self, event):
        try:
            self.broadcast(event)
        except Exception:
            LOG.exception("Не вдалося надіслати стан сесії")

    def snapshot(self):
        with self._lock:
            s = self._session
            if s is None:
                return {"active": False, "finishing": False, "id": None,
                        "sources": {"system": False, "mic": False}, "started_at": None, "backlog": 0}
            return {"active": s["active"], "finishing": s["finishing"], "id": s["id"],
                    "sources": dict(s["sources"]), "started_at": s["meta"]["started_at"],
                    "backlog": s["queue"].qsize()}

    def start(self, system=True, mic=False):
        with self._lock:
            if self._session and (self._session["active"] or self._session["finishing"]):
                return self._session["id"], False
            if self._closed:
                raise RuntimeError("Сервіс запису вже зупинено.")
            now = datetime.now().astimezone()
            sid = f"meeting_{now:%Y-%m-%d_%H-%M-%S_%f}"
            folder = self.recordings / sid
            folder.mkdir(parents=True, exist_ok=False)
            settings = self.settings_store.snapshot()
            meta = {"id": sid, "started_at": now.isoformat(), "ended_at": None,
                    "sources": {"system": bool(system), "mic": bool(mic)},
                    "live_model": settings.live_model, "language": settings.language}
            (folder / "transcript.txt").write_text(f"Транскрипція: {now:%Y-%m-%d %H:%M:%S}\n\n", encoding="utf-8")
            s = {"id": sid, "folder": folder, "meta": meta, "active": True, "finishing": False,
                 "sources": dict(meta["sources"]), "captures": {}, "segmenters": {}, "wavs": {},
                 "frames": {}, "queue": queue.Queue(), "mono_start": time.monotonic(),
                 "recent": {"system": "", "mic": ""}, "echo": EchoDeduper(), "backlog": -1,
                 "warned": False, "worker": None}
            self._session = s
            self._write_meta(s)
            worker = threading.Thread(target=self._worker, args=(s,), name="vts-live", daemon=True)
            s["worker"] = worker
            worker.start()
        self.set_sources(system, mic)
        self._emit({"type": "status"})
        return sid, True

    def _write_meta(self, s):
        path = s["folder"] / "meta.json"
        temporary = path.with_suffix(f".json.{threading.get_native_id()}.tmp")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if "job" in existing:
                s["meta"]["job"] = existing["job"]
        except (OSError, ValueError):
            pass
        temporary.write_text(json.dumps(s["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def set_sources(self, system, mic):
        stops = []
        starts = []
        with self._lock:
            s = self._session
            if not s or not s["active"]:
                return self.snapshot()
            settings = self.settings_store.snapshot()
            for source, enabled in (("system", system), ("mic", mic)):
                s["sources"][source] = bool(enabled)
                if enabled:
                    s["meta"]["sources"][source] = True
                selected = settings.system_source if source == "system" else settings.mic_source
                device = ("@DEFAULT_MONITOR@" if source == "system" else "@DEFAULT_SOURCE@") if selected == "default" else selected
                if enabled and source in s["captures"] and getattr(s["captures"][source], "device", device) != device:
                    stops.append(s["captures"].pop(source))
                    segmenter = s["segmenters"].pop(source, None)
                    if segmenter is not None:
                        self._enqueue(s, source, segmenter.flush())
                if not enabled and source in s["captures"]:
                    stops.append(s["captures"].pop(source))
                    segmenter = s["segmenters"].pop(source, None)
                    if segmenter is not None:
                        self._enqueue(s, source, segmenter.flush())
                elif enabled and source not in s["captures"]:
                    capture = self.capture_factory(source=source, device=device,
                                                  on_block=lambda src, block, arrival, session=s: self._on_block(session, src, block, arrival))
                    s["captures"][source] = capture
                    starts.append(capture)
            self._write_meta(s)
        for capture in stops:
            capture.stop()
        for capture in starts:
            capture.start()
        self._emit({"type": "status"})
        return self.snapshot()

    def _enqueue(self, s, source, segments):
        for segment in segments:
            s["queue"].put((source, segment))
        self._backlog(s)

    def _backlog(self, s):
        size = s["queue"].qsize()
        if size != s["backlog"]:
            s["backlog"] = size
            if size >= 3:
                self._emit({"type": "backlog", "size": size})

    def _on_block(self, s, source, samples, arrival):
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(data):
            return
        with self._lock:
            if not s["active"] or not s["sources"].get(source):
                return
            offset = max(0.0, float(arrival) - len(data) / 16000 - s["mono_start"])
            if source not in s["wavs"]:
                raw = (s["folder"] / ("system.wav" if source == "system" else "microphone.wav")).open("wb")
                wav = wave.open(raw, "wb")
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                s["wavs"][source] = (wav, raw)
                s["frames"][source] = 0
            wav, raw = s["wavs"][source]
            gap = int(round(offset * 16000)) - s["frames"][source]
            if gap > 8000 or (s["frames"][source] == 0 and gap > 0):
                # Write silence in bounded blocks when a source is re-enabled.
                remaining = gap
                while remaining > 0:
                    count = min(remaining, 16000)
                    wav.writeframes(b"\0\0" * count)
                    remaining -= count
                s["frames"][source] += gap
            wav.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
            raw.flush()
            s["frames"][source] += len(data)
            segmenter = s["segmenters"].get(source)
            if segmenter is None:
                segmenter = self.segmenter_factory()
                s["segmenters"][source] = segmenter
            self._enqueue(s, source, segmenter.push(data, offset))

    def stop(self):
        with self._lock:
            s = self._session
            if not s or not s["active"]:
                return (s["id"] if s else None), False
            s["active"] = False
            s["finishing"] = True
            captures = list(s["captures"].values())
            s["captures"].clear()
        # Stop outside the callback lock; capture.stop joins its reader thread.
        for capture in captures:
            capture.stop()
        with self._lock:
            for source, segmenter in s["segmenters"].items():
                self._enqueue(s, source, segmenter.flush())
            s["queue"].put(None)
        self._emit({"type": "status"})
        return s["id"], True

    def _worker(self, s):
        handle = None
        try:
            while True:
                item = s["queue"].get()
                try:
                    if item is None:
                        break
                    source, segment = item
                    if handle is None:
                        handle = self.engine.get(s["meta"]["live_model"])
                    settings = self.settings_store.snapshot()
                    prompt = (settings.glossary + " " + s["recent"][source][-200:]).strip() or None
                    live_fn = getattr(self.engine, "transcribe_live", None)
                    if live_fn is None:
                        from .engine import transcribe_live
                        live_fn = transcribe_live
                    try:
                        result = live_fn(handle, segment.audio, None if settings.language == "auto" else settings.language, prompt)
                    except Exception as exc:
                        if not is_cuda_failure(exc):
                            raise
                        # The instance that failed is unusable: replace it with a fresh one, then retry once.
                        self._emit({"type": "log", "level": "warning",
                                    "text": f"Помилка CUDA у live-транскрипції ({exc}). Перезавантажую модель і повторюю фрагмент."})
                        broken, handle = handle, None
                        discard = getattr(self.engine, "discard", None)
                        if discard is not None:
                            discard(s["meta"]["live_model"], handle=broken)
                        del broken
                        trim_memory()
                        reload = getattr(self.engine, "reload", None)
                        handle = reload(s["meta"]["live_model"]) if reload else self.engine.get(s["meta"]["live_model"])
                        result = live_fn(handle, segment.audio, None if settings.language == "auto" else settings.language, prompt)
                    text = clean_text(result.text)
                    if not text or is_hallucination(text, segment.end_s - segment.start_s):
                        continue
                    if source == "system":
                        s["echo"].remember(text, segment.start_s, segment.end_s)
                    elif settings.echo_dedupe and all(s["sources"].values()) and s["echo"].is_echo(text, segment.start_s, segment.end_s):
                        continue
                    language = getattr(result, "language", None)
                    suffix = f" ({language})" if settings.language == "auto" and language else ""
                    stamp = clock(segment.start_s)
                    line = f"[{stamp}] [{LABELS[source]}] {text}{suffix}\n"
                    with (s["folder"] / "transcript.txt").open("a", encoding="utf-8") as out:
                        out.write(line)
                    s["recent"][source] = (s["recent"][source] + " " + text)[-200:]
                    self._emit({"type": "transcript", "session_id": s["id"], "source": source,
                                "label": LABELS[source], "start": segment.start_s, "end": segment.end_s,
                                "time": stamp, "text": text, "language": language})
                except Exception as exc:
                    LOG.error("Помилка транскрипції наживо: %s", exc)
                    self._emit({"type": "log", "level": "error", "text": f"Помилка транскрипції наживо: {exc}"})
                finally:
                    s["queue"].task_done()
                    self._backlog(s)
                    if handle and handle.device == "cpu" and s["queue"].qsize() > 6 and not s["warned"]:
                        s["warned"] = True
                        self._emit({"type": "log", "level": "warning", "text": "Черга транскрипції на CPU зростає. Виправте CUDA для швидшої роботи."})
        finally:
            with self._lock:
                for wav, raw in s["wavs"].values():
                    wav.close()
                    raw.close()
                s["wavs"].clear()
                s["meta"]["ended_at"] = datetime.now().astimezone().isoformat()
                self._write_meta(s)
                s["finishing"] = False
            trim_memory()
            self._emit({"type": "status"})
            if self.settings_store.snapshot().auto_hq and self.enqueue_hq is not None and not self._closed:
                self.enqueue_hq(s["id"])

    def shutdown(self, timeout=25):
        self._closed = True
        self.stop()
        with self._lock:
            worker = self._session["worker"] if self._session else None
        if worker and worker is not threading.current_thread():
            worker.join(timeout)

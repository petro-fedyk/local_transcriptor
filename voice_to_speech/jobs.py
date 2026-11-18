"""One low priority file transcription worker with durable partial output."""
from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import uuid
import wave

from .engine import is_cuda_failure, trim_memory
from .textfilter import EchoDeduper, clean_text, is_hallucination

LOG = logging.getLogger(__name__)
TERMINAL = {"done", "error", "cancelled"}
SRT_CUE = re.compile(r"^(\d+)\n(\d+):(\d\d):(\d\d),(\d{3}) --> (\d+):(\d\d):(\d\d),(\d{3})$", re.MULTILINE)


def last_srt_cue(path):
    """Return (cue_number, end_seconds) of the last cue in an SRT file, or (0, 0.0)."""
    try:
        matches = list(SRT_CUE.finditer(Path(path).read_text(encoding="utf-8")))
    except OSError:
        return 0, 0.0
    if not matches:
        return 0, 0.0
    cue, *_, hours, minutes, seconds, millis = matches[-1].groups()
    return int(cue), int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def stamp(seconds, srt=False):
    millis = max(0, round(float(seconds) * 1000))
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}" if srt else f"{hours}:{minutes:02}:{secs:02}"


def now():
    return datetime.now().astimezone().isoformat()


class JobQueue:
    def __init__(self, recordings, settings_store, engine, broadcast=lambda event: None, live_state=None):
        self.recordings = Path(recordings)
        self.settings_store = settings_store
        self.engine = engine
        self.broadcast = broadcast
        self._live_state = live_state or (lambda: {"active": False, "finishing": False})
        self._condition = threading.Condition(threading.RLock())
        self._jobs = {}
        self._pending = []
        self._closed = False
        self._file_handle = None
        self._file_requested = None
        self._last_file_use = None
        self._worker = threading.Thread(target=self._work, name="vts-file-jobs", daemon=True)
        self._worker.start()

    @staticmethod
    def _public(job):
        return {key: value for key, value in job.items() if not key.startswith("_")}

    def _notify(self, job, force=False):
        current = time.monotonic()
        if not force and current - job.get("_last_event", 0) < .5:
            return
        job["_last_event"] = current
        event = {"type": "job", **self._public(job)}
        try:
            self.broadcast(event)
        except Exception:
            LOG.exception("Не вдалося надіслати стан задачі")
        self._save_meta(job)

    def _save_meta(self, job):
        path = self.recordings / job["session_id"] / "meta.json"
        try:
            meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"id": job["session_id"]}
            meta["job"] = self._public(job)
            if job["kind"] == "upload":
                meta["original_filename"] = job["_original_name"]
            temporary = path.with_suffix(f".json.{threading.get_native_id()}.tmp")
            temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        except (OSError, ValueError):
            LOG.warning("Не вдалося зберегти метадані задачі")

    def enqueue_upload(self, session_id, path, original_name):
        return self._enqueue("upload", session_id, path, original_name)

    def enqueue_session_hq(self, session_id):
        return self._enqueue("session_hq", session_id, None, session_id)

    def resume_job(self, job_id):
        with self._condition:
            job = self._jobs.get(job_id)
        return None if job is None else self.resume_session(job["session_id"])

    def resume_session(self, session_id):
        """Continue an interrupted job using only files on disk (works after a server restart)."""
        folder = self.recordings / session_id
        if not folder.is_dir():
            return None
        with self._condition:
            if any(job["session_id"] == session_id and job["status"] not in TERMINAL for job in self._jobs.values()):
                raise RuntimeError("Для цієї сесії вже є активна задача.")
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        previous = meta.get("job") or {}
        if previous.get("status") == "done":
            raise ValueError("Задачу вже завершено — продовжувати нічого.")
        if session_id.startswith("meeting_"):
            # HQ output is merged at the end, so the reliable way to continue is a fresh HQ pass.
            return self.enqueue_session_hq(session_id)
        originals = sorted(folder.glob("original.*"))
        if not originals:
            raise ValueError("Немає початкового файлу для продовження.")
        cue, resume_from = last_srt_cue(folder / "transcript.srt")
        name = meta.get("original_filename") or originals[0].name
        return self._enqueue("upload", session_id, originals[0], name, resume_from=resume_from, cue_start=cue)

    def _enqueue(self, kind, session_id, path, original_name, resume_from=None, cue_start=0):
        with self._condition:
            if self._closed:
                raise RuntimeError("Чергу задач уже зупинено.")
            # Reserve the job atomically, including duplicate HQ requests.
            if kind == "session_hq":
                for existing in self._jobs.values():
                    if existing["session_id"] == session_id and existing["kind"] == kind and existing["status"] not in TERMINAL:
                        return self._public(existing)
            job = {"id": uuid.uuid4().hex, "kind": kind, "session_id": session_id,
                   "status": "queued", "progress": 0.0, "eta_s": None, "error": None,
                   "created_at": now(), "started_at": None, "finished_at": None,
                   "model_used": None, "output_files": [], "_path": Path(path) if path else None,
                   "_original_name": original_name, "_cancel": threading.Event(), "_last_event": 0.0,
                   "_resume_from": resume_from, "_cue_start": cue_start}
            self._jobs[job["id"]] = job
            self._pending.append(job["id"])
            self._notify(job, True)
            self._condition.notify_all()
            return self._public(job)

    def list(self):
        with self._condition:
            return [self._public(job) for job in self._jobs.values()]

    def counts(self):
        with self._condition:
            return {name: sum(job["status"] == name for job in self._jobs.values()) for name in ("queued", "running")}

    def cancel(self, job_id):
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] not in TERMINAL:
                job["_cancel"].set()
                if job["status"] == "queued":
                    job["status"] = "cancelled"
                    job["finished_at"] = now()
                    self._append_note(job, "Мови не виявлено (лише музика, шум або тиша).")
                    self._append_note(job, "Скасовано користувачем. Частковий результат збережено.")
                self._notify(job, True)
            self._condition.notify_all()
            return self._public(job)

    def shutdown(self, timeout=25):
        with self._condition:
            self._closed = True
            for job in self._jobs.values():
                if job["status"] not in TERMINAL:
                    self.cancel(job["id"])
            self._condition.notify_all()
        if self._worker is not threading.current_thread():
            self._worker.join(timeout)

    def _unload_idle(self):
        if self._file_handle is None or self._last_file_use is None or time.monotonic() - self._last_file_use < 120:
            return
        live_name = self.settings_store.snapshot().live_model
        actual = getattr(self._file_handle, "name", self._file_requested)
        if actual != live_name and self._file_requested != live_name and hasattr(self.engine, "unload"):
            self.engine.unload(self._file_requested)
        if getattr(self._file_handle, "device", None) == "cpu" and hasattr(self.engine, "unload"):
            self.engine.unload(self._file_requested, device="cpu")
        self._file_handle = None
        self._file_requested = None
        trim_memory()

    def _work(self):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)
        except (OSError, AttributeError):
            pass
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait(timeout=1)
                    self._unload_idle()
                if self._closed and not self._pending:
                    return
                job = self._jobs[self._pending.pop(0)]
                if job["status"] == "cancelled":
                    continue
                job["status"] = "running"
                job["started_at"] = now()
                job["_started_mono"] = time.monotonic()
                self._notify(job, True)
            try:
                self._process(job)
                with self._condition:
                    if job["_cancel"].is_set():
                        job["status"] = "cancelled"
                        self._append_note(job, "Скасовано користувачем. Частковий результат збережено.")
                    else:
                        job["status"] = "done"
                        job["progress"] = 1.0
                        job["eta_s"] = 0
            except Exception as exc:
                with self._condition:
                    job["status"] = "cancelled" if job["_cancel"].is_set() else "error"
                    job["error"] = str(exc)
                    self._append_note(job, f"Помилка транскрипції: {exc}")
                LOG.exception("Помилка файлової транскрипції: %s", exc)
            finally:
                with self._condition:
                    job["finished_at"] = now()
                    self._last_file_use = time.monotonic()
                    self._notify(job, True)
                trim_memory()

    def _output_paths(self, job):
        folder = self.recordings / job["session_id"]
        basename = "transcript_hq" if job["kind"] == "session_hq" else "transcript"
        return folder / f"{basename}.txt", folder / f"{basename}.srt"

    def _append_note(self, job, note):
        transcript, _ = self._output_paths(job)
        transcript.parent.mkdir(parents=True, exist_ok=True)
        with transcript.open("a", encoding="utf-8") as out:
            out.write(f"\n{note}\n")
        partial = transcript.with_name("transcript_hq.partial.txt")
        if job["kind"] == "session_hq":
            with partial.open("a", encoding="utf-8") as out:
                out.write(f"\n{note}\n")
                out.flush()
                os.fsync(out.fileno())
            for path in (transcript, partial):
                if str(path) not in job["output_files"]:
                    job["output_files"].append(str(path))

    @staticmethod
    def _wav_duration(path):
        try:
            with wave.open(str(path), "rb") as audio:
                return audio.getnframes() / audio.getframerate()
        except (OSError, EOFError, wave.Error):
            return 0.0

    def _warning(self, text):
        LOG.warning("%s", text)
        try:
            self.broadcast({"type": "log", "level": "warning", "text": text})
        except Exception:
            LOG.exception("Не вдалося надіслати попередження задачі")

    @staticmethod
    def _close_iterator(iterator):
        close = getattr(iterator, "close", None)
        if close:
            close()

    @staticmethod
    def _wait_after_oom(job):
        return job["_cancel"].wait(2)

    def _live_busy(self):
        try:
            state = self._live_state() or {}
        except Exception:
            return True
        return bool(state.get("active") or state.get("finishing"))

    def _free_live_model(self, settings):
        """Give the file job the VRAM held by an idle live model (it is reloaded lazily on the next session)."""
        if settings.live_model == settings.file_model or self._live_busy() or not hasattr(self.engine, "unload"):
            return
        peek = getattr(self.engine, "peek", None)
        if peek is not None and peek(settings.live_model) is None:
            return
        self.engine.unload(settings.live_model)

    def _replace_broken_handle(self, job, state, settings, device):
        """A CTranslate2 instance that hit a CUDA failure is unusable forever: drop it and load a fresh one."""
        broken, state["handle"], self._file_handle = state["handle"], None, None
        discard = getattr(self.engine, "discard", None)
        if discard is not None:
            discard(settings.file_model, handle=broken)
        del broken
        self._free_live_model(settings)
        trim_memory()
        reload = getattr(self.engine, "reload", None)
        if reload is not None:
            handle = reload(settings.file_model, device=device)
        else:
            handle = self.engine.get(settings.file_model, device=device)
        state["handle"] = self._file_handle = handle
        job["model_used"] = handle.name
        return handle

    CUDA_RETRY_S = 120  # after a CPU fallback, try the GPU again this often (VRAM is often freed by then)

    def _try_return_to_cuda(self, job, state, settings, resume_from):
        """Switch a CPU-fallback job back to a fresh CUDA instance if VRAM is available again."""
        self._free_live_model(settings)
        try:
            handle = self.engine.get(settings.file_model, device="cuda")
        except Exception as exc:
            LOG.info("CUDA ще недоступна для файлової задачі: %s", exc)
            handle = None
        if handle is None or handle.device != "cuda" or handle.name != settings.file_model:
            if handle is not None and hasattr(self.engine, "unload"):
                self.engine.unload(settings.file_model, device="cuda")
            state["cpu_since"] = time.monotonic()
            return False
        cpu_handle, state["handle"], self._file_handle = state["handle"], handle, handle
        state["cpu_since"] = None
        job["model_used"] = handle.name
        if hasattr(self.engine, "unload") and getattr(cpu_handle, "device", None) == "cpu":
            self.engine.unload(settings.file_model, device="cpu")
        del cpu_handle
        trim_memory()
        note = f"VRAM звільнилась — обробку повернуто на CUDA з {stamp(resume_from)}."
        job["_fallback_notes"].append(note)
        self._warning(note)
        self._notify(job, True)
        return True

    def _file_segments(self, job, state, path, settings, resume_from=0.0):
        fn = getattr(self.engine, "transcribe_file", None)
        if fn is None:
            from .engine import transcribe_file
            fn = transcribe_file
        resume_from = float(resume_from or 0.0)
        resumed = resume_from > 0
        cuda_reloads = 0
        language = None if settings.language == "auto" else settings.language
        while not job["_cancel"].is_set():
            iterator = None
            retry_cuda = False
            try:
                options = {"clip_timestamps": [resume_from]} if resumed else {}
                iterator = iter(fn(state["handle"], path, language, settings.glossary, **options))
                while not job["_cancel"].is_set():
                    try:
                        segment, info = next(iterator)
                    except StopIteration:
                        return
                    # faster-whisper's clip timestamps and segment offsets are absolute.
                    if float(segment.end) <= resume_from + 1e-6:
                        continue
                    resume_from = max(resume_from, float(segment.end))
                    yield segment, info
                    cpu_since = state.get("cpu_since")
                    if cpu_since is not None and time.monotonic() - cpu_since >= self.CUDA_RETRY_S:
                        retry_cuda = True
                        break
                if not retry_cuda:
                    return
            except Exception as exc:
                if state["handle"].device != "cuda" or not is_cuda_failure(exc):
                    raise
                failure = str(exc)
            finally:
                if iterator is not None:
                    self._close_iterator(iterator)
                iterator = None
            resumed = True
            if retry_cuda:
                self._try_return_to_cuda(job, state, settings, resume_from)
                continue
            if self._wait_after_oom(job):
                return
            if cuda_reloads < 2:
                cuda_reloads += 1
                self._warning(f"Помилка CUDA ({failure}). Перезавантажую модель і продовжую з {stamp(resume_from)} "
                              f"(спроба {cuda_reloads}/2).")
                try:
                    self._replace_broken_handle(job, state, settings, "cuda")
                    continue
                except Exception as exc:
                    if not is_cuda_failure(exc):
                        raise
            self._replace_broken_handle(job, state, settings, "cpu")
            state["cpu_since"] = time.monotonic()
            cuda_reloads = 0
            note = f"Після повторних помилок CUDA обробку продовжено на CPU з {stamp(resume_from)}."
            job["_fallback_notes"].append(note)
            self._warning(note)
            self._notify(job, True)

    def _update_progress(self, job, positions, durations):
        total = sum(durations)
        progress = sum(min(pos, duration) for pos, duration in zip(positions, durations)) / total if total else 0.0
        job["progress"] = max(job["progress"], min(.999, progress))
        elapsed = time.monotonic() - job["_started_mono"]
        job["eta_s"] = max(0, elapsed * (1 / job["progress"] - 1)) if job["progress"] else None
        self._notify(job)

    @staticmethod
    def _write_line(out, start, end, text, durable=False):
        out.write(f"[{stamp(start)} - {stamp(end)}] {text}\n")
        out.flush()
        if durable:
            os.fsync(out.fileno())

    @staticmethod
    def _write_cue(out, cue, start, end, text):
        out.write(f"{cue}\n{stamp(start, True)} --> {stamp(end, True)}\n{text}\n\n")
        out.flush()

    def _finish_hq(self, job, records, header, echo_dedupe):
        transcript, srt = self._output_paths(job)
        systems = [row for row in records if row[2] == "Система"]
        retained = []
        for row in sorted(records, key=lambda value: (value[0], value[2] != "Система", value[1])):
            start, end, label, text = row
            if label == "Мікрофон" and echo_dedupe:
                # Each source was decoded separately; compare against system spans
                # on both sides of this microphone span before merging.
                echo_hit = False
                for sys_start, sys_end, _, sys_text in systems:
                    echo = EchoDeduper()
                    echo.remember(sys_text, sys_start, sys_end)
                    if echo.is_echo(text, start, end):
                        echo_hit = True
                        break
                if echo_hit:
                    continue
            retained.append(row)
        temporary = transcript.with_suffix(".txt.tmp")
        temporary_srt = srt.with_suffix(".srt.tmp")
        with temporary.open("w", encoding="utf-8") as txt, temporary_srt.open("w", encoding="utf-8") as subtitles:
            txt.write(header)
            for cue, (start, end, label, text) in enumerate(retained, 1):
                content = f"[{label}] {text}"
                self._write_line(txt, start, end, content)
                self._write_cue(subtitles, cue, start, end, content)
            if not retained:
                txt.write("Мови не виявлено (лише музика, шум або тиша).\n")
            for note in job["_fallback_notes"]:
                txt.write(f"\n{note}\n")
            txt.flush()
            os.fsync(txt.fileno())
            subtitles.flush()
            os.fsync(subtitles.fileno())
        temporary.replace(transcript)
        temporary_srt.replace(srt)
        directory_fd = os.open(transcript.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _process(self, job):
        settings = self.settings_store.snapshot()
        transcript, srt = self._output_paths(job)
        transcript.parent.mkdir(parents=True, exist_ok=True)
        hq = job["kind"] == "session_hq"
        partial = transcript.with_name("transcript_hq.partial.txt") if hq else None
        resume_from = 0.0 if hq else float(job.get("_resume_from") or 0.0)
        resuming = not hq and job.get("_resume_from") is not None
        header = f"Файл: {job['_original_name']}\nПочаток обробки: {job['started_at']}\n"
        if resuming:
            # Keep everything already transcribed (including the old error note) and append after it.
            with transcript.open("a", encoding="utf-8") as out:
                out.write(f"\nПродовжено з {stamp(resume_from)} ({job['started_at']})\n")
            if not srt.exists():
                srt.write_text("", encoding="utf-8")
        else:
            transcript.write_text(header, encoding="utf-8")
            srt.write_text("", encoding="utf-8")
        job["output_files"] = [str(transcript), str(srt)] + ([str(partial)] if partial else [])
        job["_fallback_notes"] = []
        if partial:
            partial.write_text(header, encoding="utf-8")
        records = []
        line_count = int(job.get("_cue_start") or 0) if resuming else 0
        new_lines = 0
        completed = False
        try:
            if job["_cancel"].is_set():
                return
            self._free_live_model(settings)
            state = {"handle": self.engine.get(settings.file_model)}
            self._file_handle = state["handle"]
            self._file_requested = settings.file_model
            job["model_used"] = state["handle"].name
            device_info = f"Модель: {state['handle'].name}\nПристрій: {state['handle'].device}\n\n"
            header += device_info
            with transcript.open("a", encoding="utf-8") as out:
                out.write(device_info)
            if partial:
                partial.write_text(header, encoding="utf-8")
            self._notify(job)
            if not hq:
                sources = [(None, job["_path"])]
            else:
                folder = self.recordings / job["session_id"]
                sources = [(label, folder / filename) for label, filename in (("Система", "system.wav"), ("Мікрофон", "microphone.wav"))
                           if (folder / filename).exists() and (folder / filename).stat().st_size > 44]
                if not sources:
                    raise FileNotFoundError("У сесії немає аудіо для транскрипції.")
            durations = [self._wav_duration(path) for _, path in sources]
            positions = [resume_from] + [0.0] * (len(sources) - 1)
            # Keep only one decoder/generator alive, even for two HQ sources.
            for index, (label, path) in enumerate(sources):
                if job["_cancel"].is_set():
                    break
                iterator = self._file_segments(job, state, path, settings, resume_from if index == 0 else 0.0)
                try:
                    with (partial or transcript).open("a", encoding="utf-8") as txt, srt.open("a", encoding="utf-8") as subtitles:
                        for segment, info in iterator:
                            start, end = float(segment.start), float(segment.end)
                            duration = getattr(info, "duration", None)
                            durations[index] = max(durations[index], float(duration or 0), end)
                            positions[index] = max(positions[index], end)
                            text = clean_text(segment.text)
                            if text and not is_hallucination(text, end - start):
                                line_count += 1
                                new_lines += 1
                                content = f"[{label}] {text}" if label else text
                                self._write_line(txt, start, end, content, durable=hq)
                                if hq:
                                    records.append((start, end, label, text))
                                else:
                                    self._write_cue(subtitles, line_count, start, end, content)
                            self._update_progress(job, positions, durations)
                finally:
                    self._close_iterator(iterator)
                if not job["_cancel"].is_set():
                    positions[index] = durations[index]
                    self._update_progress(job, positions, durations)
            completed = not job["_cancel"].is_set()
        finally:
            if hq:
                self._finish_hq(job, records, header, settings.echo_dedupe)
                if completed:
                    partial.unlink(missing_ok=True)
                    job["output_files"] = [str(transcript), str(srt)]
            else:
                with transcript.open("a", encoding="utf-8") as out:
                    if not new_lines and not resuming:
                        out.write("Мови не виявлено (лише музика, шум або тиша).\n")
                    for note in job["_fallback_notes"]:
                        out.write(f"\n{note}\n")

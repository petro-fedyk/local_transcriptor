"""Offline, resumable meeting summaries using a local llama.cpp server."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from difflib import SequenceMatcher
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import threading
import time
from urllib import error, request
import uuid

from .textfilter import _normalized
from . import guard

LOG = logging.getLogger(__name__)
SESSION_ID = re.compile(r"^(meeting|upload|summary)_[0-9_-]+$")
ALLOWED_FILES = {"transcript_hq.txt", "transcript.txt", "transcript_hq.srt", "summary_input.txt", "summary_input.srt", "summary_input.md"}
TIME = r"(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{3}))?"
RANGE = re.compile(rf"^\[{TIME}\s*-\s*{TIME}\]\s*(.*)$")
SINGLE = re.compile(rf"^\[{TIME}\]\s*(.*)$")
LABEL = re.compile(r"^\[(Система|Мікрофон)\]\s*(.*)$")
LANGUAGE = re.compile(r"\s+\([a-zA-Z]{2}\)$")
SRT_TIME = re.compile(rf"^{TIME}\s*-->\s*{TIME}$")
THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
WORD = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


@dataclass(frozen=True)
class Row:
    start: float | None
    end: float | None
    source: str | None
    text: str


def _seconds(groups):
    h, m, s, ms = groups
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms or 0) / 1000


def _content(value):
    match = LABEL.match(value.strip())
    if match:
        source = "system" if match[1] == "Система" else "mic"
        value = match[2]
    else:
        source = None
    return source, LANGUAGE.sub("", value.strip())


def parse_transcript(text: str, filename: str) -> list[Row]:
    """Parse live, HQ, SRT, or paragraph text without removing spoken words."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("Транскрипція:"))
    if filename.lower().endswith(".srt"):
        rows = []
        for block in re.split(r"\n\s*\n", text.strip()):
            lines = block.splitlines()
            if len(lines) > 1 and lines[0].strip().isdigit() and SRT_TIME.match(lines[1].strip()):
                lines = lines[1:]
            match = SRT_TIME.match(lines[0].strip()) if lines else None
            if match:
                source, content = _content(" ".join(line.strip() for line in lines[1:] if line.strip()))
                if content:
                    rows.append(Row(_seconds(match.groups()[:4]), _seconds(match.groups()[4:]), source, content))
            elif block.strip():
                rows.append(Row(None, None, None, block.strip()))
        return rows
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(RANGE.match(line) or SINGLE.match(line) for line in lines):
        rows = []
        for line in lines:
            match = RANGE.match(line)
            if match:
                source, content = _content(match[9])
                rows.append(Row(_seconds(match.groups()[:4]), _seconds(match.groups()[4:8]), source, content))
                continue
            match = SINGLE.match(line)
            if match:
                source, content = _content(match[5])
                rows.append(Row(_seconds(match.groups()[:4]), None, source, content))
            else:
                rows.append(Row(None, None, None, line))
        return rows
    return [Row(None, None, None, paragraph.strip()) for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]


def _tokens(value):
    return [(match.group(), _normalized(match.group()), match.start(), match.end()) for match in WORD.finditer(value)]


def _cut_tokens(value, indices):
    tokens = _tokens(value)
    if not indices:
        return value
    # Retain punctuation next to surviving words whenever possible.
    pieces = []
    for index, token in enumerate(tokens):
        if index in indices:
            continue
        start = token[2]
        end = tokens[index + 1][2] if index + 1 < len(tokens) else len(value)
        pieces.append(value[start:end])
    return " ".join(piece.strip() for piece in pieces).strip() if pieces else ""


def _close(a: Row, b: Row, window: float):
    return a.start is not None and b.start is not None and a.start <= (b.end if b.end is not None else b.start) + window and b.start <= (a.end if a.end is not None else a.start) + window


def _overlap(a: Row, b: Row, window: float):
    # Live transcript rows have start times only; their approximate overlap is
    # bounded by the same window used for full echo detection.
    if a.end is None or b.end is None:
        return _close(a, b, window)
    return a.start <= b.end and b.start <= a.end


def dedupe_rows(rows, window_s=8.0) -> list[Row]:
    ordered = sorted(rows, key=lambda row: (row.start is None, row.start if row.start is not None else 0))
    system = [row for row in ordered if row.source == "system" and row.start is not None]
    result = []
    previous = {}
    for row in ordered:
        if row.start is None or row.source is None:
            result.append(row)
            continue
        text = row.text
        if row.source == "mic":
            candidates = [other for other in system if _close(row, other, window_s)]
            normalized = _normalized(text)
            if normalized and any(SequenceMatcher(None, normalized, _normalized(other.text)).ratio() >= .7 for other in candidates):
                continue
            tokens = _tokens(text)
            words = [token[1] for token in tokens]
            remove = set()
            for other in candidates:
                if not _overlap(row, other, window_s):
                    continue
                other_words = [token[1] for token in _tokens(other.text)]
                for block in SequenceMatcher(None, words, other_words, autojunk=False).get_matching_blocks():
                    if block.size >= 4:
                        remove.update(range(block.a, block.a + block.size))
            text = _cut_tokens(text, remove)
            if remove and len(_tokens(text)) < 3:
                continue
        prior = previous.get(row.source)
        if prior is not None and prior.start is not None and row.start - (prior.end if prior.end is not None else prior.start) <= 3:
            previous_words = [token[1] for token in _tokens(prior.text)]
            current_words = [token[1] for token in _tokens(text)]
            count = min(len(previous_words), len(current_words))
            while count >= 4 and previous_words[-count:] != current_words[:count]:
                count -= 1
            if count >= 4:
                text = _cut_tokens(text, set(range(count)))
                if not text:
                    continue
        current = replace(row, text=text)
        result.append(current)
        previous[row.source] = current
    return result


def _stamp(seconds):
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def format_rows(rows) -> str:
    lines = []
    for row in rows:
        parts = []
        if row.start is not None:
            parts.append(f"[{_stamp(row.start)}]")
        if row.source:
            parts.append("[Система]" if row.source == "system" else "[Мікрофон]")
        parts.append(row.text)
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def chunk_rows(rows, max_chars=9000) -> list[list[Row]]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks, current, length = [], [], 0
    for row in rows:
        pieces = [row]
        prefix_size = len(format_rows([replace(row, text="")]).rstrip("\n"))
        capacity = max(1, max_chars - prefix_size)
        if len(row.text) > capacity:
            words = re.findall(r"\S+\s*", row.text)
            pieces, part = [], ""
            for word in words:
                if part and len(part) + len(word) > capacity:
                    pieces.append(replace(row, text=part.strip()))
                    part = ""
                if len(word) > capacity:
                    if part:
                        pieces.append(replace(row, text=part.strip()))
                        part = ""
                    pieces.extend(replace(row, text=word[i:i + capacity]) for i in range(0, len(word), capacity))
                else:
                    part += word
            if part:
                pieces.append(replace(row, text=part.strip()))
        for piece in pieces:
            size = len(format_rows([piece]).rstrip("\n"))
            if current and length + size + 1 > max_chars:
                chunks.append(current)
                current, length = [], 0
            current.append(piece)
            length += size + 1
    if current:
        chunks.append(current)
    return chunks


class LlmError(RuntimeError):
    pass


class LlamaServer:
    _no_webui = {}

    def __init__(self, server_path, model_path, log_path, ctx=8192, ngl=99):
        self.server_path, self.model_path, self.log_path = map(Path, (server_path, model_path, log_path))
        self.ctx, self.ngl = ctx, ngl
        self.process = None
        self._log = None
        self.port = None
        self.key = None
        self.model_used = None

    def _tail(self):
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(getattr(self, "_log_start", 0))
                return handle.read()[-3000:].decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _oom(self, exc):
        response = exc.read().decode("utf-8", errors="replace") if isinstance(exc, error.HTTPError) else ""
        value = (str(exc) + response + self._tail()).casefold()
        return any(mark in value for mark in ("out of memory", "cudamalloc", "failed to allocate", "erroroutofdevicememory"))

    def _start(self):
        if self.server_path not in self._no_webui:
            try:
                probe = subprocess.run([str(self.server_path), "--help"], capture_output=True, text=True, timeout=15)
                help_text = probe.stdout + probe.stderr
            except (OSError, subprocess.TimeoutExpired):
                help_text = ""
            self._no_webui[self.server_path] = "--no-webui" in help_text
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        self._log_start = self._log.tell()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.key = secrets.token_hex(16)
        args = [str(self.server_path), "-m", str(self.model_path), "-c", str(self.ctx), "-ngl", str(self.ngl), "--host", "127.0.0.1", "--port", str(self.port), "--api-key", self.key, "-np", "1"]
        if self._no_webui[self.server_path]:
            args.append("--no-webui")
        try:
            self.process = subprocess.Popen(args, stdout=self._log, stderr=subprocess.STDOUT,
                                            preexec_fn=lambda: os.nice(10))
        except OSError as exc:
            raise LlmError(f"Не вдалося запустити модель: {exc}") from exc
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise LlmError("Не вдалося запустити модель: " + self._tail())
            try:
                with request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                    if response.status == 200:
                        self.model_used = f"{self.model_path.stem} · " + ("GPU" if self.ngl == 99 else "частково GPU" if self.ngl else "CPU")
                        return
            except (OSError, error.URLError):
                pass
            time.sleep(.2)
        raise LlmError("Час очікування моделі вичерпано: " + self._tail())

    def __enter__(self):
        memory = guard.gpu_memory()
        if memory is not None:
            free = memory["free_mb"]
            safe_ngl = (99 if free >= guard.DESKTOP_RESERVE_MB + 3000 else
                        20 if free >= guard.DESKTOP_RESERVE_MB + 1200 else 0)
            self.ngl = min(self.ngl, safe_ngl)
        for ngl in (99, 20, 0):
            if ngl > self.ngl:
                continue
            self.ngl = ngl
            try:
                self._start()
                return self
            except Exception as exc:
                oom = self._oom(exc)
                self.stop()
                if not oom or ngl == 0:
                    raise
        raise LlmError("Не вдалося запустити модель.")

    def chat(self, messages, max_tokens, temperature=.3):
        payload = json.dumps({"model": self.model_path.stem, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}).encode()
        while True:
            req = request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            try:
                with request.urlopen(req, timeout=900) as response:
                    return json.loads(response.read())["choices"][0]["message"]["content"]
            except Exception as exc:
                if self.ngl == 0 or not self._oom(exc):
                    raise LlmError(f"Помилка відповіді моделі: {exc}") from exc
                self.stop()
                self.ngl = 20 if self.ngl == 99 else 0
                self._start()

    def stop(self):
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._log is not None:
            self._log.close()
            self._log = None

    def __exit__(self, *_):
        self.stop()


def _answer(chat, prompt, max_tokens):
    answer = chat([{"role": "system", "content": "Відповідай українською Markdown. Не вигадуй фактів, осіб або строків. Якщо даних немає, напиши «Немає.»."}, {"role": "user", "content": prompt}], max_tokens)
    return THINK.sub("", answer).strip()


def _final_sections(answer):
    headings = ("Коротко", "Теми", "Рішення", "Завдання", "Відкриті питання")
    sections = {heading: [] for heading in headings}
    current = None
    for line in answer.splitlines():
        match = re.fullmatch(r"##\s+(.+?)\s*", line)
        if match:
            if match[1] in sections:
                current = match[1]
            else:
                sections[current or "Коротко"].append("### " + match[1])
        elif line.strip() == "# Підсумок зустрічі":
            continue
        elif line.startswith("# "):
            sections[current or "Коротко"].append(line[2:])
        elif current is not None:
            sections[current].append(line)
        elif line.strip():
            sections["Коротко"].append(line)
    parts = ["# Підсумок зустрічі"]
    for heading in headings:
        content = "\n".join(sections[heading]).strip() or "Немає."
        parts.append(f"## {heading}\n{content}")
    return "\n\n".join(parts)


def summarize(rows, chat, on_progress, cancelled, state) -> str:
    chunks = chunk_rows(rows)
    notes = state.setdefault("notes", {})
    for index, chunk in enumerate(chunks):
        if cancelled():
            raise InterruptedError("Підсумок скасовано.")
        if str(index) not in notes:
            start = next((row.start for row in chunk if row.start is not None), None)
            end = next((row.end or row.start for row in reversed(chunk) if row.start is not None), None)
            span = f"{_stamp(start)[:5]}–{_stamp(end)[:5]}" if start is not None and end is not None else "без часу"
            prompt = f"Зроби стислий конспект частини {index + 1} із {len(chunks)}. Почни з «### {span}». Далі маркери: теми, рішення, домовленості/завдання (хто — що — строк, якщо сказано), відкриті питання. Зберігай часові позначки. Не додавай вигаданого.\n\n{format_rows(chunk)}"
            notes[str(index)] = _answer(chat, prompt, 1200)
        on_progress(index + 1, len(chunks), state)
    combined = "\n\n".join(notes[str(i)] for i in range(len(chunks)))
    while len(combined) > 9000:
        groups = []
        pieces = [combined[i:i + 8000] for i in range(0, len(combined), 8000)]
        for piece in pieces:
            if cancelled():
                raise InterruptedError("Підсумок скасовано.")
            groups.append(_answer(chat, "Зведи ці нотатки стисло, зберігаючи факти, рішення, завдання, відкриті питання та часові позначки:\n\n" + piece, 1800))
        next_combined = "\n\n".join(groups)
        if len(next_combined) >= len(combined):
            combined = next_combined[:9000]
            break
        combined = next_combined
    if cancelled():
        raise InterruptedError("Підсумок скасовано.")
    prompt = "Створи підсумок зустрічі рівно з такими розділами: # Підсумок зустрічі; ## Коротко (3–6 маркерів); ## Теми (для кожної теми ### Назва (HH:MM–HH:MM) та 1–3 речення); ## Рішення; ## Завдання (формат - [ ] хто — що — строк); ## Відкриті питання. Порожній розділ: Немає. Не вигадуй нічого.\n\n" + combined
    return _final_sections(_answer(chat, prompt, 3000))


def _atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{threading.get_native_id()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class SummaryManager:
    def __init__(self, recordings, config, broadcast, gpu_busy, gpu_release, server_factory=LlamaServer):
        self.recordings = Path(recordings)
        self.config = config
        self.broadcast = broadcast
        self.gpu_busy = gpu_busy
        self.gpu_release = gpu_release
        self.server_factory = server_factory
        self._condition = threading.Condition(threading.RLock())
        self._jobs = {}
        self._pending = []
        self._closed = False
        self._load()
        self._worker = threading.Thread(target=self._work, name="vts-summary", daemon=True)
        self._worker.start()

    @staticmethod
    def _public(job):
        return {key: value for key, value in job.items() if not key.startswith("_")}

    def _save(self, job):
        path = self.recordings / job["session_id"] / "meta.json"
        try:
            meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            meta["summary_job"] = self._public(job)
            _atomic_text(path, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        except (OSError, ValueError):
            LOG.exception("Не вдалося зберегти стан підсумку")

    def _notify(self, job, force=False):
        moment = time.monotonic()
        if not force and moment - job.get("_last_event", 0) < 1:
            return
        job["_last_event"] = moment
        self._save(job)
        try:
            self.broadcast({"type": "job", "job": self._public(job)})
        except Exception:
            LOG.exception("Не вдалося надіслати стан підсумку")

    def _load(self):
        if not self.recordings.is_dir():
            return
        for folder in self.recordings.iterdir():
            if not folder.is_dir() or not SESSION_ID.fullmatch(folder.name):
                continue
            try:
                job = json.loads((folder / "meta.json").read_text(encoding="utf-8")).get("summary_job")
            except (OSError, ValueError):
                continue
            if not isinstance(job, dict) or job.get("kind") != "summary" or not job.get("id"):
                continue
            if job.get("status") in ("queued", "running"):
                job["status"] = "error"
                job["error"] = "Сервер було перезапущено — натисніть Продовжити."
                job["finished_at"] = datetime.now().astimezone().isoformat()
                self._save(job)
            job["_cancel"] = threading.Event()
            job["_last_event"] = 0
            self._jobs[job["id"]] = job

    def enqueue(self, session_id, file) -> dict:
        if not SESSION_ID.fullmatch(session_id) or file not in ALLOWED_FILES or (not session_id.startswith("summary_") and file.startswith("summary_input.")):
            raise ValueError("Некоректний файл підсумку.")
        folder = self.recordings / session_id
        if not folder.is_dir() or not (folder / file).is_file():
            raise FileNotFoundError("Файл транскрипції не знайдено.")
        with self._condition:
            if self._closed:
                raise RuntimeError("Чергу підсумків зупинено.")
            for existing in self._jobs.values():
                if existing["session_id"] == session_id and existing["status"] in ("queued", "running"):
                    return self._public(existing)
            job = {"id": uuid.uuid4().hex, "kind": "summary", "session_id": session_id, "name": f"Підсумок · {file}", "file": file,
                   "status": "queued", "progress": 0.0, "eta_s": None, "stage": "У черзі", "error": None,
                   "model_used": None, "result_url": None, "created_at": datetime.now().astimezone().isoformat(),
                   "started_at": None, "finished_at": None, "_cancel": threading.Event(), "_last_event": 0}
            self._jobs[job["id"]] = job
            self._pending.append(job["id"])
            self._notify(job, True)
            self._condition.notify_all()
            return self._public(job)

    def list(self):
        with self._condition:
            return [self._public(job) for job in self._jobs.values()]

    def get(self, job_id):
        with self._condition:
            job = self._jobs.get(job_id)
            return self._public(job) if job else None

    def owns(self, job_id):
        with self._condition:
            return job_id in self._jobs

    def counts(self):
        with self._condition:
            return {key: sum(job["status"] == key for job in self._jobs.values()) for key in ("queued", "running")}

    def busy(self):
        return any(self.counts().values())

    def status(self):
        server_found = self.config.llm_server_path.is_file()
        model_found = self.config.llm_model_path.is_file()
        if not server_found:
            hint = "Виконайте vts models, щоб установити локальний сервер підсумків."
        elif not model_found:
            hint = "Виконайте vts models удома, щоб завантажити модель підсумків."
        else:
            hint = "Модель підсумків готова."
        return {"server_found": server_found, "model_found": model_found, "model": self.config.llm_model_path.name, "ready": server_found and model_found, "hint": hint}

    def cancel(self, job_id):
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] in ("queued", "running"):
                job["_cancel"].set()
                if job["status"] == "queued":
                    job["status"] = "cancelled"
                    job["finished_at"] = datetime.now().astimezone().isoformat()
                self._notify(job, True)
                self._condition.notify_all()
            return self._public(job)

    def resume(self, job_id):
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] not in ("cancelled", "error"):
                raise ValueError("Цю задачу не можна продовжити.")
            return self.enqueue(job["session_id"], job["file"])

    def shutdown(self, timeout=25):
        with self._condition:
            self._closed = True
            for job in self._jobs.values():
                if job["status"] in ("queued", "running"):
                    job["_cancel"].set()
                    if job["status"] == "queued":
                        job["status"] = "cancelled"
                        job["finished_at"] = datetime.now().astimezone().isoformat()
                        self._notify(job, True)
            self._condition.notify_all()
        if self._worker is not threading.current_thread():
            self._worker.join(timeout)

    def _wait_gpu(self, job, server=None):
        reason = self.gpu_busy()
        while reason:
            if server is not None:
                server.stop()
                server = None
            job["stage"] = reason
            self._notify(job, True)
            if job["_cancel"].wait(2):
                raise InterruptedError("Підсумок скасовано.")
            reason = self.gpu_busy()
        self.gpu_release()
        return server

    def _process(self, job):
        folder = self.recordings / job["session_id"]
        source = (folder / job["file"]).read_text(encoding="utf-8", errors="replace")
        rows = dedupe_rows(parse_transcript(source, job["file"]))
        formatted = format_rows(rows)
        _atomic_text(folder / "summary_source.txt", formatted)
        chunks = chunk_rows(rows)
        source_hash = hashlib.sha256(formatted.encode()).hexdigest()
        partial = folder / "summary.partial.json"
        try:
            state = json.loads(partial.read_text(encoding="utf-8"))
            if state.get("source_hash") != source_hash or state.get("chunk_count") != len(chunks):
                state = {}
        except (OSError, ValueError):
            state = {}
        state.update(source_hash=source_hash, chunk_count=len(chunks))
        state.setdefault("notes", {})
        _atomic_text(partial, json.dumps(state, ensure_ascii=False))
        if job["_cancel"].is_set():
            raise InterruptedError("Підсумок скасовано.")
        readiness = self.status()
        if not readiness["ready"]:
            raise LlmError(readiness["hint"])
        self._wait_gpu(job)
        job["stage"] = "Запуск моделі"
        self._notify(job, True)
        with self.server_factory(self.config.llm_server_path, self.config.llm_model_path, self.config.llm_dir / "llama-server.log") as server:
            job["model_used"] = server.model_used
            self._notify(job, True)

            def chat(messages, max_tokens):
                nonlocal server
                if job["_cancel"].is_set():
                    raise InterruptedError("Підсумок скасовано.")
                prompt = messages[-1]["content"]
                if prompt.startswith(("Зведи ці нотатки", "Створи підсумок")):
                    job["stage"] = "Зведення"
                    job["progress"] = max(job["progress"], .9 if prompt.startswith("Створи підсумок") else .85)
                    self._notify(job, True)
                if self.gpu_busy():
                    self._wait_gpu(job, server)
                    server.__enter__()
                    job["model_used"] = server.model_used
                result = server.chat(messages, max_tokens)
                job["model_used"] = server.model_used
                return result

            def progress(done, total, state):
                _atomic_text(partial, json.dumps(state, ensure_ascii=False))
                job["progress"] = .85 * done / max(total, 1)
                job["eta_s"] = (time.monotonic() - job["_started_mono"]) * (1 - job["progress"]) / job["progress"] if job["progress"] else None
                job["stage"] = f"Частина {done} з {total}"
                self._notify(job)

            result = summarize(rows, chat, progress, job["_cancel"].is_set, state)
            job["stage"] = "Зведення"
            job["progress"] = .95
            self._notify(job, True)
        if job["_cancel"].is_set():
            raise InterruptedError("Підсумок скасовано.")
        _atomic_text(folder / "summary.md", result + "\n")
        partial.unlink(missing_ok=True)
        job["result_url"] = f"/recordings/{job['session_id']}/summary.md"

    def _work(self):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)
        except (OSError, AttributeError):
            pass
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                job = self._jobs[self._pending.pop(0)]
                if job["status"] == "cancelled":
                    continue
                job["status"] = "running"
                job["started_at"] = datetime.now().astimezone().isoformat()
                job["_started_mono"] = time.monotonic()
                self._notify(job, True)
            status, failure = "error", "Помилка підсумку: невідома помилка."
            try:
                self._process(job)
                status, failure = "done", None
            except InterruptedError:
                status, failure = "cancelled", None
            except Exception as exc:
                status, failure = "error", f"Помилка підсумку: {exc}"
                LOG.exception("Помилка підсумку")
            finally:
                with self._condition:
                    job["status"] = "cancelled" if job["_cancel"].is_set() else status
                    if job["status"] == "done":
                        job["progress"], job["eta_s"], job["stage"] = 1.0, 0, "Готово"
                    elif job["status"] == "error":
                        job["error"] = failure
                    job["finished_at"] = datetime.now().astimezone().isoformat()
                    self._notify(job, True)

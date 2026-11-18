import json
import io
from pathlib import Path
import stat
import threading
import time
from urllib import error

import pytest

from voice_to_speech.config import Config
from voice_to_speech.summary import (LlamaServer, Row, SummaryManager, chunk_rows,
                                     dedupe_rows, format_rows, parse_transcript, summarize)


def wait_for(manager, job_id, status, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] == status:
            return job
        time.sleep(.02)
    pytest.fail(f"job did not become {status}: {manager.get(job_id)}")


def test_parse_formats():
    live = parse_transcript("Транскрипція: зустріч\n[00:00:01] [Система] Привіт (uk)\n[00:00:02] [Мікрофон] Так (en)\n", "transcript.txt")
    assert live == [Row(1, None, "system", "Привіт"), Row(2, None, "mic", "Так")]
    hq = parse_transcript("[00:00:01.100 - 00:00:02.500] [Система] Перше\n[00:00:03.000 - 00:00:04.000] Без мітки", "transcript_hq.txt")
    assert hq == [Row(1.1, 2.5, "system", "Перше"), Row(3, 4, None, "Без мітки")]
    srt = parse_transcript("1\n00:00:01,000 --> 00:00:04,000\n[Система] Перший\nрядок\n\n2\n00:00:05,000 --> 00:00:06,000\nДругий\n", "uploaded.srt")
    assert srt == [Row(1, 4, "system", "Перший рядок"), Row(5, 6, None, "Другий")]
    plain = parse_transcript("Один абзац.\nЩе рядок.\n\nДругий абзац.", "summary_input.md")
    assert plain == [Row(None, None, None, "Один абзац.\nЩе рядок."), Row(None, None, None, "Другий абзац.")]


def test_dedupe_full_partial_boundary_and_time():
    rows = [
        Row(1, 3, "system", "Ми обговорили запуск нового продукту"),
        Row(2, 4, "mic", "Ми обговорили запуск нового продукту"),
        Row(4, 7, "mic", "Ми обговорили запуск нового продукту і вирішили почати завтра та запросити команду на обговорення плану"),
        Row(20, 22, "mic", "Ми обговорили запуск нового продукту"),
        Row(24, 26, "system", "Тепер поговорімо про бюджет проекту"),
        Row(27, 29, "system", "поговорімо про бюджет проекту і строки виконання"),
    ]
    result = dedupe_rows(rows)
    assert result[0] == rows[0]
    assert len(result) == 5
    assert "вирішили почати завтра" in result[1].text
    assert result[2] == rows[3]
    assert result[4].text.startswith("і строки")
    assert format_rows([result[0]]).startswith("[00:00:01] [Система]")
    assert dedupe_rows([Row(None, None, None, "Повтор"), Row(None, None, None, "Повтор")]) == [Row(None, None, None, "Повтор")] * 2


def test_partial_echo_requires_time_overlap():
    system = Row(0, 2, "system", "обговорили новий важливий план сьогодні")
    mic = Row(4, 6, "mic", "обговорили новий важливий план сьогодні та додали бюджет і строки закупівель відповідальних осіб графік перевірок звітності ризиків запуску")
    assert dedupe_rows([system, mic]) == [system, mic]


def test_chunk_rows_respects_size_and_order():
    rows = [Row(i, None, "system", f"Рядок {i} " + "слово " * 8) for i in range(8)]
    chunks = chunk_rows(rows, max_chars=130)
    assert [row for chunk in chunks for row in chunk] == rows
    assert all(len(format_rows(chunk)) <= 130 for chunk in chunks)
    long = chunk_rows([Row(None, None, None, "Слово " * 40)], max_chars=50)
    assert all(len(format_rows(chunk)) <= 50 for chunk in long)
    timed = chunk_rows([Row(1, None, "system", "Слово " * 40)], max_chars=50)
    assert all(len(format_rows(chunk)) <= 50 for chunk in timed)


def test_summarize_resume_and_hierarchical_reduce():
    rows = [Row(None, None, None, "слово " * 1400) for _ in range(3)]
    calls = []
    def chat(messages, max_tokens):
        prompt = messages[-1]["content"]
        calls.append(prompt)
        if "Зроби стислий конспект" in prompt:
            return "<think>hidden</think>### 00:00–00:01\n" + "- Факт.\n" * 1500
        if "Зведи ці нотатки" in prompt:
            return "<think>hidden</think>- Стислий факт."
        return "# Підсумок зустрічі\n## Коротко\n- Факт\n## Теми\nНемає.\n## Рішення\nНемає.\n## Завдання\nНемає.\n## Відкриті питання\nНемає."
    state = {"notes": {"0": "### 00:00–00:01\n" + "- Факт.\n" * 1500}}
    output = summarize(rows, chat, lambda *args: None, lambda: False, state)
    assert all(heading in output for heading in ("# Підсумок зустрічі", "## Коротко", "## Теми", "## Рішення", "## Завдання", "## Відкриті питання"))
    assert "hidden" not in output
    assert any("Зведи ці нотатки" in call for call in calls)
    assert sum("Зроби стислий конспект" in call for call in calls) == len(chunk_rows(rows)) - 1


def test_summarize_fills_missing_final_sections():
    def chat(messages, max_tokens):
        if "Зроби стислий конспект" in messages[-1]["content"]:
            return "### без часу\n- Факт"
        return "<think>hidden</think>## Коротко\n- Факт"
    output = summarize([Row(None, None, None, "Факт")], chat,
                       lambda *_: None, lambda: False, {})
    assert output.startswith("# Підсумок зустрічі\n\n## Коротко\n- Факт")
    assert output.count("## ") == 5
    assert output.endswith("## Відкриті питання\nНемає.")
    assert "hidden" not in output


class FakeServer:
    started = 0
    entered = threading.Event()
    release = threading.Event()
    block = False

    def __init__(self, *_):
        self.model_used = "fake · CPU"

    def __enter__(self):
        type(self).started += 1
        return self

    def __exit__(self, *_):
        pass

    def stop(self):
        pass

    def chat(self, messages, max_tokens):
        type(self).entered.set()
        if type(self).block:
            type(self).release.wait(5)
        if "Зроби стислий конспект" in messages[-1]["content"]:
            return "### 00:00–00:01\n- Факт"
        return "# Підсумок зустрічі\n## Коротко\n- Факт\n## Теми\nНемає.\n## Рішення\nНемає.\n## Завдання\nНемає.\n## Відкриті питання\nНемає."


def make_manager(tmp_path, busy=lambda: None, release=lambda: None, broadcast=lambda _: None):
    config = Config(data_dir=tmp_path)
    config.llm_server_path.parent.mkdir(parents=True)
    config.llm_server_path.touch()
    config.llm_model_path.parent.mkdir(parents=True)
    config.llm_model_path.touch()
    folder = config.recordings / "meeting_2026-01-01_00-00-00"
    folder.mkdir(parents=True)
    (folder / "transcript.txt").write_text("[00:00:01] [Система] Перша тема\n[00:00:02] [Мікрофон] Друга тема", encoding="utf-8")
    manager = SummaryManager(config.recordings, config, broadcast, busy, release, FakeServer)
    return manager, folder


def test_manager_complete_wait_and_invalid(tmp_path):
    gate = threading.Event()
    releases, events = [], []
    manager, folder = make_manager(tmp_path, lambda: "Очікує завершення запису" if not gate.is_set() else None, lambda: releases.append(1), events.append)
    try:
        with pytest.raises(ValueError):
            manager.enqueue(folder.name, "../secret.txt")
        job = manager.enqueue(folder.name, "transcript.txt")
        assert manager.enqueue(folder.name, "transcript.txt")["id"] == job["id"]
        wait_for(manager, job["id"], "running")
        gate.set()
        done = wait_for(manager, job["id"], "done")
        assert done["result_url"].endswith("/summary.md")
        assert (folder / "summary.md").is_file()
        assert (folder / "summary_source.txt").is_file()
        assert not (folder / "summary.partial.json").exists()
        assert releases and any(event.get("job", {}).get("status") == "done" for event in events)
        assert json.loads((folder / "meta.json").read_text())["summary_job"]["status"] == "done"
    finally:
        manager.shutdown()


def test_manager_cancel_and_resume(tmp_path):
    FakeServer.block = True
    FakeServer.entered.clear()
    FakeServer.release.clear()
    manager, folder = make_manager(tmp_path)
    try:
        job = manager.enqueue(folder.name, "transcript.txt")
        assert FakeServer.entered.wait(2)
        manager.cancel(job["id"])
        FakeServer.release.set()
        wait_for(manager, job["id"], "cancelled")
        assert (folder / "summary.partial.json").exists()
        FakeServer.block = False
        resumed = manager.resume(job["id"])
        wait_for(manager, resumed["id"], "done")
        assert not (folder / "summary.partial.json").exists()
    finally:
        FakeServer.block = False
        FakeServer.release.set()
        manager.shutdown()


def test_manager_restart_marks_interrupted_and_resumes(tmp_path):
    manager, folder = make_manager(tmp_path, lambda: "Очікує завершення запису")
    job = manager.enqueue(folder.name, "transcript.txt")
    wait_for(manager, job["id"], "running")
    manager.shutdown()
    meta = json.loads((folder / "meta.json").read_text())
    meta["summary_job"]["status"] = "running"
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    restarted = SummaryManager(manager.recordings, manager.config, lambda _: None, lambda: None, lambda: None, FakeServer)
    try:
        interrupted = restarted.get(job["id"])
        assert interrupted["status"] == "error"
        assert "перезапущено" in interrupted["error"]
        resumed = restarted.resume(job["id"])
        wait_for(restarted, resumed["id"], "done")
    finally:
        restarted.shutdown()


def test_status_missing_model_hint(tmp_path, monkeypatch):
    monkeypatch.delenv("VTS_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("VTS_LLM_MODEL", raising=False)
    config = Config(data_dir=tmp_path)
    config.llm_server_path.parent.mkdir(parents=True)
    config.llm_server_path.touch()
    manager = SummaryManager(config.recordings, config, lambda _: None,
                             lambda: None, lambda: None, FakeServer)
    try:
        status = manager.status()
        assert status["ready"] is False
        assert status["model_found"] is False
        assert status["server_found"] is True
        assert status["hint"] == "Виконайте vts models удома, щоб завантажити модель підсумків."
    finally:
        manager.shutdown()


def test_llama_chat_retries_oom_response(tmp_path, monkeypatch):
    server = LlamaServer(tmp_path / "server", tmp_path / "model.gguf", tmp_path / "llm.log")
    server.port, server.key = 12345, "secret"
    attempts = []
    monkeypatch.setattr(server, "_start", lambda: attempts.append(server.ngl))

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def read(self):
            return '{"choices":[{"message":{"content":"Готово"}}]}'.encode()

    def urlopen(*_, **__):
        if not attempts:
            raise error.HTTPError("http://127.0.0.1", 500, "server error", {},
                                  io.BytesIO(b'{"error":"CUDA out of memory"}'))
        return Response()

    monkeypatch.setattr("voice_to_speech.summary.request.urlopen", urlopen)
    assert server.chat([{"role": "user", "content": "Привіт"}], 20) == "Готово"
    assert attempts == [20]


def test_llama_server_fake_script(tmp_path):
    import socket
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("sandbox blocks loopback sockets")
    script = tmp_path / "llama-server"
    script.write_text('''#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
if '--help' in sys.argv:
    print('--no-webui'); sys.exit(0)
open(sys.argv[sys.argv.index('-m') + 1] + '.args', 'w').write(' '.join(sys.argv[1:]))
key = sys.argv[sys.argv.index('--api-key') + 1]
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/health' else 404); self.end_headers()
    def do_POST(self):
        authorized = self.headers.get('Authorization') == 'Bearer ' + key
        self.send_response(200 if authorized else 401); self.end_headers()
        if authorized: self.wfile.write(json.dumps({'choices':[{'message':{'content':'Готово'}}]}).encode())
    def log_message(self, *_): pass
HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port') + 1])), Handler).serve_forever()
''')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "model.gguf"
    model.touch()
    server = LlamaServer(script, model, tmp_path / "llm.log")
    with server:
        assert server.chat([{"role": "user", "content": "Привіт"}], 20) == "Готово"
        args = Path(str(model) + ".args").read_text()
        assert "--host 127.0.0.1" in args and "--no-webui" in args and "-np 1" in args
        assert server.model_used.endswith("GPU")
    assert server.process is None

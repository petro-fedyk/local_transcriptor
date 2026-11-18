"""Keep active work awake and reserve video memory for the desktop."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import threading
import time

LOG = logging.getLogger(__name__)


def _reserve_mb() -> int:
    try:
        return max(0, int(os.environ.get("VTS_VRAM_RESERVE_MB", "900")))
    except ValueError:
        LOG.warning("Некоректне значення VTS_VRAM_RESERVE_MB; використовую 900 МіБ.")
        return 900


DESKTOP_RESERVE_MB = _reserve_mb()
_gpu_lock = threading.Lock()
_gpu_cached_at = 0.0
_gpu_cached_value = None


def gpu_memory() -> dict[str, int] | None:
    """Read free VRAM, caching successful and failed probes for two seconds."""
    global _gpu_cached_at, _gpu_cached_value
    with _gpu_lock:
        now = time.monotonic()
        if now - _gpu_cached_at < 2:
            return _gpu_cached_value
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True,
            )
            first = result.stdout.strip().splitlines()[0]
            total, free = (int(part.strip()) for part in first.split(",")[:2])
            value = {"total_mb": total, "free_mb": free} if 0 <= free <= total else None
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            value = None
        _gpu_cached_at, _gpu_cached_value = now, value
        return value


class SleepGuard:
    """Hold a logind inhibitor only while work is active."""

    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._what = None
        self._error = None
        self._last_attempt = 0.0
        atexit.register(self.shutdown)

    def _stop(self):
        process = self._process
        self._process = None
        self._what = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            if process.stderr is not None:
                process.stderr.close()

    def update(self, busy: bool) -> None:
        with self._lock:
            if not busy:
                self._stop()
                self._error = None
                return
            if self._process is not None and self._process.poll() is None:
                return
            self._stop()
            if self._error and time.monotonic() - self._last_attempt < 5:
                return
            self._last_attempt = time.monotonic()
            if shutil.which("systemd-inhibit") is None:
                self._error = "systemd-inhibit недоступний; блокування сну не працює."
                LOG.warning(self._error)
                return
            for what in ("sleep:idle:handle-lid-switch", "sleep:idle"):
                command = ["systemd-inhibit", f"--what={what}", "--who=Локальна транскрипція",
                           "--why=Йде запис або обробка", "--mode=block", "sleep", "infinity"]
                try:
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                               start_new_session=False)
                    try:
                        _, stderr = process.communicate(timeout=0.15)
                    except subprocess.TimeoutExpired:
                        self._process, self._what, self._error = process, what, None
                        return
                    reason = stderr.decode("utf-8", errors="replace").strip()
                    self._error = reason or f"systemd-inhibit завершився з кодом {process.returncode}."
                except OSError as exc:
                    self._error = str(exc)
                LOG.warning("Не вдалося заблокувати %s: %s", what, self._error)

    def state(self) -> dict:
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                self._error = "Блокування сну несподівано завершилося."
                self._stop()
            return {"active": self._process is not None, "what": self._what, "error": self._error}

    def shutdown(self) -> None:
        with self._lock:
            self._stop()

"""PulseAudio/PipeWire capture through the local parec utility."""
from __future__ import annotations
import os
import re
import select
import subprocess
import threading
import time
import numpy as np


def _parse_pactl(output):
    items = []
    current = None
    for line in output.splitlines():
        stripped = line.strip()
        header = re.fullmatch(r'(?:Source|Sink Input) #(\d+)', stripped)
        if header:
            current = {'index': int(header.group(1)), 'properties': {}}
            items.append(current)
            continue
        if current is None:
            continue
        property_match = re.fullmatch(r'([\w.-]+)\s*=\s*(.*)', stripped)
        if property_match:
            value = property_match.group(2)
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = re.sub(r'\\([\\"])', r'\1', value[1:-1])
            current['properties'][property_match.group(1)] = value
            continue
        key, separator, value = stripped.partition(':')
        field = {'Name': 'name', 'Description': 'description',
                 'Monitor of Sink': 'monitor_of_sink'}.get(key)
        if separator and field:
            current[field] = value.strip()
    return items


def _pactl(kind):
    environment = {**os.environ, 'LC_ALL': 'C'}
    output = subprocess.check_output(['pactl', 'list', kind], env=environment, timeout=5)
    return _parse_pactl(output.decode('utf-8', errors='replace'))


def list_sources():
    result = {'monitors': [], 'microphones': [], 'apps': [], 'error': None}
    try:
        for item in _pactl('sources'):
            entry = {'name': item['name'], 'description': item.get('description', item['name'])}
            monitor = item.get('monitor_of_sink')
            result['monitors' if monitor not in (None, '', 'n/a', 4294967295, '4294967295') else 'microphones'].append(entry)
        for item in _pactl('sink-inputs'):
            props = item.get('properties', {})
            name = props.get('application.name', props.get('application.process.binary', str(item['index'])))
            result['apps'].append({'name': name, 'media': props.get('media.name', ''), 'index': item['index']})
    except Exception as exc:
        result['error'] = f'Не вдалося отримати аудіоджерела: {exc}'
    return result


class CaptureSource:
    def __init__(self, callback=None, source='system', device=None, popen_factory=None,
                 samplerate=16000, blocksize=8000, on_block=None):
        self.callback = on_block or callback
        if self.callback is None:
            raise ValueError('callback is required')
        self.source, self.device = source, device or 'default'
        self.samplerate, self.blocksize = samplerate, blocksize
        self._popen = popen_factory or subprocess.Popen
        self._stop = threading.Event()
        self._thread = None
        self._process = None
        self.error = None

    def _command(self):
        cmd = ['parec', '--raw', '--format=float32le', '--rate=' + str(self.samplerate), '--channels=1', '--latency-msec=200', '--client-name=voice-to-speech', '--stream-name=' + self.source]
        if self.device.startswith('app:'):
            target = self.device[4:]
            matches = [item for item in _pactl('sink-inputs')
                       if item.get('properties', {}).get('application.name') == target]
            if not matches:
                raise RuntimeError('Застосунок не відтворює звук')
            cmd.append('--monitor-stream=' + str(matches[0]['index']))
        else:
            device = self.device
            if device == 'default':
                device = '@DEFAULT_MONITOR@' if self.source == 'system' else '@DEFAULT_SOURCE@'
            cmd.append('--device=' + device)
        return cmd

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name='capture-' + self.source)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                cmd = self._command()
                process = self._popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                self._process = process
                pending = bytearray()
                resolved = time.monotonic()
                try:
                    stream_fd = process.stdout.fileno()
                except (AttributeError, OSError, ValueError):
                    stream_fd = None
                while not self._stop.is_set():
                    if self.device.startswith('app:') and time.monotonic() - resolved >= 2:
                        if cmd != self._command():
                            break
                        resolved = time.monotonic()
                    if stream_fd is not None:
                        if not select.select([stream_fd], [], [], .2)[0]:
                            if getattr(process, 'poll', lambda: None)() is not None:
                                break
                            continue
                        block = os.read(stream_fd, self.blocksize * 4 - len(pending))
                    else:
                        block = process.stdout.read(self.blocksize * 4 - len(pending))
                    if not block:
                        break
                    pending.extend(block)
                    if len(pending) == self.blocksize * 4:
                        self.callback(self.source, np.frombuffer(pending, dtype='<f4').copy(), time.monotonic())
                        pending.clear()
                self.error = None
            except Exception as exc:
                self.error = str(exc)
            finally:
                self._terminate()
            self._stop.wait(1)

    def _terminate(self):
        process, self._process = self._process, None
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=1)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

    def stop(self):
        self._stop.set()
        self._terminate()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

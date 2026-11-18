"""Offline Whisper lifecycle and serialized inference adapters."""
from __future__ import annotations
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import ctypes
import gc
import threading
import warnings
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from .textfilter import clean_text, is_hallucination
from . import guard

# NVIDIA pip wheels keep libraries outside the dynamic loader's usual paths.
def _preload_cuda():
    import sys
    handles = []
    dirs = []
    for root in sys.path:
        for directory in Path(root or '.').glob('nvidia/*/lib'):
            if directory.is_dir():
                dirs.append(str(directory.resolve()))
    os.environ['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(dirs + [os.environ.get('LD_LIBRARY_PATH', '')]))
    for filename in ('libcublas.so.12', 'libcublasLt.so.12', 'libcudnn.so.9'):
        for directory in dirs:
            path = Path(directory) / filename
            if path.is_file():
                try:
                    handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
                    break
                except OSError:
                    pass
    return tuple(handles)
_CUDA_LIBRARIES = _preload_cuda()
# Older/newer PyAV builds differ in whether open accepts metadata_errors.
try:
    import av
    import io
    try:
        av.open(io.BytesIO(b''), metadata_errors='ignore')
    except TypeError as exc:
        if 'metadata_errors' in str(exc):
            _av_open = av.open
            def _compatible_av_open(*args, **kwargs):
                kwargs.pop('metadata_errors', None)
                return _av_open(*args, **kwargs)
            av.open = _compatible_av_open
    except Exception:
        pass
except ImportError:
    pass
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

_CUDA_HINT = 'CUDA зламалась після сну. Виконайте: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm, потім перезапустіть сервер (vts stop). Постійне виправлення — див. README.'

class ModelMissingError(RuntimeError):
    pass


def is_cuda_failure(exc):
    """Recognize CUDA errors, including the invalid-device error after an OOM."""
    markers = ('out of memory', 'cudaerrorinvaliddevice',
               'invalid device ordinal', 'cuda failed')
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(marker in str(exc).casefold() for marker in markers):
            return True
        exc = exc.__cause__ or exc.__context__
    return False

@dataclass
class ModelHandle:
    model: object
    name: str
    device: str
    lock: object = field(default_factory=threading.RLock)

class ModelManager:
    def __init__(self, models_dir=None, device='auto'):
        self.models_dir = Path(models_dir or '~/.local/share/voice-to-speech/models').expanduser()
        self.device = device
        self._handles, self._states = {}, {}
        self._lock = threading.RLock()
        self._cuda_hint = None

    @property
    def cuda_hint(self):
        return self._cuda_hint

    def _device(self):
        if self.device != 'auto':
            return self.device
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count():
                return 'cuda'
            if Path('/dev/nvidia0').exists():
                try:
                    driver = ctypes.CDLL('libcuda.so.1')
                    if driver.cuInit(0) == 999:
                        self._cuda_hint = _CUDA_HINT
                except OSError:
                    pass
            return 'cpu'
        except Exception:
            return 'cpu'

    def _path(self, name):
        path = self.models_dir / name
        if not (path / 'model.bin').is_file():
            raise ModelMissingError(f'Модель {name} відсутня в {path}. Виконайте vts models для встановлення моделей.')
        return str(path)

    def _load(self, name, device):
        source = self._path(name)
        if WhisperModel is None:
            raise RuntimeError('faster-whisper недоступний')
        return WhisperModel(source, device=device, compute_type='int8_float16' if device == 'cuda' else 'int8',
                            cpu_threads=4, num_workers=1, local_files_only=True)

    def _cuda_has_headroom(self, name):
        memory = guard.gpu_memory()
        if memory is None:
            return True
        # CUDA uses int8_float16 here, roughly half a float16 model's VRAM.
        estimate = {'large-v3-turbo': 1700, 'large-v3': 3200}.get(name, 3200) // 2
        if memory['free_mb'] - estimate >= guard.DESKTOP_RESERVE_MB:
            return True
        self._cuda_hint = 'Недостатньо відеопам’яті з запасом для робочого столу — модель запущено на CPU.'
        warnings.warn(self._cuda_hint, RuntimeWarning)
        return False

    def get(self, name, device=None):
        if device not in (None, 'cpu', 'cuda'):
            raise ValueError('Невідомий пристрій моделі.')
        key = name if device is None else f'{name}@{device}'
        with self._lock:
            if key in self._handles:
                return self._handles[key]
            target = device or self._device()
            if target == 'cuda' and not self._cuda_has_headroom(name):
                target = 'cpu'
            shared_key = name if device is not None else f'{name}@cpu'
            shared = self._handles.get(shared_key)
            if target == 'cpu' and shared is not None and shared.device == 'cpu' and shared.name == name:
                self._handles[key] = shared
                self._states[key] = {'state':'ready', 'device':'cpu', 'error':None}
                return shared
            self._states[key] = {'state':'loading', 'device':target, 'error':None}
            actual = name
            try:
                self._path(name)
                try:
                    model = self._load(name, target)
                except Exception as exc:
                    if target != 'cuda':
                        raise
                    if '999' in str(exc) or 'cuInit' in str(exc):
                        self._cuda_hint = _CUDA_HINT
                    model = None
                    if name == 'large-v3':
                        shared = self._handles.get('large-v3-turbo')
                        if shared is not None:
                            self._handles[key] = shared
                            self._states[key] = {'state':'ready', 'device':shared.device, 'error':None}
                            warnings.warn(f'Не вдалося завантажити {name} на CUDA: {exc}. Використовується large-v3-turbo ({shared.device}).', RuntimeWarning)
                            return shared
                        try:
                            if not self._cuda_has_headroom('large-v3-turbo'):
                                raise RuntimeError('Недостатньо відеопам’яті для CUDA.')
                            model = self._load('large-v3-turbo', 'cuda')
                            actual = 'large-v3-turbo'
                        except Exception:
                            try:
                                model = self._load('large-v3-turbo', 'cpu')
                                actual, target = 'large-v3-turbo', 'cpu'
                            except Exception:
                                pass
                    if model is None:
                        target = 'cpu'
                        model = self._load(name, target)
                    warnings.warn(f'Не вдалося завантажити {name} на CUDA: {exc}. Використовується {actual} ({target}).', RuntimeWarning)
                handle = ModelHandle(model, actual, target)
                self._handles[key] = handle
                self._states[key] = {'state':'ready', 'device':target, 'error':None}
                return handle
            except Exception as exc:
                self._states[key] = {'state':'error', 'device':target, 'error':str(exc)}
                raise

    def states(self):
        # Do not wait for an in-progress model load to report loading state.
        return {name: dict(state) for name, state in self._states.copy().items()}

    def peek(self, name, device=None):
        """Inspect a cached handle without loading a model."""
        key = name if device is None else f'{name}@{device}'
        with self._lock:
            return self._handles.get(key)

    def unload(self, name, device=None):
        key = name if device is None else f'{name}@{device}'
        with self._lock:
            self._handles.pop(key, None)
            self._states.pop(key, None)
        trim_memory()

    def discard(self, name, device=None, handle=None):
        """Evict a failed model and every cache alias of that same instance."""
        if device not in (None, 'cpu', 'cuda'):
            raise ValueError('Невідомий пристрій моделі.')
        with self._lock:
            if handle is not None:
                targets = [handle]
            else:
                key = name if device is None else f'{name}@{device}'
                target = self._handles.get(key)
                if target is None and device is None:
                    target = self._handles.get(f'{name}@cuda')
                targets = [target] if target is not None else []
            identities = {id(target) for target in targets}
            for key, cached in tuple(self._handles.items()):
                if id(cached) in identities:
                    del self._handles[key]
                    self._states.pop(key, None)
            del targets
            del handle
        trim_memory()

    def reload(self, name, device=None):
        """Replace a cached instance with a newly constructed model."""
        self.discard(name, device=device)
        handle = self.get(name, device=device)
        if device == 'cuda':
            # A later ordinary get(name) must reuse the recovered instance.
            with self._lock:
                self._handles[name] = handle
                self._states[name] = {'state':'ready', 'device':handle.device, 'error':None}
        return handle

@dataclass
class TranscriptResult:
    text: str
    language: str | None
    segments_info: list

_OPTIONS = dict(beam_size=5, best_of=5, patience=1.0, temperature=(0,.2,.4,.6,.8,1),
    compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=.6)

def build_prompt(language, text=None):
    styles = {
        'uk': 'Добре, домовились. Тоді зробимо так, як обговорювали, — і перевіримо результат.',
        'en': "Okay, sounds good. Let's do it the way we discussed, and then check the results.",
    }
    return ' '.join(part for part in (styles.get(language), (text or '').strip()) if part) or None

def transcribe_live(handle, audio, language=None, prompt=None):
    with handle.lock:
        segments, info = handle.model.transcribe(np.asarray(audio, dtype=np.float32), language=language,
            initial_prompt=build_prompt(language, prompt), condition_on_previous_text=False, vad_filter=False,
            without_timestamps=True, **_OPTIONS)
        accepted = []
        for segment in segments:
            if ((getattr(segment, 'no_speech_prob', 0) > .6 and getattr(segment, 'avg_logprob', 0) < -1)
                    or getattr(segment, 'compression_ratio', 0) > 2.4):
                continue
            text = clean_text(segment.text)
            if text and not is_hallucination(text, segment.end-segment.start):
                accepted.append((segment, text))
    return TranscriptResult(' '.join(text for _,text in accepted), getattr(info, 'language', language),
        [{'start':s.start, 'end':s.end, 'text':text} for s,text in accepted])

def transcribe_file(handle, audio_path, language=None, glossary=None, clip_timestamps=None):
    clip_options = {'clip_timestamps': clip_timestamps} if clip_timestamps is not None else {}
    with handle.lock:
        segments, info = handle.model.transcribe(str(audio_path), language=language, initial_prompt=build_prompt(language, glossary),
            condition_on_previous_text=True, vad_filter=True,
            vad_parameters={'min_silence_duration_ms':500, 'speech_pad_ms':400}, **clip_options, **_OPTIONS)
        iterator = iter(segments)
    try:
        while True:
            with handle.lock:
                try:
                    segment = next(iterator)
                except StopIteration:
                    return
            yield segment, info
    finally:
        close = getattr(iterator, 'close', None)
        if close is not None:
            with handle.lock:
                close()

def trim_memory():
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

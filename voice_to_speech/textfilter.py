"""Transcript cleanup and overlapping microphone echo detection."""
import re
import time
import unicodedata
from difflib import SequenceMatcher

_HALLUCINATIONS = {'дякую за перегляд', 'дякуємо за перегляд', 'підписуйтесь на канал', 'субтитри створено', 'субтитры сделал', 'субтитры создавал', 'редактор субтитров', 'корректор', 'продолжение следует', 'спасибо за просмотр', 'thank you for watching', 'thanks for watching', 'please subscribe', 'subtitles by', 'amaraorg', 'you'}

def clean_text(text):
    value = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text or ''))).strip()
    # Prefer the shortest repeated phrase and collapse four or more copies.
    for size in range(1, 9):
        pattern = r'(?<!\w)((?:\w+[^\w]+){' + str(size-1) + r'}\w+)(?:[\W_]+\1){3,}(?!\w)'
        value = re.sub(pattern, r'\1', value, flags=re.IGNORECASE)
    return value if any(ch.isalnum() for ch in value) else ''

def _normalized(text):
    return ' '.join(''.join(ch if ch.isalnum() or ch.isspace() else '' for ch in clean_text(text).casefold()).split())

def is_hallucination(text, duration_s=0):
    lowered = str(text).casefold()
    if 'amara.org' in lowered or 'dimatorzok' in lowered:
        return True
    return duration_s < 6 and _normalized(text) in _HALLUCINATIONS

class EchoDeduper:
    def __init__(self, window_s=8, ratio=.7, window_seconds=None):
        self.window_s = window_s if window_seconds is None else window_seconds
        self.ratio = ratio
        self._system = []
        self._seen = {}

    def remember(self, system_text, start_s, end_s):
        self._system.append((_normalized(system_text), start_s, end_s))
        self._system = [x for x in self._system if x[2] >= start_s-self.window_s]

    def is_echo(self, mic_text, start_s, end_s):
        value = _normalized(mic_text)
        return bool(value) and any(start_s <= end+self.window_s and end_s >= start-self.window_s and
            SequenceMatcher(None, value, text).ratio() >= self.ratio for text, start, end in self._system)

    def is_duplicate(self, text, now=None):
        now = time.monotonic() if now is None else now
        value = _normalized(text)
        self._seen = {k:v for k,v in self._seen.items() if now-v <= self.window_s}
        duplicate = bool(value) and value in self._seen
        if value and not duplicate:
            self._seen[value] = now
        return duplicate

    def reset(self):
        self._system.clear()
        self._seen.clear()

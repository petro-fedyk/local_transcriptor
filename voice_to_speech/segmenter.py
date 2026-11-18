"""Buffered streaming segmentation using the bundled Silero VAD."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Segment:
    audio: np.ndarray
    start_s: float
    end_s: float

class StreamingSegmenter:
    def __init__(self, sample_rate=16000, vad_fn=None, min_silence_s=.8,
                 max_segment_s=25, pad_s=.25, min_speech_s=.4):
        self.sample_rate = sample_rate
        self.vad_fn = vad_fn
        self.min_silence_s, self.max_segment_s = min_silence_s, max_segment_s
        self.pad_s, self.min_speech_s = pad_s, min_speech_s
        self.reset()

    def reset(self):
        self._audio = np.empty(0, dtype=np.float32)
        self._start = None
        self._new_samples = 0

    def _speech(self):
        if self.vad_fn is not None:
            return self.vad_fn(self._audio)
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        return get_speech_timestamps(self._audio, VadOptions(threshold=.5,
            min_silence_duration_ms=300, speech_pad_ms=0), sampling_rate=self.sample_rate)

    def push(self, samples, t_start_offset_s):
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        out = []
        if self._start is not None:
            expected = self._start + len(self._audio)/self.sample_rate
            if t_start_offset_s > expected + 1.0:
                out.extend(self.flush())
        if self._start is None:
            self._start = t_start_offset_s
        self._audio = np.concatenate((self._audio, data))
        self._new_samples += len(data)
        if self._new_samples < self.sample_rate:
            return out
        self._new_samples = 0
        return out + self._emit(False)

    def feed(self, audio, start_s=None):
        return self.push(audio, start_s if start_s is not None else (self._start or 0))

    def _discard(self, count):
        self._audio = self._audio[count:]
        self._start += count/self.sample_rate

    def _emit(self, flush):
        out = []
        sr = self.sample_rate
        while len(self._audio):
            speech = self._speech()
            if not speech:
                if flush:
                    self.reset()
                elif len(self._audio) > 3*sr:
                    self._discard(len(self._audio)-int(.5*sr))
                break
            last_end = int(speech[-1]['end'])
            hard_limit = int(self.max_segment_s*sr)
            forced = len(self._audio) >= hard_limit
            complete = flush or (len(self._audio)-last_end)/sr >= self.min_silence_s
            if not complete and not forced:
                break
            cut = last_end
            if forced:
                gaps = [int(item['end']) for index,item in enumerate(speech)
                        if item['end'] <= hard_limit and
                        (speech[index+1]['start'] if index+1 < len(speech) else len(self._audio))-item['end'] >= .2*sr]
                cut = max(gaps) if gaps else hard_limit
            start = int(speech[0]['start'])
            total_speech = sum(max(0,min(int(item['end']),cut)-int(item['start'])) for item in speech if item['start'] < cut)
            begin = max(0,start-int(self.pad_s*sr))
            finish = min(len(self._audio), cut+int(self.pad_s*sr))
            if forced:
                finish = min(finish, hard_limit)
            if total_speech >= int(self.min_speech_s*sr):
                out.append(Segment(self._audio[begin:finish].copy(), self._start+begin/sr, self._start+finish/sr))
            # Keep all samples after the speech end, including trailing silence.
            if cut <= 0:
                break
            self._discard(cut)
            if not forced or len(self._audio) < hard_limit:
                break
        if flush:
            self.reset()
        return out

    def flush(self):
        return self._emit(True)

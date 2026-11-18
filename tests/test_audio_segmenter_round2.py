"""UTF-8 pactl enumeration and capture discontinuity regression tests."""
import numpy as np
import pytest

from voice_to_speech import audio
from voice_to_speech.segmenter import StreamingSegmenter


SOURCES = '''Source #52
\tState: SUSPENDED
\tName: alsa_output.pci-0000_00_1f.3.analog-stereo.monitor
\tDescription: Monitor of Built-in Audio Аналогове стерео
\tMonitor of Sink: alsa_output.pci-0000_00_1f.3.analog-stereo
Source #53
\tName: alsa_input.pci-0000_00_1f.3.analog-stereo
\tDescription: Built-in Audio Аналогове стерео
\tMonitor of Sink: n/a
'''
SINK_INPUTS = '''Sink Input #71
\tDriver: PipeWire
\tProperties:
\t\tmedia.name = "YouTube — Музика"
\t\tapplication.name = "Google Chrome"
\t\tapplication.process.binary = "chrome"
'''


def test_pactl_text_preserves_utf8_and_resolves_application(monkeypatch):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return (SOURCES if command[-1] == 'sources' else SINK_INPUTS).encode('utf-8')

    monkeypatch.setattr(audio.subprocess, 'check_output', check_output)
    result = audio.list_sources()
    assert result['error'] is None
    assert result['monitors'] == [{
        'name': 'alsa_output.pci-0000_00_1f.3.analog-stereo.monitor',
        'description': 'Monitor of Built-in Audio Аналогове стерео',
    }]
    assert result['microphones'][0]['description'] == 'Built-in Audio Аналогове стерео'
    assert result['apps'] == [{'name': 'Google Chrome', 'media': 'YouTube — Музика', 'index': 71}]
    capture = audio.CaptureSource(source='system', device='app:Google Chrome', on_block=lambda *args: None)
    assert '--monitor-stream=71' in capture._command()
    assert all(command[:2] == ['pactl', 'list'] for command, _ in calls)
    assert all(options['env']['LC_ALL'] == 'C' for _, options in calls)


def test_pactl_replaces_invalid_utf8_and_monitor_uses_field(monkeypatch):
    sample = b'Source #9\n\tName: misleading.monitor\n\tDescription: Device \xff\n\tMonitor of Sink: n/a\n'
    monkeypatch.setattr(audio.subprocess, 'check_output', lambda command, **kwargs: sample if command[-1] == 'sources' else b'')
    result = audio.list_sources()
    assert result['monitors'] == []
    assert result['microphones'][0]['description'] == 'Device \ufffd'


def test_segmenter_flushes_before_capture_gap_and_restarts_timeline():
    def vad(samples):
        active = np.flatnonzero(samples)
        return [{'start': int(active[0]), 'end': int(active[-1]) + 1}] if len(active) else []

    segmenter = StreamingSegmenter(sample_rate=100, vad_fn=vad, pad_s=0, min_speech_s=.4)
    assert segmenter.push(np.ones(60, np.float32), 2.0) == []
    first = segmenter.push(np.ones(60, np.float32), 5.0)
    assert len(first) == 1
    assert first[0].start_s == pytest.approx(2.0)
    assert first[0].end_s == pytest.approx(2.6)
    second = segmenter.flush()
    assert len(second) == 1
    assert second[0].start_s == pytest.approx(5.0)
    assert second[0].end_s == pytest.approx(5.6)


def test_segmenter_exactly_one_second_gap_does_not_restart():
    segmenter = StreamingSegmenter(sample_rate=100,
        vad_fn=lambda samples: [{'start': 0, 'end': len(samples)}], pad_s=0)
    assert segmenter.push(np.ones(50, np.float32), 0) == []
    assert segmenter.push(np.ones(50, np.float32), 1.5) == []
    assert segmenter.flush()[0].start_s == 0

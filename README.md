# Local Transcriptor

Local Transcriptor records system audio and microphone input on Ubuntu with PipeWire, transcribes speech with faster-whisper, and creates post-meeting summaries with a local LLM through llama.cpp. Transcription and summarization run on your computer; audio and text do not leave it. The web interface binds to `127.0.0.1:8765` by default.

## Features

- Live transcription from system audio, microphone, or both, with source switching during a session.
- Capture from the default output, a specific monitor, or an application audio stream.
- File transcription and an optional high-quality pass over a recorded meeting, with TXT and SRT outputs.
- A separate `/summary` page with recorded sessions, one-click summaries, inline results, and text file upload.
- Cancellable, resumable file and summary jobs with progress shown in the web interface.
- A live transcript viewer that follows only while you are at the bottom. Use **↓ Нові рядки (N)** (new lines) to catch up or **Розгорнути** (expand) for a larger view.
- Local model storage and automatic CPU fallback when GPU memory is unavailable.

## Requirements

- Ubuntu 24.04 with PipeWire and a user systemd session.
- Python 3.10 or newer (tested with 3.12), `parec`, `pactl`, and `curl`.
- An NVIDIA GPU is optional. The tested machine uses an RTX 2050 with 4 GB VRAM; CPU processing is supported but slower.
- Disk space for Whisper `large-v3-turbo`, Whisper `large-v3`, the approximately 2.5 GB Gemma 3 4B GGUF, and the local llama.cpp server. Keep additional space for recordings and the Python environment.

## Installation

The initial dependency and model downloads require internet access. After preparation, ordinary server operation is local.

1. Install system packages:

   ```bash
   sudo apt install python3-venv pulseaudio-utils curl
   ```

2. Clone and enter the repository:

   ```bash
   git clone https://github.com/petro-fedyk/local_transcriptor.git
   cd local_transcriptor
   ```

3. Install the Python environment and user systemd integration:

   ```bash
   ./install.sh
   ```

   The installer also runs `vts models`. Run `vts models` again whenever you need to prepare missing models. It downloads the pinned CUDA build of `llama-server` and verifies its SHA-256, then prepares the GGUF and both Whisper models. No Vulkan package is needed for the selected build.

4. Open the interface:

   ```bash
   vts open
   ```

The installer puts `vts` in `~/.local/bin`. Add that directory to your `PATH` if your shell does not already include it. It creates a user socket and service plus an application menu shortcut, without installing a root service.

## Usage

The web interface is in Ukrainian. Press **Почати запис** (start recording), choose system audio, microphone, or both, and press **Зупинити запис** (stop recording) when finished. **Оновити джерела** (refresh sources) updates available devices and application streams; an application may appear only while it is playing audio. The default language is Ukrainian. Select a language explicitly for short phrases, and use **Словник** (glossary) for names and terms.

For an audio file, use the upload control in the web interface or `vts transcribe FILE`. Supported extensions are `.wav`, `.mp3`, `.m4a`, `.mp4`, `.ogg`, `.oga`, `.opus`, `.flac`, `.webm`, `.aac`, `.wma`, `.mkv`, and `.mov`; the upload limit is 2 GiB. The default file model is `large-v3`. For a meeting, **Якісна транскрипція** (high-quality transcription) processes the saved system and microphone WAV files and writes `transcript_hq.txt` and `transcript_hq.srt`. It can also run automatically after recording stops.

Open **Зробити підсумок** (make a summary) in the header to visit `/summary`. Recorded sessions appear automatically. Choose a transcript and start its summary with one click, or upload another UTF-8 `.txt`, `.srt`, or `.md` file (up to 20 MiB). The result is rendered on the page. **Продовжити** (resume) restarts an interrupted job; summary progress and cancellation are available in the interface.

The server normally exits after 15 minutes without recording or jobs. Set the idle timeout to zero in the interface to disable this. The user systemd socket remains available to start the server again when the page is opened.

### CLI reference

| Command | Action |
| --- | --- |
| `vts open` | Start the server and open the page. |
| `vts start` | Start the server without opening a browser. |
| `vts status` | Show server status as JSON. |
| `vts doctor` | Run a read-only meeting safety and NVIDIA suspend checklist. |
| `vts logs` | Follow the server log. |
| `vts stop` | Stop the server when no jobs are active. |
| `vts stop --force` | Cancel active jobs and stop the server. |
| `vts transcribe FILE` | Upload audio and wait for transcription. |
| `vts transcribe FILE --no-wait` | Upload audio without waiting. |
| `vts resume SESSION_ID` | Resume an interrupted audio upload or meeting high-quality pass. |
| `vts models` | Prepare Whisper models, the summary model, and llama-server. |
| `vts run` | Run the server in the current terminal. |
| `vts install` | Re-run installation and update user integration. |
| `vts uninstall` | Remove user integration while keeping data. |
| `vts help` | Show CLI help. |

Summary jobs are resumed from the web interface; `vts resume` applies to audio transcription jobs.

## Meeting summary

The summary page accepts a live or high-quality transcript, SRT subtitles, or plain text. Before summarizing, it removes overlapping microphone/system duplicates; it does not remove filler words. The local Gemma 3 4B model then takes notes in chunks and reduces those notes into a Markdown meeting summary with topics, decisions, tasks, and open questions. Progress is visible, and completed chunk notes allow an interrupted summary to resume.

Whisper and the LLM do not run on the GPU at the same time. A summary waits for active recording or Whisper jobs, unloads Whisper, and may fall back to partial GPU or CPU use based on available VRAM. If recording or transcription needs the GPU, the summary releases its llama-server process and waits before continuing.

For a meeting, outputs are in `recordings/<session_id>/`: `summary.md` is the final result, `summary_source.txt` is the deduplicated input, and `summary.partial.json` holds resumable notes until success. An uploaded text file gets its own `recordings/summary_<timestamp>/` folder with `summary_input.*`. Set `VTS_LLAMA_SERVER` to use a different local llama-server executable or `VTS_LLM_MODEL` to use a different local GGUF file.

## Data and privacy

By default, files are stored under `~/.local/share/voice-to-speech`. Set `VTS_DATA_DIR` to change that directory. It contains `venv/`, `models/`, `llm/`, `settings.json`, and `recordings/`. Meeting folders contain metadata, transcripts, and captured WAV files; audio upload folders also contain the original file. The web interface can play, download, and delete saved sessions.

The server binds to `127.0.0.1` and checks HTTP Host and WebSocket Origin. The summary model server also binds to `127.0.0.1`. The application does not contact external services during normal recording, transcription, or summarization. Installation and `vts models` use the network to install dependencies and fetch model files. `VTS_PORT` changes the server port (default `8765`); re-run `vts install` after changing it or `VTS_DATA_DIR` for the systemd service.

## Meeting safety (screen sharing / suspend)

While recording or processing, the app requests a logind sleep and lid inhibitor. It reserves GPU memory for the desktop and screen sharing via `VTS_VRAM_RESERVE_MB` (default `900`); when there is not enough VRAM, model work can fall back to CPU. The installed user service also has memory and OOM limits. Re-run `vts install` to apply updated unit settings.

Run `vts doctor` for a read-only checklist of the sleep inhibitor, GPU memory, service limits, GNOME auto-suspend setting, and NVIDIA suspend configuration. For long meetings, review GNOME auto-suspend settings; the inhibitor is held only while work is active.

If CUDA fails after laptop suspend, check whether the NVIDIA suspend services are enabled and whether the driver uses `NVreg_PreserveVideoMemoryAllocations=1` with `NVreg_TemporaryFilePath=/var/tmp`. On a system where you want to enable that driver setting, run:

```bash
echo 'options nvidia NVreg_PreserveVideoMemoryAllocations=1 NVreg_TemporaryFilePath=/var/tmp' | sudo tee /etc/modprobe.d/nvidia-power.conf
sudo update-initramfs -u
```

Reboot afterward. To revert this setting, remove `/etc/modprobe.d/nvidia-power.conf`, run `sudo update-initramfs -u`, and reboot. If you already use that file for other NVIDIA options, remove only the added options instead. A temporary recovery after a CUDA suspend failure is `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`, followed by restarting the app after active jobs finish.

## Troubleshooting

- Use `vts logs` for the server traceback, `vts status` for model and device state, and `vts doctor` for meeting safety checks. `watch -n1 nvidia-smi` helps identify VRAM use. A session's `meta.json` records the last job state.
- If models are missing, run `vts models` while connected to the internet. The running server will not fetch them.
- If an audio device changes or an application stream disappears, capture retries while that source remains enabled. Refresh sources when the stream reappears.
- If a job errors or is cancelled, use **Продовжити** (resume) in the interface. Audio transcription resumes from saved output where supported; a meeting high-quality pass starts again. Summary jobs reuse completed chunk notes.
- Use headphones to keep speaker output out of the microphone. Capture a specific application's audio if other desktop sounds should be excluded.
- If the server port is occupied, stop the other process or set `VTS_PORT` and re-run `vts install`.

## Development

Use Python 3.10+ in a virtual environment. From the repository root:

```bash
pip install -r requirements-dev.txt
pytest -q
bash -n bin/vts install.sh
```

The tests use fake audio sources and model engines; they do not need GPU, audio devices, or network access.

## Uninstall

Run `vts uninstall` to remove the user socket/service, application menu entry, and CLI link. The data directory, models, recordings, Python environment, and repository remain until you remove them yourself.

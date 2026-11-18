#!/usr/bin/env bash
set -euo pipefail

ROOT="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
DATA="${VTS_DATA_DIR:-$HOME/.local/share/voice-to-speech}"
PORT="${VTS_PORT:-8765}"
[[ $# -eq 0 ]] || { printf 'Використання: ./install.sh\n' >&2; exit 1; }
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1 && PORT <= 65535)) || { printf 'Некоректний порт VTS_PORT.\n' >&2; exit 1; }
missing=()
for command_name in python3 parec pactl curl systemctl; do command -v "$command_name" >/dev/null || missing+=("$command_name"); done
if ((${#missing[@]})); then
  printf 'Відсутні команди: %s\nВстановіть: sudo apt install python3-venv pulseaudio-utils curl\n' "${missing[*]}" >&2
  exit 1
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Потрібен Python 3.10 або новіший.")'
mkdir -p -- "$DATA"
python3 -m venv --upgrade "$DATA/venv"
PY="$DATA/venv/bin/python"
"$PY" -m pip install -r "$ROOT/requirements.txt"
VTS_DATA_DIR="$DATA" VTS_PORT="$PORT" "$ROOT/bin/vts" models

shopt -s nullglob
libraries=("$DATA"/venv/lib/python3*/site-packages/nvidia/*/lib)
LD_PATH=""
for directory in "${libraries[@]}"; do LD_PATH+="${LD_PATH:+:}$directory"; done
mkdir -p -- "$HOME/.config/systemd/user" "$HOME/.local/share/applications" "$HOME/.local/bin"
# Escape systemd specifiers and quoted values for paths containing spaces.
unit_quote() { local value="${1//%/%%}"; value="${value//\\/\\\\}"; value="${value//\"/\\\"}"; printf '"%s"' "$value"; }
cat >"$HOME/.config/systemd/user/voice-to-speech.socket" <<EOF
[Unit]
Description=Сокет локальної транскрипції

[Socket]
ListenStream=127.0.0.1:${PORT}
Accept=no

[Install]
WantedBy=sockets.target
EOF
cat >"$HOME/.config/systemd/user/voice-to-speech.service" <<EOF
[Unit]
Description=Локальна транскрипція
Requires=voice-to-speech.socket
After=voice-to-speech.socket

[Service]
Type=simple
ExecStart=$(unit_quote "$PY") -m voice_to_speech --port ${PORT}
WorkingDirectory=${ROOT//%/%%}
Environment=$(unit_quote "PYTHONPATH=$ROOT")
Environment=$(unit_quote "LD_LIBRARY_PATH=$LD_PATH")
Environment="HF_HUB_OFFLINE=1"
Environment=$(unit_quote "VTS_DATA_DIR=$DATA")
Environment="PYTHONUNBUFFERED=1"
Nice=10
CPUWeight=40
IOWeight=40
MemoryHigh=7G
MemoryMax=9G
MemorySwapMax=2G
OOMScoreAdjust=500
TasksMax=512
ManagedOOMMemoryPressure=kill
Restart=on-failure
RestartSec=3
TimeoutStopSec=30
EOF
# Desktop Exec quoting differs from shell quoting; escape its reserved characters.
DESKTOP_EXEC="${ROOT//\\/\\\\}/bin/vts"
DESKTOP_EXEC="${DESKTOP_EXEC//\"/\\\"}"
DESKTOP_EXEC="${DESKTOP_EXEC//\`/\\\`}"
DESKTOP_EXEC="${DESKTOP_EXEC//\$/\\\$}"
DESKTOP_EXEC="${DESKTOP_EXEC//%/%%}"
cat >"$HOME/.local/share/applications/voice-to-speech.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Транскрипція
Exec="${DESKTOP_EXEC}" open
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Utility;
EOF
if command -v update-desktop-database >/dev/null; then update-desktop-database "$HOME/.local/share/applications"; fi
ln -sfn -- "$ROOT/bin/vts" "$HOME/.local/bin/vts"
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then printf 'Додайте ~/.local/bin до PATH, щоб команда vts була доступна.\n'; fi
systemctl --user daemon-reload
systemctl --user enable --now voice-to-speech.socket
systemctl --user stop voice-to-speech.service || true
printf '\nГотово. Закладка: http://127.0.0.1:%s/\nЗапуск: vts open\nУ меню застосунків: Транскрипція\n' "$PORT"

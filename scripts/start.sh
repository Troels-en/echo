#!/usr/bin/env bash
# Start the whole stack: whisper-server (model in RAM) + Echo bot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WHISPER_PORT="${WHISPER_PORT:-8910}"

# 1. Start whisper-server if not already listening
if curl -s -o /dev/null "http://127.0.0.1:${WHISPER_PORT}/"; then
    echo "whisper-server already running on :${WHISPER_PORT}"
else
    echo "Starting whisper-server..."
    nohup "$ROOT/scripts/start_whisper_server.sh" > /tmp/whisper-server.log 2>&1 &
    # wait for health
    for i in $(seq 1 30); do
        if curl -s -o /dev/null "http://127.0.0.1:${WHISPER_PORT}/"; then
            echo "whisper-server up."
            break
        fi
        sleep 1
    done
fi

# 2. Start the bot (foreground)
echo "Starting Echo bot..."
exec "$ROOT/.venv/bin/python" -u -m app.bot

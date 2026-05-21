#!/usr/bin/env bash
# Start whisper-server with model kept in RAM (no per-call reload).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${WHISPER_MODEL_PATH:-$ROOT/data/models/ggml-large-v3-turbo-q5_0.bin}"
# fall back to non-quantized if quantized not present
[ -f "$MODEL" ] || MODEL="$ROOT/data/models/ggml-large-v3-turbo.bin"

HOST="${WHISPER_HOST:-127.0.0.1}"
PORT="${WHISPER_PORT:-8910}"
THREADS="${WHISPER_THREADS:-8}"

echo "Starting whisper-server"
echo "  model:   $MODEL"
echo "  listen:  $HOST:$PORT  threads=$THREADS"

exec whisper-server \
    -m "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    -t "$THREADS" \
    -l auto

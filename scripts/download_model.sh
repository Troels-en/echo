#!/usr/bin/env bash
# Download a whisper.cpp ggml model into ./data/models/
set -euo pipefail

MODEL="${1:-large-v3-turbo}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/models"
mkdir -p "$DEST_DIR"
DEST="$DEST_DIR/ggml-${MODEL}.bin"

if [ -f "$DEST" ]; then
    echo "Already present: $DEST"
    exit 0
fi

URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
echo "Downloading $URL → $DEST"
curl -L --progress-bar -o "$DEST" "$URL"
echo "Done. $(ls -lh "$DEST")"

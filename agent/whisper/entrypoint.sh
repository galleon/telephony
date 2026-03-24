#!/usr/bin/env bash
# Download the GGML model on first start (lazy), then launch whisper-server.
# Set WHISPER_MODEL to override the model (default: small.en).
# The /models volume persists the download across container restarts.
set -euo pipefail

MODEL="${WHISPER_MODEL:-small.en}"
MODEL_FILE="/models/ggml-${MODEL}.bin"

if [ ! -f "$MODEL_FILE" ]; then
    echo "[whisper] Downloading model '${MODEL}' to ${MODEL_FILE} ..."
    mkdir -p /models
    # The download script writes ggml-<model>.bin to the current directory.
    cd /models && bash /whisper/models/download-ggml-model.sh "$MODEL"
    echo "[whisper] Download complete."
fi

echo "[whisper] Starting whisper-server: model=${MODEL_FILE}, port=8080"
exec /whisper/build/bin/whisper-server \
    --host 0.0.0.0 \
    --port 8080 \
    -m "$MODEL_FILE" \
    --threads 4 \
    --processors 2

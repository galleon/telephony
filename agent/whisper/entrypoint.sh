#!/usr/bin/env bash
# Download the GGML model on first start (lazy), then launch whisper-server.
# Set WHISPER_MODEL to override the model (default: small.en).
# The /models volume persists the download across container restarts.
set -euo pipefail

MODEL="${WHISPER_MODEL:-small.en}"
MODEL_FILE="/models/ggml-${MODEL}.bin"
HF_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

mkdir -p /models

if [ ! -f "$MODEL_FILE" ]; then
    echo "[whisper] Downloading model '${MODEL}' → ${MODEL_FILE} ..."
    wget -q --show-progress -O "${MODEL_FILE}.tmp" \
        "${HF_BASE}/ggml-${MODEL}.bin" \
    && mv "${MODEL_FILE}.tmp" "${MODEL_FILE}" \
    || { rm -f "${MODEL_FILE}.tmp"; echo "[whisper] Download failed" >&2; exit 1; }
    echo "[whisper] Download complete."
fi

echo "[whisper] Starting whisper-server: model=${MODEL_FILE}, port=8080"
exec /whisper/build/bin/whisper-server \
    --host 0.0.0.0 \
    --port 8080 \
    -m "$MODEL_FILE" \
    --threads 4 \
    --processors 2

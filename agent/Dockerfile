# Voice agent for Sentinel-Voice. Run on DGX Spark.
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy agent code and install deps
COPY pyproject.toml uv.lock* ./
COPY src/ src/
COPY main.py ./
RUN uv sync --no-dev

# NOTE: onnxruntime-gpu has no aarch64 PyPI wheels (x86_64 only), so Piper TTS
# runs on CPU via the standard onnxruntime wheel. The call-filler phrase
# ("One moment please.") mitigates perceived latency while TTS synthesises.
# For GPU Piper on aarch64, build onnxruntime from source against CUDA or switch
# to a PyTorch-based TTS (e.g. Kokoro) which ships aarch64 CUDA wheels.

# Piper downloads voices to cwd if missing; ensure writable
RUN mkdir -p .cache/huggingface

ENV HF_HOME=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

CMD ["uv", "run", "main.py"]

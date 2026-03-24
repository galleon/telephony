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

# pipecat[piper] pulls in the CPU-only `onnxruntime` wheel. Replace it with the
# GPU build so Piper TTS can use CUDAExecutionProvider on DGX Spark GB10.
# CUDA libraries are injected at runtime by NVIDIA Container Toolkit when the
# container starts with `deploy.resources.reservations.devices: [nvidia]`.
RUN uv pip install "onnxruntime-gpu>=1.19"

# Piper downloads voices to cwd if missing; ensure writable
RUN mkdir -p .cache/huggingface

ENV HF_HOME=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

CMD ["uv", "run", "main.py"]

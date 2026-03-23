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

# Piper downloads voices to cwd if missing; ensure writable
RUN mkdir -p .cache/huggingface

ENV HF_HOME=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

CMD ["uv", "run", "main.py"]

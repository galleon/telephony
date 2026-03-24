# Voice agent for Sentinel-Voice. Run on DGX Spark (aarch64 + GB10 Blackwell).
#
# Two-stage build:
#   ct2-builder  — compiles CTranslate2 with CUDA from source (no aarch64 PyPI wheel)
#   runtime      — CUDA runtime image; uv installs pipecat deps, then the PyPI
#                  ctranslate2 (CPU) is swapped for our CUDA-built wheel so
#                  WhisperSTTService uses the GPU.
#
# Expected build time: 10-20 min (nvcc compiling SM 90+100 CUDA kernels).
# Subsequent rebuilds of the agent layer are fast (ct2-builder is cached).
#
# NOTE: onnxruntime-gpu still has no aarch64 PyPI wheel, so Piper TTS remains on
# CPU for now. Step 2 will address TTS (onnxruntime from source or Kokoro).

# ─── Stage 1: Build CTranslate2 with CUDA ─────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04 AS ct2-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.11 (Ubuntu 22.04 ships pip only for Python 3.10)
RUN python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --upgrade pip pybind11 wheel

RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/OpenNMT/CTranslate2.git /ct2

# SM 90 = Hopper, SM 100 = Blackwell (GB10 compute capability 10.0).
RUN cmake -S /ct2 -B /ct2/build \
        -DWITH_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="90;100" \
        -DWITH_MKL=OFF \
        -DOPENMP_RUNTIME=COMP \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/ct2-install \
    && cmake --build /ct2/build -j"$(nproc)" \
    && cmake --install /ct2/build

# Build the CTranslate2 Python wheel against the freshly-installed C++ library.
RUN CTRANSLATE2_ROOT=/ct2-install \
    python3.11 -m pip wheel /ct2 --no-build-isolation -w /ct2-wheel

# ─── Stage 2: Agent runtime ───────────────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CTranslate2 compiled C++ shared libraries
COPY --from=ct2-builder /ct2-install/lib /usr/local/lib/
RUN ldconfig

WORKDIR /app

# uv — uses python3.11 from the system
RUN python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
COPY src/ src/
COPY main.py ./

# Pin uv to Python 3.11 for the agent venv
ENV UV_PYTHON=python3.11

# Sync all deps; pipecat[whisper] pulls CPU ctranslate2 from PyPI — overridden below.
RUN uv sync --no-dev

# Replace the CPU ctranslate2 wheel with our CUDA-compiled one.
# --force-reinstall --no-deps: swap the .so in-place, leave all other packages untouched.
COPY --from=ct2-builder /ct2-wheel /tmp/ct2-wheel/
RUN .venv/bin/pip install --force-reinstall --no-deps /tmp/ct2-wheel/ctranslate2-*.whl \
    && rm -rf /tmp/ct2-wheel

RUN mkdir -p .cache/huggingface .cache/piper

ENV HF_HOME=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface
# Ensure CTranslate2 libctranslate2.so is found at runtime
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
# Explicit GPU visibility (belt-and-suspenders alongside deploy.resources in compose)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["uv", "run", "main.py"]

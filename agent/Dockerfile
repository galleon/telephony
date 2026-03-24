# Voice agent for Sentinel-Voice. Run on DGX Spark (aarch64 + GB10 Blackwell).
#
# Two-stage build:
#   ct2-builder  — installs CUDA toolkit, compiles CTranslate2 with CUDA from
#                  source (no aarch64 PyPI wheel exists), and builds a Python wheel.
#   runtime      — lean CUDA runtime image; uv syncs pipecat deps, then the PyPI
#                  ctranslate2 (CPU) is swapped for our CUDA wheel so
#                  WhisperSTTService runs on GPU.
#
# Expected build time: 10-20 min (nvcc compiling SM 90+100 CUDA kernels).
# Subsequent agent-layer rebuilds are fast — ct2-builder layer is cached.
#
# NOTE: onnxruntime-gpu still has no aarch64 PyPI wheel, so Piper TTS remains
# on CPU. Step 2 will address TTS (onnxruntime from source or alternative).

# ─── Stage 1: Build CTranslate2 with CUDA ─────────────────────────────────────
FROM nvcr.io/nvidia/cuda:13.1.1-base-ubuntu24.04 AS ct2-builder

ENV DEBIAN_FRONTEND=noninteractive

# cuda-toolkit-13-1: nvcc + headers + cuBLAS dev (meta-package from NVIDIA apt repo)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        cuda-toolkit-13-1 \
        python3.12-dev python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure nvcc is on PATH (NVIDIA base image sets this, but be explicit)
ENV PATH=/usr/local/cuda/bin:${PATH}

RUN pip3 install --break-system-packages pybind11 wheel

RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/OpenNMT/CTranslate2.git /ct2

# CUDA_ARCH_LIST is CTranslate2's own cmake/cuda.cmake variable, consumed by
# cmake 3.28's legacy FindCUDA module which requires dot-notation (e.g. "9.0").
# "9.0+PTX" compiles native SM 90 kernels and keeps PTX so CUDA JIT-compiles
# for SM 100 (Blackwell/GB10) at first inference (cached afterward).
# cmake 3.28 does not know SM 100 by name so native Blackwell code requires
# a newer cmake; PTX fallback is equivalent for correctness, ~same throughput.
RUN cmake -S /ct2 -B /ct2/build \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_LIST="9.0+PTX" \
        -DWITH_MKL=OFF \
        -DOPENMP_RUNTIME=COMP \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/ct2-install \
    && cmake --build /ct2/build -j"$(nproc)" \
    && cmake --install /ct2/build

# Build the CTranslate2 Python wheel against the freshly installed C++ library.
RUN CTRANSLATE2_ROOT=/ct2-install \
    python3.12 -m pip wheel /ct2 --no-build-isolation -w /ct2-wheel

# ─── Stage 2: Agent runtime ───────────────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# python3.12-venv: needed by uv to create the agent venv
# libgomp1: OpenMP runtime used by CTranslate2 CPU fallback path
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CTranslate2 compiled C++ shared libraries
COPY --from=ct2-builder /ct2-install/lib /usr/local/lib/
RUN ldconfig

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
COPY src/ src/
COPY main.py ./

ENV UV_PYTHON=python3.12

# Sync all deps; pipecat[whisper] pulls CPU ctranslate2 from PyPI — overridden below.
RUN uv sync --no-dev

# Replace the CPU ctranslate2 wheel with our CUDA-compiled one.
COPY --from=ct2-builder /ct2-wheel /tmp/ct2-wheel/
RUN .venv/bin/pip install --force-reinstall --no-deps /tmp/ct2-wheel/ctranslate2-*.whl \
    && rm -rf /tmp/ct2-wheel

RUN mkdir -p .cache/huggingface .cache/piper

ENV HF_HOME=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["uv", "run", "main.py"]

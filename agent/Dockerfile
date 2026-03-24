# Voice agent for Sentinel-Voice. Run on DGX Spark (aarch64 + GB10 Blackwell).
#
# Three-stage build:
#   ct2-builder  — compiles CTranslate2 with CUDA (no aarch64 PyPI wheel).
#   ort-builder  — compiles onnxruntime-gpu with CUDA (no aarch64 PyPI wheel).
#                  Enables Piper TTS to run on GPU via CUDAExecutionProvider.
#   runtime      — lean CUDA runtime; uv syncs deps; CPU wheels swapped for
#                  CUDA wheels from both builders.
#
# First-build times (cached on subsequent runs):
#   ct2-builder : ~15 min   (nvcc SM 90+PTX kernels)
#   ort-builder : ~45 min   (nvcc SM 90;100 kernels + clang CUDA 13 frontend)
#   runtime     : ~2 min
#
# CUDA 13 notes for onnxruntime:
#   - clang is required as CUDA 13+ compiler frontend (cmake sets CLANG variant)
#   - aarch64 defaults to Jetson arches — CMAKE_CUDA_ARCHITECTURES must be set
#   - libcudnn9-dev-cuda-13 provides headers; cudnn_home=/usr (apt install path)

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
    python3.12 -m pip wheel /ct2/python --no-build-isolation -w /ct2-wheel

# ─── Stage 2: Build onnxruntime-gpu ───────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:13.1.1-base-ubuntu24.04 AS ort-builder

ENV DEBIAN_FRONTEND=noninteractive

# clang: required by onnxruntime cmake for CUDA 13+ frontend (CMAKE_CUDA_COMPILER_FRONTEND_VARIANT=CLANG)
# libcudnn9-dev-cuda-13 + libcudnn9-headers-cuda-13: cuDNN headers at /usr/include/
# cuda-toolkit-13-1: nvcc + CUDA headers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git clang \
        cuda-toolkit-13-1 \
        libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 \
        python3.12-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${PATH}

# Stub needed so the linker resolves libcuda.so.1 at build time (driver only
# exists at runtime via the NVIDIA container runtime host mount).
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

RUN pip3 install --break-system-packages wheel setuptools

RUN git clone --depth 1 --branch v1.24.4 \
        --recurse-submodules --shallow-submodules \
        https://github.com/microsoft/onnxruntime.git /ort

# Build the GPU wheel.
# CMAKE_CUDA_ARCHITECTURES: aarch64 defaults to Jetson (53;62;72;87) — override
# for datacenter GPUs (SM 90 = Hopper, SM 100 = Blackwell/GB10).
# --nvcc_threads 1: limits per-file nvcc parallelism to control peak memory.
# --skip_submodule_sync: repo was freshly cloned, sync is redundant.
RUN cd /ort && python3 tools/ci_build/build.py \
        --build_dir /ort/build \
        --config Release \
        --update \
        --build \
        --skip_submodule_sync \
        --parallel \
        --build_wheel \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr \
        --cuda_version 13.1 \
        --nvcc_threads 1 \
        --cmake_extra_defines \
            CMAKE_CUDA_ARCHITECTURES="90;100" \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1

# ─── Stage 3: Agent runtime ───────────────────────────────────────────────────
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
RUN uv pip install --python .venv/bin/python --force-reinstall --no-deps /tmp/ct2-wheel/ctranslate2-*.whl \
    && rm -rf /tmp/ct2-wheel

# Replace the CPU onnxruntime wheel with our CUDA-compiled one.
# piper-tts pulls onnxruntime (CPU) from PyPI; this swap gives Piper the
# CUDAExecutionProvider so TTS inference runs on the GB10 GPU.
COPY --from=ort-builder /ort/build/Release/dist/onnxruntime_gpu-*.whl /tmp/ort-wheel/
RUN uv pip install --python .venv/bin/python --force-reinstall --no-deps /tmp/ort-wheel/onnxruntime_gpu-*.whl \
    && rm -rf /tmp/ort-wheel

RUN mkdir -p .cache/huggingface .cache/piper

ENV HF_HOME=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["uv", "run", "main.py"]

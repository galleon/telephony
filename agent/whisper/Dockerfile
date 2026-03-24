# Builds whisper.cpp server with CUDA on aarch64 (DGX Spark GB10 / Grace Blackwell).
# The server exposes POST /inference on port 8080.
#
# CUDA cmake flag: current whisper.cpp uses -DGGML_CUDA=ON (ggml unified build);
# -DWHISPER_CUDA=ON is the legacy alias — set both for compatibility.
# SM targets: 90 = Hopper, 100 = Blackwell (GB10 compute capability 10.0).
#
# The model is baked into the image at build time so startup is instant (no
# first-run download stall that would delay the agent coming up).
# Override WHISPER_MODEL build arg to bake a different model.

FROM nvcr.io/nvidia/cuda:13.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /whisper

RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp .

# Build only the server binary to keep build times reasonable.
# -DGGML_CUDA=ON  : current ggml-unified flag (replaces WHISPER_CUDA in recent builds)
# -DWHISPER_CUDA=ON: kept for older CMakeLists compatibility
RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DWHISPER_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="90;100" \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target whisper-server -j"$(nproc)"

# Download the GGML model at image build time → instant startup, no healthcheck delay.
# Pass --build-arg WHISPER_MODEL=tiny.en (or base.en) to use a smaller/faster model.
ARG WHISPER_MODEL=small.en
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN wget -q --show-progress \
        -O "models/ggml-${WHISPER_MODEL}.bin" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${WHISPER_MODEL}.bin"

EXPOSE 8080
# --processors 1: process audio as a single chunk.
# With --processors 2 whisper splits short clips (~1-2 s) at ~0.9 s boundaries;
# each half is too short for accurate transcription → hallucinated words / garbled output.
CMD ["/bin/sh", "-c", \
     "exec /whisper/build/bin/whisper-server \
        --host 0.0.0.0 --port 8080 \
        -m /whisper/models/ggml-${WHISPER_MODEL}.bin \
        --threads 8 --processors 1"]

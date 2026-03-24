# Builds whisper.cpp server with CUDA on aarch64 (DGX Spark GB10 / Grace Blackwell).
# The server exposes /v1/audio/transcriptions (OpenAI-compatible) on port 8080.
#
# SM targets: 90 = Hopper, 100 = Blackwell (GB10). Compiling both lets the binary
# run unchanged on any NVIDIA GPU in that range.
#
# The model is NOT baked into this image — it is downloaded on first start into the
# /models volume so it persists across image rebuilds.

FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /whisper

RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp .

# Build only the server binary to keep build times reasonable.
RUN cmake -B build \
        -DWHISPER_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="90;100" \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target whisper-server -j"$(nproc)"

VOLUME ["/models"]

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]

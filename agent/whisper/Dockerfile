# Builds whisper.cpp server with CUDA on aarch64 (DGX Spark GB10 / Grace Blackwell).
# The server exposes POST /inference on port 8080.
#
# CUDA cmake flags: -DGGML_CUDA=ON (current ggml-unified flag) and the legacy
# -DWHISPER_CUDA=ON alias — both set for compatibility.
# SM targets: 90 = Hopper, 100 = Blackwell (GB10 compute capability 10.0).
#
# The model is baked into the image at build time so startup is instant.
# Override WHISPER_MODEL build arg to use a different model.

FROM nvcr.io/nvidia/cuda:13.1.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# cuda-toolkit-13-1: nvcc + headers needed to compile whisper.cpp CUDA kernels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget ca-certificates \
        cuda-toolkit-13-1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /whisper

RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp .

# -DGGML_CUDA=ON  : current ggml-unified flag (replaces WHISPER_CUDA in recent builds)
# -DWHISPER_CUDA=ON: legacy alias kept for older CMakeLists compatibility
# libcuda.so.1 is not present in the image — it is mounted from the host driver
# at runtime by the NVIDIA container runtime. Create a versioned symlink to the
# CUDA stub library so the linker can resolve transitive driver-API symbols in
# libggml-cuda.so. The stub is removed after the build so it is never loaded at
# runtime. -rpath-link tells ld to search the stubs dir for transitive .so deps.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && cmake -B build \
        -DGGML_CUDA=ON \
        -DWHISPER_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="90;100" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build --target whisper-server -j"$(nproc)" \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1

ARG WHISPER_MODEL=small.en
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN wget -q --show-progress \
        -O "models/ggml-${WHISPER_MODEL}.bin" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${WHISPER_MODEL}.bin"

EXPOSE 8080
# --processors 1: process audio as a single chunk (--processors 2 splits short phone-call
# clips at ~0.9 s boundaries causing hallucinations at the seam).
CMD ["/bin/sh", "-c", \
     "exec /whisper/build/bin/whisper-server \
        --host 0.0.0.0 --port 8080 \
        -m /whisper/models/ggml-${WHISPER_MODEL}.bin \
        --threads 8 --processors 1"]

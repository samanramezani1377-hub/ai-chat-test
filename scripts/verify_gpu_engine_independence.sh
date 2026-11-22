#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine/src/main/cpp"
test -d "$ENGINE/gpu_engine"
test -f "$ENGINE/CMakeLists.txt"
test -f "$ROOT/test-app/build.gradle.kts"

# The GPU engine is intentionally standalone: GGUF parsing is implemented in
# this project and does not import llama.cpp/GGML execution APIs.
if grep -RInE '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]((llama|ggml)[^>"]*)[>"]|FetchContent_Declare\(llama|target_link_libraries\([^)]*(llama|ggml)' "$ENGINE"; then
    echo "::error::Unexpected llama.cpp/GGML execution dependency in the GPU engine."
    exit 1
fi

if grep -RInE 'LLAMA_CPP_TAG|llama_cpp_SOURCE_DIR|GGML_WEBGPU|GGML_OPENCL|GGML_VULKAN' "$ENGINE/CMakeLists.txt"; then
    echo "::error::Legacy llama.cpp/GGML build configuration remains in the engine."
    exit 1
fi

grep -q 'DAWN_ENABLE_VULKAN OFF' "$ENGINE/cmake/FindDawn.cmake"
grep -q 'AI_CHAT_GPU_BACKEND_VULKAN=0' "$ENGINE/CMakeLists.txt"
grep -q 'abiFilters += "arm64-v8a"' "$ROOT/engine/build.gradle.kts"
echo "Standalone Android GPU engine boundary verified."


if grep -RInE 'NativeLlamaCpp|CpuLlamaCpp|cpu_fallback_available=true|llama\.cpp runtime' "$ENGINE"; then
    echo "::error::Legacy CPU runtime routing remains in the standalone GPU engine."
    exit 1
fi

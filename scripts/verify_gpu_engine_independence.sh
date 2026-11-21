#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
engine_dir="$root/core/runtime-android/src/main/cpp/gpu_engine"
cmake_file="$root/core/runtime-android/src/main/cpp/CMakeLists.txt"
test -d "$engine_dir"
test -f "$cmake_file"
if grep -RInE '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"].*(ggml|llama)|\b(ggml_[[:alnum:]_]*|llama_[[:alnum:]_]*)\b' "$engine_dir"; then
    echo "::error::Standalone GPU engine core contains a llama.cpp/GGML API dependency."
    exit 1
fi
extract_target() {
    local target="$1"
    sed -n "/add_library(${target} /,/^)/p" "$cmake_file"
}
core_block="$(extract_target ai_chat_gpu_engine)"
dawn_block="$(extract_target ai_chat_gpu_backend_dawn)"
gles_block="$(extract_target ai_chat_gpu_backend_gles)"
android_block="$(extract_target ai_chat_gpu_backends_android)"
test -n "$core_block" && test -n "$dawn_block" && test -n "$gles_block" && test -n "$android_block"
if printf '%s\n' "$core_block" "$dawn_block" "$gles_block" "$android_block" | grep -Eiq '(ggml|llama)'; then
    echo "::error::A standalone GPU engine target contains a llama.cpp/GGML dependency."
    exit 1
fi
link_blocks="$(awk '
    /^[[:space:]]*target_link_libraries\(ai_chat_gpu_/ { in_block = 1 }
    in_block { print }
    in_block && /\)/ { in_block = 0 }
' "$cmake_file")"
if printf '%s\n' "$link_blocks" | grep -Eiq '(ggml|llama)'; then
    echo "::error::A GPU engine target links llama.cpp/GGML."
    exit 1
fi

if ! grep -q 'set(GGML_VULKAN OFF' "$cmake_file"; then
    echo "::error::Vulkan must remain explicitly disabled."
    exit 1
fi
echo "GPU engine core, Dawn/GLES backends and Android composition are isolated from llama.cpp/GGML execution APIs."
echo "Legacy llama.cpp remains linked only by ai_chat_runtime for the preserved CPU path."

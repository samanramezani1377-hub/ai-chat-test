#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/core/runtime-android/src/main/cpp/gpu_engine"
TEST="$ROOT/core/runtime-android/src/test/cpp/gpu_engine_contract_test.cpp"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I "$ENGINE" \
  "$ENGINE/gpu_engine.cpp" \
  "$ENGINE/execution_engine.cpp" \
  "$ENGINE/gguf_reader.cpp" \
  "$ENGINE/model_registry.cpp" \
  "$ENGINE/operators.cpp" \
  "$ENGINE/weight_streamer.cpp" \
  "$ENGINE/sampler.cpp" \
  "$TEST" \
  -o "$TMP/gpu_engine_contract_test"

# The reusable engine core and Dawn backend must not import llama.cpp/GGML execution APIs.
if grep -nE '#include[[:space:]]*[<"]((llama|ggml)[^>"]*)[>"]' \
    "$ENGINE/gpu_engine.h" "$ENGINE/gpu_engine.cpp" \
    "$ENGINE/execution_engine.h" "$ENGINE/execution_engine.cpp" \
    "$ENGINE/gguf_reader.h" "$ENGINE/gguf_reader.cpp" \
    "$ENGINE/model_registry.h" "$ENGINE/model_registry.cpp" \
    "$ENGINE/operators.h" "$ENGINE/operators.cpp" \
    "$ENGINE/weight_streamer.h" "$ENGINE/weight_streamer.cpp" \
    "$ENGINE/sampler.h" "$ENGINE/sampler.cpp" "$ENGINE/dawn_backend.cpp"; then
  echo "GPU engine backend boundary violation: llama.cpp/GGML include found" >&2
  exit 1
fi

"$TMP/gpu_engine_contract_test"

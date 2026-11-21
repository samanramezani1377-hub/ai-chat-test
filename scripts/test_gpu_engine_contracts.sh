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
  "$TEST" \
  -o "$TMP/gpu_engine_contract_test"

"$TMP/gpu_engine_contract_test"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine/src/main/cpp/gpu_engine"
CLI_SOURCE="$ROOT/engine/src/test/cpp/tokenizer_parity_cli.cpp"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -I "$ENGINE" \
  "$ENGINE/gguf_reader.cpp" "$ENGINE/tokenizer.cpp" "$ENGINE/tokenizer_adapters.cpp" \
  "$CLI_SOURCE" -o "$TMP/tokenizer_parity_cli"
python3 "$ROOT/scripts/test_qwen3_tokenizer_parity.py" "$TMP/tokenizer_parity_cli"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/engine/src/main/cpp/gpu_engine"
TEST="$ROOT/engine/src/test/cpp/gpu_engine_contract_test.cpp"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# "sample" is a reserved GLSL ES interpolation qualifier on mobile drivers.
# Guard the embedded SSMConv1D shaders against reintroducing that identifier.
if grep -Eq '\\bsample\\b' "$ENGINE/hybrid_state_operators.cpp"; then
  echo "SSMConv1D shader contains the reserved GLSL identifier 'sample'." >&2
  exit 1
fi

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I "$ENGINE" \
  "$ENGINE/gpu_engine.cpp" \
  "$ENGINE/execution_engine.cpp" \
  "$ENGINE/gguf_reader.cpp" \
  "$ENGINE/model_registry.cpp" \
  "$ENGINE/model_structure_analyzer.cpp" \
  "$ENGINE/hybrid_gated_delta_net_recipe.cpp" \
  "$ENGINE/split_operator.cpp" \
  "$ENGINE/gdn_l2_norm_operator.cpp" \
  "$ENGINE/hybrid_state_operators.cpp" \
  "$ENGINE/model_graph.cpp" \
  "$ENGINE/model_lowerings.cpp" \
  "$ENGINE/operators.cpp" \
  "$ENGINE/kernel_validation.cpp" \
  "$ENGINE/weight_streamer.cpp" \
  "$ENGINE/sampler.cpp" \
  "$ENGINE/quantization.cpp" \
  "$ENGINE/kv_cache.cpp" \
  "$ENGINE/memory_planner.cpp" \
  "$ENGINE/tokenizer.cpp" \
  "$ENGINE/tokenizer_adapters.cpp" \
  "$TEST" \
  -o "$TMP/gpu_engine_contract_test"

"$TMP/gpu_engine_contract_test"

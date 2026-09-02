#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_KT="$ROOT_DIR/libs/llama.kt"
CPP_DIR="$LLAMA_KT/llama-kt/src/main/cpp"
LLAMA_CPP_DIR="$LLAMA_KT/third_party/llama.cpp"
LLAMA_KT_SHA=5a20956b1fe2f40d07252f079c31bcaa08250e6b
LLAMA_CPP_SHA=c5fc7e34885ba31217e330809437afa993d27745
LLAMA_RN_SHA=7afb5c8934aea7fa41f4522662f03b3461916927

prepare_submodule() {
  git -C "$ROOT_DIR" submodule sync -- libs/llama.kt
  git -C "$ROOT_DIR" submodule update --init --depth=1 libs/llama.kt || {
    rm -rf "$LLAMA_KT" "$ROOT_DIR/.git/modules/libs/llama.kt"
    git clone --depth=1 https://github.com/hokanosekai/llama.kt "$LLAMA_KT"
    git -C "$LLAMA_KT" fetch --depth=1 origin "$LLAMA_KT_SHA"
    git -C "$LLAMA_KT" checkout --detach "$LLAMA_KT_SHA"
  }
  test "$(git -C "$LLAMA_KT" rev-parse HEAD)" = "$LLAMA_KT_SHA"
  git -C "$LLAMA_KT" submodule sync --recursive
  git -C "$LLAMA_KT" submodule update --init --recursive
}

pin_llama_cpp() {
  test -d "$LLAMA_CPP_DIR"
  git -C "$LLAMA_CPP_DIR" fetch --depth=1 origin "$LLAMA_CPP_SHA"
  git -C "$LLAMA_CPP_DIR" checkout --detach "$LLAMA_CPP_SHA"
  git -C "$LLAMA_KT" update-index --add --cacheinfo 160000,"$LLAMA_CPP_SHA",third_party/llama.cpp
  test "$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)" = "$LLAMA_CPP_SHA"
  test "$(git -C "$LLAMA_KT" ls-files -s third_party/llama.cpp | awk '{print $2}')" = "$LLAMA_CPP_SHA"
}

prepare_llama_rn() {
  local ref=/tmp/llama.rn-ref
  if [ -d "$ref/.git" ]; then
    if ! git -C "$ref" rev-parse --verify "${LLAMA_RN_SHA}^{commit}" >/dev/null 2>&1; then
      git -C "$ref" fetch --depth=1 origin "$LLAMA_RN_SHA"
    fi
  else
    rm -rf "$ref"
    git clone --filter=blob:none --no-checkout https://github.com/mybigday/llama.rn "$ref"
    git -C "$ref" fetch --depth=1 origin "$LLAMA_RN_SHA"
  fi
  git -C "$ref" checkout --detach "$LLAMA_RN_SHA"
  test "$(git -C "$ref" rev-parse HEAD)" = "$LLAMA_RN_SHA"
  test -s "$ref/cpp/rn-llama-version.h"
}

apply_native_patches() {
  local p="$ROOT_DIR/ci/patches"
  cp "$p/0002-abort-callback-mid-graph-cancel.patch" "$LLAMA_KT/patches/0002-abort-callback-mid-graph-cancel.patch"
  rm -f "$LLAMA_KT/patches/0001-vulkan-uma-descriptor-ceildiv.patch"
  patch --dry-run -p1 --forward < "$p/0003-cache-vulkan-shader-generation.patch"
  patch -p1 --forward < "$p/0003-cache-vulkan-shader-generation.patch"
  bash "$LLAMA_KT/scripts/bootstrap.sh" 2>&1 | tee "$ROOT_DIR/bootstrap-output.log"
}

validate_and_bridge() {
  test "$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)" = "$LLAMA_CPP_SHA"
  for f in llama.h llama.cpp ggml.h; do test -s "$CPP_DIR/$f"; done
  cp /tmp/llama.rn-ref/cpp/rn-llama-version.h "$CPP_DIR/rn-llama-version.h"
  test -s "$CPP_DIR/rn-llama-version.h"

  if [ -f "$CPP_DIR/common/json-partial.cpp" ]; then
    sed -i 's/LM_LM_GGML_ASSERT/LM_GGML_ASSERT/g' "$CPP_DIR/common/json-partial.cpp"
  fi
  printf '%s\n' '#include "../../../../../third_party/llama.cpp/common/json.h"' > "$CPP_DIR/common/json.h"
  printf '%s\n' '#include "../../../../../third_party/llama.cpp/common/json.cpp"' > "$CPP_DIR/common/json.cpp"

  cp "$LLAMA_CPP_DIR/ggml/src/ggml-feats.h" "$CPP_DIR/ggml-feats.h"
  sed -i 's/GGML_/LM_GGML_/g; s/ggml_/lm_ggml_/g' "$CPP_DIR/ggml-feats.h"
  grep -Fq 'lm_ggml_feats_arch64_runtime_t' "$CPP_DIR/ggml-feats.h"
  grep -Fq 'lm_ggml_feats_get_arch64_runtime' "$CPP_DIR/ggml-feats.h"

  for header in llama-kv-cache-dsa-iswa.h llama-kv-cache-msa.h llama-kv-cache-dsv4.h; do
    cp "$LLAMA_CPP_DIR/src/$header" "$CPP_DIR/$header"
    sed -i 's/GGML_/LM_GGML_/g; s/ggml_/lm_ggml_/g' "$CPP_DIR/$header"
  done
  mkdir -p "$CPP_DIR/tools/mtmd"
  cp "$LLAMA_CPP_DIR/tools/mtmd/mtmd-internal.h" "$CPP_DIR/tools/mtmd/mtmd-internal.h"
  for f in llama.h llama.cpp ggml.h rn-llama-version.h ggml-feats.h llama-kv-cache-dsa-iswa.h llama-kv-cache-msa.h llama-kv-cache-dsv4.h tools/mtmd/mtmd-internal.h; do
    test -s "$CPP_DIR/$f"
  done
}

apply_barbet_patches() {
  local d="$CPP_DIR" p=/tmp/llama.rn-ref/scripts/patches
  cd "$d"
  apply_if_missing() {
    local marker="$1" patch_file="$2" target="$3"
    if grep -Fq "$marker" "$target"; then return 0; fi
    patch -p0 --forward < "$patch_file"
    grep -Fq "$marker" "$target"
  }
  apply_if_missing 'LLM_ARCH_BARBET,' "$p/barbet-llama-arch.h.patch" llama-arch.h
  apply_if_missing '{ LLM_ARCH_BARBET,           "barbet"           },' "$p/barbet-llama-arch.cpp.patch" llama-arch.cpp
  apply_if_missing 'case LLM_ARCH_BARBET:' "$p/barbet-llama-model.cpp.patch" llama-model.cpp
  apply_if_missing 'case LLM_TENSOR_CONVNEXT_NORM:' "$p/barbet-llama-model-loader.cpp.patch" llama-model-loader.cpp
}

prepare_submodule
pin_llama_cpp
prepare_llama_rn
if [ "${LLAMA_BOOTSTRAP_CACHE_HIT:-false}" = "true" ]; then
  echo 'Native bootstrap cache hit; skipping bootstrap.sh.'
else
  apply_native_patches
fi
validate_and_bridge
apply_barbet_patches

echo 'Pinned native dependency preparation completed.'

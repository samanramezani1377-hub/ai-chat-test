#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_KT="$ROOT_DIR/libs/llama.kt"
CPP_DIR="$LLAMA_KT/llama-kt/src/main/cpp"
LLAMA_CPP_DIR="$LLAMA_KT/third_party/llama.cpp"
LLAMA_KT_SHA=5a20956b1fe2f40d07252f079c31bcaa08250e6b
LLAMA_CPP_SHA=c5fc7e34885ba31217e330809437afa993d27745
LLAMA_RN_SHA=7afb5c8934aea7fa41f4522662f03b3461916927
VULKAN_VERSION=1.4.321.1
VULKAN_SHA256=f22a3625bd4d7a32e7a0d926ace16d5278c149e938dac63cecc00537626cbf73
NDK_VERSION=27.2.12479018
NATIVE_DEPS_REBUILT=false
LLAMA_BOOTSTRAP_REBUILT=false

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

prepare_toolchains() {
  local ndk_path="${ANDROID_HOME}/ndk/${NDK_VERSION}"
  if [ ! -x "$ndk_path/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" ]; then
    echo 'NDK cache is missing or invalid; reinstalling NDK.'
    rm -rf "$ndk_path"
    sdkmanager "ndk;${NDK_VERSION}"
    NATIVE_DEPS_REBUILT=true
  fi
  test -x "$ndk_path/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"

  local archive="/tmp/vulkansdk-linux-x86_64-${VULKAN_VERSION}.tar.xz"
  if [ -s "$archive" ] && ! echo "${VULKAN_SHA256}  ${archive}" | sha256sum -c - >/dev/null 2>&1; then
    echo 'Cached Vulkan archive is invalid; rebuilding it.'
    rm -f "$archive"
    NATIVE_DEPS_REBUILT=true
  fi
  if [ ! -s "$archive" ]; then
    curl -fL --retry 5 --retry-all-errors --connect-timeout 20 --max-time 1800 -o "$archive.part" "https://sdk.lunarg.com/sdk/download/${VULKAN_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_VERSION}.tar.xz"
    echo "${VULKAN_SHA256}  ${archive}.part" | sha256sum -c -
    mv -f "$archive.part" "$archive"
    NATIVE_DEPS_REBUILT=true
  fi
  echo "${VULKAN_SHA256}  ${archive}" | sha256sum -c -

  local vulkan_sdk="$HOME/vulkan-sdk/${VULKAN_VERSION}/x86_64"
  if [ ! -x "$vulkan_sdk/bin/glslc" ]; then
    echo 'Vulkan SDK cache is missing or invalid; extracting a fresh copy.'
    rm -rf "$HOME/vulkan-sdk"
    mkdir -p "$HOME/vulkan-sdk"
    tar -xf "$archive" -C "$HOME/vulkan-sdk"
    NATIVE_DEPS_REBUILT=true
  fi
  test -x "$vulkan_sdk/bin/glslc"

  export VULKAN_SDK="$vulkan_sdk"
  export PATH="$VULKAN_SDK/bin:$PATH"
  export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
  printf 'VULKAN_SDK=%s\nPATH=%s\nLD_LIBRARY_PATH=%s\n' "$VULKAN_SDK" "$PATH" "$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
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
      git -C "$ref" fetch --depth=1 origin "$LLAMA_RN_SHA" || {
        rm -rf "$ref"
        git clone --filter=blob:none --no-checkout https://github.com/mybigday/llama.rn "$ref"
        git -C "$ref" fetch --depth=1 origin "$LLAMA_RN_SHA"
        NATIVE_DEPS_REBUILT=true
      }
    fi
  else
    rm -rf "$ref"
    git clone --filter=blob:none --no-checkout https://github.com/mybigday/llama.rn "$ref"
    git -C "$ref" fetch --depth=1 origin "$LLAMA_RN_SHA"
    NATIVE_DEPS_REBUILT=true
  fi
  git -C "$ref" checkout --detach "$LLAMA_RN_SHA" || {
    rm -rf "$ref"
    git clone --filter=blob:none --no-checkout https://github.com/mybigday/llama.rn "$ref"
    git -C "$ref" fetch --depth=1 origin "$LLAMA_RN_SHA"
    git -C "$ref" checkout --detach "$LLAMA_RN_SHA"
    NATIVE_DEPS_REBUILT=true
  }
  test "$(git -C "$ref" rev-parse HEAD)" = "$LLAMA_RN_SHA"
  test -s "$ref/cpp/rn-llama-version.h"
}

apply_native_patches() {
  local script="$LLAMA_KT/scripts/bootstrap.sh"
  local marker='Vulkan shader cache hit; reusing generated SPIR-V and per-shader .cpp files.'

  if ! grep -Fq "$marker" "$script"; then
    echo 'Applying Vulkan shader cache transformation...'
    python3 - "$script" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
start = text.find('echo "  Generating ggml-vulkan-shaders.hpp header..."')
end_marker = '# 3. Bundle Vulkan C++ binding headers (vulkan.hpp — not in NDK sysroot)'
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit('ERROR: Vulkan shader generation block was not found; refusing to modify unknown upstream code.')

new_block = r'''VK_SHADER_CACHE_SENTINEL="$VK_OUT_DIR/.complete"
if [ -s "$VK_OUT_DIR/ggml-vulkan-shaders.hpp" ] && [ -f "$VK_SHADER_CACHE_SENTINEL" ]; then
  echo "  Vulkan shader cache hit; reusing generated SPIR-V and per-shader .cpp files."
else
  rm -rf "$VK_SPV_DIR" "$VK_OUT_DIR"
  mkdir -p "$VK_SPV_DIR" "$VK_OUT_DIR"
  echo "  Generating ggml-vulkan-shaders.hpp header..."
  "$VK_GEN" \
    --output-dir "$VK_SPV_DIR" \
    --target-hpp "$VK_OUT_DIR/ggml-vulkan-shaders.hpp"
  echo "  Compiling GLSL shaders to SPIR-V and generating per-shader .cpp files..."
  for comp in "$VK_SHADERS_SRC"/*.comp; do
    base=$(basename "$comp")
    "$VK_GEN" \
      --glslc "$(which glslc)" \
      --source "$comp" \
      --output-dir "$VK_SPV_DIR" \
      --target-hpp "$VK_OUT_DIR/ggml-vulkan-shaders.hpp" \
      --target-cpp "$VK_OUT_DIR/${base}.cpp"
  done
  test -s "$VK_OUT_DIR/ggml-vulkan-shaders.hpp"
  test "$(find "$VK_OUT_DIR" -maxdepth 1 -type f -name '*.cpp' | wc -l)" -gt 0
  : > "$VK_SHADER_CACHE_SENTINEL"
fi

cp "$VK_OUT_DIR/ggml-vulkan-shaders.hpp" "$CPP_DIR/ggml-vulkan/"

for f in "$VK_OUT_DIR"/*.cpp; do
  sed 's|#include "ggml-vulkan-shaders.hpp"|#include "../ggml-vulkan-shaders.hpp"|g' "$f" > "$CPP_DIR/ggml-vulkan/shaders/$(basename "$f")"
done

'''
text = text[:start] + new_block + text[end:]
path.write_text(text)
PY
  fi

  grep -Fq "$marker" "$script"

  # llama.kt 5a20956 still carries its historical 0001 UMA patch, but the
  # pinned llama.cpp revision already contains the upstream fix merged from
  # PR #25245. Remove only that obsolete patch after verifying the exact
  # overflow-safe macro is present; otherwise keep the fail-fast behavior.
  local vulkan_cpp="$LLAMA_CPP_DIR/ggml/src/ggml-vulkan/ggml-vulkan.cpp"
  local uma_patch="$LLAMA_KT/patches/0001-vulkan-uma-descriptor-ceildiv.patch"
  if grep -Fq '#define CEIL_DIV(M, N) (((M) / (N)) + (((M) % (N)) != 0))' "$vulkan_cpp"; then
    if [ -f "$uma_patch" ]; then
      echo 'Upstream llama.cpp already contains the Vulkan UMA CEIL_DIV fix; skipping obsolete llama.kt 0001 patch.'
      rm -f "$uma_patch"
    fi
  else
    echo 'ERROR: pinned llama.cpp does not contain the upstream Vulkan UMA CEIL_DIV fix; refusing to discard 0001 patch.'
    exit 1
  fi

  bash "$script" 2>&1 | tee "$ROOT_DIR/bootstrap-output.log"
  LLAMA_BOOTSTRAP_REBUILT=true
}

bootstrap_cache_is_valid() {
  test -s "$CPP_DIR/llama.h" && test -s "$CPP_DIR/llama.cpp" && test -s "$CPP_DIR/ggml.h" && test -s "$CPP_DIR/rn-llama-version.h" && test -s "$CPP_DIR/ggml-feats.h" && grep -Fq 'LLM_ARCH_BARBET' "$CPP_DIR/llama-arch.h" && grep -Fq 'LLM_ARCH_BARBET' "$CPP_DIR/llama-arch.cpp"
}

validate_and_bridge() {
  test "$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)" = "$LLAMA_CPP_SHA"
  for f in llama.h llama.cpp ggml.h; do test -s "$CPP_DIR/$f"; done
  cp /tmp/llama.rn-ref/cpp/rn-llama-version.h "$CPP_DIR/rn-llama-version.h"
  test -s "$CPP_DIR/rn-llama-version.h"
  if [ -f "$CPP_DIR/common/json-partial.cpp" ]; then sed -i 's/LM_LM_GGML_ASSERT/LM_GGML_ASSERT/g' "$CPP_DIR/common/json-partial.cpp"; fi
  printf '%s\n' '#include "../../../../../third_party/llama.cpp/common/json.h"' > "$CPP_DIR/common/json.h"
  printf '%s\n' '#include "../../../../../third_party/llama.cpp/common/json.cpp"' > "$CPP_DIR/common/json.cpp"
  cp "$LLAMA_CPP_DIR/ggml/src/ggml-feats.h" "$CPP_DIR/ggml-feats.h"
  sed -i 's/GGML_/LM_GGML_/g; s/ggml_/lm_ggml_/g' "$CPP_DIR/ggml-feats.h"
  grep -Fq 'lm_ggml_feats_arch64_runtime_t' "$CPP_DIR/ggml-feats.h"
  grep -Fq 'lm_ggml_feats_get_arch64_runtime' "$CPP_DIR/ggml-feats.h"
  for header in llama-kv-cache-dsa-iswa.h llama-kv-cache-msa.h llama-kv-cache-dsv4.h; do cp "$LLAMA_CPP_DIR/src/$header" "$CPP_DIR/$header"; sed -i 's/GGML_/LM_GGML_/g; s/ggml_/lm_ggml_/g' "$CPP_DIR/$header"; done
  mkdir -p "$CPP_DIR/tools/mtmd"
  cp "$LLAMA_CPP_DIR/tools/mtmd/mtmd-internal.h" "$CPP_DIR/tools/mtmd/mtmd-internal.h"
  for f in llama.h llama.cpp ggml.h rn-llama-version.h ggml-feats.h llama-kv-cache-dsa-iswa.h llama-kv-cache-msa.h llama-kv-cache-dsv4.h tools/mtmd/mtmd-internal.h; do test -s "$CPP_DIR/$f"; done
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
prepare_toolchains
pin_llama_cpp
prepare_llama_rn

if [ "${LLAMA_BOOTSTRAP_CACHE_HIT:-false}" = "true" ] && bootstrap_cache_is_valid; then
  echo 'Valid native bootstrap cache restored; skipping bootstrap.sh.'
else
  if [ "${LLAMA_BOOTSTRAP_CACHE_HIT:-false}" = "true" ]; then
    echo 'Native bootstrap cache is invalid; discarding it and rebuilding.'
    rm -rf "$CPP_DIR"
  fi
  apply_native_patches
fi

validate_and_bridge
apply_barbet_patches

printf 'native_deps_rebuilt=%s\n' "$NATIVE_DEPS_REBUILT" >> "$GITHUB_OUTPUT"
printf 'llama_bootstrap_rebuilt=%s\n' "$LLAMA_BOOTSTRAP_REBUILT" >> "$GITHUB_OUTPUT"
echo 'Pinned native dependency preparation completed.'

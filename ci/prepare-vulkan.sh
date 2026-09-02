#!/usr/bin/env bash
set -euo pipefail

VULKAN_VERSION=1.4.321.1
VULKAN_SHA256=f22a3625bd4d7a32e7a0d926ace16d5278c149e938dac63cecc00537626cbf73
ARCHIVE="/tmp/vulkansdk-linux-x86_64-${VULKAN_VERSION}.tar.xz"
SDK_ROOT="$HOME/vulkan-sdk"
SDK="$SDK_ROOT/${VULKAN_VERSION}/x86_64"

if [[ -s "$ARCHIVE" ]] && ! echo "${VULKAN_SHA256}  ${ARCHIVE}" | sha256sum -c - >/dev/null 2>&1; then
  rm -f "$ARCHIVE"
fi

if [[ ! -s "$ARCHIVE" ]]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 20 --max-time 1800 \
    -o "${ARCHIVE}.part" \
    "https://sdk.lunarg.com/sdk/download/${VULKAN_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_VERSION}.tar.xz"
  echo "${VULKAN_SHA256}  ${ARCHIVE}.part" | sha256sum -c -
  mv -f "${ARCHIVE}.part" "$ARCHIVE"
fi

echo "${VULKAN_SHA256}  ${ARCHIVE}" | sha256sum -c -

if [[ ! -x "$SDK/bin/glslc" ]]; then
  rm -rf "$SDK_ROOT"
  mkdir -p "$SDK_ROOT"
  tar -xf "$ARCHIVE" -C "$SDK_ROOT"
fi

test -x "$SDK/bin/glslc"
export VULKAN_SDK="$SDK"
export PATH="$VULKAN_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
printf 'VULKAN_SDK=%s\nPATH=%s\nLD_LIBRARY_PATH=%s\n' "$VULKAN_SDK" "$PATH" "$LD_LIBRARY_PATH" >> "$GITHUB_ENV"

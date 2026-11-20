#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE="$ROOT/core/runtime-android/src/main/cpp/CMakeLists.txt"
FIND_DAWN="$ROOT/core/runtime-android/src/main/cpp/cmake/FindDawn.cmake"
GRADLE="$ROOT/core/runtime-android/build.gradle.kts"
NATIVE="$ROOT/core/runtime-android/src/main/cpp/native_runtime_android_safe.cpp"

fail() {
  echo "::error::$1" >&2
  exit 1
}

# AndroidX WebGPU ships the native Dawn shared library in its AAR. Do not
# assume that the AAR exports a Prefab webgpuConfig.cmake package.
grep -q 'AI_CHAT_WEBGPU_LIBRARY' "$CMAKE" || fail "CMake must receive the extracted AndroidX Dawn library path"
grep -q 'webgpu_c_bundled' "$FIND_DAWN" || fail "FindDawn must expose the imported webgpu_c_bundled target"
! grep -q 'find_package(webgpu CONFIG REQUIRED)' "$FIND_DAWN" || fail "FindDawn must not require a Prefab package that the AndroidX AAR does not export"
grep -q 'extractWebGpuNativeLibrary' "$GRADLE" || fail "Gradle must extract the AndroidX WebGPU AAR native library before CMake configuration"
grep -q 'libwebgpu_c_bundled.so' "$GRADLE" || fail "Gradle extraction must select libwebgpu_c_bundled.so"
grep -q 'AI_CHAT_WEBGPU_LIBRARY' "$GRADLE" || fail "Gradle must pass the extracted Dawn library path to CMake"

# The selected inference backend is WebGPU. OpenCL may remain a separately
# diagnosed legacy backend, but must never be reported as the active WebGPU path.
! grep -q 'OPENCL_GPU_PREFERRED_MIXED_CPU_OPS' "$NATIVE" || fail "Generation diagnostics still mislabel WebGPU inference as OpenCL"
grep -q 'WEBGPU' "$NATIVE" || fail "Generation diagnostics must identify the WebGPU execution path"

# Keep the Android GPU API policy explicit and independent of a phone/model.
grep -q 'GGML_VULKAN OFF' "$CMAKE" || fail "Vulkan must remain disabled"
grep -q 'FeatureLevel::Compatibility' "$CMAKE" || fail "WebGPU compatibility mode must remain enabled"
grep -q 'BackendType::OpenGLES' "$CMAKE" || fail "WebGPU must use the OpenGL ES backend"

echo "WebGPU Android native contract: PASS"

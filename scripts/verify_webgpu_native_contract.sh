#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE="$ROOT/core/runtime-android/src/main/cpp/CMakeLists.txt"
FIND_DAWN="$ROOT/core/runtime-android/src/main/cpp/cmake/FindDawn.cmake"
GRADLE="$ROOT/core/runtime-android/build.gradle.kts"
NATIVE="$ROOT/core/runtime-android/src/main/cpp/native_runtime_android_safe.cpp"
ADAPTER="$ROOT/core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/LlamaCppAndroidRuntimeAdapter.kt"
DIAGNOSTICS="$ROOT/core/runtime/src/main/kotlin/com/woogit/aicore/runtime/RuntimeDiagnostics.kt"

fail() {
  echo "::error::$1" >&2
  exit 1
}

# AndroidX WebGPU publishes jniLibs/assets in its AAR, not a Prefab package.
grep -q 'AI_CHAT_WEBGPU_LIBRARY' "$CMAKE" || fail "CMake must validate the extracted AndroidX Dawn library path"
grep -q 'AI_CHAT_WEBGPU_LIBRARY' "$FIND_DAWN" || fail "FindDawn must import the extracted AndroidX Dawn library"
grep -q 'webgpu_c_bundled' "$FIND_DAWN" || fail "FindDawn must expose the imported webgpu_c_bundled target"
grep -q 'AI_CHAT_DAWN_METADATA_FILE' "$FIND_DAWN" || fail "FindDawn must use metadata from the same AndroidX AAR"
grep -q 'dawn_source_commit_sha' "$FIND_DAWN" || fail "Dawn headers must be generated from the AAR's recorded source SHA"
grep -q -- '--targets headers,cpp_headers' "$FIND_DAWN" || fail "FindDawn must generate both Dawn C and C++ headers"
! grep -q 'find_package(webgpu CONFIG REQUIRED)' "$FIND_DAWN" || fail "FindDawn must not require a Prefab package the AndroidX AAR does not export"
grep -q 'extractWebGpuNativeLibrary' "$GRADLE" || fail "Gradle must extract the AndroidX WebGPU AAR before CMake configuration"
grep -q 'libwebgpu_c_bundled.so' "$GRADLE" || fail "Gradle extraction must select libwebgpu_c_bundled.so"
grep -q 'dawn_build_metadata.json' "$GRADLE" || fail "Gradle must extract the Dawn build metadata"
grep -q 'AI_CHAT_WEBGPU_LIBRARY' "$GRADLE" || fail "Gradle must pass the extracted Dawn library path to CMake"
grep -q 'AI_CHAT_DAWN_METADATA_FILE' "$GRADLE" || fail "Gradle must pass the Dawn metadata path to CMake"

# The selected inference backend is WebGPU. OpenCL remains a separate legacy
# diagnostic path and must not be reported as the active WebGPU execution path.
! grep -q 'OPENCL_GPU_PREFERRED_MIXED_CPU_OPS' "$NATIVE" || fail "Generation diagnostics still mislabel WebGPU inference as OpenCL"
grep -q 'WEBGPU_GPU_PREFERRED_MIXED_CPU_OPS' "$NATIVE" || fail "Generation diagnostics must identify WebGPU execution"
! grep -q 'OPENCL_GPU_PREFERRED_CPU_FALLBACK_ALLOWED' "$ADAPTER" || fail "Kotlin runtime still labels WebGPU selection as OpenCL"
grep -q 'shouldReadOpenClProfile' "$DIAGNOSTICS" || fail "Legacy OpenCL profile must be gated by the selected backend"

# Keep the Android GPU API policy explicit and independent of a phone/model.
grep -q 'GGML_VULKAN OFF' "$CMAKE" || fail "Vulkan must remain disabled"
grep -q 'FeatureLevel::Compatibility' "$CMAKE" || fail "WebGPU compatibility mode must remain enabled"
grep -q 'BackendType::OpenGLES' "$CMAKE" || fail "WebGPU must use the OpenGL ES backend"

echo "WebGPU Android native contract: PASS"

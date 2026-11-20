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

# AndroidX's bundled .so is JNI-oriented, not the public C ABI used by ggml.
# Keep only its exact Dawn source SHA metadata and build Dawn natively for GGML.
grep -q 'AI_CHAT_DAWN_METADATA_FILE' "$CMAKE" || fail "CMake must require the pinned Dawn metadata"
grep -q 'DAWN_FETCH_DEPENDENCIES ON' "$FIND_DAWN" || fail "FindDawn must fetch Dawn's pinned dependencies"
grep -q 'GIT_SUBMODULES ""' "$FIND_DAWN" || fail "Dawn FetchContent must not clone optional Git submodules"
grep -q 'GIT_SUBMODULES_RECURSE FALSE' "$FIND_DAWN" || fail "Dawn FetchContent must not recursively clone optional submodules"
grep -q 'dawn_source_commit_sha' "$FIND_DAWN" || fail "Dawn must be built from the AAR's recorded source SHA"
grep -q 'DAWN_ENABLE_OPENGLES ON' "$FIND_DAWN" || fail "Native Dawn must enable OpenGL ES"
grep -q 'TINT_BUILD_NULL_WRITER ON' "$FIND_DAWN" || fail "Tint null writer is required by Dawn OpenGL ES compute pipeline reflection"
grep -q 'DAWN_ENABLE_NULL OFF' "$FIND_DAWN" || fail "Dawn's optional Null GPU backend must remain disabled"
grep -q 'DAWN_ENABLE_VULKAN OFF' "$FIND_DAWN" || fail "Native Dawn must keep Vulkan disabled"
grep -q 'dawn::webgpu_dawn' "$FIND_DAWN" || fail "FindDawn must expose the real native Dawn C ABI target"
grep -q 'extractWebGpuDawnMetadata' "$GRADLE" || fail "Gradle must extract Dawn build metadata before CMake configuration"
! grep -q 'implementation("androidx.webgpu:webgpu:' "$GRADLE" || fail "JNI-only AndroidX Dawn library must not be packaged into the app"
! grep -q 'AI_CHAT_WEBGPU_LIBRARY' "$GRADLE" || fail "Gradle must not pass the JNI-only AAR library to native linking"
grep -q 'AI_CHAT_DAWN_METADATA_FILE' "$GRADLE" || fail "Gradle must pass the extracted Dawn metadata path to CMake"

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
grep -q 'cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED' "$ROOT/core/runtime-android/src/main/cpp/native_runtime.cpp" || fail "Android must not select the unstable fused FLASH_ATTN_EXT path"
# Non-ShaderF16 adapters use the conservative F32/I32 path. llama.cpp probes
# plain F32 ADD when selecting a layer buffer, so this op must remain admitted.
grep -q 'op->op == GGML_OP_ADD || op->op == GGML_OP_ADD_ID' "$CMAKE" || fail "No-ShaderF16 WebGPU buffer selection must admit the F32 ADD probe"
grep -q 'AI_CHAT_WEBGPU_FP16_FAST_KERNELS_DISABLED reason=shader_f16_unavailable' "$CMAKE" || fail "No-ShaderF16 fast-kernel fallback must be explicit in diagnostics"
grep -q 'quantized_f32_matvec_fallback=enabled' "$CMAKE" || fail "No-ShaderF16 quantized mat-vec fallback must be reported as enabled"
grep -q 'AI_CHAT_WEBGPU_NO_SHADER_F16_QUANT_MATVEC_V1' "$CMAKE" || fail "F32-decoded quantized mat-vec shader patch is missing"
grep -q 'AI_CHAT_NO_SHADER_F16_COMMON_V1' "$CMAKE" || fail "FP16 storage decoding must have a baseline-F32 shader path"
grep -q 'AI_CHAT_NO_SHADER_F16_Q8_V1' "$CMAKE" || fail "Q8 activation quantizer must support packed scale storage without ShaderF16"
grep -q 'AI_CHAT_NO_SHADER_F16_Q8_READ_V1' "$CMAKE" || fail "Quantized mat-vec must decode packed Q8 scales with F32"
grep -q 'AI_CHAT_NO_SHADER_F16_QUANT_MATVEC_OP_V1' "$CMAKE" || fail "No-ShaderF16 quantized MUL_MAT dispatch is missing"
grep -q 'f32_quantized_matvec' "$CMAKE" || fail "Quantized no-ShaderF16 dispatch must be limited to the validated mat-vec shape"
grep -q 'pack2x16float(vec2<f32>(d, s))' "$CMAKE" || fail "Q8_1 scale/sum must be packed without WGSL f16 arithmetic"
grep -q 'unpack2x16float(src1q\[block\].ds)' "$CMAKE" || fail "Q8_1 packed scales must be decoded with baseline F32 operations"
grep -q 'AI_CHAT_NO_SHADER_F16_MMV_ACC_V1' "$CMAKE" || fail "Quantized mat-vec accumulator still depends on the WGSL f16 type"

# The imported AAR library must never be used as a native WebGPU C ABI provider.
! grep -q 'webgpu_c_bundled' "$FIND_DAWN" || fail "FindDawn must not link the AndroidX JNI-only library"
echo "WebGPU Android native Dawn/OpenGL ES contract: PASS"

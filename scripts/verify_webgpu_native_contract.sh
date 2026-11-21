#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE="$ROOT/core/runtime-android/src/main/cpp/CMakeLists.txt"
WEBGPU_PATCH="$ROOT/core/runtime-android/src/main/cpp/cmake/patch_webgpu_backend.cmake"
OPENCL_PATCH="$ROOT/core/runtime-android/src/main/cpp/cmake/patch_android_opencl.cmake"
PATCH_HELPERS="$ROOT/core/runtime-android/src/main/cpp/cmake/AIChatPatchHelpers.cmake"
CMAKE_FILES=( "${CMAKE_FILES[@]}" "$WEBGPU_PATCH" )
FIND_DAWN="$ROOT/core/runtime-android/src/main/cpp/cmake/FindDawn.cmake"
GRADLE="$ROOT/core/runtime-android/build.gradle.kts"
NATIVE="$ROOT/core/runtime-android/src/main/cpp/native_runtime_android_safe.cpp"
ADAPTER="$ROOT/core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/LlamaCppAndroidRuntimeAdapter.kt"
DIAGNOSTICS="$ROOT/core/runtime/src/main/kotlin/com/woogit/aicore/runtime/RuntimeDiagnostics.kt"

fail() {
  echo "::error::$1" >&2
  exit 1
}

# The native CMake entry point delegates to independently validated patch modules.
grep -q 'patch_webgpu_backend.cmake' "${CMAKE}" || fail "CMake must include the isolated WebGPU patch module"
grep -q 'patch_android_opencl.cmake' "${CMAKE}" || fail "CMake must include the isolated Android OpenCL patch module"
grep -q 'AIChatPatchHelpers.cmake' "${CMAKE}" || fail "CMake must load the shared strict patch helper"
grep -q 'function(ai_chat_replace_once' "${PATCH_HELPERS}" || fail "Strict patch helper definition is missing"
grep -q 'AI_CHAT_WEBGPU_QUANT_MATMUL_OP_VERIFY' "${WEBGPU_PATCH}" || fail "Quantized MatMul patch verification must use its defined variable"
! grep -q 'AI_CHAT_WEBGPU_QUANT_MATVEC_OP_VERIFY' "${WEBGPU_PATCH}" || fail "Undefined MatVec verification variable must not be used"

# AndroidX's bundled .so is JNI-oriented, not the public C ABI used by ggml.
# Keep only its exact Dawn source SHA metadata and build Dawn natively for GGML.
grep -q 'AI_CHAT_DAWN_METADATA_FILE' "${CMAKE_FILES[@]}" || fail "CMake must require the pinned Dawn metadata"
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
grep -q 'GGML_VULKAN OFF' "${CMAKE_FILES[@]}" || fail "Vulkan must remain disabled"
grep -q 'FeatureLevel::Compatibility' "${CMAKE_FILES[@]}" || fail "WebGPU compatibility mode must remain enabled"
grep -q 'BackendType::OpenGLES' "${CMAKE_FILES[@]}" || fail "WebGPU must use the OpenGL ES backend"
grep -q 'cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED' "$ROOT/core/runtime-android/src/main/cpp/native_runtime.cpp" || fail "Android must not select the unstable fused FLASH_ATTN_EXT path"
# Non-ShaderF16 adapters use the conservative F32/I32 path. llama.cpp probes
# plain F32 ADD when selecting a layer buffer, so this op must remain admitted.
grep -q 'op->op == GGML_OP_ADD || op->op == GGML_OP_ADD_ID' "${CMAKE_FILES[@]}" || fail "No-ShaderF16 WebGPU buffer selection must admit the F32 ADD probe"
grep -q 'AI_CHAT_WEBGPU_FP16_FAST_KERNELS_DISABLED reason=shader_f16_unavailable' "${CMAKE_FILES[@]}" || fail "No-ShaderF16 fast-kernel fallback must be explicit in diagnostics"
grep -q 'quantized_f32_matvec_fallback=enabled' "${CMAKE_FILES[@]}" || fail "No-ShaderF16 quantized mat-vec fallback must be reported as enabled"
grep -q 'AI_CHAT_WEBGPU_NO_SHADER_F16_QUANT_MATVEC_V1' "${CMAKE_FILES[@]}" || fail "F32-decoded quantized mat-vec shader patch is missing"
grep -q 'AI_CHAT_NO_SHADER_F16_COMMON_V1' "${CMAKE_FILES[@]}" || fail "FP16 storage decoding must have a baseline-F32 shader path"
grep -q 'AI_CHAT_NO_SHADER_F16_Q8_V1' "${CMAKE_FILES[@]}" || fail "Q8 activation quantizer must support packed scale storage without ShaderF16"
grep -q 'AI_CHAT_NO_SHADER_F16_Q8_READ_V1' "${CMAKE_FILES[@]}" || fail "Quantized mat-vec must decode packed Q8 scales with F32"
grep -q 'AI_CHAT_NO_SHADER_F16_QUANT_MATMUL_OP_V4' "${CMAKE_FILES[@]}" || fail "No-ShaderF16 quantized MUL_MAT dispatch is missing"
grep -q 'f32_quantized_matmul' "${CMAKE_FILES[@]}" || fail "Quantized no-ShaderF16 F32-input GEMM must use the portable register-tile shader"
grep -q 'pack2x16float(vec2<f32>(d, s))' "${CMAKE_FILES[@]}" || fail "Q8_1 scale/sum must be packed without WGSL f16 arithmetic"
grep -q 'unpack2x16float(src1q\[block\].ds)' "${CMAKE_FILES[@]}" || fail "Q8_1 packed scales must be decoded with baseline F32 operations"
grep -q 'AI_CHAT_NO_SHADER_F16_MMV_ACC_V1' "${CMAKE_FILES[@]}" || fail "Quantized mat-vec accumulator still depends on the WGSL f16 type"
grep -q 'AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_PIPELINE_V1' "${CMAKE_FILES[@]}" || fail "Portable no-ShaderF16 register-tile GEMM pipeline is missing"
grep -q 'AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_TILE_V1' "${CMAKE_FILES[@]}" || fail "Portable no-ShaderF16 register-tile WGSL variant is missing"
grep -q 'AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_DECLS_V1' "${CMAKE_FILES[@]}" || fail "Portable no-ShaderF16 quantized dequant declarations are missing"
grep -q 'key.use_subgroup_matrix = context.supports_subgroup_matrix && context.supports_shader_f16' "${CMAKE_FILES[@]}" || fail "No-ShaderF16 GEMM must avoid the F16 subgroup-matrix shader"
grep -q 'NO_SHADER_F16' "${CMAKE_FILES[@]}" || fail "No-ShaderF16 F32 arithmetic define is missing"
grep -q 'apply F32 alias before all shared quantized declarations' "${CMAKE_FILES[@]}" || fail "F32 alias must be applied before common quantized WGSL declarations"
grep -q 'f32_quantized_matmul' "${CMAKE_FILES[@]}" || fail "No-ShaderF16 quantized prefill GEMM must not be restricted to decode batch sizes"
grep -q 'AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_DECLS_V1' "${CMAKE_FILES[@]}" || fail "Packed IQ scales must be decoded through the no-F16 branch"
grep -q 'prompt_prefill_gemm_fallback=portable_f32_register_tile' "${CMAKE_FILES[@]}" || fail "Portable no-ShaderF16 prompt-prefill GEMM must be reported"
grep -q 'quantized_prefill_gemm_fallback=portable_f32_register_tile' "${CMAKE_FILES[@]}" || fail "Portable no-ShaderF16 quantized prefill GEMM must be reported"
! grep -q 'f32_quantized_matvec' "${CMAKE_FILES[@]}" || fail "No-ShaderF16 quantized matmul must not remain limited to decode-only MatVec"

# The imported AAR library must never be used as a native WebGPU C ABI provider.
! grep -q 'webgpu_c_bundled' "$FIND_DAWN" || fail "FindDawn must not link the AndroidX JNI-only library"
echo "WebGPU Android native Dawn/OpenGL ES contract: PASS"

# Android GPU Backend Investigation (new-kor)

## Scope and constraints

- Keep all GPU work inside the existing llama.cpp/GGML native runtime.
- Do not add UI screens, controls, or settings.
- Preserve the current CPU backend and make every fallback observable with a concrete reason.
- Preserve arm64-v8a and the pinned llama.cpp revision until a deliberate compatibility decision is made.
- Do not change model quantization, context length, or generation limits as a performance workaround.

## Historical baseline (before WebGPU/Dawn integration)

The following notes describe the branch before the later AndroidX WebGPU/Dawn work recorded below. They are retained for investigation history and are not a description of the current branch state.

The active Android runtime is `core/runtime-android`. Its CMake build pins llama.cpp to
`9871df5911a03a813518bcd623ee2eba91bf32aa`.

The current build:
- Enables and statically registers GGML OpenCL.
- Explicitly forces `GGML_VULKAN=OFF`.
- Does not declare, fetch, build, or link Dawn.
- Uses an Android-specific direct OpenCL dispatch shim.
- Has an existing GPU-preferred / CPU-retry model-load path and emits OpenCL-specific fallback events.

The separate top-level `runtime-android/src/main/cpp/native_runtime_android_safe.cpp`
is not the active Gradle CMake source path: the active Android library points at
`core/runtime-android/src/main/cpp/CMakeLists.txt`. This duplicate is a cleanup candidate,
but it must not be deleted until all references and historical build workflows are checked.

## Historical WebGPU / Dawn feasibility assessment (superseded below)

The pinned llama.cpp revision contains a GGML WebGPU backend whose native build path
requires `find_package(Dawn REQUIRED)`, C++20, Python shader embedding, and a Dawn
package. The project currently has none of those dependencies or Android packaging
rules. Current upstream llama.cpp documentation describes the WebGPU backend as
Dawn-based and still in progress; it does not provide a verified Android NDK build
recipe for this app's integration.

Therefore WebGPU/Dawn is not currently a safe drop-in backend for this Android runtime.
Do not turn it on by merely setting `GGML_WEBGPU=ON`.

## Vulkan feasibility (rejected by the current project constraint)

Vulkan is intentionally excluded from the selected implementation. The following feasibility notes are historical only; they do not authorize enabling Vulkan.

The pinned llama.cpp revision contains a GGML Vulkan backend and exports
`ggml_backend_vk_reg()` in `ggml-vulkan.h`. Its CMake build requires Vulkan headers,
a Vulkan loader/library, SPIR-V headers, and the host `glslc` shader compiler. The
repository already contains `ci/prepare-vulkan.sh`, but the active Android CMake file
currently disables Vulkan and does not link its backend.

Vulkan is the more practical first GPU route for Mali and other Android GPU vendors,
provided the Android NDK Vulkan loader is linked correctly and shaders are compiled
with a host compiler. Device discovery alone is not proof of working inference:
the exact target must pass model-load, tensor-residency, Q6_K operation, decode, and
stability tests on hardware.

## Initial architecture proposal (superseded by the implementation below)

A native-only backend coordinator should:
1. Discover compiled backends and their actual GGML GPU devices.
2. Prefer a verified Vulkan GPU device; retain OpenCL only when its device and kernel
   capabilities pass the runtime checks.
3. Set model device preferences explicitly rather than relying on a requested GPU-layer
   percentage as proof of offload.
4. Validate actual GPU tensor residency after model load.
5. Retry the same model and same inference parameters on CPU if GPU initialization,
   model placement, context creation, or execution fails.
6. Emit a structured fallback reason such as
   `vulkan_backend_not_compiled`, `vulkan_no_device`,
   `vulkan_model_load_failed`, `vulkan_no_gpu_tensor_residency`,
   `vulkan_context_init_failed`, or `vulkan_decode_failed`.
7. Keep the existing CPU path as a first-class guaranteed recovery path.
8. Expose the selected backend and fallback reason through the existing diagnostics
   channel only; do not add UI controls.

A custom coordinator around GGML's backend API is preferable to implementing a new
GPU driver or shader compiler. The math kernels should initially come from the pinned
GGML Vulkan backend; custom kernels should be added only where measured profiling shows
a gap and correctness tests exist.

## Validation gates before claiming completion

- Android arm64-v8a CMake configure and native build succeeds in CI.
- APK contains the expected native runtime and Android Vulkan system dependency is
  resolved by the platform loader (not a host Vulkan library).
- CPU-only execution remains functional when Vulkan is absent or disabled.
- Runtime reports Vulkan device identity and backend registration separately.
- Model-load tests prove actual GPU tensor residency.
- Q6_K prefill and decode produce correct outputs on a physical Mali-G57 device.
- Forced failure tests prove CPU fallback and preserve a machine-readable reason.
- Voice and text paths remain unchanged at the UI/API boundary.
- No performance claim is made until repeated on-device benchmarks compare the same
  model, prompt, context, and output length.

## Cleanup policy

No runtime, test, workflow, or native source is deleted solely because its filename
looks old or experimental. A deletion requires proof that it is unreferenced by Gradle,
CMake, CI, scripts, runtime loading, and current product paths. The top-level duplicate
runtime source is a candidate for this audit, not yet approved for deletion.


## AndroidX Jetpack WebGPU integration (new-kor)

The Android runtime now declares the official `androidx.webgpu:webgpu:1.0.0-alpha06`
artifact and consumes its Dawn native implementation through Android Gradle Prefab.
The existing GGML WebGPU backend is compiled and statically linked into
`ai_chat_runtime`; no WebView, browser surface, Activity, Compose element, or UI
control is created for inference.

The pinned GGML WebGPU adapter request is forced to
`FeatureLevel::Compatibility` and `BackendType::OpenGLES`. Vulkan-specific Dawn
toggles are removed from this request. The native runtime registers WebGPU, selects
its GPU/IGPU device for model offload, checks actual model tensor residency, and
retains the existing same-model CPU retry if adapter discovery, model loading,
residency validation, or context creation fails.

OpenCL remains compiled as legacy diagnostic infrastructure but is not selected as
an inference fallback in this WebGPU path. CPU is the fallback. Vulkan remains
disabled in the Android CMake configuration.

### Implementation qualification

This change wires the source-level backend integration; it is not a verified
device-support or performance claim. AndroidX WebGPU is still Alpha and the
OpenGL ES Compatibility path depends on each device's graphics driver. The build
and physical-device inference have not been run as part of this change. Q6_K
operator coverage, adapter creation, tensor residency, and end-to-end generation
must be validated before calling this production-ready.


## Backend integration hardening

The native residency gate and memory report must identify buffers owned by the
GGML WebGPU backend specifically. A generic GPU/IGPU classification is insufficient
because legacy OpenCL may also be registered in the process; another GPU backend must
not accidentally satisfy WebGPU residency validation or be reported as the selected
WebGPU device.

The CMake source patch independently enforces both adapter settings:
- `FeatureLevel::Compatibility`
- `BackendType::OpenGLES`

This avoids a partial-patch case where one setting is already present in the pinned
source but the other is absent. Vulkan-specific Dawn toggles remain removed and
`GGML_VULKAN=OFF` remains mandatory.

The AndroidX WebGPU AAR is consumed as a native Dawn dependency through Prefab. The
GGML WebGPU backend remains the actual inference backend; Kotlin/Java is not placed
in the token-generation or tensor-compute hot path. No WebView or UI surface is added.

## Universal native GPU adapter foundation (2026-11-20)

This section supersedes earlier statements in this investigation that described
Vulkan as the preferred next implementation. Vulkan remains explicitly disabled
and is not part of the selected architecture.

### Runtime-only implementation

The active native target now compiles
`core/runtime-android/src/main/cpp/universal_gpu_adapter.cpp` and links the
Android NDK `EGL` and `GLESv3` system libraries. During native backend
initialization, `probe_universal_gpu_adapter()` creates a temporary 1x1 EGL
pbuffer and an OpenGL ES 3 context, reads the real driver identity and compute
limits, writes a structured `UNIVERSAL_GPU_ADAPTER` record to the existing
native trace, and releases its EGL/GL objects.

If a valid GLES context is created and reports a version below 3.1, the native coordinator rejects the Dawn/OpenGL ES inference attempt with the explicit reason `universal_gles_compute_not_supported`. If EGL probing itself fails, the result remains unknown and Dawn is still allowed to perform its own adapter discovery; a failed independent probe is not treated as proof that no adapter exists.

The probe enumerates OpenGL ES 3.x extensions with `glGetStringi(GL_EXTENSIONS, index)` and `GL_NUM_EXTENSIONS`; it does not use the legacy `glGetString(GL_EXTENSIONS)` query, which is not the valid extension enumeration path for ES 3.x contexts. The extension count and FP16 hint therefore come from the actual indexed driver extension list.

The probe records:
- EGL display/context creation and EGL version/vendor;
- GL vendor, renderer, version and shading-language version;
- OpenGL ES compute availability (3.1+);
- compute workgroup count/size, invocation and shared-memory limits;
- shader-storage-block size;
- a separately labelled FP16 shader-extension hint.

The probe runs inside the native runtime. It adds no Activity, View, Compose
screen, setting, control, or UI-to-runtime API change. The existing UI continues
to call the same runtime adapter and JNI entry points.

### Execution boundary and current limitation

This is the vendor-neutral discovery/capability layer, not a new independent
GGML compute backend. Actual inference remains on the existing GGML WebGPU
backend through Dawn's OpenGL ES implementation. Dawn/GGML remains responsible
for adapter/device creation, kernel compilation, operation support and tensor
residency. OpenCL remains a separately registered legacy/diagnostic backend;
it is not silently selected as a substitute for WebGPU. CPU fallback remains
the correctness recovery path.

The GLES FP16 extension hint is not equivalent to WebGPU's
`ShaderF16` feature and must never be used to bypass GGML's current feature
requirements. In particular, this change does not claim that a GPU without
WebGPU ShaderF16 can run the existing GGML shader set. A generic FP32 WGSL
kernel family and per-operation capability dispatch are still required before
that device class can be admitted for inference.

### Validation status

CI now checks that the native adapter source is compiled and linked against
EGL/GLESv3, that the runtime calls the probe, and that the expected capability
fields are present. The Android native build and unit tests must pass before
merging. Physical-device validation is still required to verify EGL pbuffer
creation on different Android vendor drivers and to compare the adapter report
with Dawn's actual WebGPU adapter/device report.

No UI source file is part of this change. No Vulkan dependency is introduced.


## Conservative WebGPU path for devices without ShaderF16 (2026-11-20)

The pinned upstream GGML WebGPU backend originally rejected the entire adapter
unless it exposed `ShaderF16`. That is unnecessarily strict for operations whose
inputs and outputs are strictly F32/I32, but simply removing the feature check
would be unsafe because several existing shader paths use F16 arithmetic,
including quantized matrix kernels.

The Android CMake integration now applies a narrowly scoped, version-anchored
patch to the pinned GGML source:

- `ShaderF16` is requested only when the selected WebGPU adapter reports it.
- The WebGPU backend can register without `ShaderF16` as long as native
  `ImplicitDeviceSynchronization` is still available.
- When `ShaderF16` is absent, the GGML `supports_op` callback rejects every
  operation whose input/output tensors are F16 or quantized. GGML's scheduler
  can then keep those operations on CPU while placing eligible F32/I32
  operations on WebGPU.
- The backend does not advertise unsupported quantized kernels and does not
  pretend a GLES FP16 extension is equivalent to WebGPU `ShaderF16`.
- CMake fails closed if the pinned source no longer matches the expected patch
  anchors, rather than silently applying a partial patch.

This is a conservative partial-offload path, not a claim that Q4/Q5/Q6_K
matrix multiplication is accelerated without ShaderF16. It does not yet add a
new native GLES backend or custom FP32 WGSL kernels. Existing upstream GGML
WebGPU F32 kernels are reused. CI checks both that the patch is applied to the
fetched pinned source and that the Android APK build completes.


### No-ShaderF16 operation whitelist

A further safety constraint is applied because some upstream WGSL modules
declare `enable f16` unconditionally even when the tensor metadata is F32.
Therefore the no-ShaderF16 path does not treat all F32 operations as safe.
It admits only the upstream paths audited as F32/I32-safe in this pinned revision:
`ADD_ID`, `CONCAT`, `SET`, `RMS_NORM`, `NORM`, `L2_NORM`, `SCALE`, `UNARY`
(F32 variant), `CLAMP`, `FILL`, `LOG`, `SQR`, `SQRT`, `SIN`, `COS`, `SUM`,
`SUM_ROWS`, `CUMSUM`, `ARGMAX`, `ARGSORT`, `TOP_K`, `SOFT_MAX` (F32 mask/no
F16 mask), `DIAG`, `TRI`, `SOLVE_TRI`, `PAD`, `UPSCALE` (F32 variant),
`SSM_CONV`, `SSM_SCAN`, and `GATED_DELTA_NET`. This expanded list is encoded
as `AI_CHAT_GENERIC_FP32_OP_WHITELIST_V2`; CI verifies every admitted op name
in the version-anchored CMake patch. Each operation still has to pass the
upstream `supports_op` checks for its exact tensor types, shape, and device
limits. Operations such as `MUL_MAT`, attention, RoPE, and quantized paths
remain on CPU without `ShaderF16` until their exact shader variants have
explicit no-F16 validation. The no-ShaderF16 safety guard scans all `GGML_MAX_SRC` input slots, not just the first three, so recurrent operations with additional source tensors cannot accidentally pass an F16 or quantized input through the F32/I32 whitelist.

### Quantized kernels when ShaderF16 is available

The pinned upstream WebGPU backend includes quantized `MUL_MAT` support for
`Q1_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`,
`Q5_K`, `Q6_K`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ1_S`,
`IQ1_M`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, and `NVFP4`. This is the pinned backend's
advertised type coverage, not a promise that every tensor shape or every GPU
supports every kernel; `supports_op`, shader selection, subgroup features,
and device limits still gate each graph node. Without WebGPU `ShaderF16`,
quantized operations remain on CPU. CPU fallback preserves model compatibility
rather than pretending an unsupported quantized shader is accelerated.

### Dawn capability diagnostics

The native backend emits `AI_CHAT_WEBGPU_CAPABILITIES` from the actual Dawn
adapter, separately reporting `ShaderF16`, subgroups, packed dot product,
subgroup-matrix support, and relevant storage/compute limits. The earlier
`UNIVERSAL_GPU_ADAPTER` record reports EGL/GLES capabilities only. These two
reports are intentionally distinct: an OpenGL ES FP16 extension hint is not
proof that Dawn exposes WebGPU `ShaderF16`.



### F32 softmax without ShaderF16

The pinned `soft_max.wgsl` shader declared `enable f16` unconditionally even
though its normal logits/output path is F32 and F16 is only needed when the
optional attention mask tensor itself is F16. The Android CMake patch now
guards that declaration with `#ifdef MASK_F16`. The no-ShaderF16 whitelist
can therefore safely include F32 softmax; F16-mask operations are still
rejected by the generic tensor-type guard. CI verifies the patched shader
source before packaging.


### EGL thread-state preservation

The capability probe now snapshots the calling thread's current EGL display,
draw/read surfaces, and context before binding its temporary pbuffer context.
After collecting capabilities it restores the previous application-owned
context before destroying the temporary context. This prevents a probe invoked
from a thread with an existing graphics context from silently unbinding the
application's rendering context. CI checks the capture and restore code paths;
physical-device testing is still needed for vendor EGL implementations.


The EGL restoration also preserves the calling thread's previously bound EGL
client API before attempting to restore its context. This matters when the
application context was created under a different EGL client API from the
temporary OpenGL ES probe.


## Quantized decode MatVec without ShaderF16 (2026-11-20)

This section supersedes the earlier statements above that all quantized operations
must remain on CPU whenever Dawn does not expose \`ShaderF16\`. The previous
restriction was safe but unnecessarily broad for the decode-time matrix-vector
path.

The pinned GGML WebGPU MatVec implementation already reads quantized weight
blocks as packed integer words and decodes their scales before accumulating in
F32. The remaining ShaderF16 dependency was primarily in the representation of
FP16 scale values and the temporary Q8_1 activation scale/sum pair, not a need
for FP16 arithmetic. The Android CMake integration now patches the pinned
shader sources to:

- Decode stored 16-bit scale values with baseline WGSL
  \`unpack2x16float()\`, returning F32 values without requiring the optional
  ShaderF16 feature.
- Keep the Q8_1 scale/sum storage exactly four bytes by packing the two
  16-bit values into one \`u32\` with \`pack2x16float()\`; the MatVec shader
  unpacks them to F32.
- Remove the unconditional \`enable f16;\` declaration from the quantized
  MatVec and Q8 activation-quantization shaders.
- Replace the remaining explicit \`vec2<f16>\` decode in the MatVec accumulator
  with baseline F32 unpacking.
- Admit no-ShaderF16 quantized \`MUL_MAT\` only when the destination is F32,
  the activation input is F32, the weight input is quantized, and the output
  batch dimension is at most four. This matches the upstream MatVec dispatch
  boundary and prevents the F16 tiled-GEMM path from being selected.

This allows the existing upstream quantized MatVec implementations—including
the Q6_K path—to be considered for WebGPU decode without converting the model
weights to F32 or expanding their stored representation. Q8 activation
quantization remains available when the adapter separately exposes the
required packed 4x8 integer dot-product feature; otherwise the existing direct
quantized MatVec path is used.

### Explicit limits

- This is a quantized **MatVec/decode** path for output batch sizes up to four.
- Larger prompt-prefill matrix multiplications still use the conservative CPU
  route on adapters without ShaderF16. The upstream tiled GEMM shaders still
  use F16 workgroup storage and are not admitted by this patch.
- F16 tensor operations and other quantized operations not covered by this
  exact MatVec shape remain subject to the existing no-ShaderF16 rejection.
- The change is source-level and build-contract validated; it is not yet proof
  of successful shader compilation, correct numerical output, or faster
  inference on a physical Android GPU.

CMake now fails if the pinned shader source anchors change, if the patched
MatVec/Q8 shaders still declare \`enable f16;\`, or if the Q8_1 scale reads
retain the old F16 struct-field access. The native contract script also checks
that the no-ShaderF16 quantized MatVec admission and packed scale conversions
remain present.

The next acceptance gate is an actual Android run that reports meaningful
WebGPU tensor residency and confirms Q6_K decode output against the CPU
reference. Prompt prefill and cross-vendor performance must be measured
separately; neither is inferred from model weight residency.

## No-ShaderF16 quantized prompt-prefill GEMM (2026-11-20)

The earlier decode-only limitation is superseded by a portable variant of the
pinned upstream register-tile GEMM. For adapters that do not expose Dawn's
ShaderF16 feature, the shader library now selects the same register-tile
kernel family with NO_SHADER_F16 enabled and disables the F16 subgroup-matrix
variant.

The no-F16 variant aliases the shader's temporary f16 arithmetic/storage
type to f32 before including both common_decls.tmpl and
mul_mat_decls.tmpl. This is important: putting the alias after
common_decls.tmpl leaves F16 declarations in the WGSL module and is not a
valid no-ShaderF16 shader. The two packed IQ scale reads that previously used
bitcast<vec2<f16>> are explicitly decoded with unpack2x16float() in the
no-F16 branch. Quantized model weights remain in their original packed GGUF
formats; only workgroup temporaries and dequantized values use F32.

The operation gate now admits F32-output MUL_MAT with F32 activations and
quantized weights to the no-F16 register-tile path, rather than limiting it to
decode-time MatVec. Plain F32 x F32 GEMM uses the same F32 register-tile path.
Subgroup-matrix dispatch remains disabled without ShaderF16.

### Scope and remaining acceptance tests

- This enables the upstream register-tile family for the quantized formats
  implemented by the pinned revision's mul_mat_decls.tmpl and dispatch table.
  It does not add new quantization formats that upstream does not implement.
- Quantized weight blocks remain packed; no full-model F32 expansion is used.
- F16 tensor operations and unsupported op/type combinations remain rejected
  by the no-ShaderF16 gate.
- F32 shared-memory arithmetic uses more workgroup storage than the F16 path.
  Devices that exceed their workgroup-storage or binding limits must reject the
  operation and fall back safely; the backend must not claim GPU execution.
- The source patch is not proof of shader compilation, numerical equivalence,
  GPU residency, or performance. CI must compile the Android native target;
  device acceptance must compare outputs against CPU for Q4/Q5/Q6_K/Q8 and IQ
  families, then benchmark prefill and decode independently on multiple GPU
  vendors.

The CMake patch validates that the F32 alias appears before the common shader
declarations, and the native contract test checks that the no-F16 register-tile
variant, the quantized GEMM admission, and the packed IQ scale conversions
remain present.


## Portable no-ShaderF16 quantized prompt-prefill GEMM (2026-11-20)

The decode-only quantized MatVec restriction has now been widened. On adapters
without WebGPU ShaderF16, F32-output GGML_OP_MUL_MAT may use the upstream
register-tile GEMM when its activation input is F32 and its weight input is a
quantized type handled by the pinned upstream dequantization templates.

The portable variant keeps quantized model weights in their original packed
GGUF representation. Only the shader's temporary dequantized values,
workgroup/shared-memory tiles, and accumulation inputs are promoted to F32.
The implementation:

- propagates the adapter's actual supports_shader_f16 capability into the
  GGML WebGPU shader-library context;
- disables the F16 subgroup-matrix route when ShaderF16 is unavailable;
- selects the regular upstream register-tile shader with NO_SHADER_F16;
- conditionally removes enable f16 and aliases arithmetic f16 tokens to
  F32 after the packed tensor declarations have been parsed;
- decodes the two packed IQ scale values through unpack2x16float() in the
  no-ShaderF16 variant;
- retains the existing specialized quantized MatVec path for decode-sized
  batches and uses register-tile GEMM for larger batches/prompt prefill.

The operation gate now permits only F32-output MatMul with F32 activations and
a quantized weight type, in addition to the already-whitelisted safe F32/I32
operations. Unsupported types and F16-dependent operations remain rejected
from this no-ShaderF16 GPU route.

### CI failure found and corrected

Run 44801631943 passed unit tests and the early static WebGPU/OpenGL ES
contract checks, but failed during CMake configuration. The failure was caused
by two sequential CMake source patches competing for the same old guard text:
the first patch had already rewritten the guard before the second patch tried
to locate it. The generic guard is now inserted unchanged first; the later
whitelist patch applies the narrowly-scoped quantized MatMul exception exactly
once. The native contract script now checks the new MatMul marker and the
portable F32 tile/dequant/pipeline markers.

The new shader route is source-integrated and guarded by CI checks, but it is
not accepted as device-validated until the Android build succeeds, the WGSL
compiles on a physical adapter, quantized outputs match the CPU reference, and
prefill/decode performance and GPU residency are measured separately.

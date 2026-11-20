# Android GPU Backend Investigation (new-kor)

## Scope and constraints

- Keep all GPU work inside the existing llama.cpp/GGML native runtime.
- Do not add UI screens, controls, or settings.
- Preserve the current CPU backend and make every fallback observable with a concrete reason.
- Preserve arm64-v8a and the pinned llama.cpp revision until a deliberate compatibility decision is made.
- Do not change model quantization, context length, or generation limits as a performance workaround.

## Findings on the current branch

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

## WebGPU / Dawn feasibility

The pinned llama.cpp revision contains a GGML WebGPU backend whose native build path
requires `find_package(Dawn REQUIRED)`, C++20, Python shader embedding, and a Dawn
package. The project currently has none of those dependencies or Android packaging
rules. Current upstream llama.cpp documentation describes the WebGPU backend as
Dawn-based and still in progress; it does not provide a verified Android NDK build
recipe for this app's integration.

Therefore WebGPU/Dawn is not currently a safe drop-in backend for this Android runtime.
Do not turn it on by merely setting `GGML_WEBGPU=ON`.

## Vulkan feasibility

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

## Required implementation design

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

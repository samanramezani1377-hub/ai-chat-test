# Android Vulkan GPU-only runtime

This branch migrates the Android native inference path from the OpenCL backend to
llama.cpp's Vulkan backend. It does not change the model format: GGUF models and
the pinned llama.cpp revision remain in use.

## Build requirements

- Android NDK 27.2.12479018 and Android API 29+.
- The Vulkan SDK version pinned in `ci/prepare-vulkan.sh`, including `glslc`
  and SPIR-V headers.
- The runtime statically links `ggml-vulkan`; OpenCL is disabled for this branch.
- The Android system Vulkan loader is linked from the NDK sysroot. The SDK's
  desktop loader is used only for build-time shader tooling, not shipped in the APK.

## GPU-only activation contract

- The runtime requires a registered Vulkan GPU/IGPU before model loading.
- The model requests all layers for GPU offload.
- After loading, the runtime validates model tensor residency. A significant CPU
  weight allocation rejects activation; it does not retry with CPU weight fallback.
- CPU-side scheduling, tokenization, sampling, and any operations the upstream
  backend assigns to CPU are not proof of full GPU compute. The runtime report
  must be used to inspect the actual device and tensor residency; hardware testing
  is still required to establish which graph operations execute on Vulkan.

## Diagnostics to collect on a real Android device

Copy the full native diagnostic report after model activation and after at least
five consecutive prompts, including one long prompt and one long generated answer.
Check for these records:

- `VULKAN_BACKEND_DEVICE_STATE loaded=1`
- `VULKAN_GPU_DEVICE` with the actual device name
- `NATIVE_VULKAN_DEVICE` with device memory information when the driver exposes it
- `NATIVE_WEIGHT_RESIDENCY` and `VULKAN_MODEL_RESIDENCY_GPU_ONLY_OK`
- The native error/checkpoint lines if activation or inference fails

A green CI build only proves that the Android app and Vulkan backend compile and
that static checks/tests pass. It does not prove that every Vulkan driver supports
every model operation, nor does it replace testing on the target Mali-G57 device.

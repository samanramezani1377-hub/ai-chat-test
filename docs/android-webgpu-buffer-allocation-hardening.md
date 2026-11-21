# Android WebGPU buffer-allocation hardening

This change is limited to the Android native runtime and the pinned llama.cpp revision
`9871df5911a03a813518bcd623ee2eba91bf32aa`.

## Allocation policy

- GGML virtual buffers use bounded chunks (64 maximum) and no longer make the final
  chunk virtually unbounded with `SIZE_MAX / 2`.
- Chunk creation failure or chunk exhaustion is recorded by the allocator and
  propagated as a failed graph reservation. The allocator does not dereference an
  invalid chunk address or abort the process for this condition.
- WebGPU tensor buffers are checked against both `maxBufferSize` and
  `maxStorageBufferBindingSize` before Dawn buffer creation.
- The allocator's maximum chunk size is the smaller of those two limits.
- Dawn buffer creation is logged with label, requested bytes, usage, and outcome.
  Tensor-buffer allocation additionally records the requested/aligned sizes and
  the device limits.

## Device-error and fallback policy

- Uncaptured WebGPU device errors and device loss are recorded in an atomic
  device-error state instead of immediately calling `GGML_ABORT`.
- Once the WebGPU device enters an error state, its operation-support check
  rejects subsequent operations so GGML can place them on another backend.
- If tensor-buffer allocation or graph reservation fails, the existing Android
  model activation path releases partial GPU state and retries the same model
  using the CPU backend. Model weights and quantization are not modified.
- This is not a promise that an already-running GPU dispatch can be recovered
  after device loss. The fallback is at model/context initialization or later
  graph placement boundaries.

## Diagnostic log markers

- `AI_CHAT_WEBGPU_CREATE_BUFFER_BEGIN`
- `AI_CHAT_WEBGPU_CREATE_BUFFER_OK`
- `AI_CHAT_WEBGPU_CREATE_BUFFER_FAILED`
- `AI_CHAT_WEBGPU_BUFFER_BEGIN`
- `AI_CHAT_WEBGPU_BUFFER_RETURNED`
- `AI_CHAT_GRAPH_RESERVE_BEGIN`
- `AI_CHAT_GRAPH_RESERVE_END`

## Validation still required

CI must compile the patched pinned llama.cpp sources. A successful Android build
does not prove that a particular phone can allocate the selected model's buffers.
The final validation is an on-device model activation/inference test with the
full native log and tombstone symbols from the exact APK. The runtime does not
treat reported GPU free-memory values as a guaranteed allocation budget.

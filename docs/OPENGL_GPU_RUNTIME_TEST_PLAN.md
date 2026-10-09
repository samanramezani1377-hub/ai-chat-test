# OpenGL ES GPU-only runtime validation plan

This matrix separates checks that can run on a standard GitHub-hosted runner from
checks that require an Android device with an OpenGL ES 3.1-capable GPU. A
successful static CI run is not evidence that device inference is stable.

## Automated checks

Run from the repository root:

```bash
python3 scripts/test_gpu_only_contract.py
python3 scripts/test_runtime_safety_contract.py
python3 scripts/test_opengles_shader_math.py
python3 scripts/test_opengles_shaders.py
```

The shader compile test requires `glslangValidator` (`glslang-tools` on Ubuntu).
The math script checks CPU reference invariants and source wiring; it does not
dispatch GLSL or compare GPU output values.

The Android CI job also builds the debug APK, checks the final APK's ABI/native
library packaging and package ID, inspects native dynamic dependencies, runs
JVM/Android unit tests, and runs Android lint.

## Device acceptance matrix

Use a physical device whose OpenGL ES version is at least 3.1 and install the
debug APK produced by CI. Copy a known-good GGUF model to the device through the
app's normal model-import flow; do not hard-code a model path in the test.

For each case, save the app's diagnostic report and relevant logcat output.

| Case | Procedure | Pass criteria |
| --- | --- | --- |
| Cold start | Force-stop app, launch, load model, generate a short response | Model loads; OpenGL ES is reported; no native crash |
| Sequential turns | Run 20 short prompts in the same conversation | Every request completes; next turn remains usable |
| Long output | Generate the longest response practical for the device, then send another prompt | No crash or invalid KV/prompt-cache state |
| Long prompt | Increase prompt/history length in steps toward the configured context limit | Oversized prompt is bounded/rejected safely; no process death |
| Context sizes | Repeat at supported 2048, 4096 and 8192 contexts (never exceed model/device limits) | No crash; failures at memory limits are controlled and diagnosable |
| Cancellation | Cancel generation mid-decode, then send a new prompt | Cancellation returns; next request starts without overlap |
| Repeated load | Load, unload, and reload the same model 10 times | No double initialization; all runs either succeed or return a controlled error |
| Model switch | Load model A, unload, load model B, generate | No stale cache/model identity; output comes from active model |
| Background/foreground | Background during generation, resume after completion/cancellation | No leaked or orphaned native request; UI can continue |
| Memory pressure | Repeat long prompt/output cycles while observing Android memory stats | Memory does not grow without bound; failures are controlled |
| Warm/cold comparison | Compare first generation and subsequent generations | Diagnostics remain populated and internally consistent |
| Backend identity | Inspect diagnostics and native log for each load | OpenGL ES backend and actual loaded GPU-layer/context values are reported; no CPU inference fallback |

### Suggested log capture

```bash
adb logcat -c
# Reproduce one test case in the app, then:
adb logcat -d -v threadtime > ai-chat-runtime-logcat.txt
```

Capture the diagnostic report from the app as well. Record device model, Android
version, GPU renderer/vendor, model filename/quantization, context length, output
limit, test case, and whether the failure was a Java exception, native crash, or
controlled inference error.

## Shader numerical comparison

The current CI verifies GLSL compilation and deterministic CPU-reference
invariants. A true numerical GPU test still needs a harness that creates an EGL
OpenGL ES 3.1 context, dispatches the actual embedded shader against fixed
fixtures, reads back results, and compares them with CPU reference outputs under
operation-specific tolerances. Do not label CPU-only reference tests as GPU
numerical tests.

## Memory instrumentation

Use a separate diagnostic build for Android-compatible native memory tooling
where supported, then repeat sequential-turn, cancellation, unload/reload, and
long-context cases. Sanitizer success supplements but does not replace device
GPU testing; it does not detect every driver-side or GPU-buffer lifetime issue.

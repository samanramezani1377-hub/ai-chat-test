# OpenGL ES GPU-only runtime validation plan

This matrix separates checks that can run on a standard GitHub-hosted runner from
checks that require an Android device with an OpenGL ES 3.1-capable GPU. Static CI
success is not evidence that device inference, GPU arithmetic, or performance is
correct; device checks are a quality/compatibility gate, not an assumption that a
current stability issue exists.

## Automated checks

Run from the repository root:

```bash
python3 scripts/test_gpu_only_contract.py
python3 scripts/test_runtime_safety_contract.py
python3 scripts/test_opengles_shader_math.py
python3 scripts/test_opengles_shaders.py
python3 scripts/test_gguf_loader_contract.py
python3 scripts/test_diagnostics_contract.py
python3 scripts/verify_android_apk.py app/build/outputs/apk/debug/app-debug.apk
```

The shader compile test requires `glslangValidator` (`glslang-tools` on Ubuntu).
The math script checks CPU reference invariants and source wiring; it does not
dispatch GLSL or compare GPU output values. GGUF and diagnostics contract scripts
are source-level checks, not substitutes for running a real model.

The Android CI job builds the debug APK, verifies ABI/native-library packaging,
checks package ID and dynamic dependencies, runs JVM/Android unit tests, and runs
Android lint.

## Device acceptance matrix

Use a physical device with OpenGL ES 3.1 or newer and install the debug APK produced
by CI. Import a known-good GGUF model through the app's normal model-import flow;
do not hard-code a model path in the test. Capture the diagnostic report and logcat
for every case.

| Area | Procedure | Pass criteria |
| --- | --- | --- |
| Cold start | Force-stop, launch, load a small known-good model, generate a short response | Model loads and reports OpenGL ES; no CPU fallback |
| Sequential turns | Run 20 short prompts in one conversation | All complete and following turns remain usable |
| Long output | Generate a long response, then send another prompt | Output completes and the next prompt uses the correct active context |
| Long prompt | Increase prompt/history length toward the configured context limit | Input is bounded or rejected with a controlled, diagnosable result |
| Context sizes | Repeat supported 2048/4096/8192 contexts, respecting model/device limits | Correct completion or controlled resource-limit error |
| Cancellation | Cancel during decode, then send another prompt | Stop returns and the next request begins without overlap |
| Lifecycle | Load/unload/reload the same model 10 times | No duplicate initialization; resources are released/recreated correctly |
| Model switch | Load A, unload, load B, generate | No stale model/KV-cache identity |
| Background/foreground | Background during generation, resume after completion/cancellation | UI and runtime can continue; no orphaned generation |
| Memory | Repeat long prompt/output and unload/reload cycles while sampling process memory | No unbounded growth across repeated equivalent cycles |
| Diagnostics | Compare UI report against runtime logs and model settings | Model/backend/context/layers/timings are truthful; unavailable values are N/A |
| Performance | Run 5 warm-up + 10 measured prompts with fixed model/context/input/output settings | Record median TTFT, prefill tokens/s, decode tokens/s and peak memory; compare only same device/config |
| Backend identity | Inspect runtime and tensor-residency diagnostics | OpenGL ES is active and GPU residency/layer data reflects observed execution |

### Capturing device evidence

```bash
adb logcat -c
# Reproduce one case in the app, then:
adb logcat -d -v threadtime > ai-chat-runtime-logcat.txt
adb shell dumpsys meminfo com.samanramezani.aichattest > ai-chat-meminfo.txt
```

Record device model, Android version, GPU renderer/vendor, model filename/hash/quantization,
context, output limit, and measured result. Do not compare performance numbers across
different GPUs or thermal/power states.

## Actual shader numerical comparison

The current CI verifies GLSL compilation and deterministic CPU-reference invariants.
A true GPU numerical test needs a harness that creates an EGL OpenGL ES 3.1 context,
dispatches the actual embedded shaders on fixed fixtures, reads back outputs, and
compares against operation-specific CPU tolerances. Do not label CPU-only reference
tests as GPU numerical tests. Q6_K coverage should include quantized blocks and
non-multiple tail dimensions, not just Softmax/RoPE.

## Native memory instrumentation

Native sanitizer instrumentation must be built and executed on a compatible Android
device/emulator. Use an Android-compatible ASan/HWASan diagnostic build when supported,
then repeat lifecycle, cancellation, long-context and model-switch tests. A hosted
Linux source check cannot claim to detect runtime native memory errors or GPU driver
buffer-lifetime errors.

## Performance baseline format

Once a repeatable device benchmark produces a JSON record, compare it with a baseline using `python3 scripts/compare_device_benchmark.py --baseline baseline.json --current current.json --max-regression-percent 15`. The comparator rejects mismatched device/model/context identities and checks timing/memory regressions plus throughput regressions. Its synthetic self-test runs in CI; the project still needs a physical-device producer and a recorded baseline before real performance gates can run. Save JSON records with:
`device`, `gpu_renderer`, `model_sha256`, `quantization`, `context`,
`prompt_tokens`, `output_tokens`, `load_ms`, `ttft_ms`, `prefill_tokens_per_sec`,
`decode_tokens_per_sec`, and `peak_pss_kib`. Establish a per-device baseline before
enforcing percentage thresholds; do not invent universal timing limits.

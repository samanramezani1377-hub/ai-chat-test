#!/usr/bin/env python3
"""Guard diagnostic/runtime labels against stale backend names and missing metrics.

Source-level checks only: the runtime/device report remains the authority for actual
values, and a physical-device run is required to validate reported measurements.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/NativeLlamaCpp.kt"
ADAPTER = ROOT / "core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/LlamaCppAndroidRuntimeAdapter.kt"
DIAGNOSTICS = ROOT / "app/src/main/java/com/samanramezani/aichattest/ui/diagnostics/RuntimeDiagnostic.kt"
NATIVE = ROOT / "core/runtime-android/src/main/cpp/native_runtime_android_safe.cpp"


def main() -> int:
    files = [BRIDGE, ADAPTER, DIAGNOSTICS, NATIVE]
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        print("FAIL: missing source(s): " + ", ".join(missing), file=sys.stderr)
        return 2
    bridge, adapter, diagnostics, native = [p.read_text(encoding="utf-8") for p in files]
    checks = [
        ("preflight backend is OpenGL ES GPU-only", 'backend_mode=OpenGL_ES_GPU_ONLY' in bridge),
        ("bridge error describes OpenGL ES rather than OpenCL", "OpenGL ES GPU-only runtime requires" in bridge),
        ("adapter selects OpenGL ES", 'selectedBackend = "OpenGL ES"' in adapter),
        ("runtime info reports the actual native backend string", "backend = selectedBackend" in adapter),
        ("diagnostics expose model/runtime/context/GPU-layer fields",
         all(token in diagnostics for token in ("model", "runtime", "contextLength", "gpuLayers"))),
        ("diagnostics parse TTFT and generation timing",
         all(token in diagnostics for token in ("firstTokenTimeMs", "generationTimeMs"))),
        ("diagnostics parse native prefill/decode metrics",
         all(token in diagnostics for token in ("prefillMs", "decodeMs", "decodeTokensPerSec"))),
        ("diagnostics parse KV-cache measurements",
         all(token in diagnostics for token in ("cachedTokens", "reusedTokens", "cacheHitRatio"))),
        ("native trace emits prefill and generation measurements",
         "prefillMs=" in native and "generationMs=" in native),
        ("native trace emits decode profile measurements",
         "NATIVE_PERF_PROFILE" in native and "decodeMs=" in native),
    ]
    failures = []
    for label, ok in checks:
        print(("PASS: " if ok else "FAIL: ") + label)
        if not ok:
            failures.append(label)
    # The bridge must not emit an OpenCL label: it would misrepresent the active backend.
    if "OpenCL_GPU_ONLY" in bridge or "OpenCL GPU-only runtime" in bridge:
        failures.append("stale OpenCL wording remains in NativeLlamaCpp")
        print("FAIL: stale OpenCL wording remains in NativeLlamaCpp")
    print(f"Diagnostics contract: {len(checks) - sum(not ok for _, ok in checks)}/{len(checks)} passed")
    if failures:
        print("Failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("NOTE: validates source wiring, not the truth of runtime measurements on a device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

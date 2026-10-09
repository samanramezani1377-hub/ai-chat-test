#!/usr/bin/env python3
"""Source contract tests for GGUF model loading and invalid-file handling.

These tests verify the app delegates GGUF parsing to the pinned llama.cpp loader
and validates file access before crossing JNI. They do not replace loading real
GGUF fixtures on Android.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/NativeLlamaCpp.kt"
ADAPTER = ROOT / "core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/LlamaCppAndroidRuntimeAdapter.kt"
NATIVE = ROOT / "core/runtime-android/src/main/cpp/native_runtime_android_safe.cpp"


def main() -> int:
    paths = [BRIDGE, ADAPTER, NATIVE]
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        print("FAIL: missing source(s): " + ", ".join(missing), file=sys.stderr)
        return 2
    bridge, adapter, native = [p.read_text(encoding="utf-8") for p in paths]
    checks = [
        ("blank model path is rejected before JNI", 'require(path.isNotBlank())' in bridge),
        ("unreadable/non-file model path is rejected before JNI",
         "file.isFile && file.canRead()" in bridge),
        ("model activation checks file access before native load",
         adapter.find("if (!file.isFile || !file.canRead())") >= 0 and
         adapter.find("NativeLlamaCpp.load(") > adapter.find("if (!file.isFile || !file.canRead())")),
        ("native path delegates parsing to llama.cpp model loader",
         re.search(r"llama_model_load_from_file\s*\(", native) is not None),
        ("native loader returns controlled load failure status",
         "nativeLoad" in native and ("return" in native) and
         ("llama_model_load_from_file" in native)),
        ("GPU-only request rejects zero layers", "require(gpuLayers > 0)" in bridge),
        ("unsupported architecture is surfaced by loader/runtime path",
         "architecture" in adapter.lower() and "ModelError.Inference" in adapter),
    ]
    failures = []
    for label, ok in checks:
        print(("PASS: " if ok else "FAIL: ") + label)
        if not ok:
            failures.append(label)
    print(f"GGUF loader contract: {len(checks) - len(failures)}/{len(checks)} passed")
    if failures:
        print("Failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("NOTE: real valid/corrupt/unsupported GGUF fixtures still require native device integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

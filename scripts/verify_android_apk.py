#!/usr/bin/env python3
"""Verify APK ABI/native-library packaging and guard against runaway APK growth."""
from pathlib import Path
import os
import sys
import zipfile

EXPECTED_ABI = "arm64-v8a"
RUNTIME_LIB = f"lib/{EXPECTED_ABI}/libai_chat_runtime.so"
FORBIDDEN_MARKERS = ("libOpenCL", "libvulkan", "libVulkan")
# Deliberately broad ceiling: catches accidental model bundling/duplicate native
# runtimes without imposing a device/network-dependent exact APK size.
DEFAULT_MAX_APK_MIB = 180


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: verify_android_apk.py <apk-path>", file=sys.stderr)
        return 2
    apk = Path(sys.argv[1])
    if not apk.is_file() or apk.stat().st_size == 0:
        print(f"FAIL: APK missing or empty: {apk}", file=sys.stderr)
        return 2
    try:
        with zipfile.ZipFile(apk) as archive:
            names = archive.namelist()
            bad = archive.testzip()
            if bad:
                raise ValueError(f"corrupt ZIP entry: {bad}")
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        print(f"FAIL: invalid APK archive: {exc}", file=sys.stderr)
        return 1

    native_entries = [name for name in names if name.startswith("lib/") and name.endswith(".so")]
    abi_dirs = sorted({name.split("/")[1] for name in native_entries if len(name.split("/")) >= 3})
    size_mib = apk.stat().st_size / (1024 * 1024)
    max_size_mib = float(os.environ.get("AI_CHAT_MAX_APK_MIB", DEFAULT_MAX_APK_MIB))
    checks = [
        ("APK is a valid, non-empty ZIP", bad is None),
        ("runtime JNI library is packaged", RUNTIME_LIB in names),
        ("only arm64-v8a native ABI is packaged", bool(native_entries) and abi_dirs == [EXPECTED_ABI]),
        ("no OpenCL/Vulkan native library is packaged",
         not any(marker.lower() in name.lower() for name in names for marker in FORBIDDEN_MARKERS)),
        (f"APK size is below regression ceiling ({max_size_mib:g} MiB)", size_mib <= max_size_mib),
    ]
    failures = []
    for label, ok in checks:
        print(("PASS: " if ok else "FAIL: ") + label)
        if not ok:
            failures.append(label)
    print(f"APK verification: {len(checks) - len(failures)}/{len(checks)} passed")
    print(f"APK size: {apk.stat().st_size} bytes ({size_mib:.2f} MiB); native ABI(s): {abi_dirs}")
    if failures:
        print("Failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Deterministic CPU-reference invariants for shader math and source wiring.

This checks reference mathematics and verifies that expected shader paths still
contain their relevant operations. It does NOT execute the GLSL or compare GPU
outputs; that requires an EGL/OpenGL ES-capable runtime harness.
"""
from pathlib import Path
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "core/runtime-android/src/main/cpp/ggml-opengles/ggml-opengles.cpp"


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exps = [math.exp(v - maximum) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def rope_pair(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return x * c - y * s, x * s + y * c


def near(a: float, b: float, tolerance: float = 1e-6) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def main() -> int:
    if not SOURCE.is_file():
        print(f"FAIL: missing shader source {SOURCE}", file=sys.stderr)
        return 2
    source = SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r'static\s+const\s+char\s*\*\s*softmax_shader\s*\([^)]*\)\s*\{\s*return\s+R"\((.*?)\)"\s*;',
        source, re.DOTALL,
    )
    rope_match = re.search(
        r'static\s+const\s+char\s*\*\s*rope_shader\s*\([^)]*\)\s*\{\s*return\s+R"\((.*?)\)"\s*;',
        source, re.DOTALL,
    )
    if not match or not rope_match:
        print("FAIL: could not extract Softmax/RoPE shader source", file=sys.stderr)
        return 1
    softmax_src, rope_src = match.group(1), rope_match.group(1)
    checks = []

    # Softmax should remain finite, normalized, and stable for large logits.
    for values in ([0.0, 0.0], [1.0, 2.0, 3.0], [1000.0, 1001.0, 999.0], [-1000.0, -999.0]):
        out = softmax(list(values))
        checks.append((f"softmax finite/normalized for {values}",
                       all(math.isfinite(v) and v >= 0.0 for v in out) and near(sum(out), 1.0)))
    checks.append(("softmax preserves ordering", softmax([1.0, 2.0, 3.0])[0] < softmax([1.0, 2.0, 3.0])[1] < softmax([1.0, 2.0, 3.0])[2]))
    norm_checks = []
    for angle in (0.0, 0.1, math.pi / 2, math.pi, -2.3):
        rx, ry = rope_pair(3.0, 4.0, angle)
        norm_checks.append(near(rx * rx + ry * ry, 3.0 * 3.0 + 4.0 * 4.0, 1e-5))
    checks.append(("RoPE preserves pair norm", all(norm_checks)))
    checks.append(("RoPE zero angle is identity", all(near(a,b) for a,b in zip(rope_pair(2.5, -7.0, 0.0), (2.5, -7.0))))
    checks.append(("Softmax shader uses max-subtraction stabilization", "max" in softmax_src and re.search(r"exp\s*\(", softmax_src) is not None))
    checks.append(("Softmax output buffer is readable for normalization", bool(re.search(r"buffer\s+Y", softmax_src))))
    checks.append(("RoPE shader includes sine/cosine rotation", "sin(" in rope_src and "cos(" in rope_src))
    checks.append(("RoPE signed/unsigned index regression guard remains", "float(int(d0/2u)-cumulative)" in rope_src))

    failures = []
    for label, ok in checks:
        print(("PASS: " if ok else "FAIL: ") + label)
        if not ok:
            failures.append(label)
    print(f"Shader math/reference invariants: {len(checks) - len(failures)}/{len(checks)} passed")
    if failures:
        print("Failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("NOTE: reference math and source wiring only; this does not validate GPU numerical output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

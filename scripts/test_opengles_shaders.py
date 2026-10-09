#!/usr/bin/env python3
"""Regression tests for the embedded OpenGL ES compute shaders.

Extracts every raw GLSL shader from ggml-opengles.cpp and compiles it with
glslangValidator in OpenGL ES 3.1 mode. Also guards the exact regressions that
previously blocked Q6_K, Softmax, and RoPE shader initialization.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "core/runtime-android/src/main/cpp/ggml-opengles/ggml-opengles.cpp"


def extract_shaders(source: str) -> dict[str, str]:
    pattern = re.compile(
        r'static\\s+const\\s+char\\s*\\*\\s*(\\w+)\\s*\\([^)]*\\)\\s*\\{\\s*return\\s+R"\\((.*?)\\)"\\s*;',
        re.DOTALL,
    )
    return {match.group(1): match.group(2) for match in pattern.finditer(source)}


def main() -> int:
    if not SOURCE.is_file():
        print(f"ERROR: shader source not found: {SOURCE}", file=sys.stderr)
        return 2

    source = SOURCE.read_text(encoding="utf-8")
    shaders = extract_shaders(source)
    if not shaders:
        print("ERROR: no embedded GLSL shaders were extracted", file=sys.stderr)
        return 2

    required = {"q6k_matmul_shader", "softmax_shader", "rope_shader"}
    missing = sorted(required - shaders.keys())
    if missing:
        print(f"ERROR: expected shader(s) missing: {', '.join(missing)}", file=sys.stderr)
        return 1

    # Regression guards for the exact GLSL compiler failures previously seen on device.
    q6k = shaders["q6k_matmul_shader"]
    softmax = shaders["softmax_shader"]
    rope = shaders["rope_shader"]
    if re.search(r"\\bout\\b", q6k):
        print("FAIL: Q6_K shader uses reserved GLSL identifier 'out'", file=sys.stderr)
        return 1
    if not re.search(r"uint\\s+output_idx\\s*=", q6k) or "output_idx/(rows*cols)" not in q6k:
        print("FAIL: Q6_K output index declaration/use is inconsistent", file=sys.stderr)
        return 1
    if not re.search(r"layout\\s*\\(std430,\\s*binding\\s*=\\s*2\\)\\s*buffer\\s+Y", softmax):
        print("FAIL: Softmax output must be readable during in-place normalization", file=sys.stderr)
        return 1
    if "float(int(d0/2u)-cumulative)" not in rope:
        print("FAIL: RoPE signed/unsigned conversion regression detected", file=sys.stderr)
        return 1

    validator = shutil.which("glslangValidator")
    if not validator:
        print("ERROR: glslangValidator is required (install package glslang-tools)", file=sys.stderr)
        return 2

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="opengles-glsl-") as temp_dir:
        for name, shader in sorted(shaders.items()):
            shader_path = Path(temp_dir) / f"{name}.comp"
            shader_path.write_text(shader, encoding="utf-8")
            result = subprocess.run(
                [validator, "-S", "comp", str(shader_path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode:
                failures.append(f"{name}:\\n{result.stdout}")
                print(f"FAIL GLSL compile: {name}")
            else:
                print(f"PASS GLSL compile: {name}")

    if failures:
        print("\\n".join(failures), file=sys.stderr)
        return 1

    print(
        f"PASS: {len(shaders)} embedded OpenGL ES compute shaders compile; "
        "Q6_K, Softmax, and RoPE regression guards passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

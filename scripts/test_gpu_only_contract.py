#!/usr/bin/env python3
"""Static contract tests for the Android OpenGL ES GPU-only runtime.

These tests do not replace native compilation or real-device inference tests.
They guard build/runtime invariants that can otherwise regress silently.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "cmake": ROOT / "core/runtime-android/src/main/cpp/CMakeLists.txt",
    "native": ROOT / "core/runtime-android/src/main/cpp/native_runtime_android_safe.cpp",
    "runtime": ROOT / "core/runtime-android/src/main/cpp/opengles_runtime.cpp",
    "backend": ROOT / "core/runtime-android/src/main/cpp/ggml-opengles/ggml-opengles.cpp",
    "registration": ROOT / "core/runtime-android/src/main/cpp/ggml-opengles/ggml-opengles-registration.cpp",
    "gradle": ROOT / "core/runtime-android/build.gradle.kts",
}


def require(label: str, condition: bool, detail: str) -> bool:
    if condition:
        print(f"PASS: {label}")
        return True
    print(f"FAIL: {label} — {detail}", file=sys.stderr)
    return False


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in FILES.values() if not path.is_file()]
    if missing:
        print("FAIL: required source file(s) missing: " + ", ".join(missing), file=sys.stderr)
        return 2

    src = {name: path.read_text(encoding="utf-8") for name, path in FILES.items()}
    checks = [
        ("GPU backend is statically linked", "add_library(ai_chat_runtime SHARED" in src["cmake"] and
         "ggml-opengles/ggml-opengles.cpp" in src["cmake"] and
         "ggml-opengles/ggml-opengles-registration.cpp" in src["cmake"],
         "native backend sources must be part of ai_chat_runtime"),
        ("OpenGL ES system libraries are linked", "target_link_libraries(ai_chat_runtime PRIVATE llama llama-common android log EGL GLESv3)" in src["cmake"],
         "EGL and GLESv3 must be linked"),
        ("Dynamic backend loading remains disabled", "set(GGML_BACKEND_DL OFF" in src["cmake"] and
         "GGML_BACKEND_DL=OFF" in src["gradle"],
         "GPU backend must be statically registered"),
        ("CPU repacking remains disabled", "set(GGML_CPU_REPACK OFF CACHE BOOL" in src["cmake"] and
         "set(GGML_CPU_REPACK ON" not in src["cmake"] and "GGML_CPU_REPACK=ON" not in src["gradle"],
         "CPU weight repacking must not be enabled"),
        ("GPU-only scheduler patch excludes CPU backend", "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH" in src["cmake"] and
         "if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU) {\n                continue;" in src["cmake"],
         "the CPU backend must not be inserted into the graph scheduler"),
        ("Build fails if GPU-only scheduler patch is missing", "message(FATAL_ERROR" in src["cmake"] and
         "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH was not applied" in src["cmake"],
         "CMake must fail closed if the upstream patch stops applying"),
        ("OpenGL ES compute capability is validated", "OPENGL_ES_31_COMPUTE_UNAVAILABLE" in src["runtime"] and
         "GL_COMPUTE_SHADER" in src["runtime"],
         "runtime must reject devices without compute-shader support"),
        ("OpenGL ES backend is registered statically", "ggml_backend_register(ggml_backend_opengles_reg())" in src["registration"],
         "backend registration must not depend on dynamic loading"),
        ("GPU-only runtime compile definitions are enabled", "AI_CHAT_GPU_BACKEND_OPENGL=1" in src["cmake"] and
         "AI_CHAT_REQUIRE_OPENGL_GPU=1" in src["cmake"],
         "GPU-only OpenGL compile definitions are required"),
        ("Android native ABI remains arm64-only", 'abiFilters += "arm64-v8a"' in src["gradle"],
         "the app's native ABI contract is arm64-v8a only"),
        ("No OpenCL/Vulkan backend is configured", "GGML_OPENCL=OFF" in src["gradle"] and
         "GGML_VULKAN=OFF" in src["gradle"] and
         "GGML_USE_OPENCL" not in src["cmake"],
         "OpenCL/Vulkan must not become alternate inference backends"),
    ]
    passed = sum(require(*check) for check in checks)
    print(f"GPU-only contract tests: {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

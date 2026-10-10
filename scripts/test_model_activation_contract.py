#!/usr/bin/env python3
"""Regression checks for the model activation -> native context-ready contract.

These are source-level checks. They catch activation-order regressions in CI,
but only a real Android device can prove that its driver can create the context.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "core/runtime-android/src/main/cpp/native_runtime.cpp"
GPU = ROOT / "core/runtime-android/src/main/cpp/ggml-opengles/ggml-opengles.cpp"
ADAPTER = ROOT / "core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/LlamaCppAndroidRuntimeAdapter.kt"
BRIDGE = ROOT / "core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/NativeLlamaCpp.kt"
ACTIVE_NATIVE = ROOT / "core/runtime-android/src/main/cpp/native_runtime_android_safe.cpp"
CMAKE = ROOT / "core/runtime-android/src/main/cpp/CMakeLists.txt"


def section(text: str, start: str, end: str) -> str:
    a = text.find(start)
    if a < 0:
        return ""
    b = text.find(end, a + len(start))
    return text[a:] if b < 0 else text[a:b]


def main() -> int:
    paths = [NATIVE, GPU, ADAPTER, BRIDGE, ACTIVE_NATIVE, CMAKE]
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        print("FAIL: missing source(s): " + ", ".join(missing), file=sys.stderr)
        return 2

    native, gpu, adapter, bridge, active_native, cmake = [p.read_text(encoding="utf-8") for p in paths]
    load = section(native, "Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeLoad", "Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeStop")
    context = section(native, "static bool init_generation_context() {", "static bool init_speculative_runtime() {")
    support = section(gpu, "static bool supports_op", "static enum ggml_status graph_compute")
    ensure = section(gpu, "static bool ensure_programs", "static size_t tensor_offset")
    graph = section(gpu, "static enum ggml_status graph_compute", "static ggml_backend_i ggml_backend_opengles")
    checks = [
        ("GPU-only activation rejects unavailable OpenGL ES device without CPU fallback",
         "OPENGL_ES_GPU_ONLY_REJECTED_NO_OPENGL_ES_GPU_DEVICE" in load and
         "OPENGL_ES_GPU_ONLY_NO_CPU_FALLBACK" in load),
        ("context creation is explicitly checked and failed activation frees partial state",
         "const bool context_ready = init_generation_context();" in load and
         "checkpoint(context_ready ? \"CONTEXT_INIT_RETURNED_SUCCESS\" : \"CONTEXT_INIT_RETURNED_FAILED\")" in load and
         "if (!context_ready)" in load and "free_all(); return 2;" in load),
        ("native load success marker is emitted only after context readiness",
         load.find("if (!context_ready)") >= 0 and
         load.find("if (!context_ready)") < load.find('checkpoint("NATIVE_LOAD_COMPLETED")') and
         load.find('checkpoint("NATIVE_LOAD_COMPLETED")') > load.find("init_generation_context()")),
        ("context setup requires a loaded model and nonzero context",
         "if (!g_model || g_context_length == 0) return false;" in context),
        ("native fatal handler records broad runtime phase separately from speculative phase",
         "NATIVE_FATAL_PHASE=" in native and 'g_native_phase = "LLAMA_CONTEXT_INIT"' in load),
        ("context-init trace brackets llama_init_from_model for crash localization",
         "CONTEXT_INIT_ENTER_LLAMA_INIT_FROM_MODEL" in load and
         "CONTEXT_INIT_RETURNED_FROM_LLAMA_INIT" in load),
        ("native fatal report retains PC/LR/fault address markers",
         all(token in native for token in ("NATIVE_FATAL_PC=", "NATIVE_FATAL_LR=", "NATIVE_FATAL_FAULT_ADDR="))),
        ("installed Android fatal handler reports signal name and runtime phase",
         all(token in active_native for token in ("NATIVE_FATAL_SIGNAL_NAME=", "NATIVE_FATAL_PHASE=", "g_native_phase"))),
        ("pinned llama.cpp scheduler exposes internal reservation stage markers",
         all(token in cmake for token in ("AI_CHAT_SCHEDULER_TRACE_PATCH", "AI_CHAT_SCHED_STAGE=before_backend_scheduler_create", "AI_CHAT_SCHED_STAGE=before_memory_init_full", "AI_CHAT_SCHED_STAGE=before_resolve_fused_ops", "AI_CHAT_SCHED_STAGE=before_prompt_graph_reserve", "AI_CHAT_SCHED_STAGE=before_token_graph_reserve", "AI_CHAT_SCHED_STAGE=reserve_graphs_completed"))),
        ("GPU memory marked unknown is not reported as a measured zero",
         'if (nativeField(line, "memoryKnown") == "1")' in
         (ROOT / "core/runtime/src/main/kotlin/com/woogit/aicore/runtime/RuntimeDiagnostics.kt").read_text(encoding="utf-8")),
        ("GDN scheduler support is not gated by a shader handle compiled later",
         "GGML_OP_GATED_DELTA_NET" in support and
         re.search(r"if\s*\(op->op == GGML_OP_GATED_DELTA_NET.*?g_gdn\s*!=\s*0", support, re.S) is None),
        ("graph-result allocation stages are traced inside llama_graph_result::reset",
         all(marker in cmake for marker in (
             "AI_CHAT_GRAPH_RESULT_STAGE=before_compute_meta_resize",
             "AI_CHAT_GRAPH_RESULT_STAGE=before_ggml_init",
             "AI_CHAT_GRAPH_RESULT_STAGE=before_new_graph_custom",
             "AI_CHAT_GRAPH_RESULT_STAGE=after_new_graph_custom"))),
        ("GPU-only scheduler no longer trips llama.cpp's CPU-last assertion",
         "AI_CHAT_GPU_ONLY_SCHEDULER_TAIL_PATCH" in cmake and
         "CPU backend is intentionally absent" in cmake and
         "GGML_BACKEND_DEVICE_TYPE_CPU" in cmake),
        ("scheduler remains GPU-only rather than restoring a CPU fallback",
         "if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU) {" in cmake and
         "continue;" in cmake and "OPENGL_ES_GPU_ONLY_NO_CPU_FALLBACK" in load),
        ("required GDN shader compilation failure is detected before dispatch",
         'compile_compute(gated_delta_net_shader(), "gated_delta_net")' in ensure and
         "(caps.max_compute_ssbo_blocks < 7 || g_gdn != 0)" in ensure),
        ("GPU graph prepares shaders before dispatching graph operations",
         "if (!ensure_programs()) return GGML_STATUS_FAILED;" in graph and
         "if (op->op == GGML_OP_GATED_DELTA_NET)" in graph and "glUseProgram(g_gdn)" in graph),
        ("Kotlin activation treats nonzero native load code as failure",
         "if (result != 0) return ModelResult.Failure" in adapter),
        ("Kotlin marks the model loaded only after native load succeeds",
         adapter.find("if (result != 0) return ModelResult.Failure") >= 0 and
         adapter.find("loadedModelPath = file.absolutePath") > adapter.find("if (result != 0) return ModelResult.Failure")),
        ("JNI boundary logs activation begin/end and validates GPU-only request",
         "ACTIVATION_LOAD_BEGIN" in bridge and "ACTIVATION_LOAD_END" in bridge and
         'require(gpuLayers > 0)' in bridge),
    ]
    failed = []
    for label, ok in checks:
        print(("PASS: " if ok else "FAIL: ") + label)
        if not ok:
            failed.append(label)
    print(f"Model activation contract: {len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        print("Failures: " + "; ".join(failed), file=sys.stderr)
        return 1
    print("LIMITATION: these checks validate activation ordering and regression guards, not device/driver context creation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

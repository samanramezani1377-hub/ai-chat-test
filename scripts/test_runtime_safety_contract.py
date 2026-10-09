#!/usr/bin/env python3
"""Static regression checks for Android runtime safety boundaries.

These checks protect critical lifecycle/cancellation/context guards. They are
not a substitute for executing the native runtime with a real GGUF model.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/LlamaCppAndroidRuntimeAdapter.kt"
NATIVE_BRIDGE = ROOT / "core/runtime-android/src/main/kotlin/com/woogit/aicore/runtime/android/NativeLlamaCpp.kt"
NATIVE_CPP = ROOT / "core/runtime-android/src/main/cpp/native_runtime_android_safe.cpp"
SHARED_NATIVE_CPP = ROOT / "core/runtime-android/src/main/cpp/native_runtime.cpp"


def main() -> int:
    paths = [ADAPTER, NATIVE_BRIDGE, NATIVE_CPP]
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        print("FAIL: missing source(s): " + ", ".join(missing), file=sys.stderr)
        return 2
    adapter, bridge, native = [p.read_text(encoding="utf-8") for p in paths]
    shared_native = SHARED_NATIVE_CPP.read_text(encoding="utf-8")
    checks = [
        ("load/unload/generation serialized by coroutine mutex",
         "private val nativeOperationMutex = Mutex()" in adapter and
         "nativeOperationMutex.withLock" in adapter,
         "runtime operations must share the adapter mutex"),
        ("duplicate activation is skipped for the already-loaded model",
         "loadedModelPath == requestedPath" in adapter and
         "loadedDraftPath == requestedDraftPath" in adapter,
         "same model and draft should not trigger a second native initialization"),
        ("generation refuses to run without a loaded model",
         'ModelError.RuntimeUnavailable("No local model is loaded")' in adapter,
         "unloaded generation must return a controlled error"),
        ("prompt token budget reserves output capacity",
         "contextCapacity - (promptTokens ?: 0) - 32" in adapter and
         "minOf(settings.maxNewTokens.coerceAtLeast(1), promptBudget)" in adapter,
         "effective generation tokens must be bounded by remaining context"),
        ("oversized newest message is fitted instead of passed unbounded",
         "fitMessageToBudget(message, budget)" in adapter and
         "truncatedLatest" in adapter,
         "single oversized message must be trimmed before JNI"),
        ("cancelled native generation is stopped and joined before flow closes",
         "nativeStop()" in bridge and "worker.join()" in bridge and
         "runBlocking { worker.join() }" in bridge,
         "do not allow a later request to race an active JNI decode"),
        ("JNI load validates model path and GPU-only request",
         'require(gpuLayers > 0)' in bridge and
         'require(path.isNotBlank())' in bridge and
         'file.isFile && file.canRead()' in bridge,
         "reject invalid paths and zero-GPU-layer requests before JNI"),
        ("native operations have a process-wide mutex",
         "static std::mutex g_native_runtime_mutex" in native,
         "serialize native model/generation state"),
        ("prompt cache identity includes model and context",
         "g_cached_model" in native and "g_cached_context" in native and
         "g_cached_prompt_tokens" in native,
         "KV/prompt cache must be scoped to its owning model/context"),
        ("native cache clear resets token and recurrent state",
         "g_cached_prompt_tokens.clear()" in native and
         "g_cached_prompt_state.clear()" in native and
         "g_cached_prompt_state_on_device = false" in native,
         "all host-side cache state must be cleared together"),
        ("draft initialization is guarded against duplicate init",
         "SPECULATIVE_DOUBLE_INIT_GUARD" in native and
         "if (g_spec || g_spec_init)" in native,
         "draft runtime must not initialize twice"),
        ("active OpenGL fatal handler re-delivers signals to Android debuggerd",
         "g_android_previous_fatal_actions" in native and
         "sigaction(signal_number, &previous, nullptr)" in native and
         "syscall(SYS_tgkill" in native and
         "NATIVE_FATAL_PC=" in native and "NATIVE_FATAL_LR=" in native and
         "g_android_fatal_handlers_installed" in native and
         "AI_CHAT_EXTERNAL_FATAL_HANDLER 1" in native and
         "#ifndef AI_CHAT_EXTERNAL_FATAL_HANDLER" in shared_native and
         "default_action.sa_handler = SIG_DFL" not in native and
         "raise(signal_number)" not in native,
         "the active handler must restore the original debuggerd action and re-deliver the fatal signal, without replacing it with SIG_DFL"),
        ("shared JNI init does not shadow debuggerd with a legacy handler",
         shared_native.count("#ifndef AI_CHAT_EXTERNAL_FATAL_HANDLER") >= 2,
         "the included native runtime must leave fatal-handler ownership to the Android wrapper"),
        ("native library keeps symbols needed to resolve crash addresses",
         "-g" in (ROOT / "core/runtime-android/src/main/cpp/CMakeLists.txt").read_text(encoding="utf-8") and
         "-Wl,--strip-all" not in (ROOT / "core/runtime-android/src/main/cpp/CMakeLists.txt").read_text(encoding="utf-8"),
         "do not strip the only symbols that can map a native PC to a function/source line"),
    ]
    failed = []
    for label, ok, detail in checks:
        print(("PASS: " if ok else "FAIL: ") + label)
        if not ok:
            failed.append(f"{label}: {detail}")
    print(f"Runtime safety contract: {len(checks) - len(failed)}/{len(checks)} passed")
    if failed:
        print("\n".join(failed), file=sys.stderr)
        return 1
    print("NOTE: source-level regression checks only; real-model/device stress remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

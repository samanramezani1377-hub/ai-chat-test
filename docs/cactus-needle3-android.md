# Cactus Needle 3 on Android

This branch replaces the retired Laya INT4 decision router with the official Cactus Needle 3 native Android engine. WebGPU is intentionally not part of this implementation.

## User-facing workflow

The model controls stay in the existing Voice conversation > Settings > Local Persian decision model card.

- Download fetches the pinned official needle3.cact archive from Hugging Face.
- The active copy is kept under the app's private files/voice-models/cactus-needle3/ directory.
- The downloaded model is also published through Android MediaStore into Download/AI Chat Test/Cactus/needle3.cact, so it is visible in file managers.
- Import from phone storage accepts a .cact file and copies it into the app's active model directory.
- Model weights are not bundled in the APK.
- The downloaded weights are pinned to Hugging Face revision d6c7d6a2e9655505e0f8e23339356dd1705900eb, expected size 35,335,380 bytes, and checked against the indexed SHA-256 prefix/suffix before activation.

## Runtime

The Android module builds the official android-arm64/libneedle.a from the same pinned Hugging Face revision and links it into libcactus_needle_jni.so. The archive download is checked for its expected byte length and SHA-256 fingerprint. The app remains arm64-v8a only.

The C API is process-global and not thread-safe. The JNI bridge serializes every call with one mutex. The Kotlin router initializes the model and tool schemas once, resets the conversation before each independent utterance, and sends the original transcript unchanged. Current date/time values are resolved by the Android device executor when the corresponding tool is run. The always-on service must pass the exact private model file path ending in `needle3.cact`, not the containing directory.

## Tool contract

Needle is given five tools only:

- get_clock: choose time or date for a direct request for the current local time or today's date.
- adjust_volume: increase, decrease, mute, or unmute media audio.
- set_volume: set an explicitly stated target on a 0–10 scale. A bare number from 0–10 is a level; any explicit percentage (even 7%) and any bare number above 10 are rounded to the nearest 10% step.
- adjust_brightness: increase or decrease screen brightness.
- set_brightness: set an explicitly stated target using the same 0–10 / nearest-10%-step rule.

The model does not generate conversational answers. The existing chat model receives the unchanged utterance only when Needle emits no tool call (including an explicit suppressed/no-call result). If Needle emits a tool call but it is malformed, unknown, missing required arguments, or below the confidence gate, the app does not forward that request to the chat model; it returns a short clarification/error instead. A valid tool call is executed locally and the chat model is not invoked. No regex or keyword classifier is used as a fallback. State-changing actions require confidence >= 0.70; read-only time/date requests require >= 0.50. Persian confidence calibration has not been validated on the user's device, so this is a conservative gate rather than a measured accuracy claim.

## Native engine source

- Model: Cactus-Compute/needle3
- Pinned model/engine revision: d6c7d6a2e9655505e0f8e23339356dd1705900eb
- Platform library: android-arm64/libneedle.a
- API: needle_load, needle_init, needle_complete, needle_reset
- License: Apache-2.0

The official Cactus API exposes one process-global model and conversation. Do not add parallel inference calls or bypass the JNI mutex. Do not claim GPU acceleration: this branch links the vendor's Android ARM64 native engine and does not enable WebGPU.

## Removed Laya components

The Laya decision engine, ONNX tokenizer wrapper, Rust tokenizer module, model bundle manager, and Laya-specific router documentation are removed from this branch. The Sherpa-ONNX static AAR remains for independent STT/TTS. Microsoft ONNX Runtime Java/JNI remains because the separate MiniLM intent-routing classes still use it; only the Laya-specific tokenizer and router dependencies are removed.


## Voice output reliability

Piper playback uses a bounded AudioTrack write/drain sequence. A stalled playback head is re-primed once before reporting a segment failure; a failed segment is logged without terminating the remaining speech queue. Recoverable TTS segment failures do not by themselves mark the always-on conversation as stopped or failed. Channel/control markers are removed before transcript persistence and TTS; partial channel markers are held until complete, and an empty/control-only response is not persisted or spoken. Physical playback reliability still requires testing on the target Android device.

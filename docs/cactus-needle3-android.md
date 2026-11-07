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

Needle receives five compact tool schemas, matching the official recommendation to keep the catalogue at five or fewer because Needle 3 has no retrieval head:

- get_clock: read current local time or today's date, selected by the `action` enum.
- adjust_volume: increase/decrease media volume or mute/unmute, selected by the `action` enum.
- set_volume: set a target media volume. `value` preserves the number spoken (0–100); `is_percent` is true only when the user explicitly said percent / درصد. Bare values are interpreted as device steps when possible, with the executor retaining the existing percent fallback for values above the device's maximum.
- adjust_brightness: increase or decrease screen brightness, selected by the `action` enum.
- set_brightness: set screen brightness to the stated percentage (0–100).

Each schema includes official Needle 3 `triggers` regexes for common Persian request forms. These are part of the model's tool schema; there is no parallel Kotlin keyword/regex classifier. Descriptions include common colloquial Persian phrasings and parameter meanings. The catalogue deliberately remains at five tools because the current official Needle 3 reference states that it has no retrieval head.

The native response is handled using the official envelope: `type`, `success`, `function_calls`, `suppressed_calls`, `reasoning`, and `confidence`. Multiple calls in one response are validated as a complete set before any command executes. A non-empty `suppressed_calls` list is a withheld tool decision, never a chat fallback. Only `type=call`, `success=true`, empty `function_calls`, and empty `suppressed_calls` is treated as an explicit no-tool abstention and forwarded to the conversational Qwen model. Runtime failures, malformed envelopes, unknown tools, invalid arguments, and low-confidence calls do not fall through to Qwen.

State-changing actions require confidence >= 0.70; read-only time/date requests require >= 0.50. Cactus documents that confidence is calibrated on the base model but non-English deployments need their own validation; Persian calibration has not been measured on the target phone. The app logs the exact ASR transcript, bounded raw Needle response, selected/suppressed calls, reasoning, confidence and runtime metrics for diagnosis.

## Android permissions for the current tool set

- Current time/date: no Android permission is required.
- Media volume: `MODIFY_AUDIO_SETTINGS` is already declared in the manifest; it is a normal permission and does not require a runtime dialog.
- Screen brightness: `WRITE_SETTINGS` is declared. Android treats this as special app access, not a runtime permission; the executor opens `ACTION_MANAGE_WRITE_SETTINGS` only when the user asks to change brightness and access has not been granted.
- Voice input itself uses `RECORD_AUDIO`; always-on operation additionally uses the existing foreground-service and notification permissions. These are requested by the voice workflow, not by Needle tool selection.

The app does not request unrelated permissions in advance.

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

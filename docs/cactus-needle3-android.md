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

The C API is process-global and not thread-safe. The JNI bridge serializes every call with one mutex. The Kotlin router initializes the model and tool schemas once, resets the conversation before each independent utterance, and sends the original transcript plus current device date/time facts.

## Tool contract

Needle is given four tools only:

- get_time: direct request for the current local time.
- get_date: direct request for today's date.
- set_volume: increase, decrease, mute, unmute, or set an explicitly stated target on a 0–10 scale. A bare number from 0–10 is a level; any explicit percentage (even 7%) and any bare number above 10 are rounded to the nearest 10% step.
- set_brightness: increase, decrease, or set an explicitly stated target using the same 0–10 / nearest-10%-step rule.

The model does not generate conversational answers. If it returns no single valid tool call, returns a suppressed call, has insufficient confidence, omits a required numeric target, or returns an unknown tool, the router abstains and the unchanged utterance continues to the existing conversation model. No regex or keyword classifier is used as a fallback. State-changing actions require confidence >= 0.70; read-only time/date requests require >= 0.50. Below the threshold the router abstains and forwards the original utterance to the existing chat model. Persian confidence calibration has not been validated on the user's device, so this is a conservative gate rather than a measured accuracy claim.

## Native engine source

- Model: Cactus-Compute/needle3
- Pinned model/engine revision: d6c7d6a2e9655505e0f8e23339356dd1705900eb
- Platform library: android-arm64/libneedle.a
- API: needle_load, needle_init, needle_complete, needle_reset
- License: Apache-2.0

The official Cactus API exposes one process-global model and conversation. Do not add parallel inference calls or bypass the JNI mutex. Do not claim GPU acceleration: this branch links the vendor's Android ARM64 native engine and does not enable WebGPU.

## Removed Laya components

The Laya decision engine, ONNX tokenizer wrapper, Rust tokenizer module, model bundle manager, and Laya-specific router documentation are removed from this branch. The Sherpa-ONNX static AAR remains for independent STT/TTS. Microsoft ONNX Runtime Java/JNI remains because the separate MiniLM intent-routing classes still use it; only the Laya-specific tokenizer and router dependencies are removed.

# Qwen3.5-2B WebGPU on Android

## Verified model artifact

The WebAI Community publishes a Qwen3.5-2B ONNX Runtime GenAI export under:

- Model folder: [Qwen3.5-2B/onnx-webgpu](https://huggingface.co/webai-community/ai-models/tree/main/Qwen3.5-2B/onnx-webgpu)
- Model notes and export provenance: [Qwen3.5-2B README](https://huggingface.co/webai-community/ai-models/blob/main/Qwen3.5-2B/README.md)
- Current listed folder size: about 1.78 GB.

The package contains `genai_config.json`, `model.onnx`, `model.onnx.data`, `tokenizer.json`, `tokenizer_config.json`, `config.json`, and `chat_template.jinja`. Its GenAI config requests the `webgpu` provider and sets `enableGraphCapture=0` and `validationMode=basic`.

The publisher lists these SHA-256 values for the ONNX graph and external weights:

- `model.onnx`: `565610B25156AA0DBC2E9F8E80766ED1658CFCC47F09DCF226BC692FF20F939A`
- `model.onnx.data`: `4DF448BC35E944F053EFD0130972E8224CA17BAE330F37951BA799066CDFC74A`

## Importing into Dina

1. Download **only** the `Qwen3.5-2B/onnx-webgpu/` directory, not the whole `ai-models` repository.
2. ZIP the *contents* of that directory so `genai_config.json` is at the ZIP root. Do not leave an extra `Qwen3.5-2B/onnx-webgpu/` parent directory inside the ZIP.
3. In Dina's model-import UI, import that ZIP and activate it.
4. Check runtime diagnostics. The backend field now reports the provider requested by `genai_config.json`; it explicitly warns that unsupported operators may fall back to another provider.

## Important runtime boundary

The model artifact is a **GenAI WebGPU export**, not a generic ONNX graph that can be run by simply selecting `executionProviders: ['webgpu']` in ONNX Runtime Web. It uses GenAI-specific generation/KV-cache configuration and a WebGPU provider configuration. The current Dina branch uses the native Android ONNX Runtime GenAI Java API; it does not embed a browser-based ONNX Runtime Web engine.

Therefore, provider configuration in the model file is not by itself proof that the Android AAR contains a working WebGPU execution provider on every device. A successful APK build only proves packaging/compilation. Actual GPU inference must be confirmed on a physical Android device by activating this exact model and checking logs/per-token latency. If the native AAR does not expose WebGPU on that device, a separate WebView + ORT Web/GenAI Web runtime must be built and integrated; it cannot be achieved by switching a Kotlin flag.

## Acceptance checks for a real-device WebGPU run

- Model import validates all required assets and preserves the ONNX external-data file.
- Model activation succeeds without provider/session initialization errors.
- Runtime diagnostics identify the requested WebGPU provider.
- A short Persian prompt generates coherent output and reaches EOS/stop handling.
- Logs confirm whether unsupported operations fell back to CPU; do not infer GPU use from the model name alone.
- Record first-token latency, decode tokens/sec, peak memory, device/GPU/Android version, and whether WebGPU was actually used.
- Repeat after force-stop/relaunch to verify persistence and activation restore.

No performance claim should be made until these checks pass on a real device.

# Qwen3.5-2B WebGPU on Android (experimental)

Dina now has a **separate experimental WebGPU chat screen**. Open the app's side menu and choose **گفت‌وگوی WebGPU آزمایشی**. This path uses an Android WebView, Transformers.js, and the browser's WebGPU API. It does not replace the native ONNX Runtime GenAI path.

## Model artifact used by the WebGPU screen

- Transformers.js model: [onnx-community/Qwen3.5-2B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT)
- Reference implementation: [webml-community/Qwen3.5-WebGPU](https://huggingface.co/spaces/webml-community/Qwen3.5-WebGPU)
- Transformers.js runtime: `@huggingface/transformers@4.2.0`, loaded from jsDelivr.
- Device selection: `device: "webgpu"`; model dtype is configured as Q4 for embeddings and the merged decoder, matching the published reference implementation.

The browser path uses the Transformers.js ONNX-OPT export. **Do not import the separate `Qwen3.5-2B/onnx-webgpu` GenAI package ZIP into this browser route.** The GenAI package and Transformers.js ONNX-OPT package have different model layouts and runtime APIs. The native chat screen still uses the ONNX Runtime GenAI ZIP importer.

## How it works

1. The Compose app opens a local HTML page from Android assets using `WebViewAssetLoader` and the secure `https://appassets.androidplatform.net` origin.
2. The page checks `window.isSecureContext`, `navigator.gpu`, and `navigator.gpu.requestAdapter()` before loading any model.
3. On first use, the app downloads the Q4 text-inference files from Hugging Face into the same app-private model root used by Laya: `files/voice-models/laya-int4/webgpu/onnx-community/Qwen3.5-2B-ONNX-OPT/`. The WebView serves these files through `WebViewAssetLoader`; the model is not dependent on the evictable browser cache. Transformers.js runtime code still loads from jsDelivr.
4. Text generation runs in the WebView using WebGPU. Prompt content is passed to the local inference runtime; it is not sent to a chat inference API.
5. If WebGPU is not exposed by the device's Android System WebView, the screen shows an error and the user can return to the existing native chat. This screen does not silently claim CPU fallback.

## Network and privacy

- First launch requires internet to fetch the Transformers.js runtime and model files. The total model download is large (around gigabyte scale).
- Inference is intended to happen on-device. Model weights and JavaScript runtime code are fetched from their publishers; chat messages are not sent to a remote inference endpoint by this implementation.
- Qwen weights are stored in app-private persistent storage alongside Laya, not in public Downloads. Each completed file is reused on the next launch; incomplete `.part` files are downloaded again. App uninstall clears app-private files. The Transformers.js JavaScript package itself is still loaded from jsDelivr, so the whole WebGPU route is not yet fully offline.
- The app does not expose a JavaScript bridge to native Android APIs from this WebView. Local assets are served through WebViewAssetLoader; direct file access is disabled.

## Real-device acceptance checklist

The code and CI can validate packaging and syntax, but they cannot prove that a particular phone's WebView supports the required GPU operations. Test on physical devices from different GPU vendors.

1. Install the debug APK and open the WebGPU screen.
2. Verify it reports a secure context and detects a non-null WebGPU adapter.
3. Load the 2B model and wait for a clear `مدل آماده است` status.
4. Send several Persian prompts and verify streamed output, stopping, and a fresh conversation.
5. Inspect Android logcat for `DinaWebGPU` errors and the WebView console.
6. Force-stop and reopen the app; verify the Qwen files are detected in the shared Laya model root and are not downloaded again.
7. Record phone model, Android/WebView version, GPU adapter information (if exposed), first-load time, first-token latency, decode tokens/sec, memory pressure, heat, and any WebGPU validation errors.
8. Repeat with WebGPU unavailable (or on a device where it is disabled) and verify the app shows an actionable error without claiming GPU acceleration.

## Current limits

- This is an **experimental, independent chat route**, not yet merged into the native model manager, native conversation-history database, voice pipeline, or agent/action system.
- The WebGPU model is stored next to Laya in app-private storage, but the native ONNX GenAI import/activate controls do not manage these weights.
- It has its own in-page conversation history; reset clears that in-memory history.
- It is text-only. Image input and integration with Dina's voice/wake-word features are not wired up here.
- No real-device execution or speed claim is made until the acceptance checklist passes on hardware.

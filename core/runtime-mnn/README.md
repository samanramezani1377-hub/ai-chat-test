# MNN LLM Android runtime (experimental)

This module pins Alibaba MNN 3.6.1 (commit d407447ed56c4121a11ccbd266dc184ca1ead0c2) and builds its LLM engine with Android OpenCL enabled.

## Model format

This runtime does not load GGUF. It expects an MNN-exported model directory containing config.json, llm.mnn, llm.mnn.weight, llm_config.json, and the tokenizer/embedding files required by that export. Use the official MNN LLM export workflow to produce these artifacts. Keep all files together; the adapter path must point to config.json.

## Backend policy

1. Try MNN with backend_type=opencl.
2. If OpenCL creation/load fails, destroy that runtime and retry with backend_type=cpu.
3. If OpenCL generation reports an internal runtime error, discard partial text, recreate the runtime on CPU, and replay the full message list. GPU and CPU output are never concatenated.
4. Logs distinguish requested/selected backend. Selecting OpenCL is not proof that every operator executed on GPU; some operators may be internally assigned to CPU. Device-side profiling is still required before claiming full GPU execution.

## Build

The module fetches a pinned MNN source commit at CMake configure time and builds only arm64 Android. The first native build is large and requires network access to GitHub.

## Limitations

- Generation is buffered before it is delivered to the UI so a failed OpenCL response can be discarded safely before CPU replay. Incremental token display is not implemented in this initial adapter.
- The import flow accepts an exported MNN ZIP pack, but GGUF remains handled by llama.cpp.
- A native fatal signal (SIGABRT/SIGSEGV) can terminate the process before in-process fallback executes. A separate process is needed to recover from GPU driver crashes.

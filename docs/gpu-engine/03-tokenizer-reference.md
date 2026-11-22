# Tokenizer reference validation

## Scope

The built-in `gpt2/qwen35` adapter implements the Qwen3.5 byte-level BPE profile. It consumes the vocabulary, merge ranks, token-type table and special-token IDs parsed from the selected GGUF file. Model names and token IDs are not used as tokenizer-selection shortcuts.

The adapter is marked `Validated` only while the native reference-parity test remains part of CI. Other built-in BPE and SentencePiece profiles remain `Unverified` until they receive equivalent independent reference suites.

## Reference

CI pins the upstream `Qwen/Qwen3.5-0.8B` tokenizer assets to revision `2b48083dfa97c3cbf0220cf1a5e6cffe0a511157`:

- https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2b48083dfa97c3cbf0220cf1a5e6cffe0a511157/tokenizer.json
- https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2b48083dfa97c3cbf0220cf1a5e6cffe0a511157/tokenizer_config.json
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/tokenization_qwen3_5.py

The upstream implementation specifies NFC normalization, the Qwen3.5 Unicode pre-tokenization expression, byte-level BPE and the ByteLevel decoder. The CI test builds a minimal GGUF fixture from the pinned tokenizer assets and compares native token IDs and decoded text against the Hugging Face `tokenizers` runtime.

## Run locally

Install the pinned reference runtime and run:

```bash
python3 -m pip install tokenizers==0.22.1
bash scripts/test_qwen35_tokenizer_parity.sh
```

The test is CPU-only and does not require Android, a GPU, Vulkan, a model-weight download or a separate tokenizer file on the device.

## Model-specific metadata

The runtime continues to read these values from each GGUF:

- `tokenizer.ggml.model` and `tokenizer.ggml.pre`
- `tokenizer.ggml.tokens` and `tokenizer.ggml.merges`
- `tokenizer.ggml.token_type`
- BOS, EOS, unknown and padding token IDs
- BOS/EOS insertion and byte/whitespace behavior flags

Different valid PAD IDs are therefore retained per file. Diagnostics include both special-token IDs and their corresponding token strings to expose conversion inconsistencies.

## Limits

Passing this profile test validates the Qwen3.5 tokenizer algorithm against the pinned upstream assets. It does not validate a model architecture, graph, quantization kernel, GPU backend or end-to-end generation. It also does not validate other tokenizer profiles or arbitrary custom tokenizers that happen to use the same metadata labels.

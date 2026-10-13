# Laya Multilingual INT4 — Voice Decision Router

The live voice conversation now uses the Laya multilingual INT4 typed-choice model before invoking the local chat model. The model is a decision engine, not a text generator. Inference stays on-device.

## Import the model

In **Voice Conversation → Settings → Local Persian decision-making**, import a ZIP bundle containing the complete model repository with this layout:

```text
laya-multilingual-int4-layer32-vocab128.onnx
laya-multilingual-int4-layer32-vocab128.onnx.data
rl_agent_config.json
tokenizer/
  tokenizer.json
  tokenizer_config.json
encoder/
  config.json
```

The importer accepts a common parent directory around these files. It rejects path traversal, caps the number of archive entries and total expanded size, validates the graph and external weights, stages the bundle, and only then replaces the previous installed model. The 4-bit model weights are stored in app-private files; they are not embedded in the APK. The app requires both the ONNX graph and its external-data file, because the graph references the large weight file by its original name.

Model source: [androidli/laya-multilingual-onnx-int4](https://huggingface.co/androidli/laya-multilingual-onnx-int4/tree/main). Select `laya-multilingual-int4-layer32-vocab128.onnx` and include its matching `.data` file and the tokenizer/config files when preparing the ZIP.

## Decision flow

```text
STT transcript (original text)
  └─ Laya choice: request category
       ├─ AUDIO       → Laya chooses operation
       │    ├─ INCREASE / DECREASE → change Android media volume by one native step
       │    ├─ MUTE / UNMUTE        → apply the selected action
       │    └─ SET_VOLUME           → Laya chooses target level 0–10; executor applies percentage
       ├─ DISPLAY     → Laya chooses operation; for SET, Laya chooses target percentage
       ├─ INFORMATION → Laya chooses GET_TIME / GET_DATE / CHAT
       │    ├─ GET_TIME / GET_DATE → local device value → Persian text → local TTS
       │    └─ CHAT                → forward the exact original transcript
       ├─ APPLICATION / MEDIA / SYSTEM → reserved and inactive; no device action
       └─ OTHER       → no device action; forward the exact original transcript
```

For absolute volume, Laya chooses a target level from 0–10 itself; a percentage such as 50% must be mapped by Laya to level 5, then converted to 50% for the Android executor. Relative increase/decrease commands change the Android media stream by exactly one native volume step. The executor validates values against Android's supported range.

Laya is the sole intent and value decision-maker. If the bundle is missing, ONNX inference fails, or Laya abstains, no device command is inferred by a regex/number parser; the original transcript is forwarded unchanged to the local chat model. APPLICATION, MEDIA, and SYSTEM are reserved and inactive until their executors are implemented.

## ONNX inputs and diagnostics

The runner supplies Laya's typed-choice inputs: `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, and `qtype`. The tokenizer is read from the imported bundle and the graph runs through the app's ONNX Runtime dependency. Errors use `LAYA-*` diagnostic codes; decision stages are recorded as `LAYA-DECISION` entries without sending the transcript to a service.

## Verification

The automated CI build and unit-test workflow validates compilation and the existing voice-command rules. Device validation should additionally verify:
- the INT4 graph loads on Android with the external `.data` file;
- the tokenizer produces valid marker positions for Persian text;
- `دینا ساعت چنده` routes to local clock speech and does not start chat inference;
- `دینا صداشو کم کن` chooses a lower target from the actual current media volume;
- `دینا صدا رو بذار روی ۳` selects a level in the model's 1–10 choice set;
- `دینا صدا رو بذار روی ۳۰ درصد` follows the explicit-value branch;
- a normal chat transcript reaches `session.send(utterance, ...)` unchanged.

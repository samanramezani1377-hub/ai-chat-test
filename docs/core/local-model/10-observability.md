# 10 — Observability / Action Trace

```text
Observability
├── Import Trace
├── Validation Trace
├── Inspection Trace
├── Registration Trace
├── Activation Trace
├── Load Trace
├── Inference Trace
└── Trace Payload
    ├── operationId
    ├── executionId
    ├── traceId
    ├── modelId
    ├── stage
    ├── start/end
    ├── real duration when measurable
    ├── status
    ├── errorCode
    └── non-sensitive metadata
```

هر Action واقعی باید Trace داشته باشد. UI Workspace و Diagnostics در صورت نیاز از همین شناسه‌ها استفاده می‌کنند و نباید Trace موازی بسازند.

ارجاع: [Operations](07-operations.md)، [UI Integration](15-ui-integration.md)

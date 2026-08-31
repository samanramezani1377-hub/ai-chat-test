# Failure Matrix

| Failure | Required behavior | Success condition |
|---|---|---|
| Model/Runtime error | Stop current inference, log raw error, show Persian error | Safe terminal state |
| Network/API error | Preserve operation/state, log request/result metadata | Retry/recovery policy succeeds or safe failure |
| Capability unavailable | Do not execute | Clear unsupported-capability result |
| Validation failure | Do not execute | Invalid operation rejected |
| Approval denied | Do not execute | Task remains non-executed |
| Crash before Execute | Resume/inspect state | No duplicate execution |
| Crash during/after Execute | Verify external state before retry | State known before next action |
| Execute succeeds, Verify fails | Mark uncertain, inspect/recover | Verified final state |
| Recovery fails | Stop unsafe retries, surface error | Explicit terminal failure |
| Error Center unavailable | Logging Core continues independently | Errors remain available through supported storage/API |

## General rule

Failure is not Success. Action completion is confirmed only after the required Verification step.

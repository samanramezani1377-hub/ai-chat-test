# Test Matrix

## هدف

تبدیل Requirements اصلی به تست‌های قابل تکرار و جلوگیری از Success جعلی.

| Area | Scenario | Expected evidence |
|---|---|---|
| Model | Import valid GGUF | model metadata + load success |
| Model | Invalid model | validation error |
| Chat | Persian prompt with Local selected | streamed local response; no provider request |
| Chat | Persian prompt with Provider selected | request sent to selected provider; provider response/error surfaced |
| Settings | Change inference source | selected source persists and is shown in UI |
| Privacy | Provider selected | only request-required conversation context is sent; secrets redacted from logs |
| Routing | Selected source unavailable | explicit error; no silent fallback to another inference source (Local ↔ Provider) |
| Runtime backend | GPU available and supported | actual GPU backend/offload is reported accurately |
| Runtime backend | GPU unavailable or initialization/offload fails; CPU is available | local inference continues on CPU when feasible; diagnostics identify CPU as actual backend and retain GPU failure context |
| Runtime backend | GPU and CPU both unavailable or fail | explicit runtime error with diagnostic evidence |
| Runtime backend | CPU fallback succeeds | successful local generation; must not be classified as failure solely because CPU performed inference |
| Generation | Stop | generation stops without fake completion |
| Action | Valid request | executor result + verification |
| Action | Invalid JSON | parser error, no execution |
| Action | Unknown action | validation error, no execution |
| Action | Permission denied | blocked, no execution |
| Action | Confirmation rejected | cancelled, no execution |
| Action | Execution failure | execution error |
| Action | Verification failure | verification error, not success |
| Agent | Multi-step | each step traced |
| Agent | Retry | retry consumes a step |
| Agent | Step limit | task blocked/stopped at configured limit |
| Agent | Cancel | task enters cancelled state |
| Workspace | Action result | real result visible |
| Workspace | Before/After | state evidence visible when supported |
| Performance | TTFT | real measurement |
| Performance | Tokens/sec | real measurement |
| Network | Offline | no required cloud dependency |
| Network | Network use | actual usage observable |
| Persistence | History | save/read/delete behavior verified |
| UI Navigation | Normal cold launch | Chat is the initial destination; Wake Word entry may open Voice |
| UI Navigation | Primary destinations | Bottom navigation switches Chat / History / Settings without losing current conversation state |
| Settings | Voice settings | Shenava/STT, Piper/TTS, Laya and always-on listener controls are reachable from central Settings |
| History | Search | Title and loaded conversation message text are searchable |
| History | Open voice conversation | Voice transcript is available from the shared conversation history and can be continued in Chat |
| Voice UI | Partial assistant output | Sanitized assistant text appears during token streaming without waiting for TTS completion |
| Voice UI | Transcript growth | Latest recognized and generated lines remain visible through automatic scrolling |
| Export | Report | exported data matches source |
| Screen Awareness | Persian direct screen question | Screen-question classifier routes page/button/error questions to local Vision |
| Screen Awareness | Persian deictic follow-up | Contextual phrases such as «اینو چطور حل کنم؟» trigger screen inspection |
| Screen Awareness | Generic chat question | Unrelated questions do not trigger screen capture or Vision inference |
| Screen Awareness | Frame freshness and privacy | Only latest in-memory frame is returned; stale frames expire and returned bytes are defensive copies |
| Screen Awareness | Stop screen session | Frame buffer clears and native Vision model unload is queued |
| Screen Awareness | Vision idle timeout | Native Vision model unloads after 60 seconds without another visual request |
| Screen Awareness | Cloud provider selected | Screen pixels and visual observations remain local; no screen data is sent to the provider |
| Floating bubble | Display bounds | Initial and dragged bubble coordinates keep the full bubble inside display bounds |
| Floating bubble | Service state | Persisted enabled state follows successful overlay creation and actual service teardown |
| Vision model files | Atomic replacement | Failed replacement preserves the last known-good model file |
| Home widget | Provider discovery | Chat and voice providers appear as two separately addable home-screen widgets |
| Home widget chat | Compose interaction | Tap opens only compact composer activity; MainActivity is not launched |
| Home widget chat | Send and stream | Text request runs through AppContainer in a foreground service and streamed/final reply updates RemoteViews |
| Home widget chat | Persistence | Widget conversation is visible in app history after opening the full app |
| Home widget voice | One-shot capture | One tap starts microphone capture; Shenava endpoint silence ends capture and microphone is released |
| Home widget voice | No continuous listening | After one response, microphone stays off until the next explicit widget tap |
| Home widget voice | Spoken response | Piper speaks the assistant answer and widget status returns to ready |
| Home widget voice | Permission missing | Service does not attempt capture; actionable permission guidance is displayed |
| Home widget service | Background work | Foreground notification is present during inference/capture and removed after completion |
| Home widget service | Concurrent requests | A second request is rejected while a widget request is already running |
| Home widget service | Process/model state | Active local model is restored if possible; unavailable model/provider error is explicit |

## Evidence rules

هر تست باید در صورت امکان حداقل این موارد را ثبت کند:

- Input
- Expected
- Actual
- Execution state
- Verification state
- Error code در صورت شکست
- Timestamp

هیچ تستی نباید صرفاً بر اساس Final Answer مدل سبز اعلام شود.

## Benchmark protocol

Performance Test باید بین Cold و Warm run تفکیک شود. سناریوهای کوتاه، بلند، Single-step و Multi-step باید جداگانه قابل اجرا باشند. تعداد تکرارها باید در Test Run قابل تنظیم باشد و نتیجه خام نگهداری شود.

تصمیم نهایی درباره کفایت Performance بر اساس شواهد واقعی و بررسی کاربر است؛ Threshold اجباری در Specification وجود ندارد.

# Test Matrix

## هدف

تبدیل Requirements اصلی به تست‌های قابل تکرار و جلوگیری از Success جعلی.

| Area | Scenario | Expected evidence |
|---|---|---|
| Model | Import valid GGUF | model metadata + load success |
| Model | Invalid model | validation error |
| Chat | Persian prompt | streamed local response |
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
| Export | Report | exported data matches source |

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

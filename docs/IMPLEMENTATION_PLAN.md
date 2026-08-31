# Implementation Plan

## هدف

تبدیل Specification پروژه به مسیر اجرایی مرحله‌ای، بدون قفل‌کردن معماری به UI یا یک مدل خاص.

## Phase 1 — Foundation

- AI Core interfaces
- Model metadata
- Model import/validation
- Runtime adapter
- basic local inference

## Phase 2 — Chat

- conversation state
- streaming
- stop generation
- inference settings
- error handling

## Phase 3 — Action System

- Action Protocol
- parser
- validator
- permission layer
- confirmation flow
- executor
- verifier
- structured results

## Phase 4 — Agent

- task state machine
- multi-step loop
- configurable Maximum Agent Steps
- retry accounting
- cancellation
- trace IDs

## Phase 5 — Workspace

- Workspace state model
- Action/Tool result cards
- verification evidence
- Preview
- confirmation
- Before/After
- interactive controls

## Phase 6 — Observability

- inference metrics
- model metrics
- device metrics
- agent metrics
- network monitoring
- history
- export
- independently configurable metric visibility

## Phase 7 — QA

- Test Matrix execution
- regression tests
- failure-path tests
- offline tests
- performance benchmark runs

## Phase 8 — WooGit extraction

- isolate reusable AI Core
- define WooGit adapters
- map WooGit Operations to Action Protocol
- preserve permission/confirmation/verification
- integrate Chat and UI AI Operations
- validate with real WooGit data only after Prototype is proven

## Definition of Done

هر Phase زمانی کامل است که:

- Build/CI سالم باشد.
- قابلیت واقعی باشد، نه Mock.
- مسیر خطا تست شده باشد.
- Result قابل مشاهده و قابل Trace باشد.
- مستندات مربوطه به‌روز باشند.
- Contractهای قبلی شکسته نشده باشند.

## Source of truth

جزئیات Requirement در `PROTOTYPE_SPEC.md`، قرارداد Action در `ACTION_PROTOCOL.md`، State در `TASK_STATE_MACHINE.md`، امنیت در `SECURITY_MODEL.md`، Workspace در `AI_WORKSPACE.md` و QA در `TEST_MATRIX.md` تعریف می‌شود.

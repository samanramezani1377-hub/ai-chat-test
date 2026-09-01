# AI Chat Test — UI Specification

**Status:** 🟡 UI DESIGN DECISION PHASE  
**Version:** 0.2.0  
**Scope:** UI/UX documentation only  
**Source of truth:** The structured documents under [`docs/ui/`](ui/README.md).

> **IMPORTANT:** UI v1 is **not locked yet**. The previous broad specification is retained as a baseline, but the final UI contract will be produced from the 20 owner decisions. No UI implementation may begin before all decisions are answered, reviewed, and explicitly approved.

## Documentation structure

The UI specification is intentionally split into focused documents:

- [`ui/README.md`](ui/README.md) — UI documentation entry point and workflow
- [`ui/DESIGN_SYSTEM.md`](ui/DESIGN_SYSTEM.md) — visual system
- [`ui/NAVIGATION.md`](ui/NAVIGATION.md) — navigation contract
- [`ui/STATES.md`](ui/STATES.md) — state contract
- [`ui/SCREENS.md`](ui/SCREENS.md) — screen map
- [`ui/questions/README.md`](ui/questions/README.md) — 20-question decision process

## Baseline requirements

The baseline UI must:

- Make Chat the primary interaction.
- Provide Workspace, Errors, and Settings as primary areas.
- Clearly communicate AI/runtime state.
- Support streaming, errors, retry, loading, empty, offline, and recovery states.
- Treat Persian/RTL as first-class.
- Support accessibility and responsive layouts.
- Keep technical diagnostics separate from normal user-facing UX.
- Preserve the existing Core architecture and test quality.
- Never use fake production data or placeholders as a substitute for real state.

## Decision and lock process

```text
Baseline UI requirements
        ↓
20 owner decisions
        ↓
Documentation updates
        ↓
Review for consistency / architecture / accessibility
        ↓
Explicit UI v1 approval
        ↓
UI LOCK
        ↓
Implementation
        ↓
Tests + CI + UX verification
```

## Change-control rules

1. No UI implementation during the decision phase.
2. Every new UI decision is documented before implementation.
3. Each accepted answer is committed before the next question is asked.
4. Conflicts between decisions require an explicit revision rather than a silent change.
5. CI is not a design authority.
6. Core/API limitations must not silently degrade the approved UX.
7. After UI Lock, any UI change requires a specification revision and version increment.

## Current phase

**PHASE: UI DOCUMENTATION + 20 DECISION QUESTIONS**

The next active decision is [`Q01`](ui/questions/Q01.md).

No production UI code should be added until the decision process and final review are complete.

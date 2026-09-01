# AI Chat Test — UI Specification

**Status:** Design phase only  
**Scope:** UI/UX documentation; no UI implementation in this phase  
**Rule:** This document is the source of truth for the future UI implementation.

## 1. Product UI Goals

The application should feel like a polished, modern AI workspace rather than a technical test harness.

Primary goals:
- Make the conversation the primary interaction.
- Keep AI generation state immediately understandable.
- Make actions, errors, diagnostics, and settings accessible without cluttering the chat.
- Preserve a calm, focused visual hierarchy.
- Use consistent components and states across every screen.
- Never expose internal architecture terminology unless the user is viewing diagnostics.

## 2. Current State and Design Decision

The current application UI is a minimal native Android screen containing four buttons and placeholder screens. It is not the final design and must not be treated as the target visual implementation.

The UI implementation is intentionally frozen while this specification is being prepared.

## 3. Information Architecture

### Primary areas

1. **Chat** — primary destination and default screen.
2. **Workspace** — active tasks, actions, and execution context.
3. **Errors** — user-facing failures, recovery options, and diagnostics.
4. **Settings** — model, conversation, appearance, privacy, and diagnostic preferences.

### Secondary areas

- Conversation history
- Conversation details
- Action details
- Error details
- Runtime/model status
- About and diagnostics

## 4. Navigation

### Default entry

The application opens on **Chat**.

### Primary navigation

Use a persistent mobile-friendly navigation pattern. The exact Android component is an implementation decision and is not prescribed by this document.

Required destinations:
- Chat
- Workspace
- Errors
- Settings

### Navigation rules

- Back returns to the previous logical destination.
- Opening a detail view must preserve the originating screen state.
- Navigating away from an active conversation must not discard unsent text.
- Returning to Chat must preserve the current conversation and scroll position when practical.
- Destructive navigation must never silently discard user work.

## 5. Design Language

### Visual direction

The visual language should be **modern, premium, calm, and lightweight**, with subtle depth rather than heavy decoration.

Principles:
- Clear hierarchy over decoration.
- Generous spacing.
- Soft surfaces and restrained elevation.
- Rounded interactive surfaces.
- Minimal visual noise.
- Strong readability in both light and dark themes.

Do not use excessive gradients, excessive blur, neon effects, or ornamental animations.

## 6. Color System

Define semantic colors rather than screen-specific colors.

Required semantic roles:
- Background
- Surface
- Elevated surface
- Primary text
- Secondary text
- Disabled text
- Primary action
- Secondary action
- Success
- Warning
- Error
- Information
- User message surface
- Assistant message surface
- Code/technical surface
- Divider/border

Light and dark themes must use the same semantic roles while allowing different concrete values.

## 7. Typography

Typography must establish three clear levels:

- **Display/Screen title** — page identity.
- **Section title** — groups related content.
- **Body/UI text** — normal interaction and content.

Additional semantic styles:
- Caption/metadata
- Error text
- Button label
- Monospace technical text

Rules:
- Never rely on font size alone to communicate hierarchy.
- Persian text must remain readable and correctly aligned.
- Mixed Persian/English/technical strings must not produce awkward visual jumps.
- User-selectable system font scaling must remain usable.

## 8. Spacing and Shape

Use a consistent spacing scale across the application.

Required tokens:
- Screen horizontal padding
- Section spacing
- Component internal padding
- Message spacing
- Input spacing
- Dialog spacing

Surfaces should use a consistent corner-radius family with larger radii for major containers and smaller radii for compact controls.

## 9. Core Components

### 9.1 App Shell

Responsibilities:
- Application identity.
- Current destination.
- Runtime status when relevant.
- Navigation access.

### 9.2 Status Indicator

Must communicate:
- Connected
- Connecting
- Ready
- Generating
- Offline
- Error

Status must never depend only on color; include text/icon/state semantics.

### 9.3 Chat Message

Each message supports:
- Sender identity.
- Message content.
- Timestamp/metadata when useful.
- Generation state for assistant responses.
- Error state where applicable.

Assistant and user messages must be visually distinct without making either visually dominant unnecessarily.

### 9.4 Message Composer

Required states:
- Empty
- Typing
- Sending
- Generating
- Disabled
- Error

Required interactions:
- Enter/send behavior must be predictable.
- Send must provide immediate feedback.
- During generation, the user must have a clear way to stop generation when supported.
- Failed messages must expose retry without forcing the user to recreate the message.

### 9.5 Loading Indicator

Loading must communicate what is happening. Avoid indefinite generic spinners when a more specific state can be shown.

### 9.6 Error Banner/Card

Must include:
- What happened in user-facing language.
- Whether the operation can be retried.
- Retry action when applicable.
- Technical details only in diagnostics/details.

### 9.7 Empty State

Every empty screen must explain:
- What the area is for.
- Why it is empty.
- What the user can do next.

### 9.8 Buttons

Semantic variants:
- Primary
- Secondary
- Tertiary/text
- Destructive

Buttons must have clear pressed, disabled, and loading states.

## 10. Chat Screen

The Chat screen is the primary product surface.

### Layout

From top to bottom:
1. App/conversation header.
2. Optional runtime/model status.
3. Conversation content.
4. Composer anchored near the bottom.
5. Navigation remains accessible without covering conversation content.

### Chat behavior

- New conversation starts visually clean.
- Messages remain readable during streaming.
- Assistant streaming must not cause disruptive layout jumps.
- Long responses must support comfortable scrolling.
- Code and technical output require a visually distinct treatment.
- Errors belong close to the affected operation.

## 11. Conversation History

History should show:
- Conversation title.
- Last activity.
- Optional short preview.
- Clear active/current state.

Required states:
- Loading
- Empty
- Populated
- Search/no-result if search is introduced
- Error

## 12. Workspace Screen

Workspace represents active execution context around the conversation.

It should expose:
- Current task.
- Active action.
- Action state.
- Relevant progress.
- User intervention requirements.
- Completed/failed actions.

Do not expose raw internal state machines as the primary UI. Translate them into understandable states.

## 13. Action Details

Action detail must show:
- Action name.
- Current state.
- What the action is doing.
- Relevant input/output summary.
- Recovery option when available.
- Technical details behind an expandable diagnostics section.

States to document visually:
- Pending
- Running
- Awaiting approval
- Awaiting verification
- Completed
- Failed
- Recoverable
- Cancelled

## 14. Error Center

The Error Center is a user-facing recovery area, not merely a log viewer.

### Error categories

- Network/runtime
- Model
- Action execution
- Validation
- Permission/authorization
- Configuration
- Unknown/internal

### Error presentation

Each error should answer:
1. What happened?
2. Is anything required from the user?
3. Can it be retried?
4. What is the next safe action?

Technical diagnostics may include:
- Error identifier.
- Trace identifier.
- Timestamp.
- Technical message.
- Relevant context.

## 15. Settings

Settings should be grouped by user intent rather than implementation modules.

Suggested groups:

### AI / Model
- Selected model.
- Generation preferences.
- Runtime status.

### Conversation
- History behavior.
- New conversation behavior.
- Message preferences.

### Appearance
- Theme.
- Dynamic/system appearance where supported.
- Text/display preferences.

### Privacy
- Local data behavior.
- Diagnostics/telemetry controls where applicable.

### Diagnostics
- Runtime information.
- Logs/traces access.
- Reset/recovery tools.

## 16. Runtime States

The UI must define a visual representation for each global runtime state:

| State | Meaning | Required UX |
|---|---|---|
| Initializing | App is starting | Brief, non-blocking startup state where possible |
| Ready | Runtime can accept work | Normal interaction |
| Connecting | Runtime is being connected | Explain that connection is in progress |
| Generating | AI is producing output | Streaming feedback + stop option when supported |
| Offline | Required runtime/network unavailable | Explain impact + recovery |
| Error | Runtime cannot operate normally | User-facing error + recovery |

## 17. AI Generation States

Assistant response states:

1. Queued
2. Generating
3. Streaming
4. Completed
5. Interrupted
6. Failed
7. Retry available

The transition between states must be understandable without exposing internal implementation details.

## 18. Accessibility

Requirements:
- Adequate touch targets.
- Sufficient contrast.
- Content descriptions for meaningful icons.
- Semantic labels for controls.
- Screen-reader-friendly navigation order.
- Support for system font scaling.
- Do not communicate important information by color alone.
- Respect reduced-motion preferences where applicable.

## 19. Localization and Persian UI

The UI must support Persian as a first-class language.

Requirements:
- RTL layout support.
- Correct Persian typography.
- Correct alignment of mixed RTL/LTR content.
- Technical identifiers, URLs, code, and numbers must remain readable.
- UI strings must not be hard-coded into screen logic.

## 20. Motion

Motion should explain state changes, not decorate the interface.

Allowed purposes:
- Navigation transitions.
- Message appearance.
- Streaming state changes.
- Expand/collapse diagnostics.
- Progress/state transitions.

Avoid:
- Continuous decorative animation.
- Animation that delays interaction.
- Excessive bouncing or scaling.

## 21. Responsive Layout

The design must remain usable across:
- Small Android phones.
- Large Android phones.
- Tablets.
- Portrait.
- Landscape.

The exact breakpoints will be defined during implementation, but the design must not assume a single fixed screen size.

## 22. UI State Matrix

Every screen must document at least:

- Initial
- Loading
- Empty
- Content
- Success
- Error
- Disabled
- Offline where relevant

Interactive controls must also define:
- Normal
- Pressed
- Focused
- Disabled
- Loading

## 23. Security and Privacy UX

The UI must not expose:
- Secrets.
- API keys.
- Authentication tokens.
- Sensitive internal credentials.

Diagnostics must redact sensitive values.

## 24. Performance UX

The UI should remain responsive while AI generation and actions execute.

Requirements:
- Never block the main interaction unnecessarily.
- Streaming content should appear incrementally.
- Long-running actions should expose meaningful progress/state.
- Avoid unnecessary full-screen redraws.
- Loading states should appear promptly.

## 25. Design-to-Implementation Rules

When implementation begins:

1. This document is the UI contract.
2. Components must be reusable rather than duplicated per screen.
3. UI state must come from real application state.
4. No fake data, mocks, or placeholder production UI may be introduced.
5. UI implementation must not weaken existing tests or CI quality.
6. Existing Core architecture must remain independent from presentation concerns.
7. Accessibility and RTL support are requirements, not later polish.
8. Every new screen must define its states before implementation.

## 26. Definition of Done for UI

UI is considered ready only when:

- All primary screens have documented layouts.
- All important states have documented behavior.
- Navigation is documented.
- Component semantics are documented.
- Light/dark themes are defined.
- RTL/Persian behavior is defined.
- Accessibility requirements are satisfied.
- Loading, streaming, error, retry, and empty states are covered.
- No screen relies on placeholder content.
- UI behavior is connected to real application state.
- CI continues to pass with the existing test suite intact.

## 27. Documentation Workflow

Before writing UI code for any screen:

1. Document the screen purpose.
2. Document layout and hierarchy.
3. Document all states.
4. Document interactions.
5. Document navigation entry/exit.
6. Document accessibility/localization requirements.
7. Review the specification.
8. Only then implement.

**Current phase:** documentation only. No UI implementation should be added until the specification is reviewed and approved.

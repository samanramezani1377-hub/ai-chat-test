# UI Navigation

**Status:** Draft — decisions pending

## Primary destinations

```text
App
├── Chat
├── Workspace
├── Errors
└── Settings
```

Secondary/detail destinations will be defined by the 20 decisions.

## Rules

- Chat is the primary interaction.
- Back preserves logical screen state.
- Unsaved message text must not be silently discarded.
- Active conversation state must survive navigation.
- Navigation must not expose internal architecture unnecessarily.
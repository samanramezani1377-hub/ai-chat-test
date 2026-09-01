# UI Documentation

This directory is the structured source of truth for the AI Chat Test UI.

## Documentation tree

```text
ui/
├── README.md
├── DESIGN_SYSTEM.md
├── NAVIGATION.md
├── STATES.md
├── SCREENS.md
└── questions/
    ├── README.md
    ├── Q01.md
    ├── Q02.md
    ├── ...
    └── Q20.md
```

## Workflow

```text
Question → User decision → Documentation update → Commit → Next question
```

No UI implementation is permitted while this decision phase is active.

## Source documents

- [Design System](DESIGN_SYSTEM.md)
- [Navigation](NAVIGATION.md)
- [States](STATES.md)
- [Screens](SCREENS.md)
- [UI Decision Questions](questions/README.md)
- [UI Specification overview](../UI_SPECIFICATION.md)

## Lock policy

The UI is not considered implementation-ready until all 20 decisions are answered, documented, reviewed, and explicitly approved. The resulting specification becomes the implementation contract.
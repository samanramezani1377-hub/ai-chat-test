# UI Decision Questions

**Status:** In progress  
**Total decisions:** 20  
**Current:** Q01

## Process

Each question is answered by the project owner one at a time. The accepted answer is documented and committed before the next question is asked.

```text
Q01 → answer → commit
Q02 → answer → commit
...
Q20 → answer → commit
       ↓
UI v1 final review
       ↓
UI lock
       ↓
Implementation may begin
```

## Question tree

| ID | Decision area | Status |
|---|---|---|
| Q01 | Overall visual direction | 🔵 Current |
| Q02 | App shell and navigation | ⏳ |
| Q03 | Chat screen structure | ⏳ |
| Q04 | Message bubble design | ⏳ |
| Q05 | Composer and send controls | ⏳ |
| Q06 | AI streaming experience | ⏳ |
| Q07 | Conversation history | ⏳ |
| Q08 | Workspace design | ⏳ |
| Q09 | Action/task presentation | ⏳ |
| Q10 | Approval and verification UX | ⏳ |
| Q11 | Error Center | ⏳ |
| Q12 | Loading, empty, and error states | ⏳ |
| Q13 | Color system | ⏳ |
| Q14 | Typography | ⏳ |
| Q15 | Light/dark themes | ⏳ |
| Q16 | Motion and feedback | ⏳ |
| Q17 | RTL/Persian and localization | ⏳ |
| Q18 | Accessibility and responsive layout | ⏳ |
| Q19 | Settings and diagnostics | ⏳ |
| Q20 | Final polish and UI lock criteria | ⏳ |

## Rules

- No UI implementation during this decision phase.
- Answers must be reflected in the appropriate UI document.
- Every accepted decision is committed before moving on.
- A later decision must not silently invalidate an earlier decision; conflicts require an explicit revision.
- After Q20, the complete UI specification is reviewed before implementation.
# Dependency Rules

## Allowed direction

```text
UI → Core Contracts / Presentation State
Core → Domain Contracts
Adapters → Core Contracts
Runtime implementation → Runtime Adapter Contract
Storage implementation → Repository Contract
```

## Forbidden dependencies

```text
UI → WooGit / Runtime implementation       ❌
UI → Executor implementation               ❌
Core → llama.cpp / WooGit implementation   ❌
Core → Android UI                           ❌
Workspace → Executor directly               ❌
Error Center → raw log storage directly     ❌
```

## Principle

Dependency باید به سمت abstraction باشد. برای تعویض Runtime، Adapter یا WooGit نباید Core یا UI بازطراحی شود.

## Sensitive Actions

UI فقط Approval/Command را به Core می‌دهد. UI نباید مستقیماً Execute کند. Core همان prepared operation را به Execution pipeline می‌دهد.

## Observability

همه Componentها می‌توانند Event/Error تولید کنند، اما Central Observability مالک Contract و Correlation است. Presentation فقط Query/Subscribe می‌کند.

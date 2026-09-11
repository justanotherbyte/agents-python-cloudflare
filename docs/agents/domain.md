# Domain docs

This is a single-context repository. Engineering skills consume its domain
documentation using the following conventions.

## Before exploring

Read these files when they exist:

- `CONTEXT.md` at the repository root.
- ADRs under `docs/adr/` that affect the area being changed.

Proceed silently when either location is absent. Domain-modeling skills create
them lazily when a glossary term or a qualifying architectural decision needs to
be recorded.

## File structure

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
└── agents/
```

## Use the glossary vocabulary

Use terms as defined in `CONTEXT.md` in issue titles, plans, tests, and code.
When a needed concept is missing, reconsider whether new terminology is
necessary or record the gap for domain modeling.

## Flag ADR conflicts

Surface any conflict with an existing ADR explicitly rather than silently
overriding it.

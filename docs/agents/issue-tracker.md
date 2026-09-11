# Issue tracker: GitHub

Issues and specs for this repository live in GitHub Issues. Use the `gh` CLI
for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`.
- **Read an issue**: `gh issue view <number> --comments`, including labels.
- **List issues**: use `gh issue list` with appropriate state and label filters.
- **Comment**: `gh issue comment <number> --body "..."`.
- **Apply or remove labels**: use `gh issue edit`.
- **Close**: `gh issue close <number> --comment "..."`.

Infer `justanotherbyte/agents-python` from the repository remote. GitHub shares
one number space across issues and pull requests, so resolve an ambiguous number
with `gh pr view` and then `gh issue view`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Publishing and fetching

When a skill says to publish to the issue tracker, create a GitHub issue. When a
skill says to fetch a ticket, use `gh issue view <number> --comments`.

## Wayfinding operations

The Wayfinder map is one issue labelled `wayfinder:map`; its tickets are child
issues.

- Create child tickets as GitHub sub-issues. If sub-issues are unavailable, add
  them to a task list in the map and put `Part of #<map>` in each child body.
- Label tickets `wayfinder:research`, `wayfinder:prototype`,
  `wayfinder:grilling`, or `wayfinder:task`.
- Represent blocking with native GitHub issue dependencies. If dependencies are
  unavailable, add `Blocked by: #<number>` to the child body.
- The frontier is the map's open, unblocked, unassigned child issues.
- Claim a ticket by assigning it to `@me` before starting work.
- Resolve a ticket by posting its answer, closing it, and adding a linked gist to
  the map's Decisions-so-far section.

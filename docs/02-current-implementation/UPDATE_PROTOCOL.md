# Update Protocol For Current Implementation Docs

This protocol keeps the current-implementation docs clean and easy to maintain.

## Core Rule

Update only the specific section impacted by your code change.

Do not rewrite unrelated sections.

## Mapping: Change Type -> File To Update

1. New/changed detector, localizer, or RCA behavior
- Update: `CAPABILITY_STATUS.md`
- Update: `IMPLEMENTED_ARCHITECTURE.md` (only if execution flow changed)
- Update: `COMPONENT_RATIONALE.md` (only if rationale changed)
- Update: `CODEBASE_HEADSTART.md` (if the developer reading order, module map, runtime path, or key behavior summary changes)

2. Dashboard metric/chart/data parsing behavior change
- Update: `DASHBOARD_AND_DATA_CONTRACTS.md`
- Update: `CAPABILITY_STATUS.md` (if capability scope changed)
- Update: `CODEBASE_HEADSTART.md` (if this changes how a new developer should understand runtime behavior)

3. New limitation discovered or resolved
- Update: `LIMITATIONS_AND_GAPS.md`

4. Codebase structure, major module movement, new subsystem, or changed onboarding path
- Update: `CODEBASE_HEADSTART.md`

5. Process change for documentation maintenance
- Update: this file (`UPDATE_PROTOCOL.md`)

## Section Template For Additions

Use this simple template when adding a new section:

- What changed
- Where in code this lives
- Why this design/behavior is used
- What this means for developers/operators

Keep each point short and concrete.

## Conflict Rule

If docs conflict with each other or with code behavior:

1. Resolve conflict before merge.
2. Use code behavior as source for current implementation files.
3. If end-goal conflict exists, resolve intentionally and update affected end-goal file.

## Anti-Mess Rules

- No long chronological diary logs in these files.
- No duplicate content across files.
- Use references to existing sections instead of repeating details.
- Replace obsolete text in place rather than appending stale history.

## Review Checklist (Before Merge)

1. Is every changed behavior reflected in exactly the right file(s)?
2. Did you avoid editing unrelated sections?
3. Is wording simple and direct?
4. Are removed/changed behaviors no longer described as active?
5. Are any new gaps captured in `LIMITATIONS_AND_GAPS.md`?
6. Does this code change require a `CODEBASE_HEADSTART.md` update for new-developer orientation?

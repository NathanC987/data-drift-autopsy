# Documentation Blueprint

This directory is the core developer reference for this project.

It has two tracks:

1. `01-end-goal/`
   - Defines the complete enterprise-ready product target.
   - Changes infrequently.
   - Use this to align long-term direction.

2. `02-current-implementation/`
   - Describes exactly what is implemented now and why.
   - Changes frequently as the code changes.
   - Use this as the day-to-day implementation guide.

## How To Use This Directory

1. Start with `02-current-implementation/CODEBASE_HEADSTART.md` for fast project orientation.
2. Read `02-current-implementation/README.md` to understand current state docs usage.
3. Read `01-end-goal/README.md` to understand destination state.
4. For any code change, update only the impacted section(s) in `02-current-implementation/`.
5. If a conflict appears between docs, resolve the conflict before merge.

## Rules For Contributors

- Keep language simple and direct.
- Do not add speculative claims.
- Use code-grounded facts for current implementation.
- Keep end-goal content stable and intentional.
- Avoid duplicate content across files. Link instead.

## File Map

- `01-end-goal/README.md`
- `01-end-goal/PRODUCT_VISION.md`
- `01-end-goal/TARGET_ARCHITECTURE.md`
- `01-end-goal/ENTERPRISE_REQUIREMENTS.md`
- `01-end-goal/ROADMAP.md`
- `01-end-goal/THRESHOLD_CALIBRATION_STRATEGY.md`

- `02-current-implementation/README.md`
- `02-current-implementation/CODEBASE_HEADSTART.md`
- `02-current-implementation/CAPABILITY_STATUS.md`
- `02-current-implementation/IMPLEMENTED_ARCHITECTURE.md`
- `02-current-implementation/COMPONENT_RATIONALE.md`
- `02-current-implementation/DASHBOARD_AND_DATA_CONTRACTS.md`
- `02-current-implementation/LIMITATIONS_AND_GAPS.md`
- `02-current-implementation/UPDATE_PROTOCOL.md`

# Sounding (project_brown_pants)

A KSP-inspired space-flight sandbox in Rust + Bevy, built as an academic exercise.
This is the **code** repo. The **design and planning** live in the `tickets` repo at
`/home/dave/Documents/tickets/docs/projects/sounding/`:
- `project.md` — scope and the work-item backlog
- `design.md` — conceptual architecture, rationale, the toys roadmap

## Orient here first
- [ARCHITECTURE.md](ARCHITECTURE.md) — the code map (crates, boundaries, data flow).
- [README.md](README.md) — build/run/test, controls, the bus, the companion.

## Load-bearing conventions (also in workflow `skills/rust.md`)
- **Headless core:** `crates/sim` (`sounding_sim`) depends on Bevy **sub-crates**, never the `bevy` umbrella, so it stays rendering-free. Verify: `cargo tree -p sounding_sim` shows no `bevy_render`/`bevy_winit`/`wgpu`.
- **One writer:** all simulation-state mutation goes through the `Command` executor (`apply_command`). Sources emit commands; they do not mutate state.
- **API as contract:** the `Command` envelope and `Telemetry` snapshot are a versioned public surface (humans, autopilots, the bus, a future MCP AI all depend on them). Change them as contract changes (sweep all consumers).
- **Dev tooling:** Bevy Remote Protocol is behind the app's `dev` feature; the runtime bus is separate and always-on.
- **Toolchain:** use rustup (`. "$HOME/.cargo/env"`). One patch version bump per work item.

## Workflow
Built under the workflow framework (`/home/dave/Documents/workflow`). Per-work-item
docs (plan/code/reflect) live in the tickets repo under `docs/pending/<id>-*/`.

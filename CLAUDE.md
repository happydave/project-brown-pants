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
- **Command-routed state:** every simulation-state change is `Command`-driven through the executor — sources emit commands, they never mutate state directly. Field mutations (warp, pause, maneuver) go through the pure `apply_command`; structural changes (component insert/remove — e.g. the `SetGear` gear swap) go through a dedicated `Command`-triggered system. The invariant is the routing, not one function.
- **API as contract:** the `Command` envelope and `Telemetry` snapshot are a versioned public surface (humans, autopilots, the bus, a future MCP AI all depend on them). Change them as contract changes (sweep all consumers).
- **Scale-relative physics:** per-body coefficients (angular/linear damping, spring & contact terms, inertia) must be **rates scaled by the body's mass/inertia**, never absolute constants — an absolute tuned at one build scale silently breaks at another (it has bitten twice: wheel inertia, then angular drag pinning light editor-scale rovers upright). Keep a **light editor-scale (0.1 m) fixture** in the rover/physics tests so scale regressions surface in CI, not playtest.
  - **Relationships, not just constants:** when two coupled rates must keep an ordering (e.g. tire stiffness > suspension rate), the **relationship** must be scale-relative too — a mass-sized rate crossing an *absolute* one is the bug. Tie them (`k_tire = k_tire.max(RATIO * k_susp)`) or derive both from the same mass-relative base; don't let one scale with the build while the other stays fixed (it bit a third time: WI 631b — an auto-sized suspension overtook an absolute tire rate, giving an under-damped landing wallow). Sanity-check against the real-world analogue (ride frequency, damping ratio, stiffness ratio) at both the smallest and largest build.
  - **Dissipative over deletion:** if a stability hack *removes* a force entirely (e.g. fading traction when inverted), prefer a **dissipative-only** version instead — a friction opposing the contact velocity can only remove energy, so it gives the safety (no pumping) without the side effect (a frictionless wreck sliding forever). WI 631b inverted-slide fix.
- **Dev tooling:** Bevy Remote Protocol is behind the app's `dev` feature; the runtime bus is separate and always-on.
- **Toolchain:** use rustup (`. "$HOME/.cargo/env"`). One patch version bump per work item.

## Workflow
Built under the workflow framework (`/home/dave/Documents/workflow`). Per-work-item
docs (plan/code/reflect) live in the tickets repo under `docs/pending/<id>-*/`.

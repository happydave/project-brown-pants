## Version Context

- Rust: stable via rustup (developed on 1.96), edition 2021.
- Bevy: 0.18.1. The simulation core uses Bevy **sub-crates** only; the app uses the `bevy` umbrella.
- Key deps: `glam` 0.30 (f64 math, `serde` feature), `bevy_app`/`bevy_ecs`/`bevy_time`/`bevy_diagnostic`/`bevy_log` 0.18, `serde` 1, `serde_json` 1, `tiny_http` 0.12 (app bus), `ureq` 3 (companion).
- Workspace: virtual manifest at `/Cargo.toml`; members `/crates/sim`, `/crates/app`, `/crates/companion`. One `[workspace.package]` version, bumped one patch per work item.

## Architectural Boundaries

Required: `/crates/sim/` (`sounding_sim`) depends only on Bevy sub-crates (`bevy_app`, `bevy_ecs`, `bevy_time`, `bevy_diagnostic`, `bevy_log`), never the `bevy` umbrella — so it is rendering-free and runs headless.
Forbidden: `/crates/sim/` pulling in rendering or windowing. Verify: `cargo tree -p sounding_sim` shows no `bevy_render`, `bevy_winit`, or `wgpu`.
Required: all mutation of simulation state routes through `apply_command` in `/crates/sim/src/command.rs`. No input system, autopilot, or bus client mutates `SimClock` or `Craft` directly — they emit `Command` messages.
Required: external processes (`/crates/companion/`, future second screen / MCP AI) act on the simulation only via the runtime bus (`GET /telemetry`, `POST /command`). The bus injects `Command`s; it never mutates state.
Required: the Bevy Remote Protocol is compiled only under the app's `dev` feature, absent from default/release builds, and is distinct from the runtime bus.

## Component Index

- **sounding_sim** `/crates/sim/` — the headless simulation core (library).
  - `/crates/sim/src/orbit.rs` — `Orbit`: analytic 2D Kepler propagation. Inputs: μ, state vectors, time. Outputs: position/velocity, derived orbit, maneuver result.
  - `/crates/sim/src/sim.rs` — `SimClock` (time + warp + paused), `CentralBody`, `Craft`, `OrbitPlugin`. Inputs: real frame time. Outputs: advancing simulated time; the spawned craft entity.
  - `/crates/sim/src/command.rs` — `Command` (serde message envelope), `apply_command`, `FlightControlPlugin`. Inputs: `Command` messages. Outputs: the only writes to `SimClock`/`Craft`.
  - `/crates/sim/src/diagnostics.rs` — `SimDiagnosticsPlugin`, conservation-drift metrics. Inputs: clock + craft. Outputs: Bevy diagnostics (`sim/energy_drift`, `sim/angular_momentum_drift`).
  - `/crates/sim/src/telemetry.rs` — `Telemetry`/`CraftTelemetry` serde snapshot + `capture`. Inputs: clock, orbit, μ, drift. Outputs: serializable snapshot.
- **sounding** `/crates/app/` — the windowed application (binary `sounding`).
  - `/crates/app/src/main.rs` — `DefaultPlugins` + sim plugins; 2D gizmo renderer; keyboard input → `Command` messages. Inputs: keyboard. Outputs: rendered window; `Command` messages.
  - `/crates/app/src/bus.rs` — `BusPlugin`: a `tiny_http` server (default `127.0.0.1:8787`) on its own thread, bridged by channels. Inputs: HTTP GET/POST. Outputs: telemetry JSON; `Command` messages into the executor.
- **companion** `/crates/companion/` — external agent (binary `companion`).
  - `/crates/companion/src/brain.rs` — `Brain` trait, `NavigatorBrain` (circularize policy). Inputs: `Telemetry`. Outputs: `Decision` (idle or a `Command`).
  - `/crates/companion/src/main.rs` — bus-client loop (`ureq`). Inputs: `GET /telemetry`. Outputs: `POST /command`.

## Dependency Chains

- `/crates/app/` depends on `/crates/sim/` + the `bevy` umbrella + `tiny_http` + `serde_json`.
- `/crates/companion/` depends on `/crates/sim/` (shared `Command`/`Telemetry` types) + `ureq` + `serde_json` + `glam`. No Bevy.
- `/crates/sim/` depends on Bevy sub-crates + `glam` + `serde`. No rendering; no `bevy` umbrella.
- `Command` and `Telemetry` (`/crates/sim/`) are the shared contract every other crate and the bus depend on — treat changes to them as public-contract changes.

## Linear Data Flow

Command lifecycle (the one to know):
1. A source emits a `Command` — keyboard (`/crates/app/src/main.rs`), the companion via `POST /command` (`/crates/app/src/bus.rs` → `drain_commands`), or a future MCP/tablet client.
2. The `Command` becomes a Bevy message.
3. `apply_command` (`/crates/sim/src/command.rs`, in `Update`) drains messages and applies them to `SimClock`/`Craft` — the sole writer of simulation state. Maneuvers apply at the current `clock.time` (impulsive, position-continuous).
4. `OrbitPlugin::advance_clock` advances simulated time by real-frame-time × warp.
5. The renderer (`/crates/app/src/main.rs`) and `publish_telemetry` (`/crates/app/src/bus.rs`) read the craft's `Orbit` at `clock.time`.
6. `publish_telemetry` writes a `Telemetry` JSON snapshot to a shared buffer each frame; the `tiny_http` thread serves it on `GET /telemetry`.

## Known Gaps (not yet built)

- Only the on-rails (analytic) gear exists. The active/numerical gear, the warp-gearbox hand-off, and floating origin are not built (see the design's Validation Roadmap, Toys 4–9).
- The hand-off-discontinuity and contact-jitter diagnostics are documented placeholders awaiting their systems.
- Conceptual architecture, rationale, and the roadmap live in the **design doc**, not here: `/home/dave/Documents/tickets/docs/projects/sounding/design.md` (scope + backlog in `project.md` alongside it).

## Version Context

- Rust: stable via rustup (developed on 1.96), edition 2021.
- Bevy: 0.18.1. The simulation core uses Bevy **sub-crates** only; the app uses the `bevy` umbrella.
- Key deps: `glam` 0.30 (f64 math — `DVec2` orbits, `DVec3` world coordinates; `serde` feature), `bevy_app`/`bevy_ecs`/`bevy_time`/`bevy_diagnostic`/`bevy_log` 0.18, `serde` 1, `serde_json` 1, `tiny_http` 0.12 (app bus), `ureq` 3 (companion).
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
  - `/crates/sim/src/frame.rs` — `FrameId` (body-centered inertial frame identity), `WorldPos`/`WorldVel` (f64 `DVec3`, frame-tagged). `WorldPos::transform_to` is identity for the single frame today and is the documented seam for multi-body SOI transforms (WI 508) and floating-origin rebasing (WI 504). Foundational coordinate substrate (WI 497).
  - `/crates/sim/src/fluid.rs` — `FluidMedium` (data-driven atmosphere+ocean profile), `FluidSample`, `MediumKind`. A field sampled by signed altitude (`sample_altitude`) or world position (`sample_at`): one shape, different constants for vacuum/air/ocean. Canonical instances `VACUUM`, `EARTHLIKE`. Consumed later by aero/hydro and flooding (WI 497).
  - `/crates/sim/src/surface.rs` — `SurfaceMaterial` (friction + rolling-resistance coefficients) with `REGOLITH`/`BEDROCK`/`ICE`. The ground-contact parallel to the fluid field; consumed later by the wheel/contact model (WI 506). Foundational (WI 497).
  - `/crates/sim/src/persist.rs` — the **durable, versioned** serialization format (WI 498), distinct from the ephemeral bus. `SavedDocument` (envelope: `format_version` + `Payload`), internally-tagged `Payload` (craft/subassembly/blueprint share `CraftSubgraph`; world-save → `WorldPayload`), `Kind`, `FORMAT_VERSION` (= 1, decoupled from the crate version), typed `FormatError`. `from_json` does a two-stage parse (a version-stable `VersionProbe` reads the version before the payload, so a newer file is rejected by version) and carries the migration seam. Inputs: JSON. Outputs: `SavedDocument` or a typed error. Payloads are skeletal at v1; later toys fill them.
- **sounding** `/crates/app/` — the windowed application (binary `sounding`).
  - `/crates/app/src/main.rs` — the **Toy 4** 3D scene (WI 504): a planetary-scale sphere, a small craft, a sun, and an HDR `Camera3d` with Bevy's physically-based `Atmosphere`; a free-fly camera (fly surface↔orbit) and a sun that sweeps the terminator. The sim/bus plugins (Toys 1–3) stay registered and run headless behind it (the 2D gizmo view was retired). Inputs: keyboard. Outputs: rendered window.
  - `/crates/app/src/floating_origin.rs` — floating-origin rebasing (WI 504). Pure `render_translation(world_f64, anchor) -> Vec3` (precision core), `WorldPlacement(WorldPos)` component, `FloatingOrigin` anchor resource, `AnchorCamera` marker, `FloatingOriginPlugin`. The anchor tracks the camera's X/Z (Y stays 0 = sea level), so the camera is pinned to the render origin and the world is translated around it in f32 without precision loss. Consumes WI 497's `WorldPos`.
  - `/crates/app/src/bus.rs` — `BusPlugin`: a `tiny_http` server (default `127.0.0.1:8787`) on its own thread, bridged by channels. Inputs: HTTP GET/POST. Outputs: telemetry JSON; `Command` messages into the executor.
- **companion** `/crates/companion/` — external agent (binary `companion`).
  - `/crates/companion/src/brain.rs` — `Brain` trait, `NavigatorBrain` (circularize policy). Inputs: `Telemetry`. Outputs: `Decision` (idle or a `Command`).
  - `/crates/companion/src/main.rs` — bus-client loop (`ureq`). Inputs: `GET /telemetry`. Outputs: `POST /command`.

## Dependency Chains

- `/crates/app/` depends on `/crates/sim/` + the `bevy` umbrella + `tiny_http` + `serde_json`.
- `/crates/companion/` depends on `/crates/sim/` (shared `Command`/`Telemetry` types) + `ureq` + `serde_json` + `glam`. No Bevy.
- `/crates/sim/` depends on Bevy sub-crates + `glam` + `serde` + `serde_json` (the latter for the WI 498 persistence format, in library code now — not just tests). No rendering; no `bevy` umbrella.
- `Command` and `Telemetry` (`/crates/sim/`) are the shared contract every other crate and the bus depend on — treat changes to them as public-contract changes. The `persist` format (`SavedDocument` + `FORMAT_VERSION`) is a second, **durable** versioned contract: it has its own `FORMAT_VERSION` line (independent of the crate version) and must stay backward-loadable via the two-stage `from_json` migration seam. Do not conflate it with the ephemeral bus.

## Linear Data Flow

Command lifecycle (the one to know):
1. A source emits a `Command` — keyboard (`/crates/app/src/main.rs`), the companion via `POST /command` (`/crates/app/src/bus.rs` → `drain_commands`), or a future MCP/tablet client.
2. The `Command` becomes a Bevy message.
3. `apply_command` (`/crates/sim/src/command.rs`, in `Update`) drains messages and applies them to `SimClock`/`Craft` — the sole writer of simulation state. Maneuvers apply at the current `clock.time` (impulsive, position-continuous).
4. `OrbitPlugin::advance_clock` advances simulated time by real-frame-time × warp.
5. `publish_telemetry` (`/crates/app/src/bus.rs`) reads the craft's `Orbit` at `clock.time` and writes a `Telemetry` JSON snapshot to a shared buffer each frame; the `tiny_http` thread serves it on `GET /telemetry`. (The on-rails orbit now runs headless behind the 3D scene; it is not the rendered craft.)

Render placement (Toy 4): entities carry an f64 `WorldPlacement`; each frame (`PostUpdate`, before transform propagation) `track_anchor` sets the `FloatingOrigin` to the camera's X/Z and `rebase` derives every entity's f32 `Transform.translation` as `world − anchor`. The camera stays at the render origin in X/Z with its true altitude in Y, which Bevy's atmosphere consumes directly.

## Known Gaps (not yet built)

- Only the on-rails (analytic) gear exists. The active/numerical gear and the warp-gearbox hand-off are not built (see the design's Validation Roadmap, Toys 4–9).
- The f64 3D world-coordinate + reference-frame substrate and the data-driven fluid/surface field types exist (WI 497, `frame.rs`/`fluid.rs`/`surface.rs`). Floating-origin rendering now consumes `WorldPos` (WI 504, `app/floating_origin.rs`), but the field *consumers* (aero/hydro, wheels) and the 2D-orbit→3D-world bridge are still not built. The orbit propagator remains 2D and runs headless behind the 3D scene; the rendered planet/craft are a static placement, not orbit-driven.
- The Toy 4 scene renders a single static planet sphere with no terrain LOD, collision, or surface micro-precision (Toy 6), and a placeholder craft mesh (Toy 5). Depth uses Bevy's default reverse-Z (no logarithmic depth added).
- The versioned persistence format exists (WI 498, `persist.rs`) but its payloads are skeletal: the craft-subgraph contents (lattice/devices/resources/crew) and the world-save payload are reserved empty containers, filled by later toys. There is no save/load-to-disk wiring, no UI, and no migration engine yet (nothing to migrate at format version 1).
- The hand-off-discontinuity and contact-jitter diagnostics are documented placeholders awaiting their systems.
- Conceptual architecture, rationale, and the roadmap live in the **design doc**, not here: `/home/dave/Documents/tickets/docs/projects/sounding/design.md` (scope + backlog in `project.md` alongside it).

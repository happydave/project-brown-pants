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
  - `/crates/sim/src/orbit.rs` — `Orbit`: analytic 2D Kepler propagation (the **on-rails** gear). Inputs: μ, state vectors, time. Outputs: position/velocity, derived orbit, maneuver result.
  - `/crates/sim/src/active.rs` — the **active** gear (WI 515): `ActiveBody` numerically integrated under point-mass gravity (velocity Verlet, symplectic) at a fixed `dt`, with torque-free rotation driven by stored world-frame angular momentum (consumes the WI 505 inertia tensor). `advance` caps sub-steps per frame (the active-vehicle warp cap); `ActivePlugin`/`Gravity` drive it from `SimClock`. `integrate_wrench` (semi-implicit Euler under an external force + torque) is the integrator for dissipative/state-dependent actuation — its first consumer is the rover (WI 506). Pure energy/angular-momentum drift functions (WI 499 style). Standalone — the on-rails↔active hand-off is WI 508. Headless.
  - `/crates/sim/src/sim.rs` — `SimClock` (time + warp + paused), `CentralBody`, `Craft`, `OrbitPlugin`. Inputs: real frame time. Outputs: advancing simulated time; the spawned craft entity.
  - `/crates/sim/src/command.rs` — `Command` (serde message envelope), `apply_command`, `FlightControlPlugin`. Inputs: `Command` messages. Outputs: the only writes to `SimClock`/`Craft`.
  - `/crates/sim/src/diagnostics.rs` — `SimDiagnosticsPlugin`, conservation-drift metrics. Inputs: clock + craft. Outputs: Bevy diagnostics (`sim/energy_drift`, `sim/angular_momentum_drift`).
  - `/crates/sim/src/telemetry.rs` — `Telemetry`/`CraftTelemetry` serde snapshot + `capture`. Inputs: clock, orbit, μ, drift. Outputs: serializable snapshot.
  - `/crates/sim/src/frame.rs` — `FrameId` (body-centered inertial frame identity), `WorldPos`/`WorldVel` (f64 `DVec3`, frame-tagged). `WorldPos::transform_to` is identity for the single frame today and is the documented seam for multi-body SOI transforms (WI 508) and floating-origin rebasing (WI 504). Foundational coordinate substrate (WI 497).
  - `/crates/sim/src/fluid.rs` — `FluidMedium` (data-driven atmosphere+ocean profile), `FluidSample`, `MediumKind`. A field sampled by signed altitude (`sample_altitude`) or world position (`sample_at`): one shape, different constants for vacuum/air/ocean. Canonical instances `VACUUM`, `EARTHLIKE`. Consumed later by aero/hydro and flooding (WI 497).
  - `/crates/sim/src/surface.rs` — `SurfaceMaterial` (friction + rolling-resistance coefficients) with `REGOLITH`/`BEDROCK`/`ICE`. The ground-contact parallel to the fluid field; consumed by the wheel/contact model (WI 506). Foundational (WI 497).
  - `/crates/sim/src/terrain.rs` — analytic procedural `Terrain` (WI 506): deterministic f64 height/normal/material over the local tangent plane. **This function is the collision surface** — the wheels query it in f64, never the rendered mesh, so contact cannot pop under LOD/rebasing (the kraken fix). Headless.
  - `/crates/sim/src/rover.rs` — wheels & ground contact (WI 506), the one genuinely new primitive (frictional contact). `Rover` = an `ActiveBody` + `Wheel`s; `step` probes the analytic terrain per wheel → spring-damper normal force + slip-based tyre (friction ellipse, material-scaled `μ`) + wheel spin DOF → net wrench → `integrate_wrench`, sub-stepped at `SUBSTEP_DT` (≈ 1/1920 s — stiff contact needs it). Drag (top speed), angular drag (anti-tumble), and a motor speed limit keep it stable and controllable. `contact_jitter` is the kraken detector. Mass/inertia come from the voxel lattice (WI 505). Headless, validated by no-launch/no-tumble/no-spin tests.
  - `/crates/sim/src/voxel.rs` — the **single craft representation** (WI 505): `VoxelCraft` (sparse `Voxel` lattice + `Device`s + `AttachmentPoint`s + cell size) and the derivations from it — `mass_properties` (mass, centre of mass, inertia tensor about the CoM via solid-cube + parallel-axis, principal moments/axes via a Jacobi eigensolve) and `area_curve` (voxel-occupancy slices: the aero cross-section input). `Material` is data-driven (density). Pure, headless, unit-tested. Validates two of the voxel model's four roles (mass/inertia and aero cross-section); breakage and compartments come later.
  - `/crates/sim/src/persist.rs` — the **durable, versioned** serialization format (WI 498), distinct from the ephemeral bus. `SavedDocument` (envelope: `format_version` + `Payload`), internally-tagged `Payload` (craft/subassembly/blueprint share `CraftSubgraph`; world-save → `WorldPayload`), `Kind`, `FORMAT_VERSION` (= 1, decoupled from the crate version), typed `FormatError`. `from_json` does a two-stage parse (a version-stable `VersionProbe` reads the version before the payload, so a newer file is rejected by version) and carries the migration seam. `CraftSubgraph` now embeds a real `VoxelCraft` (WI 505, evolved in place at v1); `resources`/`crew` remain reserved opaque containers.
- **sounding** `/crates/app/` — the windowed application (binary `sounding`).
  - `/crates/app/src/main.rs` — a **scene dispatcher** (WI 514): selects a toy scene at launch (`EditorPlugin` by default, `PlanetPlugin` with `planet`, `RoverScenePlugin` with `rover`) and registers the common Toy 1–3 sim/bus plugins, which run headless behind whichever scene is shown. Inputs: keyboard + launch arg. Outputs: rendered window.
  - `/crates/app/src/editor.rs` — the **Toy 5** voxel editor (WI 505): `EditorState` (craft + cursor + material + subassembly buffer) and `EditorPlugin` (self-contained: its own orbit camera + light). Cursor-based add/remove of voxels and devices, material palette, blueprint/subassembly save·load·insert via the WI 498 format, and immediate-mode gizmo visualization (voxel cubes by material, cursor, CoM marker, principal-inertia axes, area-curve plot). Consumes `sounding_sim::voxel`.
  - `/crates/app/src/planet.rs` — the **Toy 4** floating-origin planet scene (WI 504) as `PlanetPlugin` (WI 514): planetary sphere + small craft + sun + HDR camera with Bevy's `Atmosphere`; a free-fly camera (surface↔orbit) and a terminator-sweeping sun. Adds `FloatingOriginPlugin`.
  - `/crates/app/src/rover_scene.rs` — the **Toy 6** rover scene (WI 506) as `RoverScenePlugin`: drives the headless `sounding_sim::rover` over the analytic terrain with rover-anchored floating-origin gizmo rendering (terrain grid, chassis, wheels, wheel-track trail), driving input, a chase camera, and a speed/height HUD. Sub-steps the rover to `SUBSTEP_DT` each frame.
  - `/crates/app/src/floating_origin.rs` — floating-origin rebasing (WI 504), used by `planet.rs`. Pure `render_translation(world_f64, anchor) -> Vec3` plus `WorldPlacement`/`FloatingOrigin`/`AnchorCamera`/`FloatingOriginPlugin`. The anchor tracks the camera's X/Z (Y stays sea level) so the camera is pinned to the render origin and the world moves around it in f32.
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

Craft derivation (Toy 5): a `VoxelCraft` (in `EditorState`, or loaded from a `SavedDocument`) is the single input; `mass_properties` and `area_curve` (`sim/voxel.rs`) derive mass/CoM/inertia and the aero cross-section from the same voxels. The editor draws them as gizmos each frame; nothing is precomputed or duplicated.

Render placement (Toy 4, retained): entities carry an f64 `WorldPlacement`; `track_anchor` sets the `FloatingOrigin` to the camera's X/Z and `rebase` derives f32 `Transform.translation` as `world − anchor`, pinning the camera to the render origin in X/Z with altitude in Y (the atmosphere consumes that directly).

Wheel contact (Toy 6): `Rover::step` queries the **analytic** `Terrain` (f64) per wheel — never the mesh — computes suspension + tyre + drag forces into a net wrench, and integrates the `ActiveBody` via `integrate_wrench`, sub-stepped at `SUBSTEP_DT`. The rover scene anchors floating-origin rendering at the rover (it sits at a large f64 world offset) and sub-steps the rover each frame.

## Known Gaps (not yet built)

- Both gearbox gears exist: the on-rails analytic `Orbit` and the active numerical `ActiveBody` (WI 515), the latter now driven with applied forces by the rover (WI 506). The **hand-off** between gears — putting an active craft onto a conic and waking a rails craft into active physics — is **not** built (WI 508), and nothing switches a craft between gears yet. Units are normalised (μ = 1); metric reconciliation is deferred.
- The f64 3D world-coordinate + reference-frame substrate and the data-driven fluid/surface field types exist (WI 497, `frame.rs`/`fluid.rs`/`surface.rs`). Floating-origin rendering now consumes `WorldPos` (WI 504, `app/floating_origin.rs`), but the field *consumers* (aero/hydro, wheels) and the 2D-orbit→3D-world bridge are still not built. The orbit propagator remains 2D and runs headless behind the 3D scene; the rendered planet/craft are a static placement, not orbit-driven.
- The Toy 4 planet scene (retained) renders a single static sphere with no terrain LOD. Depth uses Bevy's default reverse-Z.
- The rover (WI 506) drives on a **local tangent heightfield** with constant gravity; spherical-planet driving, a full Pacejka tyre, and explicit multi-LOD terrain are the quarantined WI 516 spike. Contact is LOD-independent by construction (analytic surface), so single-resolution origin-local tiles suffice. Wheels are point-contact (one patch each); tracks are a transient gizmo trail, not persistent terrain deformation.
- The voxel craft (WI 505, `voxel.rs`) derives mass/inertia and the aero cross-section, but its other two roles — connected-component **breakage** and airtight **compartments** — are not built; **devices are inert mass** (no engine/tank behavior); and the area curve is *produced* here but *consumed* (drag/lift/area-ruling) only at Toy 9. The editor saves blueprints/subassemblies to JSON files in the working directory.
- The `CraftSubgraph` payload now carries real voxels/devices (WI 505); `resources`/`crew` and the world-save payload remain reserved. The format stayed at version 1 (evolved in place, pre-release); there is still no migration *engine* and nothing to migrate, but the two-stage `from_json` seam is intact for the first post-release change.
- The contact-jitter detector exists in the rover (`Rover::contact_jitter`, WI 506); the hand-off-discontinuity diagnostic remains a documented placeholder awaiting WI 508. Wiring the active gear's live drift diagnostics into a scene also awaits WI 508.
- Conceptual architecture, rationale, and the roadmap live in the **design doc**, not here: `/home/dave/Documents/tickets/docs/projects/sounding/design.md` (scope + backlog in `project.md` alongside it).

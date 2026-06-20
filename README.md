# Sounding (project_brown_pants)

A KSP-inspired space-flight sandbox, built as an academic exercise in Rust + the
Bevy engine. This repository is the implementation; the architecture and design
live in the project's `tickets` repository under `docs/projects/sounding/`.

> Status: bootstrap (WI 496). A two-crate workspace skeleton — a headless
> simulation core and a windowed application — with dev-only remote tooling.

## Workspace layout

- [crates/sim/](crates/sim/) — `sounding_sim`, the **headless, rendering-free**
  simulation core. Depends on Bevy *sub-crates* (`bevy_app`, `bevy_ecs`), never
  the `bevy` umbrella, so it builds and runs with no display, GPU, or windowing
  libraries. The simulation logic lives here.
- [crates/app/](crates/app/) — `sounding`, the windowed Bevy application. Pulls in
  rendering (the `bevy` umbrella) and wraps the core.

## Prerequisites

- A modern Rust toolchain via [rustup](https://rustup.rs) (stable; developed on
  1.96). A distro `rustc` may be too old for Bevy 0.18.
- For the **windowed app** on Linux, Bevy's system libraries — at minimum
  `libwayland-dev` and `libxkbcommon-dev`; audio/gamepad support may also need
  `libasound2-dev` and `libudev-dev`. The headless core needs none of these.

## Build, run, test

- Build everything: `cargo build`
- Run the windowed app: `cargo run -p sounding`
- Run with dev tooling (Bevy Remote Protocol over HTTP): `cargo run -p sounding --features dev`
- Test the headless core (no display required): `cargo test -p sounding_sim`
- Quality gates: `cargo fmt --all --check` and `cargo clippy --all-targets`

## Toy 5 — voxel ship editor

`cargo run -p sounding` opens the voxel editor: build a craft from voxels and
devices and watch its centre of mass, principal inertia axes, and aero
cross-sectional-area curve update live (all derived from the same voxels). A
craft saves and loads as a blueprint or a reusable subassembly through the
versioned serialization format. Editor controls:

- Arrow keys / `PageUp`·`PageDown` — move the build cursor (X/Z, then Y)
- `Space` add a voxel · `Backspace` remove voxel/device · `Tab` cycle material
- `G` place a device · `M` log mass properties
- `B` save blueprint · `N` save subassembly · `L` load subassembly · `V` insert it at the cursor

Camera: `Q`/`E` orbit · `R`/`F` pitch · `Z`/`C` zoom. The magenta marker is the
centre of mass, the RGB lines are the principal inertia axes, and the cyan plot
is the cross-section curve. Earlier toys keep running headless: the on-rails
orbit (Toy 1) and the runtime bus (Toys 2–3).

The app selects a toy scene at launch:

- `cargo run -p sounding` — the Toy 5 voxel editor (default)
- `cargo run -p sounding -- planet` — the Toy 4 floating-origin planet + atmosphere
- `cargo run -p sounding -- rover` — the Toy 6 rover on terrain (`W`/`S` drive, `A`/`D` steer, `Space` brake; HUD shows speed/height)
- `cargo run -p sounding -- dive` — the Toy 9 dive: a craft falls vacuum→atmosphere→ocean, drag/buoyancy/pressure all from one fluid field (HUD shows altitude/speed/medium, static hull pressure, and dynamic ram pressure / max-Q)
- `cargo run -p sounding -- break` — structural breakage: a voxel bar spins up until the centripetal load snaps it into connected-component fragments that tumble apart
- `cargo run -p sounding -- compartments` — airtight compartments: a hollow craft's sealed volumes, colour-coded; `H` toggles a hatch (merge/split), `B` breaches the hull (vent)
- `cargo run -p sounding -- flooding` — decompression/flooding: a submerged craft; `B` breaches a compartment and it floods, tilts, and sinks as floodwater mass shifts the centre of mass

## Runtime bus

While the app runs, a runtime state/command bus listens on
`http://127.0.0.1:8787` (a synchronous HTTP server on its own thread). It is the
shared substrate later consumers — the AI companion, second screen, multiplayer
sync — adapt onto. It is distinct from the dev-only Bevy Remote Protocol.

- `GET /telemetry` — current simulation snapshot as JSON (time, warp, paused,
  craft orbit and position, energy-drift metric).
- `POST /command` — inject a JSON command into the flight-control executor;
  malformed input returns HTTP 400.

```bash
curl -s localhost:8787/telemetry
curl -s -X POST localhost:8787/command -d '{"SetWarp":8.0}'
curl -s -X POST localhost:8787/command -d '{"SetPaused":true}'
curl -s -X POST localhost:8787/command \
  -d '{"ExecuteManeuver":{"delta_v":[0.0,0.1]}}'
```

## AI companion

`cargo run -p companion` starts an external agent that flies the craft through
the bus alone — it only reads `GET /telemetry` and issues `POST /command`,
reasoning purely from exposed telemetry (no privileged access). The shipped
deterministic navigator circularizes the orbit (coast to apoapsis, then a
prograde burn), narrating as it goes. The decision logic sits behind a `Brain`
trait, so an LLM-backed brain can replace it without changing the bus loop.

Run `cargo run -p sounding` first, then `cargo run -p companion` in a second
terminal. The app starts on a mildly eccentric orbit, so the navigator has a
visible circularization to perform; watch its narration in the companion's
terminal (the orbit runs headless behind the 3D scene).

## Notes

- **Headless invariant.** The core's freedom from rendering is verifiable:
  `cargo tree -p sounding_sim` shows no `bevy_render`, `bevy_winit`, or `wgpu`.
  Keep core-side crates on Bevy sub-crates so workspace feature unification does
  not drag rendering in.
- **Dev tooling is opt-in.** The Bevy Remote Protocol is compiled only under the
  `dev` feature, never in default or release builds.

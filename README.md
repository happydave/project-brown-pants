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

## Notes

- **Headless invariant.** The core's freedom from rendering is verifiable:
  `cargo tree -p sounding_sim` shows no `bevy_render`, `bevy_winit`, or `wgpu`.
  Keep core-side crates on Bevy sub-crates so workspace feature unification does
  not drag rendering in.
- **Dev tooling is opt-in.** The Bevy Remote Protocol is compiled only under the
  `dev` feature, never in default or release builds.

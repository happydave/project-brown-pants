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
- Dev-build performance: the dev profile optimizes dependencies and the sim crate (root `Cargo.toml`
  `[profile.dev...]`, WI 778) so a debug `cargo run` runs the engine + procedural surface at a playable
  frame rate. The **first** build after a clean/checkout is slower (deps compile optimized once), then
  cached; release builds are unaffected.

## Toy 5 — voxel ship editor

`cargo run -p sounding` opens the voxel editor: build a craft from voxels and
devices and watch its centre of mass, principal inertia axes, and aero
cross-sectional-area curve update live (all derived from the same voxels). A
craft saves and loads as a blueprint or a reusable subassembly through the
versioned serialization format. Editor controls:

- Arrow keys / `PageUp`·`PageDown` — move the build cursor (X/Z, then Y)
- `Space` add a voxel · `Backspace` remove voxel/device · `Tab` cycle material
- `G` place a device · `M` log mass properties
- `K` save vehicle · `O` load vehicle — a **named library** of many builds under
  `saves/crafts/` (WI 675): `K` opens a name prompt, `O` a load browser (↑/↓/Enter)
- `B` save blueprint · `N` save subassembly · `L` load subassembly · `V` insert it at the cursor

Camera: `Q`/`E` orbit · `R`/`F` pitch · `Z`/`C` zoom. The magenta marker is the
centre of mass, the RGB lines are the principal inertia axes, and the cyan plot
is the cross-section curve. Earlier toys keep running headless: the on-rails
orbit (Toy 1) and the runtime bus (Toys 2–3).

The app selects a toy scene at launch:

- `cargo run -p sounding` — the Toy 5 voxel editor (default)
- `cargo run -p sounding -- planet` — the Toy 4 floating-origin planet + atmosphere
- `cargo run -p sounding -- bodies` — the **body generator / keep loop** (WI 762): generate a celestial body from a seed + archetype (`Space` next seed, `Tab` Moon / Rocky Planet / Ocean World), shown as a coarse medium-tinted sphere with an atmosphere shell; `K` **keeps** it to `saves/bodies` (the reusable BodyAsset library). Middle-drag orbit, scroll zoom. Coarse render is a placeholder until the procedural surface (WI 763/764)
- `cargo run -p sounding -- surface [seed] [archetype]` — the **procedural surface renderer** (WI 764): fly a generated body from orbit down to its real cratered surface, tessellated live on a **spherified-cube quadtree LOD** over WI 763's analytic field (chunks meshed off-thread on the async compute pool, uploaded under a per-frame budget, **crack-free** via skirts). `W/A/S/D` move, `R/F` up/down, arrows look; `F3` toggles a debug overlay (per-LOD chunk wireframe + the nadir contact-patch marker for WI 765), `F4` a telemetry box (FPS graph + streaming stats). Per-body atmosphere is **data-driven** (airless bodies render without one). LOD transitions are hole-free (coverage-gated chunk despawn, WI 771). Render only — physics contact is `-- moon`
- `cargo run -p sounding -- moon` — **land and drive on a generated cratered moon** (WI 765, the world-building acceptance milestone): a rover drives on the **analytic** procedural surface field via the contact rebind (`W`/`S` throttle, `A`/`D` steer, `Space` brake, `P` pause), while the WI 764 streamer renders that same surface textured around it. Physics queries the field, never the mesh — so the rover can't fall through ungenerated ground and contact is stable under LOD/rebasing. Validated headless by a per-axis kraken test on the light editor-scale fixture
- `cargo run -p sounding -- rover` — the Toy 6 rover on terrain (`W`/`S` drive, `A`/`D` steer, `Space` brake, `P` pause, `.` step, `G` cockpit overlay; HUD shows speed/height; the overlay boxes live signal sparklines — contact_jitter/speed/ang.vel/hull_pen — WI 645/646). Built on the **same drivetrain as the workshop rover** (WI 641): off-road tyre/suspension wheel stations, quarter-car wheel hop, speed-eased steering — a fixed sandbox rover (no build UI) for exercising and inspecting the current rover physics
- `cargo run -p sounding -- dive` — the Toy 9 dive, the **full live chain in SI** (WI 527): one craft starts on a Kepler orbit, coasts down under time warp, **auto-drops** to active physics at the atmospheric entry interface, then **glides** (lift + transonic wave drag, weathervaning to trim — WI 526) vacuum→atmosphere→ocean to splashdown — drag/buoyancy/pressure all from one fluid field, and it enters at genuine **orbital speed** (~7 km/s, WI 693) and **heats on re-entry** (WI 691): a two-node thermal model glows it red→white-hot, and its **ablative heat-shield nose** (WI 688) ablates to survive — a draining `shield: NN %` on the HUD — while the composite hull survives (HUD shows gear, altitude/speed/medium, static hull pressure, dynamic ram pressure / max-Q, and **skin temperature** + an OVERHEAT flag)
- `cargo run -p sounding -- harbor` — **build-a-boat sandbox** (WI 706+): a **Build ↔ Float** loop (toggle `Enter`) that makes the buoyancy / compartment / flooding physics buildable. **Build** a hull on the shared editor — mouse place/remove, the left-edge **clickable palette** (WI 738), `Tab` material, **`T` panel mode** (thin, light plates for floating hulls, WI 716) — with a **live float/sink predictor** (WI 720). **Float** it on calm water at its **real material mass** (WI 717): a panel hull floats upright and **self-rights** (WI 705 righting buoyancy + WI 711 enclosed-air), a solid one **sinks** to the sea floor. A floated hull **sails and steers** (`W`/`S` throttle, `A`/`D` rudder — WI 708/725), **dives / surfaces** on ballast (`F` flood / `G` blow — WI 709), **breaches and floods** (`X` — WI 718/520), or **swamps** over the rim if open-topped (WI 713); interior water renders through one region-driven renderer (WI 729). Middle-drag orbit + wheel zoom; HUD shows draft / heel / net-buoyancy / thrust / ballast
- `cargo run -p sounding -- break` — structural breakage: a voxel bar spins up until the centripetal load snaps it into connected-component fragments that tumble apart
- `cargo run -p sounding -- compartments` — airtight compartments: a hollow craft's sealed volumes, colour-coded; `H` toggles a hatch (merge/split), `B` breaches the hull (vent)
- `cargo run -p sounding -- flooding` — decompression/flooding: a submerged craft; `B` breaches a compartment and it floods, tilts, and sinks as floodwater mass shifts the centre of mass
- `cargo run -p sounding -- windtunnel` — aero: live lift curve (Cl vs angle of attack) and wave-drag curve (Cd vs speed); `M` cycles the medium so the transonic spike appears in air and vanishes in water/vacuum
- `cargo run -p sounding -- launch` — surface lift-off (first-playable): a rocket rests on the pad, then auto-throttles up and ascends under thrust against gravity and drag (WI 531 propulsion + WI 532 launch-pad rest)
- `cargo run -p sounding -- autopilot` — a continuous one-craft session flown automatically (first-playable shell): Launch → Flight → Recovery (a sounding) on the unified flight pipeline; HUD shows phase, throttle, G-force, altitude/speed, and tilt, with an attitude gizmo (WI 534)
- `cargo run -p sounding -- play` — fly a craft by hand (WI 535): Shift/Ctrl throttle · Z/X full/cut · WSAD/QE attitude · T hold / R kill-rot / F SAS off / G re-capture-toggle · **1 prograde / 2 retrograde / 3 gravity-turn / 0 autopilot-off** (WI 565) · **`[`/`]` tune kp, `-`/`=` tune kd** (WI 566) · `,`/`.` warp; full flight HUD with Δv, apoapsis/periapsis, specific energy, the **control tier** (direct/stabilized/canned/tunable/uncontrolled — WI 562/565/566), SAS availability/re-capture (WI 564), the engaged autopilot, live SAS gains, and (WI 570) a **battery charge gauge** for a craft assembled from placed devices (control point + computer + battery) — on depletion the installed tier label is unchanged but an **ASSIST OFFLINE (low power)** marker appears and assistance drops to the unpowered floor. **C/V** (WI 571) downshift / restore the **player-selected control tier** (fly below capability for skill or to conserve power); the HUD shows **avail / sel / eff** tiers
- `cargo run -p sounding -- skins` — voxel-skin comparison (WI 582/583): the same craft rendered two ways and flown side by side from one sim state under the `hull_panel` PBR — **blocky** (per-cell cubes, Stormworks-style) vs the **greedy-meshed hull** (Starbase-style, the primary look), over a tiled **rocky-ground** surface (WI 588). WI 582 lands the blocky skin + the scene; WI 583 adds the hull
- `cargo run -p sounding -- land` — craft↔terrain collision demo (WI 590–592): a craft is dropped onto the tiled ground and the penalty contact response (detection via `parry3d-f64`) brings it to rest — `R` re-drop, `1`/`2` low/high drop
- `cargo run -p sounding -- collide` — craft↔craft collision demo (WI 593): fire a projectile craft (`SPACE`) at a target, plus a settling debris pile — the same penalty response generalized to body↔body — `R` reset
- `cargo run -p sounding -- crash` — breakage-on-impact demo (WI 594): hold `SPACE` to ram a frangible craft into a heavy block; a hard impact routes the contact force into fracture and it shatters into collidable fragments — `R` reset
- `cargo run -p sounding -- workshop` — grounded build-and-test sandbox (WI 599/602/603/604/607/611/612/613/614/618/630/631a/631b/634): **Build** a craft with the **mouse** on a solid 0.1 m grid (left-click a face to add, right-click to remove, middle-drag orbit, scroll zoom; pick blocks/devices/parts/wheels from the on-screen **palette** down the left edge — wheels come as **off-road / road / slick** rim+tyre presets (each drives differently) with an **optional suspension** strut (omit it and the wheel rides on the tyre's give) — or the keyboard: `Tab` material · `1`-`5` devices · `6`/`7` road wheel · `U` suspension · `8`/`9`/`0`/`-` seat/antenna/solar/bumper · **`K` save vehicle / `O` load vehicle** into a named library under `saves/crafts/` — `K` names it, `O` browses saved builds (WI 637/675); each material renders with its own surface) ↔ **Test** — *what you built*: a build with **wheels drives as a rover** (solid chassis + steering tyres + parts; W/S drive · A/D steer (smoothed + eased at speed) · Space brake; **change the tyres** and feel it — the HUD shows the wheel's grip/radius/slip, and each wheel now has its **own mass and suspension travel**, so it **hops and soaks up bumps** (you can watch the wheels work) instead of tracking the ground rigidly; **roll it** with a hard turn at speed or an obstacle clip, while normal driving stays upright; drive off the **30° ramp** to catch air and tumble; **ram an obstacle** and the facing **wheels shear off**, or **land a big drop too hard** and they tear off too (it then drives lopsided / rests on its **hull** — flip it over and it rests on its chassis, not bouncing on its wheels); hit it hard but short of shearing and a **part fails first** — a **tyre blows** (it runs on the rim, low grip, more drag), a **rim bends** (the corner pulls), or a **damper blows** (that corner goes bouncy), and the HUD/console names what broke; hit an obstacle **hard enough** and the **chassis itself fractures** (WI 629) into tumbling, collidable debris that settles on the pad — driving ends, **`Backspace` → BUILD** to rebuild; strong brakes lock at the grip limit; powered by an engine+tank → **fuel** or battery+solar → **charge**, shown in the HUD — run dry and it coasts; collides with **obstacles** on the pad), anything else **flies** on the textured ground with live collision (land, rest, crash → shatter); toggle with `Enter`, `Backspace` rebuilds, **`P` pauses/resumes** the world and **`.` steps** a paused world forward a beat (WI 638/643; also in `-- rover` and `-- play`; `.`/`Step` is also a bus/MCP command for inspection); **`G`** toggles a **cockpit overlay** of live signal sparklines (WI 645/646) and **`R`** enters a **replay cam** — scrub the last few seconds with `[`/`]` (WI 648; replay + screenshot are bus/MCP-drivable for AI inspection). **WI 624/652/653/654/655:** structural materials render with **bespoke per-material PBR textures** (brushed aluminium / steel / titanium / carbon, asset-harness); catalog **parts and devices render as their real mechanical-kit glb meshes**, seated on the clicked chassis face (Build **and** while driving); pick a **motor tier** with **`M`** (Economy / Standard / Performance / Heavy — sizes the drivetrain's torque, top-speed, mass, draw), shown in the HUD. **WI 617 (gamepad):** a connected **game controller** drives/flies/orbits alongside keyboard+mouse (additive) — left stick steer/roll + pitch, **triggers** throttle, **bumpers** yaw (air) / handbrake (ground), **right stick** camera (build orbit **and** free-look while driving/flying, WI 665), **Y** SAS, **Start** pause, **Back** → BUILD; rebindable mapping table with a deadzone (layout rationale in `docs/projects/sounding/controller-mapping-research.md`). **WI 775 (drive on a moon):** add `moon [seed]` — `cargo run -p sounding -- workshop moon` — to Test a built rover on a generated cratered **moonlet** instead of the flat pad. The wheels contact the **analytic** procedural surface (WI 765 rebind, streamed textured via WI 764) so it can't fall through and stays kraken-free; the HUD shows the moonlet's radius/gravity. The default `-- workshop` (flat pad) is unchanged
- `cargo run -p sounding -- gallery` — **part catalog viewer** (WI 653): every mechanical-kit part laid out on the ground grouped by category, slowly spinning; **middle-drag** orbit · **scroll** zoom · **WASD** pan · **Space** spin; **click a part** to inspect its read-only properties (name, category, material, orientation, verts, device mass)

## Runtime bus

While the app runs, a runtime state/command bus listens on
`http://127.0.0.1:8787` (a synchronous HTTP server on its own thread). It is the
shared substrate later consumers — the AI companion, second screen, multiplayer
sync — adapt onto. It is distinct from the dev-only Bevy Remote Protocol.

- `GET /telemetry` — current simulation snapshot as JSON (time, warp, paused,
  craft orbit and position, energy-drift metric; plus additive `active` / `rover` /
  `thermal` blocks when a scene publishes them — e.g. `thermal.max_skin_temp` in the dive).
- `POST /command` — inject a JSON command into the flight-control executor;
  malformed input returns HTTP 400.

```bash
curl -s localhost:8787/telemetry
curl -s -X POST localhost:8787/command -d '{"SetWarp":8.0}'
curl -s -X POST localhost:8787/command -d '{"SetPaused":true}'
curl -s -X POST localhost:8787/command \
  -d '{"ExecuteManeuver":{"delta_v":[0.0,0.1]}}'
```

A dev-only **MCP bridge** over this bus lives at [`dev/mcp/`](dev/mcp/README.md) (WI 639): a
stdlib-only stdio server exposing `get_telemetry` / `send_command` so an AI assistant can observe and
steer the running scene. It is registered by the user, never part of the game build.

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

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option. This is the standard dual license of the Rust ecosystem.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

See [ATTRIBUTION.md](ATTRIBUTION.md) for third-party code and bundled-asset
attribution.

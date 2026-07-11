//! Sounding application: the windowed Bevy app that wraps the rendering-free
//! simulation core (`sounding_sim`).
//!
//! The app hosts one of several **toy scenes**, selected at launch (WI 514):
//!
//! - `cargo run -p sounding` — Toy 5 voxel ship editor (default)
//! - `cargo run -p sounding -- planet` — Toy 4 floating-origin planet + atmosphere
//! - `cargo run -p sounding -- rover` — Toy 6 rover on terrain
//! - `cargo run -p sounding -- dive` — the dive (orbit → atmosphere → ocean), **from scenario data** (WI 739): `content/scenarios/dive.ron`
//! - `cargo run -p sounding -- break` — structural breakage (a spinning craft snaps apart)
//! - `cargo run -p sounding -- compartments` — airtight compartments (hatch + breach)
//! - `cargo run -p sounding -- flooding` — decompression/flooding (breach a submerged craft)
//! - `cargo run -p sounding -- windtunnel` — aero: lift curve + transonic area-ruling plots
//! - `cargo run -p sounding -- launch` — surface lift-off **from scenario data** (WI 739): `content/scenarios/launch.ron` on the scenario scene (the liftoff mission throttles it up)
//! - `cargo run -p sounding -- autopilot` — a hands-off sounding **from scenario data** (WI 739): `content/scenarios/autopilot.ron`
//! - `cargo run -p sounding -- play` — fly a craft by hand **from scenario data** (WI 739): `content/scenarios/play.ron`, with the full flight HUD (Δv, apsides, energy)
//! - `cargo run -p sounding -- resume [slug]` — resume a world save (WI 553): `saves/worlds/<slug>.json`, or the most recent save when the slug is omitted
//! - `cargo run -p sounding -- skins` — voxel-skin comparison: the same craft flown side by side, blocky vs greedy-meshed hull
//! - `cargo run -p sounding -- land` — drop a craft and watch the collision response bring it to rest
//! - `cargo run -p sounding -- collide [projectile] [target]` — fire a craft at another (and a debris pile) — craft↔craft collision; optional saved-craft fixtures (WI 843)
//! - `cargo run -p sounding -- crash [projectile] [target]` — ram a frangible craft into a block — breakage-on-impact (it shatters); optional saved-craft fixtures (WI 843, try `wedge-dart`)
//! - `cargo run -p sounding -- workshop` — grounded build-and-test sandbox: Build (edit a craft) ↔ Test (fly what you built, with live ground collision), toggle with Enter. Add `moon [seed]` (`-- workshop moon`) to Test a built rover on a generated cratered moon (WI 775)
//! - `cargo run -p sounding -- materials` — preview a generated PBR material set on lit geometry
//! - `cargo run -p sounding -- terrainmesh` — preview a generated MoGe terrain relief (glTF)
//! - `cargo run -p sounding -- gallery` — part catalog viewer: every mechanical-kit part laid out by category, click to inspect
//! - `cargo run -p sounding -- harbor` — float a built hull on calm water by a dock (WI 705/711 righting + enclosed buoyancy); seed hull + Float spawn **from scenario data** (WI 739): `content/scenarios/harbor.ron`
//! - `cargo run -p sounding -- surface [seed] [archetype]` — fly a generated body from orbit to its streamed procedural surface (quadtree LOD, WI 764)
//! - `cargo run -p sounding -- moon` — land and drive a rover on a generated cratered moon: analytic-surface contact (WI 765) under the WI 764 render
//!
//! The Toy 1–3 simulation and runtime bus run headless behind whichever scene is
//! shown, so the companion still works. Per-scene controls are documented in
//! `editor.rs` and `planet.rs`.

use bevy::math::DVec2;
use bevy::prelude::*;
use sounding_sim::command::FlightControlPlugin;
use sounding_sim::diagnostics::SimDiagnosticsPlugin;
use sounding_sim::orbit::Orbit;
use sounding_sim::sim::{CentralBody, OrbitPlugin};

mod bodies_scene;
mod break_scene;
mod build;
mod bus;
mod check_cli;
mod collide_scene;
mod compartments_scene;
mod craft_library;
mod crash_scene;
mod debug_control;
mod dive_scene;
mod editor;
mod export_cli;
mod floating_origin;
mod flooding_scene;
mod gallery_scene;
mod gamepad;
mod ground;
mod harbor_scene;
mod harness_fixture;
mod input_inject;
mod land_scene;
mod materials_scene;
mod moon_scene;
mod net;
mod overlay;
mod parts;
mod pause;
mod planet;
mod replay;
mod rover_scene;
mod scenario_scene;
mod scene_cam;
mod scene_water;
mod skins_scene;
mod sparkline;
mod surface_scene;
mod surface_stream;
mod terrain_mesh_scene;
mod voxel_skin;
mod wind_tunnel_scene;
mod workshop_scene;

use bodies_scene::BodiesScenePlugin;
use break_scene::BreakScenePlugin;
use collide_scene::CollideScenePlugin;
use compartments_scene::CompartmentsScenePlugin;
use crash_scene::CrashScenePlugin;
use dive_scene::DiveScenePlugin;
use editor::EditorPlugin;
use flooding_scene::FloodingScenePlugin;
use gallery_scene::GalleryScenePlugin;
use harbor_scene::HarborScenePlugin;
use land_scene::LandScenePlugin;
use materials_scene::MaterialsScenePlugin;
use moon_scene::MoonScenePlugin;
use planet::PlanetPlugin;
use rover_scene::RoverScenePlugin;
use scenario_scene::ScenarioScenePlugin;
use skins_scene::SkinsScenePlugin;
use surface_scene::SurfaceScenePlugin;
use terrain_mesh_scene::TerrainMeshScenePlugin;
use wind_tunnel_scene::WindTunnelScenePlugin;
use workshop_scene::WorkshopScenePlugin;

/// Which toy scene the windowed app shows. The flight-family flags
/// (`play`/`launch`/`autopilot`) are **scenario aliases** (WI 739): one
/// scenario scene, different shipped documents.
enum Scene {
    Editor,
    Bodies,
    Planet,
    Rover,
    Dive,
    Break,
    Compartments,
    Flooding,
    WindTunnel,
    Materials,
    Skins,
    Land,
    Collide,
    Crash,
    Workshop,
    TerrainMesh,
    Gallery,
    Harbor,
    Surface,
    Moon,
    /// The scenario scene with the given default document (an explicit
    /// `-- <alias> <path>` still overrides it).
    Scenario(&'static str),
    /// Resume a world save (`-- resume [slug]`, WI 553): the slug names
    /// `saves/worlds/<slug>.json`; omitted, the most recent save resumes.
    Resume,
}

fn selected_scene() -> Scene {
    match std::env::args().nth(1).as_deref() {
        Some("bodies") => Scene::Bodies,
        Some("planet") => Scene::Planet,
        Some("rover") => Scene::Rover,
        Some("dive") => Scene::Dive,
        Some("break") => Scene::Break,
        Some("compartments") => Scene::Compartments,
        Some("flooding") => Scene::Flooding,
        Some("windtunnel") => Scene::WindTunnel,
        Some("launch") => Scene::Scenario("content/scenarios/launch.ron"),
        Some("autopilot") => Scene::Scenario("content/scenarios/autopilot.ron"),
        Some("play") => Scene::Scenario("content/scenarios/play.ron"),
        Some("materials") => Scene::Materials,
        Some("skins") => Scene::Skins,
        Some("land") => Scene::Land,
        Some("collide") => Scene::Collide,
        Some("crash") => Scene::Crash,
        Some("workshop") => Scene::Workshop,
        Some("terrainmesh") => Scene::TerrainMesh,
        Some("gallery") => Scene::Gallery,
        Some("harbor") => Scene::Harbor,
        Some("surface") => Scene::Surface,
        Some("moon") => Scene::Moon,
        Some("scenario") => Scene::Scenario("content/scenarios/first-flight.ron"),
        Some("resume") => Scene::Resume,
        _ => Scene::Editor,
    }
}

fn main() {
    // `sounding check` (WI 896): a headless authoring report — dispatched
    // before any Bevy `App` exists, prints and exits (warnings exit 0).
    if std::env::args().nth(1).as_deref() == Some("check") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        std::process::exit(check_cli::run(&args));
    }
    // `sounding export-body` (WI 897): the write-side sibling — emit a kept
    // body as an authored, self-verified pack; same pre-Bevy dispatch.
    if std::env::args().nth(1).as_deref() == Some("export-body") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        std::process::exit(export_cli::run(&args));
    }
    // Toys 1–3 keep running headless: the on-rails orbit and the runtime bus stay
    // live so the companion still works, whichever scene is shown. All SI (WI 527):
    // the one canonical unit system, shared with the scenes via `CentralBody::EARTHLIKE`.
    let central_body = CentralBody::EARTHLIKE;
    // A low, eccentric Earth orbit (periapsis ~200 km, prograde, faster than
    // circular) so the companion navigator has an orbit to circularize.
    let initial_orbit = Orbit::from_state(
        central_body.mu,
        DVec2::new(central_body.radius + 200_000.0, 0.0),
        DVec2::new(0.0, 8_200.0),
        0.0,
    )
    .expect("initial orbit is bound");

    let mut app = App::new();
    // Gamepad mapping table (WI 617): one rebindable resource read by the keyboard input systems
    // (rover/rocket/flight/build-camera) as an additive controller source.
    app.init_resource::<gamepad::GamepadMap>();
    // Shared chase-camera free-look offset (WI 665), driven by the right stick in Test / -- play.
    app.init_resource::<gamepad::ChaseLook>();
    // Craft save library (WI 675): the modal state + last-used name are read by the editor's
    // input guard in any scene that runs the voxel editor (standalone editor + workshop Build).
    app.init_resource::<craft_library::CraftLibraryModal>();
    app.init_resource::<craft_library::CurrentCraftName>();
    app.add_plugins(DefaultPlugins)
        .add_plugins(OrbitPlugin {
            central_body,
            initial_orbit,
        })
        .add_plugins(FlightControlPlugin)
        .add_plugins(SimDiagnosticsPlugin)
        .add_plugins(bus::BusPlugin::default())
        .add_plugins(replay::ReplayPlugin)
        .add_plugins(debug_control::DebugControlPlugin)
        // Multiplayer net adapter (WI 857): dormant unless SOUNDING_SERVER/
        // SOUNDING_INVITE/SOUNDING_PLAYER are set — single-player is untouched.
        .add_plugins(net::NetPlugin);

    match selected_scene() {
        Scene::Editor => {
            app.add_plugins(EditorPlugin);
        }
        Scene::Bodies => {
            app.add_plugins(BodiesScenePlugin);
        }
        Scene::Planet => {
            app.add_plugins(PlanetPlugin);
        }
        Scene::Rover => {
            app.add_plugins(RoverScenePlugin);
        }
        Scene::Dive => {
            app.add_plugins(DiveScenePlugin);
        }
        Scene::Break => {
            app.add_plugins(BreakScenePlugin);
        }
        Scene::Compartments => {
            app.add_plugins(CompartmentsScenePlugin);
        }
        Scene::Flooding => {
            app.add_plugins(FloodingScenePlugin);
        }
        Scene::WindTunnel => {
            app.add_plugins(WindTunnelScenePlugin);
        }
        Scene::Materials => {
            app.add_plugins(MaterialsScenePlugin);
        }
        Scene::Skins => {
            app.add_plugins(SkinsScenePlugin);
        }
        Scene::Land => {
            app.add_plugins(LandScenePlugin);
        }
        Scene::Collide => {
            app.add_plugins(CollideScenePlugin);
        }
        Scene::Crash => {
            app.add_plugins(CrashScenePlugin);
        }
        Scene::Workshop => {
            app.add_plugins(WorkshopScenePlugin);
        }
        Scene::Gallery => {
            app.add_plugins(GalleryScenePlugin);
        }
        Scene::Harbor => {
            app.add_plugins(HarborScenePlugin);
        }
        Scene::TerrainMesh => {
            app.add_plugins(TerrainMeshScenePlugin);
        }
        Scene::Surface => {
            app.add_plugins(SurfaceScenePlugin);
        }
        Scene::Moon => {
            app.add_plugins(MoonScenePlugin);
        }
        Scene::Scenario(default_doc) => {
            app.add_plugins(ScenarioScenePlugin {
                default_doc,
                resume: false,
            });
        }
        Scene::Resume => {
            app.add_plugins(ScenarioScenePlugin {
                resume: true,
                ..Default::default()
            });
        }
    }

    #[cfg(feature = "dev")]
    add_dev_tools(&mut app);

    app.run();
}

/// Registers dev-only tooling. Compiled only under the `dev` feature so that
/// the Bevy Remote Protocol is absent from default and release builds.
#[cfg(feature = "dev")]
fn add_dev_tools(app: &mut App) {
    use bevy::remote::http::RemoteHttpPlugin;
    use bevy::remote::RemotePlugin;

    app.add_plugins(RemotePlugin::default())
        .add_plugins(RemoteHttpPlugin::default());
    info!("dev: Bevy Remote Protocol enabled (HTTP transport)");
}

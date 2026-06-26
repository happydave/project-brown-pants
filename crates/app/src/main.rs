//! Sounding application: the windowed Bevy app that wraps the rendering-free
//! simulation core (`sounding_sim`).
//!
//! The app hosts one of several **toy scenes**, selected at launch (WI 514):
//!
//! - `cargo run -p sounding` — Toy 5 voxel ship editor (default)
//! - `cargo run -p sounding -- planet` — Toy 4 floating-origin planet + atmosphere
//! - `cargo run -p sounding -- rover` — Toy 6 rover on terrain
//! - `cargo run -p sounding -- dive` — Toy 9 the dive (orbit → atmosphere → ocean)
//! - `cargo run -p sounding -- break` — structural breakage (a spinning craft snaps apart)
//! - `cargo run -p sounding -- compartments` — airtight compartments (hatch + breach)
//! - `cargo run -p sounding -- flooding` — decompression/flooding (breach a submerged craft)
//! - `cargo run -p sounding -- windtunnel` — aero: lift curve + transonic area-ruling plots
//! - `cargo run -p sounding -- launch` — surface lift-off: a rocket rests on the pad, then ascends under thrust
//! - `cargo run -p sounding -- autopilot` — a continuous one-craft session flown automatically: Launch → Flight → Recovery (a sounding)
//! - `cargo run -p sounding -- play` — fly a craft by hand: throttle/attitude/SAS/warp, with a full flight HUD (Δv, apsides, energy)
//! - `cargo run -p sounding -- skins` — voxel-skin comparison: the same craft flown side by side, blocky vs greedy-meshed hull
//! - `cargo run -p sounding -- land` — drop a craft and watch the collision response bring it to rest
//! - `cargo run -p sounding -- collide` — fire a craft at another (and a debris pile) — craft↔craft collision
//! - `cargo run -p sounding -- crash` — ram a frangible craft into a block — breakage-on-impact (it shatters)
//! - `cargo run -p sounding -- workshop` — grounded build-and-test sandbox: Build (edit a craft) ↔ Test (fly what you built, with live ground collision), toggle with Enter
//! - `cargo run -p sounding -- materials` — preview a generated PBR material set on lit geometry
//! - `cargo run -p sounding -- terrainmesh` — preview a generated MoGe terrain relief (glTF)
//! - `cargo run -p sounding -- gallery` — part catalog viewer: every mechanical-kit part laid out by category, click to inspect
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

mod autopilot_scene;
mod break_scene;
mod bus;
mod collide_scene;
mod compartments_scene;
mod crash_scene;
mod dive_scene;
mod editor;
mod floating_origin;
mod flooding_scene;
mod gallery_scene;
mod gamepad;
mod ground;
mod land_scene;
mod launch_scene;
mod materials_scene;
mod overlay;
mod parts;
mod pause;
mod planet;
mod play_scene;
mod replay;
mod rover_scene;
mod skins_scene;
mod sparkline;
mod terrain_mesh_scene;
mod voxel_skin;
mod wind_tunnel_scene;
mod workshop_scene;

use autopilot_scene::AutopilotScenePlugin;
use break_scene::BreakScenePlugin;
use collide_scene::CollideScenePlugin;
use compartments_scene::CompartmentsScenePlugin;
use crash_scene::CrashScenePlugin;
use dive_scene::DiveScenePlugin;
use editor::EditorPlugin;
use flooding_scene::FloodingScenePlugin;
use gallery_scene::GalleryScenePlugin;
use land_scene::LandScenePlugin;
use launch_scene::LaunchScenePlugin;
use materials_scene::MaterialsScenePlugin;
use planet::PlanetPlugin;
use play_scene::PlayScenePlugin;
use rover_scene::RoverScenePlugin;
use skins_scene::SkinsScenePlugin;
use terrain_mesh_scene::TerrainMeshScenePlugin;
use wind_tunnel_scene::WindTunnelScenePlugin;
use workshop_scene::WorkshopScenePlugin;

/// Which toy scene the windowed app shows.
enum Scene {
    Editor,
    Planet,
    Rover,
    Dive,
    Break,
    Compartments,
    Flooding,
    WindTunnel,
    Launch,
    Autopilot,
    Play,
    Materials,
    Skins,
    Land,
    Collide,
    Crash,
    Workshop,
    TerrainMesh,
    Gallery,
}

fn selected_scene() -> Scene {
    match std::env::args().nth(1).as_deref() {
        Some("planet") => Scene::Planet,
        Some("rover") => Scene::Rover,
        Some("dive") => Scene::Dive,
        Some("break") => Scene::Break,
        Some("compartments") => Scene::Compartments,
        Some("flooding") => Scene::Flooding,
        Some("windtunnel") => Scene::WindTunnel,
        Some("launch") => Scene::Launch,
        Some("autopilot") => Scene::Autopilot,
        Some("play") => Scene::Play,
        Some("materials") => Scene::Materials,
        Some("skins") => Scene::Skins,
        Some("land") => Scene::Land,
        Some("collide") => Scene::Collide,
        Some("crash") => Scene::Crash,
        Some("workshop") => Scene::Workshop,
        Some("terrainmesh") => Scene::TerrainMesh,
        Some("gallery") => Scene::Gallery,
        _ => Scene::Editor,
    }
}

fn main() {
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
    app.add_plugins(DefaultPlugins)
        .add_plugins(OrbitPlugin {
            central_body,
            initial_orbit,
        })
        .add_plugins(FlightControlPlugin)
        .add_plugins(SimDiagnosticsPlugin)
        .add_plugins(bus::BusPlugin::default())
        .add_plugins(replay::ReplayPlugin);

    match selected_scene() {
        Scene::Editor => {
            app.add_plugins(EditorPlugin);
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
        Scene::Launch => {
            app.add_plugins(LaunchScenePlugin);
        }
        Scene::Autopilot => {
            app.add_plugins(AutopilotScenePlugin);
        }
        Scene::Play => {
            app.add_plugins(PlayScenePlugin);
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
        Scene::TerrainMesh => {
            app.add_plugins(TerrainMeshScenePlugin);
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

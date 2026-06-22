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
//! - `cargo run -p sounding -- materials` — preview a generated PBR material set on lit geometry
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
mod compartments_scene;
mod dive_scene;
mod editor;
mod floating_origin;
mod flooding_scene;
mod launch_scene;
mod materials_scene;
mod planet;
mod play_scene;
mod rover_scene;
mod skins_scene;
mod voxel_skin;
mod wind_tunnel_scene;

use autopilot_scene::AutopilotScenePlugin;
use break_scene::BreakScenePlugin;
use compartments_scene::CompartmentsScenePlugin;
use dive_scene::DiveScenePlugin;
use editor::EditorPlugin;
use flooding_scene::FloodingScenePlugin;
use launch_scene::LaunchScenePlugin;
use materials_scene::MaterialsScenePlugin;
use planet::PlanetPlugin;
use play_scene::PlayScenePlugin;
use rover_scene::RoverScenePlugin;
use skins_scene::SkinsScenePlugin;
use wind_tunnel_scene::WindTunnelScenePlugin;

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
    app.add_plugins(DefaultPlugins)
        .add_plugins(OrbitPlugin {
            central_body,
            initial_orbit,
        })
        .add_plugins(FlightControlPlugin)
        .add_plugins(SimDiagnosticsPlugin)
        .add_plugins(bus::BusPlugin::default());

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

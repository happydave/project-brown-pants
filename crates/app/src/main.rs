//! Sounding application: the windowed Bevy app that wraps the rendering-free
//! simulation core (`sounding_sim`).
//!
//! Toy 5 is the **voxel ship editor** (WI 505): build a craft from voxels and
//! devices and watch its centre of mass, principal inertia axes, and aero
//! cross-sectional-area curve update live. The Toy 1–3 simulation/bus plugins keep
//! running headless behind it (the Toy 4 planet scene is retired as the default;
//! its `floating_origin` module is retained for when in-world flight returns).
//!
//! Editor controls:
//! - Arrow keys / `PageUp`·`PageDown` — move the build cursor (X/Z, then Y)
//! - `Space` — add a voxel · `Backspace` — remove voxel/device at the cursor
//! - `Tab` — cycle material · `G` — place a device · `M` — log mass properties
//! - `B` — save blueprint · `N` — save subassembly · `L` — load subassembly · `V` — insert it at the cursor
//!
//! Camera controls: `Q`/`E` orbit · `R`/`F` pitch · `Z`/`C` zoom.

use bevy::math::DVec2;
use bevy::prelude::*;
use sounding_sim::command::FlightControlPlugin;
use sounding_sim::diagnostics::SimDiagnosticsPlugin;
use sounding_sim::orbit::Orbit;
use sounding_sim::sim::{CentralBody, OrbitPlugin};

mod bus;
mod editor;
mod floating_origin;

use editor::EditorPlugin;

fn main() {
    // Toys 1–3 keep running headless: the on-rails orbit and the runtime bus stay
    // live so the companion still works, even while the editor is the view.
    let central_body = CentralBody {
        mu: 1.0,
        radius: 0.08,
    };
    let initial_orbit = Orbit::from_state(
        central_body.mu,
        DVec2::new(1.0, 0.0),
        DVec2::new(0.0, 1.15),
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
        .add_plugins(bus::BusPlugin::default())
        .add_plugins(EditorPlugin)
        .init_resource::<OrbitCam>()
        .add_systems(Startup, setup)
        .add_systems(Update, orbit_camera);

    #[cfg(feature = "dev")]
    add_dev_tools(&mut app);

    app.run();
}

/// Orbit-camera state: yaw, pitch, and distance about the craft.
#[derive(Resource)]
struct OrbitCam {
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl Default for OrbitCam {
    fn default() -> Self {
        Self {
            yaw: 0.7,
            pitch: 0.5,
            dist: 14.0,
        }
    }
}

fn setup(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Transform::default()));
    commands.spawn((
        DirectionalLight {
            illuminance: 6_000.0,
            ..default()
        },
        Transform::from_xyz(6.0, 12.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Keyboard orbit camera, always framing the editor's build volume.
fn orbit_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut cam: ResMut<OrbitCam>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    let dt = time.delta_secs();
    if keys.pressed(KeyCode::KeyQ) {
        cam.yaw += dt;
    }
    if keys.pressed(KeyCode::KeyE) {
        cam.yaw -= dt;
    }
    if keys.pressed(KeyCode::KeyR) {
        cam.pitch = (cam.pitch + dt).clamp(0.05, 1.5);
    }
    if keys.pressed(KeyCode::KeyF) {
        cam.pitch = (cam.pitch - dt).clamp(0.05, 1.5);
    }
    if keys.pressed(KeyCode::KeyZ) {
        cam.dist = (cam.dist - dt * 12.0).max(2.0);
    }
    if keys.pressed(KeyCode::KeyC) {
        cam.dist = (cam.dist + dt * 12.0).min(80.0);
    }

    let target = Vec3::new(1.5, 0.5, 1.0);
    let dir = Vec3::new(
        cam.yaw.cos() * cam.pitch.cos(),
        cam.pitch.sin(),
        cam.yaw.sin() * cam.pitch.cos(),
    );
    if let Ok(mut tf) = camera.single_mut() {
        *tf = Transform::from_translation(target + dir * cam.dist).looking_at(target, Vec3::Y);
    }
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

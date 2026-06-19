//! Sounding application: the windowed Bevy app that wraps the rendering-free
//! simulation core (`sounding_sim`).
//!
//! Toy 4 renders a planetary-scale world with **floating-origin** precision and
//! Bevy's physically-based **atmosphere** (the "beautiful from orbit" pillar). A
//! small craft sits at the surface site near the world origin; the planet centre
//! is one planet-radius below in +Y-up, surface-centred coordinates. The
//! simulation/bus plugins (Toys 1–3) keep running headless behind the scene.
//!
//! Controls:
//! - `W`/`S`, `A`/`D` — fly forward/back, strafe left/right
//! - `R`/`F`         — ascend / descend (fly from the surface up to orbit)
//! - Arrow keys      — look (yaw / pitch)
//! - `P`             — pause / resume the sun's motion (the terminator sweep)
//!
//! Fly upward (`R`) to orbital altitude and watch the sun sweep the terminator
//! for an orbital sunrise; fly back down (`F`) to see the craft jitter-free.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::{DVec2, DVec3};
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::command::FlightControlPlugin;
use sounding_sim::diagnostics::SimDiagnosticsPlugin;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::orbit::Orbit;
use sounding_sim::sim::{CentralBody, OrbitPlugin};
use std::f32::consts::PI;

mod bus;
mod floating_origin;

use floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};

/// Planet (and atmosphere ground) radius, metres — matches `Atmosphere::earthlike`.
const PLANET_RADIUS_M: f64 = 6_360_000.0;

fn main() {
    // Toys 1–3 keep running headless: the on-rails orbit and the runtime bus stay
    // live so the companion still works, even though the 2D view is retired.
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
        .add_plugins(FloatingOriginPlugin)
        .init_resource::<SunPaused>()
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (fly_camera, rotate_sun, toggle_sun));

    #[cfg(feature = "dev")]
    add_dev_tools(&mut app);

    app.run();
}

/// Marks the sun (the directional light) so it can be rotated to sweep the
/// terminator.
#[derive(Component)]
struct Sun;

/// Whether the sun's motion is paused.
#[derive(Resource, Default)]
struct SunPaused(bool);

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
) {
    // The planet: a large sphere whose top sits at sea level (world Y = 0), so
    // its centre is one radius below. Surface micro-precision near the craft is a
    // later toy's concern (Toy 6); here it is the planetary-scale backdrop.
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere {
            radius: PLANET_RADIUS_M as f32,
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.42, 0.32, 0.27),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -PLANET_RADIUS_M, 0.0),
        )),
    ));

    // The small craft, resting just above the surface site at the world origin.
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Cuboid::new(2.0, 2.0, 4.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.80, 0.81, 0.86),
            metallic: 0.7,
            perceptual_roughness: 0.3,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, 3.0, 0.0),
        )),
    ));

    // The sun: raw (pre-scattering) sunlight, the input the atmosphere filters.
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.35)),
        Sun,
    ));

    // HDR camera with Bevy's physically-based atmosphere. It starts near the
    // craft; fly up (`R`) to reach orbit. `WorldPlacement` + `AnchorCamera` make
    // it the floating-origin anchor, pinned to the render origin in X/Z.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 8.0, 0.0).looking_at(Vec3::new(0.0, 3.0, -20.0), Vec3::Y),
        Atmosphere::earthlike(scattering.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, 8.0, 20.0),
        )),
        AnchorCamera,
    ));
}

/// A free-fly camera that edits the camera's f64 world placement (movement) and
/// f32 rotation (aim). Movement speed scales with altitude so it is usable from
/// the surface up to orbit.
fn fly_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    let Ok((mut tf, mut placement)) = camera.single_mut() else {
        return;
    };
    let dt = time.delta_secs();

    // Look: yaw about world up, pitch about the camera's local x.
    let rot_speed = 1.2;
    let mut yaw = 0.0;
    let mut pitch = 0.0;
    if keys.pressed(KeyCode::ArrowLeft) {
        yaw += 1.0;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        yaw -= 1.0;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        pitch += 1.0;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        pitch -= 1.0;
    }
    tf.rotate_y(yaw * rot_speed * dt);
    tf.rotate_local_x(pitch * rot_speed * dt);

    // Move: along the camera basis plus world up/down.
    let forward = *tf.forward();
    let right = *tf.right();
    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        dir += forward;
    }
    if keys.pressed(KeyCode::KeyS) {
        dir -= forward;
    }
    if keys.pressed(KeyCode::KeyD) {
        dir += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        dir -= right;
    }
    if keys.pressed(KeyCode::KeyR) {
        dir += Vec3::Y;
    }
    if keys.pressed(KeyCode::KeyF) {
        dir -= Vec3::Y;
    }
    if dir != Vec3::ZERO {
        let dir = dir.normalize();
        // Altitude-scaled speed: gentle near the surface, fast from orbit.
        let speed = (placement.0.pos.y.max(1.0) * 0.6).clamp(5.0, 8.0e5);
        let step = DVec3::new(dir.x as f64, dir.y as f64, dir.z as f64) * speed * dt as f64;
        placement.0.pos += step;
    }
}

/// Slowly rotates the sun so the terminator sweeps the planet (orbital sunrise).
fn rotate_sun(time: Res<Time>, paused: Res<SunPaused>, mut suns: Query<&mut Transform, With<Sun>>) {
    if paused.0 {
        return;
    }
    for mut tf in &mut suns {
        tf.rotate_x(-time.delta_secs() * PI / 30.0);
    }
}

/// Toggles the sun's motion with `P`.
fn toggle_sun(keys: Res<ButtonInput<KeyCode>>, mut paused: ResMut<SunPaused>) {
    if keys.just_pressed(KeyCode::KeyP) {
        paused.0 = !paused.0;
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

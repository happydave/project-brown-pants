//! Toy 4 floating-origin planet scene (WI 504), packaged as a selectable scene
//! (WI 514). A planetary-scale sphere, a small craft, a sun, and an HDR camera
//! with Bevy's physically-based atmosphere; a free-fly camera flies from the
//! surface up to orbit and the sun sweeps the terminator. Floating origin keeps
//! the metres-scale craft jitter-free against the planet radius.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::sim::CentralBody;
use std::f32::consts::PI;

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};

/// Planet (and atmosphere ground) radius, metres — the shared canonical SI body
/// (WI 527), matching `Atmosphere::earthlike`.
const PLANET_RADIUS_M: f64 = CentralBody::EARTHLIKE.radius;

/// The Toy 4 planet scene.
pub struct PlanetPlugin;

impl Plugin for PlanetPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .init_resource::<SunPaused>()
            .add_systems(Startup, setup_scene)
            .add_systems(Update, (fly_camera, rotate_sun, toggle_sun));
    }
}

/// Marks the sun so it can be rotated to sweep the terminator.
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
    // The planet: a large sphere whose top sits at sea level (world Y = 0), so its
    // centre is one radius below. Surface micro-precision is a later toy (Toy 6).
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

    // HDR camera with Bevy's physically-based atmosphere. Starts near the craft;
    // fly up (`R`) to orbit. `WorldPlacement` + `AnchorCamera` make it the
    // floating-origin anchor, pinned to the render origin in X/Z.
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

/// A free-fly camera editing the camera's f64 world placement (movement) and f32
/// rotation (aim). Speed scales with altitude, usable from the surface to orbit.
fn fly_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    let Ok((mut tf, mut placement)) = camera.single_mut() else {
        return;
    };
    let dt = time.delta_secs();

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

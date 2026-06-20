//! Toy 9 — the dive (WI 509).
//!
//! A craft descends **vacuum → atmosphere → ocean** in one continuous fall, with
//! aero/hydro drag, buoyancy, and pressure all produced by the single
//! `FluidMedium` field (`sounding_sim::medium`) — the multi-fluid thesis made
//! visible. A "sounding" descent: the craft falls vertically from high vacuum,
//! the thickening atmosphere decelerates it, and it splashes into the ocean.
//!
//! Rendering uses the floating-origin flat-ground convention (sea level at world
//! Y = 0, planet centre at `(0, -R, 0)`) so Bevy's atmosphere reads altitude from
//! the camera Y; the descent runs in the radial sim frame (planet at the origin)
//! and is converted for display. The on-rails↔active hand-off and its automatic
//! altitude trigger are validated headless (`sounding_sim::handoff`/`medium`); this
//! scene drives the active descent directly, like the rover scene.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::active::ActiveBody;
use sounding_sim::fluid::{FluidMedium, MediumKind};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::medium::{descent_step, max_cross_section, DescentParams};
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};

/// Planet (sea-level) radius, metres — matches `Atmosphere::earthlike` and the
/// planet scene.
const SURFACE_R: f64 = 6_360_000.0;
/// Central-body gravitational parameter, m³/s² (≈ g·R²).
const MU: f64 = 3.986e14;
/// Starting altitude of the drop, metres (high vacuum).
const START_ALT: f64 = 120_000.0;
/// Dive sub-step, seconds (the stiff splashdown wants a small step).
const SUBSTEP_DT: f64 = 0.004;
/// Cap on sub-steps per frame.
const MAX_SUBSTEPS: u32 = 600;

/// The descending craft and its medium constants. Self-contained, like the rover
/// scene's world.
#[derive(Resource)]
struct DiveWorld {
    body: ActiveBody,
    craft: VoxelCraft,
    com: DVec3,
    params: DescentParams,
    accumulator: f64,
    medium: MediumKind,
}

impl DiveWorld {
    fn new() -> Self {
        // A small composite capsule: a 2×2×3 m block (denser than water — it sinks).
        let mut craft = VoxelCraft::new(1.0);
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..3 {
                    craft.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
        }
        let mp = craft.mass_properties().expect("non-empty craft");
        // Radial sim frame: planet at the origin, craft straight up at start.
        let body = ActiveBody::new(
            DVec3::new(0.0, SURFACE_R + START_ALT, 0.0),
            DVec3::new(0.0, -100.0, 0.0), // a gentle initial nudge downward
            mp.mass,
            mp.inertia,
        );
        let params = DescentParams {
            medium: FluidMedium::EARTHLIKE,
            mu: MU,
            surface_radius: SURFACE_R,
            drag_area: max_cross_section(&craft),
            drag_coefficient: 1.0,
        };
        Self {
            body,
            craft,
            com: mp.center_of_mass,
            params,
            accumulator: 0.0,
            medium: MediumKind::Vacuum,
        }
    }

    /// Altitude above sea level, metres.
    fn altitude(&self) -> f64 {
        self.body.position.length() - SURFACE_R
    }

    /// The craft's flat-ground render position (sea level at Y = 0).
    fn render_world(&self) -> DVec3 {
        self.body.position - DVec3::new(0.0, SURFACE_R, 0.0)
    }
}

/// Marks the heads-up readout.
#[derive(Component)]
struct Hud;

/// Marks the rendered craft so its placement can be updated each frame.
#[derive(Component)]
struct CraftMarker;

/// The Toy 9 dive scene.
pub struct DiveScenePlugin;

impl Plugin for DiveScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .insert_resource(DiveWorld::new())
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (step_dive, track_craft, follow_camera, update_hud).chain(),
            );
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    dive: Res<DiveWorld>,
) {
    // Seabed: an opaque sphere just below sea level, centred one radius down.
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere {
            radius: (SURFACE_R - 4_000.0) as f32,
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.20, 0.17, 0.14),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -SURFACE_R, 0.0),
        )),
    ));

    // Ocean: a translucent blue sphere whose surface is sea level (Y = 0).
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere {
            radius: SURFACE_R as f32,
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.05, 0.20, 0.40, 0.55),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.1,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -SURFACE_R, 0.0),
        )),
    ));

    // The descending craft.
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Cuboid::new(2.0, 2.0, 3.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.84, 0.88),
            metallic: 0.6,
            perceptual_roughness: 0.3,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, dive.render_world())),
        CraftMarker,
    ));

    // The sun: raw sunlight the atmosphere filters.
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5) * Quat::from_rotation_y(0.6)),
    ));

    // Heads-up readout: altitude, speed, medium.
    commands.spawn((
        Text::new("altitude:        0 m\nspeed:       0.0 m/s\nmedium:   vacuum"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.95, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        Hud,
    ));

    // HDR camera with Bevy's physically-based atmosphere, chasing the craft.
    let cam = dive.render_world() + DVec3::new(18.0, 8.0, 18.0);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(cam.as_vec3())
            .looking_at(dive.render_world().as_vec3(), Vec3::Y),
        Atmosphere::earthlike(scattering.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, cam)),
        AnchorCamera,
    ));
}

/// Sub-steps the descent under gravity + drag + buoyancy from the one medium.
fn step_dive(time: Res<Time>, mut dive: ResMut<DiveWorld>) {
    dive.accumulator += time.delta_secs_f64();
    let mut n = 0;
    while dive.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS {
        // Stop integrating once well underwater so the capsule rests on the seabed
        // rather than tunnelling (the seabed is not a collision surface this toy).
        if dive.altitude() <= -3_500.0 {
            dive.accumulator = 0.0;
            break;
        }
        let DiveWorld {
            body,
            craft,
            com,
            params,
            ..
        } = &mut *dive;
        let sample = descent_step(body, craft, *com, params, SUBSTEP_DT);
        dive.medium = sample.medium;
        dive.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

/// Updates the craft entity's world placement from the simulation.
fn track_craft(dive: Res<DiveWorld>, mut craft: Query<&mut WorldPlacement, With<CraftMarker>>) {
    if let Ok(mut wp) = craft.single_mut() {
        wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, dive.render_world());
    }
}

/// Keeps the anchor camera a fixed offset from the descending craft.
fn follow_camera(
    dive: Res<DiveWorld>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    if let Ok((mut tf, mut placement)) = camera.single_mut() {
        let target = dive.render_world();
        let eye = target + DVec3::new(18.0, 8.0, 18.0);
        placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
        // Aim at the craft (render space; the floating origin keeps both near zero).
        let look_dir = (target - eye).as_vec3().normalize_or_zero();
        if look_dir != Vec3::ZERO {
            tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
        }
    }
}

fn update_hud(dive: Res<DiveWorld>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let medium = match dive.medium {
            MediumKind::Vacuum => "vacuum",
            MediumKind::Atmosphere => "atmosphere",
            MediumKind::Liquid => "ocean",
        };
        let alt = dive.altitude();
        let speed = dive.body.velocity.length();
        text.0 = format!("altitude: {alt:8.0} m\nspeed:    {speed:7.1} m/s\nmedium:   {medium}");
    }
}

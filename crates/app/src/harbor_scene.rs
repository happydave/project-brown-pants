//! Harbor — a calm waterfront where a built hull floats (`-- harbor`, WI 706).
//!
//! The first scene that makes the WI 705 righting buoyancy + WI 711 enclosed-volume buoyancy
//! **visible**: a sealed hull spawns **already active** at the surface and floats through the same
//! `DescentPlugin` → `glide_step` engine the dive uses (gravity + drag + the buoyancy wrench +
//! free-surface damping), self-righting from a small initial heel and settling to its waterline by a
//! dock. No orbit/handoff/warp — the harbor just floats a craft on sheltered water.
//!
//! Sealed-hull note (WI 711): buoyancy displaces the hull's **enclosed** air, so the fixture is a
//! *sealed* box. An open-top hull would sink until WI 713 (open-boat displacement). Camera: middle-
//! drag orbit + wheel zoom. The HUD shows the WI 705 draft / heel / net-buoyancy gauges.

use std::f32::consts::FRAC_PI_4;

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::{DQuat, DVec3};
use bevy::mesh::VertexAttributeValues;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;

use sounding_sim::active::{ActiveBody, Gravity};
use sounding_sim::fluid::FluidMedium;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::medium::{
    buoyancy_wrench, enclosed_cells, heel_angle, max_cross_section, DescentParams, DescentPlugin,
    DivingCraft, GlideParams, DEFAULT_SLAM_COEFFICIENT,
};
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Axis, Material, Voxel, VoxelCraft};

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};

const BODY: CentralBody = CentralBody::EARTHLIKE;

/// Float-integration sub-step and per-frame cap (real-time; the hull settles in a few seconds).
const SUBSTEP_DT: f64 = 0.002;
const MAX_SUBSTEPS: u32 = 64;

/// Fraction of the fully-submerged displaced-water mass used as the hull's mass — its reserve
/// buoyancy. ~0.4 floats it at roughly 40 % draft (a designer's ballast choice; the *buoyancy* is the
/// real WI 705/711 geometric displacement, only the hull weight is chosen for a pleasing waterline).
const HULL_MASS_FRACTION: f64 = 0.4;

/// Sea level lies at `BODY.radius`; render space puts it at `Y = 0` (floating origin).
fn render_world(sim_pos: DVec3) -> DVec3 {
    sim_pos - DVec3::new(0.0, BODY.radius, 0.0)
}

/// A sealed rectangular hull (a decked pontoon): the surface cells of a `W×H×L` box are solid, the
/// interior is enclosed air (WI 711 buoyancy). Editor-scale 0.5 m cells ⇒ a 3.5 × 2 × 5.5 m hull.
fn harbor_hull() -> VoxelCraft {
    let mut c = VoxelCraft::new(0.5);
    let (w, h, l) = (7, 4, 11);
    for x in 0..w {
        for y in 0..h {
            for z in 0..l {
                let on_surface =
                    x == 0 || x == w - 1 || y == 0 || y == h - 1 || z == 0 || z == l - 1;
                if on_surface {
                    c.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::ALUMINIUM,
                    });
                }
            }
        }
    }
    c
}

/// Mouse-driven orbit/zoom state for the harbor camera (the editor/gallery convention).
#[derive(Resource)]
struct HarborCam {
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl Default for HarborCam {
    fn default() -> Self {
        Self {
            yaw: FRAC_PI_4,
            pitch: 0.3,
            dist: 22.0,
        }
    }
}

/// Spherical orbit offset (shared math with the dive/gallery cameras).
fn orbit_offset(yaw: f32, pitch: f32, dist: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(sy * cp, sp, cy * cp) * dist
}

/// The HUD readout (WI 705 hydrostatic gauges).
#[derive(Resource, Default)]
struct HarborReadout {
    draft: f64,
    heel: f64,
    net_buoyancy: f64,
}

#[derive(Component)]
struct Hud;

#[derive(Component)]
struct HullMarker;

/// The animated calm-water patch.
#[derive(Component)]
struct WaterPatch;

const WATER_HALF: f32 = 160.0;
const WATER_SUBDIV: u32 = 64;
/// Calm harbor: a much smaller amplitude than the open-ocean dive (WI 703 used 0.55).
const WATER_AMPLITUDE: f32 = 0.12;

/// Height of the calm water surface at local patch coordinate `(x, z)` and time `t` — bounded summed
/// sines (weights sum to 1). Pure; computed in the patch's local frame so it ripples in place.
fn wave_height(x: f32, z: f32, t: f32) -> f32 {
    let w1 = (x * 0.10 + t * 0.7).sin();
    let w2 = (z * 0.13 - t * 0.6).sin();
    let w3 = ((x + z) * 0.06 + t * 0.5).sin();
    WATER_AMPLITUDE * (0.45 * w1 + 0.35 * w2 + 0.20 * w3)
}

/// The harbor scene (`-- harbor`, WI 706).
pub struct HarborScenePlugin;

impl Plugin for HarborScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .init_resource::<HarborReadout>()
            .init_resource::<HarborCam>()
            // The float engine: glide_step on any active craft carrying a DivingCraft (no orbit gear).
            .insert_resource(Gravity { mu: BODY.mu })
            .add_plugins(DescentPlugin {
                substep_dt: SUBSTEP_DT,
                max_substeps: MAX_SUBSTEPS,
            })
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (track_hull, harbor_camera_input, follow_camera, update_hud).chain(),
            )
            .add_systems(Update, animate_water);
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
) {
    let hull = harbor_hull();
    let mp = hull.mass_properties().expect("non-empty hull");
    let enclosed = enclosed_cells(&hull);

    // Float mass from the *real* displaceable volume (shell + enclosed), so the hull always floats at
    // a sensible draft regardless of the box dimensions (only the weight is a design choice).
    let water_density = FluidMedium::EARTHLIKE.sample_altitude(-1.0).density;
    let displaceable = hull.occupied_volume() + enclosed.len() as f64 * hull.cell_volume();
    let mass = HULL_MASS_FRACTION * water_density * displaceable;
    let inertia = mp.inertia * (mass / mp.mass);

    let descent = DescentParams {
        medium: FluidMedium::EARTHLIKE,
        mu: BODY.mu,
        surface_radius: BODY.radius,
        drag_area: max_cross_section(&hull),
        drag_coefficient: 1.0,
        slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
    };
    let glide = GlideParams::for_craft(descent, &hull, Axis::Z);

    // Spawn the hull **already active** at the surface, heeled slightly so the righting is visible.
    let start_sim = DVec3::new(0.0, BODY.radius, 0.0);
    let mut body = ActiveBody::new(start_sim, DVec3::ZERO, mass, inertia);
    body.orientation = DQuat::from_rotation_z(0.25); // a starting list to self-correct
    let start_render = render_world(start_sim);

    // Hull bounds in metres (cells × cell_size): 7×4×11 × 0.5.
    let hull_mesh = meshes.add(Mesh::from(Cuboid::new(3.5, 2.0, 5.5)));
    let hull_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.72, 0.34, 0.22), // a painted-hull red-brown
        metallic: 0.1,
        perceptual_roughness: 0.7,
        ..default()
    });
    let cabin_mesh = meshes.add(Mesh::from(Cuboid::new(2.0, 1.0, 2.4)));
    let cabin_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.90, 0.86),
        perceptual_roughness: 0.6,
        ..default()
    });

    commands
        .spawn((
            body,
            DivingCraft::new(hull, mp.center_of_mass, glide),
            Mesh3d(hull_mesh),
            MeshMaterial3d(hull_mat),
            Transform::default(),
            WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, start_render)),
            HullMarker,
        ))
        .with_children(|parent| {
            // A small deckhouse so the fixture reads as a boat (rides with the hull via Transform).
            parent.spawn((
                Mesh3d(cabin_mesh),
                MeshMaterial3d(cabin_mat),
                Transform::from_xyz(0.0, 1.4, -0.6),
            ));
        });

    // Distant ocean: a broad reflective blue sphere for the horizon, sunk 2 m so it does not z-fight
    // the animated near-surface patch (the dive's fix, WI 703).
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere {
            radius: BODY.radius as f32,
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.04, 0.18, 0.30, 0.94),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.06,
            reflectance: 0.6,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -2.0, 0.0),
        )),
    ));

    // Shallow sea floor: an opaque plane a few metres down, so a sinking hull grounds and the depth
    // reads. Rendered as a wide dim plane just below the hull.
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Plane3d::default().mesh().size(400.0, 400.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.16, 0.14, 0.11),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, -8.0, 0.0),
        )),
    ));

    // Calm near-surface water patch at sea level (render Y = 0); animate_water follows the camera.
    commands.spawn((
        Mesh3d(
            meshes.add(Mesh::from(
                Plane3d::default()
                    .mesh()
                    .size(2.0 * WATER_HALF, 2.0 * WATER_HALF)
                    .subdivisions(WATER_SUBDIV),
            )),
        ),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.10, 0.32, 0.44, 0.80),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.08,
            reflectance: 0.6,
            ..default()
        })),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(start_render.x, 0.0, start_render.z),
        )),
        WaterPatch,
    ));

    // A dock / quay: a low concrete deck on pilings, beside the hull, partly above the waterline.
    let quay_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.52, 0.48),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Cuboid::new(4.0, 1.2, 16.0)))),
        MeshMaterial3d(quay_mat.clone()),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(-6.0, 0.5, 0.0),
        )),
    ));
    // A couple of pilings.
    for z in [-6.0_f64, 6.0] {
        commands.spawn((
            Mesh3d(meshes.add(Mesh::from(Cuboid::new(0.5, 6.0, 0.5)))),
            MeshMaterial3d(quay_mat.clone()),
            Transform::default(),
            WorldPlacement(WorldPos::new(
                FrameId::CENTRAL_BODY,
                DVec3::new(-4.4, -2.5, z),
            )),
        ));
    }

    // Sun.
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.6) * Quat::from_rotation_y(0.5)),
    ));

    // HUD.
    commands.spawn((
        Text::new("harbor\ndraft:  --\nheel:   --\nnet buoy: --"),
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

    // HDR camera with the physically-based atmosphere, orbiting the hull.
    let eye = start_render + orbit_offset(FRAC_PI_4, 0.3, 22.0).as_dvec3();
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(eye.as_vec3()).looking_at(start_render.as_vec3(), Vec3::Y),
        Atmosphere::earthlike(scattering.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, eye)),
        AnchorCamera,
    ));
}

/// Renders the floating hull (pose from its `ActiveBody`) and publishes the WI 705 hydrostatic
/// gauges (draft / heel / net buoyancy) to the HUD.
fn track_hull(
    mut readout: ResMut<HarborReadout>,
    mut hull: Query<
        (
            &ActiveBody,
            &DivingCraft,
            &mut WorldPlacement,
            &mut Transform,
        ),
        With<HullMarker>,
    >,
) {
    let Ok((body, dc, mut wp, mut tf)) = hull.single_mut() else {
        return;
    };
    wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, render_world(body.position));
    tf.rotation = body.orientation.as_quat();

    let r = body.position.length();
    let g_local = if r > 0.0 { BODY.mu / (r * r) } else { 0.0 };
    let up = if r > 0.0 { body.position / r } else { DVec3::Y };
    let sample = FluidMedium::EARTHLIKE.sample_altitude(r - BODY.radius);
    let load = buoyancy_wrench(
        &dc.craft,
        dc.com,
        body.position,
        body.orientation,
        BODY.radius,
        0.0,
        sample.density,
        g_local,
        &dc.enclosed,
    );
    readout.draft = load.draft;
    readout.heel = heel_angle(body.orientation, up);
    readout.net_buoyancy = load.force.length() - body.mass * g_local;
}

/// Middle-drag orbit, wheel zoom (the editor/gallery convention).
fn harbor_camera_input(
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut cam: ResMut<HarborCam>,
) {
    if buttons.pressed(MouseButton::Middle) {
        cam.yaw -= motion.delta.x * 0.01; // drag-right swings the view right (L/R swapped per Dave)
        cam.pitch = (cam.pitch + motion.delta.y * 0.01).clamp(-1.3, 1.3);
    }
    if scroll.delta.y != 0.0 {
        cam.dist = (cam.dist - scroll.delta.y * cam.dist * 0.1).clamp(6.0, 600.0);
    }
}

/// Keeps the anchor camera orbiting the hull's render position.
#[allow(clippy::type_complexity)]
fn follow_camera(
    cam: Res<HarborCam>,
    hull: Query<&WorldPlacement, (With<HullMarker>, Without<AnchorCamera>)>,
    mut camera: Query<
        (&mut Transform, &mut WorldPlacement),
        (With<AnchorCamera>, Without<HullMarker>),
    >,
) {
    let Ok(hull_wp) = hull.single() else {
        return;
    };
    let Ok((mut tf, mut placement)) = camera.single_mut() else {
        return;
    };
    let target = hull_wp.0.pos;
    let eye = target + orbit_offset(cam.yaw, cam.pitch, cam.dist).as_dvec3();
    placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
    let look_dir = (target - eye).as_vec3().normalize_or_zero();
    if look_dir != Vec3::ZERO {
        tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
    }
}

fn update_hud(readout: Res<HarborReadout>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        text.0 = format!(
            "harbor\ndraft:    {:6.2} m\nheel:     {:6.1} deg\nnet buoy: {:8.0} N",
            readout.draft,
            readout.heel.to_degrees(),
            readout.net_buoyancy,
        );
    }
}

/// Follows the camera horizontally at sea level and ripples the calm patch each frame.
#[allow(clippy::type_complexity)]
fn animate_water(
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera: Query<&WorldPlacement, (With<AnchorCamera>, Without<WaterPatch>)>,
    mut patch: Query<(&Mesh3d, &mut WorldPlacement), (With<WaterPatch>, Without<AnchorCamera>)>,
) {
    let Ok(cam_wp) = camera.single() else {
        return;
    };
    let Ok((mesh3d, mut wp)) = patch.single_mut() else {
        return;
    };
    let c = cam_wp.0.pos;
    wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, DVec3::new(c.x, 0.0, c.z));
    let t = time.elapsed_secs();
    let Some(mesh) = meshes.get_mut(&mesh3d.0) else {
        return;
    };
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for p in positions.iter_mut() {
            p[1] = wave_height(p[0], p[2], t);
        }
    }
    mesh.compute_normals();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harbor_hull_is_sealed_and_floatable() {
        let hull = harbor_hull();
        // A sealed box encloses interior air (WI 711) — the reason it floats.
        let enclosed = enclosed_cells(&hull);
        assert!(!enclosed.is_empty(), "the hull encloses a sealed volume");
        // At the chosen mass fraction the hull is net-buoyant (reserve buoyancy > 0).
        let water = FluidMedium::EARTHLIKE.sample_altitude(-1.0).density;
        let displaceable = hull.occupied_volume() + enclosed.len() as f64 * hull.cell_volume();
        let mass = HULL_MASS_FRACTION * water * displaceable;
        assert!(
            water * displaceable > mass,
            "fully-submerged buoyancy exceeds weight ⇒ it floats"
        );
    }

    #[test]
    fn wave_height_is_calm_and_bounded() {
        // The amplitude is calm — well under the open-ocean dive's 0.55 m — and bounds the surface.
        const _: () = assert!(WATER_AMPLITUDE < 0.2);
        for &(x, z, t) in &[(0.0, 0.0, 0.0), (12.0, -7.0, 3.0), (-30.0, 40.0, 9.0)] {
            assert!(wave_height(x, z, t).abs() <= WATER_AMPLITUDE + 1e-6);
        }
    }

    #[test]
    fn orbit_offset_points_from_target_to_eye() {
        let v = orbit_offset(0.0, 0.0, 10.0);
        assert!((v.length() - 10.0).abs() < 1e-4);
    }
}

//! Harbor — a boat workshop on the water (`-- harbor`, WI 706 + 707).
//!
//! A self-contained **Build ↔ Float** loop (toggle with `Enter`), mirroring the grounded workshop's
//! Build ↔ Test:
//!
//! - **Build** (WI 707): the voxel editor (the `editor` module's systems under a state run-condition)
//!   edits a hull lattice near the origin — mouse orbit/zoom, left-click place / right-click remove,
//!   keyboard brush. The craft renders **solid** (`voxel_skin::skin_submeshes`).
//! - **Float** (WI 706/717): the built lattice is assembled into an `ActiveBody` + `DivingCraft` at its
//!   **real material mass** (WI 717 — no auto-ballast) and floats (or **sinks**) on calm water by a
//!   dock, self-righting via the WI 705 buoyancy wrench + WI 711 enclosed-volume buoyancy + WI 716 thin
//!   panels + free-surface damping (the same `DescentPlugin` → `glide_step` the dive uses). A light
//!   **panel** hull floats; a heavy/solid one sinks to the sea floor — the harbor is the proving ground.
//!
//! Build and Float are different coordinate worlds (Build near the origin with the editor camera;
//! Float in planetary coordinates with floating origin), so each spawns/despawns its own entities on
//! the toggle — they never coexist. Sealed-hull buoyancy (WI 711): a sealed hull floats; open-top
//! hulls await WI 713. Float camera: middle-drag orbit + wheel zoom; HUD shows draft / heel / net
//! buoyancy.

use std::collections::HashSet;
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
use sounding_sim::ballast::{Ballast, BallastCommand, BallastTank};
use sounding_sim::fluid::FluidMedium;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::marine::{MarinePropulsion, MarineThruster, Rudder, ThrusterCommand};
use sounding_sim::medium::{
    buoyancy_wrench, enclosed_cells, heel_angle, max_cross_section, DescentParams, DescentPlugin,
    DivingCraft, GlideParams, DEFAULT_SLAM_COEFFICIENT,
};
use sounding_sim::powertrain::MotorTier;
use sounding_sim::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Axis, Material, Voxel, VoxelCraft, PANEL_FILL};

use crate::editor::{
    draw_editor, editor_input, mouse_build, mouse_orbit_input, orbit_camera, update_hover, Brush,
    EditorState, HoverState, OrbitCam, PointerOnPalette,
};
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::voxel_skin::{materials_present, pbr_material, skin_submeshes, VoxelSkin};

const BODY: CentralBody = CentralBody::EARTHLIKE;

/// Float-integration sub-step and per-frame cap (real-time; the hull settles in a few seconds).
const SUBSTEP_DT: f64 = 0.002;
const MAX_SUBSTEPS: u32 = 64;

/// Depth of the harbor sea floor below sea level, m (WI 717): a sinking hull rests here rather than
/// falling toward the planet centre. Matches the rendered floor plane.
const SEA_FLOOR_DEPTH: f64 = 8.0;

/// The two harbor modes: float the built hull, or edit it.
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
enum HarborMode {
    /// Float the built hull on the water (the opening view).
    #[default]
    Float,
    /// Edit the hull lattice in the voxel editor.
    Build,
}

/// Sea level lies at `BODY.radius`; render space puts it at `Y = 0` (floating origin).
fn render_world(sim_pos: DVec3) -> DVec3 {
    sim_pos - DVec3::new(0.0, BODY.radius, 0.0)
}

/// The seed hull: a sealed **panel** pontoon (WI 716/717) — surface cells are thin aluminium **panels**
/// (light, so it floats honestly under real mass), enclosing air. Editor-scale 0.5 m cells ⇒ a
/// 3.5 × 2.5 × 5.5 m starter boat the player can reshape. The same hull in **solid** cubes would sink —
/// that is the game.
fn seed_hull() -> VoxelCraft {
    let mut c = VoxelCraft::new(0.5);
    let (w, h, l) = (7, 5, 11);
    for x in 0..w {
        for y in 0..h {
            for z in 0..l {
                let on_surface =
                    x == 0 || x == w - 1 || y == 0 || y == h - 1 || z == 0 || z == l - 1;
                if on_surface {
                    let cell = IVec3::new(x, y, z);
                    c.voxels.push(Voxel {
                        cell,
                        material: Material::ALUMINIUM,
                    });
                    c.set_panel(cell, true); // a thin hull plate, not a solid cube (WI 716)
                }
            }
        }
    }
    c
}

/// Mouse-driven orbit/zoom state for the Float camera (the editor/gallery convention).
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

/// The Float HUD readout (WI 705 hydrostatic gauges + WI 708 marine drive).
#[derive(Resource, Default)]
struct HarborReadout {
    draft: f64,
    heel: f64,
    net_buoyancy: f64,
    /// Net marine thrust, N (WI 708).
    thrust: f64,
    /// Fuel/charge fraction in `[0, 1]` (WI 708).
    fuel: f64,
    /// Ballast fill fraction in `[0, 1]` (WI 709).
    ballast: f64,
    /// Rudder deflection, radians (WI 725).
    rudder: f64,
}

// --- markers (teardown: tag only the root of each spawned tree) ---

/// Tags an entity owned by Float mode (despawned on leaving Float).
#[derive(Component)]
struct FloatEntity;

/// Tags an entity owned by Build mode (despawned on leaving Build).
#[derive(Component)]
struct BuildEntity;

/// Tags a solid mesh rendering part of the Build craft (rebuilt on edit).
#[derive(Component)]
struct BuildMesh;

/// The floating hull (its `ActiveBody` is the physics body).
#[derive(Component)]
struct HullMarker;

/// The Float HUD text.
#[derive(Component)]
struct Hud;

/// The Build HUD text.
#[derive(Component)]
struct BuildHud;

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

/// The harbor boat-workshop scene (`-- harbor`, WI 706 + 707).
pub struct HarborScenePlugin;

impl Plugin for HarborScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .init_state::<HarborMode>()
            .insert_resource(EditorState {
                craft: seed_hull(),
                cursor: IVec3::new(0, 3, 0),
                material: 0,
                brush: Brush::default(),
                subassembly: None,
                motor: MotorTier::Standard,
                panel_mode: false,
            })
            .init_resource::<OrbitCam>()
            .init_resource::<HoverState>()
            .init_resource::<PointerOnPalette>()
            .init_resource::<HarborReadout>()
            .init_resource::<HarborCam>()
            .insert_resource(Gravity { mu: BODY.mu })
            .add_plugins(DescentPlugin {
                substep_dt: SUBSTEP_DT,
                max_substeps: MAX_SUBSTEPS,
            })
            .add_systems(OnEnter(HarborMode::Float), enter_float)
            .add_systems(OnExit(HarborMode::Float), exit_float)
            .add_systems(OnEnter(HarborMode::Build), enter_build)
            .add_systems(OnExit(HarborMode::Build), exit_build)
            .add_systems(Update, toggle_mode)
            .add_systems(
                Update,
                (
                    harbor_drive_input,
                    harbor_ballast_input,
                    track_hull,
                    harbor_camera_input,
                    follow_camera,
                    update_hud,
                    animate_water,
                )
                    .chain()
                    .run_if(in_state(HarborMode::Float)),
            )
            .add_systems(
                Update,
                (
                    editor_input,
                    mouse_orbit_input,
                    update_hover,
                    mouse_build,
                    orbit_camera,
                    sync_build_meshes,
                    draw_editor,
                    update_build_hud,
                )
                    .chain()
                    .run_if(in_state(HarborMode::Build)),
            );
    }
}

/// `Enter` toggles Build ↔ Float.
fn toggle_mode(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<HarborMode>>,
    mut next: ResMut<NextState<HarborMode>>,
) {
    if keys.just_pressed(KeyCode::Enter) {
        next.set(match state.get() {
            HarborMode::Float => HarborMode::Build,
            HarborMode::Build => HarborMode::Float,
        });
    }
}

/// Builds the descent/glide parameters and the floating body for a hull lattice using its **real
/// material mass** (WI 717 — no auto-ballast): a light panel hull floats, a heavy / solid one sinks.
/// `None` for an empty lattice.
fn assemble_float(craft: &VoxelCraft) -> Option<(ActiveBody, DivingCraft)> {
    let mp = craft.mass_properties()?;
    if mp.mass <= 0.0 {
        return None;
    }
    let mass = mp.mass; // real mass — floating is earned, not granted
    let inertia = mp.inertia;
    let descent = DescentParams {
        medium: FluidMedium::EARTHLIKE,
        mu: BODY.mu,
        surface_radius: BODY.radius,
        drag_area: max_cross_section(craft),
        drag_coefficient: 1.0,
        slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
    };
    let glide = GlideParams::for_craft(descent, craft, Axis::Z);
    let start_sim = DVec3::new(0.0, BODY.radius, 0.0);
    let mut body = ActiveBody::new(start_sim, DVec3::ZERO, mass, inertia);
    body.orientation = DQuat::from_rotation_z(0.2); // a starting list, so the self-righting reads
    Some((
        body,
        DivingCraft::new(craft.clone(), mp.center_of_mass, glide),
    ))
}

/// A synthesized marine drive for a built hull (WI 708): a port + starboard screw pair low at the
/// stern (the −Z face; forward is +Z, the glide forward axis), mounted near the keel so they sit in
/// the water. Differential throttle steers (a yaw couple from the ±X offset). Sized to push the
/// editor-scale starter boat; player-placed thrusters await the WI 715 device palette.
fn synth_marine(craft: &VoxelCraft) -> MarinePropulsion {
    let cs = craft.cell_size;
    let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
    for v in &craft.voxels {
        lo = lo.min(v.cell);
        hi = hi.max(v.cell);
    }
    let min_m = lo.as_dvec3() * cs;
    let max_m = (hi.as_dvec3() + DVec3::ONE) * cs;
    let centre_x = 0.5 * (min_m.x + max_m.x);
    let beam = (max_m.x - min_m.x).max(cs);
    let stern_z = min_m.z + 0.5 * cs; // just inside the stern (the −Z end)
    let keel_y = min_m.y + 0.5 * cs; // near the bottom row, so the screws are submerged
    let off = 0.30 * beam;
    let screw = |x: f64| MarineThruster {
        tank: ReservoirId(0),
        max_thrust: 5_000.0,
        reference_density: 1_025.0, // water surface — full thrust submerged, ~none in air
        max_draw: 4.0,
        mount: DVec3::new(x, keel_y, stern_z),
        axis: DVec3::Z, // push forward (+Z)
    };
    MarinePropulsion {
        graph: ResourceGraph {
            reservoirs: vec![Reservoir::new(ResourceType(0), 1_500.0, 1_500.0)],
            ..Default::default()
        },
        thrusters: vec![screw(centre_x + off), screw(centre_x - off)],
        commands: vec![ThrusterCommand::default(); 2],
        last_thrust: 0.0,
    }
}

/// A synthesized rudder for a built hull (WI 725): a control surface aft at the stern (−Z, forward is
/// +Z), low so it sits in the water. Area scales with the hull's beam×draft so bigger boats get more
/// steering authority. Player-placed rudders await the WI 715 device palette.
fn synth_rudder(craft: &VoxelCraft) -> Rudder {
    let cs = craft.cell_size;
    let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
    for v in &craft.voxels {
        lo = lo.min(v.cell);
        hi = hi.max(v.cell);
    }
    let min_m = lo.as_dvec3() * cs;
    let max_m = (hi.as_dvec3() + DVec3::ONE) * cs;
    let beam = (max_m.x - min_m.x).max(cs);
    let depth = (max_m.y - min_m.y).max(cs);
    Rudder {
        mount: DVec3::new(
            0.5 * (min_m.x + max_m.x),
            min_m.y + 0.5 * cs,  // low, in the water
            min_m.z - 0.25 * cs, // just aft of the stern
        ),
        forward: DVec3::Z,
        area: 0.25 * beam * depth, // a fraction of the stern cross-section
        slope: 6.0,                // ~2π lift-curve slope per radian
        max_angle: 0.6,            // ~34° hard over
        angle: 0.0,
    }
}

/// A synthesized ballast tank for a built hull (WI 709): one tank low at the keel, sized so a full
/// flood **clearly overcomes the hull's reserve buoyancy** (≈1.5× the reserve), so flooding it sinks a
/// boat that floats empty and blowing it surfaces — controllable dive/surface/hold. `dry_mass` is
/// cached for the per-tick fold. `None` ⇒ no ballast (an empty/barely-floating hull). Player-placed
/// ballast awaits the WI 715 device palette.
fn synth_ballast(craft: &VoxelCraft) -> Option<Ballast> {
    let mp = craft.mass_properties()?;
    let g = 9.81;
    // The hull's reserve buoyancy (fully-submerged displaced weight − real weight), as a mass.
    let deep = DVec3::new(0.0, BODY.radius - 100.0, 0.0);
    let max_buoy = buoyancy_wrench(
        craft,
        mp.center_of_mass,
        deep,
        DQuat::IDENTITY,
        BODY.radius,
        0.0,
        1_025.0,
        g,
        &enclosed_cells(craft),
    )
    .force
    .length();
    let reserve_mass = ((max_buoy - mp.mass * g) / g).max(0.0);
    if reserve_mass <= 0.0 {
        return None; // already sinks; ballast is meaningless
    }
    let cap_volume = 1.5 * reserve_mass / 1_025.0; // m³ of water to clearly overcome the reserve
    let cs = craft.cell_size;
    let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
    for v in &craft.voxels {
        lo = lo.min(v.cell);
        hi = hi.max(v.cell);
    }
    let min_m = lo.as_dvec3() * cs;
    let max_m = (hi.as_dvec3() + DVec3::ONE) * cs;
    let mount = DVec3::new(
        0.5 * (min_m.x + max_m.x),
        min_m.y + 0.5 * cs, // low in the hull, so a flooded tank sinks the bow evenly and trims down
        0.5 * (min_m.z + max_m.z),
    );
    let rate = cap_volume / 6.0; // ~6 s to fully flood or blow
    Some(Ballast {
        tanks: vec![BallastTank {
            capacity: cap_volume,
            mount,
            fill: 0.0,
            fill_rate: rate,
            blow_rate: rate,
        }],
        command: BallastCommand::Hold,
        dry_mass: mp.mass,
    })
}

/// Enters Float: assemble the built hull and spawn the floating waterfront (WI 706 + 707).
fn enter_float(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    asset_server: Res<AssetServer>,
    editor: Res<EditorState>,
) {
    let start_render = render_world(DVec3::new(0.0, BODY.radius, 0.0));

    // The floating hull, assembled from the built lattice. If the build is empty, skip it (the
    // waterfront still renders; the player can return to Build).
    if let Some((body, dc)) = assemble_float(&editor.craft) {
        let com_off = -dc.com.as_vec3();
        let craft = dc.craft.clone();
        // A synthesized stern drive (WI 708): no device palette yet (WI 715), so any built hull
        // gets a port+starboard screw pair so it can sail and steer (differential thrust).
        let marine = synth_marine(&craft);
        // A synthesized ballast tank (WI 709): flood to dive, blow to surface (no palette yet, WI 715).
        let ballast = synth_ballast(&craft);
        // A synthesized rudder (WI 725): the primary underway steering surface.
        let rudder = synth_rudder(&craft);
        let mut hull = commands.spawn((
            body,
            dc,
            marine,
            rudder,
            Transform::default(),
            Visibility::default(),
            WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, start_render)),
            HullMarker,
            FloatEntity,
        ));
        if let Some(b) = ballast {
            hull.insert(b);
        }
        hull.with_children(|parent| {
            // Render the built hull centred on the CoM — solid cubes in their per-material skin,
            // thin panels as actual thin plates on the hull faces they form (WI 722).
            let (solid, _) = split_panels(&craft);
            for (material, mesh) in skin_submeshes(&solid, VoxelSkin::Hull) {
                let mat = pbr_material(material, &asset_server, &mut materials);
                parent.spawn((
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(mat),
                    Transform::from_translation(com_off),
                ));
            }
            let plates = plate_meshes(craft.cell_size, &mut meshes);
            let mats = material_handles(&craft, &asset_server, &mut materials);
            for (material, axis, pos) in panel_plate_specs(&craft) {
                let mat = mats
                    .iter()
                    .find(|(m, _)| *m == material)
                    .map(|(_, h)| h.clone());
                parent.spawn((
                    Mesh3d(plates[axis].clone()),
                    MeshMaterial3d(mat.unwrap_or_default()),
                    Transform::from_translation(pos + com_off),
                ));
            }
        });
    }

    // Distant ocean (sunk 2 m so it does not z-fight the animated patch).
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
        FloatEntity,
    ));

    // Shallow sea floor.
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
        FloatEntity,
    ));

    // Calm near-surface water patch at sea level.
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
        FloatEntity,
    ));

    // Dock / quay + pilings.
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
        FloatEntity,
    ));
    for z in [-6.0_f64, 6.0] {
        commands.spawn((
            Mesh3d(meshes.add(Mesh::from(Cuboid::new(0.5, 6.0, 0.5)))),
            MeshMaterial3d(quay_mat.clone()),
            Transform::default(),
            WorldPlacement(WorldPos::new(
                FrameId::CENTRAL_BODY,
                DVec3::new(-4.4, -2.5, z),
            )),
            FloatEntity,
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
        FloatEntity,
    ));

    // HUD.
    commands.spawn((
        Text::new("harbor — Float (Enter: Build)\nWASD: drive / steer   F/G: dive / surface\ndraft:  --\nheel:   --\nnet buoy: --\nthrust:  --   fuel: --\nballast: --   rudder: --"),
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
        FloatEntity,
    ));

    // HDR camera with the atmosphere, orbiting the hull.
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
        FloatEntity,
    ));
}

fn exit_float(mut commands: Commands, entities: Query<Entity, With<FloatEntity>>) {
    for e in &entities {
        commands.entity(e).despawn();
    }
}

/// Enters Build: the editor camera + light + build HUD near the origin (the craft meshes are spawned
/// by `sync_build_meshes`).
fn enter_build(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Transform::default(), BuildEntity));
    commands.spawn((
        DirectionalLight {
            illuminance: 6_000.0,
            ..default()
        },
        Transform::from_xyz(6.0, 12.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        BuildEntity,
    ));
    commands.spawn((
        Text::new(
            "harbor — Build (Enter: Float)\nmouse: orbit/zoom · L-click place · R-click remove\nT: panel mode (thin, light) · Tab: material",
        ),
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
        BuildHud,
        BuildEntity,
    ));
}

fn exit_build(mut commands: Commands, entities: Query<Entity, With<BuildEntity>>) {
    // Build meshes carry `BuildEntity` too, so this clears the whole Build world.
    for e in &entities {
        commands.entity(e).despawn();
    }
}

/// Rebuilds the solid Build-craft meshes whenever the lattice changes (or after re-entering Build),
/// reusing the shared `voxel_skin` hull skin (per-material sub-meshes), near the origin.
fn sync_build_meshes(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    editor: Res<EditorState>,
    existing: Query<Entity, With<BuildMesh>>,
) {
    if !editor.is_changed() && !existing.is_empty() {
        return;
    }
    for e in &existing {
        commands.entity(e).despawn();
    }
    // Solid cubes render via the per-material hull skin; thin panels render as actual thin plates on
    // the hull faces they form (WI 722).
    let (solid, _) = split_panels(&editor.craft);
    for (material, mesh) in skin_submeshes(&solid, VoxelSkin::Hull) {
        let mat = pbr_material(material, &asset_server, &mut materials);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(mat),
            Transform::default(),
            BuildMesh,
            BuildEntity,
        ));
    }
    let plates = plate_meshes(editor.craft.cell_size, &mut meshes);
    let mats = material_handles(&editor.craft, &asset_server, &mut materials);
    for (material, axis, pos) in panel_plate_specs(&editor.craft) {
        let mat = mats
            .iter()
            .find(|(m, _)| *m == material)
            .map(|(_, h)| h.clone());
        commands.spawn((
            Mesh3d(plates[axis].clone()),
            MeshMaterial3d(mat.unwrap_or_default()),
            Transform::from_translation(pos),
            BuildMesh,
            BuildEntity,
        ));
    }
}

/// Partition a craft's voxels into a **solid-cube** craft and a **thin-panel** craft (WI 719), so each
/// can render with its own material. Hull-skin only (devices/parts unneeded here).
fn split_panels(craft: &VoxelCraft) -> (VoxelCraft, VoxelCraft) {
    let mut solid = VoxelCraft::new(craft.cell_size);
    let mut panel = VoxelCraft::new(craft.cell_size);
    for v in &craft.voxels {
        if craft.is_panel(v.cell) {
            panel.voxels.push(*v);
        } else {
            solid.voxels.push(*v);
        }
    }
    (solid, panel)
}

/// The empty cells that are **outside** the hull (WI 723): a flood-fill from the expanded bounding-box
/// border through empty cells, mirroring `compartments` — so the enclosed cavity is *not* included.
/// Used to plate only the true outer surface.
fn exterior_empty_cells(craft: &VoxelCraft) -> HashSet<IVec3> {
    let occupied: HashSet<IVec3> = craft.voxels.iter().map(|v| v.cell).collect();
    if occupied.is_empty() {
        return HashSet::new();
    }
    let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
    for &c in &occupied {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    lo -= IVec3::ONE;
    hi += IVec3::ONE;
    let in_bbox = |c: IVec3| {
        c.x >= lo.x && c.x <= hi.x && c.y >= lo.y && c.y <= hi.y && c.z >= lo.z && c.z <= hi.z
    };
    let is_air = |c: IVec3| in_bbox(c) && !occupied.contains(&c);
    let neighbours = [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ];
    let mut exterior = HashSet::new();
    let mut stack = Vec::new();
    for x in lo.x..=hi.x {
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                let c = IVec3::new(x, y, z);
                let border =
                    x == lo.x || x == hi.x || y == lo.y || y == hi.y || z == lo.z || z == hi.z;
                if border && is_air(c) && exterior.insert(c) {
                    stack.push(c);
                }
            }
        }
    }
    while let Some(c) = stack.pop() {
        for off in neighbours {
            let n = c + off;
            if is_air(n) && exterior.insert(n) {
                stack.push(n);
            }
        }
    }
    exterior
}

/// A thin-plate render spec for the hull's **outer surface** (WI 723): `(material, thin-axis, centre)`
/// in lattice coords — a plate on each panel-cell face whose neighbour is **exterior** (outside the
/// hull, not the enclosed cavity), seated flush on that grid boundary. Outer faces of a wall are
/// coplanar/contiguous ⇒ one coherent thin shell (no exploded box); the cavity is not double-plated.
fn panel_plate_specs(craft: &VoxelCraft) -> Vec<(Material, usize, Vec3)> {
    let exterior = exterior_empty_cells(craft);
    let cs = craft.cell_size;
    let inset = 0.5 * cs * (1.0 - PANEL_FILL); // seat flush against the exterior grid boundary
    let dirs: [(IVec3, usize); 6] = [
        (IVec3::X, 0),
        (IVec3::NEG_X, 0),
        (IVec3::Y, 1),
        (IVec3::NEG_Y, 1),
        (IVec3::Z, 2),
        (IVec3::NEG_Z, 2),
    ];
    let mut out = Vec::new();
    for v in &craft.voxels {
        if !craft.is_panel(v.cell) {
            continue;
        }
        let centre = (v.cell.as_dvec3() + DVec3::splat(0.5)) * cs;
        for (off, axis) in dirs {
            if exterior.contains(&(v.cell + off)) {
                let pos = centre + off.as_dvec3() * inset;
                out.push((v.material, axis, pos.as_vec3()));
            }
        }
    }
    out
}

/// The three thin-plate meshes (X/Y/Z thin) for a cell size, reused across all panel plates (WI 722).
fn plate_meshes(cell_size: f64, meshes: &mut Assets<Mesh>) -> [Handle<Mesh>; 3] {
    let s = cell_size as f32;
    let t = (cell_size * PANEL_FILL) as f32;
    [
        meshes.add(Mesh::from(Cuboid::new(t, s, s))),
        meshes.add(Mesh::from(Cuboid::new(s, t, s))),
        meshes.add(Mesh::from(Cuboid::new(s, s, t))),
    ]
}

/// One reusable PBR material handle per distinct structural material in the craft (WI 722), so the many
/// panel plates don't each allocate a material.
fn material_handles(
    craft: &VoxelCraft,
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
) -> Vec<(Material, Handle<StandardMaterial>)> {
    materials_present(craft)
        .into_iter()
        .map(|m| (m, pbr_material(m, asset_server, materials)))
        .collect()
}

/// Will this craft float, and at what draft? (WI 720, the Build-mode predictor.) Net buoyancy is the
/// fully-submerged displaced weight vs the real weight — the **same** model the Float mode runs, so
/// the prediction agrees with the live outcome. Returns `(floats, draft_fraction)`; draft is the
/// equilibrium submerged fraction (1.0 ⇒ awash / sinking).
fn would_float(craft: &VoxelCraft) -> (bool, f64) {
    let Some(mp) = craft.mass_properties() else {
        return (false, 1.0);
    };
    if mp.mass <= 0.0 {
        return (false, 1.0);
    }
    let g = 9.81;
    let deep = DVec3::new(0.0, BODY.radius - 100.0, 0.0); // fully submerged ⇒ maximum buoyancy
    let max_buoy = buoyancy_wrench(
        craft,
        mp.center_of_mass,
        deep,
        DQuat::IDENTITY,
        BODY.radius,
        0.0,
        1_025.0,
        g,
        &enclosed_cells(craft),
    )
    .force
    .length();
    let weight = mp.mass * g;
    (max_buoy > weight, (weight / max_buoy).clamp(0.0, 1.0))
}

fn update_build_hud(editor: Res<EditorState>, mut hud: Query<&mut Text, With<BuildHud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let cells = editor.craft.voxels.len();
        let panels = editor.craft.panels.len();
        let mode = if editor.panel_mode {
            "PANEL (thin)"
        } else {
            "solid"
        };
        // Live float prediction (WI 720): the same buoyancy model the Float mode runs.
        let (floats, draft) = would_float(&editor.craft);
        let prediction = if floats {
            format!("will FLOAT (draft ~{:.0}%)", draft * 100.0)
        } else {
            "will SINK".to_string()
        };
        text.0 = format!(
            "harbor — Build (Enter: Float)\nmouse: orbit/zoom · L-click place · R-click remove\nT: place mode — {mode} · Tab: material\ncells: {cells}  panels: {panels}\n>> {prediction} <<"
        );
    }
}

/// Renders the floating hull (pose from its `ActiveBody`) and publishes the WI 705 gauges.
#[allow(clippy::type_complexity)]
fn track_hull(
    mut readout: ResMut<HarborReadout>,
    mut hull: Query<
        (
            &mut ActiveBody,
            &DivingCraft,
            &mut WorldPlacement,
            &mut Transform,
        ),
        With<HullMarker>,
    >,
) {
    let Ok((mut body, dc, mut wp, mut tf)) = hull.single_mut() else {
        return;
    };

    // Rest a sinking hull on the sea floor (WI 717): a net-negative hull descends and grounds here
    // rather than falling toward the planet centre. Only engages well below the waterline, so a
    // floating (or merely low) hull is untouched.
    if body.position.length() - BODY.radius < -SEA_FLOOR_DEPTH {
        let up = body.position.normalize_or(DVec3::Y);
        body.position = up * (BODY.radius - SEA_FLOOR_DEPTH);
        let into_floor = body.velocity.dot(up).min(0.0); // downward component
        body.velocity -= into_floor * up;
    }

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

/// Player marine drive + steering (WI 708 + 725): `W`/`S` throttle forward/reverse, `A`/`D` steer.
/// Steering deflects the **rudder** (the primary underway control — yaw grows with speed) and also
/// biases **differential thrust** (so the boat still pivots at low speed / standstill). Publishes net
/// thrust, fuel, and rudder angle to the HUD readout.
#[allow(clippy::type_complexity)]
fn harbor_drive_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut readout: ResMut<HarborReadout>,
    mut hull: Query<(&mut MarinePropulsion, &mut Rudder), With<HullMarker>>,
) {
    let Ok((mut mp, mut rudder)) = hull.single_mut() else {
        return;
    };
    let mut forward = 0.0;
    let mut turn = 0.0;
    if keys.pressed(KeyCode::KeyW) {
        forward += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        forward -= 1.0;
    }
    // In the +Z-forward / +Y-up hull frame a positive `turn` yaws the bow to **port**, so `D` (steer
    // right) contributes negative turn and `A` (steer left) positive — matching the player's L/R.
    if keys.pressed(KeyCode::KeyD) {
        turn -= 1.0; // bow to starboard (right)
    }
    if keys.pressed(KeyCode::KeyA) {
        turn += 1.0; // bow to port (left)
    }
    rudder.set_turn(turn); // primary steering — yaw scales with speed
    mp.drive(forward, turn); // differential thrust — low-speed pivoting
    readout.thrust = mp.last_thrust;
    readout.fuel = mp.fuel_fraction();
    readout.rudder = rudder.angle;
}

/// Player ballast control (WI 709): hold `F` to **flood** (dive), hold `G` to **blow** (surface),
/// release both to **hold** the current fill (hold depth). Publishes fill fraction to the HUD.
fn harbor_ballast_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut readout: ResMut<HarborReadout>,
    mut hull: Query<&mut Ballast, With<HullMarker>>,
) {
    let Ok(mut b) = hull.single_mut() else {
        return;
    };
    b.command = if keys.pressed(KeyCode::KeyF) {
        BallastCommand::Fill
    } else if keys.pressed(KeyCode::KeyG) {
        BallastCommand::Blow
    } else {
        BallastCommand::Hold
    };
    readout.ballast = b.fill_fraction();
}

/// Middle-drag orbit (L/R swapped per Dave), wheel zoom.
fn harbor_camera_input(
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut cam: ResMut<HarborCam>,
) {
    if buttons.pressed(MouseButton::Middle) {
        cam.yaw -= motion.delta.x * 0.01;
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
        // Net buoyancy is the displaced weight minus the hull weight: ~0 floating at its waterline,
        // strongly negative when fully submerged and unable to float (WI 717).
        let status = if readout.net_buoyancy < -500.0 {
            "SINKING"
        } else {
            "floating"
        };
        text.0 = format!(
            "harbor — Float (Enter: Build) — {status}\nWASD: drive / steer   F/G: dive / surface\ndraft:    {:6.2} m\nheel:     {:6.1} deg\nnet buoy: {:8.0} N\nthrust:   {:8.0} N   fuel:    {:3.0}%\nballast:  {:3.0}%   rudder: {:5.1} deg",
            readout.draft,
            readout.heel.to_degrees(),
            readout.net_buoyancy,
            readout.thrust,
            readout.fuel * 100.0,
            readout.ballast * 100.0,
            readout.rudder.to_degrees(),
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
    use sounding_sim::medium::enclosed_cells;

    /// Net buoyancy of a craft fully submerged (max buoyancy − real weight): >0 floats, <0 sinks.
    fn net_buoyancy(craft: &VoxelCraft) -> f64 {
        let mp = craft.mass_properties().unwrap();
        let enc = enclosed_cells(craft);
        let g = 9.81;
        let deep = DVec3::new(0.0, BODY.radius - 100.0, 0.0);
        let buoy = buoyancy_wrench(
            craft,
            mp.center_of_mass,
            deep,
            DQuat::IDENTITY,
            BODY.radius,
            0.0,
            1_025.0,
            g,
            &enc,
        )
        .force
        .length();
        buoy - mp.mass * g
    }

    /// WI 717: the **panel** seed hull floats under **real mass**, while the same hull in **solid
    /// cubes** sinks — the proving-ground contrast (no auto-ballast).
    #[test]
    fn the_panel_seed_floats_but_a_solid_hull_sinks() {
        let seed = seed_hull();
        assert!(!seed.panels.is_empty(), "the seed is built of panels");
        assert!(net_buoyancy(&seed) > 0.0, "the panel seed hull floats");

        let mut solid = seed.clone();
        solid.panels.clear(); // same geometry, solid cubes
        assert!(
            net_buoyancy(&solid) < 0.0,
            "the same hull in solid cubes sinks"
        );
    }

    /// WI 719: `split_panels` partitions voxels into solid + panel sets for distinct rendering.
    #[test]
    fn split_panels_partitions_solid_and_panel_cells() {
        // The all-panel seed: solid set empty, panel set = the whole hull.
        let seed = seed_hull();
        let (solid, panel) = split_panels(&seed);
        assert!(solid.voxels.is_empty());
        assert_eq!(panel.voxels.len(), seed.voxels.len());

        // A no-panel craft: everything is in the solid set (renders unchanged).
        let mut plain = seed.clone();
        plain.panels.clear();
        let (s2, p2) = split_panels(&plain);
        assert_eq!(s2.voxels.len(), plain.voxels.len());
        assert!(p2.voxels.is_empty());
    }

    /// WI 723: the exterior flood-fill excludes the enclosed cavity (so only the outer surface plates).
    #[test]
    fn exterior_flood_excludes_the_cavity() {
        let seed = seed_hull(); // a 7×5×11 sealed shell
        let ext = exterior_empty_cells(&seed);
        assert!(
            ext.contains(&IVec3::new(-1, 2, 5)),
            "outside the hull is exterior"
        );
        assert!(
            !ext.contains(&IVec3::new(3, 2, 5)),
            "the enclosed cavity is NOT exterior"
        );
        assert!(
            !ext.contains(&IVec3::new(0, 2, 5)),
            "a solid wall cell is not air"
        );
    }

    /// WI 723: plates seat on the hull's **outer** faces only — coherent thin shell, no cavity
    /// double-plating; no panels ⇒ no plates.
    #[test]
    fn panel_plate_specs_plates_outer_faces() {
        // A lone panel cell: all 6 neighbours are exterior ⇒ 6 plates (a fully-skinned cube).
        let mut c = VoxelCraft::new(0.5);
        c.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        c.set_panel(IVec3::ZERO, true);
        assert_eq!(panel_plate_specs(&c).len(), 6);

        // No panels ⇒ no plates.
        let mut solid = c.clone();
        solid.panels.clear();
        assert!(panel_plate_specs(&solid).is_empty());

        // The seed shell: at least one outer plate per wall cell, but far fewer than all 6 faces
        // (internal + cavity faces are excluded — the fix for the exploded box).
        let seed = seed_hull();
        let n = panel_plate_specs(&seed).len();
        assert!(n >= seed.voxels.len());
        assert!(n < 6 * seed.voxels.len());
    }

    /// WI 720: the predictor agrees with the float — the panel seed predicts FLOAT (draft < 1), the
    /// same hull in solid cubes predicts SINK.
    #[test]
    fn would_float_predicts_panel_floats_solid_sinks() {
        let seed = seed_hull();
        let (floats, draft) = would_float(&seed);
        assert!(floats, "the panel seed is predicted to float");
        assert!(draft > 0.0 && draft < 1.0, "with a real draft: {draft}");

        let mut solid = seed.clone();
        solid.panels.clear();
        assert!(
            !would_float(&solid).0,
            "the same hull in solid cubes is predicted to sink"
        );

        // Empty craft: predicted to sink, no panic.
        assert!(!would_float(&VoxelCraft::new(0.5)).0);
    }

    /// WI 717: the float body carries the craft's **real** mass (no auto-ballast).
    #[test]
    fn assemble_float_uses_real_mass() {
        let seed = seed_hull();
        let (body, _) = assemble_float(&seed).unwrap();
        let real = seed.mass_properties().unwrap().mass;
        assert!(
            (body.mass - real).abs() < 1e-9,
            "float body uses real mass: {} vs {real}",
            body.mass
        );
    }

    #[test]
    fn an_empty_lattice_does_not_assemble() {
        assert!(assemble_float(&VoxelCraft::new(0.5)).is_none());
    }

    /// WI 708: the synthesized stern drive sits in the water on the floated seed hull and pushes it
    /// **forward** (+Z), drawing fuel — and steers (a yaw couple) under differential throttle.
    #[test]
    fn synth_marine_drives_the_seed_hull_forward_and_steers() {
        let seed = seed_hull();
        let (body, dc) = assemble_float(&seed).unwrap();
        let mut mp = synth_marine(&seed);
        let fuel0 = mp.fuel();

        // Full ahead: forward thrust, fuel drawn.
        mp.drive(1.0, 0.0);
        let (f, tq) = mp.thrust_step(
            &FluidMedium::EARTHLIKE,
            BODY.radius,
            body.position,
            body.orientation,
            dc.com,
            0.1,
        );
        assert!(
            f.length() > 0.0,
            "the keel screws are submerged and push: {f:?}"
        );
        // Forward is +Z in the body frame; the world thrust is mostly along the hull's forward axis.
        let forward_world = body.orientation * DVec3::Z;
        assert!(f.dot(forward_world) > 0.0, "drives forward");
        assert!(mp.fuel() < fuel0, "fuel drawn under power");
        assert!(tq.length() >= 0.0);

        // Differential throttle yaws (nonzero steering moment about up).
        mp.drive(0.4, 0.6);
        let (_f2, tq2) = mp.thrust_step(
            &FluidMedium::EARTHLIKE,
            BODY.radius,
            body.position,
            body.orientation,
            dc.com,
            0.1,
        );
        let up = body.position.normalize_or(DVec3::Y);
        assert!(
            tq2.dot(up).abs() > 1e-3,
            "differential thrust steers: {tq2:?}"
        );
    }

    /// WI 725: the synthesized rudder sits in the water aft of the hull and steers it **only when
    /// moving** — a yaw moment under way, nothing at a standstill.
    #[test]
    fn synth_rudder_steers_the_moving_seed_hull_only_under_way() {
        let seed = seed_hull();
        let (body, dc) = assemble_float(&seed).unwrap();
        let mut r = synth_rudder(&seed);
        r.set_turn(1.0); // hard over
        let m = FluidMedium::EARTHLIKE;

        // At rest: no flow ⇒ no steering.
        let (_, t_rest) = r.wrench(
            &m,
            BODY.radius,
            body.position,
            body.orientation,
            DVec3::ZERO,
            dc.com,
        );
        assert_eq!(t_rest, DVec3::ZERO, "no steering at a standstill");

        // Under way (forward, the hull's +Z): a real yaw moment about the local up.
        let forward = body.orientation * DVec3::Z;
        let (_, t_move) = r.wrench(
            &m,
            BODY.radius,
            body.position,
            body.orientation,
            forward * 5.0,
            dc.com,
        );
        let up = body.position.normalize_or(DVec3::Y);
        assert!(
            t_move.dot(up).abs() > 1e-3,
            "the rudder yaws the moving hull: {t_move:?}"
        );
    }

    /// WI 709: flooding the synthesized ballast flips the seed hull's net buoyancy negative (it
    /// dives), and blowing it recovers positive net buoyancy (it surfaces) — controllable and
    /// reversible.
    #[test]
    fn ballast_flips_net_buoyancy_dive_and_surface() {
        let seed = seed_hull();
        let mut b = synth_ballast(&seed).expect("the panel hull has reserve buoyancy");
        let g = 9.81;
        let enc = enclosed_cells(&seed);
        let mp = seed.mass_properties().unwrap();
        let deep = DVec3::new(0.0, BODY.radius - 100.0, 0.0);
        let buoy = buoyancy_wrench(
            &seed,
            mp.center_of_mass,
            deep,
            DQuat::IDENTITY,
            BODY.radius,
            0.0,
            1_025.0,
            g,
            &enc,
        )
        .force
        .length();
        let net = |ballast: &Ballast| buoy - ballast.wet_mass(mp.center_of_mass, 1_025.0).mass * g;

        // Blown (empty): the panel hull floats.
        assert!(net(&b) > 0.0, "blown ballast ⇒ floats");
        // Flood it fully: it sinks.
        b.command = BallastCommand::Fill;
        b.step(100.0);
        assert!(net(&b) < 0.0, "flooded ballast ⇒ sinks (dives)");
        // Blow it again: it floats once more (reversible).
        b.command = BallastCommand::Blow;
        b.step(100.0);
        assert!(net(&b) > 0.0, "blown again ⇒ floats (reversible)");
    }

    #[test]
    fn wave_height_is_calm_and_bounded() {
        const _: () = assert!(WATER_AMPLITUDE < 0.2);
        for &(x, z, t) in &[(0.0, 0.0, 0.0), (12.0, -7.0, 3.0), (-30.0, 40.0, 9.0)] {
            assert!(wave_height(x, z, t).abs() <= WATER_AMPLITUDE + 1e-6);
        }
    }
}

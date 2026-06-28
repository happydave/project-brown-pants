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
use sounding_sim::powertrain::MotorTier;
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Axis, Material, Voxel, VoxelCraft};

use crate::editor::{
    draw_editor, editor_input, mouse_build, mouse_orbit_input, orbit_camera, update_hover, Brush,
    EditorState, HoverState, OrbitCam, PointerOnPalette,
};
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::voxel_skin::{pbr_material, skin_submeshes, VoxelSkin};

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

/// The Float HUD readout (WI 705 hydrostatic gauges).
#[derive(Resource, Default)]
struct HarborReadout {
    draft: f64,
    heel: f64,
    net_buoyancy: f64,
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
        commands
            .spawn((
                body,
                dc,
                Transform::default(),
                Visibility::default(),
                WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, start_render)),
                HullMarker,
                FloatEntity,
            ))
            .with_children(|parent| {
                // Render the actual built hull centred on the CoM — solid cubes in their per-material
                // skin, thin panels in the distinct panel material (WI 719).
                let (solid, panel) = split_panels(&craft);
                for (material, mesh) in skin_submeshes(&solid, VoxelSkin::Hull) {
                    let mat = pbr_material(material, &asset_server, &mut materials);
                    parent.spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(mat),
                        Transform::from_translation(com_off),
                    ));
                }
                let pmat = panel_material(&mut materials);
                for (_m, mesh) in skin_submeshes(&panel, VoxelSkin::Hull) {
                    parent.spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(pmat.clone()),
                        Transform::from_translation(com_off),
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
        Text::new("harbor — Float (Enter: Build)\ndraft:  --\nheel:   --\nnet buoy: --"),
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
    // Render solid cubes with their per-material skin, thin panels in the distinct panel material (WI 719).
    let (solid, panel) = split_panels(&editor.craft);
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
    let pmat = panel_material(&mut materials);
    for (_m, mesh) in skin_submeshes(&panel, VoxelSkin::Hull) {
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(pmat.clone()),
            Transform::default(),
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

/// A distinct **panel** material (WI 719): a cool plate blue-grey, clearly different from the metallic
/// solid-cube skin, so thin panels read at a glance.
fn panel_material(materials: &mut Assets<StandardMaterial>) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: Color::srgb(0.42, 0.60, 0.72),
        metallic: 0.2,
        perceptual_roughness: 0.5,
        ..default()
    })
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
            "harbor — Float (Enter: Build) — {status}\ndraft:    {:6.2} m\nheel:     {:6.1} deg\nnet buoy: {:8.0} N",
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

    #[test]
    fn wave_height_is_calm_and_bounded() {
        const _: () = assert!(WATER_AMPLITUDE < 0.2);
        for &(x, z, t) in &[(0.0, 0.0, 0.0), (12.0, -7.0, 3.0), (-30.0, 40.0, 9.0)] {
            assert!(wave_height(x, z, t).abs() <= WATER_AMPLITUDE + 1e-6);
        }
    }
}

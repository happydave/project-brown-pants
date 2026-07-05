//! Harbor — a boat workshop on the water (`-- harbor`, WI 706 + 707), its
//! starting hull **from scenario data** since WI 739.
//!
//! A self-contained **Build ↔ Float** loop (toggle with `Enter`), mirroring the grounded workshop's
//! Build ↔ Test:
//!
//! - **Build** (WI 707): the voxel editor (the `editor` module's systems under a state run-condition)
//!   edits a hull lattice near the origin — mouse orbit/zoom, left-click place / right-click remove,
//!   keyboard brush. The craft renders **solid** (`voxel_skin::skin_submeshes`).
//! - **Float** (WI 706/717, spawned through the scenario director since WI 739): entering Float
//!   stages the live build as an **Afloat** scenario spawn — the sim-side assembly
//!   (`sounding_sim::afloat`) builds the `ActiveBody` + `DivingCraft` at its **real material mass**
//!   (WI 717 — no auto-ballast) with the synthesized drive/rudder/ballast and the interior-flood
//!   physics, and it floats (or **sinks**) on calm water by a dock, self-righting via the WI 705
//!   buoyancy wrench + WI 711 enclosed-volume buoyancy + WI 716 thin panels + free-surface damping
//!   (the same `DescentPlugin` → `glide_step` the dive uses). A light **panel** hull floats; a
//!   heavy/solid one sinks to the sea floor — the harbor is the proving ground.
//!
//! **The content comes from `content/scenarios/harbor.ron`** (or an explicit `-- harbor <path>`):
//! the seed hull is its referenced blueprint, loaded into the editor at startup. This scene keeps
//! Build (the shared editor) and the waterfront presentation; the Float physics assembly and the
//! flood stepping live in the sim.
//!
//! Build and Float are different coordinate worlds (Build near the origin with the editor camera;
//! Float in planetary coordinates with floating origin), so each spawns/despawns its own entities on
//! the toggle — they never coexist. Float camera: middle-drag orbit + wheel zoom; HUD shows draft /
//! heel / net buoyancy.

use std::f32::consts::FRAC_PI_4;

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::{DQuat, DVec3};
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;

use sounding_sim::active::{ActiveBody, Gravity};
use sounding_sim::afloat::{FloodComps, ScenarioVessel};
use sounding_sim::ballast::{Ballast, BallastCommand};
use sounding_sim::command::Command;
use sounding_sim::director::{DirectorPlugin, PendingSpawn, ScenarioSpawn};
use sounding_sim::fluid::FluidMedium;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::marine::{MarinePropulsion, Rudder};
use sounding_sim::medium::{
    buoyancy_wrench, enclosed_cells, heel_angle, open_cavity, open_cavity_load,
    open_cavity_rim_factor, DescentPlugin, DivingCraft,
};
use sounding_sim::powertrain::MotorTier;
use sounding_sim::scenario::{load_scenario, ScenarioRoots, StartPlacement};
use sounding_sim::sim::CentralBody;
#[cfg(test)]
use sounding_sim::voxel::Material;
use sounding_sim::voxel::VoxelCraft;

use crate::build::{self, BuildEntity, BuildHud, BuildMesh};
use crate::editor::{
    editor_input, mouse_build, mouse_orbit_input, orbit_camera, update_hover, Brush, EditorState,
    HoverState, OrbitCam, PointerOnPalette,
};
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::scene_cam::{self, OrbitFollowCam};
use crate::scene_water::{self, WaterPatch, WaveSpec};
use crate::voxel_skin::{panel_render_pieces, pbr_material, skin_submeshes, VoxelSkin};

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

/// The default scenario document when `-- harbor` is given no path (WI 739).
/// The seed hull — a sealed panel pontoon that floats honestly under real
/// mass — is this document's referenced blueprint, not code.
const DEFAULT_SCENARIO: &str = "content/scenarios/harbor.ron";

/// The loaded scenario's resolved spawn payload (WI 739), kept as the Float
/// template: entering Float re-stages it with the **live build** as the
/// craft, so the director assembles exactly what the editor holds.
#[derive(Resource)]
struct HarborTemplate(ScenarioSpawn);

/// The harbor's Float follow-camera config (WI 714): the editor/gallery orbit convention, with the
/// harbor's inverted horizontal drag and tighter zoom range.
fn harbor_cam() -> OrbitFollowCam {
    OrbitFollowCam {
        yaw: FRAC_PI_4,
        pitch: 0.3,
        dist: 22.0,
        yaw_sign: -1.0,
        pitch_limit: 1.3,
        dist_min: 6.0,
        dist_max: 600.0,
    }
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
    /// Overall interior flooded fraction in `[0, 1]` (WI 718).
    flood: f64,
}

/// The harbor's interior-water **render manifest** (WI 718 + 728 + 709, unified by WI 729; the
/// flood *physics* moved to `sounding_sim::afloat` in WI 739): every interior volume that can hold
/// water — sealed compartment, open cavity, or ballast tank — each with the source its fill level
/// comes from. One renderer draws them all (occluder + rising surface), so a new water source is
/// data (a new region), not a new render block.
#[derive(Resource, Default)]
struct FloodState {
    /// Every interior volume that can hold water, with its fill source (WI 729). The unified renderer
    /// spawns one occluder + one rising-water cuboid per region and raises it from `source`.
    regions: Vec<InteriorWaterRegion>,
}

/// One interior volume that can hold water (WI 729): an AABB to fill — hull-local, CoM-offset render
/// metres — plus the [`WaterSource`] its fill fraction is read from each frame. The unifying abstraction:
/// the renderer treats sealed flooding, open swamping, and ballast identically.
struct InteriorWaterRegion {
    min: Vec3,
    max: Vec3,
    source: WaterSource,
}

/// Where an [`InteriorWaterRegion`]'s fill fraction comes from (WI 729).
#[derive(Clone, Copy)]
enum WaterSource {
    /// A sealed compartment (WI 718): fill = its WI 520 flooded fraction (breach-driven).
    Sealed { comp: usize },
    /// The open cavity (WI 728): fill = `1 − open_cavity_rim_factor` (swamp-driven).
    Open,
    /// A ballast tank (WI 709): fill = the tank's fill fraction (player flood/blow) — so ballast water is
    /// finally visible in its tank, not just a HUD number.
    Ballast { tank: usize },
}

/// Tags a translucent rising-water cuboid, indexing [`FloodState::regions`] (WI 729). One component for
/// every source (sealed / open / ballast), replacing the per-source `FloodWater`/`OpenWater`.
#[derive(Component)]
struct InteriorWater(usize);

/// How much of an **open** boat's interior the dry-hold occluder fills, bottom-up (WI 734). The cap sits
/// just above the typical floating waterline so it still hides the sea-level water plane, while the top is
/// left open so you can see down into the real hollow interior (a recess, not a flush grey lid). Sealed
/// compartments and ballast tanks fill fully — their interiors are genuinely closed.
const OPEN_HOLD_FILL: f32 = 0.72;

/// The vertical scale + centre of a rising-water cuboid (WI 729): a unit cube scaled to `frac` of a
/// region's vertical `extent`, its base pinned at `min_y` so the surface rises from the floor. Returns
/// `(scale_y, translation_y)`. Pure — the testable core of the interior-water render raise.
fn fill_cuboid_y(min_y: f32, extent: f32, frac: f32) -> (f32, f32) {
    let h = (frac.clamp(0.0, 1.0) * extent).max(0.0001);
    (h, min_y + 0.5 * h)
}

/// The hull-local CoM-offset render AABB of a set of cells (WI 729): the cell lattice ∩ extents mapped to
/// metres, used to size an interior-water region's occluder + rising surface.
fn cell_aabb(cells: &[IVec3], cs: f64, com_off: Vec3) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
    for &p in cells {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    (
        (lo.as_dvec3() * cs).as_vec3() + com_off,
        ((hi.as_dvec3() + DVec3::ONE) * cs).as_vec3() + com_off,
    )
}

/// Build the harbor interior-water **render manifest** (WI 718/728/709, unified by WI 729; physics
/// sim-side since WI 739): one region per interior volume that can hold water — sealed compartments
/// (from the sim-side flood physics aboard the vessel), the open cavity, and the ballast tanks (so
/// ballast shows as water in its tank).
fn build_flood_state(
    craft: &VoxelCraft,
    com: DVec3,
    flood: &FloodComps,
    ballast: Option<&Ballast>,
) -> FloodState {
    let cs = craft.cell_size;
    let com_off = -com.as_vec3();
    let mut regions = Vec::new();
    // Sealed compartments (WI 718): a region fed by each sim-side flood model's flooded fraction.
    for (comp, fc) in flood.comps.iter().enumerate() {
        let (min, max) = cell_aabb(&fc.cells, cs, com_off);
        regions.push(InteriorWaterRegion {
            min,
            max,
            source: WaterSource::Sealed { comp },
        });
    }
    // The open cavity (WI 713/728): a region fed by the swamp state, so an open boat's interior occludes
    // the ocean (dry hold) and fills with water as it swamps over the rim.
    let cavity = open_cavity(craft);
    if !cavity.cells.is_empty() {
        let (min, max) = cell_aabb(&cavity.cells, cs, com_off);
        regions.push(InteriorWaterRegion {
            min,
            max,
            source: WaterSource::Open,
        });
    }
    // Ballast tanks (WI 709, made visible by WI 729): a region fed by the tank's fill fraction. The tank
    // carries a buoyancy-overcoming `capacity` (m³), not a physical size, so the **render extent** is
    // decoupled from it (WI 733 — a `cbrt(capacity)` cube was bigger than the hull): a low bilge box
    // centred on the mount, a fraction of the hull's footprint, sitting on the floor and clamped **inside**
    // the hull so it never pokes through the skin. Water still rises in it as the tank floods.
    if let Some(b) = ballast {
        // Hull bounds from solids and paneled cells (WI 824 — a hull can be all
        // plates); the plate's owner cell is the inside row for a shell hull.
        let mut hull_cells: Vec<IVec3> = craft.voxels.iter().map(|v| v.cell).collect();
        for p in &craft.face_panels {
            hull_cells.push(p.cell);
        }
        let (hull_min, hull_max) = cell_aabb(&hull_cells, cs, com_off);
        let span = hull_max - hull_min;
        for (tank, t) in b.tanks.iter().enumerate() {
            let centre = t.mount.as_vec3() + com_off;
            let half = Vec3::new(0.3 * span.x, 0.0, 0.3 * span.z);
            let min = (Vec3::new(centre.x, hull_min.y, centre.z) - half).max(hull_min);
            let max =
                (Vec3::new(centre.x, hull_min.y + 0.35 * span.y, centre.z) + half).min(hull_max);
            regions.push(InteriorWaterRegion {
                min,
                max,
                source: WaterSource::Ballast { tank },
            });
        }
    }
    FloodState { regions }
}

// --- markers (teardown: tag only the root of each spawned tree) ---

/// Tags an entity owned by Float mode (despawned on leaving Float).
#[derive(Component)]
struct FloatEntity;

/// The floating hull (its `ActiveBody` is the physics body).
#[derive(Component)]
struct HullMarker;

/// The Float HUD text.
#[derive(Component)]
struct Hud;

const WATER_HALF: f32 = 160.0;
const WATER_SUBDIV: u32 = 64;
// The water patch + wave motion are the shared `scene_water` module (WI 714); the harbor uses the
// `WaveSpec::CALM_HARBOR` preset on its `scene_water::WaterPatch`.

/// The harbor boat-workshop scene (`-- harbor`, WI 706 + 707).
pub struct HarborScenePlugin;

impl Plugin for HarborScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .init_state::<HarborMode>()
            .insert_resource(EditorState {
                // The lattice is loaded from the scenario's blueprint at
                // startup (WI 739) — content, not code.
                craft: VoxelCraft::new(0.5),
                cursor: IVec3::new(0, 3, 0),
                material: 0,
                brush: Brush::default(),
                subassembly: None,
                motor: MotorTier::Standard,
                panel_mode: false,
                form: sounding_sim::shape::Form::Cube,
                orientation_pick: None,
            })
            .init_resource::<OrbitCam>()
            .init_resource::<HoverState>()
            .init_resource::<PointerOnPalette>()
            .init_resource::<HarborReadout>()
            .insert_resource(harbor_cam())
            .init_resource::<FloodState>()
            .insert_resource(Gravity { mu: BODY.mu })
            // The scenario director (WI 739): the Afloat spawn arm assembles
            // the staged hull sim-side (+ the flood-physics stepper).
            .add_plugins(DirectorPlugin)
            .add_plugins(DescentPlugin {
                substep_dt: SUBSTEP_DT,
                max_substeps: MAX_SUBSTEPS,
            })
            .add_systems(Startup, load_harbor_scenario)
            .add_systems(OnEnter(HarborMode::Float), enter_float)
            .add_systems(OnExit(HarborMode::Float), exit_float)
            .add_systems(OnEnter(HarborMode::Build), enter_build)
            .add_systems(OnExit(HarborMode::Build), build::exit_build)
            .add_systems(Update, toggle_mode)
            .add_systems(
                Update,
                (
                    dress_vessel,
                    harbor_drive_input,
                    harbor_ballast_input,
                    harbor_breach_input,
                    raise_interior_water,
                    track_hull,
                    scene_cam::orbit_follow_input,
                    scene_cam::orbit_follow_camera,
                    update_hud,
                    scene_water::animate_water,
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
                    build::track_pointer_over_palette,
                    build::palette_click,
                    mouse_build,
                    orbit_camera,
                    sync_build_meshes,
                    // The shared Build overlays (WI 826 consistency audit): hover
                    // highlight + panel-ghost preview + CoM/axes — the same drawer
                    // the workshop runs. Replaces `draw_editor`, whose wireframes
                    // duplicated the solid meshes (voxels since WI 612, plates
                    // since WI 825) and offered no hover affordance.
                    build::draw_build_overlays,
                    build::update_palette_highlight,
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

/// Loads the harbor scenario document at startup (WI 739): the editor's
/// starting lattice is the referenced blueprint, and the resolved payload is
/// kept as the Float template. A bad document fails fast with the loader's
/// message.
fn load_harbor_scenario(
    mut commands: Commands,
    mut editor: ResMut<EditorState>,
    mut pending: ResMut<PendingSpawn>,
    mut messages: MessageWriter<Command>,
) {
    let path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| DEFAULT_SCENARIO.to_string());
    let roots = ScenarioRoots::default();
    let scenario = match load_scenario(std::path::Path::new(&path), &roots) {
        Ok(s) => s,
        Err(e) => panic!("scenario `{path}` failed to load: {e}"),
    };
    if scenario.placement != StartPlacement::Afloat {
        panic!(
            "scenario `{path}` is not an afloat scenario — the harbor presents Afloat placements"
        );
    }
    info!(
        "scenario `{}` ({}) loaded: seed hull of {} voxels",
        scenario.id,
        scenario.name,
        scenario.blueprint.voxels.len(),
    );
    editor.craft = scenario.blueprint.clone();
    let template = ScenarioSpawn::from_scenario(&scenario);
    // The initial Float entry ran before this loader (Bevy applies the
    // default-state OnEnter ahead of Startup), so stage the opening spawn
    // here; later Build → Float toggles stage through `enter_float`.
    pending.0 = Some(template.clone());
    messages.write(Command::SpawnScenario);
    commands.insert_resource(HarborTemplate(template));
}

/// Enters Float: stage the live build for the director's Afloat spawn (WI 739)
/// and spawn the floating waterfront (WI 706 + 707).
#[allow(clippy::too_many_arguments)]
fn enter_float(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    editor: Res<EditorState>,
    template: Option<Res<HarborTemplate>>,
    mut pending: ResMut<PendingSpawn>,
    mut messages: MessageWriter<Command>,
) {
    let start_render = render_world(DVec3::new(0.0, BODY.radius, 0.0));

    // The floating hull: stage the **live build** as an Afloat scenario spawn
    // (WI 739) — the sim-side director assembles it (real material mass,
    // synthesized drive/rudder/ballast, flood physics) and the shared descent
    // step floats it; `dress_vessel` attaches the render tree when it
    // appears. If the build is empty the director logs and skips it (the
    // waterfront still renders; the player can return to Build).
    // On the very first entry the loader has not run yet (the initial
    // OnEnter precedes Startup) — it stages the opening spawn itself.
    if let Some(template) = template {
        pending.0 = Some(ScenarioSpawn {
            craft: editor.craft.clone(),
            ..template.0.clone()
        });
        messages.write(Command::SpawnScenario);
    }
    commands.insert_resource(FloodState::default());

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
        WaterPatch {
            wave: WaveSpec::CALM_HARBOR,
        },
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
        Text::new("harbor — Float (Enter: Build)\nWASD: drive / steer   F/G: dive / surface   X: breach\ndraft:  --\nheel:   --\nnet buoy: --\nthrust:  --   fuel: --\nballast: --   rudder: --\nflood: --"),
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
    let eye = start_render + scene_cam::orbit_offset(FRAC_PI_4, 0.3, 22.0).as_dvec3();
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

/// Dresses a director-spawned vessel (WI 739): when the Afloat spawn appears,
/// attach the presentation — teardown/camera markers, render placement, the
/// hull skin + panel plates as children, and the interior-water render
/// manifest (occluders + rising-water cuboids) built over the sim-side flood
/// physics.
#[allow(clippy::type_complexity)]
fn dress_vessel(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    vessels: Query<(Entity, &DivingCraft, &FloodComps, Option<&Ballast>), Added<ScenarioVessel>>,
) {
    let Ok((entity, dc, flood, ballast)) = vessels.single() else {
        return;
    };
    let start_render = render_world(DVec3::new(0.0, BODY.radius, 0.0));
    let com_off = -dc.com.as_vec3();
    let craft = dc.craft.clone();
    // Interior water (WI 718/728/709, unified by WI 729): one render manifest of every interior
    // volume that can hold water — sealed compartments (occlude the ocean, flood when breached),
    // the open cavity (swamps over the rim), and the ballast tanks (fill on command).
    let flood_state = build_flood_state(&craft, dc.com, flood, ballast);
    let dry_fill_mat = materials.add(StandardMaterial {
        // A painted-hold interior tone (WI 733): opaque so it occludes the ocean / sea-level water
        // patch, but **self-lit** (emissive) so it reads as the inside of the boat through an open
        // hatch — the hold is an enclosed recess that direct sun + sky IBL never reach, so a merely
        // lit grey rendered black. Emissive ~0.2 matches the dive scene's soft-glow convention under
        // this camera's exposure (not blooming).
        base_color: Color::srgb(0.24, 0.25, 0.27),
        emissive: LinearRgba::rgb(0.20, 0.21, 0.23),
        perceptual_roughness: 1.0,
        ..default()
    });
    let water_fill_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.08, 0.30, 0.42, 0.75),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.1,
        ..default()
    });
    let unit_cube = meshes.add(Mesh::from(Cuboid::new(1.0, 1.0, 1.0)));
    let mut hull = commands.entity(entity);
    hull.insert((
        Transform::default(),
        Visibility::default(),
        WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, start_render)),
        HullMarker,
        scene_cam::CameraTarget, // the follow camera tracks the hull (WI 714)
        FloatEntity,
    ));
    hull.with_children(|parent| {
        // Render the built hull centred on the CoM — solid cubes in their per-material skin,
        // face panels as actual thin plates on their grid boundaries (WI 722 → 824).
        for (material, mesh) in skin_submeshes(&craft, VoxelSkin::Hull) {
            let mat = pbr_material(material, &asset_server, &mut materials);
            parent.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(mat),
                Transform::from_translation(com_off),
            ));
        }
        for (mesh, mat) in panel_render_pieces(&craft, &asset_server, &mut materials, &mut meshes) {
            parent.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mat),
                Transform::from_translation(com_off),
            ));
        }
        // Interior water (WI 729): one renderer for every region — sealed compartment, open cavity,
        // or ballast tank. Each gets an opaque dry hold (occludes the ocean / reads as a tank) + a
        // flat rising-water cuboid (a unit cube `raise_interior_water` scales to its source's level).
        for (i, region) in flood_state.regions.iter().enumerate() {
            let size = region.max - region.min;
            let centre = 0.5 * (region.min + region.max);
            let eps = 0.02;
            // WI 734: an OPEN hold is capped just above the floating waterline (`OPEN_HOLD_FILL`) — it
            // still hides the sea-level water plane, but the top is left open so you see **down into**
            // the real hollow interior (a recess, not a flush lid). Sealed compartments + ballast tanks
            // fill fully (their interiors are genuinely closed).
            let fill = if matches!(region.source, WaterSource::Open) {
                OPEN_HOLD_FILL
            } else {
                1.0
            };
            let occ_h = (size.y * fill - eps).max(0.01);
            parent.spawn((
                Mesh3d(meshes.add(Mesh::from(Cuboid::new(
                    (size.x - eps).max(0.01),
                    occ_h,
                    (size.z - eps).max(0.01),
                )))),
                MeshMaterial3d(dry_fill_mat.clone()),
                Transform::from_translation(Vec3::new(
                    centre.x,
                    region.min.y + 0.5 * size.y * fill,
                    centre.z,
                )),
            ));
            parent.spawn((
                Mesh3d(unit_cube.clone()),
                MeshMaterial3d(water_fill_mat.clone()),
                Transform {
                    translation: Vec3::new(centre.x, region.min.y, centre.z),
                    scale: Vec3::new((size.x - eps).max(0.01), 0.0001, (size.z - eps).max(0.01)),
                    ..default()
                },
                InteriorWater(i),
            ));
        }
    });
    commands.insert_resource(flood_state);
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
    // HUD at top-right to clear the left-edge palette (WI 738).
    commands.spawn((
        Text::new(
            "harbor — Build (Enter: Float)\nmouse: orbit/zoom · L-click place · R-click remove · click palette (left) to pick\nT: panel mode (thin, light) · Tab: material",
        ),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.95, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            right: Val::Px(12.0),
            ..default()
        },
        BuildHud,
        BuildEntity,
    ));
    // The shared clickable palette (WI 738): the harbor gains the workshop's left-edge palette.
    build::spawn_palette(&mut commands);
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
    // Solid cubes render via the per-material hull skin; face panels render as actual thin plates
    // on their grid boundaries (WI 722 → 824).
    for (material, mesh) in skin_submeshes(&editor.craft, VoxelSkin::Hull) {
        let mat = pbr_material(material, &asset_server, &mut materials);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(mat),
            Transform::default(),
            BuildMesh,
            BuildEntity,
        ));
    }
    for (mesh, mat) in
        panel_render_pieces(&editor.craft, &asset_server, &mut materials, &mut meshes)
    {
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(mat),
            Transform::default(),
            BuildMesh,
            BuildEntity,
        ));
    }
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
    let shell_buoy = buoyancy_wrench(
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
    // Open-boat hold-out volume (WI 713): an un-sealed hull floats on the water it holds out up to its
    // rim — the maximum un-swamped displacement (the full cavity, not the fully-submerged swamped value).
    let open = open_cavity(craft);
    let open_buoy = 1_025.0 * g * open.cells.len() as f64 * craft.cell_volume();
    let max_buoy = shell_buoy + open_buoy;
    let weight = mp.mass * g;
    (max_buoy > weight, (weight / max_buoy).clamp(0.0, 1.0))
}

fn update_build_hud(editor: Res<EditorState>, mut hud: Query<&mut Text, With<BuildHud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let cells = editor.craft.voxels.len();
        // The live face-panel store (WI 824/825) — the legacy cell-flag set is
        // decode-only and always empty in play.
        let panels = editor.craft.face_panels.len();
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
        // Shape read-out (WI 833): the same shared-editor state the workshop names.
        let shape = crate::editor::shape_hud_label(&editor);
        text.0 = format!(
            "harbor — Build (Enter: Float)\nmouse: orbit/zoom · L-click place · R-click remove · click palette (left) to pick\nT: place mode — {mode} · Tab: material · shape: {shape}\ncells: {cells}  panels: {panels}\n>> {prediction} <<"
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
    // The gauge uses the **ocean** density: `buoyancy_wrench` already gates each
    // cell/plate by its own submerged fraction, so sampling the medium at the
    // CoM's altitude (the old code) mis-read air density — and "SINKING" — for
    // any hull floating high enough to lift its CoM above the waterline
    // (latent; exposed when the WI 824 plate hull floated higher).
    let water = FluidMedium::EARTHLIKE.sample_altitude(-0.1);
    let load = buoyancy_wrench(
        &dc.craft,
        dc.com,
        body.position,
        body.orientation,
        BODY.radius,
        0.0,
        water.density,
        g_local,
        &dc.enclosed,
    );
    // Open-boat displacement (WI 713): add the held-out-volume buoyancy (zero for a sealed hull, and
    // zero once swamped over the rim) so the gauge matches the physics the descent applies.
    let open = open_cavity_load(
        &dc.craft,
        dc.com,
        body.position,
        body.orientation,
        BODY.radius,
        0.0,
        water.density,
        g_local,
        &dc.open,
    );
    readout.draft = load.draft.max(open.draft);
    readout.heel = heel_angle(body.orientation, up);
    readout.net_buoyancy = (load.force + open.force).length() - body.mass * g_local;
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

/// Breach the hull (WI 718): `X` opens the sealed compartments to the sea so they begin to flood
/// (the flood physics is the sim-side `FloodComps` aboard the vessel, WI 739). One-way — re-enter
/// Float (toggle to Build and back) to reset to a dry hull.
fn harbor_breach_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut hull: Query<&mut FloodComps, With<HullMarker>>,
) {
    let Ok(mut flood) = hull.single_mut() else {
        return;
    };
    if keys.just_pressed(KeyCode::KeyX) && !flood.breached {
        flood.breach();
    }
}

/// Raise the interior-water render (WI 718/728/709, unified by WI 729; the flood *physics* stepped
/// sim-side since WI 739): read each region's fill level from its source — sealed flooding (the
/// vessel's `FloodComps`), open swamping (WI 713), or ballast fill (WI 709) — publish the flooded
/// fraction, and scale every interior-water cuboid to its level (transform-only, no mesh rebuild).
#[allow(clippy::type_complexity)]
fn raise_interior_water(
    flood: Res<FloodState>,
    mut readout: ResMut<HarborReadout>,
    hull: Query<(&ActiveBody, &DivingCraft, &FloodComps, Option<&Ballast>), With<HullMarker>>,
    mut water: Query<(&InteriorWater, &mut Transform)>,
) {
    let Ok((body, dc, comps, ballast)) = hull.single() else {
        return;
    };
    readout.flood = comps.flooded_fraction();
    // The open cavity's swamp share, computed once (1 floating ⇒ dry, 0 swamped ⇒ full). `1 − rim_factor`.
    let open_frac = if flood
        .regions
        .iter()
        .any(|r| matches!(r.source, WaterSource::Open))
    {
        let rim_factor = open_cavity_rim_factor(
            &dc.craft,
            dc.com,
            body.position,
            body.orientation,
            BODY.radius,
            0.0,
            &dc.open,
        );
        let f = (1.0 - rim_factor).clamp(0.0, 1.0);
        readout.flood = readout.flood.max(f);
        f as f32
    } else {
        0.0
    };
    for (iw, mut tf) in &mut water {
        let Some(region) = flood.regions.get(iw.0) else {
            continue;
        };
        let frac = match region.source {
            WaterSource::Sealed { comp } => comps
                .comps
                .get(comp)
                .map_or(0.0, |c| c.flood.flooded_fraction() as f32),
            WaterSource::Open => open_frac,
            WaterSource::Ballast { tank } => ballast
                .and_then(|b| b.tanks.get(tank))
                .filter(|t| t.capacity > 0.0)
                .map_or(0.0, |t| (t.fill / t.capacity).clamp(0.0, 1.0) as f32),
        };
        let (scale_y, translation_y) =
            fill_cuboid_y(region.min.y, region.max.y - region.min.y, frac);
        tf.scale.y = scale_y;
        tf.translation.y = translation_y;
    }
}

/// Middle-drag orbit (L/R swapped per Dave), wheel zoom.
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
            "harbor — Float (Enter: Build) — {status}\nWASD: drive / steer   F/G: dive / surface   X: breach\ndraft:    {:6.2} m\nheel:     {:6.1} deg\nnet buoy: {:8.0} N\nthrust:   {:8.0} N   fuel:    {:3.0}%\nballast:  {:3.0}%   rudder: {:5.1} deg\nflood:    {:3.0}%",
            readout.draft,
            readout.heel.to_degrees(),
            readout.net_buoyancy,
            readout.thrust,
            readout.fuel * 100.0,
            readout.ballast * 100.0,
            readout.rudder.to_degrees(),
            readout.flood * 100.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sounding_sim::ballast::BallastTank;
    use sounding_sim::voxel::Voxel;

    /// The shipped harbor seed hull (the scenario's blueprint) — the tests'
    /// fixture is the same document the scene loads (WI 739).
    fn seed_hull() -> VoxelCraft {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../content/blueprints/harbor-seed.json");
        sounding_sim::library::load_craft(&path).expect("shipped harbor-seed blueprint")
    }

    /// The seed geometry in **solid cubes** — the sinking comparison hull.
    fn solid_shell() -> VoxelCraft {
        let mut c = VoxelCraft::new(0.5);
        let (w, h, l) = (7, 5, 11);
        for x in 0..w {
            for y in 0..h {
                for z in 0..l {
                    if x == 0 || x == w - 1 || y == 0 || y == h - 1 || z == 0 || z == l - 1 {
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

    /// Remove the top deck of an all-plate hull (WI 824 fixture): drop every
    /// Y-normal panel on the hull's top boundary, opening the cavity.
    fn open_top(mut craft: VoxelCraft) -> VoxelCraft {
        let top = craft
            .face_panels
            .iter()
            .filter(|p| p.axis == sounding_sim::voxel::Axis::Y)
            .map(|p| p.cell.y)
            .max()
            .expect("the hull has a deck");
        craft
            .face_panels
            .retain(|p| !(p.axis == sounding_sim::voxel::Axis::Y && p.cell.y == top));
        craft
    }

    /// WI 824 → 825: one rendered plate per face-panel record — the plate you see is
    /// the plate that weighs/floats/seals; no inference, no plates without records.
    /// The geometry now comes from the shared `panel_mesh` seam (a merged box mesh
    /// per material, 6 quads per plate); the shipped seed renders exactly its
    /// record count.
    #[test]
    fn seed_renders_one_plate_per_record_through_the_shared_seam() {
        use sounding_sim::panel_mesh::panel_submeshes;

        let seed = seed_hull();
        assert!(!seed.face_panels.is_empty(), "the seed is a plate hull");
        let boxes: usize = panel_submeshes(&seed)
            .iter()
            .map(|(_, m)| m.face_count() / 6)
            .sum();
        assert_eq!(boxes, seed.face_panels.len());

        // No panels ⇒ no plates.
        assert!(panel_submeshes(&VoxelCraft::new(0.5)).is_empty());
    }

    /// WI 713: an **open-top** hull (seed with its top removed) is predicted to FLOAT — on its held-out
    /// volume — where without the open-cavity term the thin shell alone would read as SINK.
    #[test]
    fn would_float_predicts_an_open_top_hull_floats() {
        let open = open_top(seed_hull()); // remove the deck → an open boat
        assert!(
            !open_cavity(&open).cells.is_empty(),
            "removing the deck opens the cavity"
        );
        assert!(
            would_float(&open).0,
            "an open-top panel hull floats on its held-out volume"
        );
    }

    /// WI 720: the predictor agrees with the float — the panel seed predicts FLOAT (draft < 1), the
    /// same hull in solid cubes predicts SINK.
    #[test]
    fn would_float_predicts_panel_floats_solid_sinks() {
        let seed = seed_hull();
        let (floats, draft) = would_float(&seed);
        assert!(floats, "the panel seed is predicted to float");
        assert!(draft > 0.0 && draft < 1.0, "with a real draft: {draft}");

        assert!(
            !would_float(&solid_shell()).0,
            "the same hull in solid cubes is predicted to sink"
        );

        // Empty craft: predicted to sink, no panic.
        assert!(!would_float(&VoxelCraft::new(0.5)).0);
    }

    /// WI 729: `build_flood_state` produces one render region per water source — a sealed compartment
    /// (Sealed), the open cavity (Open), and each ballast tank (Ballast) — so the renderer is data-driven.
    #[test]
    fn build_flood_state_yields_one_region_per_source() {
        let count = |fs: &FloodState, pred: fn(&WaterSource) -> bool| {
            fs.regions.iter().filter(|r| pred(&r.source)).count()
        };

        // A sealed hull, no ballast: a Sealed region, no Open, no Ballast.
        let seed = seed_hull();
        let com = seed.mass_properties().unwrap().center_of_mass;
        let comps = FloodComps::for_craft(&seed);
        let sealed = build_flood_state(&seed, com, &comps, None);
        assert!(
            count(&sealed, |s| matches!(s, WaterSource::Sealed { .. })) >= 1,
            "the sealed hull yields at least one Sealed region"
        );
        assert_eq!(
            count(&sealed, |s| matches!(s, WaterSource::Open)),
            0,
            "a sealed hull has no open cavity"
        );
        assert_eq!(
            count(&sealed, |s| matches!(s, WaterSource::Ballast { .. })),
            0,
            "no ballast ⇒ no Ballast region"
        );
        assert_eq!(
            comps.comps.len(),
            count(&sealed, |s| matches!(s, WaterSource::Sealed { .. })),
            "each Sealed region maps to a compartment"
        );

        // An open-top hull: an Open region appears.
        let open = open_top(seed_hull());
        let open_com = open.mass_properties().unwrap().center_of_mass;
        let open_state = build_flood_state(&open, open_com, &FloodComps::for_craft(&open), None);
        assert_eq!(
            count(&open_state, |s| matches!(s, WaterSource::Open)),
            1,
            "an open-top hull yields exactly one Open region"
        );

        // With ballast: a Ballast region per tank, fed by the tank's fill.
        let ballast = Ballast {
            tanks: vec![BallastTank {
                capacity: 2.0,
                mount: DVec3::new(0.0, 0.5, 0.0),
                fill: 0.0,
                fill_rate: 1.0,
                blow_rate: 1.0,
            }],
            command: BallastCommand::Hold,
            dry_mass: 1.0,
        };
        let ballasted = build_flood_state(&seed, com, &comps, Some(&ballast));
        assert_eq!(
            count(&ballasted, |s| matches!(
                s,
                WaterSource::Ballast { tank: 0 }
            )),
            1,
            "one tank ⇒ one Ballast region"
        );
    }

    /// WI 733: the ballast render box fits **inside** the hull (no black cube poking through the skin) and
    /// has positive extent (water can still rise in it) — decoupled from the buoyancy `capacity` that made
    /// it a 2.6 m cube in a 2.5 m hull.
    #[test]
    fn ballast_region_fits_within_the_hull() {
        let seed = seed_hull();
        let com = seed.mass_properties().unwrap().center_of_mass;
        let ballast = sounding_sim::afloat::synth_ballast(&seed, BODY.radius)
            .expect("the sealed seed hull synthesizes ballast");
        let fs = build_flood_state(&seed, com, &FloodComps::for_craft(&seed), Some(&ballast));
        let region = fs
            .regions
            .iter()
            .find(|r| matches!(r.source, WaterSource::Ballast { .. }))
            .expect("a ballast region exists");

        // Hull bounds from the plate hull's paneled cells (WI 824 — no voxels).
        let hull_cells: Vec<IVec3> = seed.face_panels.iter().map(|p| p.cell).collect();
        let (hull_min, hull_max) = cell_aabb(&hull_cells, seed.cell_size, -com.as_vec3());
        let eps = 1e-4;
        assert!(
            region.min.cmpge(hull_min - Vec3::splat(eps)).all()
                && region.max.cmple(hull_max + Vec3::splat(eps)).all(),
            "ballast box {:?}..{:?} stays inside the hull {hull_min:?}..{hull_max:?}",
            region.min,
            region.max
        );
        let size = region.max - region.min;
        assert!(
            size.x > 0.01 && size.y > 0.01 && size.z > 0.01,
            "the tank has positive extent so water can rise: {size:?}"
        );
    }

    /// WI 729: the shared fill-raise math — a unit cube scaled to `frac` of the region height, its base
    /// pinned at the floor so the surface rises from the bottom (empty floor-flat, full fills the region).
    #[test]
    fn fill_cuboid_rises_from_the_floor() {
        let (min_y, extent) = (2.0_f32, 4.0_f32);
        // Empty: floor-flat at the base.
        let (s0, t0) = fill_cuboid_y(min_y, extent, 0.0);
        assert!(
            s0 <= 0.001 && (t0 - min_y).abs() < 0.01,
            "empty sits on the floor"
        );
        // Half: half height, centred halfway up.
        let (s1, t1) = fill_cuboid_y(min_y, extent, 0.5);
        assert!(
            (s1 - 2.0).abs() < 1e-4 && (t1 - 3.0).abs() < 1e-4,
            "half fills to mid-height"
        );
        // Full: the whole region, centred at its middle. Over-fill clamps.
        let (s2, t2) = fill_cuboid_y(min_y, extent, 1.5);
        assert!(
            (s2 - 4.0).abs() < 1e-4 && (t2 - 4.0).abs() < 1e-4,
            "full clamps to the region"
        );
    }

    // The wave-height unit test moved to `scene_water` (WI 714).
}

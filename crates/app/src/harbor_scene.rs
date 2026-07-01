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
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::{DQuat, DVec3};
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;

use sounding_sim::active::{ActiveBody, Gravity};
use sounding_sim::ballast::{Ballast, BallastCommand, BallastTank};
use sounding_sim::compartments::compartments;
use sounding_sim::flooding::FloodCompartment;
use sounding_sim::fluid::FluidMedium;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::marine::{MarinePropulsion, MarineThruster, Rudder, ThrusterCommand};
use sounding_sim::medium::{
    buoyancy_wrench, enclosed_cells, heel_angle, max_cross_section, open_cavity, open_cavity_load,
    open_cavity_rim_factor, DescentParams, DescentPlugin, DivingCraft, GlideParams,
    DEFAULT_SLAM_COEFFICIENT,
};
use sounding_sim::powertrain::MotorTier;
use sounding_sim::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Axis, Material, Voxel, VoxelCraft, PANEL_FILL};

use crate::build::{self, BuildEntity, BuildHud, BuildMesh};
use crate::editor::{
    draw_editor, editor_input, mouse_build, mouse_orbit_input, orbit_camera, update_hover, Brush,
    EditorState, HoverState, OrbitCam, PointerOnPalette,
};
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::scene_cam::{self, OrbitFollowCam};
use crate::scene_water::{self, WaterPatch, WaveSpec};
use crate::voxel_skin::{
    materials_present, pbr_material, pbr_material_tinted, skin_submeshes, VoxelSkin,
};

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

/// The harbor's interior-water state (WI 718 + 728 + 709, unified by WI 729): the WI 520 flood physics
/// per sealed compartment, plus a single **render manifest** (`regions`) of every interior volume that can
/// hold water — sealed compartment, open cavity, or ballast tank — each with the source its fill level
/// comes from. One renderer draws them all (occluder + rising surface), so a new water source is data (a
/// new region), not a new render block. Driven by the existing WI 519/520/713 systems — no `sounding_sim`
/// change.
#[derive(Resource, Default)]
struct FloodState {
    /// The sealed-compartment flood models (WI 520) — the physics that drives buoyancy loss. The render
    /// manifest references these by index via [`WaterSource::Sealed`].
    comps: Vec<FloodComp>,
    /// Every interior volume that can hold water, with its fill source (WI 729). The unified renderer
    /// spawns one occluder + one rising-water cuboid per region and raises it from `source`.
    regions: Vec<InteriorWaterRegion>,
    /// Whether the hull has been breached (the player opened it to the sea).
    breached: bool,
}

struct FloodComp {
    /// The compartment's empty cells, sorted by height (ascending) — water fills bottom-up.
    cells: Vec<IVec3>,
    /// The WI 520 transient flood model (volume, centroid, breach/flood state).
    flood: FloodCompartment,
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

/// Flood relaxation rate (1/s) — a several-second flood once breached (WI 520 `step` constant).
const FLOOD_RATE: f64 = 0.15;

/// How much of an **open** boat's interior the dry-hold occluder fills, bottom-up (WI 734). The cap sits
/// just above the typical floating waterline so it still hides the sea-level water plane, while the top is
/// left open so you can see down into the real hollow interior (a recess, not a flush grey lid). Sealed
/// compartments and ballast tanks fill fully — their interiors are genuinely closed.
const OPEN_HOLD_FILL: f32 = 0.72;

/// The cells of a compartment that still hold **air** at a given flooded fraction (WI 718): water fills
/// **bottom-up**, so the lowest `fraction · n` cells have flooded (lost their air) and the upper ones
/// remain. `cells` must be sorted by height ascending. The hull's enclosed-buoyancy set is rebuilt from
/// these, so buoyancy falls as it floods. Pure (the testable core of the buoyancy feedback).
fn unflooded_cells(cells: &[IVec3], flooded_fraction: f64) -> Vec<IVec3> {
    let n = cells.len();
    let flooded = (flooded_fraction.clamp(0.0, 1.0) * n as f64).round() as usize;
    cells.iter().skip(flooded.min(n)).copied().collect()
}

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

/// Build the harbor interior-water state (WI 718/728/709, unified by WI 729), dry: the sealed-compartment
/// flood models plus the render manifest of every interior volume that can hold water — sealed
/// compartments, the open cavity, and the ballast tanks (so ballast shows as water in its tank).
fn build_flood_state(craft: &VoxelCraft, com: DVec3, ballast: Option<&Ballast>) -> FloodState {
    let cs = craft.cell_size;
    let com_off = -com.as_vec3();
    let atm = 101_325.0;
    let crush = 5.0e6; // ~500 m of water (well beyond the harbor)
    let mut comps = Vec::new();
    let mut regions = Vec::new();
    // Sealed compartments (WI 718): each a flood model + a region fed by its flooded fraction.
    for c in &compartments(craft).compartments {
        let mut cells = c.cells.clone();
        cells.sort_by_key(|p| p.y);
        let (min, max) = cell_aabb(&cells, cs, com_off);
        let comp = comps.len();
        comps.push(FloodComp {
            cells,
            flood: FloodCompartment::from_compartment(c, cs, atm, crush),
        });
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
        let (hull_min, hull_max) = cell_aabb(
            &craft.voxels.iter().map(|v| v.cell).collect::<Vec<_>>(),
            cs,
            com_off,
        );
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
    FloodState {
        comps,
        regions,
        breached: false,
    }
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
            .insert_resource(harbor_cam())
            .init_resource::<FloodState>()
            .insert_resource(Gravity { mu: BODY.mu })
            .add_plugins(DescentPlugin {
                substep_dt: SUBSTEP_DT,
                max_substeps: MAX_SUBSTEPS,
            })
            .add_systems(OnEnter(HarborMode::Float), enter_float)
            .add_systems(OnExit(HarborMode::Float), exit_float)
            .add_systems(OnEnter(HarborMode::Build), enter_build)
            .add_systems(OnExit(HarborMode::Build), build::exit_build)
            .add_systems(Update, toggle_mode)
            .add_systems(
                Update,
                (
                    harbor_drive_input,
                    harbor_ballast_input,
                    harbor_breach_input,
                    harbor_flood_step,
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
                    draw_editor,
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
        // Interior water (WI 718/728/709, unified by WI 729): one render manifest of every interior volume
        // that can hold water — sealed compartments (occlude the ocean, flood when breached), the open
        // cavity (swamps over the rim), and the ballast tanks (fill on command). Driven by WI 519/520/713.
        let flood_state = build_flood_state(&craft, dc.com, ballast.as_ref());
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
        let mut hull = commands.spawn((
            body,
            dc,
            marine,
            rudder,
            Transform::default(),
            Visibility::default(),
            WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, start_render)),
            HullMarker,
            scene_cam::CameraTarget, // the follow camera tracks the hull (WI 714)
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
            // Interior water (WI 729): one renderer for every region — sealed compartment, open cavity,
            // or ballast tank. Each gets an opaque dry hold (occludes the ocean / reads as a tank) + a
            // flat rising-water cuboid (a unit cube `harbor_flood_step` scales to its source's fill level).
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
                        scale: Vec3::new(
                            (size.x - eps).max(0.01),
                            0.0001,
                            (size.z - eps).max(0.01),
                        ),
                        ..default()
                    },
                    InteriorWater(i),
                ));
            }
        });
        commands.insert_resource(flood_state);
    } else {
        commands.insert_resource(FloodState::default());
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

/// A steel-blue cast for **panel** plates (WI 727): multiplied over the material's albedo so panels read
/// as panels at a glance (the at-a-glance cue the old blue-cube WI 719 gave), distinct from solid cubes
/// which keep their neutral material colour. The plate geometry (WI 723) is unchanged.
const PANEL_TINT: Color = Color::srgb(0.42, 0.62, 1.0);

/// One reusable **panel-tinted** PBR material handle per distinct structural material in the craft
/// (WI 722/727), so the many panel plates don't each allocate a material — and panels read distinct from
/// solid cubes.
fn material_handles(
    craft: &VoxelCraft,
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
) -> Vec<(Material, Handle<StandardMaterial>)> {
    materials_present(craft)
        .into_iter()
        .map(|m| {
            (
                m,
                pbr_material_tinted(m, PANEL_TINT, asset_server, materials),
            )
        })
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
            "harbor — Build (Enter: Float)\nmouse: orbit/zoom · L-click place · R-click remove · click palette (left) to pick\nT: place mode — {mode} · Tab: material\ncells: {cells}  panels: {panels}\n>> {prediction} <<"
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
    // Open-boat displacement (WI 713): add the held-out-volume buoyancy (zero for a sealed hull, and
    // zero once swamped over the rim) so the gauge matches the physics the descent applies.
    let open = open_cavity_load(
        &dc.craft,
        dc.com,
        body.position,
        body.orientation,
        BODY.radius,
        0.0,
        sample.density,
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

/// Breach the hull (WI 718): `X` opens the sealed compartments to the sea so they begin to flood.
/// One-way — re-enter Float (toggle to Build and back) to reset to a dry hull.
fn harbor_breach_input(keys: Res<ButtonInput<KeyCode>>, mut flood: ResMut<FloodState>) {
    if keys.just_pressed(KeyCode::KeyX) && !flood.breached {
        flood.breached = true;
        for c in &mut flood.comps {
            c.flood.breached = true;
        }
    }
}

/// Advance interior water (WI 718/728/709, unified by WI 729): step each breached compartment against the
/// water at its keel, rebuild the hull's enclosed-buoyancy set from the still-dry (upper) cells so
/// **buoyancy falls as it floods** (it sits lower / sinks), then raise **every** interior-water cuboid to
/// its source's level — sealed flooding (WI 520), open swamping (WI 713), or ballast fill (WI 709) — in
/// one loop. Physics driven entirely by the WI 519/520/713 models; no `sounding_sim` change. Publishes the
/// flooded (flood + swamp) fraction; ballast is reported separately by `track_ballast`.
#[allow(clippy::type_complexity)]
fn harbor_flood_step(
    time: Res<Time>,
    mut flood: ResMut<FloodState>,
    mut readout: ResMut<HarborReadout>,
    mut hull: Query<(&ActiveBody, &mut DivingCraft, Option<&Ballast>), With<HullMarker>>,
    mut water: Query<(&InteriorWater, &mut Transform)>,
) {
    let Ok((body, mut dc, ballast)) = hull.single_mut() else {
        return;
    };
    let dt = time.delta_secs_f64();
    let mut total_vol = 0.0;
    let mut total_water = 0.0;
    let mut enclosed: Vec<IVec3> = Vec::new();
    for comp in &mut flood.comps {
        // Sample the water at the compartment; the breach is taken to sit at/below the keel, so a
        // breached hull always takes on water, and the inflow grows as it sinks deeper (real).
        let centroid_world = body.position + body.orientation * (comp.flood.centroid - dc.com);
        let alt = (centroid_world.length() - BODY.radius).min(-0.1);
        let sample = FluidMedium::EARTHLIKE.sample_altitude(alt);
        comp.flood.step(&sample, FLOOD_RATE, dt);
        total_vol += comp.flood.volume;
        total_water += comp.flood.floodwater;
        enclosed.extend(unflooded_cells(&comp.cells, comp.flood.flooded_fraction()));
    }
    // Rebuild the buoyancy set from the still-dry cells (the shared `advance_descent` reads this).
    dc.enclosed = enclosed;
    readout.flood = if total_vol > 0.0 {
        (total_water / total_vol).clamp(0.0, 1.0)
    } else {
        0.0
    };
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
    // Raise every interior-water cuboid to its source's fill level (transform-only, no mesh rebuild).
    for (iw, mut tf) in &mut water {
        let Some(region) = flood.regions.get(iw.0) else {
            continue;
        };
        let frac = match region.source {
            WaterSource::Sealed { comp } => flood
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

    /// WI 713: an **open-top** hull (seed with its top removed) is predicted to FLOAT — on its held-out
    /// volume — where without the open-cavity term the thin shell alone would read as SINK.
    #[test]
    fn would_float_predicts_an_open_top_hull_floats() {
        let mut open = seed_hull();
        let max_y = open.voxels.iter().map(|v| v.cell.y).max().unwrap();
        open.voxels.retain(|v| v.cell.y != max_y); // remove the deck → an open boat
        open.panels = open
            .panels
            .iter()
            .copied()
            .filter(|c| c.y != max_y)
            .collect();
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

    /// WI 718: flooding removes the hull's enclosed buoyancy **bottom-up**, so net buoyancy falls
    /// monotonically as it floods — a hull that floats dry sinks fully flooded.
    #[test]
    fn flooding_removes_buoyancy_bottom_up_and_sinks() {
        let seed = seed_hull();
        let mp = seed.mass_properties().unwrap();
        let mut cells: Vec<IVec3> = compartments(&seed)
            .compartments
            .iter()
            .flat_map(|c| c.cells.iter().copied())
            .collect();
        cells.sort_by_key(|c| c.y);
        assert!(!cells.is_empty(), "the sealed hull has an enclosed cavity");

        let g = 9.81;
        let deep = DVec3::new(0.0, BODY.radius - 100.0, 0.0);
        let net = |frac: f64| {
            let enc = unflooded_cells(&cells, frac);
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
            buoy - mp.mass * g
        };
        assert!(net(0.0) > 0.0, "dry, the panel hull floats");
        assert!(net(1.0) < 0.0, "fully flooded, it sinks");
        // Monotone buoyancy loss as it floods.
        assert!(
            net(0.0) > net(0.5) && net(0.5) > net(1.0),
            "buoyancy falls as it floods"
        );
        // Bottom-up: at half flood the dry cells are the upper half.
        let dry = unflooded_cells(&cells, 0.5);
        assert!(dry.len() < cells.len() && !dry.is_empty());
        let min_dry_y = dry.iter().map(|c| c.y).min().unwrap();
        let max_flooded_y = cells[(0.5 * cells.len() as f64).round() as usize - 1].y;
        assert!(
            min_dry_y >= max_flooded_y,
            "the dry cells are the upper ones"
        );
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
        let sealed = build_flood_state(&seed, com, None);
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
            sealed.comps.len(),
            count(&sealed, |s| matches!(s, WaterSource::Sealed { .. })),
            "each Sealed region maps to a compartment"
        );

        // An open-top hull: an Open region appears.
        let mut open = seed_hull();
        let max_y = open.voxels.iter().map(|v| v.cell.y).max().unwrap();
        open.voxels.retain(|v| v.cell.y != max_y);
        open.panels = open
            .panels
            .iter()
            .copied()
            .filter(|c| c.y != max_y)
            .collect();
        let open_com = open.mass_properties().unwrap().center_of_mass;
        let open_state = build_flood_state(&open, open_com, None);
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
        let ballasted = build_flood_state(&seed, com, Some(&ballast));
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
        let ballast = synth_ballast(&seed).expect("the sealed seed hull synthesizes ballast");
        let fs = build_flood_state(&seed, com, Some(&ballast));
        let region = fs
            .regions
            .iter()
            .find(|r| matches!(r.source, WaterSource::Ballast { .. }))
            .expect("a ballast region exists");

        let (hull_min, hull_max) = cell_aabb(
            &seed.voxels.iter().map(|v| v.cell).collect::<Vec<_>>(),
            seed.cell_size,
            -com.as_vec3(),
        );
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
    // The wave-height unit test moved to `scene_water` (WI 714).
}

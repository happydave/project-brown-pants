//! Workshop — grounded build-and-test sandbox (`-- workshop`).
//!
//! The build-and-test loop in one scene with a **Build ↔ Test** toggle (`Enter`):
//!
//! - **Build** (WI 603): the voxel editor runs on the workshop's editable craft (reusing the
//!   `editor` module's systems under a state run-condition) — place/remove cells, materials,
//!   devices, the live mass/inertia gizmos. The edited lattice persists across toggles.
//! - **Test** (WI 599 / 602): a controllable craft on the textured ground, hand-flown through
//!   `flight::flight_step` with **live collision** — it lands, rests, drives, and shatters on a
//!   hard crash (`breakage::fracture_on_impact`), substep-capped near the surface
//!   (`warp::safe_substep_dt`). `Backspace` rebuilds the test craft.
//!
//! **Test drives what you built if it's a rover** (WI 607): when the Build lattice carries wheel
//! parts (placed with `6`/`7`), entering Test assembles a `rover::Rover` (mass/inertia from the
//! voxels + parts, wheels from the wheel parts, drive/steer groups from their flags) and drives it
//! on a flat pad via `rover::Rover::step` — rendered rover-anchored with gizmos and a fixed chase
//! camera, like `-- rover`. Otherwise Test **flies** the build as a `FlightCraft` (WI 604). The
//! rover-vs-rocket discriminator is `rover::assemble_rover` returning Some (wheels ⇒ rover).
//!
//! Build and Test are different coordinate worlds (the editor works near the origin; the rocket
//! Test runs in planetary coordinates with floating origin; the rover Test is rover-anchored), so
//! each mode spawns and despawns its own entities on transition — they never coexist.
//!
//! Test controls (rocket): Shift/Ctrl throttle · Z/X full/cut · W/S/A/D/Q/E attitude · T SAS ·
//! F off · `,`/`.` warp · Backspace reset. Test controls (rover): W/S drive · A/D steer ·
//! Space brake · Backspace reset. Build controls (WI 612): **mouse** — left-click places the active
//! brush on the hovered face, right-click removes, middle-drag orbits, scroll zooms. The brush is
//! chosen with Tab (material) and 1-7 (1 control · 2 computer · 3 battery · 4 engine · 5 tank ·
//! 6/7 wheel drive / drive+steer); the craft renders as a **solid** mesh, gizmos only overlay the
//! CoM / inertia axes / hover. Arrows + Space remain a keyboard fallback.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;

use sounding_sim::active::ActiveBody;
use sounding_sim::attitude::{AttitudeControl, AttitudePilot, ReactionWheels, Sas};
use sounding_sim::breakage::fracture_on_impact;
use sounding_sim::collision::{
    craft_bounding_radius, craft_bounds, craft_collision_shape, ground_half_space, Bounds,
    BoxShape, CollisionShape,
};
use sounding_sim::command::{Command, SasMode};
use sounding_sim::contact::{body_contact_wrench, ground_contact_wrench, ContactParams};
use sounding_sim::control::{assemble_control, BatterySpec, ControlComputer};
use sounding_sim::flight::{flight_step, FlightCraft, FlightParams, GroundContact};
use sounding_sim::fluid::FluidMedium;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::launch::LaunchPad;
use sounding_sim::medium::max_cross_section;
use sounding_sim::powertrain::RoverPowertrain;
use sounding_sim::propulsion::{Engine, EngineCommand, Propulsion};
use sounding_sim::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use sounding_sim::rover::{assemble_rover, Rover, RoverAssembly, SUBSTEP_DT as ROVER_SUBSTEP_DT};
use sounding_sim::sim::CentralBody;
use sounding_sim::terrain::Terrain;
use sounding_sim::voxel::{device_mass, Device, DeviceKind, Material, PartKind, Voxel, VoxelCraft};
use sounding_sim::warp::safe_substep_dt;

use crate::editor::{
    editor_input, material_label, mouse_build, mouse_orbit_input, orbit_camera, update_hover,
    Brush, EditorState, HoverState, OrbitCam, PaletteEntry, PointerOnPalette, PALETTE_GROUPS,
};
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::voxel_skin::{pbr_material, skin_submeshes, VoxelSkin};

const BODY: CentralBody = CentralBody::EARTHLIKE;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
const PROPELLANT: ResourceType = ResourceType(0);
const THROTTLE_RATE: f64 = 1.0;
const MIN_WARP: f64 = 1.0;
const MAX_WARP: f64 = 8.0;
/// Contact tolerance for the anti-tunnel substep cap, m.
const CONTACT_TOL: f64 = 0.1;
/// A lightweight test frame: flies and lands fine, but a hard crash overruns its bonds.
const FRAME: Material = Material {
    density: 1_600.0,
    strength: 3.0e6,
};

/// Which half of the build-and-test loop is active.
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
enum WorkshopMode {
    #[default]
    Build,
    Test,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum CraftState {
    Intact,
    Fractured,
}

/// Rover acceleration gravity in the workshop (m/s²).
const ROVER_GRAVITY: f64 = 9.81;
/// Max rover physics substeps per frame (the rover sub-steps far finer than the rocket path).
const ROVER_MAX_SUBSTEPS: u32 = 64;
/// Brake torque per kg of rover mass (N·m/kg).
const ROVER_BRAKE_PER_KG: f64 = 9.0;
/// Steering angle applied to the steer-group wheels (rad).
const ROVER_STEER: f64 = 0.35;
/// Supplemental inward-velocity damping for obstacle contact (1/s): unclamped, approach-only damping
/// that bleeds the rover's speed *into* an obstacle so it thuds and stops instead of springing back
/// off the (elastic) penalty contact. Safe against a static obstacle (no reduced-mass instability).
const OBSTACLE_CONTACT_DAMP: f64 = 40.0;

/// A few obstacles scattered on the pad to drive into (WI 610): a low wall ahead and a couple of
/// rocks off to the sides, clear of the spawn.
fn rover_obstacles() -> Vec<Obstacle> {
    vec![
        Obstacle::new(DVec3::new(0.0, 0.5, 4.0), DVec3::new(2.0, 0.5, 0.25)),
        Obstacle::new(DVec3::new(2.2, 0.4, 2.0), DVec3::new(0.4, 0.4, 0.4)),
        Obstacle::new(DVec3::new(-2.0, 0.35, 2.8), DVec3::new(0.35, 0.35, 0.35)),
    ]
}

/// A static collidable obstacle on the rover pad (WI 610): a fixed box the rover bumps into.
struct Obstacle {
    /// Fixed body at the box centre (zero velocity, effectively infinite mass).
    body: ActiveBody,
    shape: CollisionShape,
    bounds: Option<Bounds>,
    /// Half-extents (m), for rendering the box mesh.
    half: DVec3,
}

impl Obstacle {
    fn new(center: DVec3, half: DVec3) -> Self {
        Self {
            body: ActiveBody::new(center, DVec3::ZERO, 1.0e12, DMat3::IDENTITY),
            shape: CollisionShape::CuboidCompound(vec![BoxShape {
                center: DVec3::ZERO,
                half_extents: half,
            }]),
            bounds: Some(Bounds {
                aabb_min: -half,
                aabb_max: half,
                sphere_center: DVec3::ZERO,
                sphere_radius: half.length(),
            }),
            half,
        }
    }
}

/// The grounded workshop Test state for a **rover** build (WI 607): the assembled rover, its
/// (flat) pad terrain, the drivetrain groups, the source lattice (for reset), and a substep
/// accumulator. Present only when the build is a rover; the rocket path leaves it `None`.
struct RoverState {
    rover: Rover,
    terrain: Terrain,
    drive: Vec<usize>,
    steer: Vec<usize>,
    lattice: VoxelCraft,
    /// The drive power source (combustion / electric) — gates drive torque by fuel/charge (WI 609).
    powertrain: RoverPowertrain,
    /// Throttle intent (−1..1), set by input and routed through the powertrain each frame.
    throttle: f64,
    /// Brake torque magnitude applied to every wheel.
    brake: f64,
    accumulator: f64,
    /// Static obstacles on the pad the rover collides with (WI 610).
    obstacles: Vec<Obstacle>,
    /// World-space breadcrumb trail the rover leaves, so motion is visible against the
    /// (otherwise self-similar, rover-anchored) flat ground.
    track: Vec<DVec3>,
    /// Substep counter for sampling the trail.
    record: u32,
    /// The lattice centre of mass (body frame) — for placing the chassis skin mesh.
    com: DVec3,
    /// Accumulated wheel spin angle (rad) per wheel, for the rolling-wheel render.
    spin_angle: Vec<f64>,
}

/// The grounded workshop Test state: one controllable craft, or its debris after a crash.
#[derive(Resource)]
struct WorkshopWorld {
    body: ActiveBody,
    craft: FlightCraft,
    params: FlightParams,
    pad: LaunchPad,
    accumulator: f64,
    throttle: f64,
    warp: f64,
    state: CraftState,
    fragments: Vec<(VoxelCraft, ActiveBody)>,
    dirty: bool,
    /// When the build is a rover, its rover state; `None` for a rocket (the existing path).
    rover: Option<RoverState>,
}

/// The default workshop craft as an **editable lattice**: a 2×2×2 "test frame" with a control
/// point, a computer, a battery, an engine, and a tank — so it assembles into a flyable craft and
/// the player can edit it in Build mode. Seeds the workshop's `EditorState`.
fn default_lattice() -> VoxelCraft {
    // 0.1 m cells — fine enough to build vehicles, not castles (WI 612 feedback).
    let mut v = VoxelCraft::new(0.1);
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                v.voxels.push(Voxel {
                    cell: IVec3::new(x, y, z),
                    material: FRAME,
                });
            }
        }
    }
    // Device masses scale with cell volume (WI 615) so the seed craft isn't device-mass-dominated.
    let s = v.cell_size;
    v.devices.push(Device::control_point(
        IVec3::new(0, 0, 0),
        device_mass(DeviceKind::Command, s),
        true,
    ));
    v.devices.push(Device::computer(
        IVec3::new(1, 1, 1),
        device_mass(DeviceKind::Computer, s),
        ControlComputer::tuning_computer(0.4),
    ));
    v.devices.push(Device::battery(
        IVec3::new(0, 1, 0),
        device_mass(DeviceKind::Battery, s),
        BatterySpec::full(120.0),
    ));
    v.devices.push(Device::structural(
        IVec3::new(1, 0, 1),
        device_mass(DeviceKind::Engine, s),
        DeviceKind::Engine,
    ));
    v.devices.push(Device::structural(
        IVec3::new(0, 0, 1),
        device_mass(DeviceKind::Tank, s),
        DeviceKind::Tank,
    ));
    v
}

/// Assemble a flyable `FlightCraft` (+ its resting body and a released pad) **from a built
/// lattice** (WI 604). Mass/inertia/CoM and the skin come from the voxels; **engines** are
/// derived from the placed `Engine` devices (thrust through the CoM, +Y), with propellant from
/// the `Tank` devices (or a default if engines but no tanks); **control** comes from
/// `assemble_control` (so a build with no control point is uncontrolled). `None` for an empty
/// lattice (no mass).
fn assemble_from_lattice(voxels: &VoxelCraft) -> Option<(FlightCraft, ActiveBody, LaunchPad)> {
    let mp = voxels.mass_properties()?;
    let s = voxels.cell_size;
    let com = mp.center_of_mass;

    let engine_cells: Vec<IVec3> = voxels
        .devices
        .iter()
        .filter(|d| d.kind == DeviceKind::Engine)
        .map(|d| d.cell)
        .collect();
    let tanks = voxels
        .devices
        .iter()
        .filter(|d| d.kind == DeviceKind::Tank)
        .count();
    let propellant = if engine_cells.is_empty() {
        0.0
    } else {
        tanks.max(1) as f64 * 1_500.0
    };

    let mut propulsion = Propulsion {
        graph: ResourceGraph {
            reservoirs: vec![Reservoir::new(PROPELLANT, propellant, propellant)],
            ..Default::default()
        },
        tank_mounts: vec![com],
        // Thrust along +Y, passed through the CoM in X/Z (the engine sits at the bottom of its
        // cell) so a built craft flies straight without a surprise spin.
        engines: engine_cells
            .iter()
            .map(|c| Engine {
                tank: ReservoirId(0),
                exhaust_velocity: 3_000.0,
                max_mass_flow: 90.0,
                mount: DVec3::new(com.x, c.y as f64 * s, com.z),
                axis: DVec3::Y,
                max_gimbal: 0.0,
            })
            .collect(),
        commands: vec![EngineCommand::default(); engine_cells.len()],
    };
    let mut control = assemble_control(voxels, &mut propulsion.graph);
    control.low_power_reserve = 6.0;
    let attitude = AttitudePilot {
        sas: Sas::default(),
        manual: DVec3::ZERO,
        authority: 5_000.0,
        recapture_on_release: true,
        actuators: AttitudeControl {
            wheels: Some(ReactionWheels::new(8_000.0, 1e9)),
            rcs: None,
        },
    };

    let rest_radius = BODY.radius + com.y;
    let body = ActiveBody::new(
        DVec3::new(0.0, rest_radius, 0.0),
        DVec3::ZERO,
        mp.mass + propellant,
        mp.inertia,
    );
    let mut pad = LaunchPad::resting(rest_radius);
    pad.released = true;

    let craft = FlightCraft {
        dry_mass: mp.mass,
        dry_com: com,
        voxels: voxels.clone(),
        propulsion,
        attitude,
        control,
        autopilot: None,
    };
    Some((craft, body, pad))
}

impl WorkshopWorld {
    /// Wrap an assembled craft + body + pad into a fresh Test world (on the pad, intact).
    fn wrap(craft: FlightCraft, body: ActiveBody, pad: LaunchPad) -> Self {
        Self {
            params: FlightParams {
                mu: BODY.mu,
                surface_radius: BODY.radius,
                medium: FluidMedium::EARTHLIKE,
                drag_area: max_cross_section(&craft.voxels),
                drag_coefficient: 1.0,
                lift: None,
                ground: Some(GroundContact {
                    normal: DVec3::Y,
                    offset: BODY.radius,
                    contact: ContactParams::default(),
                }),
            },
            body,
            craft,
            pad,
            accumulator: 0.0,
            throttle: 0.0,
            warp: 1.0,
            state: CraftState::Intact,
            fragments: Vec::new(),
            dirty: true,
            rover: None,
        }
    }

    /// A Test world flying the given built lattice (falling back to the default craft for an
    /// empty/unassemblable lattice).
    fn from_lattice(voxels: &VoxelCraft) -> Self {
        match assemble_from_lattice(voxels) {
            Some((craft, body, pad)) => Self::wrap(craft, body, pad),
            None => Self::new(),
        }
    }

    fn new() -> Self {
        let (craft, body, pad) =
            assemble_from_lattice(&default_lattice()).expect("default lattice is non-empty");
        Self::wrap(craft, body, pad)
    }

    /// A Test world **driving** an assembled rover (WI 607), resting on a flat pad terrain. The
    /// rocket fields carry a harmless placeholder craft (never stepped — the rover branch handles
    /// stepping/render/input); `rover` is `Some`.
    fn rover(asm: RoverAssembly, lattice: VoxelCraft) -> Self {
        let mut world = Self::new();
        let terrain = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        let mut rover = asm.rover;
        let com = lattice
            .mass_properties()
            .map(|mp| mp.center_of_mass)
            .unwrap_or(DVec3::ZERO);
        // Rest the rover on the pad: place the CoM (`body.position`) high enough that **both** every
        // wheel hub sits at its suspension free length above the surface **and** the chassis bottom
        // clears the ground — so it never spawns partly underground (the "front falls through" bug),
        // then it settles a little under load.
        let ground = terrain.height(0.0, 0.0);
        let wheel_drop = rover
            .wheels
            .iter()
            .map(|w| w.rest_length + w.radius - w.mount.y)
            .fold(0.0_f64, f64::max);
        // Distance from the CoM down to the lowest chassis voxel (CoM-relative), so the chassis
        // bottom lands at/above the ground.
        let min_cell_y =
            lattice.voxels.iter().map(|v| v.cell.y).min().unwrap_or(0) as f64 * lattice.cell_size;
        let chassis_drop = com.y - min_cell_y;
        let drop = wheel_drop.max(chassis_drop) + 0.05;
        rover.body.position = DVec3::new(0.0, ground + drop, 0.0);
        let spin_angle = vec![0.0; rover.wheels.len()];
        world.rover = Some(RoverState {
            rover,
            terrain,
            drive: asm.drive,
            steer: asm.steer,
            lattice,
            powertrain: asm.powertrain,
            throttle: 0.0,
            brake: 0.0,
            accumulator: 0.0,
            obstacles: rover_obstacles(),
            track: Vec::new(),
            record: 0,
            com,
            spin_angle,
        });
        world
    }

    /// Rebuild the *current* test craft on the pad (the Backspace reset). For a rover, re-assemble
    /// from its source lattice; otherwise re-assemble the flight craft from the lattice it flew.
    fn reset(&mut self) {
        if let Some(rs) = &self.rover {
            let lattice = rs.lattice.clone();
            if let Some(asm) = assemble_rover(&lattice, DVec3::ZERO, ROVER_GRAVITY) {
                *self = Self::rover(asm, lattice);
                return;
            }
        }
        let voxels = self.craft.voxels.clone();
        *self = Self::from_lattice(&voxels);
    }

    fn render_of(&self, pos: DVec3) -> DVec3 {
        pos - DVec3::new(0.0, BODY.radius, 0.0)
    }

    /// Render position for a skin mesh: the mesh is built in **raw lattice coordinates** (cells,
    /// not centred on the CoM), while `body.position` is the **CoM**. Place the mesh's lattice
    /// origin at the physical lattice origin (`body.position − orientation·com`) — exactly where
    /// `flight_step`'s collision shape sits — so the rendered hull coincides with the physics
    /// (no float/sink), then rebase to render space.
    fn mesh_origin(&self, body: &ActiveBody, com: DVec3) -> DVec3 {
        self.render_of(body.position - body.orientation * com)
    }

    fn focus(&self) -> DVec3 {
        match self.state {
            CraftState::Intact => self.render_of(self.body.position),
            CraftState::Fractured => {
                if self.fragments.is_empty() {
                    DVec3::ZERO
                } else {
                    let sum: DVec3 = self.fragments.iter().map(|(_, b)| b.position).sum();
                    self.render_of(sum / self.fragments.len() as f64)
                }
            }
        }
    }

    fn altitude(&self) -> f64 {
        self.body.position.length() - BODY.radius
    }

    fn gravity_force(body: &ActiveBody) -> DVec3 {
        let r = body.position;
        let r2 = r.length_squared();
        if r2 <= 0.0 {
            return DVec3::ZERO;
        }
        -BODY.mu * body.mass * r / (r2 * r2.sqrt())
    }

    fn ground_shape(&self) -> CollisionShape {
        ground_half_space(BODY.radius)
    }

    /// Advance the intact craft one substep through the live flight pipeline, capping the step
    /// near the surface (anti-tunnel). Returns `true` if the craft fractured.
    fn step_intact(&mut self, frame_dt: f64) -> bool {
        let radius = craft_bounding_radius(&self.craft.voxels).unwrap_or(0.0);
        let gap = self.body.position.y - BODY.radius - radius;
        let approach = -self.body.velocity.y;
        let dt = safe_substep_dt(gap, approach, frame_dt, CONTACT_TOL);

        let WorkshopWorld {
            body,
            craft,
            params,
            pad,
            ..
        } = self;
        flight_step(body, craft, params, pad, dt);

        let shape = craft_collision_shape(&self.craft.voxels);
        let bounds = craft_bounds(&self.craft.voxels);
        let ground = self.ground_shape();
        let (cf, _) = ground_contact_wrench(
            &self.body,
            &shape,
            bounds,
            self.craft.dry_com,
            &ground,
            &ContactParams::default(),
        );
        if let Some(frags) = fracture_on_impact(&self.craft.voxels, &self.body, cf) {
            self.fragments = frags;
            self.state = CraftState::Fractured;
            self.dirty = true;
            return true;
        }
        false
    }

    /// Advance the debris one substep: gravity + ground + pairwise contact, then integrate.
    fn step_fragments(&mut self, dt: f64) {
        let ground = self.ground_shape();
        let params = ContactParams::default();
        let n = self.fragments.len();
        let shapes: Vec<CollisionShape> = self
            .fragments
            .iter()
            .map(|(v, _)| craft_collision_shape(v))
            .collect();
        let bounds: Vec<Option<Bounds>> = self
            .fragments
            .iter()
            .map(|(v, _)| craft_bounds(v))
            .collect();
        let coms: Vec<DVec3> = self
            .fragments
            .iter()
            .map(|(v, _)| {
                v.mass_properties()
                    .map(|mp| mp.center_of_mass)
                    .unwrap_or(DVec3::ZERO)
            })
            .collect();

        let mut acc = vec![(DVec3::ZERO, DVec3::ZERO); n];
        for i in 0..n {
            let (_, b) = &self.fragments[i];
            acc[i].0 += Self::gravity_force(b);
            let (gf, gt) =
                ground_contact_wrench(b, &shapes[i], bounds[i], coms[i], &ground, &params);
            acc[i].0 += gf;
            acc[i].1 += gt;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let (_, bi) = &self.fragments[i];
                let (_, bj) = &self.fragments[j];
                let ((fa, ta), (fb, tb)) = body_contact_wrench(
                    bi, &shapes[i], bounds[i], coms[i], bj, &shapes[j], bounds[j], coms[j], &params,
                );
                acc[i].0 += fa;
                acc[i].1 += ta;
                acc[j].0 += fb;
                acc[j].1 += tb;
            }
        }
        for (i, (_, b)) in self.fragments.iter_mut().enumerate() {
            b.integrate_wrench(acc[i].0, acc[i].1, dt);
        }
    }
}

// --- Entity markers ---

/// Tags every entity owned by Test mode (despawned on leaving Test).
#[derive(Component)]
struct TestEntity;
/// Tags every entity owned by Build mode (despawned on leaving Build).
#[derive(Component)]
struct BuildEntity;
#[derive(Component)]
struct CraftMarker;
#[derive(Component)]
struct FragmentMarker(usize);
#[derive(Component)]
struct TestHud;
#[derive(Component)]
struct BuildHud;
/// Tags a solid mesh entity rendering part of the Build craft (rebuilt on edit).
#[derive(Component)]
struct BuildMesh;
/// The root container of the Build palette (WI 613); carries `Interaction` so hovering its
/// background/gaps still counts as "pointer over the palette".
#[derive(Component)]
struct PaletteRoot;
/// A clickable Build-palette entry button (WI 613): clicking it selects that block/device/part.
#[derive(Component)]
struct PaletteButton(PaletteEntry);
/// The rover Test's solid chassis skin mesh (WI 608).
#[derive(Component)]
struct RoverChassisMesh;
/// A rover Test wheel (tyre) mesh by wheel index (WI 608).
#[derive(Component)]
struct RoverWheelMesh(usize);
/// A rover Test cosmetic part (seat/antenna/solar/bumper) mesh by part index (WI 608).
#[derive(Component)]
struct RoverPartMesh(usize);
/// A rover Test obstacle box mesh by obstacle index (WI 610).
#[derive(Component)]
struct RoverObstacleMesh(usize);

/// The grounded build-and-test workshop scene.
pub struct WorkshopScenePlugin;

impl Plugin for WorkshopScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .init_state::<WorkshopMode>()
            .insert_resource(WorkshopWorld::new())
            // Seed Build with the default flyable lattice (a control point + engine + battery +
            // tank), so it can be edited and immediately Tested.
            .insert_resource(EditorState {
                craft: default_lattice(),
                cursor: IVec3::new(0, 2, 0),
                material: 0,
                brush: Brush::default(),
                subassembly: None,
            })
            .init_resource::<OrbitCam>()
            .init_resource::<HoverState>()
            .init_resource::<PointerOnPalette>()
            .add_systems(OnEnter(WorkshopMode::Build), enter_build)
            .add_systems(OnExit(WorkshopMode::Build), exit_build)
            .add_systems(OnEnter(WorkshopMode::Test), enter_test)
            .add_systems(OnExit(WorkshopMode::Test), exit_test)
            .add_systems(Update, toggle_mode)
            .add_systems(
                Update,
                (
                    editor_input,
                    mouse_orbit_input,
                    update_hover,
                    track_pointer_over_palette,
                    palette_click,
                    mouse_build,
                    orbit_camera,
                    sync_build_meshes,
                    draw_build_overlays,
                    update_palette_highlight,
                    update_build_hud,
                )
                    .chain()
                    .run_if(in_state(WorkshopMode::Build)),
            )
            .add_systems(
                Update,
                (
                    workshop_input,
                    step_workshop,
                    reconcile_meshes,
                    track_meshes,
                    track_rover_meshes,
                    follow_camera,
                    draw_rover,
                    update_test_hud,
                )
                    .chain()
                    .run_if(in_state(WorkshopMode::Test)),
            );
    }
}

/// `Enter` toggles between Build and Test (from either mode).
fn toggle_mode(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<WorkshopMode>>,
    mut next: ResMut<NextState<WorkshopMode>>,
) {
    if keys.just_pressed(KeyCode::Enter) {
        next.set(match state.get() {
            WorkshopMode::Build => WorkshopMode::Test,
            WorkshopMode::Test => WorkshopMode::Build,
        });
    }
}

// --- Build mode ---

fn enter_build(mut commands: Commands) {
    // The editor's orbit camera (positioned each frame by `orbit_camera`). An ambient term on the
    // camera (Bevy 0.18 makes AmbientLight per-camera) fills shadowed faces of the solid build mesh.
    commands.spawn((
        Camera3d::default(),
        Transform::default(),
        AmbientLight {
            brightness: 250.0,
            ..default()
        },
        BuildEntity,
    ));
    // A sun so the solid (PBR) build meshes are lit.
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(6.0, 14.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        BuildEntity,
    ));
    // Status HUD moved to the top-right (WI 613) to clear the left-edge palette.
    commands.spawn((
        Text::new("workshop · BUILD"),
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
    commands.spawn((
        Text::new(
            "left-click place · right-click remove · middle-drag orbit · scroll zoom · Tab cycles material · Enter → TEST (4 wheels ⇒ drive it)",
        ),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgb(0.7, 0.75, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        BuildEntity,
    ));
    spawn_palette(&mut commands);
}

/// Spawns the left-edge Build palette (WI 613): a docked column of grouped, clickable swatch+label
/// entries — Blocks, Devices, Parts — one [`PaletteButton`] per buildable item. The root carries an
/// `Interaction` so hovering its background (between buttons) still registers as "over the palette".
fn spawn_palette(commands: &mut Commands) {
    let idle = Color::srgb(0.16, 0.16, 0.18);
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(12.0),
                width: Val::Px(168.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            Interaction::default(),
            PaletteRoot,
            BuildEntity,
        ))
        .with_children(|root| {
            for (group, entries) in PALETTE_GROUPS {
                root.spawn((
                    Text::new(*group),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.55, 0.6, 0.68)),
                    Node {
                        margin: UiRect::top(Val::Px(4.0)),
                        ..default()
                    },
                ));
                for &entry in *entries {
                    root.spawn((
                        Button,
                        Node {
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(8.0),
                            padding: UiRect::all(Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(idle),
                        PaletteButton(entry),
                        // No BuildEntity here: buttons are children of the PaletteRoot, so the
                        // recursive despawn in exit_build removes them with the root (avoids a
                        // double-despawn warning on each Build→Test switch).
                    ))
                    .with_children(|btn| {
                        // Identity swatch.
                        btn.spawn((
                            Node {
                                width: Val::Px(16.0),
                                height: Val::Px(16.0),
                                ..default()
                            },
                            BackgroundColor(entry.swatch_color()),
                            BorderColor::all(Color::srgb(0.0, 0.0, 0.0)),
                        ));
                        // Label (so identity never rests on colour alone).
                        btn.spawn((
                            Text::new(entry.label()),
                            TextFont {
                                font_size: 13.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.88, 0.9, 0.94)),
                        ));
                    });
                }
            }
        });
}

/// Sets [`PointerOnPalette`] when the cursor is over the palette root or any entry (WI 613), so
/// `mouse_build` can skip a click that lands on the UI.
fn track_pointer_over_palette(
    mut flag: ResMut<PointerOnPalette>,
    roots: Query<&Interaction, With<PaletteRoot>>,
    buttons: Query<&Interaction, With<PaletteButton>>,
) {
    let over = roots.iter().any(|i| *i != Interaction::None)
        || buttons.iter().any(|i| *i != Interaction::None);
    flag.0 = over;
}

/// Applies a palette entry to the editor selection when its button is pressed (WI 613).
fn palette_click(
    buttons: Query<(&PaletteButton, &Interaction), Changed<Interaction>>,
    mut editor: ResMut<EditorState>,
) {
    for (button, interaction) in &buttons {
        if *interaction == Interaction::Pressed {
            button.0.apply(&mut editor);
        }
    }
}

/// Highlights the active palette entry and reflects hover (WI 613): selected reads from the editor
/// state, so keyboard shortcuts and palette clicks stay in sync through the one source of truth.
fn update_palette_highlight(
    editor: Res<EditorState>,
    mut buttons: Query<(&PaletteButton, &Interaction, &mut BackgroundColor)>,
) {
    for (button, interaction, mut bg) in &mut buttons {
        *bg = if button.0.is_active(&editor) {
            BackgroundColor(Color::srgb(0.20, 0.42, 0.78))
        } else if *interaction == Interaction::Hovered {
            BackgroundColor(Color::srgb(0.30, 0.30, 0.34))
        } else {
            BackgroundColor(Color::srgb(0.16, 0.16, 0.18))
        };
    }
}

fn exit_build(mut commands: Commands, q: Query<Entity, With<BuildEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

fn update_build_hud(editor: Res<EditorState>, mut hud: Query<&mut Text, With<BuildHud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let mass = editor
            .craft
            .mass_properties()
            .map(|mp| mp.mass)
            .unwrap_or(0.0);
        let brush = match editor.brush {
            Brush::Voxel => format!("voxel ({})", material_label(editor.material)),
            other => other.label().to_string(),
        };
        text.0 = format!(
            "workshop · BUILD\nbrush:   {brush}\nvoxels:  {}\ndevices: {}\nwheels:  {}\nmass:    {mass:.0} kg",
            editor.craft.voxels.len(),
            editor.craft.devices.len(),
            editor.craft.parts.len(),
        );
    }
}

/// Rebuilds the **solid** Build meshes when the lattice changes (WI 612): the hull via the skin
/// pipeline (the same one the rocket Test uses), devices as small cubes, wheel parts as cylinders.
/// Replaces the old wireframe-cuboid gizmos; overlays (CoM / axes / cursor) stay gizmos.
fn sync_build_meshes(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    editor: Res<EditorState>,
    existing: Query<Entity, With<BuildMesh>>,
) {
    // Rebuild on edit, and whenever the meshes are missing (e.g. after re-entering Build).
    if !editor.is_changed() && !existing.is_empty() {
        return;
    }
    for e in &existing {
        commands.entity(e).despawn();
    }

    let s = editor.craft.cell_size as f32;
    // Solid hull from the voxels, one sub-mesh per material so each renders with its own appearance
    // (WI 614; same skin + PBR pipeline as the rocket Test).
    for (material, mesh) in skin_submeshes(&editor.craft, VoxelSkin::Hull) {
        let hull = pbr_material(material, &asset_server, &mut materials);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(hull),
            Transform::default(),
            BuildMesh,
            BuildEntity,
        ));
    }
    // Devices: small orange cubes at their cell centres.
    let dev_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.55, 0.0),
        perceptual_roughness: 0.8,
        ..default()
    });
    for d in &editor.craft.devices {
        let c = ((d.cell.as_dvec3() + DVec3::splat(0.5)) * editor.craft.cell_size).as_vec3();
        let m = meshes.add(Mesh::from(Cuboid::new(s * 0.55, s * 0.55, s * 0.55)));
        commands.spawn((
            Mesh3d(m),
            MeshMaterial3d(dev_mat.clone()),
            Transform::from_translation(c),
            BuildMesh,
            BuildEntity,
        ));
    }
    // Wheel parts: dark cylinders at their mount, axis along X (the spin axis).
    let wheel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.10, 0.10, 0.13),
        perceptual_roughness: 0.9,
        ..default()
    });
    for p in &editor.craft.parts {
        if let PartKind::Wheel(spec) = p.kind {
            let m = meshes.add(Mesh::from(Cylinder::new(
                spec.radius as f32,
                (spec.radius * 0.6) as f32,
            )));
            let tf = Transform::from_translation(p.mount.as_vec3())
                .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2));
            commands.spawn((
                Mesh3d(m),
                MeshMaterial3d(wheel_mat.clone()),
                tf,
                BuildMesh,
                BuildEntity,
            ));
        } else {
            // Cosmetic parts (seat/antenna/solar/bumper): recognisable solids at their mount.
            let (mesh, mat) =
                part_mesh(p.kind, editor.craft.cell_size, &mut meshes, &mut materials);
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mat),
                Transform::from_translation(p.mount.as_vec3()),
                BuildMesh,
                BuildEntity,
            ));
        }
    }
}

/// Draws Build **overlays** as gizmos (WI 612): the mouse hover highlight + add-ghost, the keyboard
/// cursor, and the derived CoM / principal inertia axes. The solid geometry itself is meshes
/// (`sync_build_meshes`); gizmos are only for these overlays.
fn draw_build_overlays(mut gizmos: Gizmos, editor: Res<EditorState>, hover: Res<HoverState>) {
    let s = editor.craft.cell_size as f32;
    let cc = |c: IVec3| ((c.as_dvec3() + DVec3::splat(0.5)) * editor.craft.cell_size).as_vec3();

    // Keyboard cursor (faint yellow) — the precise fallback.
    gizmos.primitive_3d(
        &Cuboid::new(s * 1.04, s * 1.04, s * 1.04),
        cc(editor.cursor),
        Color::srgba(1.0, 1.0, 0.1, 0.45),
    );
    // Mouse hover: highlight the hovered cell and ghost where a click would add.
    if let Some(h) = hover.0 {
        gizmos.primitive_3d(
            &Cuboid::new(s * 1.08, s * 1.08, s * 1.08),
            cc(h.highlight),
            Color::srgb(0.2, 1.0, 0.45),
        );
        gizmos.primitive_3d(
            &Cuboid::new(s * 0.94, s * 0.94, s * 0.94),
            cc(h.add_cell),
            Color::srgba(0.2, 1.0, 0.45, 0.4),
        );
    }

    if let Some(mp) = editor.craft.mass_properties() {
        let com = mp.center_of_mass.as_vec3();
        gizmos.sphere(com, s * 0.3, Color::srgb(1.0, 0.1, 1.0));
        // Forward indicator: +Z is the assembled craft/rover's forward (cyan arrow).
        let fwd_len = (s * 5.0).max(1.5);
        gizmos.arrow(com, com + Vec3::Z * fwd_len, Color::srgb(0.1, 0.8, 1.0));
        let colors = [
            Color::srgb(1.0, 0.3, 0.3),
            Color::srgb(0.3, 1.0, 0.3),
            Color::srgb(0.4, 0.5, 1.0),
        ];
        let moments = [
            mp.principal_moments.x,
            mp.principal_moments.y,
            mp.principal_moments.z,
        ];
        let max_m = moments.iter().cloned().fold(0.0_f64, f64::max).max(1e-9);
        for i in 0..3 {
            let axis = mp.principal_axes.col(i).as_vec3().normalize_or_zero();
            let len = s * 2.5 * (moments[i] / max_m).sqrt() as f32;
            gizmos.line(com, com + axis * len, colors[i]);
            gizmos.line(com, com - axis * len, colors[i]);
        }
    }
}

// --- Test mode ---

fn enter_test(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    mut world: ResMut<WorkshopWorld>,
    editor: Res<EditorState>,
) {
    // Drive **what was built** if it is a rover (the build has wheel parts): assemble a rover and
    // run the rover Test path (rover-anchored gizmos + a fixed chase camera, the proven
    // `-- rover` rendering). The rover-vs-rocket discriminator is `assemble_rover` returning Some.
    if let Some(asm) = assemble_rover(&editor.craft, DVec3::ZERO, ROVER_GRAVITY) {
        *world = WorkshopWorld::rover(asm, editor.craft.clone());
        // A fixed chase camera: the rover is rendered anchored at its own position, so a static
        // camera keeps it framed while the terrain scrolls beneath it.
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 7.0, -16.0).looking_at(Vec3::new(0.0, 1.0, 4.0), Vec3::Y),
            TestEntity,
        ));
        commands.spawn((
            DirectionalLight {
                illuminance: 8_000.0,
                ..default()
            },
            Transform::from_xyz(6.0, 14.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
            TestEntity,
        ));
        commands.spawn((
            Text::new("workshop · TEST (rover)"),
            TextFont {
                font_size: 18.0,
                ..default()
            },
            TextColor(Color::srgb(0.9, 0.95, 1.0)),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(12.0),
                ..default()
            },
            TestHud,
            TestEntity,
        ));
        commands.spawn((
            Text::new("W/S drive · A/D steer · Space brake · Backspace reset · Enter → BUILD"),
            TextFont {
                font_size: 14.0,
                ..default()
            },
            TextColor(Color::srgb(0.7, 0.75, 0.8)),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(10.0),
                left: Val::Px(12.0),
                ..default()
            },
            TestEntity,
        ));

        // Solid render (WI 608): chassis skin mesh + a tyre mesh per wheel + cosmetic part meshes,
        // all positioned each frame by `track_rover_meshes`. Replaces the gizmo cuboid + spheres.
        // Chassis skin, one sub-mesh per material (WI 614); all positioned by `track_rover_meshes`.
        for (material, mesh) in skin_submeshes(&editor.craft, VoxelSkin::Hull) {
            let chassis_mat = pbr_material(material, &asset_server, &mut materials);
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(chassis_mat),
                Transform::default(),
                RoverChassisMesh,
                TestEntity,
            ));
        }
        let tyre_mat = materials.add(StandardMaterial {
            base_color: Color::srgb(0.07, 0.07, 0.09),
            perceptual_roughness: 0.95,
            ..default()
        });
        if let Some(rs) = &world.rover {
            for (i, w) in rs.rover.wheels.iter().enumerate() {
                let r = w.radius as f32;
                commands.spawn((
                    Mesh3d(meshes.add(Mesh::from(Cylinder::new(r, r * 0.5)))),
                    MeshMaterial3d(tyre_mat.clone()),
                    Transform::default(),
                    RoverWheelMesh(i),
                    TestEntity,
                ));
            }
            for (j, part) in rs.lattice.parts.iter().enumerate() {
                if matches!(part.kind, PartKind::Wheel(_)) {
                    continue; // wheels handled above
                }
                let (mesh, mat) =
                    part_mesh(part.kind, rs.lattice.cell_size, &mut meshes, &mut materials);
                commands.spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    Transform::default(),
                    RoverPartMesh(j),
                    TestEntity,
                ));
            }
            // Obstacles (WI 610): solid boxes to drive into.
            let obs_mat = materials.add(StandardMaterial {
                base_color: Color::srgb(0.42, 0.36, 0.30),
                perceptual_roughness: 1.0,
                ..default()
            });
            for (k, obs) in rs.obstacles.iter().enumerate() {
                let m = meshes.add(Mesh::from(Cuboid::new(
                    (obs.half.x * 2.0) as f32,
                    (obs.half.y * 2.0) as f32,
                    (obs.half.z * 2.0) as f32,
                )));
                commands.spawn((
                    Mesh3d(m),
                    MeshMaterial3d(obs_mat.clone()),
                    Transform::default(),
                    RoverObstacleMesh(k),
                    TestEntity,
                ));
            }
        }
        return;
    }

    // Otherwise fly **what was built**: assemble the editor's lattice into a fresh craft on the
    // pad (WI 604). An empty/unassemblable build falls back to the default craft.
    *world = WorkshopWorld::from_lattice(&editor.craft);

    let ground =
        crate::ground::spawn_ground(&mut commands, &mut meshes, &mut materials, &asset_server);
    commands.entity(ground).insert(TestEntity); // so it's cleaned up on exit
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5) * Quat::from_rotation_y(0.6)),
        TestEntity,
    ));
    commands.spawn((
        Text::new("workshop · TEST"),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.95, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        TestHud,
        TestEntity,
    ));
    commands.spawn((
        Text::new(
            "Shift/Ctrl throttle · Z/X full/cut · WSAD QE attitude · T SAS  F off · ,/. warp · Backspace reset · Enter → BUILD",
        ),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgb(0.7, 0.75, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        TestEntity,
    ));

    let cam = world.focus() + DVec3::new(14.0, 7.0, 16.0);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(cam.as_vec3()).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
        Atmosphere::earthlike(scattering.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, cam)),
        AnchorCamera,
        TestEntity,
    ));
}

#[allow(clippy::type_complexity)]
fn exit_test(
    mut commands: Commands,
    q: Query<Entity, Or<(With<TestEntity>, With<CraftMarker>, With<FragmentMarker>)>>,
) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

/// Translates keys into commands (throttle/attitude/SAS), plus warp and reset.
fn workshop_input(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut world: ResMut<WorkshopWorld>,
) {
    if keys.just_pressed(KeyCode::Backspace) {
        world.reset();
        return;
    }
    if world.rover.is_some() {
        drive_rover(&keys, &mut world);
        return;
    }
    if world.state != CraftState::Intact {
        return; // debris isn't controllable
    }
    let dt = time.delta_secs_f64();
    if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        world.throttle = (world.throttle + THROTTLE_RATE * dt).min(1.0);
    }
    if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
        world.throttle = (world.throttle - THROTTLE_RATE * dt).max(0.0);
    }
    if keys.just_pressed(KeyCode::KeyZ) {
        world.throttle = 1.0;
    }
    if keys.just_pressed(KeyCode::KeyX) {
        world.throttle = 0.0;
    }
    let orientation = world.body.orientation;
    let throttle = world.throttle;
    world
        .craft
        .apply_command(&Command::SetThrottle(throttle), orientation);

    let mut manual = DVec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        manual.x += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        manual.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        manual.z += 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        manual.z -= 1.0;
    }
    if keys.pressed(KeyCode::KeyQ) {
        manual.y += 1.0;
    }
    if keys.pressed(KeyCode::KeyE) {
        manual.y -= 1.0;
    }
    world
        .craft
        .apply_command(&Command::SetAttitude(manual), orientation);

    if keys.just_pressed(KeyCode::KeyT) {
        let mode = if world.craft.attitude.sas.mode == SasMode::Hold {
            SasMode::Off
        } else {
            SasMode::Hold
        };
        world
            .craft
            .apply_command(&Command::SetSas(mode), orientation);
    }
    if keys.just_pressed(KeyCode::KeyF) {
        world
            .craft
            .apply_command(&Command::SetSas(SasMode::Off), orientation);
    }

    if keys.just_pressed(KeyCode::Period) {
        world.warp = (world.warp * 2.0).min(MAX_WARP);
    }
    if keys.just_pressed(KeyCode::Comma) {
        world.warp = (world.warp / 2.0).max(MIN_WARP);
    }
}

/// Drive the rover by **group**: throttle the drive wheels, steer the steer wheels, brake all.
fn drive_rover(keys: &ButtonInput<KeyCode>, world: &mut WorkshopWorld) {
    let Some(rs) = world.rover.as_mut() else {
        return;
    };
    // Set the throttle/brake **intent**; the powertrain turns throttle into (fuel-gated) torque each
    // frame in `step_workshop` (WI 609).
    rs.throttle = if keys.pressed(KeyCode::KeyW) {
        1.0
    } else if keys.pressed(KeyCode::KeyS) {
        -1.0
    } else {
        0.0
    };
    rs.brake = if keys.pressed(KeyCode::Space) {
        rs.rover.body.mass * ROVER_BRAKE_PER_KG
    } else {
        0.0
    };
    let steer_input = if keys.pressed(KeyCode::KeyA) {
        1.0
    } else if keys.pressed(KeyCode::KeyD) {
        -1.0
    } else {
        0.0
    };
    // Coordinated counter-steer: each steered wheel's angle ∝ its longitudinal offset from the CoM,
    // so rear steer-wheels invert and the rover turns about itself instead of fighting itself.
    let steer = rs.steer.clone();
    rs.rover.set_steer(steer_input, ROVER_STEER, &steer);
}

fn step_workshop(time: Res<Time>, mut world: ResMut<WorkshopWorld>) {
    if world.rover.is_some() {
        let frame_dt = time.delta_secs_f64();
        let rs = world.rover.as_mut().expect("rover present");
        // Route throttle through the powertrain (consumes fuel/charge over the frame); the realized
        // torque drives the drive-group wheels, brake applies to all.
        let torque = rs.powertrain.drive_torque(rs.throttle, frame_dt);
        let brake = rs.brake;
        for (i, w) in rs.rover.wheels.iter_mut().enumerate() {
            w.drive_torque = if rs.drive.contains(&i) { torque } else { 0.0 };
            w.brake = brake;
        }
        rs.accumulator += frame_dt;
        let terrain = rs.terrain;
        // Rover↔obstacle contact (WI 610): the chassis collision shape vs. each static obstacle,
        // injected as an external wrench per sub-step (the seam — `rover.step` integrates it).
        let rover_shape = craft_collision_shape(&rs.lattice);
        let rover_bounds = craft_bounds(&rs.lattice);
        let com = rs.com;
        let contact = ContactParams::default();
        let mut n = 0;
        while rs.accumulator >= ROVER_SUBSTEP_DT && n < ROVER_MAX_SUBSTEPS {
            for obs in &rs.obstacles {
                let ((mut force, torque), _) = body_contact_wrench(
                    &rs.rover.body,
                    &rover_shape,
                    rover_bounds,
                    com,
                    &obs.body,
                    &obs.shape,
                    obs.bounds,
                    DVec3::ZERO,
                    &contact,
                );
                // Kill the elastic rebound: when in contact and still moving *into* the obstacle,
                // add unclamped damping along the contact normal so it thuds and stops.
                if force.length_squared() > 1e-9 {
                    let n = force.normalize();
                    let vn = rs.rover.body.velocity.dot(n);
                    if vn < 0.0 {
                        force -= n * (vn * OBSTACLE_CONTACT_DAMP * rs.rover.body.mass);
                    }
                }
                rs.rover.apply_external(force, torque);
            }
            rs.rover.step(&terrain, ROVER_SUBSTEP_DT);
            rs.accumulator -= ROVER_SUBSTEP_DT;
            n += 1;
            // Accumulate each wheel's spin angle for the rolling-wheel render.
            for (i, w) in rs.rover.wheels.iter().enumerate() {
                rs.spin_angle[i] += w.spin * ROVER_SUBSTEP_DT;
            }
            // Drop a breadcrumb under the rover every so often (motion reference).
            rs.record += 1;
            if rs.record.is_multiple_of(48) {
                let p = rs.rover.body.position;
                rs.track
                    .push(DVec3::new(p.x, rs.terrain.height(p.x, p.z), p.z));
                if rs.track.len() > 400 {
                    rs.track.remove(0);
                }
            }
        }
        return;
    }
    world.accumulator += time.delta_secs_f64() * world.warp;
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS {
        match world.state {
            CraftState::Intact => {
                if world.step_intact(SUBSTEP_DT) {
                    world.accumulator = 0.0;
                    break;
                }
            }
            CraftState::Fractured => world.step_fragments(SUBSTEP_DT),
        }
        world.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

/// Rebuilds the rendered craft/debris entities when the Test world changes (enter, fracture,
/// reset). Cheap: only on `dirty` frames.
fn reconcile_meshes(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut world: ResMut<WorkshopWorld>,
    craft_q: Query<Entity, With<CraftMarker>>,
    frag_q: Query<Entity, With<FragmentMarker>>,
) {
    // The rover path renders with gizmos (`draw_rover`); no skin meshes to reconcile.
    if world.rover.is_some() {
        world.dirty = false;
        return;
    }
    if !world.dirty {
        return;
    }
    for e in &craft_q {
        commands.entity(e).despawn();
    }
    for e in &frag_q {
        commands.entity(e).despawn();
    }

    // One sub-mesh per material so the hull renders as the materials it's made of (WI 614).
    match world.state {
        CraftState::Intact => {
            let render = world.mesh_origin(&world.body, world.craft.dry_com);
            for (material, mesh) in skin_submeshes(&world.craft.voxels, VoxelSkin::Hull) {
                let mat = pbr_material(material, &asset_server, &mut materials);
                commands.spawn((
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(mat),
                    Transform::default(),
                    WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, render)),
                    CraftMarker,
                    TestEntity,
                ));
            }
        }
        CraftState::Fractured => {
            for (i, (voxels, body)) in world.fragments.iter().enumerate() {
                let com = voxels
                    .mass_properties()
                    .map(|mp| mp.center_of_mass)
                    .unwrap_or(DVec3::ZERO);
                let render = world.mesh_origin(body, com);
                for (material, mesh) in skin_submeshes(voxels, VoxelSkin::Hull) {
                    let mat = pbr_material(material, &asset_server, &mut materials);
                    commands.spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(mat),
                        Transform::default(),
                        WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, render)),
                        FragmentMarker(i),
                        TestEntity,
                    ));
                }
            }
        }
    }
    world.dirty = false;
}

#[allow(clippy::type_complexity)]
fn track_meshes(
    world: Res<WorkshopWorld>,
    mut sets: ParamSet<(
        Query<(&mut WorldPlacement, &mut Transform), With<CraftMarker>>,
        Query<(&FragmentMarker, &mut WorldPlacement, &mut Transform)>,
    )>,
) {
    if world.rover.is_some() {
        return; // rover meshes are gizmos, not tracked entities
    }
    match world.state {
        CraftState::Intact => {
            // Per-material sub-meshes (WI 614) all carry CraftMarker; place them together.
            for (mut wp, mut tf) in &mut sets.p0() {
                wp.0 = WorldPos::new(
                    FrameId::CENTRAL_BODY,
                    world.mesh_origin(&world.body, world.craft.dry_com),
                );
                tf.rotation = world.body.orientation.as_quat();
            }
        }
        CraftState::Fractured => {
            for (tag, mut wp, mut tf) in &mut sets.p1() {
                if let Some((voxels, body)) = world.fragments.get(tag.0) {
                    let com = voxels
                        .mass_properties()
                        .map(|mp| mp.center_of_mass)
                        .unwrap_or(DVec3::ZERO);
                    wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, world.mesh_origin(body, com));
                    tf.rotation = body.orientation.as_quat();
                }
            }
        }
    }
}

fn follow_camera(
    world: Res<WorkshopWorld>,
    mut camera: Query<(&mut Transform, &mut WorldPlacement), With<AnchorCamera>>,
) {
    if world.rover.is_some() {
        return; // the rover uses a fixed chase camera (rover-anchored rendering)
    }
    if let Ok((mut tf, mut placement)) = camera.single_mut() {
        let target = world.focus();
        let eye = target + DVec3::new(14.0, 7.0, 16.0);
        placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
        let look_dir = (target - eye).as_vec3().normalize_or_zero();
        if look_dir != Vec3::ZERO {
            tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
        }
    }
}

/// A procedural mesh + material for a catalog part (WI 608), sized to `cell_size`. Recognisable
/// primitive shapes (textured asset-harness versions are deferred to WI 614).
fn part_mesh(
    kind: PartKind,
    s: f64,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) -> (Handle<Mesh>, Handle<StandardMaterial>) {
    let s = s as f32;
    let (mesh, color) = match kind {
        PartKind::Seat => (
            Mesh::from(Cuboid::new(s * 1.2, s * 0.7, s * 1.2)),
            Color::srgb(0.15, 0.16, 0.2),
        ),
        PartKind::Antenna => (
            Mesh::from(Cylinder::new(s * 0.12, s * 4.0)),
            Color::srgb(0.7, 0.72, 0.78),
        ),
        PartKind::SolarPanel => (
            Mesh::from(Cuboid::new(s * 3.0, s * 0.1, s * 2.0)),
            Color::srgb(0.06, 0.1, 0.35),
        ),
        PartKind::Bumper => (
            Mesh::from(Cuboid::new(s * 3.0, s * 0.5, s * 0.5)),
            Color::srgb(0.5, 0.5, 0.55),
        ),
        PartKind::Wheel(w) => (
            Mesh::from(Cylinder::new(w.radius as f32, (w.radius * 0.5) as f32)),
            Color::srgb(0.07, 0.07, 0.09),
        ),
    };
    (
        meshes.add(mesh),
        materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.8,
            ..default()
        }),
    )
}

/// Positions the rover's solid meshes (WI 608) each frame, rover-anchored: the chassis skin at the
/// lattice origin, each tyre at its wheel (steered, riding the suspension), each cosmetic part at
/// its mount — all oriented with the body.
#[allow(clippy::type_complexity)]
fn track_rover_meshes(
    world: Res<WorkshopWorld>,
    mut chassis_q: Query<
        &mut Transform,
        (
            With<RoverChassisMesh>,
            Without<RoverWheelMesh>,
            Without<RoverPartMesh>,
            Without<RoverObstacleMesh>,
        ),
    >,
    mut wheel_q: Query<
        (&RoverWheelMesh, &mut Transform),
        (
            Without<RoverChassisMesh>,
            Without<RoverPartMesh>,
            Without<RoverObstacleMesh>,
        ),
    >,
    mut part_q: Query<
        (&RoverPartMesh, &mut Transform),
        (
            Without<RoverChassisMesh>,
            Without<RoverWheelMesh>,
            Without<RoverObstacleMesh>,
        ),
    >,
    mut obstacle_q: Query<
        (&RoverObstacleMesh, &mut Transform),
        (
            Without<RoverChassisMesh>,
            Without<RoverWheelMesh>,
            Without<RoverPartMesh>,
        ),
    >,
) {
    let Some(rs) = &world.rover else {
        return;
    };
    let body = &rs.rover.body;
    let anchor = body.position;
    let q = body.orientation;

    // Per-material chassis sub-meshes (WI 614) all share the chassis transform.
    for mut tf in &mut chassis_q {
        tf.translation = (-(q * rs.com)).as_vec3();
        tf.rotation = q.as_quat();
    }

    let up = q * DVec3::Y;
    let fwd = q * DVec3::Z;
    for (tag, mut tf) in &mut wheel_q {
        if let Some(w) = rs.rover.wheels.get(tag.0) {
            let hub = body.position + q * w.mount;
            let ground = rs.terrain.height(hub.x, hub.z);
            let normal = rs.terrain.normal(hub.x, hub.z);
            let center = DVec3::new(hub.x, ground + w.radius, hub.z);
            let steer_rot = DQuat::from_axis_angle(up, w.steer);
            let heading = steer_rot * fwd;
            let forward = (heading - normal * heading.dot(normal)).normalize_or_zero();
            let axle = normal.cross(forward).normalize_or_zero();
            let align = Quat::from_rotation_arc(Vec3::Y, axle.as_vec3());
            let spin = Quat::from_axis_angle(axle.as_vec3(), rs.spin_angle[tag.0] as f32);
            tf.translation = (center - anchor).as_vec3();
            tf.rotation = spin * align;
        }
    }
    for (tag, mut tf) in &mut part_q {
        if let Some(part) = rs.lattice.parts.get(tag.0) {
            let world_pos = body.position + q * (part.mount - rs.com);
            tf.translation = (world_pos - anchor).as_vec3();
            tf.rotation = q.as_quat();
        }
    }
    // Obstacles are world-static; rover-anchored, they slide relative to the rover.
    for (tag, mut tf) in &mut obstacle_q {
        if let Some(obs) = rs.obstacles.get(tag.0) {
            tf.translation = (obs.body.position - anchor).as_vec3();
            tf.rotation = Quat::IDENTITY;
        }
    }
}

fn draw_rover(mut gizmos: Gizmos, world: Res<WorkshopWorld>) {
    let Some(rs) = &world.rover else {
        return;
    };
    let body = &rs.rover.body;
    let anchor = body.position;
    let to_render = |p: DVec3| (p - anchor).as_vec3();
    let terrain = &rs.terrain;

    // Terrain grid, **world-locked** (snapped to world coordinates) so it scrolls under the rover as
    // it drives — a rover-relative grid looks identical everywhere on flat ground (the "feels like
    // sitting still" bug).
    let step = 1.0;
    let n = 18;
    let base_x = (anchor.x / step).round() * step;
    let base_z = (anchor.z / step).round() * step;
    let grid = Color::srgb(0.30, 0.26, 0.22);
    for i in -n..=n {
        let mut row = Vec::new();
        let mut col = Vec::new();
        for j in -n..=n {
            let (xi, zj) = (base_x + i as f64 * step, base_z + j as f64 * step);
            let (xj, zi) = (base_x + j as f64 * step, base_z + i as f64 * step);
            row.push(to_render(DVec3::new(xi, terrain.height(xi, zj), zj)));
            col.push(to_render(DVec3::new(xj, terrain.height(xj, zi), zi)));
        }
        gizmos.linestrip(row, grid);
        gizmos.linestrip(col, grid);
    }

    // Breadcrumb trail (world-space) — recedes behind the rover as it moves.
    if rs.track.len() > 1 {
        gizmos.linestrip(
            rs.track.iter().map(|p| to_render(*p)),
            Color::srgb(0.9, 0.7, 0.2),
        );
    }

    // The chassis, tyres, and parts are **solid meshes** (positioned by `track_rover_meshes`); the
    // gizmos here are just overlays.

    // Forward indicator: +Z in the body frame (cyan arrow).
    let fwd = body.orientation * DVec3::Z;
    gizmos.arrow(
        to_render(body.position),
        to_render(body.position + fwd * 3.0),
        Color::srgb(0.1, 0.8, 1.0),
    );

    // Spin spokes: a rotating cross on each tyre's outer face so the (rotationally symmetric) tyre
    // mesh visibly rolls.
    let up = body.orientation * DVec3::Y;
    for (i, w) in rs.rover.wheels.iter().enumerate() {
        let hub = body.position + body.orientation * w.mount;
        let ground = terrain.height(hub.x, hub.z);
        let normal = terrain.normal(hub.x, hub.z);
        let center = DVec3::new(hub.x, ground + w.radius, hub.z);
        let steer_rot = DQuat::from_axis_angle(up, w.steer);
        let heading = steer_rot * (body.orientation * DVec3::Z);
        let forward = (heading - normal * heading.dot(normal)).normalize_or_zero();
        let axle = normal.cross(forward).normalize_or_zero();
        let face = center + axle * (w.radius * 0.27); // just outside the tyre's outer face
        let spin = DQuat::from_axis_angle(axle, rs.spin_angle[i]);
        let a = spin * forward * (w.radius * 0.85);
        let b = spin * axle.cross(forward) * (w.radius * 0.85);
        let spoke = Color::srgb(0.55, 0.55, 0.6);
        gizmos.line(to_render(face - a), to_render(face + a), spoke);
        gizmos.line(to_render(face - b), to_render(face + b), spoke);
    }
}

fn update_test_hud(world: Res<WorkshopWorld>, mut hud: Query<&mut Text, With<TestHud>>) {
    if let Ok(mut text) = hud.single_mut() {
        if let Some(rs) = &world.rover {
            let speed = rs.rover.body.velocity.length();
            let height = rs.rover.height_above_terrain(&rs.terrain);
            text.0 = format!(
                "workshop · TEST (rover)\nspeed:  {speed:6.2} m/s\nheight: {height:6.2} m\nwheels: {}\n{}:  {:3.0}%",
                rs.rover.wheels.len(),
                rs.powertrain.label(),
                rs.powertrain.fraction() * 100.0,
            );
            return;
        }
        match world.state {
            CraftState::Intact => {
                let speed = world.body.velocity.length();
                let resting = speed < 0.1;
                let state = if resting { "RESTING" } else { "flying" };
                let sas = match world.craft.attitude.sas.mode {
                    SasMode::Off => "off",
                    SasMode::KillRotation => "kill-rot",
                    SasMode::Hold => "hold",
                    SasMode::Point(_) => "point",
                };
                text.0 = format!(
                    "workshop · TEST: {state}\nthrottle: {:3.0}%\naltitude: {:6.2} m\nv-speed:  {:+6.2} m/s\nspeed:    {:6.2} m/s\nSAS {sas}   warp {:.0}x",
                    world.throttle * 100.0,
                    world.altitude(),
                    world.body.velocity.y,
                    speed,
                    world.warp,
                );
            }
            CraftState::Fractured => {
                text.0 = format!(
                    "workshop · TEST: CRASHED — fractured into {} pieces\nBackspace to rebuild",
                    world.fragments.len()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default lattice assembles into a flyable craft: controllable, with an engine and
    /// propellant, mass/inertia from the voxels.
    #[test]
    fn default_lattice_assembles_a_flyable_craft() {
        let (craft, _body, _pad) =
            assemble_from_lattice(&default_lattice()).expect("default lattice is non-empty");
        assert!(
            craft.resolve_control().allows_manual(),
            "a control point makes it controllable"
        );
        assert_eq!(
            craft.propulsion.engines.len(),
            1,
            "one engine device → one engine"
        );
        assert!(
            craft.propulsion.propellant() > 0.0,
            "a tank device gives it propellant"
        );
        let mp = default_lattice().mass_properties().unwrap();
        assert!(
            (craft.dry_mass - mp.mass).abs() < 1e-9,
            "mass from the lattice"
        );
    }

    /// A bare lattice (no devices) assembles into an **uncontrolled**, engineless craft — control
    /// reflects what was built (the WI 604 acceptance case).
    #[test]
    fn deviceless_build_is_uncontrolled() {
        let mut v = VoxelCraft::new(1.0);
        for x in 0..2 {
            for z in 0..2 {
                v.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::ALUMINIUM,
                });
            }
        }
        let (craft, _, _) = assemble_from_lattice(&v).expect("non-empty");
        assert!(
            !craft.resolve_control().allows_manual(),
            "no control point → uncontrolled"
        );
        assert!(
            craft.propulsion.engines.is_empty(),
            "no engine device → no engine"
        );
    }

    /// An empty lattice has no mass, so it can't be assembled (the scene falls back to default).
    #[test]
    fn empty_lattice_does_not_assemble() {
        assert!(assemble_from_lattice(&VoxelCraft::new(1.0)).is_none());
    }

    /// A lattice with wheel parts is a rover: `assemble_rover` returns Some, and the rover Test
    /// world places it resting on the pad with its drivetrain groups intact.
    #[test]
    fn wheeled_lattice_drives_as_a_rover() {
        use sounding_sim::voxel::{Part, PartKind, WheelPart};
        let mut v = default_lattice();
        for (x, z, steer) in [(0, 0, false), (1, 0, false), (0, 1, true), (1, 1, true)] {
            v.parts.push(Part {
                mount: DVec3::new(x as f64, -0.3, z as f64),
                mass: 60.0,
                kind: PartKind::Wheel(WheelPart::new(true, steer)),
            });
        }
        let asm = assemble_rover(&v, DVec3::ZERO, ROVER_GRAVITY).expect("wheels ⇒ rover");
        assert_eq!(asm.rover.wheels.len(), 4);
        assert_eq!(asm.steer.len(), 2);

        let world = WorkshopWorld::rover(asm, v);
        let rs = world.rover.as_ref().expect("rover world");
        // Rests on the pad: the CoM sits above the flat surface (height 0), finite.
        assert!(rs.rover.body.position.y > 0.0 && rs.rover.body.position.y.is_finite());
        assert_eq!(rs.drive.len(), 4);
    }

    /// The default (wheel-less) lattice is a rocket: `assemble_rover` is None, so the Test path
    /// flies it (the discriminator).
    #[test]
    fn default_lattice_is_not_a_rover() {
        assert!(assemble_rover(&default_lattice(), DVec3::ZERO, ROVER_GRAVITY).is_none());
    }
}

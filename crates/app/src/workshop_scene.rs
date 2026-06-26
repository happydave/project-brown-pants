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
//! Space brake · Backspace reset. `P` pauses/resumes either mode (WI 638). `K` saves the build as a
//! craft, `O` opens one back into Build (WI 637). Build controls (WI 612): **mouse** — left-click places the active
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
use sounding_sim::rover::{
    assemble_rover, Rover, RoverAssembly, Wheel, SUBSTEP_DT as ROVER_SUBSTEP_DT,
};
use sounding_sim::sim::{CentralBody, SimClock};
use sounding_sim::telemetry::RoverTelemetry;
use sounding_sim::terrain::{Ramp, Terrain};
use sounding_sim::voxel::{device_mass, Device, DeviceKind, Material, PartKind, Voxel, VoxelCraft};
use sounding_sim::warp::safe_substep_dt;

use crate::bus::GroundedRover;
use crate::editor::{
    editor_input, material_label, mouse_build, mouse_orbit_input, orbit_camera, update_hover,
    Brush, EditorState, HoverState, OrbitCam, PaletteEntry, PointerOnPalette, PALETTE_GROUPS,
};
use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::overlay::{spawn_overlay, update_overlay, CockpitOverlay};
use crate::replay::Replayable;
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
/// Brake torque per kg of rover mass (N·m/kg). Well above the drive torque (≈4 N·m/kg) so braking
/// firmly locks the wheels and stops at the tire-grip limit — brakes bite harder than the throttle.
const ROVER_BRAKE_PER_KG: f64 = 35.0;
/// Maximum steering angle applied to the steer-group wheels at low speed (rad). Reduced with speed
/// (see [`STEER_SPEED_REF`]) so a flick at speed doesn't spin the rover.
const ROVER_STEER: f64 = 0.35;
/// How fast keyboard steering ramps toward full lock / recenters (1/s): a quick tap yields a small
/// angle instead of instant full lock (WI 630 feel tuning), so light corrections don't spin.
const STEER_RATE: f64 = 3.0;
/// Speed (m/s) at which steering authority halves: effective lock = `ROVER_STEER / (1 + v/ref)`. Keeps
/// low-speed manoeuvring sharp while making high-speed steering progressively gentler (WI 630).
const STEER_SPEED_REF: f64 = 7.0;
/// Supplemental inward-velocity damping for obstacle contact (1/s): unclamped, approach-only damping
/// that bleeds the rover's speed *into* an obstacle so it thuds and stops instead of springing back
/// off the (elastic) penalty contact. Safe against a static obstacle (no reduced-mass instability).
const OBSTACLE_CONTACT_DAMP: f64 = 40.0;
/// Minimum closing speed (m/s) for an obstacle contact to count as an "impact" for the diagnostic
/// (WI 618) — filters out resting/leaning/sliding contact so only real knocks are reported.
const IMPACT_MIN_CLOSING: f64 = 1.0;
/// Duration of the post-impact watch (s): how long after an impact to watch for a kraken / fall-through.
const IMPACT_WATCH_SECONDS: f64 = 2.5;
/// Post-impact kraken thresholds (WI 618): angular speed (rad/s) and vertical bounce (m/s) above which
/// the watch flags a kraken; and the height (m, below terrain) that counts as falling through the world.
const KRAKEN_OMEGA: f64 = 8.0;
const KRAKEN_BOUNCE: f64 = 8.0;
const FALL_THROUGH_HEIGHT: f64 = -0.5;
/// Minimum landing (downward) speed (m/s) for a touchdown to count as a fall worth reporting / damaging
/// (WI 630) — normal driving over bumps stays below this, so only real drops register. The actual wheel
/// shear is still gated per-wheel by the rated shear speed.
const FALL_MIN_SPEED: f64 = 3.0;

/// A 30° wedge ramp off the left of the pad (WI 630 test affordance): steer left and drive up it to
/// launch off the lip and check the rover tumbles. Clear of the spawn and the obstacle course.
const ROVER_TEST_RAMP: Ramp = Ramp {
    center_x: -4.5,
    half_width: 1.8,
    start_z: 1.0,
    run: 3.0,
    angle: std::f64::consts::FRAC_PI_6, // 30°
};

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

/// The post-impact watch result (WI 618): what happened in the seconds after an impact — used to
/// catch a kraken (excessive rotation/bounce) or the rover falling through the world.
#[derive(Clone)]
struct WatchResult {
    max_speed: f64,
    max_omega: f64,
    max_bounce: f64,
    min_height: f64,
    verdict: String,
}

/// A captured rover↔obstacle impact (WI 618 diagnostic): the impact (closing) speed, what was hit,
/// the most-stressed wheel's effective impact speed vs. its rated speed, which wheels sheared, and —
/// once the post-impact watch completes — what happened next. Surfaced on the HUD and logged as a
/// copy-pasteable block so impacts can be tuned against real data.
#[derive(Clone)]
struct ImpactReport {
    /// Rover speed at impact (m/s).
    speed: f64,
    /// Closing speed into the obstacle (m/s).
    closing: f64,
    /// What was struck — "chassis" today (the obstacle collision is the hull only); true tire/rim
    /// attribution is WI 631.
    impacted: &'static str,
    /// The most-stressed wheel at impact (index into `rover.wheels`), if any.
    peak_wheel: Option<usize>,
    /// Effective impact speed that wheel felt (closing × side share), m/s.
    demand: f64,
    /// That wheel's rated shear speed, m/s.
    capacity: f64,
    /// Wheels that sheared off at this impact.
    sheared: Vec<usize>,
    /// Wheels whose tire blew out at this impact (WI 631b).
    blown_tires: Vec<usize>,
    /// Wheels whose rim bent at this impact (WI 631b).
    bent_rims: Vec<usize>,
    /// Wheels whose damper blew at this impact (WI 631b).
    blown_dampers: Vec<usize>,
    /// Filled in when the post-impact watch finishes.
    watch: Option<WatchResult>,
}

/// The in-progress post-impact watch (WI 618): tracks the seconds after an impact for kraken /
/// fall-through signs, then writes a [`WatchResult`] onto the last impact.
#[derive(Clone)]
struct ImpactWatch {
    steps_left: u32,
    max_speed: f64,
    max_omega: f64,
    max_bounce: f64,
    min_height: f64,
}

impl ImpactReport {
    /// A compact "what failed" summary across components and shear (WI 631b), or "intact".
    fn damage_summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.blown_tires.is_empty() {
            parts.push(format!("blew tire {:?}", self.blown_tires));
        }
        if !self.bent_rims.is_empty() {
            parts.push(format!("bent rim {:?}", self.bent_rims));
        }
        if !self.blown_dampers.is_empty() {
            parts.push(format!("blew damper {:?}", self.blown_dampers));
        }
        if !self.sheared.is_empty() {
            parts.push(format!("sheared {:?}", self.sheared));
        }
        if parts.is_empty() {
            "intact".to_string()
        } else {
            parts.join(", ")
        }
    }

    /// A one-line HUD summary (plus the post-impact verdict once the watch finishes).
    fn hud_line(&self) -> String {
        let result = self.damage_summary();
        let base = format!(
            "impact: {:.1} m/s ({}) · load {:.1}/{:.1} m/s · {result}",
            self.closing, self.impacted, self.demand, self.capacity
        );
        match &self.watch {
            Some(w) => format!("{base} · {}", w.verdict),
            None => base,
        }
    }

    /// A multi-line, copy-pasteable diagnostic block (logged to the console at each impact).
    fn log_block(&self) -> String {
        format!(
            "===== ROVER IMPACT (WI 618) =====\n\
             impacted:      {}\n\
             rover speed:   {:.2} m/s\n\
             closing speed: {:.2} m/s (into obstacle)\n\
             peak wheel:    {}\n\
             impact speed:  {:.2} m/s (effective, this wheel)\n\
             rated speed:   {:.2} m/s  ({})\n\
             damage:        {}\n\
             =================================",
            self.impacted,
            self.speed,
            self.closing,
            self.peak_wheel
                .map_or("none".to_string(), |i| i.to_string()),
            self.demand,
            self.capacity,
            if self.demand > self.capacity {
                "exceeded"
            } else {
                "held"
            },
            self.damage_summary(),
        )
    }

    /// The copy-pasteable post-impact watch block (logged when the watch finishes).
    fn watch_block(w: &WatchResult) -> String {
        format!(
            "----- post-impact (next few s) -----\n\
             verdict:    {}\n\
             max speed:  {:.2} m/s\n\
             max omega:  {:.2} rad/s\n\
             max bounce: {:.2} m/s (vertical)\n\
             min height: {:.2} m (above terrain)\n\
             ------------------------------------",
            w.verdict, w.max_speed, w.max_omega, w.max_bounce, w.min_height,
        )
    }
}

impl ImpactWatch {
    /// Begin a watch covering [`IMPACT_WATCH_SECONDS`] of sub-steps.
    fn start() -> Self {
        Self {
            steps_left: (IMPACT_WATCH_SECONDS / ROVER_SUBSTEP_DT) as u32,
            max_speed: 0.0,
            max_omega: 0.0,
            max_bounce: 0.0,
            min_height: f64::INFINITY,
        }
    }

    /// Finish the watch and classify what happened (kraken / fell through / OK).
    fn finish(&self) -> WatchResult {
        let verdict = if self.min_height < FALL_THROUGH_HEIGHT {
            format!("FELL THROUGH WORLD (min height {:.1} m)", self.min_height)
        } else if self.max_omega > KRAKEN_OMEGA || self.max_bounce > KRAKEN_BOUNCE {
            format!(
                "KRAKEN (omega {:.1} rad/s, bounce {:.1} m/s)",
                self.max_omega, self.max_bounce
            )
        } else {
            "OK (recovered)".to_string()
        };
        WatchResult {
            max_speed: self.max_speed,
            max_omega: self.max_omega,
            max_bounce: self.max_bounce,
            min_height: self.min_height,
            verdict,
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
    /// Whether a contact episode is open, and its peak closing speed so far (WI 618 diagnostic) —
    /// used to emit a "survived" report when a notable but non-shearing episode ends.
    episode: Option<(f64, ImpactReport)>,
    /// Whether the open episode already logged a shear (so its end doesn't double-report).
    episode_reported: bool,
    /// The last completed impact, shown on the HUD and logged to the console.
    last_impact: Option<ImpactReport>,
    /// The in-progress post-impact watch, if any.
    watch: Option<ImpactWatch>,
    /// Whether the rover is currently airborne (all wheels off the ground), for fall-damage detection.
    airborne: bool,
    /// Peak downward speed (m/s) accumulated while airborne — the landing speed when it touches down.
    fall_peak: f64,
    /// Smoothed steering input (−1..1): ramps toward the key target so a tap is a gentle correction.
    steer_input: f64,
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
            // A 30° wedge off to the side to drive up and launch off (WI 630 test affordance): steer
            // left, climb it, and catch air over the lip to check the rover tumbles when it should.
            ramp: Some(ROVER_TEST_RAMP),
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
            episode: None,
            episode_reported: false,
            last_impact: None,
            watch: None,
            airborne: false,
            fall_peak: 0.0,
            steer_input: 0.0,
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
/// A rover Test wheel (tyre) mesh: wheel index, plus the radius it was built at (WI 608); the second
/// field lets the render shrink the mesh when a tire blows and the wheel runs on its smaller rim
/// (WI 631b).
#[derive(Component)]
struct RoverWheelMesh(usize, f32);
/// A rover Test cosmetic part (seat/antenna/solar/bumper) mesh by part index (WI 608).
#[derive(Component)]
struct RoverPartMesh(usize);
/// A rover Test obstacle box mesh by obstacle index (WI 610).
#[derive(Component)]
struct RoverObstacleMesh(usize);
/// The rover Test wedge-ramp mesh (WI 630 test affordance).
#[derive(Component)]
struct RoverRampMesh;

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
            .init_resource::<CockpitOverlay>()
            .add_systems(OnEnter(WorkshopMode::Build), enter_build)
            .add_systems(OnExit(WorkshopMode::Build), exit_build)
            .add_systems(OnEnter(WorkshopMode::Test), enter_test)
            .add_systems(
                OnExit(WorkshopMode::Test),
                (exit_test, clear_rover_telemetry),
            )
            .add_systems(
                Update,
                (
                    toggle_mode,
                    crate::pause::toggle_pause,
                    crate::pause::step_scene,
                ),
            )
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
                    publish_rover,
                    reconcile_meshes,
                    track_meshes,
                    track_rover_meshes,
                    track_ramp_mesh,
                    follow_camera,
                    draw_rover,
                    update_test_hud,
                    update_overlay,
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
            "left-click place · right-click remove · middle-drag orbit · scroll zoom · Tab material · K save · O open craft · P pause · Enter → TEST (4 wheels ⇒ drive it)",
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
            Text::new(
                "W/S drive · A/D steer · Space brake · P pause · R replay [/] scrub · G cockpit · Backspace reset · Enter → BUILD",
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
        // Cockpit overlay (WI 646), fresh each Test session; TestEntity-tagged so it is cleaned up on
        // exit (recursive despawn covers its panels). `G` toggles it.
        commands.insert_resource(CockpitOverlay::default());
        let overlay = spawn_overlay(&mut commands);
        commands.entity(overlay).insert(TestEntity);

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
                Replayable,
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
                    RoverWheelMesh(i, r),
                    Replayable,
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
                    Replayable,
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
            // Test ramp (WI 630): a wedge to drive up and launch off the lip.
            let r = ROVER_TEST_RAMP;
            let slope_len = (r.run / r.angle.cos()) as f32;
            let ramp_mesh = meshes.add(Mesh::from(Cuboid::new(
                (r.half_width * 2.0) as f32,
                0.15,
                slope_len,
            )));
            commands.spawn((
                Mesh3d(ramp_mesh),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.50, 0.45, 0.30),
                    perceptual_roughness: 1.0,
                    ..default()
                })),
                Transform::default(),
                RoverRampMesh,
                TestEntity,
            ));
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
        drive_rover(&keys, &mut world, time.delta_secs_f64());
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
fn drive_rover(keys: &ButtonInput<KeyCode>, world: &mut WorkshopWorld, dt: f64) {
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
    // Smooth the keyboard steer toward its target so a quick tap is a small correction, not instant
    // full lock (WI 630 feel tuning).
    let target = if keys.pressed(KeyCode::KeyA) {
        1.0
    } else if keys.pressed(KeyCode::KeyD) {
        -1.0
    } else {
        0.0
    };
    let step = STEER_RATE * dt;
    rs.steer_input += (target - rs.steer_input).clamp(-step, step);
    // Speed-sensitive authority: full lock when slow, progressively gentler with speed so a flick at
    // speed can't spin the rover.
    let speed = rs.rover.body.velocity.length();
    let max_angle = ROVER_STEER / (1.0 + speed / STEER_SPEED_REF);
    // Coordinated counter-steer: each steered wheel's angle ∝ its longitudinal offset from the CoM,
    // so rear steer-wheels invert and the rover turns about itself instead of fighting itself.
    let steer = rs.steer.clone();
    let steer_input = rs.steer_input;
    rs.rover.set_steer(steer_input, max_angle, &steer);
}

/// Publishes the Test rover's live state onto the bus bridge each frame (WI 640), so
/// `GET /telemetry` and the dev-MCP bridge can introspect a workshop rover. Carries `None`
/// when Test is flying a rocket (no assembled rover); cleared on leaving Test.
fn publish_rover(world: Res<WorkshopWorld>, mut grounded: ResMut<GroundedRover>) {
    grounded.0 = world
        .rover
        .as_ref()
        .map(|rs| RoverTelemetry::from_rover(&rs.rover));
}

/// Clears the rover bridge when leaving Test (WI 640) so Build mode publishes no stale rover.
fn clear_rover_telemetry(mut grounded: ResMut<GroundedRover>) {
    grounded.0 = None;
}

fn step_workshop(time: Res<Time>, mut clock: ResMut<SimClock>, mut world: ResMut<WorkshopWorld>) {
    // Paused (WI 638): freeze the active physics. Camera, HUD, and inspection stay live. While paused,
    // a step (WI 643) advances a bounded chunk; `frame_step_dt` returns `None` to stay frozen.
    let Some(frame_dt) = crate::pause::frame_step_dt(&mut clock, &time) else {
        return;
    };
    if world.rover.is_some() {
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
            let mut any_contact = false;
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
                // Impact damage (WI 618): keyed to the *closing speed* into the obstacle (not the
                // sustained contact force, so leaning never shears). A hit shears the wheels facing it
                // when their effective impact speed exceeds their material-rated speed; sheared wheels
                // drop from the drive/steer groups and their tyres hide.
                let mut closing = 0.0;
                let mut into = DVec3::ZERO;
                if force.length_squared() > 1e-9 {
                    let nrm = force.normalize();
                    closing = (-rs.rover.body.velocity.dot(nrm)).max(0.0);
                    into = -nrm; // toward the obstacle
                }
                let outcome = rs.rover.shear_on_impact(closing, into);
                for &idx in &outcome.sheared {
                    rs.drive.retain(|&x| x != idx);
                    rs.steer.retain(|&x| x != idx);
                }
                // Diagnostic: build a report for a notable impact; log shears immediately (so a
                // sustained square-on hit still reports), and remember a non-shearing episode's peak
                // to emit a "survived" report when it ends.
                if closing > IMPACT_MIN_CLOSING {
                    any_contact = true;
                    let report = ImpactReport {
                        speed: rs.rover.body.velocity.length(),
                        closing,
                        impacted: "chassis",
                        peak_wheel: outcome.peak_wheel,
                        demand: outcome.peak_demand,
                        capacity: outcome.peak_capacity,
                        sheared: outcome.sheared.clone(),
                        blown_tires: outcome.blown_tires.clone(),
                        bent_rims: outcome.bent_rims.clone(),
                        blown_dampers: outcome.blown_dampers.clone(),
                        watch: None,
                    };
                    // Any component failure (not just a clean shear) is a notable, loggable event.
                    let damaged = !outcome.sheared.is_empty()
                        || !outcome.blown_tires.is_empty()
                        || !outcome.bent_rims.is_empty()
                        || !outcome.blown_dampers.is_empty();
                    if damaged {
                        info!("{}", report.log_block());
                        rs.watch = Some(ImpactWatch::start());
                        rs.last_impact = Some(report);
                        rs.episode_reported = true;
                    } else if rs.episode.as_ref().is_none_or(|(c, _)| closing > *c) {
                        rs.episode = Some((closing, report));
                    }
                }
                rs.rover.apply_external(force, torque);
            }
            // Episode ended this sub-step: if a notable hit didn't shear anything, emit it now.
            if !any_contact {
                if let Some((_, report)) = rs.episode.take() {
                    if !rs.episode_reported {
                        info!("{}", report.log_block());
                        rs.watch = Some(ImpactWatch::start());
                        rs.last_impact = Some(report);
                    }
                }
                rs.episode_reported = false;
            }
            rs.rover.step(&terrain, ROVER_SUBSTEP_DT);
            // Post-impact watch (WI 618): tally kraken / fall-through signs for a few seconds after an
            // impact, then write the verdict onto the last impact (console + HUD).
            if let Some(w) = rs.watch.as_mut() {
                w.max_speed = w.max_speed.max(rs.rover.body.velocity.length());
                w.max_omega = w.max_omega.max(rs.rover.body.angular_velocity().length());
                w.max_bounce = w.max_bounce.max(rs.rover.body.velocity.y.abs());
                w.min_height = w.min_height.min(rs.rover.height_above_terrain(&terrain));
                w.steps_left = w.steps_left.saturating_sub(1);
                if w.steps_left == 0 {
                    let result = w.finish();
                    info!("{}", ImpactReport::watch_block(&result));
                    if let Some(last) = rs.last_impact.as_mut() {
                        last.watch = Some(result);
                    }
                    rs.watch = None;
                }
            }
            // Fall damage (WI 630): accumulate downward speed while airborne; on touchdown, a hard
            // enough landing shears wheels (graded by their rated shear speed) and reports like an
            // impact. Normal bump-hopping stays below FALL_MIN_SPEED and is ignored.
            let vy = rs.rover.body.velocity.y;
            if rs.rover.airborne(&terrain) {
                rs.airborne = true;
                if vy < 0.0 {
                    rs.fall_peak = rs.fall_peak.max(-vy);
                }
            } else if rs.airborne {
                rs.airborne = false;
                let landing = rs.fall_peak;
                rs.fall_peak = 0.0;
                if landing > FALL_MIN_SPEED {
                    let outcome = rs.rover.shear_on_landing(landing);
                    for &idx in &outcome.sheared {
                        rs.drive.retain(|&x| x != idx);
                        rs.steer.retain(|&x| x != idx);
                    }
                    let report = ImpactReport {
                        speed: rs.rover.body.velocity.length(),
                        closing: landing,
                        impacted: "landing",
                        peak_wheel: outcome.peak_wheel,
                        demand: outcome.peak_demand,
                        capacity: outcome.peak_capacity,
                        sheared: outcome.sheared.clone(),
                        blown_tires: outcome.blown_tires.clone(),
                        bent_rims: outcome.bent_rims.clone(),
                        blown_dampers: outcome.blown_dampers.clone(),
                        watch: None,
                    };
                    info!("{}", report.log_block());
                    rs.watch = Some(ImpactWatch::start());
                    rs.last_impact = Some(report);
                }
            }
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
    world.accumulator += frame_dt * world.warp;
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
        // Wheel-station components (WI 630): the suspension strut, the metallic rim, and the rubber
        // tire. (Station render is refined in Phase C; these are recognisable placeholders.)
        PartKind::Suspension(susp) => (
            Mesh::from(Cylinder::new(s * 0.08, susp.rest_length as f32)),
            Color::srgb(0.55, 0.45, 0.2),
        ),
        PartKind::Rim(r) => (
            Mesh::from(Cylinder::new(r.radius as f32, (r.radius * 0.45) as f32)),
            Color::srgb(0.6, 0.62, 0.68),
        ),
        PartKind::Tire(t) => (
            Mesh::from(Cylinder::new(t.profile as f32 * 2.0, t.profile as f32)),
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

/// Where a wheel's mesh sits and how it aligns, given its suspension mount `hub` (world) and the body
/// `up`. For a quarter-car wheel (WI 631a) this is the **actual axle height** (`hub − axle_drop`
/// vertically), so the wheel visibly hops, squats under load, and droops when airborne — the ride
/// dynamics are seen, not re-derived. For a legacy wheel (no unsprung DOF) it falls back to the WI 630
/// rule: on the ground when in contact, else hanging at full suspension droop (so an airborne wheel
/// stays with the rover instead of shadowing the terrain). The returned normal aligns the axle: the
/// terrain normal when the tyre touches, the body up while airborne.
fn wheel_render_pose(w: &Wheel, hub: DVec3, terrain: &Terrain, up: DVec3) -> (DVec3, DVec3) {
    let ground = terrain.height(hub.x, hub.z);
    if w.unsprung_mass > 0.0 {
        let axle_y = hub.y - w.axle_drop;
        let center = DVec3::new(hub.x, axle_y, hub.z);
        let normal = if axle_y <= ground + w.radius + 1e-3 {
            terrain.normal(hub.x, hub.z)
        } else {
            up
        };
        (center, normal)
    } else {
        let contact_center = DVec3::new(hub.x, ground + w.radius, hub.z);
        let droop_center = hub - up * w.rest_length;
        if contact_center.y >= droop_center.y {
            (contact_center, terrain.normal(hub.x, hub.z))
        } else {
            (droop_center, up)
        }
    }
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
    cam: Res<crate::replay::ReplayCam>,
) {
    // During replay (WI 648) the recorded poses drive the meshes — don't overwrite them with the live
    // rover state (which is frozen anyway).
    if cam.is_playback() {
        return;
    }
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
            // A sheared-off wheel (WI 618) is hidden by collapsing its mesh to zero scale.
            if w.inert {
                tf.scale = Vec3::ZERO;
                continue;
            }
            // A blown tire runs on the smaller rim (WI 631b): shrink the mesh to the current radius
            // relative to the radius it was built at.
            tf.scale = Vec3::splat((w.radius as f32 / tag.1).clamp(0.05, 1.0));
            let hub = body.position + q * w.mount;
            let (center, align_normal) = wheel_render_pose(w, hub, &rs.terrain, up);
            // Include the bent-rim steer/camber bias (WI 631b) so a damaged wheel visibly points off.
            let steer_rot = DQuat::from_axis_angle(up, w.steer + w.steer_bias);
            let heading = steer_rot * fwd;
            let forward = (heading - align_normal * heading.dot(align_normal)).normalize_or_zero();
            let axle = align_normal.cross(forward).normalize_or_zero();
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

/// Positions the world-static test ramp mesh (WI 630), rover-anchored, tilted to the incline so its
/// top face is the surface the wheels climb.
fn track_ramp_mesh(world: Res<WorkshopWorld>, mut q: Query<&mut Transform, With<RoverRampMesh>>) {
    let Some(rs) = &world.rover else { return };
    let Some(r) = rs.terrain.ramp else { return };
    let anchor = rs.rover.body.position;
    // Centre of the inclined slab (base terrain is flat here, so y is half the peak height).
    let center = DVec3::new(
        r.center_x,
        0.5 * r.run * r.angle.tan(),
        r.start_z + 0.5 * r.run,
    );
    for mut tf in &mut q {
        tf.translation = (center - anchor).as_vec3();
        tf.translation.y -= 0.075; // sink half the slab thickness so the top face is the surface
        tf.rotation = Quat::from_rotation_x(-r.angle as f32);
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
        // A sheared wheel has no tyre, so no spokes.
        if w.inert {
            continue;
        }
        let hub = body.position + body.orientation * w.mount;
        let (center, align_normal) = wheel_render_pose(w, hub, terrain, up);
        let steer_rot = DQuat::from_axis_angle(up, w.steer);
        let heading = steer_rot * (body.orientation * DVec3::Z);
        let forward = (heading - align_normal * heading.dot(align_normal)).normalize_or_zero();
        let axle = align_normal.cross(forward).normalize_or_zero();
        let face = center + axle * (w.radius * 0.27); // just outside the tyre's outer face
        let spin = DQuat::from_axis_angle(axle, rs.spin_angle[i]);
        let a = spin * forward * (w.radius * 0.85);
        let b = spin * axle.cross(forward) * (w.radius * 0.85);
        // Spin-out indicator (WI 650): tint the spokes cool→hot by |slip| so wheelspin (which at high
        // spin aliases into a static-looking blur) reads at a glance — grippy is grey, lit-up is red.
        let slip = (w.slip_ratio.abs() / 1.0).clamp(0.0, 1.0) as f32;
        let spoke = Color::srgb(0.55, 0.55, 0.6).mix(&Color::srgb(1.0, 0.25, 0.1), slip);
        gizmos.line(to_render(face - a), to_render(face + a), spoke);
        gizmos.line(to_render(face - b), to_render(face + b), spoke);
    }
}

fn update_test_hud(
    world: Res<WorkshopWorld>,
    clock: Res<SimClock>,
    cam: Res<crate::replay::ReplayCam>,
    mut hud: Query<&mut Text, With<TestHud>>,
) {
    // A clear paused banner (WI 638); during replay (WI 648) show the scrub position instead.
    let paused = if cam.is_playback() {
        format!(
            "\n⏪ REPLAY {}/{} (R live · [ ] scrub)",
            cam.cursor() + 1,
            cam.len()
        )
    } else {
        crate::pause::paused_banner(&clock).to_string()
    };
    if let Ok(mut text) = hud.single_mut() {
        if let Some(rs) = &world.rover {
            let speed = rs.rover.body.velocity.length();
            let height = rs.rover.height_above_terrain(&rs.terrain);
            let live = rs.rover.wheels.iter().filter(|w| !w.inert).count();
            let impact = rs
                .last_impact
                .as_ref()
                .map(|r| format!("\n{}", r.hud_line()))
                .unwrap_or_default();
            // Per-wheel component readout (WI 630): the lead wheel's grip, effective radius, slip, and
            // ride (sprung vs riding on the tire), so swapping a component shows a measurable change.
            let tires = rs
                .rover
                .wheels
                .first()
                .map(|w| {
                    let ride = if w.rigid_suspension { "tire" } else { "sprung" };
                    format!(
                        "\ntire:   grip ×{:.2}  R {:.2} m  slip {:.1}/{:.1}  [{ride}]",
                        w.grip_scale, w.radius, w.slip_long, w.slip_lat,
                    )
                })
                .unwrap_or_default();
            text.0 = format!(
                "workshop · TEST (rover){paused}\nspeed:  {speed:6.2} m/s\nheight: {height:6.2} m\nwheels: {live}/{}{tires}\n{}:  {:3.0}%{impact}",
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
                    "workshop · TEST: {state}{paused}\nthrottle: {:3.0}%\naltitude: {:6.2} m\nv-speed:  {:+6.2} m/s\nspeed:    {:6.2} m/s\nSAS {sas}   warp {:.0}x",
                    world.throttle * 100.0,
                    world.altitude(),
                    world.body.velocity.y,
                    speed,
                    world.warp,
                );
            }
            CraftState::Fractured => {
                text.0 = format!(
                    "workshop · TEST: CRASHED — fractured into {} pieces{paused}\nBackspace to rebuild",
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
                station: None,
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

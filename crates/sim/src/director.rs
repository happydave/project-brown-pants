//! Scenario director (WI 550, content Slice 1).
//!
//! The runtime piece of the scenario layer: it takes a loaded, validated
//! scenario's **spawn payload** and instantiates the starting state into the
//! one-craft flight chain — the same [`FlightCraft`] + [`ActiveBody`] +
//! [`LaunchPad`] + [`flight_step`] pipeline the play/workshop scenes drive —
//! then advances it each frame. The craft is thereafter flown by ordinary
//! [`Command`]s through the tier-gated executor; the director has exactly the
//! authority a scene has, and mutates nothing outside the command routing.
//!
//! **How the spawn is command-driven.** The loader boundary (a scene, a test,
//! eventually a save-restore) stages the payload in [`PendingSpawn`] and posts
//! [`Command::SpawnScenario`]; [`apply_spawn_scenario`] — the structural arm,
//! the `SetGear` pattern — consumes the stage and performs the spawn. There is
//! no file IO in the command path: the payload was fully loaded and validated
//! before it was staged.
//!
//! **The catalog is consumed here.** Engine exhaust velocity / max mass flow
//! and tank capacity come from the scenario's device bindings into the
//! resolved catalog — final physical values with the balance scalars already
//! baked (the sim never sees a scalar). The workshop's hardcoded assembly
//! constants are exactly what this replaces for scenario craft.

use crate::active::ActiveBody;
use crate::attitude::{AttitudeControl, AttitudePilot, ReactionWheels, Sas};
use crate::command::Command;
use crate::content::{DeviceSpec, Record, Setting};
use crate::control::assemble_control;
use crate::flight::{flight_step, FlightCraft, FlightParams, GroundContact};
use crate::fluid::FluidMedium;
use crate::launch::LaunchPad;
use crate::medium::max_cross_section;
use crate::propulsion::{Engine, EngineCommand, Propulsion};
use crate::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use crate::scenario::{Scenario, StartPlacement};
use crate::sim::SimClock;
use crate::voxel::{DeviceKind, VoxelCraft};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_time::prelude::*;
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The scenario propellant resource (the flight chain's single-tank model).
const PROPELLANT: ResourceType = ResourceType(0);
/// Flight sub-step, s (the play/workshop convention).
const SUBSTEP_DT: f64 = 0.004;
/// Per-frame sub-step cap (the active-vehicle warp cap).
const MAX_SUBSTEPS: u32 = 250;
/// Bounded chunk of the WI 643 step budget consumed per frame while paused.
const STEP_CHUNK_SECONDS: f64 = 1.0 / 60.0;
/// Attitude authority + reaction-wheel defaults (the workshop's assembly
/// constants — control-system tuning stays code/blueprint-authored this
/// slice; the catalog owns engine/tank physics).
const ATTITUDE_AUTHORITY: f64 = 5_000.0;
const WHEEL_TORQUE: f64 = 8_000.0;
const WHEEL_MOMENTUM: f64 = 1e9;
const LOW_POWER_RESERVE: f64 = 6.0;

/// Everything the director needs to spawn a scenario's starting state —
/// **fully resolved**: catalog values are final physical numbers, the world is
/// reduced to its root-body constants. Serde-able (it is the operand of a
/// [`Command`]-driven structural change, staged rather than carried inline).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSpawn {
    /// Scenario identity (surfaced on telemetry).
    pub id: String,
    /// Display name.
    pub name: String,
    /// The starting craft's lattice (from the blueprint).
    pub craft: VoxelCraft,
    /// Engine physics from the bound catalog record, if the craft has engines:
    /// (exhaust velocity m/s, max mass flow kg/s).
    pub engine: Option<(f64, f64)>,
    /// Per-tank propellant capacity (kg) from the bound catalog record, if the
    /// craft has tanks.
    pub tank_capacity: Option<f64>,
    /// Root-body gravitational parameter, m³/s².
    pub mu: f64,
    /// Root-body surface radius, m.
    pub surface_radius: f64,
    /// The root body's fluid-medium field.
    pub medium: FluidMedium,
    /// Where the craft starts.
    pub placement: StartPlacement,
    /// The composed balance settings (names are a semi-public contract;
    /// surfaced on telemetry as "real × named modifier").
    pub settings: BTreeMap<String, Setting>,
}

impl ScenarioSpawn {
    /// Resolves a loaded scenario into its spawn payload: device bindings →
    /// final catalog values, world → root-body constants.
    ///
    /// The scenario was validated at load (bindings exist, classes match,
    /// required classes bound), so the lookups here cannot fail; they are
    /// written defensively anyway (`None` when absent) because a payload can
    /// also be constructed by hand in tests.
    pub fn from_scenario(s: &Scenario) -> ScenarioSpawn {
        let device_spec = |class: crate::content::DeviceClass| {
            s.bindings
                .get(&class)
                .and_then(|id| s.catalog.get(id))
                .and_then(|e| match &e.record {
                    Record::Device(d) => Some(d.spec.clone()),
                    _ => None,
                })
        };
        let engine = device_spec(crate::content::DeviceClass::Engine).and_then(|spec| match spec {
            DeviceSpec::Engine {
                exhaust_velocity,
                max_mass_flow,
            } => Some((exhaust_velocity, max_mass_flow)),
            _ => None,
        });
        let tank_capacity =
            device_spec(crate::content::DeviceClass::Tank).and_then(|spec| match spec {
                DeviceSpec::Tank { capacity } => Some(capacity),
                _ => None,
            });
        let body = s.root_asset.central_body();
        ScenarioSpawn {
            id: s.id.clone(),
            name: s.name.clone(),
            craft: s.blueprint.clone(),
            engine,
            tank_capacity,
            mu: body.mu,
            surface_radius: body.radius,
            medium: s.root_asset.fluid_medium(),
            placement: s.placement,
            settings: s.catalog.settings.clone(),
        }
    }
}

/// The staged spawn payload [`Command::SpawnScenario`] consumes. The loader
/// boundary inserts it (validated), the command triggers it, the director
/// takes it. `None` when nothing is staged (the command is then a no-op).
#[derive(Resource, Default)]
pub struct PendingSpawn(pub Option<ScenarioSpawn>);

/// The spawned scenario flight — the one-craft chain as a resource, stepped by
/// the director and flown through the ordinary command systems.
#[derive(Resource)]
pub struct ScenarioFlight {
    /// Scenario identity (telemetry).
    pub id: String,
    /// Display name (HUD).
    pub name: String,
    /// The rigid body.
    pub body: ActiveBody,
    /// The assembled craft (propulsion, attitude, control).
    pub craft: FlightCraft,
    /// Step parameters (world constants + drag reference).
    pub params: FlightParams,
    /// The launch pad (rest + release).
    pub pad: LaunchPad,
    /// The composed balance settings (telemetry).
    pub settings: BTreeMap<String, Setting>,
    /// Frame-time accumulator for fixed sub-stepping.
    pub accumulator: f64,
}

/// Assembles the spawn payload into the one-craft chain. `None` for an empty
/// lattice (no mass) — the same honest failure as the workshop. Engines take
/// their physics from the payload's catalog-resolved values; a craft with
/// engine devices but no engine parameters assembles **engine-less** rather
/// than inventing numbers (the validated path can't produce that state).
pub fn instantiate(spawn: &ScenarioSpawn) -> Option<ScenarioFlight> {
    let voxels = spawn.craft.clone();
    let mp = voxels.mass_properties()?;
    let s = voxels.cell_size;
    let com = mp.center_of_mass;

    let engine_cells: Vec<glam::IVec3> = voxels
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

    // Catalog-resolved physics: propellant mass per tank, engine performance.
    let propellant = match (&spawn.engine, spawn.tank_capacity) {
        (Some(_), Some(capacity)) if !engine_cells.is_empty() => tanks.max(1) as f64 * capacity,
        _ => 0.0,
    };
    let engines: Vec<Engine> = match spawn.engine {
        Some((exhaust_velocity, max_mass_flow)) => engine_cells
            .iter()
            .map(|c| Engine {
                tank: ReservoirId(0),
                exhaust_velocity,
                max_mass_flow,
                // Thrust along +Y through the CoM in X/Z (the workshop's
                // fly-straight convention).
                mount: DVec3::new(com.x, c.y as f64 * s, com.z),
                axis: DVec3::Y,
                max_gimbal: 0.0,
            })
            .collect(),
        None => Vec::new(),
    };
    let n_engines = engines.len();
    let mut propulsion = Propulsion {
        graph: ResourceGraph {
            reservoirs: vec![Reservoir::new(PROPELLANT, propellant, propellant)],
            ..Default::default()
        },
        // Two mounts, both at the CoM: the propellant tank, and the battery
        // reservoir `assemble_control` appends. `wet_mass` counts *every*
        // reservoir's amount as mass at its mount (pre-existing quirk — an
        // unmounted reservoir masses at the lattice origin), and an offset
        // wet CoM turns thrust-through-the-dry-CoM into a constant tipping
        // torque the SAS cannot hold. Mounting both at the CoM keeps
        // wet CoM = dry CoM, so the craft flies straight.
        tank_mounts: vec![com, com],
        engines,
        commands: vec![EngineCommand::default(); n_engines],
    };
    let mut control = assemble_control(&voxels, &mut propulsion.graph);
    control.low_power_reserve = LOW_POWER_RESERVE;
    let attitude = AttitudePilot {
        sas: Sas::default(),
        manual: DVec3::ZERO,
        authority: ATTITUDE_AUTHORITY,
        recapture_on_release: true,
        actuators: AttitudeControl {
            wheels: Some(ReactionWheels::new(WHEEL_TORQUE, WHEEL_MOMENTUM)),
            rcs: None,
        },
    };

    // Placement: only Pad exists this slice — at rest on the root body's
    // surface, supported by the launch pad until thrust beats weight.
    let StartPlacement::Pad = spawn.placement;
    let rest_radius = spawn.surface_radius + com.y;
    let body = ActiveBody::new(
        DVec3::new(0.0, rest_radius, 0.0),
        DVec3::ZERO,
        mp.mass + propellant,
        mp.inertia,
    );
    let pad = LaunchPad::resting(rest_radius);

    let params = FlightParams {
        mu: spawn.mu,
        surface_radius: spawn.surface_radius,
        medium: spawn.medium,
        drag_area: max_cross_section(&voxels),
        drag_coefficient: 1.0,
        lift: None,
        // A ground plane at the surface so a craft that comes back down
        // lands/rests instead of tunnelling (WI 592 model).
        ground: Some(GroundContact {
            normal: DVec3::Y,
            offset: spawn.surface_radius,
            contact: Default::default(),
        }),
    };

    Some(ScenarioFlight {
        id: spawn.id.clone(),
        name: spawn.name.clone(),
        body,
        craft: FlightCraft {
            dry_mass: mp.mass,
            dry_com: com,
            voxels,
            propulsion,
            attitude,
            control,
            autopilot: None,
        },
        params,
        pad,
        settings: spawn.settings.clone(),
        accumulator: 0.0,
    })
}

/// The structural arm for [`Command::SpawnScenario`] (the `SetGear` pattern):
/// consumes the staged payload and inserts the spawned [`ScenarioFlight`].
/// No stage ⇒ no-op (logged); an empty-lattice payload ⇒ no spawn (logged).
///
/// The director's initial state is issued as **ordinary commands**: SAS
/// hold-attitude on spawn (a fixed engine below the CoM is pendulum-unstable;
/// the starter carries a Tier-0 command core exactly so its first flight
/// flies straight — a craft without one simply ignores the command, the
/// tier gate's honest behaviour).
fn apply_spawn_scenario(
    mut messages: ParamSet<(MessageReader<Command>, MessageWriter<Command>)>,
    mut pending: ResMut<PendingSpawn>,
    mut commands: Commands,
) {
    let triggered = messages
        .p0()
        .read()
        .any(|cmd| matches!(cmd, Command::SpawnScenario));
    if !triggered {
        return;
    }
    match pending.0.take() {
        Some(spawn) => match instantiate(&spawn) {
            Some(flight) => {
                bevy_log::info!(
                    "scenario `{}` spawned: {} on the pad",
                    flight.id,
                    flight.name
                );
                commands.insert_resource(flight);
                messages
                    .p1()
                    .write(Command::SetSas(crate::command::SasMode::Hold));
            }
            None => bevy_log::warn!("scenario spawn: blueprint has no mass — nothing spawned"),
        },
        None => bevy_log::warn!("SpawnScenario with no staged payload — no-op"),
    }
}

/// Routes flight commands (throttle, attitude, SAS, autopilot, tier) from the
/// message envelope into the spawned craft's tier-gated applicator — so the
/// scenario craft is flown by ordinary [`Command`]s from any source (keyboard,
/// bus/MCP, a future mission effect) through one path.
fn apply_flight_commands(
    mut reader: MessageReader<Command>,
    flight: Option<ResMut<ScenarioFlight>>,
) {
    let Some(mut flight) = flight else { return };
    for cmd in reader.read() {
        let orientation = flight.body.orientation;
        flight.craft.apply_command(cmd, orientation);
    }
}

/// Advances the spawned flight with fixed sub-steps, honouring pause + the
/// WI 643 step budget and the command-driven [`SimClock::warp`].
fn step_scenario_flight(
    time: Res<Time>,
    mut clock: ResMut<SimClock>,
    flight: Option<ResMut<ScenarioFlight>>,
) {
    let Some(mut flight) = flight else { return };
    let dt = if !clock.paused {
        time.delta_secs_f64()
    } else if clock.step_budget > 0.0 {
        let chunk = clock.step_budget.min(STEP_CHUNK_SECONDS);
        clock.step_budget -= chunk;
        chunk
    } else {
        return;
    };
    flight.accumulator += dt * clock.warp;
    let mut n = 0;
    while flight.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS {
        let ScenarioFlight {
            body,
            craft,
            params,
            pad,
            ..
        } = &mut *flight;
        // `SimClock.time` is NOT advanced here — the orbit plugin's
        // `advance_clock` owns it (advancing it from two places would
        // double-count sim time in the composed app).
        flight_step(body, craft, params, pad, SUBSTEP_DT);
        flight.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

/// The scenario director: registers the command message (idempotent with
/// [`crate::command::FlightControlPlugin`]), the pending-spawn stage, the
/// structural spawn arm, and the flight stepper. Headless — composes into the
/// windowed app and into a test [`App`] identically.
pub struct DirectorPlugin;

impl Plugin for DirectorPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Command>()
            .init_resource::<PendingSpawn>()
            .add_systems(
                Update,
                (
                    apply_spawn_scenario,
                    apply_flight_commands,
                    step_scenario_flight,
                )
                    .chain(),
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{BatterySpec, ControlComputer};
    use crate::voxel::{Device, Material, Voxel};
    use glam::IVec3;

    /// The starter blueprint (also the shipped `content/blueprints` artifact):
    /// a light 0.35 m-cell build — one composite cell, crewed control point,
    /// Tier-0 computer, battery, engine, tank — sized so the shipped small
    /// engine (3200 × 0.85 × 0.9 m/s × 1.5 kg/s ≈ 3.7 kN) out-thrusts its
    /// ~310 kg all-up weight (with the 100 kg starter tank): it rests, then
    /// lifts off at full throttle. Note the battery charge counts toward wet
    /// mass (the pre-existing `wet_mass` sums *every* reservoir as kg), so
    /// the starter carries a small 50-unit battery.
    fn blueprint() -> VoxelCraft {
        let mut craft = VoxelCraft::new(0.35);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::COMPOSITE,
        });
        craft
            .devices
            .push(Device::control_point(IVec3::new(0, 2, 0), 40.0, true));
        craft.devices.push(Device::computer(
            IVec3::new(0, 2, 0),
            5.0,
            ControlComputer::command_core(0.5),
        ));
        craft.devices.push(Device::battery(
            IVec3::new(0, 1, 0),
            10.0,
            BatterySpec::full(50.0),
        ));
        craft
            .devices
            .push(Device::structural(IVec3::ZERO, 30.0, DeviceKind::Engine));
        craft.devices.push(Device::structural(
            IVec3::new(0, 1, 0),
            5.0,
            DeviceKind::Tank,
        ));
        craft
    }

    fn spawn_payload() -> ScenarioSpawn {
        ScenarioSpawn {
            id: "s".into(),
            name: "S".into(),
            craft: blueprint(),
            engine: Some((2448.0, 2.0)),
            tank_capacity: Some(100.0),
            mu: crate::sim::CentralBody::EARTHLIKE.mu,
            surface_radius: crate::sim::CentralBody::EARTHLIKE.radius,
            medium: FluidMedium::EARTHLIKE,
            placement: StartPlacement::Pad,
            settings: BTreeMap::new(),
        }
    }

    #[test]
    fn instantiate_assembles_catalog_values_and_pad_rest() {
        let spawn = spawn_payload();
        let flight = instantiate(&spawn).unwrap();
        // Engine physics are the payload's catalog-resolved values.
        assert_eq!(flight.craft.propulsion.engines.len(), 1);
        assert_eq!(flight.craft.propulsion.engines[0].exhaust_velocity, 2448.0);
        assert_eq!(flight.craft.propulsion.engines[0].max_mass_flow, 2.0);
        // Propellant = tanks × bound capacity.
        assert_eq!(flight.craft.propulsion.graph.reservoirs[0].amount, 100.0);
        // At rest on the pad at surface radius + CoM height.
        assert!(!flight.pad.released);
        assert!(flight.body.position.y > spawn.surface_radius);
        assert_eq!(flight.body.velocity, DVec3::ZERO);
    }

    #[test]
    fn empty_lattice_does_not_instantiate() {
        let mut spawn = spawn_payload();
        spawn.craft = VoxelCraft::new(1.0);
        assert!(instantiate(&spawn).is_none());
    }

    /// Regenerates the shipped starter blueprint. Run explicitly when the
    /// fixture craft changes:
    /// `cargo test -p sounding_sim --lib write_first_flight_blueprint -- --ignored`
    #[test]
    #[ignore = "writes the shipped content/blueprints/first-flight.json artifact"]
    fn write_first_flight_blueprint() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/blueprints");
        let path = crate::library::save_craft(&dir, "First Flight", &blueprint()).unwrap();
        assert!(path.ends_with("first-flight.json"));
    }

    /// The shipped fixture, end to end: content/scenarios/first-flight.ron
    /// composes the three Slice 0 example documents + the starter blueprint,
    /// and the assembled engine carries the provenance-honest value
    /// 3200 (pack, physical) × 0.85 (setting) × 0.9 (scenario override).
    #[test]
    fn shipped_first_flight_scenario_composes_end_to_end() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let roots = crate::scenario::ScenarioRoots {
            content: root.join("content"),
            saves: root.join("saves"),
        };
        let s = crate::scenario::load_scenario(
            &root.join("content/scenarios/first-flight.ron"),
            &roots,
        )
        .unwrap();
        assert_eq!(s.name, "First Flight");
        let spawn = ScenarioSpawn::from_scenario(&s);
        assert_eq!(spawn.engine, Some((3200.0 * 0.85 * 0.9, 1.5)));
        assert_eq!(spawn.tank_capacity, Some(100.0));
        // The composed settings ride the payload for telemetry, rationale included.
        assert!(spawn.settings.contains_key("engine_efficiency"));
        let flight = instantiate(&spawn).unwrap();
        assert_eq!(
            flight.craft.propulsion.engines[0].exhaust_velocity,
            3200.0 * 0.85 * 0.9
        );
        assert_eq!(
            flight.craft.propulsion.graph.reservoirs[0].amount, 100.0,
            "one shipped tank × bound capacity"
        );
        assert!(!flight.pad.released, "starts at rest on the pad");
    }

    /// Advances the paused app by `seconds` of sim time through the WI 643
    /// step budget — deterministic, wall-clock-free test time.
    fn step_sim(app: &mut App, seconds: f64) {
        app.world_mut().write_message(Command::Step { seconds });
        // Budget is consumed in bounded chunks, one per update.
        let updates = (seconds / STEP_CHUNK_SECONDS).ceil() as usize + 2;
        for _ in 0..updates {
            app.update();
        }
    }

    #[test]
    fn spawn_is_command_driven_then_flies_on_commands() {
        let mut app = App::new();
        app.add_plugins(bevy_time::TimePlugin);
        app.init_resource::<SimClock>();
        // The real executor applies SetPaused/Step; the director applies the
        // spawn and routes flight commands — all through one envelope.
        app.add_plugins(crate::command::FlightControlPlugin);
        app.add_plugins(DirectorPlugin);

        // Nothing staged: the command is a no-op, not a panic.
        app.world_mut().write_message(Command::SpawnScenario);
        app.update();
        assert!(app.world().get_resource::<ScenarioFlight>().is_none());

        // Stage + trigger: the director spawns the chain.
        app.world_mut().resource_mut::<PendingSpawn>().0 = Some(spawn_payload());
        app.world_mut().write_message(Command::SpawnScenario);
        app.update();
        let rest_y = {
            let flight = app.world().resource::<ScenarioFlight>();
            assert_eq!(flight.id, "s");
            assert!(!flight.pad.released);
            flight.body.position.y
        };

        // Freeze the clock and drive time by command: at zero throttle the
        // pad holds the craft at exact rest.
        app.world_mut().write_message(Command::SetPaused(true));
        app.update();
        step_sim(&mut app, 1.0);
        {
            let flight = app.world().resource::<ScenarioFlight>();
            assert!(!flight.pad.released);
            assert!((flight.body.position.y - rest_y).abs() < 1e-9, "held");
        }

        // Full throttle by ordinary command: the engine out-thrusts the
        // weight, the pad releases, and the craft climbs.
        app.world_mut().write_message(Command::SetThrottle(1.0));
        app.update();
        step_sim(&mut app, 4.0);
        let flight = app.world().resource::<ScenarioFlight>();
        assert!(flight.pad.released, "thrust > weight releases the pad");
        assert!(flight.body.position.is_finite());
        assert!(
            flight.body.position.y > rest_y + 1.0,
            "climbing: {} vs rest {}",
            flight.body.position.y,
            rest_y
        );
        assert!(flight.body.velocity.y > 0.0, "ascending");
        // The spawn-time SAS hold keeps the unguided ascent upright (the
        // starter's command core doing its job).
        assert!(
            (flight.body.orientation * glam::DVec3::Y).y > 0.99,
            "upright under SAS hold"
        );
        // Propellant drew from the catalog-bound tank over the burn.
        assert!(flight.craft.propulsion.graph.reservoirs[0].amount < 100.0);
    }
}

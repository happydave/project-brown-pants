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
use crate::mission::{Effect, Mission, MissionState, NodeState, Offer};
use crate::propulsion::{Engine, EngineCommand, Propulsion};
use crate::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
use crate::scenario::{Scenario, StartPlacement};
use crate::session::GameSession;
use crate::sim::SimClock;
use crate::voxel::{DeviceKind, VoxelCraft};
use crate::world_save::{ContentIdentity, MissionSave, ScenarioSaveState};
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
/// Mission-evaluation poll cadence, sim-seconds of flight time (WI 551) —
/// bounded, warp-safe (latching makes coarse polls defined semantics).
const MISSION_POLL_SECONDS: f64 = 0.25;
/// Attitude authority + reaction-wheel defaults (the workshop's assembly
/// constants — control-system tuning stays code/blueprint-authored this
/// slice; the catalog owns engine/tank physics).
///
/// SCAFFOLD: these should be catalog device records (reaction wheels / RCS as
/// content) resolved through scenario bindings like engines/tanks are, not
/// engine constants.
const ATTITUDE_AUTHORITY: f64 = 5_000.0;
const WHEEL_TORQUE: f64 = 8_000.0;
const WHEEL_MOMENTUM: f64 = 1e9;
const LOW_POWER_RESERVE: f64 = 6.0;
/// Standard gravity for the G-force readout, m/s² (WI 739).
const G0: f64 = 9.80665;
/// Ambient / radiative-sink temperature for an orbit-entry spawn, K (WI 739,
/// the dive's value).
///
/// SCAFFOLD: thermal ambience is a property of the world/body (content), not
/// a director constant.
const ORBIT_AMBIENT_K: f64 = 250.0;
/// Initial rails-coast time-warp on an orbit-entry spawn (the entry trigger
/// drops it back to 1× at the interface).
///
/// SCAFFOLD: a pacing choice the scenario document should make.
const RAILS_COAST_WARP: f64 = 30.0;

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
    /// The scenario's mission definitions, in declared order (WI 551).
    #[serde(default)]
    pub missions: Vec<Mission>,
    /// The resolved-content identity (WI 553) — recorded into world saves.
    #[serde(default)]
    pub content: ContentIdentity,
    /// Saved state to restore after assembly (WI 553): resume = fresh spawn
    /// + dynamic overwrite, one spawn path. Boxed — it embeds a full craft.
    #[serde(default)]
    pub resume: Option<Box<ScenarioSaveState>>,
    /// Per-body world-save records for this world (WI 891): built from the
    /// scenario's resolved assets (root ⇒ snapshot tier, the rest ⇒ digest
    /// tier — the production default), carried to the flight and written into
    /// world saves by `capture`. On resume, loaded snapshot pins are
    /// preserved verbatim (`world_save::reconcile_body_records`).
    #[serde(default)]
    pub bodies: Vec<crate::world_save::SavedBodyRecord>,
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
            missions: s.missions.clone(),
            content: ContentIdentity::from_scenario(s),
            resume: None,
            bodies: crate::world_save::body_records(
                &s.assets,
                &std::iter::once(s.root_asset.id.clone()).collect(),
            ),
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
    /// Mission runtime state, in declared order (WI 551).
    pub missions: Vec<MissionRun>,
    /// The most recent lore beat surfaced by a mission effect (WI 551).
    pub lore: Option<String>,
    /// Integrated flight sim time, s — the evaluator's poll clock (the orbit
    /// plugin owns `SimClock.time`; this is the scenario flight's own time,
    /// warp-scaled by construction).
    pub elapsed: f64,
    /// Flight time of the last mission poll.
    pub last_poll: f64,
    /// The played session — Launch → Flight → Recovery with landed/crashed
    /// outcome, driven by simulation state (WI 739, absorbing the play/
    /// autopilot scenes' session tracking). Terminal Recovery freezes the
    /// stepper, matching the migrated scenes' behaviour.
    pub session: GameSession,
    /// Felt (proper) acceleration over the last sub-step, in g (WI 739).
    pub g_force: f64,
    /// The resolved-content identity this flight was composed from (WI 553)
    /// — what a world save records.
    pub content: ContentIdentity,
    /// Per-body world-save records (WI 891) — what `capture` writes. On a
    /// resumed flight, loaded snapshot pins were preserved verbatim so a
    /// re-save never silently re-stamps a pinned body's output version.
    pub bodies: Vec<crate::world_save::SavedBodyRecord>,
}

/// One mission's runtime state: its definition, lifecycle state, and the
/// latch tree mirroring its objective (WI 551).
pub struct MissionRun {
    /// The authored definition.
    pub def: Mission,
    /// Lifecycle state (Pending / Active / Completed).
    pub state: MissionState,
    /// Objective latch state (monotone progress).
    pub nodes: NodeState,
}

impl MissionRun {
    fn new(def: Mission) -> MissionRun {
        let state = match def.offer {
            Offer::Immediate => MissionState::Active,
            Offer::AfterMission(_) => MissionState::Pending,
        };
        let nodes = NodeState::for_condition(&def.objective);
        MissionRun { def, state, nodes }
    }
}

impl ScenarioFlight {
    /// The scenario telemetry block — **one construction** shared by the bus
    /// publisher and the mission evaluator, so objective leaves query exactly
    /// the shape the wire serves (WI 551).
    pub fn telemetry(&self) -> crate::telemetry::ScenarioTelemetry {
        crate::telemetry::ScenarioTelemetry {
            id: self.id.clone(),
            name: self.name.clone(),
            settings: self.settings.clone(),
            altitude: self.pad.altitude(&self.body),
            speed: self.body.velocity.length(),
            airborne: self.pad.released,
            elapsed: self.elapsed,
            session: Some(self.session),
            g_force: self.g_force,
            missions: self
                .missions
                .iter()
                .map(|m| crate::telemetry::MissionTelemetry {
                    id: m.def.id.clone(),
                    name: m.def.name.clone(),
                    state: m.state,
                    progress: m.nodes.progress(&m.def.objective),
                })
                .collect(),
            lore: self.lore.clone(),
        }
    }
}

/// Assembles a **Pad-placement** spawn payload into the one-craft chain.
/// `None` for an empty lattice (no mass) — the same honest failure as the
/// workshop — or for a non-Pad placement (those spawn through their own arms;
/// see [`apply_spawn_scenario`]). Engines take their physics from the
/// payload's catalog-resolved values; a craft with engine devices but no
/// engine parameters assembles **engine-less** rather than inventing numbers
/// (the validated path can't produce that state).
pub fn instantiate(spawn: &ScenarioSpawn) -> Option<ScenarioFlight> {
    if !matches!(spawn.placement, StartPlacement::Pad) {
        return None;
    }
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
    // SCAFFOLD: one engine record for every Engine device, thrust forced
    // along +Y through the CoM with no gimbal, and one pooled propellant
    // reservoir of `tanks × capacity` — proper per-device assembly mounts
    // each engine/tank at its own cell with its own record binding and
    // gimbal. (Resource mass-typing itself landed with WI 810.)
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
        // The pooled propellant tank masses at the CoM, so wet CoM = dry CoM
        // and thrust-through-the-CoM flies straight. The battery reservoir
        // `assemble_control` appends needs no mount: it is massless (WI 810).
        // SCAFFOLD: the mount belongs at the tank devices' cells; goes away
        // with per-device assembly.
        tank_mounts: vec![com],
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

    // Pad placement: at rest on the root body's surface, supported by the
    // launch pad until thrust beats weight.
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

    let mut flight = ScenarioFlight {
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
        missions: spawn
            .missions
            .iter()
            .cloned()
            .map(MissionRun::new)
            .collect(),
        lore: None,
        elapsed: 0.0,
        last_poll: 0.0,
        session: {
            let mut s = GameSession::new();
            s.begin_launch();
            s
        },
        g_force: 1.0,
        content: spawn.content.clone(),
        bodies: spawn.bodies.clone(),
    };
    if let Some(saved) = &spawn.resume {
        apply_resume(&mut flight, saved, &spawn.missions);
        // Carry loaded snapshot pins through the resume verbatim (WI 891) —
        // a rebuilt record must not re-stamp a pin's recorded output version.
        flight.bodies =
            crate::world_save::reconcile_body_records(spawn.bodies.clone(), &saved.bodies);
    }
    Some(flight)
}

/// Overwrites a freshly-assembled flight with saved dynamic state (WI 553):
/// resume = fresh spawn + overwrite, so there is exactly one assembly path.
///
/// The restored craft replaces the assembled one wholesale (it carries its
/// own reservoirs, SAS, control, autopilot); the body's dynamic degrees of
/// freedom come from the save while mass stays stepper-derived (the flight
/// step recomputes wet mass from the craft's reservoirs every sub-step) and
/// inertia is rebuilt from the restored lattice, so nothing saved can
/// disagree with what the stepper derives. The mission evaluator's poll
/// clock re-seeds from the restored elapsed time (no burst poll); the dry
/// mass/CoM caches re-derive from the restored lattice.
fn apply_resume(flight: &mut ScenarioFlight, saved: &ScenarioSaveState, defs: &[Mission]) {
    flight.craft = saved.craft.clone();
    if let Some(mp) = flight.craft.voxels.mass_properties() {
        flight.craft.dry_mass = mp.mass;
        flight.craft.dry_com = mp.center_of_mass;
        flight.body = ActiveBody::new(
            saved.body.position,
            saved.body.velocity,
            mp.mass,
            mp.inertia,
        );
    } else {
        flight.body.position = saved.body.position;
        flight.body.velocity = saved.body.velocity;
    }
    flight.body.orientation = saved.body.orientation;
    flight.body.angular_momentum = saved.body.angular_momentum;
    flight.pad = saved.pad;
    flight.session = saved.session;
    flight.elapsed = saved.elapsed;
    flight.last_poll = saved.elapsed;
    flight.lore = saved.lore.clone();
    flight.g_force = saved.g_force;
    let (missions, report) = reconcile_missions(defs, &saved.missions);
    flight.missions = missions;
    for line in report {
        bevy_log::warn!("resume: {line}");
    }
}

/// Rebuilds mission runtime state from saved rows against the re-resolved
/// definitions (WI 553's drift-tolerant reconcile):
///
/// - a saved row applies where its id exists **and** its latch tree still
///   matches the current objective's shape; a shape mismatch resets that
///   mission's latches (reported);
/// - a saved id absent from the current scenario is dropped (reported);
/// - a definition with no saved row starts fresh;
/// - `AfterMission` activation is then recomputed from restored completion,
///   so a completed prerequisite's successor is Active after resume.
pub fn reconcile_missions(
    defs: &[Mission],
    saved: &[MissionSave],
) -> (Vec<MissionRun>, Vec<String>) {
    let mut report = Vec::new();
    let mut runs: Vec<MissionRun> = defs.iter().cloned().map(MissionRun::new).collect();
    for row in saved {
        match runs.iter_mut().find(|r| r.def.id == row.id) {
            Some(run) => {
                // The lifecycle state always applies (a Completed mission
                // must stay Completed — its effects were already issued);
                // the latch tree applies only while it still matches the
                // re-resolved objective's shape.
                run.state = row.state;
                if row.nodes.matches(&run.def.objective) {
                    run.nodes = row.nodes.clone();
                } else {
                    report.push(format!(
                        "mission `{}` objective changed since the save — progress reset",
                        row.id
                    ));
                }
            }
            None => report.push(format!(
                "mission `{}` is no longer in the scenario — saved state dropped",
                row.id
            )),
        }
    }
    // Recompute AfterMission activation from restored completion.
    let completed: Vec<String> = runs
        .iter()
        .filter(|r| r.state == MissionState::Completed)
        .map(|r| r.def.id.clone())
        .collect();
    for run in &mut runs {
        if run.state == MissionState::Pending {
            if let Offer::AfterMission(dep) = &run.def.offer {
                if completed.iter().any(|c| c == dep) {
                    run.state = MissionState::Active;
                }
            }
        }
    }
    (runs, report)
}

/// The structural arm for [`Command::SpawnScenario`] (the `SetGear` pattern):
/// consumes the staged payload and spawns the placement's regime. No stage ⇒
/// no-op (logged); an empty-lattice payload ⇒ no spawn (logged).
///
/// **Pad** inserts the [`ScenarioFlight`] one-craft chain. **Orbit** (WI 739,
/// the dive) configures the app's single on-rails craft entity instead: the
/// entry orbit, a real gear state, the diving description + thermal state,
/// and the entry-interface resource — the existing sim plugins
/// ([`crate::handoff::HandoffPlugin`], the [`EntryInterface`] trigger,
/// [`crate::medium::DescentPlugin`]) then run the rails → wake → descent
/// chain; the director configures, it does not step.
///
/// The director's initial state is issued as **ordinary commands**: SAS
/// hold-attitude on a Pad spawn (a fixed engine below the CoM is
/// pendulum-unstable; the starter carries a Tier-0 command core exactly so
/// its first flight flies straight — a craft without one simply ignores the
/// command, the tier gate's honest behaviour), and the rails-coast time-warp
/// on an Orbit spawn (the entry trigger's warp filter drops it back to 1×).
#[allow(clippy::type_complexity)]
fn apply_spawn_scenario(
    mut messages: ParamSet<(MessageReader<Command>, MessageWriter<Command>)>,
    mut pending: ResMut<PendingSpawn>,
    mut commands: Commands,
    mut rails: Query<(
        Entity,
        &mut crate::sim::Craft,
        &mut crate::handoff::GearState,
    )>,
) {
    let triggered = messages
        .p0()
        .read()
        .any(|cmd| matches!(cmd, Command::SpawnScenario));
    if !triggered {
        return;
    }
    let Some(spawn) = pending.0.take() else {
        bevy_log::warn!("SpawnScenario with no staged payload — no-op");
        return;
    };
    // A saved-state resume is honest only for the Pad flight family (WI 553);
    // if the recorded scenario's placement drifted to another regime, the
    // spawn proceeds fresh and says so.
    if spawn.resume.is_some() && !matches!(spawn.placement, StartPlacement::Pad) {
        bevy_log::warn!(
            "resume: scenario `{}` placement is no longer Pad — saved state ignored, spawning fresh",
            spawn.id
        );
    }
    match spawn.placement {
        StartPlacement::Pad => {
            let resumed = spawn.resume.is_some();
            match instantiate(&spawn) {
                Some(flight) => {
                    bevy_log::info!(
                        "scenario `{}` spawned: {} on the pad{}",
                        flight.id,
                        flight.name,
                        if resumed { " (resumed from save)" } else { "" }
                    );
                    commands.insert_resource(flight);
                    if !resumed {
                        // The restored craft carries its own SAS state; only a
                        // fresh spawn gets the director's hold-attitude nudge.
                        messages
                            .p1()
                            .write(Command::SetSas(crate::command::SasMode::Hold));
                    }
                }
                None => bevy_log::warn!("scenario spawn: blueprint has no mass — nothing spawned"),
            }
        }
        StartPlacement::Orbit {
            altitude,
            speed,
            interface,
        } => {
            let Some(mp) = spawn.craft.mass_properties() else {
                bevy_log::warn!("scenario spawn: blueprint has no mass — nothing spawned");
                return;
            };
            let Ok((entity, mut craft, mut gear)) = rails.single_mut() else {
                bevy_log::warn!("orbit-entry spawn: no on-rails craft entity — nothing spawned");
                return;
            };
            // The entry orbit: start at the +Y high point with +X tangential
            // velocity (the dive scene's flat-ground render convention).
            let r0 = spawn.surface_radius + altitude;
            let Some(orbit) = crate::orbit::Orbit::from_state(
                spawn.mu,
                glam::DVec2::new(0.0, r0),
                glam::DVec2::new(speed, 0.0),
                0.0,
            ) else {
                bevy_log::warn!("orbit-entry spawn: unbound start state — nothing spawned");
                return;
            };
            craft.orbit = orbit;
            *gear = crate::handoff::GearState::new(mp.mass, mp.inertia);
            let descent = crate::medium::DescentParams {
                medium: spawn.medium,
                mu: spawn.mu,
                surface_radius: spawn.surface_radius,
                drag_area: max_cross_section(&spawn.craft),
                drag_coefficient: 1.0,
                slam_coefficient: crate::medium::DEFAULT_SLAM_COEFFICIENT,
            };
            // SCAFFOLD: forward axis +Z and drag coefficient 1.0 are
            // conventions the blueprint/catalog should declare.
            let glide =
                crate::medium::GlideParams::for_craft(descent, &spawn.craft, crate::voxel::Axis::Z);
            let thermal = crate::medium::CraftThermal::new(
                &spawn.craft,
                ORBIT_AMBIENT_K,
                ORBIT_AMBIENT_K,
                crate::medium::DIVE_HEAT_SCALE,
            );
            commands.entity(entity).insert((
                crate::medium::DivingCraft::new(spawn.craft.clone(), mp.center_of_mass, glide),
                thermal,
            ));
            commands.insert_resource(crate::medium::EntryInterface {
                surface_radius: spawn.surface_radius,
                altitude: interface,
            });
            bevy_log::info!(
                "scenario `{}` spawned: {} on rails at {altitude} m, entry interface {interface} m",
                spawn.id,
                spawn.name
            );
            messages.p1().write(Command::SetWarp(RAILS_COAST_WARP));
        }
        StartPlacement::Afloat => {
            // The harbor regime (WI 739): assemble at real material mass and
            // spawn the floating chain the shared DescentPlugin steps —
            // synthesized drive/rudder/ballast until the device palette
            // (WI 715), interior-flood physics alongside.
            let Some((body, dc)) =
                crate::afloat::assemble_float(&spawn.craft, spawn.mu, spawn.surface_radius)
            else {
                bevy_log::warn!("scenario spawn: blueprint has no mass — nothing spawned");
                return;
            };
            let marine = crate::afloat::synth_marine(&spawn.craft);
            let ballast = crate::afloat::synth_ballast(&spawn.craft, spawn.surface_radius);
            let rudder = crate::afloat::synth_rudder(&spawn.craft);
            let flood = crate::afloat::FloodComps::for_craft(&spawn.craft);
            let mut vessel = commands.spawn((
                body,
                dc,
                marine,
                rudder,
                flood,
                crate::afloat::ScenarioVessel,
            ));
            if let Some(b) = ballast {
                vessel.insert(b);
            }
            bevy_log::info!(
                "scenario `{}` spawned: {} afloat at the origin",
                spawn.id,
                spawn.name
            );
        }
    }
}

/// Advances every vessel's interior flooding (WI 739, the physics half of
/// the old harbor flood step): breached compartments take on water and the
/// hull's enclosed-buoyancy set shrinks, so the shared descent step feels
/// the lost buoyancy. Render (occluders, rising water) stays scene-side.
fn step_vessel_flooding(
    time: Res<Time>,
    mut vessels: Query<(
        &crate::active::ActiveBody,
        &mut crate::medium::DivingCraft,
        &mut crate::afloat::FloodComps,
    )>,
) {
    let dt = time.delta_secs_f64();
    for (body, mut dc, mut comps) in &mut vessels {
        crate::afloat::step_flooding(&mut comps, body, &mut dc, dt);
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
    // A terminal session (Recovery) freezes the flight — the migrated play/
    // autopilot behaviour (WI 739).
    if flight.session.is_terminal() {
        return;
    }
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
    while flight.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS && !flight.session.is_terminal() {
        let v0 = flight.body.velocity;
        let r0 = flight.body.position.length();
        let up0 = if r0 > 0.0 {
            flight.body.position / r0
        } else {
            DVec3::Y
        };
        let mu = flight.params.mu;
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

        // Felt acceleration (what an accelerometer reads) and the session
        // phase, from simulation state (WI 739 — the play/autopilot wiring).
        let gravity_accel = -mu / (r0 * r0) * up0;
        let felt = (flight.body.velocity - v0) / SUBSTEP_DT - gravity_accel;
        flight.g_force = felt.length() / G0;
        let altitude = flight.pad.altitude(&flight.body);
        let vertical_speed = flight.body.velocity.dot(up0);
        let speed = flight.body.velocity.length();
        let released = flight.pad.released;
        flight
            .session
            .update(released, altitude, vertical_speed, speed);

        flight.accumulator -= SUBSTEP_DT;
        flight.elapsed += SUBSTEP_DT;
        n += 1;
    }
}

/// The mission evaluator (WI 551): at a bounded flight-time cadence, builds
/// the bus-shaped snapshot (the same construction the publisher uses) and
/// polls every Active mission's objective tree. Node satisfaction latches
/// (monotone — warp-coarse polling is defined semantics, not a race). On
/// completion the mission's effects issue **once**: `Command` effects as
/// ordinary envelope messages (validated by the executor/tier gates exactly
/// like player commands — fire-and-forget, a rejection does not un-complete),
/// `Lore` effects onto the scenario state (telemetry + HUD). Completion
/// activates any Pending mission whose `AfterMission` offer names it (the
/// linear campaign primitive). No new warp-drop triggers; paused flights
/// accrue no flight time, so no polls.
fn evaluate_missions(
    clock: Res<SimClock>,
    flight: Option<ResMut<ScenarioFlight>>,
    mut commands: MessageWriter<Command>,
) {
    let Some(mut flight) = flight else { return };
    if flight.missions.is_empty() || flight.elapsed - flight.last_poll < MISSION_POLL_SECONDS {
        return;
    }
    flight.last_poll = flight.elapsed;

    // The snapshot the leaves query — the wire shape, not sim internals.
    let snap = crate::telemetry::Telemetry::capture(&clock, None, flight.params.mu, None)
        .with_scenario(flight.telemetry());

    let mut completed: Vec<String> = Vec::new();
    let mut beats: Vec<String> = Vec::new();
    for run in flight.missions.iter_mut() {
        if run.state != MissionState::Active {
            continue;
        }
        if run.nodes.poll(&run.def.objective, &snap) {
            run.state = MissionState::Completed;
            bevy_log::info!("mission `{}` complete: {}", run.def.id, run.def.name);
            for effect in &run.def.effects {
                match effect {
                    Effect::Lore(text) => beats.push(text.clone()),
                    Effect::Command(cmd) => {
                        commands.write(*cmd);
                    }
                }
            }
            completed.push(run.def.id.clone());
        }
    }
    if let Some(beat) = beats.pop() {
        flight.lore = Some(beat);
    }
    // Offer successors of anything that just completed.
    for done in &completed {
        for run in flight.missions.iter_mut() {
            if run.state == MissionState::Pending
                && matches!(&run.def.offer, Offer::AfterMission(after) if after == done)
            {
                run.state = MissionState::Active;
                bevy_log::info!("mission `{}` offered: {}", run.def.id, run.def.name);
            }
        }
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
                    step_vessel_flooding,
                    evaluate_missions,
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
    /// lifts off at full throttle. (The 50-unit battery was sized small when
    /// charge still counted toward wet mass; since WI 810 charge is massless
    /// and the size is a content choice, not a physics mitigation.)
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
            missions: Vec::new(),
            content: ContentIdentity::default(),
            resume: None,
            bodies: Vec::new(),
        }
    }

    /// A payload carrying the given missions.
    fn spawn_with_missions(missions: Vec<Mission>) -> ScenarioSpawn {
        ScenarioSpawn {
            missions,
            ..spawn_payload()
        }
    }

    fn hop(id: &str, offer: Offer, altitude: f64) -> Mission {
        Mission {
            format: 2,
            id: id.into(),
            name: format!("hop {id}"),
            offer,
            objective: crate::mission::Condition::AltitudeAbove(altitude),
            effects: vec![Effect::Lore(format!("{id} done"))],
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
        let path = crate::library::save_blueprint(&dir, "First Flight", &blueprint()).unwrap();
        assert!(path.ends_with("first-flight.json"));
    }

    /// The flight-family blueprint (WI 739): the play/launch/autopilot
    /// scenes' 1 m-cell composite stack — crewed control point, Tier-2
    /// tuning computer, battery — plus Engine/Tank devices so the director's
    /// catalog-bound assembly supplies the propulsion (the scenes hardcoded
    /// it). With the core pack's medium engine/tank records the all-up mass
    /// gives the scenes' TWR ≈ 1.6.
    fn sounding_rocket_blueprint() -> VoxelCraft {
        let mut craft = VoxelCraft::new(1.0);
        for y in 0..5 {
            craft.voxels.push(Voxel {
                cell: IVec3::new(0, y, 0),
                material: Material::COMPOSITE,
            });
        }
        craft
            .devices
            .push(Device::control_point(IVec3::new(0, 0, 0), 120.0, true));
        craft.devices.push(Device::computer(
            IVec3::new(0, 2, 0),
            40.0,
            ControlComputer::tuning_computer(0.4),
        ));
        craft.devices.push(Device::battery(
            IVec3::new(0, 3, 0),
            60.0,
            BatterySpec::full(120.0),
        ));
        craft
            .devices
            .push(Device::structural(IVec3::ZERO, 150.0, DeviceKind::Engine));
        craft.devices.push(Device::structural(
            IVec3::new(0, 1, 0),
            50.0,
            DeviceKind::Tank,
        ));
        craft
    }

    /// The dive-capsule blueprint (WI 739): the dive scene's slender re-entry
    /// body along +Z — a 3×3×4 composite hull with an ablative heat-shield
    /// nose tip at the windward front (positive static margin → it
    /// weathervanes into the airflow). No devices — a passive capsule.
    fn dive_capsule_blueprint() -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        for z in 0..4 {
            for x in 0..3 {
                for y in 0..3 {
                    c.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
        }
        c.voxels.push(Voxel {
            cell: IVec3::new(1, 1, 4),
            material: Material::ABLATOR,
        });
        c
    }

    /// Regenerates the shipped dive-capsule blueprint (WI 739):
    /// `cargo test -p sounding_sim --lib write_dive_capsule_blueprint -- --ignored`
    #[test]
    #[ignore = "writes the shipped content/blueprints/dive-capsule.json artifact"]
    fn write_dive_capsule_blueprint() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/blueprints");
        let path = crate::library::save_blueprint(&dir, "Dive Capsule", &dive_capsule_blueprint())
            .unwrap();
        assert!(path.ends_with("dive-capsule.json"));
    }

    /// Regenerates the shipped flight-family blueprint (WI 739):
    /// `cargo test -p sounding_sim --lib write_sounding_rocket_blueprint -- --ignored`
    #[test]
    #[ignore = "writes the shipped content/blueprints/sounding-rocket.json artifact"]
    fn write_sounding_rocket_blueprint() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/blueprints");
        let path =
            crate::library::save_blueprint(&dir, "Sounding Rocket", &sounding_rocket_blueprint())
                .unwrap();
        assert!(path.ends_with("sounding-rocket.json"));
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
        // The shipped mission rides the payload and starts Active (WI 551).
        assert_eq!(spawn.missions.len(), 1);
        assert_eq!(spawn.missions[0].id, "first-hop");
        assert_eq!(flight.missions[0].state, MissionState::Active);
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

    /// Builds the standard headless app: executor + director + a spawned
    /// payload, paused, ready to be driven by `Command::Step`.
    fn mission_app(missions: Vec<Mission>) -> App {
        let mut app = App::new();
        app.add_plugins(bevy_time::TimePlugin);
        app.init_resource::<SimClock>();
        app.add_plugins(crate::command::FlightControlPlugin);
        app.add_plugins(DirectorPlugin);
        app.world_mut().resource_mut::<PendingSpawn>().0 = Some(spawn_with_missions(missions));
        app.world_mut().write_message(Command::SpawnScenario);
        app.update();
        app.world_mut().write_message(Command::SetPaused(true));
        app.update();
        app
    }

    #[test]
    fn first_hop_completes_end_to_end_with_lore() {
        let mut app = mission_app(vec![hop("hop", Offer::Immediate, 100.0)]);
        // At rest: Active, zero progress, no lore.
        step_sim(&mut app, 1.0);
        {
            let flight = app.world().resource::<ScenarioFlight>();
            assert_eq!(flight.missions[0].state, MissionState::Active);
            let block = flight.telemetry();
            assert_eq!(block.missions[0].progress, 0.0);
            assert_eq!(block.lore, None);
            assert!(!block.airborne);
        }
        // Full throttle: the climb passes 100 m; the mission completes and
        // the lore beat surfaces on the (bus-shaped) scenario block.
        app.world_mut().write_message(Command::SetThrottle(1.0));
        app.update();
        step_sim(&mut app, 4.0);
        step_sim(&mut app, 4.0);
        let flight = app.world().resource::<ScenarioFlight>();
        assert_eq!(flight.missions[0].state, MissionState::Completed);
        let block = flight.telemetry();
        assert_eq!(block.missions[0].progress, 1.0);
        assert_eq!(block.lore.as_deref(), Some("hop done"));
        assert!(block.altitude > 100.0);
    }

    #[test]
    fn completion_latches_and_after_mission_chains() {
        // Mission 2 offers after mission 1; both are one-leaf altitude hops.
        let mut app = mission_app(vec![
            hop("one", Offer::Immediate, 50.0),
            hop("two", Offer::AfterMission("one".into()), 120.0),
        ]);
        {
            let flight = app.world().resource::<ScenarioFlight>();
            assert_eq!(flight.missions[0].state, MissionState::Active);
            assert_eq!(flight.missions[1].state, MissionState::Pending, "gated");
        }
        app.world_mut().write_message(Command::SetThrottle(1.0));
        app.update();
        step_sim(&mut app, 4.0);
        step_sim(&mut app, 4.0);
        {
            let flight = app.world().resource::<ScenarioFlight>();
            assert_eq!(flight.missions[0].state, MissionState::Completed);
            assert_eq!(
                flight.missions[1].state,
                MissionState::Completed,
                "offered on one's completion, then completed by the same climb"
            );
        }
        // Latching: cut throttle, fall back — completed missions stay completed.
        app.world_mut().write_message(Command::SetThrottle(0.0));
        app.update();
        step_sim(&mut app, 4.0);
        let flight = app.world().resource::<ScenarioFlight>();
        assert_eq!(flight.missions[0].state, MissionState::Completed);
        assert_eq!(flight.missions[1].state, MissionState::Completed);
    }

    /// WI 739 Stage 1: the shipped launch scenario end to end — the craft
    /// rests on the pad with no player input, the liftoff mission's
    /// `ElapsedAbove` objective completes at ~2 s of flight time, its effect
    /// throttles the engine through the envelope, and the rocket lifts off:
    /// the session tracks Launch → Flight and the telemetry block carries
    /// elapsed time, session, and G-force.
    #[test]
    fn shipped_launch_scenario_lifts_off_by_mission_effect() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let roots = crate::scenario::ScenarioRoots {
            content: root.join("content"),
            saves: root.join("saves"),
        };
        let s = crate::scenario::load_scenario(&root.join("content/scenarios/launch.ron"), &roots)
            .unwrap();
        let spawn = ScenarioSpawn::from_scenario(&s);
        assert_eq!(spawn.engine, Some((3000.0, 70.0)), "physical, unscaled");
        assert_eq!(spawn.tank_capacity, Some(5000.0));

        let mut app = mission_app(spawn.missions.clone());
        // Re-stage with the real payload (mission_app staged the test one).
        app.world_mut().resource_mut::<PendingSpawn>().0 = Some(spawn);
        app.world_mut().write_message(Command::SpawnScenario);
        app.update();
        {
            let flight = app.world().resource::<ScenarioFlight>();
            assert!(!flight.pad.released);
            assert_eq!(flight.session.phase, crate::session::Phase::Launch);
        }
        // No input at all: one second in, still held (the mission waits).
        step_sim(&mut app, 1.0);
        assert!(!app.world().resource::<ScenarioFlight>().pad.released);
        // Past the 2 s hold the effect throttles up; TWR ≈ 1.6 releases the
        // pad and the rocket climbs.
        step_sim(&mut app, 4.0);
        let flight = app.world().resource::<ScenarioFlight>();
        assert_eq!(flight.missions[0].state, MissionState::Completed);
        assert!(flight.pad.released, "mission effect throttled it up");
        assert!(flight.body.velocity.y > 0.0, "ascending");
        assert_eq!(flight.session.phase, crate::session::Phase::Flight);
        let block = flight.telemetry();
        assert!(block.elapsed > 4.0);
        assert_eq!(
            block.session.map(|s| s.phase),
            Some(crate::session::Phase::Flight)
        );
        assert!(block.g_force > 0.0);
    }

    /// WI 739 Stage 2: the shipped dive scenario loads (empty pack list, no
    /// bindings — a passive capsule) and its orbit-entry payload spawn
    /// configures the on-rails craft entity; the existing sim plugins then
    /// run the chain: rails coast under the spawn-issued warp, auto-wake at
    /// the entry interface (warp filter back to 1×), finite active descent.
    #[test]
    fn shipped_dive_scenario_spawns_rails_and_wakes_at_the_interface() {
        use crate::handoff::{GearState, HandoffPlugin};
        use crate::medium::{DescentPlugin, DiveTriggerPlugin, EntryInterface};
        use crate::sim::{CentralBody, Craft, OrbitPlugin};
        use bevy_time::Time;
        use std::time::Duration;

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let roots = crate::scenario::ScenarioRoots {
            content: root.join("content"),
            saves: root.join("saves"),
        };
        let s = crate::scenario::load_scenario(&root.join("content/scenarios/dive.ron"), &roots)
            .unwrap();
        assert!(s.bindings.is_empty(), "a passive capsule needs no bindings");
        let spawn = ScenarioSpawn::from_scenario(&s);
        assert_eq!(
            spawn.placement,
            StartPlacement::Orbit {
                altitude: 120_000.0,
                speed: 7_000.0,
                interface: 100_000.0,
            }
        );

        let body = CentralBody::EARTHLIKE;
        // Any parking orbit — the spawn arm replaces it with the entry orbit.
        let parking = crate::orbit::Orbit::from_state(
            body.mu,
            glam::DVec2::new(body.radius + 500_000.0, 0.0),
            glam::DVec2::new(0.0, (body.mu / (body.radius + 500_000.0)).sqrt()),
            0.0,
        )
        .unwrap();
        let mut app = App::new();
        app.insert_resource(Time::<()>::default()); // drive time manually (deterministic)
        app.insert_resource(crate::active::Gravity { mu: body.mu });
        app.add_plugins(OrbitPlugin {
            central_body: body,
            initial_orbit: parking,
        });
        app.add_plugins(crate::command::FlightControlPlugin);
        app.add_plugins(HandoffPlugin);
        app.add_plugins(DiveTriggerPlugin {
            // Placeholder config — the spawn arm overwrites it from the payload.
            interface: EntryInterface {
                surface_radius: body.radius,
                altitude: 1.0,
            },
        });
        app.add_plugins(DescentPlugin {
            substep_dt: 0.004,
            max_substeps: 250,
        });
        app.add_plugins(DirectorPlugin);

        app.world_mut().resource_mut::<PendingSpawn>().0 = Some(spawn);
        app.world_mut().write_message(Command::SpawnScenario);
        app.update();
        // The spawn arm configured the interface from the document and issued
        // the rails-coast warp through the envelope.
        assert_eq!(app.world().resource::<EntryInterface>().altitude, 100_000.0);
        {
            let mut q = app.world_mut().query::<&Craft>();
            let orbit = q.single(app.world()).unwrap().orbit;
            assert!(
                orbit.periapsis_radius() < body.radius,
                "an entry trajectory: periapsis inside the atmosphere"
            );
        }
        app.update();
        assert_eq!(app.world().resource::<SimClock>().warp, RAILS_COAST_WARP);

        // Coast to the interface and wake; then descend actively and finitely.
        let mut transitioned = false;
        for _ in 0..2_000 {
            app.world_mut()
                .resource_mut::<Time<()>>()
                .advance_by(Duration::from_secs_f64(0.5));
            app.update();
            let mut q = app
                .world_mut()
                .query::<Option<&crate::active::ActiveBody>>();
            if let Some(active) = q.single(app.world()).unwrap() {
                transitioned = true;
                assert!(
                    active.position.is_finite() && active.velocity.is_finite(),
                    "active descent state must stay finite"
                );
                let altitude = active.position.length() - body.radius;
                assert!(altitude < 100_500.0, "woke at/below the interface");
                if altitude < 80_000.0 {
                    break; // well into active descent — the chain runs
                }
            }
        }
        assert!(
            transitioned,
            "rails must hand off to active at the interface"
        );
        // The entry trigger's warp filter dropped the coast warp.
        assert_eq!(app.world().resource::<SimClock>().warp, 1.0);
        // The gear state carries the blueprint's real mass.
        let mp = dive_capsule_blueprint().mass_properties().unwrap();
        let mut q = app.world_mut().query::<&GearState>();
        assert_eq!(q.single(app.world()).unwrap().mass, mp.mass);
    }

    /// WI 739 Stage 3: the shipped harbor scenario loads and its afloat
    /// payload spawns the floating chain — the seed panel pontoon settles on
    /// the water (finite, near the waterline, righting from its starting
    /// list) on the shared descent step, with the synthesized drive/rudder/
    /// ballast and the flood physics aboard.
    #[test]
    fn shipped_harbor_scenario_spawns_afloat_and_settles() {
        use crate::afloat::{FloodComps, ScenarioVessel};
        use crate::medium::{DescentPlugin, DivingCraft};
        use bevy_time::Time;
        use std::time::Duration;

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let roots = crate::scenario::ScenarioRoots {
            content: root.join("content"),
            saves: root.join("saves"),
        };
        let s = crate::scenario::load_scenario(&root.join("content/scenarios/harbor.ron"), &roots)
            .unwrap();
        let spawn = ScenarioSpawn::from_scenario(&s);
        assert_eq!(spawn.placement, StartPlacement::Afloat);

        let mut app = App::new();
        app.insert_resource(Time::<()>::default()); // drive time manually
        app.init_resource::<SimClock>();
        app.add_plugins(crate::command::FlightControlPlugin);
        app.add_plugins(DescentPlugin {
            substep_dt: 0.002,
            max_substeps: 64,
        });
        app.add_plugins(DirectorPlugin);

        app.world_mut().resource_mut::<PendingSpawn>().0 = Some(spawn);
        app.world_mut().write_message(Command::SpawnScenario);
        app.update();
        {
            let mut q = app
                .world_mut()
                .query_filtered::<(&ActiveBody, &FloodComps), With<ScenarioVessel>>();
            let (body, flood) = q.single(app.world()).unwrap();
            assert!(body.orientation.to_axis_angle().1 > 0.1, "starting list");
            assert!(
                !flood.comps.is_empty(),
                "sealed pontoon: flood physics aboard"
            );
        }

        // Let it settle for ~12 s of sim time.
        for _ in 0..240 {
            app.world_mut()
                .resource_mut::<Time<()>>()
                .advance_by(Duration::from_secs_f64(0.05));
            app.update();
        }
        let surface = crate::sim::CentralBody::EARTHLIKE.radius;
        let mut q = app
            .world_mut()
            .query_filtered::<(&ActiveBody, &DivingCraft), With<ScenarioVessel>>();
        let (body, _dc) = q.single(app.world()).unwrap();
        assert!(
            body.position.is_finite() && body.velocity.is_finite(),
            "afloat state stays finite"
        );
        let altitude = body.position.length() - surface;
        assert!(
            altitude.abs() < 3.0,
            "the panel pontoon rides near the waterline: {altitude} m"
        );
        assert!(
            body.velocity.length() < 1.0,
            "settled: {} m/s",
            body.velocity.length()
        );
        // Righted from the 0.2 rad starting list.
        let up_alignment =
            (body.orientation * glam::DVec3::Y).dot(body.position.normalize_or(glam::DVec3::Y));
        assert!(up_alignment > 0.98, "self-righted: {up_alignment}");
    }

    #[test]
    fn command_effects_act_through_the_executor() {
        // A mission whose effect is an envelope command: on completion the
        // executor applies it exactly as a player command (warp changes).
        let mut m = hop("warpme", Offer::Immediate, 50.0);
        m.effects = vec![Effect::Command(Command::SetWarp(4.0))];
        let mut app = mission_app(vec![m]);
        assert_eq!(app.world().resource::<SimClock>().warp, 1.0);
        app.world_mut().write_message(Command::SetThrottle(1.0));
        app.update();
        step_sim(&mut app, 4.0);
        step_sim(&mut app, 4.0);
        assert_eq!(
            app.world().resource::<ScenarioFlight>().missions[0].state,
            MissionState::Completed
        );
        assert_eq!(app.world().resource::<SimClock>().warp, 4.0);
    }

    /// WI 891: `capture` writes the flight's per-body records; an empty list
    /// stays absent from the JSON (pre-891 world saves are byte-unchanged by
    /// this WI); records round-trip through the envelope; and a resume
    /// carries a loaded snapshot pin through reconciliation verbatim.
    #[test]
    fn capture_carries_body_records_and_resume_preserves_pins() {
        use crate::world_save::SavedBodyRecord;
        // Empty records: the member is absent (the additive rule's byte test).
        let spawn = spawn_payload();
        let flight = instantiate(&spawn).unwrap();
        let json = crate::persist::SavedDocument::new(crate::persist::Payload::WorldSave(
            crate::world_save::capture(&flight, vec![]),
        ))
        .to_json()
        .unwrap();
        assert!(!json.contains("\"bodies\""), "absent member stays absent");

        // With records: capture writes them; the envelope round-trips them.
        let root = crate::body_asset::BodyAsset::earthlike();
        let mut spawn = spawn_payload();
        spawn.bodies = crate::world_save::body_records(
            std::slice::from_ref(&root),
            &std::iter::once(root.id.clone()).collect(),
        );
        let flight = instantiate(&spawn).unwrap();
        let payload = crate::world_save::capture(&flight, vec![]);
        let json = crate::persist::SavedDocument::new(crate::persist::Payload::WorldSave(payload))
            .to_json()
            .unwrap();
        let saved = match crate::persist::SavedDocument::from_json(&json)
            .unwrap()
            .payload
        {
            crate::persist::Payload::WorldSave(w) => w.scenario.unwrap(),
            _ => panic!("world scope"),
        };
        assert_eq!(saved.bodies, spawn.bodies);

        // Resume: a loaded pin (older output version) survives reconciliation
        // verbatim — a re-save must not re-stamp it with this build's version.
        let mut old = saved.clone();
        match &mut old.bodies[0] {
            SavedBodyRecord::Snapshot { output_version, .. } => *output_version = 0,
            other => panic!("root record should be a snapshot, got {other:?}"),
        }
        let resumed = instantiate(&ScenarioSpawn {
            resume: Some(old.clone()),
            ..spawn.clone()
        })
        .unwrap();
        assert_eq!(resumed.bodies, old.bodies, "pin preserved verbatim");
    }

    /// WI 892 (scenario A-3b): a v2 world save carrying WI 891 per-body
    /// records migrates coherently — the snapshot record's embedded body gets
    /// the flat→stack conversion and its digest is restored over the migrated
    /// body (integrity by construction) while its recorded output_version is
    /// left untouched (an old snapshot stays an old-generator pin: the WI 891
    /// drift line fires, the snapshot wins); a digest-tier record is carried
    /// unmodified (its stale version already routes to "expected reroll").
    #[test]
    fn v2_world_save_body_records_migrate_coherently() {
        use crate::world_save::{apply_body_records, SavedBodyRecord};
        let root = crate::body_asset::BodyAsset::earthlike();
        let visited = crate::bodygen::generate(42, crate::bodygen::Archetype::RockyPlanet);
        let mut spawn = spawn_payload();
        spawn.bodies = crate::world_save::body_records(
            &[root.clone(), visited.clone()],
            &std::iter::once(root.id.clone()).collect(),
        );
        let flight = instantiate(&spawn).unwrap();
        let payload = crate::world_save::capture(&flight, vec![]);
        let doc = crate::persist::SavedDocument::new(crate::persist::Payload::WorldSave(payload));

        // Regress the document to v2 shape: version tag, flat surfaces on the
        // snapshot record's embedded body, a stale (old-layout) digest, and
        // output_version 1 (the v2 era's value).
        let mut v2 = serde_json::to_value(&doc).unwrap();
        v2["format_version"] = 2.into();
        let rec = &mut v2["payload"]["scenario"]["bodies"][0];
        assert_eq!(rec["tier"], "snapshot");
        rec["output_version"] = 1.into();
        rec["digest"] = "0123456789abcdef".into(); // stale old-layout digest
        rec["body"]["surface"] = serde_json::json!({
            "seed": 0, "terrain": null, "crater": null,
            "material": { "temperature": -5.0 }
        });
        v2["payload"]["scenario"]["bodies"][1]["output_version"] = 1.into();

        let migrated = crate::persist::SavedDocument::from_json(&v2.to_string())
            .expect("v2 world save migrates");
        assert_eq!(migrated.format_version, crate::persist::FORMAT_VERSION);
        let crate::persist::Payload::WorldSave(w) = &migrated.payload else {
            panic!("world scope");
        };
        let records = &w.scenario.as_ref().unwrap().bodies;

        // Snapshot record: surface converted, digest restored, version kept.
        let SavedBodyRecord::Snapshot {
            output_version,
            digest,
            body,
            ..
        } = &records[0]
        else {
            panic!("snapshot tier");
        };
        assert_eq!(*output_version, 1, "recorded output_version untouched");
        assert_eq!(
            *digest,
            crate::body_digest::digest_hex(body),
            "digest restored over the migrated body"
        );
        assert_eq!(
            body.surface.layers.len(),
            1,
            "flat material area became one layer"
        );
        let pinned_body = (**body).clone();

        // And the load path accepts it: integrity passes, the stale versions
        // are the designed drift lines, the snapshot substitutes.
        let mut assets = vec![root.clone(), visited.clone()];
        let drift = apply_body_records(records, &mut assets).expect("no integrity failure");
        assert!(
            drift.iter().any(|l| l.contains("pins output version")),
            "{drift:?}"
        );
        assert!(
            drift.iter().any(|l| l.contains("expected reroll")),
            "digest-tier stale version reads as the designed reroll: {drift:?}"
        );
        assert_eq!(assets[0], pinned_body, "snapshot substituted");
    }

    /// WI 553: capture → JSON → restore is a live-state round-trip — the
    /// resumed flight matches the saved one (dynamic state, reservoirs,
    /// session, elapsed, mission latches) and keeps flying.
    #[test]
    fn world_save_resume_restores_an_equivalent_flight() {
        // Fly: two missions (one completes mid-burn, one chained), full
        // throttle, a few seconds of ascent.
        let missions = vec![
            hop("first", Offer::Immediate, 10.0),
            hop("second", Offer::AfterMission("first".into()), 1.0e7),
        ];
        let mut app = mission_app(missions.clone());
        app.world_mut().write_message(Command::SetThrottle(1.0));
        app.update();
        step_sim(&mut app, 4.0);
        step_sim(&mut app, 4.0);

        // Capture through the persist envelope (the real save path).
        let json = {
            let flight = app.world().resource::<ScenarioFlight>();
            assert!(flight.pad.released, "airborne before capture");
            assert_eq!(flight.missions[0].state, MissionState::Completed);
            assert_eq!(flight.missions[1].state, MissionState::Active);
            let payload = crate::world_save::capture(flight, vec![]);
            crate::persist::SavedDocument::new(crate::persist::Payload::WorldSave(payload))
                .to_json()
                .unwrap()
        };
        let world = match crate::persist::SavedDocument::from_json(&json)
            .unwrap()
            .payload
        {
            crate::persist::Payload::WorldSave(w) => w,
            _ => panic!("world scope"),
        };
        let saved = world.scenario.expect("scenario state present");

        // Resume in a fresh app: same defs, saved state riding the spawn.
        let mut app2 = App::new();
        app2.add_plugins(bevy_time::TimePlugin);
        app2.init_resource::<SimClock>();
        app2.add_plugins(crate::command::FlightControlPlugin);
        app2.add_plugins(DirectorPlugin);
        app2.world_mut().resource_mut::<PendingSpawn>().0 = Some(ScenarioSpawn {
            resume: Some(saved.clone()),
            ..spawn_with_missions(missions)
        });
        app2.world_mut().write_message(Command::SpawnScenario);
        app2.update();
        // Freeze the clock (the harness convention) so continuation is
        // driven deterministically by Command::Step.
        app2.world_mut().write_message(Command::SetPaused(true));
        app2.update();

        {
            let a = app2.world().resource::<ScenarioFlight>();
            let tol = 1e-9;
            assert!((a.body.position - saved.body.position).length() < tol);
            assert!((a.body.velocity - saved.body.velocity).length() < tol);
            assert!((a.body.angular_momentum - saved.body.angular_momentum).length() < tol);
            assert!(a.pad.released, "pad release restored");
            assert_eq!(a.session, saved.session, "session restored verbatim");
            assert!((a.elapsed - saved.elapsed).abs() < tol);
            assert_eq!(a.last_poll, a.elapsed, "poll clock re-seeded, no burst");
            assert_eq!(a.missions[0].state, MissionState::Completed);
            assert_eq!(a.missions[1].state, MissionState::Active, "chain restored");
            assert!(
                (a.craft.propulsion.graph.reservoirs[0].amount
                    - saved.craft.propulsion.graph.reservoirs[0].amount)
                    .abs()
                    < tol,
                "reservoir levels restored"
            );
            // The telemetry block (the wire equivalence surface) agrees.
            let t = a.telemetry();
            assert!((t.elapsed - saved.elapsed).abs() < tol);
            assert_eq!(t.missions.len(), 2);
        }

        // The resumed flight keeps flying: still under the throttle the
        // restored craft carries, it climbs on.
        let y0 = app2.world().resource::<ScenarioFlight>().body.position.y;
        step_sim(&mut app2, 2.0);
        let flight = app2.world().resource::<ScenarioFlight>();
        assert!(flight.body.position.is_finite());
        assert!(flight.body.position.y > y0, "resumed flight continues");
    }

    /// WI 553: the drift-tolerant mission reconcile — unknown saved ids
    /// drop (reported), shape mismatches reset (reported), and AfterMission
    /// activation recomputes from restored completion.
    #[test]
    fn reconcile_missions_handles_drift() {
        use crate::world_save::MissionSave;
        let defs = vec![
            hop("a", Offer::Immediate, 10.0),
            hop("b", Offer::AfterMission("a".into()), 20.0),
        ];
        // `a` completed with a matching (leaf) tree; `ghost` no longer
        // exists; `b` saved with a wrong-shape tree (composite of 2). The
        // latched leaf arrives through serde — exactly how a save delivers it.
        let done: NodeState =
            serde_json::from_str(r#"{"latched":true,"children":[]}"#).expect("latch tree decodes");
        let wrong_shape = NodeState::for_condition(&crate::mission::Condition::All(vec![
            crate::mission::Condition::Airborne,
            crate::mission::Condition::Airborne,
        ]));
        let saved = vec![
            MissionSave {
                id: "a".into(),
                state: MissionState::Completed,
                nodes: done,
            },
            MissionSave {
                id: "ghost".into(),
                state: MissionState::Active,
                nodes: NodeState::for_condition(&crate::mission::Condition::Airborne),
            },
            MissionSave {
                id: "b".into(),
                state: MissionState::Active,
                nodes: wrong_shape,
            },
        ];
        let (runs, report) = reconcile_missions(&defs, &saved);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].state, MissionState::Completed);
        assert!(runs[0].nodes.matches(&defs[0].objective));
        // b keeps its lifecycle state (Active) but its mismatched latch tree
        // was reset to the re-resolved objective's fresh shape.
        assert_eq!(runs[1].state, MissionState::Active);
        assert!(runs[1].nodes.matches(&defs[1].objective), "fresh tree");
        assert_eq!(runs[1].nodes.progress(&defs[1].objective), 0.0, "reset");
        let text = report.join("\n");
        assert!(text.contains("ghost"), "{text}");
        assert!(text.contains("`b` objective changed"), "{text}");
    }
}

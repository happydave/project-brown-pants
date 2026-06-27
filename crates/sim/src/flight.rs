//! Unified active-flight step (WI 534).
//!
//! One place that composes **every** active-flight force and torque — gravity,
//! atmospheric/hydro drag, buoyancy, engine thrust (WI 531), attitude control
//! (WI 533), and optional aero lift + transonic wave drag (WI 526) — into a single
//! net wrench applied through the launch-pad gate (WI 532) and `integrate_wrench`.
//! Before this, the dive and launch scenes each hand-assembled these forces; this
//! is the consolidation those reflections kept asking for, and the substrate the
//! game session (`session.rs`) and the `-- play` scene run on.
//!
//! Headless. (The dive/launch scenes keep their own assembly for now to preserve
//! their confirmed visuals; migrating them onto this step is a follow-up.)

use crate::active::ActiveBody;
use crate::aero;
use crate::attitude::AttitudePilot;
use crate::autopilot::Autopilot;
use crate::collision::{craft_bounds, craft_collision_shape, CollisionShape};
use crate::contact::{ground_contact_wrench, ContactParams};
use crate::control::{ControlSystem, ControlTier};
use crate::fluid::{FluidMedium, FluidSample};
use crate::launch::LaunchPad;
use crate::medium::{buoyancy_force, drag_force, submerged_volume, GlideParams};
use crate::propulsion::Propulsion;
use crate::voxel::VoxelCraft;
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// A flyable craft: geometry + propulsion + attitude, sharing one resource graph
/// (the propulsion graph) for main-engine and RCS propellant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlightCraft {
    /// Voxel geometry (drag reference, submerged volume, dry mass/inertia).
    pub voxels: VoxelCraft,
    /// Dry mass, kg (cached from `voxels`).
    pub dry_mass: f64,
    /// Dry centre of mass, body frame, m (cached from `voxels`).
    pub dry_com: DVec3,
    /// Engines + propellant tanks (the shared resource graph).
    pub propulsion: Propulsion,
    /// Attitude actuators + SAS (RCS draws from `propulsion.graph`).
    pub attitude: AttitudePilot,
    /// Control devices + the autonomy-tier gate (WI 562). The battery, if any, is a
    /// reservoir in `propulsion.graph`.
    pub control: ControlSystem,
    /// Engaged Tier-1 canned autopilot, if any (WI 565). Requires a powered `Canned`
    /// computer to engage and to act; drives SAS target + throttle each step.
    #[serde(default)]
    pub autopilot: Option<Autopilot>,
}

impl FlightCraft {
    /// The craft's current **effective** control tier — what is operating given power,
    /// resolved from its control devices and the power in its (shared) resource graph
    /// (WI 562). This is the tier the executor gates on.
    pub fn resolve_control(&self) -> ControlTier {
        self.control.resolve(&self.propulsion.graph)
    }

    /// The craft's **available** (installed) control tier — charge-independent (WI 570).
    /// A HUD shows this as the craft's capability; it does not change when the battery
    /// drains (only [`Self::resolve_control`] does).
    pub fn available_control(&self) -> ControlTier {
        self.control.available_tier()
    }

    /// Whether powered assistance is currently offline because of low power (WI 570):
    /// the effective tier has fallen below the available tier due to the battery
    /// reaching its low-power reserve.
    pub fn assist_offline(&self) -> bool {
        self.control.assist_offline(&self.propulsion.graph)
    }

    /// Apply a manual command, **gated by the resolved control tier** (WI 562). An
    /// uncontrolled craft rejects every manual command (returns `false`, no state
    /// change); a controllable craft routes throttle/gimbal to propulsion and
    /// attitude/SAS to the attitude pilot. `current` is the craft orientation,
    /// captured as the SAS hold target. Returns whether the command was applied.
    ///
    /// Note: the *gate* governs whether a command is accepted. Engaging an active SAS
    /// mode additionally requires `Stabilized` (a powered command core, WI 564) — at
    /// Direct it is refused, not silently accepted-then-suppressed; `SetSas(Off)` is
    /// always allowed. Whether SAS produces torque is still enforced in [`flight_step`].
    pub fn apply_command(&mut self, cmd: &crate::command::Command, current: DQuat) -> bool {
        use crate::command::{Command, SasMode};
        let tier = self.resolve_control();
        if !tier.allows_manual() {
            return false;
        }
        match cmd {
            Command::SetThrottle(_) | Command::SetGimbal(_) => self.propulsion.apply_command(cmd),
            // Engaging an active SAS mode needs a powered command core (WI 564).
            Command::SetSas(mode) if *mode != SasMode::Off && !tier.allows_stabilization() => false,
            Command::SetAttitude(_) | Command::SetSas(_) | Command::SetSasRecapture(_) => {
                self.attitude.apply_command(cmd, current)
            }
            // Live controller tuning needs a powered Tier-2 computer (WI 566).
            Command::SetSasGains(..) if !tier.allows_tuning() => false,
            Command::SetSasGains(..) => self.attitude.apply_command(cmd, current),
            // Engaging a canned autopilot needs a powered Tier-1 computer (WI 565);
            // disengaging (None) is always allowed.
            Command::SetAutopilot(ap) => {
                if ap.is_some() && !tier.allows_canned() {
                    false
                } else {
                    self.autopilot = *ap;
                    true
                }
            }
            // Select a control-tier cap (WI 571). Always permitted on a controllable
            // craft (the `allows_manual` gate above already rejected uncontrolled): the
            // cap can only lower the operating tier, never exceed capability.
            Command::SetControlTier(sel) => {
                self.control.selected = *sel;
                true
            }
            _ => false,
        }
    }
}

/// A flat-ground collision plane for the active step (WI 592): an upward plane (`normal`,
/// `offset`) the craft collides with, plus the penalty [`ContactParams`]. Copy (no `Vec`),
/// so [`FlightParams`] stays `Copy`; the craft's own collision shape is derived from its
/// voxels each step.
#[derive(Clone, Copy, Debug)]
pub struct GroundContact {
    pub normal: DVec3,
    pub offset: f64,
    pub contact: ContactParams,
}

/// The fixed environment of an active flight. (Not serialized — reconstructed from
/// the body/medium constants; only the craft state, [`FlightCraft`], is persisted.)
#[derive(Clone, Copy, Debug)]
pub struct FlightParams {
    /// Central-body gravitational parameter, m³/s².
    pub mu: f64,
    /// Surface (sea-level) radius, m.
    pub surface_radius: f64,
    /// The unified fluid-medium field (atmosphere + ocean).
    pub medium: FluidMedium,
    /// Drag reference area, m².
    pub drag_area: f64,
    /// Drag coefficient.
    pub drag_coefficient: f64,
    /// Optional aero lift / wave drag (a winged craft); `None` for a ballistic body.
    pub lift: Option<GlideParams>,
    /// Optional ground-collision plane (WI 592); `None` ⇒ no collision (existing behaviour).
    pub ground: Option<GroundContact>,
}

/// Advance one active-flight sub-step: compose all forces/torques into one wrench
/// and apply it through the launch-pad gate (which holds the craft at rest until
/// thrust > weight, then integrates). Returns the medium sample at the craft.
pub fn flight_step(
    body: &mut ActiveBody,
    craft: &mut FlightCraft,
    params: &FlightParams,
    pad: &mut LaunchPad,
    dt: f64,
) -> FluidSample {
    // Resolve the control tier before borrowing the fields mutably: it gates how
    // much of the attitude demand is applied (WI 562).
    let tier = craft.resolve_control();

    // Canned autopilot (WI 565): if engaged and the tier permits, drive the SAS
    // target (and throttle, for an ascent autopilot) the same way a player would —
    // command arbitration (563) then lets manual input override it per axis.
    if tier.allows_canned() {
        if let Some(out) = craft.autopilot.map(|ap| {
            ap.evaluate(
                body.position,
                body.velocity,
                params.surface_radius,
                params.mu,
            )
        }) {
            if let Some(dir) = out.attitude_target {
                craft
                    .attitude
                    .sas
                    .set_mode(crate::command::SasMode::Point(dir), body.orientation);
            }
            if let Some(th) = out.throttle {
                craft
                    .propulsion
                    .apply_command(&crate::command::Command::SetThrottle(th));
            }
        }
    }

    let FlightCraft {
        voxels,
        dry_mass,
        dry_com,
        propulsion,
        attitude,
        control: _,
        autopilot: _,
    } = craft;

    // Wet mass + CoM (propellant folds into mass/CoM in real time, WI 531).
    let wet = propulsion.wet_mass(*dry_mass, *dry_com);
    body.mass = wet.mass;
    let com = wet.center_of_mass;

    let r = body.position.length();
    let up = if r > 0.0 { body.position / r } else { DVec3::Y };
    let altitude = r - params.surface_radius;
    let sample = params.medium.sample_altitude(altitude);
    let g_local = if r > 0.0 { params.mu / (r * r) } else { 0.0 };

    // Central forces: gravity, drag, buoyancy.
    let gravity = if r > 0.0 {
        -params.mu * body.mass / (r * r) * up
    } else {
        DVec3::ZERO
    };
    let drag = drag_force(
        &sample,
        body.velocity,
        params.drag_area,
        params.drag_coefficient,
    );
    let sub_vol = submerged_volume(
        voxels,
        com,
        body.position,
        body.orientation,
        params.surface_radius,
    );
    let buoyancy = buoyancy_force(sample.density, sub_vol, g_local, up);

    // Thrust (force + moment about the CoM).
    let (thrust, thrust_torque) = propulsion.thrust_step(body.orientation, com, dt);

    // Optional aero lift + transonic wave drag (winged craft), with the
    // restoring/damping pitching moment — the same terms `glide_step` applies.
    let (lift, lift_torque) = match &params.lift {
        Some(g) => {
            let forward = body.orientation * g.forward_local;
            let lift = aero::lift_force(&sample, body.velocity, forward, g.lift_area);
            let wave =
                aero::wave_drag_force(&sample, body.velocity, g.area_ruling_factor, g.lift_area);
            let cop = body.orientation * g.cop_offset_local;
            let restoring = aero::pitching_moment(cop, lift);
            let damping = aero::pitch_damping_moment(
                &sample,
                body.velocity.length(),
                body.angular_velocity(),
                g.lift_area,
                g.damping_length,
                g.pitch_damping,
            );
            (lift + wave, restoring + damping)
        }
        None => (DVec3::ZERO, DVec3::ZERO),
    };

    // Attitude control torque (RCS draws from the shared propellant graph), gated by
    // the resolved tier: an uncontrolled craft actuates nothing; a Direct craft
    // applies manual only (no stabilization); Stabilized applies SAS too (WI 562).
    let att_torque = attitude.control_torque_gated(
        body,
        &mut propulsion.graph,
        dt,
        tier.allows_manual(),
        tier.allows_stabilization(),
    );

    // Drain the control computer's electrical battery over the step (the standing
    // electricity consumer in the shared graph). Analytic catch-up; the only standing
    // graph consumer (thrust/RCS deduct propellant directly), so this touches only
    // the battery reservoir (WI 562).
    let g = &mut propulsion.graph;
    g.integrate(g.time + dt);

    let mut force = gravity + drag + buoyancy + thrust + lift;
    let mut torque = thrust_torque + lift_torque + att_torque;

    // Ground collision (WI 592): when a scene supplies a ground plane, add the penalty
    // contact wrench (craft shape derived from the voxels, placed at the CoM). `None`
    // leaves the wrench untouched — existing scenes are unaffected.
    if let Some(gc) = params.ground {
        let shape = craft_collision_shape(voxels);
        let bounds = craft_bounds(voxels);
        let ground = CollisionShape::HalfSpace {
            normal: gc.normal,
            offset: gc.offset,
        };
        let (cf, ct) = ground_contact_wrench(body, &shape, bounds, *dry_com, &ground, &gc.contact);
        force += cf;
        torque += ct;
    }

    pad.step(body, force, torque, dt);
    sample
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attitude::{AttitudeControl, ReactionWheels, Sas};
    use crate::command::SasMode;
    use crate::propulsion::{Engine, EngineCommand};
    use crate::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
    use crate::sim::CentralBody;
    use crate::voxel::{Material, Voxel};
    use glam::IVec3;

    const PROP: ResourceType = ResourceType(0);
    const BODY: CentralBody = CentralBody::EARTHLIKE;

    fn rocket() -> (FlightCraft, f64) {
        let mut voxels = VoxelCraft::new(1.0);
        for y in 0..5 {
            voxels.voxels.push(Voxel {
                cell: IVec3::new(0, y, 0),
                material: Material::COMPOSITE,
            });
        }
        let mp = voxels.mass_properties().unwrap();
        let propellant = 4_000.0;
        let propulsion = Propulsion {
            graph: ResourceGraph {
                reservoirs: vec![Reservoir::new(PROP, propellant, propellant)],
                ..Default::default()
            },
            tank_mounts: vec![DVec3::new(mp.center_of_mass.x, 0.5, mp.center_of_mass.z)],
            engines: vec![Engine {
                tank: ReservoirId(0),
                exhaust_velocity: 3_000.0,
                max_mass_flow: 60.0,
                mount: DVec3::new(mp.center_of_mass.x, 0.0, mp.center_of_mass.z),
                axis: DVec3::Y,
                max_gimbal: 0.0,
            }],
            commands: vec![EngineCommand::default()],
        };
        let attitude = AttitudePilot {
            sas: Sas::default(),
            manual: DVec3::ZERO,
            authority: 1_000.0,
            recapture_on_release: true,
            actuators: AttitudeControl {
                wheels: Some(ReactionWheels::new(5_000.0, 1e9)),
                rcs: None,
            },
        };
        let craft = FlightCraft {
            dry_mass: mp.mass,
            dry_com: mp.center_of_mass,
            voxels,
            propulsion,
            attitude,
            control: crate::control::ControlSystem::crewed_stabilized(),
            autopilot: None,
        };
        (craft, mp.mass + propellant)
    }

    #[test]
    fn uncontrolled_craft_rejects_manual_commands() {
        use crate::command::Command;
        let (mut craft, _) = rocket();
        craft.control = ControlSystem::default(); // no control point → Uncontrolled
        assert_eq!(craft.resolve_control(), ControlTier::Uncontrolled);
        let ok = craft.apply_command(&Command::SetThrottle(1.0), DQuat::IDENTITY);
        assert!(!ok, "uncontrolled craft rejects throttle");
        assert_eq!(
            craft.propulsion.commands[0].throttle, 0.0,
            "throttle unchanged on rejection"
        );
    }

    #[test]
    fn direct_craft_accepts_manual_throttle() {
        use crate::command::Command;
        let (mut craft, _) = rocket();
        craft.control = ControlSystem::crewed_manual(); // Direct
        assert_eq!(craft.resolve_control(), ControlTier::Direct);
        let ok = craft.apply_command(&Command::SetThrottle(0.5), DQuat::IDENTITY);
        assert!(ok, "direct craft accepts throttle");
        assert!((craft.propulsion.commands[0].throttle - 0.5).abs() < 1e-9);
    }

    #[test]
    fn engaging_sas_requires_command_core() {
        use crate::command::Command;
        let (mut craft, _) = rocket(); // crewed_stabilized → Stabilized
        assert!(
            craft.apply_command(&Command::SetSas(SasMode::Hold), DQuat::IDENTITY),
            "stabilized craft engages SAS"
        );
        // Drop to Direct (crewed manual, no command core).
        craft.attitude.sas.set_mode(SasMode::Off, DQuat::IDENTITY);
        craft.control = crate::control::ControlSystem::crewed_manual();
        assert_eq!(craft.resolve_control(), ControlTier::Direct);
        assert!(
            !craft.apply_command(&Command::SetSas(SasMode::Hold), DQuat::IDENTITY),
            "Direct refuses engaging SAS"
        );
        assert!(
            craft.apply_command(&Command::SetSas(SasMode::Off), DQuat::IDENTITY),
            "SetSas(Off) always allowed"
        );
    }

    #[test]
    fn sas_lost_when_command_core_battery_drains() {
        use crate::control::{ControlComputer, ControlPoint, ControlSystem, ELECTRICITY};
        let (mut craft, wet0) = rocket();
        // A crewed control point + a powered command core on a tiny battery.
        let bi = craft.propulsion.graph.reservoirs.len();
        craft
            .propulsion
            .graph
            .reservoirs
            .push(Reservoir::new(ELECTRICITY, 0.02, 100.0));
        let sys = ControlSystem {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::command_core(10.0)], // 10/s draw
            battery: Some(ReservoirId(bi)),
            ..Default::default()
        };
        craft
            .propulsion
            .graph
            .consumers
            .push(sys.power_consumer().unwrap());
        craft.control = sys;
        assert_eq!(
            craft.resolve_control(),
            ControlTier::Stabilized,
            "powered: stabilized"
        );

        let p = params(crate::medium::max_cross_section(&craft.voxels));
        let pad_radius = BODY.radius + craft.dry_com.y;
        let mut pad = LaunchPad::resting(pad_radius);
        let mut body = ActiveBody::new(
            DVec3::new(0.0, pad_radius, 0.0),
            DVec3::ZERO,
            wet0,
            craft.voxels.mass_properties().unwrap().inertia,
        );
        // 0.02 units at 10/s drains in 0.002 s — a couple of 4 ms steps.
        for _ in 0..5 {
            flight_step(&mut body, &mut craft, &p, &mut pad, 0.004);
        }
        assert!(
            craft.propulsion.graph.reservoirs[bi].amount <= 1e-9,
            "battery drained"
        );
        assert_eq!(
            craft.resolve_control(),
            ControlTier::Direct,
            "unpowered: SAS lost, falls to Direct"
        );
    }

    #[test]
    fn assembled_battery_drains_to_assist_offline_in_flight() {
        // WI 570: assemble the control system from placed devices (control point +
        // computer + battery), then drain it through flight_step. The available tier
        // stays Stabilized; the effective tier falls to Direct (crewed floor) and
        // `assist_offline` reports the low-power cause.
        use crate::control::{assemble_control, BatterySpec, ControlComputer};
        use crate::voxel::Device;
        use glam::IVec3;
        let (mut craft, wet0) = rocket();
        // Place real functional devices on the lattice.
        craft
            .voxels
            .devices
            .push(Device::control_point(IVec3::new(0, 0, 0), 50.0, true));
        craft.voxels.devices.push(Device::computer(
            IVec3::new(0, 1, 0),
            20.0,
            ControlComputer::command_core(10.0), // 10/s draw
        ));
        craft.voxels.devices.push(Device::battery(
            IVec3::new(0, 2, 0),
            30.0,
            BatterySpec::full(0.02), // tiny: drains in ~2 ms
        ));
        // Assemble control from the devices into the shared propulsion graph.
        craft.control = assemble_control(&craft.voxels, &mut craft.propulsion.graph);
        assert_eq!(craft.available_control(), ControlTier::Stabilized);
        assert_eq!(craft.resolve_control(), ControlTier::Stabilized);
        assert!(!craft.assist_offline());

        let p = params(crate::medium::max_cross_section(&craft.voxels));
        let pad_radius = BODY.radius + craft.dry_com.y;
        let mut pad = LaunchPad::resting(pad_radius);
        let mut body = ActiveBody::new(
            DVec3::new(0.0, pad_radius, 0.0),
            DVec3::ZERO,
            wet0,
            craft.voxels.mass_properties().unwrap().inertia,
        );
        for _ in 0..5 {
            flight_step(&mut body, &mut craft, &p, &mut pad, 0.004);
        }
        assert_eq!(
            craft.available_control(),
            ControlTier::Stabilized,
            "installed tier unchanged by depletion"
        );
        assert_eq!(
            craft.resolve_control(),
            ControlTier::Direct,
            "effective tier falls to the crewed floor"
        );
        assert!(craft.assist_offline(), "assist offline due to low power");
    }

    #[test]
    fn autopilot_engagement_requires_canned_tier() {
        use crate::autopilot::Autopilot;
        use crate::command::Command;
        let (mut craft, _) = rocket(); // crewed_stabilized → Stabilized (no Tier 1)
        assert!(
            !craft.apply_command(
                &Command::SetAutopilot(Some(Autopilot::Prograde)),
                DQuat::IDENTITY
            ),
            "Stabilized refuses engaging an autopilot"
        );
        assert!(craft.autopilot.is_none());
        // Disengage (None) is always allowed.
        assert!(craft.apply_command(&Command::SetAutopilot(None), DQuat::IDENTITY));
        // Upgrade to a Tier-1 computer → engaging is accepted.
        craft.control = crate::control::ControlSystem::crewed_canned();
        assert_eq!(craft.resolve_control(), ControlTier::Canned);
        assert!(craft.apply_command(
            &Command::SetAutopilot(Some(Autopilot::Prograde)),
            DQuat::IDENTITY
        ));
        assert_eq!(craft.autopilot, Some(Autopilot::Prograde));
    }

    #[test]
    fn gravity_turn_autopilot_ascends_and_gains_horizontal_velocity() {
        use crate::autopilot::{Autopilot, GravityTurn};
        let (mut craft, wet0) = rocket();
        craft.control = crate::control::ControlSystem::crewed_canned();
        // A capable engine + ample propellant so the demo has TWR to climb and time to
        // thrust through the turn (the stock rocket is deliberately marginal).
        craft.propulsion.engines[0].max_mass_flow = 200.0;
        craft.propulsion.graph.reservoirs[0] = Reservoir::new(PROP, 8_000.0, 8_000.0);
        craft.autopilot = Some(Autopilot::GravityTurn(GravityTurn {
            pitchover_altitude: 800.0,
            turn_end_altitude: 2_500.0,
            target_apoapsis: 120_000.0,
        }));

        let p = params(crate::medium::max_cross_section(&craft.voxels));
        let pad_radius = BODY.radius + craft.dry_com.y;
        let mut pad = LaunchPad::resting(pad_radius);
        let mut body = ActiveBody::new(
            DVec3::new(0.0, pad_radius, 0.0), // up = +Y; downrange = +X
            DVec3::ZERO,
            wet0,
            craft.voxels.mass_properties().unwrap().inertia,
        );
        let apo0 = crate::autopilot::apoapsis_radius(body.position, body.velocity, p.mu);
        // The autopilot owns throttle (gravity turn); no manual input. Climb + turn.
        for _ in 0..12_000 {
            flight_step(&mut body, &mut craft, &p, &mut pad, 0.004);
        }
        assert!(pad.released, "lifted off");
        assert!(
            body.position.length() - BODY.radius > 1_000.0,
            "climbed past pitchover"
        );
        assert!(
            body.velocity.x.abs() > 50.0,
            "gained horizontal (downrange) velocity"
        );
        let apo1 = crate::autopilot::apoapsis_radius(body.position, body.velocity, p.mu);
        assert!(
            apo1.unwrap_or(0.0) > apo0.unwrap_or(0.0).max(body.position.length()),
            "raised apoapsis (ascending toward orbit)"
        );
        assert!(body.position.is_finite() && body.velocity.is_finite());
    }

    #[test]
    fn gain_tuning_requires_tunable_tier() {
        use crate::command::Command;
        let (mut craft, _) = rocket();
        craft.control = crate::control::ControlSystem::crewed_canned(); // Canned < Tunable
        assert!(
            !craft.apply_command(&Command::SetSasGains(20.0, 10.0), DQuat::IDENTITY),
            "Canned tier refuses gain tuning"
        );
        craft.control = crate::control::ControlSystem::crewed_tunable();
        assert_eq!(craft.resolve_control(), ControlTier::Tunable);
        assert!(craft.apply_command(&Command::SetSasGains(20.0, 10.0), DQuat::IDENTITY));
        assert_eq!((craft.attitude.sas.kp, craft.attitude.sas.kd), (20.0, 10.0));
    }

    #[test]
    fn selecting_direct_disables_assist_then_restores() {
        // WI 571: a Stabilized craft downshifted to Direct refuses SAS engagement; the
        // available tier is unchanged; clearing the selection restores assist.
        use crate::command::Command;
        let (mut craft, _) = rocket(); // crewed_stabilized → Stabilized
        assert_eq!(craft.resolve_control(), ControlTier::Stabilized);

        assert!(
            craft.apply_command(
                &Command::SetControlTier(Some(ControlTier::Direct)),
                DQuat::IDENTITY
            ),
            "downshift is always permitted"
        );
        assert_eq!(
            craft.resolve_control(),
            ControlTier::Direct,
            "operating at Direct"
        );
        assert_eq!(
            craft.available_control(),
            ControlTier::Stabilized,
            "installed tier unchanged"
        );
        assert!(
            !craft.apply_command(&Command::SetSas(SasMode::Hold), DQuat::IDENTITY),
            "Direct refuses engaging SAS (assist off)"
        );

        // A selection above capability cannot lift the operating tier.
        craft.apply_command(
            &Command::SetControlTier(Some(ControlTier::Tunable)),
            DQuat::IDENTITY,
        );
        assert_eq!(craft.resolve_control(), ControlTier::Stabilized);

        // Clear the cap → assist available again.
        assert!(craft.apply_command(&Command::SetControlTier(None), DQuat::IDENTITY));
        assert_eq!(craft.resolve_control(), ControlTier::Stabilized);
        assert!(
            craft.apply_command(&Command::SetSas(SasMode::Hold), DQuat::IDENTITY),
            "assist restored after clearing the downshift"
        );
    }

    fn params(drag_area: f64) -> FlightParams {
        FlightParams {
            mu: BODY.mu,
            surface_radius: BODY.radius,
            medium: FluidMedium::EARTHLIKE,
            drag_area,
            drag_coefficient: 1.0,
            lift: None,
            ground: None,
        }
    }

    #[test]
    fn flight_step_with_ground_lands_and_rests() {
        // WI 599: collision live in the flight pipeline. A craft released above a flat ground,
        // no thrust, falls under gravity and the in-step penalty contact brings it to rest just
        // above the surface — no tunnelling, no kraken.
        let (mut craft, wet0) = rocket();
        let mut p = params(crate::medium::max_cross_section(&craft.voxels));
        let surface = BODY.radius;
        p.ground = Some(GroundContact {
            normal: DVec3::Y,
            offset: surface,
            contact: ContactParams::default(),
        });
        // Released pad so collision (not the pad) supports it; start a few metres up.
        let mut pad = LaunchPad::resting(surface);
        pad.released = true;
        let mut body = ActiveBody::new(
            DVec3::new(0.0, surface + craft.dry_com.y + 5.0, 0.0),
            DVec3::ZERO,
            wet0,
            craft.voxels.mass_properties().unwrap().inertia,
        );
        for _ in 0..8_000 {
            flight_step(&mut body, &mut craft, &p, &mut pad, 0.004);
        }
        let altitude = body.position.y - surface; // CoM height above the plane
        assert!(
            body.velocity.length() < 0.1,
            "came to rest: v={}",
            body.velocity.length()
        );
        // Rests with the base on the surface: CoM ≈ dry_com.y above it (tiny penalty sink).
        assert!(
            altitude > craft.dry_com.y - 0.2 && altitude <= craft.dry_com.y + 1e-3,
            "resting on the surface: altitude={altitude}, com.y={}",
            craft.dry_com.y
        );
        assert!(body.position.is_finite());
    }

    #[test]
    fn flight_step_launches_and_ascends() {
        let (mut craft, wet0) = rocket();
        let p = params(crate::medium::max_cross_section(&craft.voxels));
        let pad_radius = BODY.radius + craft.dry_com.y;
        let mut pad = LaunchPad::resting(pad_radius);
        let mut body = ActiveBody::new(
            DVec3::new(0.0, pad_radius, 0.0),
            DVec3::ZERO,
            wet0,
            craft.voxels.mass_properties().unwrap().inertia,
        );

        // Idle: held on the pad.
        for _ in 0..50 {
            flight_step(&mut body, &mut craft, &p, &mut pad, 0.004);
        }
        assert!(!pad.released, "idle: on the pad");

        // Throttle up via the propulsion command, hold attitude via SAS.
        craft
            .propulsion
            .apply_command(&crate::command::Command::SetThrottle(1.0));
        craft.attitude.sas.set_mode(SasMode::Hold, body.orientation);
        for _ in 0..2_000 {
            flight_step(&mut body, &mut craft, &p, &mut pad, 0.004);
        }
        assert!(pad.released, "lifts off under thrust");
        assert!(pad.altitude(&body) > 0.0, "climbed off the pad");
        assert!(body.velocity.y > 0.0, "ascending");
        assert!(body.position.is_finite() && body.velocity.is_finite());
        assert!(
            craft.propulsion.graph.reservoirs[0].amount < 4_000.0,
            "propellant burned"
        );
        // SAS hold kept the craft from tumbling.
        assert!(
            body.angular_velocity().length() < 1e-2,
            "attitude held during ascent"
        );
    }

    #[test]
    fn sounding_runs_launch_then_flight_then_recovery() {
        use crate::session::{GameSession, Phase};
        // A short sounding: small propellant → quick up-and-down.
        let (mut craft, _) = rocket();
        craft.propulsion.graph.reservoirs[0] = Reservoir::new(PROP, 150.0, 150.0);
        let wet0 = craft.dry_mass + 150.0;
        let p = params(crate::medium::max_cross_section(&craft.voxels));
        let pad_radius = BODY.radius + craft.dry_com.y;
        let mut pad = LaunchPad::resting(pad_radius);
        let mut body = ActiveBody::new(
            DVec3::new(0.0, pad_radius, 0.0),
            DVec3::ZERO,
            wet0,
            craft.voxels.mass_properties().unwrap().inertia,
        );
        let mut session = GameSession::new();
        session.begin_launch();
        craft
            .propulsion
            .apply_command(&crate::command::Command::SetThrottle(1.0));
        craft.attitude.sas.set_mode(SasMode::Hold, body.orientation);

        let mut saw_flight = false;
        for _ in 0..200_000 {
            flight_step(&mut body, &mut craft, &p, &mut pad, 0.004);
            let r = body.position.length();
            let up = body.position / r;
            let altitude = r - BODY.radius;
            let vertical_speed = body.velocity.dot(up);
            session.update(
                pad.released,
                altitude,
                vertical_speed,
                body.velocity.length(),
            );
            if session.phase == Phase::Flight {
                saw_flight = true;
            }
            if session.is_terminal() {
                break;
            }
        }
        assert!(saw_flight, "passed through Flight");
        assert!(
            session.is_terminal(),
            "reached Recovery (landed or crashed)"
        );
        assert!(body.position.is_finite());
    }

    #[test]
    fn flight_step_one_pipeline_is_finite_with_lift() {
        // With an optional lift profile present, the step stays finite (a winged
        // craft path through the same pipeline).
        let (mut craft, wet0) = rocket();
        let mut p = params(crate::medium::max_cross_section(&craft.voxels));
        p.lift = Some(GlideParams::for_craft(
            crate::medium::DescentParams {
                medium: FluidMedium::EARTHLIKE,
                mu: BODY.mu,
                surface_radius: BODY.radius,
                drag_area: p.drag_area,
                drag_coefficient: 1.0,
                slam_coefficient: 0.0, // flight pipeline does not model water-entry slam (WI 700)
            },
            &craft.voxels,
            crate::voxel::Axis::Y,
        ));
        let pad_radius = BODY.radius + craft.dry_com.y;
        let mut pad = LaunchPad::resting(pad_radius);
        let mut body = ActiveBody::new(
            DVec3::new(0.0, pad_radius, 0.0),
            DVec3::ZERO,
            wet0,
            craft.voxels.mass_properties().unwrap().inertia,
        );
        craft
            .propulsion
            .apply_command(&crate::command::Command::SetThrottle(1.0));
        for _ in 0..1_000 {
            flight_step(&mut body, &mut craft, &p, &mut pad, 0.004);
            assert!(body.position.is_finite() && body.velocity.is_finite());
        }
    }
}

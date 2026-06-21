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
}

impl FlightCraft {
    /// The craft's current control tier, resolved from its control devices and the
    /// power in its (shared) resource graph (WI 562).
    pub fn resolve_control(&self) -> ControlTier {
        self.control.resolve(&self.propulsion.graph)
    }

    /// Apply a manual command, **gated by the resolved control tier** (WI 562). An
    /// uncontrolled craft rejects every manual command (returns `false`, no state
    /// change); a controllable craft routes throttle/gimbal to propulsion and
    /// attitude/SAS to the attitude pilot. `current` is the craft orientation,
    /// captured as the SAS hold target. Returns whether the command was applied.
    ///
    /// Note: the *gate* only governs whether a command is accepted. Whether SAS
    /// actually produces stabilizing torque at the resolved tier is enforced in
    /// [`flight_step`] (Direct ⇒ no stabilization).
    pub fn apply_command(&mut self, cmd: &crate::command::Command, current: DQuat) -> bool {
        use crate::command::Command;
        if !self.resolve_control().allows_manual() {
            return false;
        }
        match cmd {
            Command::SetThrottle(_) | Command::SetGimbal(_) => self.propulsion.apply_command(cmd),
            Command::SetAttitude(_) | Command::SetSas(_) => {
                self.attitude.apply_command(cmd, current)
            }
            _ => false,
        }
    }
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
    let FlightCraft {
        voxels,
        dry_mass,
        dry_com,
        propulsion,
        attitude,
        control: _,
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

    let force = gravity + drag + buoyancy + thrust + lift;
    let torque = thrust_torque + lift_torque + att_torque;
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

    fn params(drag_area: f64) -> FlightParams {
        FlightParams {
            mu: BODY.mu,
            surface_radius: BODY.radius,
            medium: FluidMedium::EARTHLIKE,
            drag_area,
            drag_coefficient: 1.0,
            lift: None,
        }
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

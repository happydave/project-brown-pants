//! Telemetry: a serializable snapshot of the queryable simulation state.
//!
//! The snapshot reflects the authoritative current state (it is built from it,
//! not a separate model). WI 502's bus serves this over a transport; the AI
//! companion, second-screen, and replay read the same shape. Rendering-free.

use crate::control::ControlTier;
use crate::flight::FlightCraft;
use crate::orbit::Orbit;
use crate::rover::Rover;
use crate::sim::SimClock;
use serde::{Deserialize, Serialize};

/// A point-in-time snapshot of the simulation, as served to external clients.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Telemetry {
    pub time: f64,
    pub warp: f64,
    pub paused: bool,
    /// Gravitational parameter of the central body (lets a client plan burns).
    pub mu: f64,
    pub craft: Option<CraftTelemetry>,
    /// Energy-drift invariant metric, if available (WI 499).
    pub energy_drift: Option<f64>,
    /// Active-gear autonomy state of the live craft (WI 569), when an active
    /// `FlightCraft` exists. Distinct from the orbit-derived `craft` block because an
    /// active craft need not have an orbit-gear `Craft` (e.g. the `-- play` scene).
    /// Additive/serde-defaulted: snapshots without it deserialize to `None`.
    #[serde(default)]
    pub active: Option<ActiveFlightTelemetry>,
    /// Grounded-vehicle (rover) state of the live craft (WI 640), when a scene owns a
    /// [`Rover`] (the `-- rover` scene or the workshop Test driving an assembled rover).
    /// Pose, velocity, and contact/wheel signals — the introspection the 631/634 work
    /// lacked. Distinct from `active` (a `FlightCraft`'s autonomy gear) and `craft` (an
    /// orbit). Additive/serde-defaulted: snapshots without it deserialize to `None`.
    #[serde(default)]
    pub rover: Option<RoverTelemetry>,
}

/// The active craft's autonomy state on the bus (WI 569): the control-tier model from
/// WI 562/570, derived live from a [`FlightCraft`].
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveFlightTelemetry {
    /// The **effective** control tier — what is currently operating (the executor gate).
    pub control_tier: ControlTier,
    /// The **available** (installed) control tier — charge-independent (WI 570).
    pub available_tier: ControlTier,
    /// Whether powered assistance is offline due to low power (WI 570): effective is
    /// below available because the battery reached its reserve.
    pub assist_offline: bool,
}

impl ActiveFlightTelemetry {
    /// Snapshot the autonomy state of a live craft.
    pub fn from_flight(craft: &FlightCraft) -> Self {
        Self {
            control_tier: craft.resolve_control(),
            available_tier: craft.available_control(),
            assist_offline: craft.assist_offline(),
        }
    }
}

/// A grounded rover's live state on the bus (WI 640): pose, velocity, and the
/// contact/wheel signals that make the running rover introspectable (the data the
/// 631/634 work could not see). Captured purely from a [`Rover`]; rendering-free.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RoverTelemetry {
    /// Body position relative to the attractor `[x, y, z]` (normalised units).
    pub position: [f64; 3],
    /// Orientation (body → world) as a quaternion `[x, y, z, w]`.
    pub orientation: [f64; 4],
    /// Linear velocity `[x, y, z]`.
    pub velocity: [f64; 3],
    /// Angular velocity (world frame) `[x, y, z]` (rad/s).
    pub angular_velocity: [f64; 3],
    /// Contact-jitter metric driving the rollover-safe angular damping (WI 611).
    pub contact_jitter: f64,
    /// Hull penetration depth into the terrain (m) — the hull-shell contact signal (WI 634).
    pub hull_penetration: f64,
    /// Whether the rover is touching the ground at all (WI 642): any wheel in contact or the hull
    /// penetrating. Disambiguates `contact_jitter` — `false` here means a jitter spike is the wheels
    /// leaving the ground, not a live load. Additive/serde-defaulted.
    #[serde(default)]
    pub grounded: bool,
    /// Per-wheel state, in build order.
    pub wheels: Vec<WheelTelemetry>,
}

/// One wheel's live state on the bus (WI 640).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct WheelTelemetry {
    /// Axle (wheel-centre) drop below the chassis mount (m) — the quarter-car state (WI 631a).
    pub axle_drop: f64,
    /// Static ground load this corner carries at rest (N) (WI 631a).
    pub static_load: f64,
    /// Steering angle about the body up axis (rad), including any bent-rim bias.
    pub steer: f64,
    /// Wheel spin (rad/s).
    pub spin: f64,
    /// Longitudinal slip stiffness in effect (WI 630).
    pub slip_long: f64,
    /// Lateral slip stiffness in effect (WI 630).
    pub slip_lat: f64,
    /// Tire grip multiplier currently in effect (WI 630) — collapses on a blowout.
    pub grip_scale: f64,
    /// Sheared off by a hard impact (WI 618): the corner behaves as if it has no wheel.
    pub inert: bool,
    /// Tire blown — runs rigid on the rim (WI 631b). Latched.
    pub tire_blown: bool,
    /// Rim bent — added rolling resistance and a steer bias (WI 631b). Latched.
    pub rim_bent: bool,
    /// Suspension damper blown — bouncy corner (WI 631b). Latched.
    pub damper_blown: bool,
    /// Whether this wheel carried ground load on the last step (WI 642): `false` if airborne or
    /// sheared. Additive/serde-defaulted. Lets a reader tell a grazing wheel from a loaded one.
    #[serde(default)]
    pub tire_contact: bool,
}

impl RoverTelemetry {
    /// Snapshot the live state of a grounded rover. Pure — borrows the authoritative
    /// [`Rover`] only (no terrain, clock, or rendering), so any scene that owns a rover
    /// can publish it each frame.
    pub fn from_rover(rover: &Rover) -> Self {
        let b = &rover.body;
        let omega = b.angular_velocity();
        Self {
            position: [b.position.x, b.position.y, b.position.z],
            orientation: [
                b.orientation.x,
                b.orientation.y,
                b.orientation.z,
                b.orientation.w,
            ],
            velocity: [b.velocity.x, b.velocity.y, b.velocity.z],
            angular_velocity: [omega.x, omega.y, omega.z],
            contact_jitter: rover.contact_jitter,
            hull_penetration: rover.hull_penetration,
            grounded: rover.hull_penetration > 0.0 || rover.wheels.iter().any(|w| w.tire_contact),
            wheels: rover
                .wheels
                .iter()
                .map(|w| WheelTelemetry {
                    axle_drop: w.axle_drop,
                    static_load: w.static_load,
                    steer: w.steer,
                    spin: w.spin,
                    slip_long: w.slip_long,
                    slip_lat: w.slip_lat,
                    grip_scale: w.grip_scale,
                    inert: w.inert,
                    tire_blown: w.tire_blown,
                    rim_bent: w.rim_bent,
                    damper_blown: w.damper_blown,
                    tire_contact: w.tire_contact,
                })
                .collect(),
        }
    }
}

/// The craft's orbit and current state.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CraftTelemetry {
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    pub arg_periapsis: f64,
    pub periapsis_radius: f64,
    pub apoapsis_radius: f64,
    /// Current world position `[x, y]` at the snapshot time.
    pub position: [f64; 2],
    /// Current world velocity `[x, y]` at the snapshot time.
    pub velocity: [f64; 2],
}

impl Telemetry {
    /// Builds a snapshot from the authoritative state. Pure. `mu` is the central
    /// body's gravitational parameter.
    pub fn capture(
        clock: &SimClock,
        orbit: Option<&Orbit>,
        mu: f64,
        energy_drift: Option<f64>,
    ) -> Telemetry {
        let craft = orbit.map(|o| {
            let (p, v) = o.position_velocity(clock.time);
            CraftTelemetry {
                semi_major_axis: o.semi_major_axis,
                eccentricity: o.eccentricity,
                arg_periapsis: o.arg_periapsis,
                periapsis_radius: o.periapsis_radius(),
                apoapsis_radius: o.apoapsis_radius(),
                position: [p.x, p.y],
                velocity: [v.x, v.y],
            }
        });
        Telemetry {
            time: clock.time,
            warp: clock.warp,
            paused: clock.paused,
            mu,
            craft,
            energy_drift,
            active: None,
            rover: None,
        }
    }

    /// Attach active-gear autonomy state to this snapshot (WI 569). The active block is
    /// the single home for the control tier; clients read `active.control_tier`. (The
    /// `craft` block describes the orbit-gear propagator — a distinct craft — so the
    /// tier is deliberately not mirrored onto it. WI 579.) Builder-style so the publisher
    /// can layer it onto `capture` without changing the orbit-only `capture` signature.
    pub fn with_active_flight(mut self, active: ActiveFlightTelemetry) -> Self {
        self.active = Some(active);
        self
    }

    /// Attach grounded-rover state to this snapshot (WI 640). Builder-style so a
    /// rover-bearing scene can layer it onto `capture` without changing the orbit-only
    /// `capture` signature, mirroring [`Telemetry::with_active_flight`].
    pub fn with_rover(mut self, rover: RoverTelemetry) -> Self {
        self.rover = Some(rover);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;

    #[test]
    fn capture_reflects_state_and_serializes() {
        let clock = SimClock {
            time: 0.0,
            warp: 8.0,
            paused: true,
            ..Default::default()
        };
        let orbit =
            Orbit::from_state(1.0, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0), 0.0).unwrap();

        let snap = Telemetry::capture(&clock, Some(&orbit), 1.0, Some(1e-12));
        assert_eq!(snap.warp, 8.0);
        assert!(snap.paused);
        let craft = snap.craft.as_ref().unwrap();
        assert!((craft.semi_major_axis - 1.0).abs() < 1e-9);
        // At t=0 the craft is at (1, 0).
        assert!((craft.position[0] - 1.0).abs() < 1e-9 && craft.position[1].abs() < 1e-9);

        let json = serde_json::to_value(&snap).unwrap();
        assert_eq!(json["warp"], 8.0);
        assert_eq!(json["craft"]["eccentricity"], 0.0);
        assert!(json["craft"]["position"].is_array());
    }

    #[test]
    fn capture_without_craft_is_none() {
        let clock = SimClock::default();
        let snap = Telemetry::capture(&clock, None, 1.0, None);
        assert!(snap.craft.is_none());
        assert!(snap.energy_drift.is_none());
        assert!(snap.active.is_none(), "no active block by default (WI 569)");
    }

    // --- WI 569: flight-aware active block ---

    #[test]
    fn active_block_derives_from_flight_and_attaches() {
        use crate::control::{assemble_control, BatterySpec, ControlComputer};
        use crate::voxel::{Device, VoxelCraft};
        use glam::IVec3;
        // A device-assembled craft: crewed point + Tier-2 computer + battery → Tunable.
        let mut voxels = VoxelCraft::new(1.0);
        voxels
            .devices
            .push(Device::control_point(IVec3::ZERO, 50.0, true));
        voxels.devices.push(Device::computer(
            IVec3::new(0, 1, 0),
            10.0,
            ControlComputer::tuning_computer(0.5),
        ));
        voxels.devices.push(Device::battery(
            IVec3::new(0, 2, 0),
            20.0,
            BatterySpec::full(100.0),
        ));
        let mut craft = FlightCraft {
            voxels: voxels.clone(),
            dry_mass: 1.0,
            dry_com: glam::DVec3::ZERO,
            propulsion: crate::propulsion::Propulsion {
                graph: crate::resource::ResourceGraph::default(),
                tank_mounts: vec![],
                engines: vec![],
                commands: vec![],
            },
            attitude: crate::attitude::AttitudePilot {
                sas: crate::attitude::Sas::default(),
                manual: glam::DVec3::ZERO,
                authority: 1.0,
                recapture_on_release: true,
                actuators: crate::attitude::AttitudeControl {
                    wheels: None,
                    rcs: None,
                },
            },
            control: crate::control::ControlSystem::default(),
            autopilot: None,
        };
        craft.control = assemble_control(&voxels, &mut craft.propulsion.graph);

        let active = ActiveFlightTelemetry::from_flight(&craft);
        assert_eq!(active.control_tier, ControlTier::Tunable);
        assert_eq!(active.available_tier, ControlTier::Tunable);
        assert!(!active.assist_offline);

        // Attaching to an orbit-bearing snapshot fills the top-level active block; the
        // orbit `craft` block is a distinct craft and carries no tier (WI 579).
        let clock = SimClock::default();
        let orbit =
            Orbit::from_state(1.0, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0), 0.0).unwrap();
        let snap = Telemetry::capture(&clock, Some(&orbit), 1.0, None).with_active_flight(active);
        assert!(snap.craft.is_some());
        assert_eq!(snap.active.unwrap().control_tier, ControlTier::Tunable);
    }

    #[test]
    fn active_block_attaches_without_an_orbit_craft() {
        // The orbit-less case (e.g. -- play): no `craft` block, but the active block is
        // still published at the top level so a client can read the tier.
        let clock = SimClock::default();
        let active = ActiveFlightTelemetry {
            control_tier: ControlTier::Direct,
            available_tier: ControlTier::Stabilized,
            assist_offline: true,
        };
        let snap = Telemetry::capture(&clock, None, 1.0, None).with_active_flight(active);
        assert!(snap.craft.is_none());
        assert_eq!(snap.active.unwrap().control_tier, ControlTier::Direct);
        assert!(snap.active.unwrap().assist_offline);
    }

    #[test]
    fn active_block_is_backward_compatible_over_json() {
        // A legacy snapshot (no `active`) deserializes to None; a snapshot with it
        // round-trips and a client can read the tier.
        let legacy =
            r#"{"time":0.0,"warp":1.0,"paused":false,"mu":1.0,"craft":null,"energy_drift":null}"#;
        let snap: Telemetry = serde_json::from_str(legacy).unwrap();
        assert!(snap.active.is_none());

        let active = ActiveFlightTelemetry {
            control_tier: ControlTier::Canned,
            available_tier: ControlTier::Canned,
            assist_offline: false,
        };
        let snap =
            Telemetry::capture(&SimClock::default(), None, 1.0, None).with_active_flight(active);
        let json = serde_json::to_string(&snap).unwrap();
        let back: Telemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.active.unwrap().control_tier, ControlTier::Canned);
    }

    // --- WI 640: grounded/rover block ---

    fn test_rover() -> crate::rover::Rover {
        use crate::active::ActiveBody;
        use crate::rover::{Rover, Wheel};
        use glam::{DMat3, DVec3};
        let body = ActiveBody::new(
            DVec3::new(1.0, 2.0, 3.0),
            DVec3::new(0.5, 0.0, -0.5),
            1000.0,
            DMat3::IDENTITY,
        )
        .with_angular_velocity(DVec3::new(0.0, 0.1, 0.0));
        let wheels = vec![
            Wheel::new(DVec3::new(1.0, 0.0, 1.0)),
            Wheel::new(DVec3::new(-1.0, 0.0, 1.0)),
        ];
        Rover::new(body, wheels, 9.81)
    }

    #[test]
    fn rover_block_derives_from_rover_and_attaches() {
        let mut rover = test_rover();
        rover.hull_penetration = 0.04;
        rover.wheels[1].tire_blown = true;
        rover.wheels[1].inert = true;

        rover.wheels[0].tire_contact = true;
        let rt = RoverTelemetry::from_rover(&rover);
        assert_eq!(rt.position, [1.0, 2.0, 3.0]);
        assert_eq!(rt.velocity, [0.5, 0.0, -0.5]);
        assert!((rt.hull_penetration - 0.04).abs() < 1e-12);
        assert_eq!(rt.wheels.len(), 2);
        assert!(!rt.wheels[0].tire_blown);
        assert!(rt.wheels[1].tire_blown && rt.wheels[1].inert);
        // Contact flags (WI 642): wheel 0 carries load, wheel 1 doesn't; the rover is grounded
        // (a wheel in contact, and the hull penetrating).
        assert!(rt.wheels[0].tire_contact && !rt.wheels[1].tire_contact);
        assert!(rt.grounded);
        // The angular-velocity y-component reflects the imposed spin.
        assert!((rt.angular_velocity[1] - 0.1).abs() < 1e-9);

        // Attaches alongside the orbit-less case (a rover scene has no orbit `craft`).
        let snap = Telemetry::capture(&SimClock::default(), None, 1.0, None).with_rover(rt);
        assert!(snap.craft.is_none());
        assert!(snap.active.is_none());
        let attached = snap.rover.as_ref().unwrap();
        assert_eq!(attached.wheels.len(), 2);
    }

    #[test]
    fn rover_block_is_backward_compatible_over_json() {
        // A legacy snapshot (no `rover`, no `active`) deserializes to None for both.
        let legacy =
            r#"{"time":0.0,"warp":1.0,"paused":false,"mu":1.0,"craft":null,"energy_drift":null}"#;
        let snap: Telemetry = serde_json::from_str(legacy).unwrap();
        assert!(snap.rover.is_none());
        assert!(snap.active.is_none());
    }

    #[test]
    fn rover_block_round_trips_over_json() {
        let rt = RoverTelemetry::from_rover(&test_rover());
        let snap = Telemetry::capture(&SimClock::default(), None, 1.0, None).with_rover(rt);
        let json = serde_json::to_string(&snap).unwrap();
        let back: Telemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, snap);
    }
}

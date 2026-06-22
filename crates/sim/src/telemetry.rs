//! Telemetry: a serializable snapshot of the queryable simulation state.
//!
//! The snapshot reflects the authoritative current state (it is built from it,
//! not a separate model). WI 502's bus serves this over a transport; the AI
//! companion, second-screen, and replay read the same shape. Rendering-free.

use crate::control::ControlTier;
use crate::flight::FlightCraft;
use crate::orbit::Orbit;
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
}

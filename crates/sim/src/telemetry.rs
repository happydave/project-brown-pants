//! Telemetry: a serializable snapshot of the queryable simulation state.
//!
//! The snapshot reflects the authoritative current state (it is built from it,
//! not a separate model). WI 502's bus serves this over a transport; the AI
//! companion, second-screen, and replay read the same shape. Rendering-free.

use crate::control::ControlTier;
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
    /// The craft's resolved control tier (WI 562), when known. The orbit-gear bus
    /// `capture` leaves this `None` (it has no `FlightCraft`); a flight-aware path
    /// (e.g. the flight HUD reading `FlightCraft::resolve_control`) supplies it.
    #[serde(default)]
    pub control_tier: Option<ControlTier>,
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
                control_tier: None,
            }
        });
        Telemetry {
            time: clock.time,
            warp: clock.warp,
            paused: clock.paused,
            mu,
            craft,
            energy_drift,
        }
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
    }
}

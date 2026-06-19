//! Telemetry: a serializable snapshot of the queryable simulation state.
//!
//! The snapshot reflects the authoritative current state (it is built from it,
//! not a separate model). WI 502's bus serves this over a transport; the AI
//! companion, second-screen, and replay read the same shape. Rendering-free.

use crate::orbit::Orbit;
use crate::sim::SimClock;
use serde::Serialize;

/// A point-in-time snapshot of the simulation, as served to external clients.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Telemetry {
    pub time: f64,
    pub warp: f64,
    pub paused: bool,
    pub craft: Option<CraftTelemetry>,
    /// Energy-drift invariant metric, if available (WI 499).
    pub energy_drift: Option<f64>,
}

/// The craft's orbit and current position.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct CraftTelemetry {
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    pub arg_periapsis: f64,
    pub periapsis_radius: f64,
    pub apoapsis_radius: f64,
    /// Current world position `[x, y]` at the snapshot time.
    pub position: [f64; 2],
}

impl Telemetry {
    /// Builds a snapshot from the authoritative state. Pure.
    pub fn capture(
        clock: &SimClock,
        orbit: Option<&Orbit>,
        energy_drift: Option<f64>,
    ) -> Telemetry {
        let craft = orbit.map(|o| {
            let p = o.position(clock.time);
            CraftTelemetry {
                semi_major_axis: o.semi_major_axis,
                eccentricity: o.eccentricity,
                arg_periapsis: o.arg_periapsis,
                periapsis_radius: o.periapsis_radius(),
                apoapsis_radius: o.apoapsis_radius(),
                position: [p.x, p.y],
            }
        });
        Telemetry {
            time: clock.time,
            warp: clock.warp,
            paused: clock.paused,
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

        let snap = Telemetry::capture(&clock, Some(&orbit), Some(1e-12));
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
        let snap = Telemetry::capture(&clock, None, None);
        assert!(snap.craft.is_none());
        assert!(snap.energy_drift.is_none());
    }
}

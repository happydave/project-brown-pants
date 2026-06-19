//! The companion's decision logic.
//!
//! A [`Brain`] maps a telemetry snapshot to a [`Decision`]. It reasons only from
//! telemetry — no privileged access to the simulation. An LLM-backed brain can
//! replace [`NavigatorBrain`] without touching the bus-client loop.

use glam::DVec2;
use sounding_sim::command::Command;
use sounding_sim::telemetry::Telemetry;

/// What the companion decides to do this tick.
pub enum Decision {
    /// Do nothing; carries narration.
    Idle(String),
    /// Issue a command; carries narration.
    Act(Command, String),
}

/// Maps telemetry to a decision. Implementors reason only from telemetry.
pub trait Brain {
    fn decide(&mut self, telemetry: &Telemetry) -> Decision;
}

/// Eccentricity at or below which the orbit is treated as circular.
const CIRCULAR_E: f64 = 1e-3;
/// Warp factor used while coasting toward apoapsis.
const COAST_WARP: f64 = 8.0;

/// A deterministic navigator pursuing one goal: circularize the orbit. It coasts
/// (under warp) to apoapsis, slows on approach, and orders a prograde burn sized
/// to circular speed. The "autopilot" end of the tutor↔autopilot spectrum.
#[derive(Default)]
pub struct NavigatorBrain;

impl Brain for NavigatorBrain {
    fn decide(&mut self, t: &Telemetry) -> Decision {
        let Some(craft) = &t.craft else {
            return Decision::Idle("No craft on telemetry. Standing by.".to_string());
        };
        if craft.eccentricity <= CIRCULAR_E {
            return Decision::Idle("Orbit circular. Nominal.".to_string());
        }

        let r = (craft.position[0].powi(2) + craft.position[1].powi(2)).sqrt();
        let r_apo = craft.apoapsis_radius;

        if r < r_apo * 0.95 {
            return Decision::Act(
                Command::SetWarp(COAST_WARP),
                "Eccentric orbit. Warping toward apoapsis.".to_string(),
            );
        }
        if r < r_apo * 0.99 {
            return Decision::Act(
                Command::SetWarp(1.0),
                "Approaching apoapsis. Slowing down.".to_string(),
            );
        }

        // At apoapsis: a prograde burn from apoapsis speed up to circular speed.
        let speed = (craft.velocity[0].powi(2) + craft.velocity[1].powi(2)).sqrt();
        let v_circ = (t.mu / r_apo).sqrt();
        let dv_mag = v_circ - speed;
        let delta_v = DVec2::new(craft.velocity[0], craft.velocity[1]).normalize_or_zero() * dv_mag;
        Decision::Act(
            Command::ExecuteManeuver { delta_v },
            format!("At apoapsis. Circularizing with a {dv_mag:.4} prograde burn."),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sounding_sim::orbit::Orbit;
    use sounding_sim::sim::SimClock;

    fn telemetry_at(orbit: &Orbit, time: f64) -> Telemetry {
        let clock = SimClock {
            time,
            warp: 1.0,
            paused: false,
        };
        Telemetry::capture(&clock, Some(orbit), 1.0, None)
    }

    fn eccentric() -> Orbit {
        Orbit::from_state(1.0, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.3), 0.0).unwrap()
    }

    #[test]
    fn idles_when_circular() {
        let orbit =
            Orbit::from_state(1.0, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0), 0.0).unwrap();
        let mut brain = NavigatorBrain;
        assert!(matches!(
            brain.decide(&telemetry_at(&orbit, 0.0)),
            Decision::Idle(_)
        ));
    }

    #[test]
    fn warps_toward_apoapsis_when_far() {
        // At periapsis (t=0) the craft is far from apoapsis.
        let orbit = eccentric();
        let mut brain = NavigatorBrain;
        match brain.decide(&telemetry_at(&orbit, 0.0)) {
            Decision::Act(Command::SetWarp(w), _) => assert!(w > 1.0),
            _ => panic!("expected a warp command far from apoapsis"),
        }
    }

    #[test]
    fn circularizes_at_apoapsis() {
        let orbit = eccentric();
        let t_apo = orbit.period() / 2.0; // half a period from periapsis
        let mut brain = NavigatorBrain;
        match brain.decide(&telemetry_at(&orbit, t_apo)) {
            Decision::Act(Command::ExecuteManeuver { delta_v }, _) => {
                let after = orbit.with_maneuver(t_apo, delta_v).unwrap();
                assert!(
                    after.eccentricity < 1e-3,
                    "burn should circularize, got e={}",
                    after.eccentricity
                );
            }
            _ => panic!("expected a circularizing maneuver at apoapsis"),
        }
    }
}

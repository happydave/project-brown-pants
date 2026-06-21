//! Tier-1 canned autopilots (WI 565).
//!
//! Outer-loop control policies: each reads the craft's flight state and produces an
//! attitude target (a world-frame nose direction, driving `SasMode::Point`) and an
//! optional throttle. They are **bus clients in authority** — the flight step applies
//! their output through the same SAS/throttle paths a player uses, so command
//! arbitration (WI 563) lets manual input override them and the tier gate (WI 562/564)
//! requires a powered Tier-1 (`Canned`) computer to run them.
//!
//! Pure and deterministic (state in, output out); evaluated every active sub-step.
//! In scope here: orbital-frame attitude holds and a gravity-turn ascent. Hover-hold
//! (throttle PID) and maneuver-node execution are a separate follow-up.

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Parameters of a gravity-turn ascent.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GravityTurn {
    /// Altitude (m above surface) at which to begin pitching over from vertical.
    pub pitchover_altitude: f64,
    /// Altitude (m above surface) by which the turn has blended fully to prograde.
    pub turn_end_altitude: f64,
    /// Target apoapsis (m above surface); throttle cuts once apoapsis reaches it.
    pub target_apoapsis: f64,
}

impl GravityTurn {
    /// A gravity turn to a circular-ish target apoapsis (sensible defaults).
    pub fn to_apoapsis(target_apoapsis: f64) -> Self {
        Self {
            pitchover_altitude: 1_000.0,
            turn_end_altitude: 45_000.0,
            target_apoapsis,
        }
    }

    fn evaluate(
        &self,
        position: DVec3,
        velocity: DVec3,
        surface_radius: f64,
        mu: f64,
    ) -> AutopilotOutput {
        let altitude = position.length() - surface_radius;
        let up = position.normalize_or_zero();
        // A horizontal downrange direction perpendicular to `up` (deterministic): the
        // turn blends vertical → downrange with altitude. Pitching toward a *horizontal*
        // direction (not toward prograde, which on a vertical climb equals up) is what
        // actually builds horizontal velocity.
        let mut downrange = (DVec3::X - DVec3::X.dot(up) * up).normalize_or_zero();
        if downrange == DVec3::ZERO {
            downrange = (DVec3::Z - DVec3::Z.dot(up) * up).normalize_or_zero();
        }
        let frac = ((altitude - self.pitchover_altitude)
            / (self.turn_end_altitude - self.pitchover_altitude).max(1.0))
        .clamp(0.0, 1.0);
        let target = if frac <= 0.0 || downrange == DVec3::ZERO {
            up
        } else {
            let blended = up.lerp(downrange, frac).normalize_or_zero();
            if blended == DVec3::ZERO {
                up
            } else {
                blended
            }
        };
        // Full throttle until apoapsis reaches the target, then coast.
        let target_apo_r = surface_radius + self.target_apoapsis;
        let throttle = match apoapsis_radius(position, velocity, mu) {
            Some(a) if a >= target_apo_r => 0.0,
            _ => 1.0,
        };
        AutopilotOutput {
            attitude_target: Some(target),
            throttle: Some(throttle),
        }
    }
}

/// A canned autopilot behaviour (Tier 1).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Autopilot {
    /// Point the nose along the velocity vector.
    Prograde,
    /// Point opposite the velocity vector.
    Retrograde,
    /// Point along the orbit normal (r×v).
    Normal,
    /// Point opposite the orbit normal.
    Antinormal,
    /// Point toward the central body (−r).
    RadialIn,
    /// Point away from the central body (+r).
    RadialOut,
    /// Vertical-then-prograde ascent to a target apoapsis.
    GravityTurn(GravityTurn),
}

/// What an autopilot commands this step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AutopilotOutput {
    /// Desired world-frame nose direction (drives `SasMode::Point`), if any.
    pub attitude_target: Option<DVec3>,
    /// Desired throttle in `[0, 1]`, if this autopilot controls throttle.
    pub throttle: Option<f64>,
}

impl Autopilot {
    /// Whether this autopilot drives throttle (and so should own it while engaged).
    pub fn controls_throttle(self) -> bool {
        matches!(self, Autopilot::GravityTurn(_))
    }

    /// Compute the commands for this autopilot from the craft's flight state.
    pub fn evaluate(
        self,
        position: DVec3,
        velocity: DVec3,
        surface_radius: f64,
        mu: f64,
    ) -> AutopilotOutput {
        match self {
            Autopilot::GravityTurn(gt) => gt.evaluate(position, velocity, surface_radius, mu),
            hold => AutopilotOutput {
                attitude_target: hold_direction(hold, position, velocity),
                throttle: None,
            },
        }
    }
}

/// The world-frame unit direction for an orbital-frame attitude hold, or `None` for a
/// degenerate state (e.g. zero velocity for prograde, r∥v for normal) or a non-hold.
pub fn hold_direction(mode: Autopilot, position: DVec3, velocity: DVec3) -> Option<DVec3> {
    let d = match mode {
        Autopilot::Prograde => velocity.normalize_or_zero(),
        Autopilot::Retrograde => -velocity.normalize_or_zero(),
        Autopilot::RadialOut => position.normalize_or_zero(),
        Autopilot::RadialIn => -position.normalize_or_zero(),
        Autopilot::Normal => position.cross(velocity).normalize_or_zero(),
        Autopilot::Antinormal => -position.cross(velocity).normalize_or_zero(),
        Autopilot::GravityTurn(_) => DVec3::ZERO,
    };
    (d != DVec3::ZERO).then_some(d)
}

/// Apoapsis radius (m from the body centre) of the conic through `(position,
/// velocity)` under `mu`, or `None` if unbound/degenerate (no apoapsis).
pub fn apoapsis_radius(position: DVec3, velocity: DVec3, mu: f64) -> Option<f64> {
    let r = position.length();
    if r <= 0.0 || mu <= 0.0 {
        return None;
    }
    let energy = velocity.length_squared() / 2.0 - mu / r;
    if energy >= 0.0 {
        return None; // unbound: no apoapsis
    }
    let a = -mu / (2.0 * energy);
    let h = position.cross(velocity).length();
    let e = (1.0 + 2.0 * energy * h * h / (mu * mu)).max(0.0).sqrt();
    Some(a * (1.0 + e))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU: f64 = 3.986e14;
    const RE: f64 = 6.36e6;

    #[test]
    fn holds_point_along_orbital_frame() {
        let pos = DVec3::new(RE + 200_000.0, 0.0, 0.0);
        let vel = DVec3::new(0.0, 7_700.0, 0.0);
        assert!(
            (hold_direction(Autopilot::Prograde, pos, vel).unwrap() - DVec3::Y).length() < 1e-9
        );
        assert!(
            (hold_direction(Autopilot::Retrograde, pos, vel).unwrap() + DVec3::Y).length() < 1e-9
        );
        assert!(
            (hold_direction(Autopilot::RadialOut, pos, vel).unwrap() - DVec3::X).length() < 1e-9
        );
        assert!(
            (hold_direction(Autopilot::RadialIn, pos, vel).unwrap() + DVec3::X).length() < 1e-9
        );
        assert!((hold_direction(Autopilot::Normal, pos, vel).unwrap() - DVec3::Z).length() < 1e-9);
    }

    #[test]
    fn degenerate_hold_is_none() {
        let pos = DVec3::new(RE, 0.0, 0.0);
        assert!(hold_direction(Autopilot::Prograde, pos, DVec3::ZERO).is_none());
    }

    #[test]
    fn apoapsis_of_circular_orbit_is_radius() {
        let r = RE + 300_000.0;
        let v = (MU / r).sqrt(); // circular speed
        let apo = apoapsis_radius(DVec3::new(r, 0.0, 0.0), DVec3::new(0.0, v, 0.0), MU).unwrap();
        assert!((apo - r).abs() / r < 1e-6, "circular apoapsis ≈ radius");
        // A fast (escape) state has no apoapsis.
        assert!(
            apoapsis_radius(DVec3::new(r, 0.0, 0.0), DVec3::new(0.0, 2.0 * v, 0.0), MU).is_none()
        );
    }

    #[test]
    fn gravity_turn_is_vertical_low_then_tilts_and_cuts() {
        let gt = GravityTurn::to_apoapsis(200_000.0);
        // On the pad (just above surface, ~no horizontal speed): vertical, full throttle.
        let low = Autopilot::GravityTurn(gt).evaluate(
            DVec3::new(RE + 100.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            RE,
            MU,
        );
        let up = DVec3::X;
        assert!(
            (low.attitude_target.unwrap() - up).length() < 1e-3,
            "vertical low"
        );
        assert_eq!(low.throttle, Some(1.0));

        // High with horizontal velocity: target tilts toward prograde (away from up).
        let high = Autopilot::GravityTurn(gt).evaluate(
            DVec3::new(RE + 40_000.0, 0.0, 0.0),
            DVec3::new(1_000.0, 2_000.0, 0.0),
            RE,
            MU,
        );
        let t = high.attitude_target.unwrap();
        assert!(t.dot(up) < 0.999, "tilted off vertical toward downrange");

        // Apoapsis already above target → throttle cut.
        let r = RE + 250_000.0;
        let v = (MU / r).sqrt();
        let coast = Autopilot::GravityTurn(gt).evaluate(
            DVec3::new(r, 0.0, 0.0),
            DVec3::new(0.0, v, 0.0),
            RE,
            MU,
        );
        assert_eq!(coast.throttle, Some(0.0), "coast once apoapsis reached");
    }
}

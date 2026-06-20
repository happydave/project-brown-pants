//! Launch-pad rest and release — surface launch (WI 532).
//!
//! The ground for a launching craft, modelled as a **unilateral vertical support**
//! — *not* the wheel-contact problem (no rolling, slip, or LOD), and *not* a stiff
//! spring contact. A craft on the pad is held at exact rest (zero velocity, clamped
//! to the pad) until the net force along the local up turns positive — i.e. thrust
//! exceeds weight — at which point it **releases** into pure active physics
//! ([`ActiveBody::integrate_wrench`], WI 515; the thrust comes from
//! [`crate::propulsion`], WI 531). Because a held craft is exactly still, this is
//! kraken-proof by construction — no contact stiffness, no sub-step tuning.
//!
//! The support is unilateral: while held it provides whatever upward force is
//! needed (implicitly), but the instant it would have to *pull down* (net force
//! already upward) it releases. A [`LaunchPad`] also surfaces a launch-stability
//! diagnostic (lift-off acceleration, max surface penetration). Headless.

use crate::active::ActiveBody;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// A launch pad holding a craft at the surface until thrust overcomes weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LaunchPad {
    /// Pad radius from the body centre (the craft's CoM rests here; altitude 0), m.
    pub surface_radius: f64,
    /// Whether the craft has lifted off (one-way: a released craft is not re-held).
    pub released: bool,
    /// Net upward acceleration recorded at the release instant, m/s² (diagnostic).
    pub liftoff_acceleration: f64,
    /// Worst sub-pad penetration ever seen, m — the kraken detector (≈ 0 expected).
    pub max_penetration: f64,
}

impl LaunchPad {
    /// A pad holding a craft at rest at `surface_radius`.
    pub fn resting(surface_radius: f64) -> Self {
        Self {
            surface_radius,
            released: false,
            liftoff_acceleration: 0.0,
            max_penetration: 0.0,
        }
    }

    /// Advance one step under the net world-frame wrench `(force, torque)` the caller
    /// assembles (gravity + thrust + drag). While held, the unilateral support keeps
    /// the craft at exact rest and clamps it to the pad (never pulling it down). When
    /// the net force along local up turns positive, release and integrate freely
    /// thereafter. Updates the launch-stability diagnostic.
    pub fn step(&mut self, body: &mut ActiveBody, force: DVec3, torque: DVec3, dt: f64) {
        let r = body.position.length();
        let up = if r > 0.0 { body.position / r } else { DVec3::Y };

        if !self.released {
            let net_up = force.dot(up);
            if net_up > 0.0 && body.mass > 0.0 {
                // Thrust overcomes weight: release into active physics.
                self.released = true;
                self.liftoff_acceleration = net_up / body.mass;
            } else {
                // Held: exact rest, clamped to the pad (no integration → no jitter,
                // no penetration). The support force is implicit and non-negative.
                body.velocity = DVec3::ZERO;
                let altitude = r - self.surface_radius;
                if altitude < 0.0 {
                    self.max_penetration = self.max_penetration.max(-altitude);
                    body.position = up * self.surface_radius;
                }
                return;
            }
        }

        body.integrate_wrench(force, torque, dt);
        let altitude = body.position.length() - self.surface_radius;
        if altitude < 0.0 {
            self.max_penetration = self.max_penetration.max(-altitude);
        }
    }

    /// Altitude of the craft above the pad, m.
    pub fn altitude(&self, body: &ActiveBody) -> f64 {
        body.position.length() - self.surface_radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propulsion::{Engine, EngineCommand, Propulsion};
    use crate::resource::{Reservoir, ReservoirId, ResourceGraph, ResourceType};
    use glam::DMat3;

    const R: f64 = 1_000.0; // arbitrary pad radius
    const G: f64 = 9.81;

    fn body_on_pad(mass: f64) -> ActiveBody {
        ActiveBody::new(DVec3::new(0.0, R, 0.0), DVec3::ZERO, mass, DMat3::IDENTITY)
    }

    /// Weight as a downward (−up) world force.
    fn gravity(mass: f64) -> DVec3 {
        DVec3::new(0.0, -mass * G, 0.0)
    }

    #[test]
    fn rests_under_gravity_without_jitter_or_penetration() {
        let mass = 1_000.0;
        let mut pad = LaunchPad::resting(R);
        let mut body = body_on_pad(mass);
        for _ in 0..1_000 {
            pad.step(&mut body, gravity(mass), DVec3::ZERO, 0.01);
        }
        assert!(!pad.released, "no lift-off under gravity alone");
        assert_eq!(body.velocity, DVec3::ZERO, "held at exact rest (no jitter)");
        assert!(
            (body.position.length() - R).abs() < 1e-9,
            "stays on the pad"
        );
        assert!(pad.max_penetration < 1e-9, "no surface penetration");
    }

    #[test]
    fn does_not_lift_off_below_weight() {
        let mass = 1_000.0;
        let weight = mass * G;
        let mut pad = LaunchPad::resting(R);
        let mut body = body_on_pad(mass);
        // Thrust at half weight → net force still downward.
        let force = gravity(mass) + DVec3::new(0.0, 0.5 * weight, 0.0);
        for _ in 0..500 {
            pad.step(&mut body, force, DVec3::ZERO, 0.01);
        }
        assert!(!pad.released, "sub-weight thrust does not lift off");
        assert_eq!(body.velocity, DVec3::ZERO);
        assert!(pad.max_penetration < 1e-9);
    }

    #[test]
    fn lifts_off_and_ascends_when_thrust_exceeds_weight() {
        let mass = 1_000.0;
        let weight = mass * G;
        let mut pad = LaunchPad::resting(R);
        let mut body = body_on_pad(mass);
        // Thrust at 1.5× weight → net upward 0.5·weight.
        let force = gravity(mass) + DVec3::new(0.0, 1.5 * weight, 0.0);
        pad.step(&mut body, force, DVec3::ZERO, 0.01);
        assert!(pad.released, "lifts off when thrust > weight");
        // Net upward acceleration ≈ 0.5·g.
        assert!(
            (pad.liftoff_acceleration - 0.5 * G).abs() < 1e-6,
            "lift-off accel {} ≈ {}",
            pad.liftoff_acceleration,
            0.5 * G
        );
        let alt0 = pad.altitude(&body);
        for _ in 0..200 {
            pad.step(&mut body, force, DVec3::ZERO, 0.01);
        }
        assert!(body.velocity.y > 0.0, "ascending");
        assert!(pad.altitude(&body) > alt0, "altitude climbing");
        assert!(pad.max_penetration < 1e-9, "never penetrates");
        assert!(body.position.is_finite() && body.velocity.is_finite());
    }

    #[test]
    fn diagnostic_bounded_across_the_throttle_range() {
        let mass = 1_000.0;
        let weight = mass * G;
        for k in 0..=10 {
            let thrust = (k as f64 / 10.0) * 2.0 * weight; // 0 .. 2×weight
            let mut pad = LaunchPad::resting(R);
            let mut body = body_on_pad(mass);
            let force = gravity(mass) + DVec3::new(0.0, thrust, 0.0);
            for _ in 0..500 {
                pad.step(&mut body, force, DVec3::ZERO, 0.01);
            }
            assert!(
                pad.max_penetration < 1e-9,
                "no penetration at thrust {thrust}"
            );
            assert!(pad.liftoff_acceleration.is_finite());
            assert!(body.position.is_finite() && body.velocity.is_finite());
            // Releases iff thrust strictly exceeds weight.
            assert_eq!(
                pad.released,
                thrust > weight,
                "release matches thrust>weight"
            );
        }
    }

    #[test]
    fn full_chain_engine_throttle_up_lifts_off() {
        // A craft + engine on the pad: throttle up → propellant burns, thrust builds,
        // it lifts off and climbs (WI 531 propulsion through the WI 532 pad).
        const PROP: ResourceType = ResourceType(0);
        let dry = 1_000.0;
        let mut prop = Propulsion {
            graph: ResourceGraph {
                reservoirs: vec![Reservoir::new(PROP, 500.0, 500.0)],
                ..Default::default()
            },
            tank_mounts: vec![DVec3::ZERO],
            engines: vec![Engine {
                tank: ReservoirId(0),
                exhaust_velocity: 3_000.0,
                max_mass_flow: 5.0, // max thrust 15 000 N > weight ≈ 14 715 N (wet)
                mount: DVec3::ZERO,
                axis: DVec3::Y, // thrust up
                max_gimbal: 0.0,
            }],
            commands: vec![EngineCommand::default()],
        };
        let mut pad = LaunchPad::resting(R);
        let wet0 = prop.wet_mass(dry, DVec3::ZERO).mass;
        let mut body = body_on_pad(wet0);

        // Idle (zero throttle): stays on the pad.
        for _ in 0..50 {
            let g = gravity(body.mass);
            let (t, tq) = prop.thrust_step(body.orientation, DVec3::ZERO, 0.01);
            pad.step(&mut body, g + t, tq, 0.01);
        }
        assert!(!pad.released, "idle: held on the pad");

        // Throttle up → lift off.
        prop.apply_command(&crate::command::Command::SetThrottle(1.0));
        for _ in 0..500 {
            body.mass = prop.wet_mass(dry, DVec3::ZERO).mass;
            let g = gravity(body.mass);
            let (t, tq) = prop.thrust_step(body.orientation, DVec3::ZERO, 0.01);
            pad.step(&mut body, g + t, tq, 0.01);
        }
        assert!(pad.released, "throttle-up lifts off");
        assert!(pad.altitude(&body) > 0.0, "climbed off the pad");
        assert!(body.velocity.y > 0.0);
        assert!(prop.graph.reservoirs[0].amount < 500.0, "propellant burned");
        assert!(pad.max_penetration < 1e-9);
    }
}

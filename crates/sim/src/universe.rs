//! Multi-body universe and patched-conic sphere-of-influence transitions (WI 528).
//!
//! The universe is a **tree of bodies**: a root body at the origin of the root
//! frame, each non-root body on a conic ([`Orbit`]) about its parent. Each body
//! defines a body-centered **inertial** frame ([`FrameId`]). A craft on-rails is on
//! a conic about its current **primary** (the body whose sphere of influence it
//! occupies).
//!
//! Crossing an SOI boundary **re-bases** the craft's conic into the new primary's
//! frame: because the frames are inertial, the transform is a pure translation by
//! the relative state of the two body origins (composed up the tree by
//! [`Universe::body_state`]); the craft's physical (position, velocity) is expressed
//! in the new frame and a fresh conic is fitted through it ([`Orbit::from_state`],
//! now bound *or* hyperbolic, WI 528). Fitting *through* the same physical state
//! makes the transition continuous by construction; [`Universe::soi_transition`]
//! reports the residual as a [`SoiTransition::discontinuity`], the frame-transform
//! analogue of the WI 508 hand-off-discontinuity check. Headless.

use crate::frame::{FrameId, FrameTree, WorldPos, WorldVel};
use crate::orbit::Orbit;
use glam::{DVec2, DVec3};

/// Embeds a planar (2D, z = 0) coordinate into 3D world space.
fn embed(v: DVec2) -> DVec3 {
    DVec3::new(v.x, v.y, 0.0)
}

/// A celestial body: the origin of an inertial frame, with its gravity, size,
/// sphere-of-influence radius, and (for non-root bodies) its conic about a parent.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    /// This body's inertial frame.
    pub frame: FrameId,
    /// Gravitational parameter (μ = G·M), m³/s².
    pub mu: f64,
    /// Surface radius, metres.
    pub radius: f64,
    /// Sphere-of-influence radius, metres; `INFINITY` for the root body.
    pub soi_radius: f64,
    /// Parent body's frame, or `None` for the root.
    pub parent: Option<FrameId>,
    /// Conic about the parent (in the parent's frame); `None` for the root.
    pub orbit: Option<Orbit>,
}

impl Body {
    /// The root body: origin of the root frame, no parent, infinite SOI.
    pub fn root(frame: FrameId, mu: f64, radius: f64) -> Self {
        Self {
            frame,
            mu,
            radius,
            soi_radius: f64::INFINITY,
            parent: None,
            orbit: None,
        }
    }

    /// A child body orbiting `parent` on `orbit`, with its SOI radius from the
    /// patched-conic formula `a · (μ_body / μ_parent)^(2/5)` (`a` = the orbit's
    /// semi-major axis).
    pub fn child(
        frame: FrameId,
        mu: f64,
        radius: f64,
        parent: FrameId,
        parent_mu: f64,
        orbit: Orbit,
    ) -> Self {
        let soi_radius = orbit.semi_major_axis.abs() * (mu / parent_mu).powf(0.4);
        Self {
            frame,
            mu,
            radius,
            soi_radius,
            parent: Some(parent),
            orbit: Some(orbit),
        }
    }
}

/// The result of an on-rails SOI transition: the re-based conic in the new frame,
/// plus the continuity residual (root-frame position/velocity jump, ≈ 0).
#[derive(Clone, Copy, Debug)]
pub struct SoiTransition {
    /// The new primary's frame.
    pub new_frame: FrameId,
    /// The craft's conic re-based into the new frame.
    pub new_orbit: Orbit,
    /// The injected discontinuity (the larger of the root-frame position and
    /// velocity jumps); ≈ 0 for a clean re-base.
    pub discontinuity: f64,
}

/// A tree of bodies forming a patched-conic universe.
#[derive(Clone, Debug, Default)]
pub struct Universe {
    bodies: Vec<Body>,
}

impl Universe {
    /// An empty universe.
    pub fn new() -> Self {
        Self { bodies: Vec::new() }
    }

    /// Adds a body (builder style).
    pub fn with_body(mut self, body: Body) -> Self {
        self.bodies.push(body);
        self
    }

    /// The body owning `frame`, if any.
    pub fn body(&self, frame: FrameId) -> Option<&Body> {
        self.bodies.iter().find(|b| b.frame == frame)
    }

    /// A body's sphere-of-influence radius.
    pub fn soi_radius(&self, frame: FrameId) -> Option<f64> {
        self.body(frame).map(|b| b.soi_radius)
    }

    /// The (position, velocity) of `frame`'s origin in the **root** frame at time
    /// `t`, composed by summing the conics up the tree. `None` if `frame` is
    /// unknown.
    pub fn body_state(&self, frame: FrameId, t: f64) -> Option<(DVec3, DVec3)> {
        let body = self.body(frame)?;
        match (body.parent, body.orbit) {
            (Some(parent), Some(orbit)) => {
                let (pp, pv) = self.body_state(parent, t)?;
                let (op, ov) = orbit.position_velocity(t);
                Some((pp + embed(op), pv + embed(ov)))
            }
            // Root (or a malformed body without an orbit): the frame origin.
            _ => Some((DVec3::ZERO, DVec3::ZERO)),
        }
    }

    /// The craft's (position, velocity) in the **root** frame, given its conic
    /// about `frame`.
    fn craft_root_state(&self, frame: FrameId, orbit: &Orbit, t: f64) -> Option<(DVec3, DVec3)> {
        let (p, v) = orbit.position_velocity(t);
        let (op, ov) = self.body_state(frame, t)?;
        Some((op + embed(p), ov + embed(v)))
    }

    /// Re-bases a craft's conic from `from` into `to` at time `t`: transforms its
    /// (position, velocity) into the new frame and fits a fresh conic through it.
    /// `None` if a frame is unknown or the re-based state is parabolic-degenerate.
    fn rebase(&self, from: FrameId, to: FrameId, orbit: &Orbit, t: f64) -> Option<SoiTransition> {
        let (p, v) = orbit.position_velocity(t);
        let new_pos = WorldPos::new(from, embed(p)).transform_to(to, self, t)?;
        let new_vel = WorldVel::new(from, embed(v)).transform_to(to, self, t)?;
        let to_mu = self.body(to)?.mu;
        let new_orbit = Orbit::from_state(to_mu, new_pos.truncate(), new_vel.truncate(), t)?;

        let before = self.craft_root_state(from, orbit, t)?;
        let after = self.craft_root_state(to, &new_orbit, t)?;
        let discontinuity = (before.0 - after.0)
            .length()
            .max((before.1 - after.1).length());
        Some(SoiTransition {
            new_frame: to,
            new_orbit,
            discontinuity,
        })
    }

    /// If the craft (on `orbit` about `frame`) has crossed a sphere-of-influence
    /// boundary at time `t`, returns the re-based conic in the new frame plus the
    /// continuity residual. Checks an **exit** (distance from the current primary
    /// exceeds its SOI → re-base to the parent) and an **entry** (distance to a
    /// child body is within the child's SOI → re-base to the child). One transition
    /// per call; the caller applies it and re-checks. `None` if no crossing (or the
    /// re-based conic would be degenerate).
    pub fn soi_transition(&self, frame: FrameId, orbit: &Orbit, t: f64) -> Option<SoiTransition> {
        let body = self.body(frame)?;
        let craft_pos = orbit.position(t); // in `frame`
        let dist = craft_pos.length();

        // Exit: left the current SOI → parent frame.
        if dist > body.soi_radius {
            if let Some(parent) = body.parent {
                return self.rebase(frame, parent, orbit, t);
            }
        }
        // Entry: inside a child's SOI → child frame.
        let craft_world = WorldPos::new(frame, embed(craft_pos));
        for child in self.bodies.iter().filter(|b| b.parent == Some(frame)) {
            if let Some(rel) = craft_world.transform_to(child.frame, self, t) {
                if rel.length() < child.soi_radius {
                    return self.rebase(frame, child.frame, orbit, t);
                }
            }
        }
        None
    }
}

impl FrameTree for Universe {
    fn relative_state(&self, from: FrameId, reference: FrameId, t: f64) -> Option<(DVec3, DVec3)> {
        if from == reference {
            return Some((DVec3::ZERO, DVec3::ZERO));
        }
        let (fp, fv) = self.body_state(from, t)?;
        let (rp, rv) = self.body_state(reference, t)?;
        Some((fp - rp, fv - rv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLANET: FrameId = FrameId(0);
    const MOON: FrameId = FrameId(1);

    // A planet (root) with a moon on a circular orbit. Normalised-ish units (the
    // machinery is unit-agnostic): planet μ = 1, moon orbit radius 10, moon μ = 0.01.
    fn planet_moon() -> Universe {
        let moon_orbit = Orbit::from_state(
            1.0,
            DVec2::new(10.0, 0.0),
            DVec2::new(0.0, 0.316_227_766),
            0.0,
        )
        .unwrap(); // ~circular: v = sqrt(mu/r) = sqrt(0.1)
        Universe::new()
            .with_body(Body::root(PLANET, 1.0, 1.0))
            .with_body(Body::child(MOON, 0.01, 0.2, PLANET, 1.0, moon_orbit))
    }

    #[test]
    fn moon_soi_radius_matches_patched_conic_formula() {
        let u = planet_moon();
        // a·(μ_m/μ_p)^0.4 = 10·(0.01)^0.4 ≈ 1.585.
        let soi = u.soi_radius(MOON).unwrap();
        assert!((soi - 10.0 * 0.01_f64.powf(0.4)).abs() < 1e-9);
        assert!((soi - 1.585).abs() < 0.01, "soi = {soi}");
        assert_eq!(u.soi_radius(PLANET), Some(f64::INFINITY));
    }

    #[test]
    fn body_state_walks_the_tree() {
        let u = planet_moon();
        // Planet (root) origin is always the world origin.
        assert_eq!(u.body_state(PLANET, 0.0), Some((DVec3::ZERO, DVec3::ZERO)));
        // Moon starts at (10, 0) and moves; magnitude stays ~10 (circular).
        let (p0, _) = u.body_state(MOON, 0.0).unwrap();
        assert!((p0 - DVec3::new(10.0, 0.0, 0.0)).length() < 1e-6);
        let (pt, _) = u.body_state(MOON, 5.0).unwrap();
        assert!(
            (pt.length() - 10.0).abs() < 1e-6,
            "moon stays on its circle"
        );
    }

    #[test]
    fn transform_between_frames_via_universe() {
        let u = planet_moon();
        // A point at the moon's centre: ZERO in the moon frame, the moon's position
        // in the planet frame.
        let at_moon = WorldPos::new(MOON, DVec3::ZERO);
        let (moon_pos, _) = u.body_state(MOON, 3.0).unwrap();
        assert!((at_moon.transform_to(PLANET, &u, 3.0).unwrap() - moon_pos).length() < 1e-9);
    }

    // --- SOI transitions (the headline) ---

    #[test]
    fn craft_exits_moon_soi_into_a_bound_planet_orbit_continuously() {
        let u = planet_moon();
        // A craft in the moon's frame beyond the moon's SOI (dist 2.0 > ~1.585) and
        // slow relative to the moon → its planet-frame orbit is bound.
        let craft =
            Orbit::from_state(0.01, DVec2::new(2.0, 0.0), DVec2::new(0.0, 0.05), 0.0).unwrap();
        let trans = u
            .soi_transition(MOON, &craft, 0.0)
            .expect("craft should exit the moon SOI");
        assert_eq!(trans.new_frame, PLANET, "re-based into the parent frame");
        assert!(trans.new_orbit.is_bound(), "planet-frame orbit is bound");
        assert!(
            trans.discontinuity < 1e-9,
            "re-base continuous in the root frame: {}",
            trans.discontinuity
        );
    }

    #[test]
    fn craft_enters_moon_soi_on_a_hyperbolic_flyby_continuously() {
        let u = planet_moon();
        // A craft on a bound planet orbit that passes through the moon's SOI fast
        // relative to the moon → hyperbolic about the moon (a flyby / gravity assist).
        let craft =
            Orbit::from_state(1.0, DVec2::new(11.0, 0.0), DVec2::new(0.0, -0.3), 0.0).unwrap();
        assert!(craft.is_bound(), "approach is a bound planet orbit");
        let trans = u
            .soi_transition(PLANET, &craft, 0.0)
            .expect("craft should enter the moon SOI");
        assert_eq!(trans.new_frame, MOON, "re-based into the moon's frame");
        assert!(
            !trans.new_orbit.is_bound() && trans.new_orbit.eccentricity > 1.0,
            "the flyby is hyperbolic about the moon (e = {})",
            trans.new_orbit.eccentricity
        );
        assert!(
            trans.discontinuity < 1e-9,
            "re-base continuous in the root frame: {}",
            trans.discontinuity
        );
    }

    #[test]
    fn parabolic_degenerate_rebase_is_rejected() {
        let u = planet_moon();
        // Entering the moon SOI at exactly moon-escape speed (energy ≈ 0 about the
        // moon) → the re-based conic is parabolic-degenerate → no transition.
        // Derive the craft's planet-frame velocity from the moon's *actual* state so
        // the moon-relative velocity is exactly radial escape (energy ≈ 0).
        let (moon_p, moon_v) = u.body_state(MOON, 0.0).unwrap();
        let r = 1.0;
        let v_esc = (2.0_f64 * 0.01 / r).sqrt(); // escape speed at r about the moon
        let craft = Orbit::from_state(
            1.0,
            moon_p.truncate() + DVec2::new(r, 0.0),
            moon_v.truncate() + DVec2::new(v_esc, 0.0),
            0.0,
        )
        .unwrap();
        assert!(
            u.soi_transition(PLANET, &craft, 0.0).is_none(),
            "a parabolic-degenerate re-base must not transition"
        );
    }

    #[test]
    fn no_transition_when_comfortably_within_the_primary() {
        let u = planet_moon();
        // A low planet orbit, far from the moon and well inside the (infinite)
        // planet SOI → no transition.
        let craft =
            Orbit::from_state(1.0, DVec2::new(2.0, 0.0), DVec2::new(0.0, 0.7), 0.0).unwrap();
        assert!(u.soi_transition(PLANET, &craft, 0.0).is_none());
    }
}

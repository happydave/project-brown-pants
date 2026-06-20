//! Two-body orbital mechanics for the on-rails gear.
//!
//! Analytic Kepler propagation of a planar (2D) bound orbit about a single fixed
//! central body at the origin. Position and velocity are closed-form functions of
//! time, so there is no integrator drift and time warp is exact at any step. This
//! is the simulation spine (WI 501); floating origin, 3D, and multiple bodies are
//! later work.

use glam::DVec2;
use std::f64::consts::{PI, TAU};

/// A bound (elliptical) two-body orbit in the plane, about a fixed central body
/// at the origin.
///
/// The orbit is stored as classical planar elements plus the central body's
/// gravitational parameter (one body in this toy, so it is carried here for
/// self-contained evaluation). `sense` is the motion direction: `+1.0` for
/// counter-clockwise (prograde), `-1.0` for clockwise (retrograde).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orbit {
    /// Gravitational parameter (μ = G·M) of the central body.
    pub mu: f64,
    /// Semi-major axis; `> 0` for bound orbits.
    pub semi_major_axis: f64,
    /// Eccentricity, `0 <= e < 1` for bound orbits.
    pub eccentricity: f64,
    /// Argument of periapsis: angle of the periapsis direction from +x, radians.
    pub arg_periapsis: f64,
    /// Mean anomaly at `epoch`, radians.
    pub mean_anomaly_at_epoch: f64,
    /// Time at which `mean_anomaly_at_epoch` holds.
    pub epoch: f64,
    /// Motion sense: `+1.0` counter-clockwise (prograde), `-1.0` clockwise.
    pub sense: f64,
}

impl Orbit {
    /// Derives the orbit from a position/velocity state at `epoch`.
    ///
    /// Returns `None` if the state is not bound (specific orbital energy ≥ 0,
    /// i.e. parabolic or hyperbolic) — out of scope for this toy.
    pub fn from_state(mu: f64, position: DVec2, velocity: DVec2, epoch: f64) -> Option<Orbit> {
        let r = position.length();
        let v2 = velocity.length_squared();
        if !(r.is_finite() && v2.is_finite()) || r == 0.0 {
            return None;
        }
        let energy = 0.5 * v2 - mu / r;
        if !energy.is_finite() {
            return None;
        }
        // Reject only the parabolic knife-edge (energy ≈ 0): a measure-zero,
        // ill-conditioned case. Bound (energy < 0 → a > 0) and hyperbolic
        // (energy > 0 → a < 0) conics are both represented.
        if energy.abs() <= 1e-9 * (mu / r) {
            return None;
        }
        let semi_major_axis = -mu / (2.0 * energy); // > 0 elliptical, < 0 hyperbolic

        // Specific angular momentum (z-component in 2D) gives the motion sense.
        let h = position.x * velocity.y - position.y * velocity.x;
        let sense = if h >= 0.0 { 1.0 } else { -1.0 };

        // Eccentricity vector: e_vec = ((v² - μ/r)·r - (r·v)·v) / μ.
        let rv = position.dot(velocity);
        let e_vec = ((v2 - mu / r) * position - rv * velocity) / mu;
        let eccentricity = e_vec.length();
        let arg_periapsis = if eccentricity > 1e-9 {
            e_vec.y.atan2(e_vec.x)
        } else {
            0.0 // near-circular: periapsis direction is undefined; pick +x
        };

        // Invert the perifocal position formula for the (eccentric / hyperbolic)
        // anomaly at epoch and reduce to the mean anomaly. Rotate into the
        // perifocal frame and read the anomaly off — consistent across motion sense.
        let pos_pf = rotate(position, -arg_periapsis);
        let mean_anomaly_at_epoch = if semi_major_axis > 0.0 {
            // Elliptical: M = E − e·sin E.
            let sqrt_1me2 = (1.0 - eccentricity * eccentricity).max(0.0).sqrt();
            let cos_e = (pos_pf.x / semi_major_axis + eccentricity).clamp(-1.0, 1.0);
            let sin_e = if sqrt_1me2 > 1e-12 {
                pos_pf.y / (semi_major_axis * sqrt_1me2)
            } else {
                0.0
            };
            let e0 = sin_e.atan2(cos_e);
            e0 - eccentricity * e0.sin()
        } else {
            // Hyperbolic: x_pf = a(cosh F − e), y_pf = −a·√(e²−1)·sinh F, so
            // sinh F = −y_pf / (a·√(e²−1)); F = asinh(sinh F); M = e·sinh F − F.
            let sqrt_e2m1 = (eccentricity * eccentricity - 1.0).max(0.0).sqrt();
            let sinh_f = if sqrt_e2m1 > 1e-12 {
                -pos_pf.y / (semi_major_axis * sqrt_e2m1)
            } else {
                0.0
            };
            let f0 = sinh_f.asinh();
            eccentricity * f0.sinh() - f0
        };

        Some(Orbit {
            mu,
            semi_major_axis,
            eccentricity,
            arg_periapsis,
            mean_anomaly_at_epoch,
            epoch,
            sense,
        })
    }

    /// Signed mean motion (rad/time): carries the motion sense. Uses `|a|³`, so it
    /// is well-defined for hyperbolic conics (a < 0) as well as elliptical.
    pub fn mean_motion(&self) -> f64 {
        self.sense * (self.mu / self.semi_major_axis.abs().powi(3)).sqrt()
    }

    /// Orbital period (always positive); `INFINITY` for a hyperbolic conic.
    pub fn period(&self) -> f64 {
        if self.semi_major_axis > 0.0 {
            TAU * (self.semi_major_axis.powi(3) / self.mu).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Whether the conic is bound (elliptical). Hyperbolic conics are not.
    pub fn is_bound(&self) -> bool {
        self.semi_major_axis > 0.0
    }

    /// Periapsis (closest) radius. Valid for both conics: `a(1−e)` is positive for
    /// elliptical (a>0, e<1) and for hyperbolic (a<0, e>1).
    pub fn periapsis_radius(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.eccentricity)
    }

    /// Apoapsis (farthest) radius; `INFINITY` for a hyperbolic conic (no apoapsis).
    pub fn apoapsis_radius(&self) -> f64 {
        if self.semi_major_axis > 0.0 {
            self.semi_major_axis * (1.0 + self.eccentricity)
        } else {
            f64::INFINITY
        }
    }

    /// Specific orbital energy (constant of motion); useful for verification.
    pub fn specific_energy(&self) -> f64 {
        -self.mu / (2.0 * self.semi_major_axis)
    }

    /// Specific angular momentum (signed; sign follows the motion sense).
    pub fn specific_angular_momentum(&self) -> f64 {
        self.sense
            * (self.mu * self.semi_major_axis * (1.0 - self.eccentricity * self.eccentricity))
                .max(0.0)
                .sqrt()
    }

    /// Position and velocity at time `t`, in world coordinates. Closed form: the
    /// same `t` always yields the same result, and arbitrarily large steps are
    /// exact rather than approximate.
    pub fn position_velocity(&self, t: f64) -> (DVec2, DVec2) {
        let e = self.eccentricity;
        let a = self.semi_major_axis;
        let mean_anomaly = self.mean_anomaly_at_epoch + self.mean_motion() * (t - self.epoch);

        let (p_pf, v_pf) = if a > 0.0 {
            // Elliptical: M = E − e·sin E.
            let ea = solve_kepler(mean_anomaly, e);
            let (sin_e, cos_e) = ea.sin_cos();
            let sqrt_1me2 = (1.0 - e * e).max(0.0).sqrt();
            let r = a * (1.0 - e * cos_e);
            let p = DVec2::new(a * (cos_e - e), a * sqrt_1me2 * sin_e);
            // Velocity carries the motion sense (derivative w.r.t. signed mean motion).
            let factor = self.sense * (self.mu * a).sqrt() / r;
            let v = DVec2::new(-factor * sin_e, factor * sqrt_1me2 * cos_e);
            (p, v)
        } else {
            // Hyperbolic: M = e·sinh F − F. Perifocal position and its time
            // derivative via the (signed) mean motion: dF/dt = ṅ / (e·cosh F − 1).
            let f = solve_kepler_hyperbolic(mean_anomaly, e);
            let (sh, ch) = (f.sinh(), f.cosh());
            let sqrt_e2m1 = (e * e - 1.0).max(0.0).sqrt();
            let p = DVec2::new(a * (ch - e), -a * sqrt_e2m1 * sh);
            let df_dt = self.mean_motion() / (e * ch - 1.0);
            let v = DVec2::new(a * sh * df_dt, -a * sqrt_e2m1 * ch * df_dt);
            (p, v)
        };

        (
            rotate(p_pf, self.arg_periapsis),
            rotate(v_pf, self.arg_periapsis),
        )
    }

    /// Position at time `t`, in world coordinates.
    pub fn position(&self, t: f64) -> DVec2 {
        self.position_velocity(t).0
    }

    /// The orbit resulting from an instantaneous delta-v (world frame) applied at
    /// time `t`. Returns `None` if the burn makes the orbit unbound.
    pub fn with_maneuver(&self, t: f64, delta_v: DVec2) -> Option<Orbit> {
        let (pos, vel) = self.position_velocity(t);
        Orbit::from_state(self.mu, pos, vel + delta_v, t)
    }

    /// Samples `samples` points around the orbit path (world coordinates) for
    /// rendering. Spacing is by eccentric anomaly; the shape is independent of
    /// motion sense.
    pub fn sample_path(&self, samples: usize) -> Vec<DVec2> {
        let e = self.eccentricity;
        let a = self.semi_major_axis;
        let sqrt_1me2 = (1.0 - e * e).max(0.0).sqrt();
        (0..samples)
            .map(|i| {
                let ea = TAU * (i as f64) / (samples as f64);
                let (s, c) = ea.sin_cos();
                rotate(
                    DVec2::new(a * (c - e), a * sqrt_1me2 * s),
                    self.arg_periapsis,
                )
            })
            .collect()
    }
}

/// Rotates `v` by `angle` radians (counter-clockwise).
fn rotate(v: DVec2, angle: f64) -> DVec2 {
    let (s, c) = angle.sin_cos();
    DVec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
}

/// Solves Kepler's equation `M = E - e·sin E` for the eccentric anomaly `E`.
///
/// The mean anomaly is reduced into `[-π, π]` first (the orbit is periodic in
/// `M`), which keeps Newton–Raphson fast and stable and prevents any secular
/// drift over many revolutions.
fn solve_kepler(mean_anomaly: f64, e: f64) -> f64 {
    let mut m = mean_anomaly.rem_euclid(TAU);
    if m > PI {
        m -= TAU;
    }
    // A good initial guess speeds convergence for high eccentricity.
    let mut ea = if e < 0.8 { m } else { PI.copysign(m) };
    for _ in 0..64 {
        let delta = (ea - e * ea.sin() - m) / (1.0 - e * ea.cos());
        ea -= delta;
        if delta.abs() < 1e-13 {
            break;
        }
    }
    ea
}

/// Solves the **hyperbolic** Kepler equation `M = e·sinh F − F` for the hyperbolic
/// anomaly `F` (e > 1). Newton–Raphson from the seed `F₀ = asinh(M/e)` (asymptotically
/// exact for large `|M|`), clamped to avoid `cosh` overflow. The hyperbolic mean
/// anomaly is **not** periodic, so `M` is used as-is (no reduction).
fn solve_kepler_hyperbolic(m: f64, e: f64) -> f64 {
    if m == 0.0 {
        return 0.0;
    }
    // asinh(M/e) is the large-|M| asymptote and a stable seed everywhere for e > 1.
    let mut f = (m / e).asinh().clamp(-100.0, 100.0);
    for _ in 0..100 {
        let (sh, ch) = (f.sinh(), f.cosh());
        let delta = (e * sh - f - m) / (e * ch - 1.0);
        f -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU: f64 = 1.0;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// A unit circular orbit returns to its start every period and never drifts,
    /// even after many revolutions evaluated in single large steps.
    #[test]
    fn circular_orbit_is_periodic_without_drift() {
        // r=1, v=sqrt(mu/r)=1, prograde (CCW).
        let orbit = Orbit::from_state(MU, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0), 0.0).unwrap();
        assert!(approx(orbit.eccentricity, 0.0, 1e-9), "should be circular");
        assert!(approx(orbit.semi_major_axis, 1.0, 1e-9));

        let p = orbit.period();
        let start = orbit.position(0.0);
        for k in [1, 2, 10, 1000] {
            let p_k = orbit.position(k as f64 * p);
            assert!(
                (p_k - start).length() < 1e-6,
                "drift after {k} revolutions: {:?}",
                p_k - start
            );
            // Circular: radius stays unit at all times.
            assert!(approx(p_k.length(), 1.0, 1e-9));
        }
    }

    /// Specific orbital energy is conserved across arbitrary time steps.
    #[test]
    fn energy_is_conserved() {
        // An eccentric orbit: periapsis speed above circular.
        let orbit = Orbit::from_state(MU, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.2), 0.0).unwrap();
        let expected = orbit.specific_energy();
        for i in 0..200 {
            let t = i as f64 * 0.137;
            let (pos, vel) = orbit.position_velocity(t);
            let energy = 0.5 * vel.length_squared() - MU / pos.length();
            assert!(
                approx(energy, expected, 1e-9),
                "energy drift at t={t}: {energy} vs {expected}"
            );
        }
    }

    /// A large warped step lands in exactly the same place as many small steps.
    #[test]
    fn large_step_matches_small_steps() {
        let orbit =
            Orbit::from_state(MU, DVec2::new(1.0, 0.2), DVec2::new(-0.1, 1.1), 0.0).unwrap();
        let target = 53.21;
        let big = orbit.position(target);
        // "Small steps" is just sampling the closed form; equality is exact-ish.
        let small = orbit.position(target);
        assert!((big - small).length() < 1e-12);
        // And a whole number of periods later is the same point.
        let later = orbit.position(target + 100.0 * orbit.period());
        assert!((later - big).length() < 1e-6, "drift over 100 periods");
    }

    /// A prograde burn at periapsis raises apoapsis and leaves periapsis put.
    #[test]
    fn prograde_burn_raises_apoapsis() {
        let circular =
            Orbit::from_state(MU, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0), 0.0).unwrap();
        let before_apo = circular.apoapsis_radius();
        // At t=0 the craft is at (1,0) moving +y; prograde delta-v is +y.
        let raised = circular.with_maneuver(0.0, DVec2::new(0.0, 0.1)).unwrap();
        assert!(
            raised.apoapsis_radius() > before_apo + 1e-3,
            "apoapsis should rise: {} -> {}",
            before_apo,
            raised.apoapsis_radius()
        );
        assert!(
            approx(raised.periapsis_radius(), 1.0, 1e-6),
            "burn point stays periapsis: {}",
            raised.periapsis_radius()
        );
    }

    /// Retrograde (clockwise) orbits propagate correctly: a quarter period from
    /// (1,0) going clockwise lands near (0,-1).
    #[test]
    fn retrograde_orbit_goes_clockwise() {
        let orbit =
            Orbit::from_state(MU, DVec2::new(1.0, 0.0), DVec2::new(0.0, -1.0), 0.0).unwrap();
        assert!(approx(orbit.sense, -1.0, 0.0), "should be retrograde");
        let quarter = orbit.position(orbit.period() / 4.0);
        assert!(
            (quarter - DVec2::new(0.0, -1.0)).length() < 1e-6,
            "quarter-period clockwise position: {:?}",
            quarter
        );
    }

    /// from_state followed by evaluation at the epoch reproduces the input state.
    #[test]
    fn state_round_trips() {
        let pos = DVec2::new(0.7, -0.4);
        let vel = DVec2::new(0.3, 0.9);
        let orbit = Orbit::from_state(MU, pos, vel, 5.0).unwrap();
        let (p, v) = orbit.position_velocity(5.0);
        assert!((p - pos).length() < 1e-9, "position round-trip: {:?}", p);
        assert!((v - vel).length() < 1e-9, "velocity round-trip: {:?}", v);
    }

    /// The parabolic knife-edge (energy ≈ 0, escape speed) is rejected as a
    /// measure-zero, ill-conditioned case rather than producing garbage.
    #[test]
    fn parabolic_state_is_rejected() {
        let escape = (2.0 * MU / 1.0).sqrt(); // exactly escape speed → energy ≈ 0
        assert!(
            Orbit::from_state(MU, DVec2::new(1.0, 0.0), DVec2::new(0.0, escape), 0.0).is_none()
        );
    }

    // --- WI 528: hyperbolic conics ---

    /// A state above escape speed is now a represented **hyperbolic** conic (e > 1,
    /// a < 0, infinite period/apoapsis), not rejected.
    #[test]
    fn hyperbolic_state_is_represented() {
        let escape = (2.0 * MU / 1.0).sqrt() + 0.5; // well above escape
        let orbit =
            Orbit::from_state(MU, DVec2::new(1.0, 0.0), DVec2::new(0.0, escape), 0.0).unwrap();
        assert!(orbit.eccentricity > 1.0, "e = {}", orbit.eccentricity);
        assert!(orbit.semi_major_axis < 0.0, "a = {}", orbit.semi_major_axis);
        assert!(!orbit.is_bound());
        assert_eq!(orbit.period(), f64::INFINITY);
        assert_eq!(orbit.apoapsis_radius(), f64::INFINITY);
        assert!(orbit.periapsis_radius() > 0.0);
    }

    /// `from_state` then `position_velocity(epoch)` recovers a hyperbolic input
    /// state — the strongest correctness check on the hyperbolic inversion.
    #[test]
    fn hyperbolic_state_round_trips() {
        let pos = DVec2::new(1.0, 0.3);
        let vel = DVec2::new(0.4, 1.9); // fast → hyperbolic
        let orbit = Orbit::from_state(MU, pos, vel, 2.0).unwrap();
        assert!(orbit.eccentricity > 1.0);
        let (p, v) = orbit.position_velocity(2.0);
        assert!((p - pos).length() < 1e-9, "position round-trip: {p:?}");
        assert!((v - vel).length() < 1e-9, "velocity round-trip: {v:?}");
    }

    /// Specific energy and angular momentum are conserved along a propagated
    /// hyperbolic trajectory (through periapsis), at relative tolerance.
    #[test]
    fn hyperbolic_conserves_energy_and_momentum() {
        // A non-radial hyperbolic state (clear angular momentum).
        let orbit =
            Orbit::from_state(MU, DVec2::new(-3.0, 2.0), DVec2::new(1.2, 0.3), 0.0).unwrap();
        assert!(orbit.eccentricity > 1.0, "e = {}", orbit.eccentricity);
        let e0 = orbit.specific_energy();
        let (p0, v0) = orbit.position_velocity(0.0);
        let h0 = p0.x * v0.y - p0.y * v0.x;
        for i in -50..=50 {
            let t = i as f64 * 0.2;
            let (p, v) = orbit.position_velocity(t);
            let energy = 0.5 * v.length_squared() - MU / p.length();
            let h = p.x * v.y - p.y * v.x;
            assert!(
                (energy - e0).abs() <= 1e-9 * e0.abs().max(1.0),
                "energy drift at t={t}: {energy} vs {e0}"
            );
            assert!(
                (h - h0).abs() <= 1e-9 * h0.abs().max(1.0),
                "angular-momentum drift at t={t}: {h} vs {h0}"
            );
        }
    }

    /// Hyperbolic propagation satisfies vis-viva: v² = μ(2/r − 1/a) (a < 0).
    #[test]
    fn hyperbolic_obeys_vis_viva() {
        let orbit = Orbit::from_state(MU, DVec2::new(2.0, 0.0), DVec2::new(0.2, 1.6), 0.0).unwrap();
        for i in 0..40 {
            let t = i as f64 * 0.1;
            let (p, v) = orbit.position_velocity(t);
            let vis_viva = MU * (2.0 / p.length() - 1.0 / orbit.semi_major_axis);
            assert!(
                (v.length_squared() - vis_viva).abs() <= 1e-9 * vis_viva.abs().max(1.0),
                "vis-viva mismatch at t={t}: {} vs {vis_viva}",
                v.length_squared()
            );
        }
    }

    /// WI 527: the propagator is unit-agnostic, so the conservation/periodicity
    /// invariants hold at **SI / planetary scale** too — verified with *relative*
    /// tolerances (absolute sub-millimetre equality is meaningless on 6.6e6 m).
    #[test]
    fn si_scale_orbit_conserves_energy_and_is_periodic() {
        const MU_SI: f64 = 3.986e14; // m³/s²
        let r0 = 6_560_000.0; // ~200 km altitude
        let orbit =
            Orbit::from_state(MU_SI, DVec2::new(r0, 0.0), DVec2::new(0.0, 8_200.0), 0.0).unwrap();
        let e0 = orbit.specific_energy();
        let period = orbit.period();
        // Specific energy conserved to a tight relative tolerance around the orbit.
        for i in 0..200 {
            let t = i as f64 * period / 200.0;
            let (p, v) = orbit.position_velocity(t);
            let e = 0.5 * v.length_squared() - MU_SI / p.length();
            assert!(
                (e - e0).abs() <= 1e-9 * e0.abs(),
                "relative energy drift at t={t}: {e} vs {e0}"
            );
        }
        // No secular drift after many revolutions (relative position tolerance).
        let start = orbit.position(0.0);
        let after = orbit.position(1000.0 * period);
        assert!(
            (after - start).length() <= 1e-6 * r0,
            "relative drift over 1000 revolutions"
        );
    }
}

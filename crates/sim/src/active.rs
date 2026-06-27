//! The active gear: numerical rigid-body integration (WI 515).
//!
//! The counterpart to the analytic on-rails gear (`orbit.rs`). A single focused
//! [`ActiveBody`] is integrated under the dominant attractor's point-mass gravity
//! at a fixed timestep, using a **symplectic** scheme (velocity Verlet) so energy
//! and angular momentum stay bounded — the property the design demands of the
//! propagator. Rotation is torque-free: the rotational state is the **angular
//! momentum** (world frame), which is constant when no torque acts, so its
//! conservation is exact by construction; angular velocity is derived from it and
//! the current orientation each step.
//!
//! This gear is built standalone (validated against the analytic Kepler orbit and
//! the WI 499 drift pattern); it does not switch a craft between gears — that
//! hand-off is WI 508. Mass and inertia come from the voxel craft (WI 505).

use crate::sim::SimClock;
use crate::voxel::MassProperties;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_time::prelude::*;
use glam::{DMat3, DQuat, DVec3};

/// Fixed integration timestep, in simulated seconds (normalised units).
pub const FIXED_DT: f64 = 1.0 / 64.0;
/// Maximum fixed sub-steps integrated per frame — the active-vehicle warp cap.
pub const MAX_SUBSTEPS: u32 = 8;

/// A numerically-integrated rigid body: the active gear's state.
#[derive(Component, Clone, Copy, Debug)]
pub struct ActiveBody {
    /// Position relative to the attractor, normalised units.
    pub position: DVec3,
    /// Velocity, normalised units per second.
    pub velocity: DVec3,
    /// Orientation (body → world).
    pub orientation: DQuat,
    /// Angular momentum in the world frame. Constant under no torque, so its
    /// conservation is exact; angular velocity is derived from it.
    pub angular_momentum: DVec3,
    /// Total mass.
    pub mass: f64,
    /// Inertia tensor about the centre of mass, body frame.
    pub inertia: DMat3,
    /// Cached body-frame inverse inertia (`ZERO` if the inertia is singular).
    inertia_inv: DMat3,
}

impl ActiveBody {
    /// A body at rest orientation with the given linear state and mass/inertia.
    pub fn new(position: DVec3, velocity: DVec3, mass: f64, inertia: DMat3) -> Self {
        let inv = inertia.inverse();
        let inertia_inv = if inv.is_finite() { inv } else { DMat3::ZERO };
        Self {
            position,
            velocity,
            orientation: DQuat::IDENTITY,
            angular_momentum: DVec3::ZERO,
            mass,
            inertia,
            inertia_inv,
        }
    }

    /// Builds a body from a voxel craft's derived mass properties (WI 505).
    pub fn from_mass_properties(position: DVec3, velocity: DVec3, mp: &MassProperties) -> Self {
        Self::new(position, velocity, mp.mass, mp.inertia)
    }

    /// Sets the spin by specifying angular velocity (world frame), storing the
    /// corresponding world-frame angular momentum `L = I_world · ω`.
    pub fn with_angular_velocity(mut self, omega: DVec3) -> Self {
        let r = DMat3::from_quat(self.orientation);
        let i_world = r * self.inertia * r.transpose();
        self.angular_momentum = i_world * omega;
        self
    }

    /// Angular velocity (world frame), derived from the angular momentum and the
    /// current orientation: `ω = R · I_body⁻¹ · Rᵀ · L`.
    pub fn angular_velocity(&self) -> DVec3 {
        let r = DMat3::from_quat(self.orientation);
        let inv_world = r * self.inertia_inv * r.transpose();
        inv_world * self.angular_momentum
    }

    /// Specific orbital energy of the current state (per unit mass).
    pub fn specific_energy(&self, mu: f64) -> f64 {
        0.5 * self.velocity.length_squared() - mu / self.position.length()
    }

    /// Specific orbital angular momentum of the current state (`r × v`).
    pub fn specific_angular_momentum(&self) -> DVec3 {
        self.position.cross(self.velocity)
    }

    /// Deviation of the current specific energy from a reference value (WI 499 style).
    pub fn energy_drift(&self, mu: f64, reference: f64) -> f64 {
        (self.specific_energy(mu) - reference).abs()
    }

    /// Deviation of the current specific angular momentum from a reference value.
    pub fn angular_momentum_drift(&self, reference: DVec3) -> f64 {
        (self.specific_angular_momentum() - reference).length()
    }

    /// Advances the body by one fixed step `dt` under gravity `mu` (velocity
    /// Verlet for translation; torque-free quaternion integration for rotation).
    pub fn step(&mut self, mu: f64, dt: f64) {
        // Velocity Verlet: half kick, drift, half kick.
        let a0 = gravity_accel(self.position, mu);
        self.velocity += 0.5 * dt * a0;
        self.position += dt * self.velocity;
        let a1 = gravity_accel(self.position, mu);
        self.velocity += 0.5 * dt * a1;

        // Torque-free rotation: angular momentum is constant; integrate the
        // orientation quaternion with the derived angular velocity.
        self.integrate_orientation(dt);
    }

    /// Advances the body by `dt` under an external **force and torque** (world
    /// frame), using semi-implicit Euler. This is the integrator for dissipative,
    /// state-dependent contact forces (wheels, WI 506) — distinct from the
    /// conservative-gravity velocity-Verlet [`ActiveBody::step`]. The caller adds
    /// gravity into `force`.
    pub fn integrate_wrench(&mut self, force: DVec3, torque: DVec3, dt: f64) {
        if self.mass > 0.0 {
            self.velocity += (force / self.mass) * dt;
        }
        self.position += self.velocity * dt;
        self.angular_momentum += torque * dt;
        self.integrate_orientation(dt);
    }

    /// Integrates the orientation quaternion by `dt` from the derived angular
    /// velocity (`q̇ = ½ ω ⊗ q`), renormalising.
    fn integrate_orientation(&mut self, dt: f64) {
        let omega = self.angular_velocity();
        if omega != DVec3::ZERO {
            let omega_q = DQuat::from_xyzw(omega.x, omega.y, omega.z, 0.0);
            let q = self.orientation;
            let q_dot = omega_q * q;
            self.orientation = DQuat::from_xyzw(
                q.x + 0.5 * dt * q_dot.x,
                q.y + 0.5 * dt * q_dot.y,
                q.z + 0.5 * dt * q_dot.z,
                q.w + 0.5 * dt * q_dot.w,
            )
            .normalize();
        }
    }

    /// Integrates `sim_seconds` of simulated time in fixed `dt` sub-steps, capped
    /// at `max_substeps` (the warp cap). Returns the number of sub-steps taken.
    pub fn advance(&mut self, mu: f64, sim_seconds: f64, dt: f64, max_substeps: u32) -> u32 {
        if dt <= 0.0 || !sim_seconds.is_finite() || sim_seconds <= 0.0 {
            return 0;
        }
        let want = (sim_seconds / dt).floor().max(0.0);
        let n = (want as u32).min(max_substeps);
        for _ in 0..n {
            self.step(mu, dt);
        }
        n
    }
}

/// Point-mass gravitational acceleration toward the attractor at the origin.
/// Returns zero at the singularity (`|r| = 0`) rather than producing a non-finite.
fn gravity_accel(position: DVec3, mu: f64) -> DVec3 {
    let r2 = position.length_squared();
    if r2 <= 0.0 || !r2.is_finite() {
        return DVec3::ZERO;
    }
    let r = r2.sqrt();
    -mu * position / (r2 * r)
}

/// The dominant attractor's gravitational parameter, for the active integrator.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Gravity {
    pub mu: f64,
}

/// Drives the active gear: advances every [`ActiveBody`] from the clock each frame,
/// capped sub-stepping. Wired by an app once a scene hosts an active body.
pub struct ActivePlugin {
    pub mu: f64,
}

impl Plugin for ActivePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Gravity { mu: self.mu })
            .add_systems(Update, advance_active_bodies);
    }
}

fn advance_active_bodies(
    time: Res<Time>,
    clock: Res<SimClock>,
    gravity: Res<Gravity>,
    mut bodies: Query<&mut ActiveBody>,
) {
    if clock.paused {
        return;
    }
    let sim_seconds = time.delta_secs_f64() * clock.warp;
    for mut body in &mut bodies {
        body.advance(gravity.mu, sim_seconds, FIXED_DT, MAX_SUBSTEPS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::Orbit;
    use crate::voxel::{Material, Thermal, Voxel, VoxelCraft};
    use glam::{DVec2, IVec3};
    use std::f64::consts::TAU;

    fn test_orbit() -> Orbit {
        Orbit::from_state(1.0, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.1), 0.0).unwrap()
    }

    fn body_on(orbit: &Orbit) -> ActiveBody {
        let (p, v) = orbit.position_velocity(0.0);
        ActiveBody::new(
            DVec3::new(p.x, p.y, 0.0),
            DVec3::new(v.x, v.y, 0.0),
            1.0,
            DMat3::IDENTITY,
        )
    }

    #[test]
    fn tracks_analytic_orbit_over_one_period() {
        let orbit = test_orbit();
        let mut body = body_on(&orbit);
        let mu = 1.0;
        let period = TAU * (orbit.semi_major_axis.powi(3) / mu).sqrt();
        let steps = (period / FIXED_DT) as u32;

        let mut t = 0.0;
        let mut max_err = 0.0_f64;
        for _ in 0..steps {
            body.step(mu, FIXED_DT);
            t += FIXED_DT;
            let (apos, _) = orbit.position_velocity(t);
            max_err = max_err.max((body.position.truncate() - apos).length());
        }
        assert!(
            max_err < 1e-2,
            "max position error over one orbit: {max_err}"
        );
    }

    #[test]
    fn energy_and_momentum_bounded_over_many_orbits() {
        let orbit = test_orbit();
        let mut body = body_on(&orbit);
        let mu = 1.0;
        let e0 = body.specific_energy(mu);
        let l0 = body.specific_angular_momentum();

        let mut max_e_drift = 0.0_f64;
        let mut max_l_drift = 0.0_f64;
        for _ in 0..20_000 {
            body.step(mu, FIXED_DT);
            max_e_drift = max_e_drift.max(body.energy_drift(mu, e0));
            max_l_drift = max_l_drift.max(body.angular_momentum_drift(l0));
        }
        // ~50 orbits: energy oscillates within a small bound, momentum ~exact.
        assert!(max_e_drift < 1e-2, "energy drift unbounded: {max_e_drift}");
        assert!(
            max_l_drift < 1e-9,
            "angular momentum not conserved: {max_l_drift}"
        );
    }

    #[test]
    fn torque_free_rotation_conserves_momentum_and_unit_quaternion() {
        // No gravity (mu = 0): isolate rotation. Asymmetric inertia, spinning.
        let inertia = DMat3::from_cols(
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 3.0, 0.0),
            DVec3::new(0.0, 0.0, 5.0),
        );
        let mut body = ActiveBody::new(DVec3::new(5.0, 0.0, 0.0), DVec3::ZERO, 1.0, inertia)
            .with_angular_velocity(DVec3::new(0.3, 0.1, 0.2));
        let l0 = body.angular_momentum;

        for _ in 0..2_000 {
            body.step(0.0, FIXED_DT);
        }
        // Angular momentum conserved (exact, by construction) and quaternion unit.
        assert!((body.angular_momentum - l0).length() < 1e-12);
        assert!((body.orientation.length() - 1.0).abs() < 1e-9);
        // The body actually rotated away from the identity.
        assert!(body.orientation.dot(DQuat::IDENTITY).abs() < 0.999);
    }

    #[test]
    fn integration_is_deterministic() {
        let orbit = test_orbit();
        let mut a = body_on(&orbit);
        let mut b = body_on(&orbit);
        for _ in 0..500 {
            a.step(1.0, FIXED_DT);
            b.step(1.0, FIXED_DT);
        }
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
    }

    /// WI 527: the symplectic integrator's conservation invariants hold at **SI /
    /// planetary scale** too (relative tolerances). A low-Earth-orbit circular state
    /// integrated for several minutes keeps energy bounded and momentum ~exact.
    #[test]
    fn si_scale_active_conserves_energy_and_momentum() {
        const MU_SI: f64 = 3.986e14;
        let r0 = 6_560_000.0;
        let v_circ = (MU_SI / r0).sqrt();
        let mut body = ActiveBody::new(
            DVec3::new(r0, 0.0, 0.0),
            DVec3::new(0.0, v_circ, 0.0),
            1.0,
            DMat3::IDENTITY,
        );
        let e0 = body.specific_energy(MU_SI);
        let l0 = body.specific_angular_momentum();
        let mut max_e_rel = 0.0_f64;
        let mut max_l_rel = 0.0_f64;
        // ~5 minutes of flight at the fixed step.
        for _ in 0..20_000 {
            body.step(MU_SI, FIXED_DT);
            max_e_rel = max_e_rel.max((body.specific_energy(MU_SI) - e0).abs() / e0.abs());
            max_l_rel =
                max_l_rel.max((body.specific_angular_momentum() - l0).length() / l0.length());
        }
        assert!(
            max_e_rel < 1e-6,
            "relative energy drift bounded at SI scale: {max_e_rel}"
        );
        assert!(
            max_l_rel < 1e-9,
            "angular momentum ~conserved at SI scale: {max_l_rel}"
        );
    }

    #[test]
    fn warp_is_capped() {
        let orbit = test_orbit();
        let mut body = body_on(&orbit);
        // A huge elapsed time integrates only up to the cap.
        let n = body.advance(1.0, 1_000.0, FIXED_DT, MAX_SUBSTEPS);
        assert_eq!(n, MAX_SUBSTEPS);
    }

    #[test]
    fn from_mass_properties_uses_voxel_inertia() {
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material {
                density: 1_000.0,
                strength: 1.0e9,
                thermal: Thermal::INERT,
            },
        });
        let mp = craft.mass_properties().unwrap();
        let body = ActiveBody::from_mass_properties(DVec3::ZERO, DVec3::ZERO, &mp);
        assert!((body.mass - mp.mass).abs() < 1e-9);
        assert!((body.inertia.col(0).x - mp.inertia.col(0).x).abs() < 1e-9);
    }

    #[test]
    fn gravity_singularity_and_degenerate_inertia_are_guarded() {
        // Gravity at the centre is zero, not non-finite.
        assert_eq!(gravity_accel(DVec3::ZERO, 1.0), DVec3::ZERO);
        // A body with singular (zero) inertia has zero angular velocity, no NaN.
        let body = ActiveBody::new(DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO, 5.0, DMat3::ZERO)
            .with_angular_velocity(DVec3::new(1.0, 1.0, 1.0));
        assert_eq!(body.angular_velocity(), DVec3::ZERO);
    }

    #[test]
    fn plugin_advances_without_panic() {
        let mut app = App::new();
        app.add_plugins(bevy_time::TimePlugin);
        app.insert_resource(SimClock::default());
        app.add_plugins(ActivePlugin { mu: 1.0 });
        let orbit = test_orbit();
        app.world_mut().spawn(body_on(&orbit));
        app.update();
        app.update();
    }
}

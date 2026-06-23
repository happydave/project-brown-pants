//! Contact response (WI 592, WI 593) — resolve collision manifolds into the f64 rigid body.
//!
//! A **penalty** response (the rover's proven philosophy, generalized to the hull): each
//! contact (WI 591) contributes a spring-damper **normal** force plus **regularized Coulomb
//! friction**, evaluated against the **relative** contact-point velocity and applied at the
//! contact point so it produces force + torque about the CoM through
//! `ActiveBody::integrate_wrench`. No new mutation path — collision is a physical wrench in the
//! active-flight step.
//!
//! WI 592 handled a single craft against **static ground**; WI 593 generalizes the same
//! per-contact penalty to **body↔body** ([`body_contact_wrench`]): both partners move, the
//! force is evaluated against their relative contact-point velocity, and the equal-and-opposite
//! reaction is applied to the partner (torque about each body's own CoM). The ground case is
//! the body↔body case with a static partner, so [`ground_contact_wrench`] now delegates to the
//! shared per-contact helper and its observable results are unchanged.

use crate::active::ActiveBody;
use crate::collision::{Bounds, CollisionShape};
use crate::collision_detect::{contacts, ContactPoint};
use glam::{DQuat, DVec3};

/// Penalty contact parameters.
#[derive(Clone, Copy, Debug)]
pub struct ContactParams {
    /// Normal spring stiffness, N/m (per contact point).
    pub normal_stiffness: f64,
    /// Normal damping, N·s/m (per contact point).
    pub normal_damping: f64,
    /// Coulomb friction coefficient (μ); the friction magnitude is capped at `μ·N`.
    pub friction: f64,
}

impl Default for ContactParams {
    fn default() -> Self {
        Self {
            normal_stiffness: 2.0e6,
            normal_damping: 8.0e4,
            friction: 0.8,
        }
    }
}

/// Tangential speed below which friction is regularized to avoid chatter at rest.
const FRICTION_V_EPS: f64 = 1.0e-3;

/// The penalty **force on body A** at a single contact, given the contact `normal` (pointing to
/// push A out), the penetration `depth`, and the **relative** contact-point velocity
/// `v_rel = v_A(point) − v_B(point)`. `None` when the normal magnitude is non-positive (the
/// contact is separating faster than the spring pushes — never adhesive). The reaction on body
/// B is the negation of this force.
fn penalty_contact_force(
    normal: DVec3,
    depth: f64,
    v_rel: DVec3,
    params: &ContactParams,
) -> Option<DVec3> {
    let vn = v_rel.dot(normal);
    // Spring (penetration) plus damping (resisting approach when `vn < 0`). The damping
    // magnitude is **clamped to the spring force** so it cannot exceed `k·depth`: at first
    // contact (`depth ≈ 0`) this kills the discontinuous "damping kick" that an explicit
    // integrator turns into spurious energy — the instability that appears for body↔body, where
    // the reduced mass halves the linear-damping stability margin. It is a no-op for any
    // already-stable contact (resting/landing, where `|c·vn| ≤ k·depth`), so the WI 592/597
    // craft↔ground behavior is preserved. Never adhesive (`fn_mag ≥ 0`).
    let spring = params.normal_stiffness * depth;
    let damping = (-params.normal_damping * vn).clamp(-spring, spring);
    let fn_mag = (spring + damping).max(0.0);
    if fn_mag <= 0.0 {
        return None;
    }
    let f_normal = fn_mag * normal;
    // Regularized Coulomb friction opposing the tangential relative velocity, capped at μ·N by
    // construction (magnitude → μ·N as |v_t| grows, →0 at rest).
    let v_t = v_rel - vn * normal;
    let f_t = -(params.friction * fn_mag) * v_t / (v_t.length() + FRICTION_V_EPS);
    Some(f_normal + f_t)
}

/// World pose `(position, orientation)` that places `body`'s lattice-frame collision shape so
/// the CoM at `dry_com` sits at `body.position`.
fn shape_pose(body: &ActiveBody, dry_com: DVec3) -> (DVec3, DQuat) {
    (body.position - body.orientation * dry_com, body.orientation)
}

/// Contact-point velocity of `body` at world `point`: `v + ω × (point − CoM)`.
fn point_velocity(body: &ActiveBody, point: DVec3) -> DVec3 {
    body.velocity + body.angular_velocity().cross(point - body.position)
}

/// Net contact **force and torque** (world frame, torque about the CoM) on a craft whose
/// collision `shape` is in the lattice frame with the CoM at `dry_com`, resting on / impacting
/// the static `ground`. The craft's `ActiveBody::position` is its CoM, so the shape is placed
/// with the `dry_com` offset and torque arms are taken about `body.position`. The ground is the
/// body↔body case with a static partner (`v_B = 0`).
pub fn ground_contact_wrench(
    body: &ActiveBody,
    shape: &CollisionShape,
    bounds: Option<Bounds>,
    dry_com: DVec3,
    ground: &CollisionShape,
    params: &ContactParams,
) -> (DVec3, DVec3) {
    let a_pose = shape_pose(body, dry_com);
    let cs = contacts(
        shape,
        a_pose,
        bounds,
        ground,
        (DVec3::ZERO, DQuat::IDENTITY),
        None,
        0.0,
    );
    let mut force = DVec3::ZERO;
    let mut torque = DVec3::ZERO;
    for c in cs {
        let v_rel = point_velocity(body, c.point); // static ground → v_B = 0
        if let Some(f) = penalty_contact_force(c.normal, c.depth, v_rel, params) {
            force += f;
            torque += (c.point - body.position).cross(f);
        }
    }
    (force, torque)
}

/// A net contact wrench on one body: `(force, torque-about-its-CoM)`, world frame.
pub type Wrench = (DVec3, DVec3);

/// Net contact wrenches for a **pair of movable bodies** (WI 593). Both `a` and `b` carry a
/// lattice-frame collision `shape` (a cuboid compound), broad-phase `bounds`, and a `dry_com`
/// offset; their `ActiveBody::position` is each body's CoM. Returns `(wrench_a, wrench_b)`.
///
/// Each contact's penalty force is evaluated against the **relative** contact-point velocity
/// and applied to A; the equal-and-opposite reaction is applied to B (torque about each body's
/// own CoM). Bounciness is damping-governed (no per-material restitution; WI 593 decision).
#[allow(clippy::too_many_arguments)]
pub fn body_contact_wrench(
    a: &ActiveBody,
    shape_a: &CollisionShape,
    bounds_a: Option<Bounds>,
    dry_com_a: DVec3,
    b: &ActiveBody,
    shape_b: &CollisionShape,
    bounds_b: Option<Bounds>,
    dry_com_b: DVec3,
    params: &ContactParams,
) -> (Wrench, Wrench) {
    let a_pose = shape_pose(a, dry_com_a);
    let b_pose = shape_pose(b, dry_com_b);
    // `contacts` returns normals that push body A out of B (the convention WI 591 documents).
    let cs: Vec<ContactPoint> = contacts(shape_a, a_pose, bounds_a, shape_b, b_pose, bounds_b, 0.0);
    let mut force_a = DVec3::ZERO;
    let mut torque_a = DVec3::ZERO;
    let mut force_b = DVec3::ZERO;
    let mut torque_b = DVec3::ZERO;
    for c in cs {
        let v_rel = point_velocity(a, c.point) - point_velocity(b, c.point);
        if let Some(f) = penalty_contact_force(c.normal, c.depth, v_rel, params) {
            force_a += f;
            torque_a += (c.point - a.position).cross(f);
            // Newton's third law: B gets the negation at the same world point.
            force_b -= f;
            torque_b += (c.point - b.position).cross(-f);
        }
    }
    ((force_a, torque_a), (force_b, torque_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::{craft_bounds, craft_collision_shape, ground_half_space};
    use crate::voxel::{Material, Voxel, VoxelCraft};
    use glam::IVec3;

    fn unit_craft() -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        c.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::COMPOSITE,
        });
        c
    }

    #[test]
    fn resting_craft_is_pushed_up_without_torque() {
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let mp = craft.mass_properties().unwrap();
        let ground = ground_half_space(0.0);
        let params = ContactParams::default();
        // CoM at y = 0.5 - d → the cell's 4 bottom corners penetrate by d.
        let d = 0.05;
        let body = ActiveBody::new(
            DVec3::new(0.0, 0.5 - d, 0.0),
            DVec3::ZERO,
            mp.mass,
            mp.inertia,
        );
        let (f, t) = ground_contact_wrench(
            &body,
            &shape,
            craft_bounds(&craft),
            mp.center_of_mass,
            &ground,
            &params,
        );
        assert!(f.y > 0.0, "pushed up");
        assert!(
            (f.y - 4.0 * params.normal_stiffness * d).abs() < 1e-6,
            "4 corners × k·d"
        );
        assert!(
            f.x.abs() < 1e-6 && f.z.abs() < 1e-6,
            "no lateral force at rest"
        );
        assert!(t.length() < 1e-6, "symmetric → no torque");
    }

    #[test]
    fn friction_opposes_lateral_sliding() {
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let mp = craft.mass_properties().unwrap();
        let ground = ground_half_space(0.0);
        let params = ContactParams::default();
        let d = 0.05;
        let body = ActiveBody::new(
            DVec3::new(0.0, 0.5 - d, 0.0),
            DVec3::new(2.0, 0.0, 0.0), // sliding +x
            mp.mass,
            mp.inertia,
        );
        let (f, _) = ground_contact_wrench(
            &body,
            &shape,
            craft_bounds(&craft),
            mp.center_of_mass,
            &ground,
            &params,
        );
        assert!(f.x < 0.0, "friction opposes +x sliding");
        // Capped at μ·N.
        let n = 4.0 * params.normal_stiffness * d;
        assert!(f.x.abs() <= params.friction * n + 1e-6, "friction ≤ μ·N");
    }

    #[test]
    fn clear_craft_has_no_contact_wrench() {
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let mp = craft.mass_properties().unwrap();
        let body = ActiveBody::new(DVec3::new(0.0, 10.0, 0.0), DVec3::ZERO, mp.mass, mp.inertia);
        let (f, t) = ground_contact_wrench(
            &body,
            &shape,
            craft_bounds(&craft),
            mp.center_of_mass,
            &ground_half_space(0.0),
            &ContactParams::default(),
        );
        assert_eq!(f, DVec3::ZERO);
        assert_eq!(t, DVec3::ZERO);
    }

    /// Net contact wrench on a body from gravity helper: `-g·m` along world Y.
    fn weight(body: &ActiveBody, g: f64) -> DVec3 {
        DVec3::new(0.0, -g * body.mass, 0.0)
    }

    /// Total kinetic energy (translational + rotational) of a body.
    fn kinetic_energy(body: &ActiveBody) -> f64 {
        0.5 * body.mass * body.velocity.length_squared()
            + 0.5 * body.angular_velocity().dot(body.angular_momentum)
    }

    #[test]
    fn contact_force_is_equal_and_opposite() {
        // Two overlapping unit craft → the force on B is the negation of the force on A.
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let bounds = craft_bounds(&craft);
        let mp = craft.mass_properties().unwrap();
        let params = ContactParams::default();
        let a = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, mp.mass, mp.inertia);
        let b = ActiveBody::new(DVec3::new(0.6, 0.0, 0.0), DVec3::ZERO, mp.mass, mp.inertia);
        let ((fa, _), (fb, _)) = body_contact_wrench(
            &a,
            &shape,
            bounds,
            mp.center_of_mass,
            &b,
            &shape,
            bounds,
            mp.center_of_mass,
            &params,
        );
        assert!(fa.length() > 0.0, "overlap produces a force");
        assert!((fa + fb).length() < 1e-9, "Newton's third law: f_b = -f_a");
        assert!(fa.x < 0.0, "A (left) is pushed in −x, away from B");
    }

    #[test]
    fn head_on_pair_separates_without_energy_injection() {
        // Two craft approach head-on in free space (no gravity); they collide and bounce apart,
        // and the pair's total kinetic energy does not increase (penalty is dissipative).
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let bounds = craft_bounds(&craft);
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let params = ContactParams::default();
        let dt = 0.004;
        let mut a = ActiveBody::new(
            DVec3::new(-2.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            mp.mass,
            mp.inertia,
        );
        let mut b = ActiveBody::new(
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(-2.0, 0.0, 0.0),
            mp.mass,
            mp.inertia,
        );
        let ke_before = kinetic_energy(&a) + kinetic_energy(&b);
        for _ in 0..6_000 {
            let (wa, wb) =
                body_contact_wrench(&a, &shape, bounds, com, &b, &shape, bounds, com, &params);
            a.integrate_wrench(wa.0, wa.1, dt);
            b.integrate_wrench(wb.0, wb.1, dt);
        }
        let ke_after = kinetic_energy(&a) + kinetic_energy(&b);
        // They bounced: each CoM velocity reversed sign.
        assert!(a.velocity.x < -0.01, "A reversed: {}", a.velocity.x);
        assert!(b.velocity.x > 0.01, "B reversed: {}", b.velocity.x);
        // Separated, no interpenetration runaway.
        assert!(
            (b.position.x - a.position.x) > 1.0,
            "separated: gap {}",
            b.position.x - a.position.x
        );
        // No energy injection.
        assert!(
            ke_after <= ke_before + 1e-6,
            "energy not injected: before={ke_before}, after={ke_after}"
        );
        assert!(a.position.is_finite() && b.position.is_finite());
    }

    #[test]
    fn stack_of_two_craft_rests_stably() {
        // B rests on the ground; A rests on B. Released slightly high, the stack settles and
        // stays stacked, finite, near-stationary — penalty stacking is stable.
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let bounds = craft_bounds(&craft);
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let ground = ground_half_space(0.0);
        let params = ContactParams::default();
        let g = 9.81;
        let dt = 0.004;
        let mut bottom =
            ActiveBody::new(DVec3::new(0.0, 1.0, 0.0), DVec3::ZERO, mp.mass, mp.inertia);
        let mut top = ActiveBody::new(DVec3::new(0.0, 2.0, 0.0), DVec3::ZERO, mp.mass, mp.inertia);
        for _ in 0..8_000 {
            // Top↔bottom (top is body A so its out-normal points up, off the bottom).
            let ((fa_t, ta_t), (fb_t, tb_t)) = body_contact_wrench(
                &top, &shape, bounds, com, &bottom, &shape, bounds, com, &params,
            );
            // Bottom↔ground.
            let (fg, tg) = ground_contact_wrench(&bottom, &shape, bounds, com, &ground, &params);
            top.integrate_wrench(fa_t + weight(&top, g), ta_t, dt);
            bottom.integrate_wrench(fb_t + fg + weight(&bottom, g), tb_t + tg, dt);
        }
        assert!(
            top.velocity.length() < 0.05,
            "top settled: v={}",
            top.velocity.length()
        );
        assert!(
            bottom.velocity.length() < 0.05,
            "bottom settled: v={}",
            bottom.velocity.length()
        );
        // Bottom resting just above the ground; top resting just above the bottom.
        assert!(
            bottom.position.y > 0.40 && bottom.position.y <= 0.5,
            "bottom: {}",
            bottom.position.y
        );
        assert!(
            top.position.y > 1.40 && top.position.y <= 1.5,
            "top: {}",
            top.position.y
        );
        let gap = top.position.y - bottom.position.y;
        assert!(gap > 0.95 && gap <= 1.0, "no interpenetration: gap={gap}");
        assert!(top.position.is_finite() && bottom.position.is_finite());
    }

    #[test]
    fn body_contact_is_deterministic() {
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let bounds = craft_bounds(&craft);
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let params = ContactParams::default();
        let dt = 0.004;
        let run = || {
            let mut a = ActiveBody::new(
                DVec3::new(-1.0, 0.1, 0.0),
                DVec3::new(1.5, 0.0, 0.0),
                mp.mass,
                mp.inertia,
            );
            let mut b =
                ActiveBody::new(DVec3::new(1.0, 0.0, 0.0), DVec3::ZERO, mp.mass, mp.inertia);
            for _ in 0..2_000 {
                let (wa, wb) =
                    body_contact_wrench(&a, &shape, bounds, com, &b, &shape, bounds, com, &params);
                a.integrate_wrench(wa.0, wa.1, dt);
                b.integrate_wrench(wb.0, wb.1, dt);
            }
            (a.position, a.velocity, b.position, b.velocity)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn craft_dropped_under_gravity_settles_to_rest() {
        // Isolate collision: constant gravity + the contact wrench through integrate_wrench.
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let bounds = craft_bounds(&craft);
        let mp = craft.mass_properties().unwrap();
        let ground = ground_half_space(0.0);
        let params = ContactParams::default();
        let g = 9.81;
        let dt = 0.004;
        let mut body = ActiveBody::new(DVec3::new(0.0, 2.0, 0.0), DVec3::ZERO, mp.mass, mp.inertia);
        for _ in 0..6_000 {
            let weight = DVec3::new(0.0, -g * body.mass, 0.0);
            let (cf, ct) =
                ground_contact_wrench(&body, &shape, bounds, mp.center_of_mass, &ground, &params);
            body.integrate_wrench(weight + cf, ct, dt);
        }
        // Settled: nearly stationary, resting just above the ground (CoM ≈ 0.5 − small sink).
        assert!(
            body.velocity.length() < 0.05,
            "came to rest: v={}",
            body.velocity.length()
        );
        assert!(
            body.position.y > 0.40 && body.position.y <= 0.5,
            "resting near the surface: y={}",
            body.position.y
        );
        assert!(body.position.is_finite());
    }
}

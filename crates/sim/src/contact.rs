//! Contact response (WI 592) — resolve collision manifolds into the f64 rigid body.
//!
//! A **penalty** response (the rover's proven philosophy, generalized to the hull): each
//! contact (WI 591) contributes a spring-damper **normal** force plus **regularized Coulomb
//! friction**, applied at the contact point so it produces force + torque about the CoM
//! through `ActiveBody::integrate_wrench`. No new mutation path — collision is a physical
//! wrench in the active-flight step.

use crate::active::ActiveBody;
use crate::collision::{Bounds, CollisionShape};
use crate::collision_detect::contacts;
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

/// Net contact **force and torque** (world frame, torque about the CoM) on a craft whose
/// collision `shape` is in the lattice frame with the CoM at `dry_com`, resting on / impacting
/// `ground`. The craft's `ActiveBody::position` is its CoM, so the shape is placed with the
/// `dry_com` offset and torque arms are taken about `body.position`.
pub fn ground_contact_wrench(
    body: &ActiveBody,
    shape: &CollisionShape,
    bounds: Option<Bounds>,
    dry_com: DVec3,
    ground: &CollisionShape,
    params: &ContactParams,
) -> (DVec3, DVec3) {
    // Place the lattice-frame shape so the CoM sits at body.position.
    let a_pose = (body.position - body.orientation * dry_com, body.orientation);
    let cs = contacts(
        shape,
        a_pose,
        bounds,
        ground,
        (DVec3::ZERO, DQuat::IDENTITY),
        None,
        0.0,
    );
    let omega = body.angular_velocity();
    let mut force = DVec3::ZERO;
    let mut torque = DVec3::ZERO;
    for c in cs {
        let r = c.point - body.position;
        let v_p = body.velocity + omega.cross(r);
        let vn = v_p.dot(c.normal);
        // Normal: spring (penetration) minus damping (approach); never adhesive.
        let fn_mag = (params.normal_stiffness * c.depth - params.normal_damping * vn).max(0.0);
        if fn_mag <= 0.0 {
            continue;
        }
        let f_normal = fn_mag * c.normal;
        // Regularized Coulomb friction opposing the tangential contact-point velocity, capped
        // at μ·N by construction (magnitude → μ·N as |v_t| grows, →0 at rest).
        let v_t = v_p - vn * c.normal;
        let f_t = -(params.friction * fn_mag) * v_t / (v_t.length() + FRICTION_V_EPS);
        let f = f_normal + f_t;
        force += f;
        torque += r.cross(f);
    }
    (force, torque)
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

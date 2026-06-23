//! Collision detection adapter (WI 591) — the `parry3d-f64` seam.
//!
//! The **only** module that names parry/glamx types. It converts our `CollisionShape`s + f64
//! poses (glam 0.30) into parry's types (glamx / glam 0.33, via `[f64;n]` array round-trips),
//! runs narrow-phase contact, and returns **contact manifolds in our types**. A
//! `CuboidCompound` (the craft) is the movable **body A**; it is queried **box-by-box** so a
//! craft's bottom cells each contribute a contact — a natural multi-point manifold for stable
//! resting, without parry's persistent-manifold pipeline. Broad phase culls by bounding
//! sphere first.
//!
//! Two contact paths: **box↔convex (box) via parry** (the general case, craft↔craft); and
//! **box-corner↔plane analytically** for flat ground (exact, no half-space-orientation
//! ambiguity, and it yields a contact per penetrating corner → 4-corner support for a
//! resting craft).
//!
//! Convention: a [`ContactPoint`]'s `normal` is the direction to push **body A out** of the
//! contact (the separating/MTV direction for A), with a non-negative penetration `depth`.

use crate::collision::{Bounds, BoxShape, CollisionShape};
use glam::{DQuat, DVec3};
use parry3d_f64::math::{Pose, Rot3, Vector};
use parry3d_f64::query;
use parry3d_f64::shape::Cuboid;

/// A single contact: a world point, a unit `normal` (the direction to push **body A** out of
/// penetration), and a non-negative penetration `depth`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactPoint {
    pub point: DVec3,
    pub normal: DVec3,
    pub depth: f64,
}

/// A body's pose: world position + orientation (body → world).
pub type Pose6 = (DVec3, DQuat);

/// The eight corner sign-combinations of a box.
const CORNER_SIGNS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];

fn to_v(v: DVec3) -> Vector {
    Vector::from_array(v.to_array())
}
fn from_v(v: Vector) -> DVec3 {
    DVec3::from_array(v.to_array())
}
/// World pose of a sub-box of body `body`: translate by the rotated box centre.
fn box_pose(body: Pose6, b: &BoxShape) -> Pose {
    let world_center = body.0 + body.1 * b.center;
    Pose::from_parts(to_v(world_center), Rot3::from_array(body.1.to_array()))
}

/// Contact manifold between two posed shapes. **Body A must be a `CuboidCompound`** (the
/// movable craft); body B is the ground half-space or another craft. `normal`s push A out.
/// `prediction` is the separation below which near contacts are reported (0 = touching/
/// penetrating only).
pub fn contacts(
    a: &CollisionShape,
    a_pose: Pose6,
    a_bounds: Option<Bounds>,
    b: &CollisionShape,
    b_pose: Pose6,
    b_bounds: Option<Bounds>,
    prediction: f64,
) -> Vec<ContactPoint> {
    // Broad phase: cull by bounding sphere when both bodies have bounds.
    if let (Some(ba), Some(bb)) = (a_bounds, b_bounds) {
        let ca = a_pose.0 + a_pose.1 * ba.sphere_center;
        let cb = b_pose.0 + b_pose.1 * bb.sphere_center;
        if (ca - cb).length() > ba.sphere_radius + bb.sphere_radius + prediction {
            return Vec::new();
        }
    }

    let CollisionShape::CuboidCompound(boxes_a) = a else {
        return Vec::new(); // A is expected to be the movable craft compound.
    };
    match b {
        CollisionShape::HalfSpace { normal, offset } => {
            boxes_vs_plane(boxes_a, a_pose, *normal, *offset)
        }
        CollisionShape::CuboidCompound(boxes_b) => {
            boxes_vs_boxes(boxes_a, a_pose, boxes_b, b_pose, prediction)
        }
    }
}

/// Each box corner below the ground plane (`p·normal < offset`) is a contact pushing the
/// craft along `normal` (up). Exact; gives a contact per penetrating corner.
fn boxes_vs_plane(
    boxes: &[BoxShape],
    pose: Pose6,
    normal: DVec3,
    offset: f64,
) -> Vec<ContactPoint> {
    let mut out = Vec::new();
    for b in boxes {
        for s in CORNER_SIGNS {
            let local = b.center + b.half_extents * DVec3::from_array(s);
            let corner = pose.0 + pose.1 * local;
            let sd = corner.dot(normal) - offset;
            if sd < 0.0 {
                out.push(ContactPoint {
                    point: corner,
                    normal,
                    depth: -sd,
                });
            }
        }
    }
    out
}

/// Box↔box via parry, per box pair. `normal2` (= push A out) is our contact normal.
fn boxes_vs_boxes(
    boxes_a: &[BoxShape],
    a_pose: Pose6,
    boxes_b: &[BoxShape],
    b_pose: Pose6,
    prediction: f64,
) -> Vec<ContactPoint> {
    let mut out = Vec::new();
    for ba in boxes_a {
        let ca = Cuboid::new(to_v(ba.half_extents));
        let pa = box_pose(a_pose, ba);
        for bb in boxes_b {
            let cb = Cuboid::new(to_v(bb.half_extents));
            let pb = box_pose(b_pose, bb);
            if let Ok(Some(c)) = query::contact(&pa, &ca, &pb, &cb, prediction) {
                if c.dist <= 0.0 {
                    out.push(ContactPoint {
                        point: from_v(c.point1),
                        normal: from_v(c.normal2), // points from B toward A → push A out
                        depth: -c.dist,
                    });
                }
            }
        }
    }
    out
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
            material: Material::ALUMINIUM,
        });
        c
    }

    #[test]
    fn box_penetrating_ground_reports_upward_corner_contacts() {
        // A unit craft lowered so its box (local y∈[0,1]) sinks below y=0.
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let ground = ground_half_space(0.0);
        // Body at y=-0.7 → box spans world y∈[-0.7, 0.3]; the 4 bottom corners are below 0.
        let pose = (DVec3::new(0.0, -0.7, 0.0), DQuat::IDENTITY);
        let cs = contacts(
            &shape,
            pose,
            craft_bounds(&craft),
            &ground,
            (DVec3::ZERO, DQuat::IDENTITY),
            None,
            0.0,
        );
        assert_eq!(cs.len(), 4, "four bottom corners penetrate");
        for c in &cs {
            assert!(c.normal.dot(DVec3::Y) > 0.9, "normal points up");
            assert!((c.depth - 0.7).abs() < 1e-9, "0.7 below the ground");
        }
    }

    #[test]
    fn box_clear_of_ground_has_no_contact() {
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let ground = ground_half_space(0.0);
        let pose = (DVec3::new(0.0, 10.0, 0.0), DQuat::IDENTITY);
        let cs = contacts(
            &shape,
            pose,
            craft_bounds(&craft),
            &ground,
            (DVec3::ZERO, DQuat::IDENTITY),
            None,
            0.0,
        );
        assert!(cs.is_empty());
    }

    #[test]
    fn broad_phase_culls_distant_bodies() {
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let bounds = craft_bounds(&craft);
        let cs = contacts(
            &shape,
            (DVec3::ZERO, DQuat::IDENTITY),
            bounds,
            &shape,
            (DVec3::new(1000.0, 0.0, 0.0), DQuat::IDENTITY),
            bounds,
            0.0,
        );
        assert!(cs.is_empty());
    }

    #[test]
    fn two_overlapping_craft_report_contact() {
        let craft = unit_craft();
        let shape = craft_collision_shape(&craft);
        let bounds = craft_bounds(&craft);
        // Boxes (each spans 0..1 locally) overlapped by 0.5 along x.
        let cs = contacts(
            &shape,
            (DVec3::ZERO, DQuat::IDENTITY),
            bounds,
            &shape,
            (DVec3::new(0.5, 0.0, 0.0), DQuat::IDENTITY),
            bounds,
            0.0,
        );
        assert!(!cs.is_empty(), "overlapping craft report a contact");
        assert!(cs[0].depth > 0.0);
    }
}

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
//! Two contact paths: **box↔box via parry's multi-point contact manifold** (the general case,
//! craft↔craft; SAT + face clipping yields up to 4 coplanar points for a face-face rest, which
//! is what makes stacking stable — a single deepest point cannot resist tipping); and
//! **box-corner↔plane analytically** for flat ground (exact, no half-space-orientation
//! ambiguity, and it yields a contact per penetrating corner → 4-corner support for a
//! resting craft).
//!
//! Convention: a [`ContactPoint`]'s `normal` is the direction to push **body A out** of the
//! contact (the separating/MTV direction for A), with a non-negative penetration `depth`.

use crate::collision::{Bounds, BoxShape, CollisionShape};
use glam::{DQuat, DVec3};
use parry3d_f64::math::{Pose, Rot3, Vector};
use parry3d_f64::query::details::contact_manifold_cuboid_cuboid;
use parry3d_f64::query::{
    ContactManifold, ContactManifoldsWorkspace, DefaultQueryDispatcher, PersistentQueryDispatcher,
};
use parry3d_f64::shape::{Cuboid, HeightField};
use parry3d_f64::utils::Array2;

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
        CollisionShape::Heightfield {
            heights,
            nx,
            nz,
            origin,
            size_x,
            size_z,
        } => boxes_vs_heightfield(
            boxes_a, a_pose, heights, *nx, *nz, *origin, *size_x, *size_z, prediction,
        ),
    }
}

/// A **pre-built** parry heightfield + its world placement (WI 636), so the expensive `HeightField`
/// construction (an O(n²) AABB tree) happens once and is reused across sub-steps while the rover sits on
/// the same local patch — instead of rebuilding it every `contacts` call. The headless `collision`
/// module stays parry-free; this parry-typed handle lives in the `collision_detect` seam and is cached
/// by the rover. `Clone`/`Debug` (parry's `HeightField` is both) so it can sit in `Rover`.
#[derive(Clone, Debug)]
pub struct BuiltHeightfield {
    hf: HeightField,
    origin: DVec3,
}

impl BuiltHeightfield {
    /// Build from a [`CollisionShape::Heightfield`]'s data (`None` for any other shape).
    pub fn from_shape(shape: &CollisionShape) -> Option<Self> {
        let CollisionShape::Heightfield {
            heights,
            nx,
            nz,
            origin,
            size_x,
            size_z,
        } = shape
        else {
            return None;
        };
        // parry `Array2` is column-major with rows = z (nz), cols = x (nx): `heights[ix*nz + iz]`.
        let arr = Array2::new(*nz, *nx, heights.clone());
        let hf = HeightField::new(arr, Vector::new(*size_x, 1.0, *size_z));
        Some(Self {
            hf,
            origin: *origin,
        })
    }

    /// Contact points for each box (body A, posed by `a_pose`) against this heightfield (body B, static).
    /// `local_n1` is the heightfield's outward normal toward the box — already the push-**A**-out
    /// direction (no negation, unlike box↔box where A is g1). Only **genuinely penetrating** points are
    /// kept (strict `dist < 0`): a heightfield manifold also emits zero-depth grazing/edge contacts at
    /// triangle boundaries whose diagonal normals would fabricate lateral support. A fresh workspace per
    /// box keeps the call stateless (the rover rebuilds the patch occasionally, not the per-box query).
    pub fn box_contacts(
        &self,
        boxes: &[BoxShape],
        a_pose: Pose6,
        prediction: f64,
    ) -> Vec<ContactPoint> {
        let hf_pose = Pose::from_parts(to_v(self.origin), Rot3::IDENTITY);
        let dispatcher = DefaultQueryDispatcher;
        let mut out = Vec::new();
        for ba in boxes {
            let cuboid = Cuboid::new(to_v(ba.half_extents));
            let box_world = box_pose(a_pose, ba);
            let pos12 = hf_pose.inv_mul(&box_world); // box pose in the heightfield's frame
            let mut manifolds: Vec<ContactManifold<(), ()>> = Vec::new();
            let mut workspace: Option<ContactManifoldsWorkspace> = None;
            if dispatcher
                .contact_manifolds(
                    &pos12,
                    &self.hf,
                    &cuboid,
                    prediction,
                    &mut manifolds,
                    &mut workspace,
                )
                .is_err()
            {
                continue;
            }
            for m in &manifolds {
                if m.points.is_empty() {
                    continue;
                }
                let normal = from_v(hf_pose.transform_vector(m.local_n1));
                for p in &m.points {
                    if p.dist < 0.0 {
                        let point = from_v(box_world.transform_point(p.local_p2));
                        out.push(ContactPoint {
                            point,
                            normal,
                            depth: -p.dist,
                        });
                    }
                }
            }
        }
        out
    }
}

/// Each craft box against a terrain **heightfield** (WI 636): the one-shot path used by `contacts`
/// (and the slice-(a) unit tests). Builds the heightfield then runs the per-box manifold. Hot callers
/// (the rover) cache a [`BuiltHeightfield`] instead, so the build is not repeated every sub-step.
#[allow(clippy::too_many_arguments)]
fn boxes_vs_heightfield(
    boxes_a: &[BoxShape],
    a_pose: Pose6,
    heights: &[f64],
    nx: usize,
    nz: usize,
    origin: DVec3,
    size_x: f64,
    size_z: f64,
    prediction: f64,
) -> Vec<ContactPoint> {
    let shape = CollisionShape::Heightfield {
        heights: heights.to_vec(),
        nx,
        nz,
        origin,
        size_x,
        size_z,
    };
    match BuiltHeightfield::from_shape(&shape) {
        Some(built) => built.box_contacts(boxes_a, a_pose, prediction),
        None => Vec::new(),
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

/// Box↔box via parry's **multi-point contact manifold**, per box pair. parry's `local_n1` is
/// box-A's outward normal (A→B); we negate it to push **A** out (B→A). Each manifold point that
/// is penetrating (`dist ≤ 0`) becomes a [`ContactPoint`] at the point on A's surface — so a
/// face-resting pair contributes its (up to 4) coplanar corners and rests/stacks without
/// tipping.
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
            let pos12 = pa.inv_mul(&pb); // pose of B in A's frame
            let mut manifold: ContactManifold<(), ()> = ContactManifold::new();
            contact_manifold_cuboid_cuboid(&pos12, &ca, &cb, prediction, &mut manifold);
            if manifold.points.is_empty() {
                continue;
            }
            // local_n1 is A's outward normal (A→B); the push-A-out direction is its negation.
            let normal = -from_v(pa.transform_vector(manifold.local_n1));
            for p in &manifold.points {
                if p.dist <= 0.0 {
                    out.push(ContactPoint {
                        point: from_v(pa.transform_point(p.local_p1)),
                        normal,
                        depth: -p.dist,
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
    use crate::collision::{
        craft_bounds, craft_collision_shape, ground_half_space, terrain_heightfield, BoxShape,
    };
    use crate::terrain::{Ramp, Terrain};
    use crate::voxel::{Material, Voxel, VoxelCraft};
    use glam::IVec3;

    /// A single axis-aligned box (half-extents `h`) at the origin of body A.
    fn box_a(h: f64) -> CollisionShape {
        CollisionShape::CuboidCompound(vec![BoxShape {
            center: DVec3::ZERO,
            half_extents: DVec3::splat(h),
        }])
    }

    /// A flat heightfield (all heights `y`) of `n×n` nodes over a `size×size` footprint at the origin.
    fn flat_heightfield(y: f64, size: f64, n: usize) -> CollisionShape {
        CollisionShape::Heightfield {
            heights: vec![y; n * n],
            nx: n,
            nz: n,
            origin: DVec3::ZERO,
            size_x: size,
            size_z: size,
        }
    }

    fn hf_contacts(a: &CollisionShape, a_pose: Pose6, hf: &CollisionShape) -> Vec<ContactPoint> {
        contacts(
            a,
            a_pose,
            None,
            hf,
            (DVec3::ZERO, DQuat::IDENTITY),
            None,
            0.0,
        )
    }

    #[test]
    fn box_on_flat_heightfield_pushes_up() {
        // A 0.5-half box lowered so its bottom is 0.2 below a flat (y=0) heightfield.
        let a = box_a(0.5);
        let hf = flat_heightfield(0.0, 4.0, 5);
        let pose = (DVec3::new(0.0, 0.3, 0.0), DQuat::IDENTITY);
        let cs = hf_contacts(&a, pose, &hf);
        assert!(!cs.is_empty(), "the penetrating box reports contacts");
        for c in &cs {
            assert!(
                c.normal.dot(DVec3::Y) > 0.9,
                "normal points up: {:?}",
                c.normal
            );
            assert!(
                c.depth > 0.0 && c.depth < 0.5,
                "sane penetration: {}",
                c.depth
            );
        }
        // Deepest contact ≈ 0.2 m (the bottom face below the surface).
        let max_pen = cs.iter().map(|c| c.depth).fold(0.0_f64, f64::max);
        assert!(
            (max_pen - 0.2).abs() < 0.05,
            "deepest ≈ 0.2 m, got {max_pen}"
        );
    }

    #[test]
    fn box_clear_of_heightfield_has_no_contact() {
        let a = box_a(0.5);
        let hf = flat_heightfield(0.0, 4.0, 5);
        let pose = (DVec3::new(0.0, 5.0, 0.0), DQuat::IDENTITY);
        assert!(hf_contacts(&a, pose, &hf).is_empty());
    }

    #[test]
    fn box_off_the_footprint_has_no_contact() {
        // A box well outside the patch footprint reads as no contact (never a clamped-edge artifact).
        let a = box_a(0.5);
        let hf = flat_heightfield(0.0, 4.0, 5);
        let pose = (DVec3::new(100.0, 0.3, 0.0), DQuat::IDENTITY);
        assert!(hf_contacts(&a, pose, &hf).is_empty());
    }

    #[test]
    fn box_on_sampled_incline_normal_tilts() {
        // A heightfield sampled from a planar incline (rising along +Z) → contact normals tilt back
        // (negative Z component), matching the analytic terrain normal.
        let terrain = Terrain {
            amplitude: 0.0, // a clean planar incline (no sinusoidal bumps)
            ramp: Some(Ramp {
                center_x: 0.0,
                half_width: 5.0,
                start_z: -5.0,
                run: 10.0,
                angle: 0.5,
            }),
            ..Terrain::default()
        };
        let hf = terrain_heightfield(&terrain, DVec3::new(0.0, 0.0, 0.0), 2.0, 2.0, 9, 9);
        let a = box_a(0.3);
        // Rest the box where the incline height at z=0 is ~ tan(0.5)*5 ≈ 2.73; lower it to touch.
        let h = terrain.height(0.0, 0.0);
        let pose = (DVec3::new(0.0, h + 0.25, 0.0), DQuat::IDENTITY);
        let cs = hf_contacts(&a, pose, &hf);
        assert!(!cs.is_empty(), "box rests on the incline");
        let mean_n = cs.iter().map(|c| c.normal).sum::<DVec3>().normalize();
        assert!(mean_n.z < -0.1, "normal tilts back along -Z: {:?}", mean_n);
        assert!(mean_n.y > 0.5, "still mostly upward: {:?}", mean_n);
    }

    #[test]
    fn box_against_sampled_cliff_sees_a_steep_face() {
        // A steep cliff: heights drop from 1.0 (x<0) to 0.0 (x>0) over one 0.1 m cell → an ~84° face.
        // A box pressed against the low side gets a contact with a large horizontal normal — the face
        // is contacted as a face, not sampled as a shallow ramp (the WI 636 fidelity point).
        let n = 11;
        let size = 1.0; // 0.1 m cells
        let mut heights = vec![0.0; n * n];
        for ix in 0..n {
            let x = (ix as f64 / (n - 1) as f64 - 0.5) * size; // node x
            for iz in 0..n {
                heights[ix * n + iz] = if x < 0.0 { 1.0 } else { 0.0 };
            }
        }
        let hf = CollisionShape::Heightfield {
            heights,
            nx: n,
            nz: n,
            origin: DVec3::ZERO,
            size_x: size,
            size_z: size,
        };
        // A small box straddling the cliff base on the low side, sunk into the face.
        let a = box_a(0.15);
        let pose = (DVec3::new(0.02, 0.1, 0.0), DQuat::IDENTITY);
        let cs = hf_contacts(&a, pose, &hf);
        assert!(!cs.is_empty(), "the box contacts the cliff region");
        let max_horiz = cs.iter().map(|c| c.normal.x.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_horiz > 0.3,
            "a steep face yields a horizontal-ish normal (not flat-ground +Y): max |n.x| = {max_horiz}"
        );
    }

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

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

use crate::collision::{Bounds, BoxShape, CollisionShape, ConvexShape};
use glam::{DQuat, DVec3};
use parry3d_f64::math::{Pose, Rot3, Vector};
use parry3d_f64::query::details::{contact_manifold_cuboid_cuboid, contact_manifold_pfm_pfm};
use parry3d_f64::query::ContactManifold;
use parry3d_f64::shape::{ConvexPolyhedron, Cuboid, PolygonalFeatureMap};

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

    let Some((boxes_a, convexes_a)) = compound_parts(a) else {
        return Vec::new(); // A is expected to be the movable craft compound.
    };
    match b {
        CollisionShape::HalfSpace { normal, offset } => {
            let mut out = boxes_vs_plane(boxes_a, a_pose, *normal, *offset);
            out.extend(convexes_vs_plane(convexes_a, a_pose, *normal, *offset));
            out
        }
        _ => {
            let Some((boxes_b, convexes_b)) = compound_parts(b) else {
                return Vec::new();
            };
            let mut out = boxes_vs_boxes(boxes_a, a_pose, boxes_b, b_pose, prediction);
            out.extend(convex_pairs(
                (boxes_a, convexes_a),
                a_pose,
                (boxes_b, convexes_b),
                b_pose,
                prediction,
            ));
            out
        }
    }
}

/// A compound's (boxes, convexes) part slices — a plain cuboid compound has no
/// convex parts (WI 837). `None` for a half-space.
fn compound_parts(shape: &CollisionShape) -> Option<(&[BoxShape], &[ConvexShape])> {
    match shape {
        CollisionShape::CuboidCompound(b) => Some((b, &[])),
        CollisionShape::Compound { boxes, convexes } => Some((boxes, convexes)),
        CollisionShape::HalfSpace { .. } => None,
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

/// Each convex-part hull vertex below the ground plane is a contact pushing the craft
/// along `normal` (the box corner path generalized — WI 837). Exact; a contact per
/// penetrating vertex, so a wedge resting on its hypotenuse gets its quad's support.
fn convexes_vs_plane(
    convexes: &[ConvexShape],
    pose: Pose6,
    normal: DVec3,
    offset: f64,
) -> Vec<ContactPoint> {
    let mut out = Vec::new();
    for c in convexes {
        for &p in &c.points {
            let vertex = pose.0 + pose.1 * p;
            let sd = vertex.dot(normal) - offset;
            if sd < 0.0 {
                out.push(ContactPoint {
                    point: vertex,
                    normal,
                    depth: -sd,
                });
            }
        }
    }
    out
}

/// A convex part's parry polyhedron. The part's points are already in the **body**
/// frame (cell offset baked in), so its parry pose is the body pose itself. `None`
/// only for degenerate input, which admitted catalog forms cannot produce
/// (positive volume is an admission invariant) — skipped defensively.
fn convex_poly(c: &ConvexShape) -> Option<ConvexPolyhedron> {
    let pts: Vec<Vector> = c.points.iter().map(|&p| to_v(p)).collect();
    ConvexPolyhedron::from_convex_hull(&pts)
}

/// World pose of a body itself (a convex part bakes its offset into its points).
fn body_pose(body: Pose6) -> Pose {
    Pose::from_parts(to_v(body.0), Rot3::from_array(body.1.to_array()))
}

/// One PFM↔PFM pair via parry's **face-clipped multi-point manifold** (WI 837:
/// the same manifold quality the cuboid path has — up to 4 coplanar points for a
/// face rest, which is what makes resting/stacking stable). Border radii 0; fresh
/// manifold per pair (the cuboid path's lifecycle); normals negated to push A out.
fn pfm_pair<S1, S2>(
    s1: &S1,
    p1: &Pose,
    s2: &S2,
    p2: &Pose,
    prediction: f64,
    out: &mut Vec<ContactPoint>,
) where
    S1: ?Sized + PolygonalFeatureMap,
    S2: ?Sized + PolygonalFeatureMap,
{
    let pos12 = p1.inv_mul(p2);
    let mut manifold: ContactManifold<(), ()> = ContactManifold::new();
    contact_manifold_pfm_pfm(
        &pos12,
        s1,
        0.0,
        None,
        s2,
        0.0,
        None,
        prediction,
        &mut manifold,
    );
    if manifold.points.is_empty() {
        return;
    }
    // local_n1 is A's outward normal (A→B); the push-A-out direction is its negation.
    let normal = -from_v(p1.transform_vector(manifold.local_n1));
    for p in &manifold.points {
        if p.dist <= 0.0 {
            out.push(ContactPoint {
                point: from_v(p1.transform_point(p.local_p1)),
                normal,
                depth: -p.dist,
            });
        }
    }
}

/// Every compound pair that involves a convex part (WI 837): convex↔box,
/// box↔convex, convex↔convex — all through the PFM manifold (`Cuboid` and
/// `ConvexPolyhedron` both implement `PolygonalFeatureMap`). Box↔box pairs stay
/// on the dedicated cuboid path (`boxes_vs_boxes`), keeping unshaped behaviour
/// bit-identical.
fn convex_pairs(
    a: (&[BoxShape], &[ConvexShape]),
    a_pose: Pose6,
    b: (&[BoxShape], &[ConvexShape]),
    b_pose: Pose6,
    prediction: f64,
) -> Vec<ContactPoint> {
    let mut out = Vec::new();
    if a.1.is_empty() && b.1.is_empty() {
        return out;
    }
    let polys_a: Vec<ConvexPolyhedron> = a.1.iter().filter_map(convex_poly).collect();
    let polys_b: Vec<ConvexPolyhedron> = b.1.iter().filter_map(convex_poly).collect();
    let pa_body = body_pose(a_pose);
    let pb_body = body_pose(b_pose);
    for ca in &polys_a {
        for bb in b.0 {
            let cb = Cuboid::new(to_v(bb.half_extents));
            pfm_pair(
                ca,
                &pa_body,
                &cb,
                &box_pose(b_pose, bb),
                prediction,
                &mut out,
            );
        }
        for cb in &polys_b {
            pfm_pair(ca, &pa_body, cb, &pb_body, prediction, &mut out);
        }
    }
    for ba in a.0 {
        let ca = Cuboid::new(to_v(ba.half_extents));
        let pa = box_pose(a_pose, ba);
        for cb in &polys_b {
            pfm_pair(&ca, &pa, cb, &pb_body, prediction, &mut out);
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
    fn every_form_hull_volume_matches_the_catalog_volume() {
        // WI 837 derived-vs-derived: the mesh vertex hull IS the form. A future
        // non-convex catalog form would make the hull overclaim volume and fail
        // here loudly, by name — the convexity guard the collision fidelity
        // rests on.
        use crate::shape::{constants, hull_vertices, FORMS};
        use parry3d_f64::shape::Shape;
        for form in FORMS {
            let c = constants(form);
            for &o in &c.distinct_orientations {
                let pts: Vec<Vector> = hull_vertices(form, o).iter().map(|&p| to_v(p)).collect();
                let poly = ConvexPolyhedron::from_convex_hull(&pts).expect("admitted form hulls");
                let volume = 1.0 / poly.mass_properties(1.0).inv_mass;
                assert!(
                    (volume - c.volume).abs() < 1e-9,
                    "{form:?} o{o}: hull {volume} vs catalog {}",
                    c.volume
                );
            }
        }
    }

    fn wedge_craft() -> VoxelCraft {
        use crate::shape::{FillMode, Form, ShapedCell};
        let mut c = unit_craft();
        c.set_shape(ShapedCell {
            cell: IVec3::ZERO,
            form: Form::Wedge,
            orientation: 0,
            fill: FillMode::Solid,
        });
        c
    }

    /// The body rotation that faces the wedge's hypotenuse (outward normal
    /// (0,1,−1)/√2) straight down: −135° about X. Under it the four hypotenuse-
    /// quad vertices sit level (height 0 from the body origin), the two real
    /// front vertices +1/√2 above, and the two *empty* cube corners 1/√2 BELOW
    /// the resting face — the phantom-corner geometry in one pose.
    fn hypotenuse_down() -> DQuat {
        DQuat::from_axis_angle(DVec3::X, (-135.0_f64).to_radians())
    }

    #[test]
    fn a_hypotenuse_down_wedge_rests_on_its_quad_not_a_phantom_corner() {
        // WI 837 AC (geometric half). Same pose, shaped vs unshaped craft:
        // the wedge touches at exactly its four hypotenuse vertices at the
        // posed 0.01 penetration; the cube reports corners 1/√2 deeper — the
        // contact the wedge no longer has.
        let ground = ground_half_space(0.0);
        let pose = (DVec3::new(0.0, -0.01, 0.0), hypotenuse_down());
        let shaped = wedge_craft();
        let cs = contacts(
            &craft_collision_shape(&shaped),
            pose,
            craft_bounds(&shaped),
            &ground,
            (DVec3::ZERO, DQuat::IDENTITY),
            None,
            0.0,
        );
        assert_eq!(cs.len(), 4, "the hypotenuse quad supports");
        for c in &cs {
            assert!((c.depth - 0.01).abs() < 1e-9, "flush rest: {}", c.depth);
            assert!(c.normal.dot(DVec3::Y) > 0.9);
        }
        let plain = unit_craft();
        let cb = contacts(
            &craft_collision_shape(&plain),
            pose,
            craft_bounds(&plain),
            &ground,
            (DVec3::ZERO, DQuat::IDENTITY),
            None,
            0.0,
        );
        let max_depth = cb.iter().map(|c| c.depth).fold(0.0, f64::max);
        assert!(
            max_depth > 0.7,
            "the cube's phantom corner digs 1/√2 in: {max_depth}"
        );
    }

    #[test]
    fn a_face_resting_convex_pair_gets_a_multi_point_manifold() {
        // WI 837: a cube craft face-resting on a wedge's full back (z = 1) face
        // through the PFM path yields the multi-point (face-clipped) manifold
        // that makes stacking stable — not a single deepest point.
        let cube = unit_craft();
        let wedge = wedge_craft();
        let cs = contacts(
            &craft_collision_shape(&cube),
            (DVec3::new(0.0, 0.0, 0.98), DQuat::IDENTITY), // 0.02 into the back face
            craft_bounds(&cube),
            &craft_collision_shape(&wedge),
            (DVec3::ZERO, DQuat::IDENTITY),
            craft_bounds(&wedge),
            0.0,
        );
        assert!(cs.len() >= 3, "face rest is multi-point: {}", cs.len());
        for c in &cs {
            assert!(
                c.normal.dot(DVec3::Z) > 0.9,
                "pushed out +Z: {:?}",
                c.normal
            );
            assert!(c.depth > 0.0 && c.depth < 0.05);
        }
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

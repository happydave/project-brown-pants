//! Collision shapes (WI 590) — the backend-agnostic geometry the collision system runs on.
//!
//! These are `sounding_sim`'s **own** f64/glam shape types, derived from existing data (the
//! voxel lattice, the ground convention). The detection adapter (WI 591) converts them into
//! the detection backend's types (`parry3d-f64`); keeping parry out of this module means the
//! shapes stay pure, headless, and unit-testable, and the backend is swappable.

use crate::voxel::{FacePanel, VoxelCraft, PANEL_FILL};
use glam::DVec3;

/// An axis-aligned box: centre + half-extents, in some frame (a craft's local frame here).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxShape {
    pub center: DVec3,
    pub half_extents: DVec3,
}

/// A convex collision part (WI 837): the hull vertices of a shaped cell's oriented
/// canonical form, in the body's local frame. Pure glam — the parry conversion
/// lives in the detection adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexShape {
    /// The hull vertex set (a form mesh's vertices, oriented, scaled, translated).
    pub points: Vec<DVec3>,
}

/// A collision shape in a body's local frame. The craft is a union of axis-aligned boxes;
/// flat ground is a half-space. (Heightfield / trimesh terrain are future shapes.)
#[derive(Clone, Debug, PartialEq)]
pub enum CollisionShape {
    /// A union of axis-aligned boxes (one per occupied cell plus one thin box per face
    /// panel — WI 828; greedy solid-box merging is a future optimization).
    CuboidCompound(Vec<BoxShape>),
    /// A **mixed** compound (WI 837): unshaped cells + panels as boxes, shaped cells
    /// as their oriented form hulls. Only produced for crafts with shape records —
    /// an unshaped craft stays a [`CollisionShape::CuboidCompound`] (the fast path),
    /// so a mixed compound with zero convex parts never occurs.
    Compound {
        boxes: Vec<BoxShape>,
        convexes: Vec<ConvexShape>,
    },
    /// A flat ground plane: points with `p·normal < offset` are below the surface
    /// (penetrating). `normal` is unit.
    HalfSpace { normal: DVec3, offset: f64 },
}

/// Broad-phase bounds for a body: a local AABB plus an **orientation-invariant** bounding
/// sphere (so broad phase can cull pairs by transforming only the sphere centre, not the
/// box extents).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub aabb_min: DVec3,
    pub aabb_max: DVec3,
    pub sphere_center: DVec3,
    pub sphere_radius: f64,
}

/// A face panel's thin axis-aligned collision box (WI 828): full face span in the
/// tangent axes, plate thickness along the normal, centred on the plate.
fn panel_box(craft: &VoxelCraft, p: &FacePanel) -> BoxShape {
    let s = craft.cell_size;
    let n = p.axis.unit().as_dvec3();
    BoxShape {
        center: craft.face_center(p),
        half_extents: DVec3::splat(0.5 * s) * (DVec3::ONE - n) + n * (0.5 * PANEL_FILL * s),
    }
}

/// The craft's collision shape: one solid cuboid per occupied cell **plus one thin box per
/// face panel** (WI 828 — so a shed plate lands on the pad instead of ghosting through it),
/// in the craft's local frame (the same frame `mass_properties`/the lattice use). Pure and
/// deterministic; voxel geometry is unchanged from pre-828. **WI 837:** a shaped cell
/// contributes its oriented form hull (the same canonical mesh that renders/weighs/seals;
/// a shell collides as its solid form — the outer surface) instead of its cell box; a
/// craft with no shape records keeps the exact pre-837 cuboid compound.
pub fn craft_collision_shape(craft: &VoxelCraft) -> CollisionShape {
    let s = craft.cell_size;
    let half = DVec3::splat(s * 0.5);
    if craft.shapes.is_empty() {
        // Fast path: no shaped cells — the exact pre-837 shape and variant.
        let boxes = craft
            .voxels
            .iter()
            .map(|v| BoxShape {
                center: (v.cell.as_dvec3() + DVec3::splat(0.5)) * s,
                half_extents: half,
            })
            .chain(craft.face_panels.iter().map(|p| panel_box(craft, p)))
            .collect();
        return CollisionShape::CuboidCompound(boxes);
    }
    let mut boxes = Vec::new();
    let mut convexes = Vec::new();
    for v in &craft.voxels {
        match craft.shape_at(v.cell) {
            // A `Cube` record hulls as its full box would — kept a convex part so
            // the fast-path/mixed fork stays the single `shapes.is_empty()` test.
            Some(sh) => convexes.push(ConvexShape {
                points: crate::shape::hull_vertices(sh.form, sh.orientation)
                    .into_iter()
                    .map(|p| (v.cell.as_dvec3() + p) * s)
                    .collect(),
            }),
            None => boxes.push(BoxShape {
                center: (v.cell.as_dvec3() + DVec3::splat(0.5)) * s,
                half_extents: half,
            }),
        }
    }
    boxes.extend(craft.face_panels.iter().map(|p| panel_box(craft, p)));
    CollisionShape::Compound { boxes, convexes }
}

/// The craft's broad-phase bounds (local AABB + bounding sphere), or `None` for an empty
/// craft. Face panels bound by their plate boxes (WI 828).
pub fn craft_bounds(craft: &VoxelCraft) -> Option<Bounds> {
    if craft.voxels.is_empty() && craft.face_panels.is_empty() {
        return None;
    }
    let s = craft.cell_size;
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for v in &craft.voxels {
        let lo = v.cell.as_dvec3() * s;
        min = min.min(lo);
        max = max.max(lo + DVec3::splat(s));
    }
    for p in &craft.face_panels {
        let b = panel_box(craft, p);
        min = min.min(b.center - b.half_extents);
        max = max.max(b.center + b.half_extents);
    }
    let center = 0.5 * (min + max);
    Some(Bounds {
        aabb_min: min,
        aabb_max: max,
        sphere_center: center,
        sphere_radius: 0.5 * (max - min).length(),
    })
}

/// The craft's bounding radius **about its centre of mass**: the farthest cell-corner distance
/// from the CoM, an orientation-invariant conservative sphere (so a rotated craft never reaches
/// past it). `None` for an empty craft. Used by the anti-tunnelling warp cap (WI 595) to bound
/// how close any part of the craft is to a surface from the CoM alone.
pub fn craft_bounding_radius(craft: &VoxelCraft) -> Option<f64> {
    let com = craft.mass_properties()?.center_of_mass;
    let s = craft.cell_size;
    let mut r2 = 0.0_f64;
    for v in &craft.voxels {
        let lo = v.cell.as_dvec3() * s;
        for cx in [0.0, s] {
            for cy in [0.0, s] {
                for cz in [0.0, s] {
                    let corner = lo + DVec3::new(cx, cy, cz);
                    r2 = r2.max((corner - com).length_squared());
                }
            }
        }
    }
    Some(r2.sqrt())
}

/// A flat-ground half-space at height `surface_offset` with an upward (+Y) normal.
pub fn ground_half_space(surface_offset: f64) -> CollisionShape {
    CollisionShape::HalfSpace {
        normal: DVec3::Y,
        offset: surface_offset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{Material, Voxel};
    use glam::IVec3;

    fn craft_from(cells: &[IVec3], cell_size: f64) -> VoxelCraft {
        let mut c = VoxelCraft::new(cell_size);
        for &cell in cells {
            c.voxels.push(Voxel {
                cell,
                material: Material::ALUMINIUM,
            });
        }
        c
    }

    fn boxes(shape: &CollisionShape) -> &[BoxShape] {
        match shape {
            CollisionShape::CuboidCompound(b) => b,
            _ => panic!("expected a cuboid compound"),
        }
    }

    #[test]
    fn single_cell_box_centre_and_extents() {
        let shape = craft_collision_shape(&craft_from(&[IVec3::ZERO], 2.0));
        let b = boxes(&shape);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].center, DVec3::splat(1.0)); // cell centre at half a 2 m cell
        assert_eq!(b[0].half_extents, DVec3::splat(1.0));
    }

    #[test]
    fn compound_covers_the_solid_volume() {
        // One box per cell; total box volume == occupied voxel volume (no gaps/overhang).
        let cells: Vec<_> = (0..2)
            .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| IVec3::new(x, y, z))))
            .collect();
        let craft = craft_from(&cells, 0.5);
        let shape = craft_collision_shape(&craft);
        let b = boxes(&shape);
        assert_eq!(b.len(), 8);
        let box_vol: f64 = b
            .iter()
            .map(|bx| 8.0 * bx.half_extents.x * bx.half_extents.y * bx.half_extents.z)
            .sum();
        assert!((box_vol - craft.occupied_volume()).abs() < 1e-9);
    }

    #[test]
    fn bounds_aabb_and_sphere_enclose_the_craft() {
        let craft = craft_from(&[IVec3::ZERO, IVec3::new(2, 0, 0)], 1.0);
        let bounds = craft_bounds(&craft).unwrap();
        assert_eq!(bounds.aabb_min, DVec3::ZERO);
        assert_eq!(bounds.aabb_max, DVec3::new(3.0, 1.0, 1.0)); // cells 0 and 2 occupied
                                                                // The sphere encloses every box corner.
        let shape = craft_collision_shape(&craft);
        for bx in boxes(&shape) {
            let far = bx.center + bx.half_extents;
            assert!((far - bounds.sphere_center).length() <= bounds.sphere_radius + 1e-9);
        }
    }

    #[test]
    fn empty_craft_has_no_bounds_and_empty_compound() {
        let craft = VoxelCraft::new(1.0);
        assert!(craft_bounds(&craft).is_none());
        assert!(boxes(&craft_collision_shape(&craft)).is_empty());
    }

    #[test]
    fn ground_half_space_is_upward_plane() {
        match ground_half_space(5.0) {
            CollisionShape::HalfSpace { normal, offset } => {
                assert_eq!(normal, DVec3::Y);
                assert_eq!(offset, 5.0);
            }
            _ => panic!("expected a half-space"),
        }
    }

    #[test]
    fn a_shaped_cell_contributes_its_form_hull_and_a_shell_matches_its_solid() {
        // WI 837: the wedge cell becomes its 6-vertex oriented hull (inside the
        // cell box, the two empty top-front cube corners absent — the phantom
        // corners are not in the geometry at all); unshaped neighbours stay
        // boxes; a shell twin produces the identical shape (solid-for-topology).
        use crate::shape::{FillMode, Form, ShapedCell};
        use glam::IVec3;
        let mut craft = craft_from(&[IVec3::ZERO, IVec3::new(1, 0, 0)], 2.0);
        craft.set_shape(ShapedCell {
            cell: IVec3::ZERO,
            form: Form::Wedge,
            orientation: 0,
            fill: FillMode::Solid,
        });
        let shape = craft_collision_shape(&craft);
        let CollisionShape::Compound { boxes, convexes } = &shape else {
            panic!("a shaped craft is a mixed compound");
        };
        assert_eq!(boxes.len(), 1, "the unshaped neighbour stays a box");
        assert_eq!(convexes.len(), 1);
        let pts = &convexes[0].points;
        assert_eq!(pts.len(), 6, "the wedge hull has six vertices");
        for p in pts {
            for i in 0..3 {
                assert!(
                    (-1e-12..=2.0 + 1e-12).contains(&p[i]),
                    "inside the cell box"
                );
            }
        }
        for empty in [DVec3::new(0.0, 2.0, 0.0), DVec3::new(2.0, 2.0, 0.0)] {
            assert!(
                pts.iter().all(|p| (*p - empty).length() > 1e-9),
                "no phantom corner at {empty:?}"
            );
        }
        let mut shell = craft.clone();
        shell.set_shape(ShapedCell {
            cell: IVec3::ZERO,
            form: Form::Wedge,
            orientation: 0,
            fill: FillMode::Shell,
        });
        assert_eq!(
            craft_collision_shape(&shell),
            shape,
            "a shell collides as its solid form"
        );
        // And the fast path: clearing the record restores the exact box compound.
        let mut plain = craft.clone();
        plain.clear_shape(IVec3::ZERO);
        assert!(matches!(
            craft_collision_shape(&plain),
            CollisionShape::CuboidCompound(b) if b.len() == 2
        ));
    }

    #[test]
    fn a_shed_plate_has_a_thin_collision_box_and_bounds() {
        // WI 828: a plate-only fragment collides as its thin plate — full face
        // span tangentially, plate thickness along the normal — so shed plates
        // land on the pad instead of ghosting through it.
        use crate::voxel::PANEL_FILL;
        let mut craft = VoxelCraft::new(2.0);
        craft.set_face_panel(IVec3::ZERO, IVec3::Y, Some(Material::ALUMINIUM));
        let b = boxes(&craft_collision_shape(&craft)).to_vec();
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].center, DVec3::new(1.0, 2.0, 1.0)); // the +Y face centre
        assert_eq!(
            b[0].half_extents,
            DVec3::new(1.0, 0.5 * PANEL_FILL * 2.0, 1.0) // tangent half-cells, thin normal
        );
        let bounds = craft_bounds(&craft).expect("a plate bounds itself");
        assert!(
            bounds.aabb_max.y - bounds.aabb_min.y < 0.5,
            "thin along the normal"
        );
        assert!((bounds.aabb_max.x - bounds.aabb_min.x - 2.0).abs() < 1e-9);
    }
}

//! Collision shapes (WI 590) — the backend-agnostic geometry the collision system runs on.
//!
//! These are `sounding_sim`'s **own** f64/glam shape types, derived from existing data (the
//! voxel lattice, the ground convention). The detection adapter (WI 591) converts them into
//! the detection backend's types (`parry3d-f64`); keeping parry out of this module means the
//! shapes stay pure, headless, and unit-testable, and the backend is swappable.

use crate::voxel::VoxelCraft;
use glam::DVec3;

/// An axis-aligned box: centre + half-extents, in some frame (a craft's local frame here).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxShape {
    pub center: DVec3,
    pub half_extents: DVec3,
}

/// A collision shape in a body's local frame. The craft is a union of axis-aligned boxes;
/// flat ground is a half-space. (Heightfield / trimesh terrain are future shapes.)
#[derive(Clone, Debug, PartialEq)]
pub enum CollisionShape {
    /// A union of axis-aligned boxes (one per occupied cell now; greedy solid-box merging is
    /// a future optimization).
    CuboidCompound(Vec<BoxShape>),
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

/// The craft's collision shape: one solid cuboid per occupied cell, in the craft's local
/// frame (the same frame `mass_properties`/the lattice use). Pure and deterministic.
pub fn craft_collision_shape(craft: &VoxelCraft) -> CollisionShape {
    let s = craft.cell_size;
    let half = DVec3::splat(s * 0.5);
    let boxes = craft
        .voxels
        .iter()
        .map(|v| BoxShape {
            center: (v.cell.as_dvec3() + DVec3::splat(0.5)) * s,
            half_extents: half,
        })
        .collect();
    CollisionShape::CuboidCompound(boxes)
}

/// The craft's broad-phase bounds (local AABB + bounding sphere), or `None` for an empty
/// craft.
pub fn craft_bounds(craft: &VoxelCraft) -> Option<Bounds> {
    if craft.voxels.is_empty() {
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
    let center = 0.5 * (min + max);
    Some(Bounds {
        aabb_min: min,
        aabb_max: max,
        sphere_center: center,
        sphere_radius: 0.5 * (max - min).length(),
    })
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
}

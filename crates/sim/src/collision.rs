//! Collision shapes (WI 590) — the backend-agnostic geometry the collision system runs on.
//!
//! These are `sounding_sim`'s **own** f64/glam shape types, derived from existing data (the
//! voxel lattice, the ground convention). The detection adapter (WI 591) converts them into
//! the detection backend's types (`parry3d-f64`); keeping parry out of this module means the
//! shapes stay pure, headless, and unit-testable, and the backend is swappable.

use crate::terrain::Terrain;
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
    /// A terrain patch as a regular height grid (WI 636): a static, body-B surface for true
    /// hull-vs-terrain manifolds at a sharp discontinuity (the ramp lip), where independent point
    /// sampling fails. Sampled from the analytic [`crate::terrain::Terrain`] over a small,
    /// rover-local footprint each step (see [`terrain_heightfield`]). The grid has `nx × nz` nodes;
    /// `heights` are **absolute world-Y** values in **column-major** order (`heights[ix*nz + iz]`,
    /// matching parry's `Array2` layout); `origin` is the world position of the footprint centre
    /// (its `y` is the surface datum, normally 0); `size_x`/`size_z` are the world footprint extents
    /// along X/Z. Node `(ix, iz)` sits at world
    /// `origin + ((ix/(nx-1) − 0.5)·size_x, 0, (iz/(nz-1) − 0.5)·size_z)` at height `heights[..]`.
    Heightfield {
        heights: Vec<f64>,
        nx: usize,
        nz: usize,
        origin: DVec3,
        size_x: f64,
        size_z: f64,
    },
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

/// Sample the analytic [`Terrain`] into a local [`CollisionShape::Heightfield`] patch (WI 636): an
/// `nx × nz`-node grid spanning `size_x × size_z` (world metres) centred on `center` (only `center`'s
/// X/Z are used; the patch datum `origin.y` is 0). Each node's height is an **exact** `Terrain::height`
/// sample, so the hull manifold and the wheel quarter-car (which still reads `Terrain` analytically)
/// never disagree about the ground. The footprint must be sized past the rover's reach by the caller so
/// no hull point is ever over un-sampled terrain; the resolution must be fine enough that a sharp
/// feature (the ramp lip) falls within one cell. `nx`, `nz` are clamped to ≥ 2.
pub fn terrain_heightfield(
    terrain: &Terrain,
    center: DVec3,
    size_x: f64,
    size_z: f64,
    nx: usize,
    nz: usize,
) -> CollisionShape {
    let nx = nx.max(2);
    let nz = nz.max(2);
    // Column-major (`heights[ix*nz + iz]`) to match parry's `Array2` layout (cols = x, rows = z).
    let mut heights = Vec::with_capacity(nx * nz);
    for ix in 0..nx {
        let lx = (ix as f64 / (nx - 1) as f64 - 0.5) * size_x;
        for iz in 0..nz {
            let lz = (iz as f64 / (nz - 1) as f64 - 0.5) * size_z;
            heights.push(terrain.height(center.x + lx, center.z + lz));
        }
    }
    CollisionShape::Heightfield {
        heights,
        nx,
        nz,
        origin: DVec3::new(center.x, 0.0, center.z),
        size_x,
        size_z,
    }
}

/// Sample the analytic [`Terrain`] into a **solid** local patch of cuboid *columns* (WI 636): one
/// upward box per grid cell, its top face at the cell's `Terrain::height` and extruded `depth` metres
/// downward, tiling an `nx × nz` grid over `size_x × size_z` (world) centred on `center`'s X/Z. Resolved
/// through the proven `boxes_vs_boxes` contact (a craft hull vs this compound), it is a true **solid**:
/// a resting hull cannot tunnel through it (unlike a thin one-sided heightfield), and a sharp feature
/// (the ramp-lip cliff) becomes the exposed **side face** of a tall column — a vertical face the hull
/// contacts with a horizontal normal, so there is no upward "toss". `depth` must exceed how far any hull
/// box could reach below the surface so the column's (never-contacted) bottom is irrelevant. Column tops
/// are flat per cell (a staircase); this is the *hull's* contact surface only — wheels still read the
/// smooth analytic terrain — and the hull contact is inert in normal driving.
pub fn terrain_columns(
    terrain: &Terrain,
    center: DVec3,
    size_x: f64,
    size_z: f64,
    nx: usize,
    nz: usize,
    depth: f64,
) -> CollisionShape {
    let nx = nx.max(2);
    let nz = nz.max(2);
    let cell_x = size_x / (nx - 1) as f64;
    let cell_z = size_z / (nz - 1) as f64;
    let half = DVec3::new(0.5 * cell_x, 0.5 * depth, 0.5 * cell_z);
    let mut boxes = Vec::with_capacity(nx * nz);
    for ix in 0..nx {
        let cx = center.x + (ix as f64 / (nx - 1) as f64 - 0.5) * size_x;
        for iz in 0..nz {
            let cz = center.z + (iz as f64 / (nz - 1) as f64 - 0.5) * size_z;
            let top = terrain.height(cx, cz);
            boxes.push(BoxShape {
                center: DVec3::new(cx, top - 0.5 * depth, cz),
                half_extents: half,
            });
        }
    }
    CollisionShape::CuboidCompound(boxes)
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

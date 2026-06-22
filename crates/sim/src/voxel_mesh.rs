//! Voxel-craft surface meshing (WI 582/583): turn a [`VoxelCraft`] lattice into
//! **engine-agnostic** triangle-mesh data so the windowed app can render the craft's
//! real geometry instead of a placeholder.
//!
//! Headless and rendering-free: this module emits plain `Vec`s of positions, normals,
//! texture coordinates, and indices — no Bevy render types — so the meshing algorithms
//! are unit-tested without a display, and the app owns the thin conversion to its mesh
//! type (+ tangents + material). Two skins share one exposed-face determination:
//! - **Blocky** ([`blocky_mesh`], WI 582) — a quad per exposed cell face (Stormworks
//!   style).
//! - **Greedy hull** (WI 583) — coplanar exposed faces merged into panels.
//!
//! The shared rule: a cell face is **exposed** only when the neighbouring cell along its
//! normal is empty; interior faces (between two occupied cells) are never emitted.

use crate::voxel::VoxelCraft;
use glam::IVec3;
use std::collections::HashSet;

/// Engine-agnostic triangle-mesh data for one skin of a craft. Parallel
/// `positions`/`normals`/`uvs` (one entry per vertex) plus a triangle-`indices` list.
/// The app converts this into its own mesh type and generates a tangent basis.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkinMesh {
    /// Vertex positions in the craft's local frame, metres (cell coordinates × cell size).
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex outward normals (flat per face for the blocky skin).
    pub normals: Vec<[f32; 3]>,
    /// Per-vertex texture coordinates.
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices into the vertex arrays (CCW front faces).
    pub indices: Vec<u32>,
}

impl SkinMesh {
    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of emitted quad faces (each face is two triangles / four vertices).
    pub fn face_count(&self) -> usize {
        self.positions.len() / 4
    }
}

/// The six axis-aligned cube faces: outward integer normal + the four corner offsets
/// (in `{0,1}³`) wound **counter-clockwise as seen from outside**, so the default
/// back-face culling shows them from the exterior. Winding verified per face by
/// cross-product (`(c1−c0)×(c2−c0)` equals the listed normal).
const FACES: [([i32; 3], [[i32; 3]; 4]); 6] = [
    ([1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
    ([-1, 0, 0], [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]),
    ([0, 1, 0], [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
    ([0, -1, 0], [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]]),
    ([0, 0, 1], [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]]),
    ([0, 0, -1], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
];

/// The unit-square texture coordinates for a face's four corners, matching `FACES`
/// corner order.
const FACE_UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// The set of occupied structural cells (devices are not surfaced here; WI 582).
fn occupied_cells(craft: &VoxelCraft) -> HashSet<IVec3> {
    craft.voxels.iter().map(|v| v.cell).collect()
}

/// Build the **blocky** skin (WI 582): one textured cube per occupied cell, emitting only
/// the faces that border empty space. Each exposed face is an independent quad (four
/// vertices, two triangles) carrying the face's outward normal and unit-square texture
/// coordinates. Pure and deterministic.
pub fn blocky_mesh(craft: &VoxelCraft) -> SkinMesh {
    let occupied = occupied_cells(craft);
    let s = craft.cell_size as f32;
    let mut mesh = SkinMesh::default();

    for v in &craft.voxels {
        let cell = v.cell;
        for (normal, corners) in FACES.iter() {
            let n = IVec3::new(normal[0], normal[1], normal[2]);
            // Skip interior faces: a face shared with another occupied cell is hidden.
            if occupied.contains(&(cell + n)) {
                continue;
            }
            let base = mesh.positions.len() as u32;
            let nf = [normal[0] as f32, normal[1] as f32, normal[2] as f32];
            for (corner, uv) in corners.iter().zip(FACE_UVS.iter()) {
                mesh.positions.push([
                    (cell.x + corner[0]) as f32 * s,
                    (cell.y + corner[1]) as f32 * s,
                    (cell.z + corner[2]) as f32 * s,
                ]);
                mesh.normals.push(nf);
                mesh.uvs.push(*uv);
            }
            // Two triangles, CCW: (0,1,2) and (0,2,3).
            mesh.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{Material, Voxel};

    fn craft_from(cells: &[IVec3]) -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        for &cell in cells {
            c.voxels.push(Voxel {
                cell,
                material: Material::ALUMINIUM,
            });
        }
        c
    }

    #[test]
    fn single_voxel_has_six_faces() {
        let m = blocky_mesh(&craft_from(&[IVec3::ZERO]));
        assert_eq!(m.face_count(), 6);
        assert_eq!(m.positions.len(), 24);
        assert_eq!(m.triangle_count(), 12);
        assert_eq!(m.indices.len(), 36);
        // UVs and normals are per-vertex parallel arrays.
        assert_eq!(m.normals.len(), 24);
        assert_eq!(m.uvs.len(), 24);
    }

    #[test]
    fn adjacent_cells_cull_the_shared_face() {
        // A 1×1×2 column: 12 faces naively, minus the shared internal pair → 10.
        let m = blocky_mesh(&craft_from(&[IVec3::new(0, 0, 0), IVec3::new(0, 0, 1)]));
        assert_eq!(m.face_count(), 10);
    }

    #[test]
    fn solid_block_emits_only_the_surface() {
        // A solid 2×2×2 block: only the 6 sides × 4 cells = 24 boundary faces; no interior.
        let mut cells = Vec::new();
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    cells.push(IVec3::new(x, y, z));
                }
            }
        }
        let m = blocky_mesh(&craft_from(&cells));
        assert_eq!(m.face_count(), 24, "surface only, interior faces culled");
    }

    #[test]
    fn cell_size_scales_positions() {
        let mut c = craft_from(&[IVec3::ZERO]);
        c.cell_size = 2.0;
        let m = blocky_mesh(&c);
        // Every coordinate is 0 or cell_size (2.0) for a unit cell at the origin.
        for p in &m.positions {
            for &x in p {
                assert!(x == 0.0 || x == 2.0, "coordinate {x} scaled by cell_size");
            }
        }
    }

    #[test]
    fn face_normals_are_unit_axis_aligned() {
        let m = blocky_mesh(&craft_from(&[IVec3::ZERO]));
        for n in &m.normals {
            let len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
            assert!((len2 - 1.0).abs() < 1e-6, "unit normal");
        }
    }

    #[test]
    fn empty_craft_is_empty_mesh() {
        let m = blocky_mesh(&VoxelCraft::new(1.0));
        assert!(m.positions.is_empty() && m.indices.is_empty());
        assert_eq!(m.face_count(), 0);
    }

    #[test]
    fn winding_matches_outward_normal() {
        // For each emitted face the geometric winding (c1−c0)×(c2−c0) must agree with the
        // stored outward normal (front faces point outward under CCW back-face culling).
        let m = blocky_mesh(&craft_from(&[IVec3::ZERO]));
        for tri in m.indices.chunks(3).step_by(2) {
            // first triangle of each quad: indices [0,1,2] relative to the quad base
            let p0 = glam::Vec3::from(m.positions[tri[0] as usize]);
            let p1 = glam::Vec3::from(m.positions[tri[1] as usize]);
            let p2 = glam::Vec3::from(m.positions[tri[2] as usize]);
            let face_n = (p1 - p0).cross(p2 - p0).normalize();
            let stored = glam::Vec3::from(m.normals[tri[0] as usize]);
            assert!(
                face_n.dot(stored) > 0.99,
                "winding normal {face_n:?} agrees with stored {stored:?}"
            );
        }
    }
}

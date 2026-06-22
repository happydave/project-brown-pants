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

use crate::voxel::{Material, VoxelCraft};
use glam::IVec3;
use std::collections::{HashMap, HashSet};

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

/// Integer cell component along axis `i` (0=x, 1=y, 2=z).
fn comp(c: IVec3, i: usize) -> i32 {
    [c.x, c.y, c.z][i]
}

/// In-plane axes `(u, v)` for face-normal axis `a`, chosen so `û × v̂ = â` (used to keep
/// merged-quad winding outward).
const INPLANE: [(usize, usize); 3] = [(1, 2), (2, 0), (0, 1)];

/// One face direction's geometry context: the normal axis `a`, its in-plane axes
/// `(ua, va)`, the face `sign` (±1), and the cell size `s`. Threaded through the greedy
/// merge so the inner functions stay within argument limits.
struct FaceAxis {
    a: usize,
    ua: usize,
    va: usize,
    sign: i32,
    s: f32,
}

/// Build the **greedy hull** skin (WI 583): the same exposed faces as [`blocky_mesh`],
/// but coplanar faces of like material are merged into maximal rectangles (per
/// face-direction and slab), with texture coordinates that **tile** across each rectangle
/// (a w×h-cell quad spans `[0,w]×[0,h]`). Far fewer vertices than blocky for large flat
/// surfaces; pure and deterministic.
pub fn greedy_mesh(craft: &VoxelCraft) -> SkinMesh {
    let occ: HashMap<IVec3, Material> = craft.voxels.iter().map(|v| (v.cell, v.material)).collect();
    let s = craft.cell_size as f32;
    let mut mesh = SkinMesh::default();

    for a in 0..3usize {
        let (ua, va) = INPLANE[a];
        for sign in [1i32, -1] {
            let mut off = [0i32; 3];
            off[a] = sign;
            let nrm = IVec3::new(off[0], off[1], off[2]);

            // Group exposed faces of this direction by slab (the cell's `a`-coordinate),
            // each slab a sparse (u,v) → material mask.
            let mut layers: HashMap<i32, HashMap<(i32, i32), Material>> = HashMap::new();
            for (cell, mat) in &occ {
                if !occ.contains_key(&(*cell + nrm)) {
                    layers
                        .entry(comp(*cell, a))
                        .or_default()
                        .insert((comp(*cell, ua), comp(*cell, va)), *mat);
                }
            }

            let fa = FaceAxis { a, ua, va, sign, s };
            for (k, mask) in &layers {
                greedy_merge_layer(*k, mask, &fa, &mut mesh);
            }
        }
    }
    mesh
}

/// Greedily merge one slab's (u,v) → material mask into maximal rectangles and emit a
/// quad per rectangle.
fn greedy_merge_layer(
    k: i32,
    mask: &HashMap<(i32, i32), Material>,
    fa: &FaceAxis,
    mesh: &mut SkinMesh,
) {
    let umin = mask.keys().map(|&(u, _)| u).min().unwrap_or(0);
    let umax = mask.keys().map(|&(u, _)| u).max().unwrap_or(0);
    let vmin = mask.keys().map(|&(_, v)| v).min().unwrap_or(0);
    let vmax = mask.keys().map(|&(_, v)| v).max().unwrap_or(0);
    let mut visited: HashSet<(i32, i32)> = HashSet::new();

    for v0 in vmin..=vmax {
        for u0 in umin..=umax {
            let Some(&mat) = mask.get(&(u0, v0)) else {
                continue;
            };
            if visited.contains(&(u0, v0)) {
                continue;
            }
            // Extend width along +u while same material, present, unvisited.
            let mut w = 1;
            while mask.get(&(u0 + w, v0)) == Some(&mat) && !visited.contains(&(u0 + w, v0)) {
                w += 1;
            }
            // Extend height along +v while the whole row [u0, u0+w) matches.
            let mut h = 1;
            'rows: loop {
                for du in 0..w {
                    let key = (u0 + du, v0 + h);
                    if mask.get(&key) != Some(&mat) || visited.contains(&key) {
                        break 'rows;
                    }
                }
                h += 1;
            }
            for dv in 0..h {
                for du in 0..w {
                    visited.insert((u0 + du, v0 + dv));
                }
            }
            emit_quad(k, u0, v0, w, h, fa, mesh);
        }
    }
}

/// Emit one merged rectangle as a quad (four vertices, two triangles) with an outward
/// normal, CCW-outward winding, and tiled texture coordinates spanning `[0,w]×[0,h]`.
fn emit_quad(k: i32, u0: i32, v0: i32, w: i32, h: i32, fa: &FaceAxis, mesh: &mut SkinMesh) {
    // The face plane: +faces sit on the high boundary of the cell layer, −faces on the low.
    let ap = if fa.sign > 0 { k + 1 } else { k };
    let xyz = |uv: i32, vv: i32| -> [f32; 3] {
        let mut c = [0i32; 3];
        c[fa.a] = ap;
        c[fa.ua] = uv;
        c[fa.va] = vv;
        [c[0] as f32 * fa.s, c[1] as f32 * fa.s, c[2] as f32 * fa.s]
    };
    let mut n = [0.0f32; 3];
    n[fa.a] = fa.sign as f32;

    // Corner + UV order differs by sign so the winding stays outward (û × v̂ = â).
    let (corners, uvs): ([[f32; 3]; 4], [[f32; 2]; 4]) = if fa.sign > 0 {
        (
            [
                xyz(u0, v0),
                xyz(u0 + w, v0),
                xyz(u0 + w, v0 + h),
                xyz(u0, v0 + h),
            ],
            [
                [0.0, 0.0],
                [w as f32, 0.0],
                [w as f32, h as f32],
                [0.0, h as f32],
            ],
        )
    } else {
        (
            [
                xyz(u0, v0),
                xyz(u0, v0 + h),
                xyz(u0 + w, v0 + h),
                xyz(u0 + w, v0),
            ],
            [
                [0.0, 0.0],
                [0.0, h as f32],
                [w as f32, h as f32],
                [w as f32, 0.0],
            ],
        )
    };

    let base = mesh.positions.len() as u32;
    for (corner, uv) in corners.iter().zip(uvs.iter()) {
        mesh.positions.push(*corner);
        mesh.normals.push(n);
        mesh.uvs.push(*uv);
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
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

    // --- WI 583: greedy hull skin ---

    /// Total surface area of an axis-aligned-rectangle quad mesh: Σ |(p1−p0)×(p3−p0)|.
    fn quad_area_sum(m: &SkinMesh) -> f32 {
        (0..m.positions.len())
            .step_by(4)
            .map(|b| {
                let p0 = glam::Vec3::from(m.positions[b]);
                let p1 = glam::Vec3::from(m.positions[b + 1]);
                let p3 = glam::Vec3::from(m.positions[b + 3]);
                (p1 - p0).cross(p3 - p0).length()
            })
            .sum()
    }

    fn block(nx: i32, ny: i32, nz: i32) -> VoxelCraft {
        let mut cells = Vec::new();
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    cells.push(IVec3::new(x, y, z));
                }
            }
        }
        craft_from(&cells)
    }

    #[test]
    fn greedy_solid_block_is_six_quads() {
        // Each side of a solid block merges to a single rectangle.
        let m = greedy_mesh(&block(2, 3, 4));
        assert_eq!(m.face_count(), 6, "one merged rectangle per side");
    }

    #[test]
    fn greedy_column_merges_long_sides() {
        // 1×1×2: 4 long sides each merge two cells (4 quads) + 2 end caps = 6 (blocky: 10).
        let c = craft_from(&[IVec3::new(0, 0, 0), IVec3::new(0, 0, 1)]);
        assert_eq!(greedy_mesh(&c).face_count(), 6);
        assert_eq!(blocky_mesh(&c).face_count(), 10);
    }

    #[test]
    fn greedy_single_voxel_matches_blocky() {
        // Nothing to merge: 6 quads, same as blocky.
        let c = craft_from(&[IVec3::ZERO]);
        assert_eq!(greedy_mesh(&c).face_count(), 6);
    }

    #[test]
    fn greedy_conserves_exposed_area() {
        // The merged rectangles cover exactly the exposed surface: greedy area == the
        // naive per-cube exposed area (blocky face count, cell_size 1) — for a block, a
        // column, and a non-rectangular L silhouette.
        let l = craft_from(&[
            IVec3::new(0, 0, 0),
            IVec3::new(1, 0, 0),
            IVec3::new(0, 1, 0),
        ]);
        for c in [block(2, 2, 2), block(1, 1, 3), l] {
            let blocky_area = blocky_mesh(&c).face_count() as f32; // cell_size 1 → area = faces
            assert!(
                (quad_area_sum(&greedy_mesh(&c)) - blocky_area).abs() < 1e-4,
                "greedy area conserved vs blocky"
            );
        }
    }

    #[test]
    fn greedy_has_fewer_vertices_than_blocky() {
        // The 2×2×5 hull the -- skins scene flies: greedy merges the broad faces.
        let hull = block(2, 2, 5);
        let g = greedy_mesh(&hull);
        let b = blocky_mesh(&hull);
        assert!(
            g.positions.len() < b.positions.len(),
            "greedy {} < blocky {} vertices",
            g.positions.len(),
            b.positions.len()
        );
    }

    #[test]
    fn greedy_winding_matches_outward_normal() {
        // Both face signs: geometric winding agrees with the stored normal.
        let m = greedy_mesh(&block(2, 2, 2));
        for tri in m.indices.chunks(3).step_by(2) {
            let p0 = glam::Vec3::from(m.positions[tri[0] as usize]);
            let p1 = glam::Vec3::from(m.positions[tri[1] as usize]);
            let p2 = glam::Vec3::from(m.positions[tri[2] as usize]);
            let face_n = (p1 - p0).cross(p2 - p0).normalize();
            let stored = glam::Vec3::from(m.normals[tri[0] as usize]);
            assert!(face_n.dot(stored) > 0.99, "outward winding");
        }
    }

    #[test]
    fn greedy_tiles_uvs_across_merged_rectangle() {
        // A solid block's merged side spans multiple cells; its UVs must reach the cell
        // dimensions (tiling), not stay within [0,1] (stretched).
        let m = greedy_mesh(&block(2, 2, 2));
        let max_u = m.uvs.iter().map(|uv| uv[0]).fold(0.0_f32, f32::max);
        assert!(
            max_u >= 2.0,
            "UVs tile across the merged rectangle (got {max_u})"
        );
    }
}

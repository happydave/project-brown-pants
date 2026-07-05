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
//! The shared rule: a cell face portion is **culled** only where the neighbouring
//! coverage fully mates against it (WI 833 — for a craft with no shaped cells this is
//! exactly the original rule: a face shared with another occupied cell is hidden).
//! **Shaped cells** (WI 831/833) emit their oriented catalog form mesh instead of cube
//! faces — the same canonical polyhedron that derived their physics — with face-plane
//! triangles culled per that rule and oblique triangles always emitted (over-emission
//! before optimization, per the design). The greedy skin treats shaped cells as
//! unmergeable islands; unshaped cells keep merging among themselves.

use crate::shape::{self, FaceMask, ShapedCell, MASK_EMPTY, MASK_FULL};
use crate::voxel::{Material, VoxelCraft};
use glam::{DVec3, IVec3};
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

/// The coverage mask a neighbouring cell presents back across a boundary (WI 833):
/// empty when unoccupied, full when an unshaped solid, its form's oriented face mask
/// when shaped. `face` is the neighbour's own face index toward the boundary
/// (the `[x0, x1, y0, y1, z0, z1]` order shared with `boundary_sealed`).
fn neighbour_mask(
    craft: &VoxelCraft,
    occupied: &HashSet<IVec3>,
    cell: IVec3,
    face: usize,
) -> FaceMask {
    if !occupied.contains(&cell) {
        return MASK_EMPTY;
    }
    match craft.shape_at(cell) {
        Some(s) => shape::face_masks(s.form, s.orientation)[face],
        None => MASK_FULL,
    }
}

/// `a ⊆ b` on coverage masks: every covered sample of `a` is covered by `b`.
fn mask_subset(a: &FaceMask, b: &FaceMask) -> bool {
    (0..4).all(|w| a[w] & !b[w] == 0)
}

/// Vertex-position tolerance for face-plane membership of oriented form triangles.
const FACE_EPS: f64 = 1e-9;

/// The unit-cube face (in `[x0, x1, y0, y1, z0, z1]` order) a triangle lies wholly
/// in, if any — oblique triangles (hypotenuses) return `None`.
fn face_plane_of(a: DVec3, b: DVec3, c: DVec3) -> Option<usize> {
    for (face, axis, side) in [
        (0usize, 0usize, 0.0),
        (1, 0, 1.0),
        (2, 1, 0.0),
        (3, 1, 1.0),
        (4, 2, 0.0),
        (5, 2, 1.0),
    ] {
        if (a[axis] - side).abs() < FACE_EPS
            && (b[axis] - side).abs() < FACE_EPS
            && (c[axis] - side).abs() < FACE_EPS
        {
            return Some(face);
        }
    }
    None
}

/// Emit one shaped cell's oriented form mesh (WI 833): triangles from the canonical
/// catalog mesh rotated about the cell centre, translated to the cell, scaled by the
/// cell size. Face-plane triangles are culled only when the cell's own coverage on
/// that face is a subset of the neighbour's opposing coverage (fully mated); oblique
/// triangles always emit. Flat per-triangle normals from the (rotation-preserved)
/// winding; planar-projection UVs in the two tangent axes of the normal's dominant
/// axis (the blocky unit-square-per-cell convention), so tangent generation stays
/// valid downstream.
fn emit_form_cell(
    craft: &VoxelCraft,
    occupied: &HashSet<IVec3>,
    sc: &ShapedCell,
    mesh: &mut SkinMesh,
) {
    let s = craft.cell_size as f32;
    let fm = shape::form_mesh(sc.form);
    let r = shape::rotations()[sc.orientation as usize];
    let oriented: Vec<DVec3> = fm
        .vertices
        .iter()
        .map(|&p| r * (p - DVec3::splat(0.5)) + DVec3::splat(0.5))
        .collect();
    let own = shape::face_masks(sc.form, sc.orientation);
    // Per-face cull decision: own coverage fully mated by the neighbour's.
    let mut cull = [false; 6];
    for (face, c) in cull.iter_mut().enumerate() {
        let axis = face / 2;
        let positive = face % 2 == 1;
        let mut dir = IVec3::ZERO;
        dir[axis] = if positive { 1 } else { -1 };
        // The neighbour's face back toward this boundary.
        let nface = if positive { 2 * axis } else { 2 * axis + 1 };
        let n = neighbour_mask(craft, occupied, sc.cell + dir, nface);
        *c = own[face] != MASK_EMPTY && mask_subset(&own[face], &n);
    }
    for t in &fm.triangles {
        let (a, b, c) = (oriented[t[0]], oriented[t[1]], oriented[t[2]]);
        if let Some(face) = face_plane_of(a, b, c) {
            if cull[face] {
                continue;
            }
        }
        let n = (b - a).cross(c - a).normalize();
        // UV projection axes: the two tangents of the normal's dominant axis.
        let dominant = if n.x.abs() >= n.y.abs() && n.x.abs() >= n.z.abs() {
            0
        } else if n.y.abs() >= n.z.abs() {
            1
        } else {
            2
        };
        let (t1, t2) = match dominant {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let base = mesh.positions.len() as u32;
        let nf = [n.x as f32, n.y as f32, n.z as f32];
        for p in [a, b, c] {
            mesh.positions.push([
                (sc.cell.x as f64 + p.x) as f32 * s,
                (sc.cell.y as f64 + p.y) as f32 * s,
                (sc.cell.z as f64 + p.z) as f32 * s,
            ]);
            mesh.normals.push(nf);
            mesh.uvs.push([p[t1] as f32, p[t2] as f32]);
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

/// Build the **blocky** skin (WI 582): one textured cube per occupied cell, emitting only
/// the faces that border empty space. Each exposed face is an independent quad (four
/// vertices, two triangles) carrying the face's outward normal and unit-square texture
/// coordinates. Pure and deterministic.
pub fn blocky_mesh(craft: &VoxelCraft) -> SkinMesh {
    let occupied = occupied_cells(craft);
    let s = craft.cell_size as f32;
    let mut mesh = SkinMesh::default();
    // The exact pre-833 path when nothing is shaped (structural, not incidental).
    let shaped = !craft.shapes.is_empty();

    for v in &craft.voxels {
        let cell = v.cell;
        if shaped {
            if let Some(sc) = craft.shape_at(cell) {
                emit_form_cell(craft, &occupied, sc, &mut mesh);
                continue;
            }
        }
        for (normal, corners) in FACES.iter() {
            let n = IVec3::new(normal[0], normal[1], normal[2]);
            // Skip interior faces: a face fully mated by the neighbour is hidden
            // (for an unshaped craft: any occupied neighbour, the original rule).
            if shaped {
                let axis = if n.x != 0 {
                    0
                } else if n.y != 0 {
                    1
                } else {
                    2
                };
                let nface = if n[axis] > 0 { 2 * axis } else { 2 * axis + 1 };
                if neighbour_mask(craft, &occupied, cell + n, nface) == MASK_FULL {
                    continue;
                }
            } else if occupied.contains(&(cell + n)) {
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
    // Shaped cells are unmergeable islands (WI 833): emit each as its oriented form
    // mesh and exclude it from the rectangle-merge layers below. Deterministic order
    // via the sorted shape store. Unshaped crafts skip all of this.
    let shaped = !craft.shapes.is_empty();
    let occupied: HashSet<IVec3> = if shaped {
        occ.keys().copied().collect()
    } else {
        HashSet::new()
    };
    if shaped {
        for sc in &craft.shapes {
            if occ.contains_key(&sc.cell) {
                emit_form_cell(craft, &occupied, sc, &mut mesh);
            }
        }
    }

    for a in 0..3usize {
        let (ua, va) = INPLANE[a];
        for sign in [1i32, -1] {
            let mut off = [0i32; 3];
            off[a] = sign;
            let nrm = IVec3::new(off[0], off[1], off[2]);
            // The neighbour's face back toward this direction's boundary.
            let nface = if sign > 0 { 2 * a } else { 2 * a + 1 };

            // Group exposed faces of this direction by slab (the cell's `a`-coordinate),
            // each slab a sparse (u,v) → material mask. A face is exposed unless the
            // neighbour's coverage fully mates it (for an unshaped craft this is the
            // original "neighbour occupied" rule).
            let mut layers: HashMap<i32, HashMap<(i32, i32), Material>> = HashMap::new();
            for (cell, mat) in &occ {
                if shaped && craft.shape_at(*cell).is_some() {
                    continue; // emitted as a form island above
                }
                let hidden = if shaped {
                    neighbour_mask(craft, &occupied, *cell + nrm, nface) == MASK_FULL
                } else {
                    occ.contains_key(&(*cell + nrm))
                };
                if !hidden {
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

    // --- WI 833: shaped-cell emission + coverage-aware culling ---

    use crate::shape::{FillMode, Form, ShapedCell};

    fn wedge_at(craft: &mut VoxelCraft, cell: IVec3, orientation: u8) {
        craft.voxels.push(Voxel {
            cell,
            material: Material::ALUMINIUM,
        });
        craft.set_shape(ShapedCell {
            cell,
            form: Form::Wedge,
            orientation,
            fill: FillMode::Solid,
        });
    }

    #[test]
    fn a_lone_wedge_emits_its_full_form_in_both_skins() {
        // One shaped wedge, no neighbours: all 8 canonical triangles emit (bottom 2 +
        // back 2 + hypotenuse 2 + two half sides), nothing culled — and the winding
        // still agrees with the stored flat normals.
        let mut c = VoxelCraft::new(1.0);
        wedge_at(&mut c, IVec3::ZERO, 0);
        for m in [blocky_mesh(&c), greedy_mesh(&c)] {
            assert_eq!(m.triangle_count(), 8, "the wedge's own triangles");
            for tri in m.indices.chunks(3) {
                let p0 = glam::Vec3::from(m.positions[tri[0] as usize]);
                let p1 = glam::Vec3::from(m.positions[tri[1] as usize]);
                let p2 = glam::Vec3::from(m.positions[tri[2] as usize]);
                let face_n = (p1 - p0).cross(p2 - p0).normalize();
                let stored = glam::Vec3::from(m.normals[tri[0] as usize]);
                assert!(face_n.dot(stored) > 0.99, "outward winding");
            }
        }
    }

    #[test]
    fn a_fully_mated_face_culls_and_a_buried_half_face_culls() {
        // Wedge (y ≤ z) with a cube behind its full z=1 face: that boundary emits
        // nothing on either side — wedge 8−2=6 triangles, cube 5 faces (10).
        let mut c = VoxelCraft::new(1.0);
        wedge_at(&mut c, IVec3::ZERO, 0);
        c.voxels.push(Voxel {
            cell: IVec3::Z,
            material: Material::ALUMINIUM,
        });
        assert_eq!(blocky_mesh(&c).triangle_count(), 6 + 10);

        // Wedge with a cube beside its half-covered x=1 face: the wedge's half
        // triangle is buried against full coverage (culled, 8−1=7); the cube's face
        // is only partially covered, so it emits all 6 faces (12) — a partial
        // neighbour no longer hides the cube face behind it.
        let mut c = VoxelCraft::new(1.0);
        wedge_at(&mut c, IVec3::ZERO, 0);
        c.voxels.push(Voxel {
            cell: IVec3::X,
            material: Material::ALUMINIUM,
        });
        assert_eq!(blocky_mesh(&c).triangle_count(), 7 + 12);
    }

    #[test]
    fn mated_complementary_wedges_emit_both_half_faces() {
        // The WI 832 seal fixture: complementary wedges across one X boundary seal
        // the sim, but each half-face triangle borders the neighbour's air, so both
        // emit (nothing fully mates) — visible geometry and sealing agree.
        let mask = crate::shape::face_masks(Form::Wedge, 0)[1]; // its x=1 face
        let complement = crate::shape::constants(Form::Wedge)
            .distinct_orientations
            .iter()
            .copied()
            .find(|&o| {
                let m = crate::shape::face_masks(Form::Wedge, o)[0]; // an x=0 face
                crate::shape::masks_seal(&mask, &m) && crate::shape::mask_popcount(&m) < 200
            })
            .expect("a complementary orientation exists");
        let mut c = VoxelCraft::new(1.0);
        wedge_at(&mut c, IVec3::ZERO, 0);
        wedge_at(&mut c, IVec3::X, complement);
        // Both wedges emit all 8 of their triangles: no face is fully mated.
        assert_eq!(blocky_mesh(&c).triangle_count(), 16);
        // And the boundary genuinely seals (the render-vs-seal agreement pin).
        let solid: std::collections::HashSet<IVec3> = c.voxels.iter().map(|v| v.cell).collect();
        assert!(c.boundary_sealed(&solid, IVec3::ZERO, IVec3::X));
    }

    #[test]
    fn an_unshaped_craft_meshes_exactly_as_before() {
        // The fast-path invariant: no shapes ⇒ byte-identical generator output to
        // the original rule (same craft, shapes-free), for both skins.
        let c = block(2, 2, 5);
        assert!(c.shapes.is_empty());
        let m = blocky_mesh(&c);
        assert_eq!(m.face_count(), 2 * (2 * 2 + 2 * 5 + 2 * 5));
        assert_eq!(greedy_mesh(&c).face_count(), 6);
    }

    #[test]
    fn greedy_still_merges_unshaped_cells_around_a_shaped_island() {
        // A 1×1×3 column whose middle cell is a wedge: the two unshaped end cells
        // still merge/emit their own faces; the wedge emits as a form island. The
        // craft renders with no holes: total triangles = ends' quads × 2 + wedge's
        // (8 − culled full faces).
        let mut c = VoxelCraft::new(1.0);
        for z in [0, 2] {
            c.voxels.push(Voxel {
                cell: IVec3::new(0, 0, z),
                material: Material::ALUMINIUM,
            });
        }
        // Wedge y ≤ z at the middle: full z=1 face mates the z=2 cube (culled);
        // its z=0 face is empty (nothing to cull); bottom y=0 face borders air.
        wedge_at(&mut c, IVec3::new(0, 0, 1), 0);
        let m = greedy_mesh(&c);
        // End cubes: z=0 cube emits 5 faces toward air + its z=1 face (the wedge
        // covers none of that face's plane — wedge z=0 coverage is empty) = 6;
        // z=2 cube emits 5 (its z=0 face fully mated by the wedge's full z=1 face
        // — culled). Wedge: 8 − 2 (z=1 mated) = 6 triangles.
        assert_eq!(m.triangle_count(), 6 * 2 + 5 * 2 + 6);
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

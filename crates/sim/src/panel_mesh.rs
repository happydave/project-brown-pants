//! Face-panel render geometry (WI 825, panels design stage 2): turn a craft's
//! [`FacePanel`] records into **engine-agnostic** mesh data — one thin plate box per
//! record, grouped per material, plus the generated **frame/trim** edge treatment
//! (mullion seams between panels, rebate lines against solids, rim rails on free
//! edges). The plate you see is the plate that weighs, floats, and seals (WI 824);
//! this module makes every solid-render scene draw it from the same data.
//!
//! Headless and rendering-free like [`crate::voxel_mesh`]: emits [`SkinMesh`] vectors
//! only, so the generators are unit-tested without a display and the app owns the
//! conversion to its mesh type. All output is **deterministic**: panels iterate in
//! their sorted-store order (WI 820 discipline) and edges in canonical key order.

use crate::voxel::{Axis, FacePanel, Material, VoxelCraft, PANEL_FILL};
use crate::voxel_mesh::SkinMesh;
use glam::{IVec3, Vec3};
use std::collections::{BTreeMap, HashSet};

/// Trim cross-section (square side) for **mullion** seams and **rim** rails, as a
/// multiple of the plate thickness (`PANEL_FILL × cell_size`). Proud of the plate on
/// both sides so the seam reads. A WI 825 visual knob.
pub const TRIM_MULLION_SCALE: f64 = 2.5;

/// Trim cross-section for **rebate** lines (where a plate seats into a solid cube
/// face), as a multiple of the plate thickness — slimmer than a mullion: most of a
/// rebate strip is buried in the solid, leaving a quarter-round colour break on the
/// cube face. A WI 825 visual knob.
pub const TRIM_REBATE_SCALE: f64 = 2.0;

/// How a panel edge meets its neighbourhood — the frame/trim visual language of the
/// panels design (stage 2). Classification is material-blind and per lattice edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeClass {
    /// At least one of the (up to four) cells around the edge is solid: the plate
    /// seats into the cube face. Takes precedence over `Mullion`.
    Rebate,
    /// No solid, and two or more panels share the edge: the seam between plates —
    /// a glass pane bounded by structural plates gets its window frame from here.
    Mullion,
    /// A single panel's free edge: the plate's frame rail.
    Rim,
}

/// One classified lattice edge touched by at least one panel: the unit segment from
/// lattice point `origin` to `origin + axis.unit()` (lattice-point coordinates: cell
/// `c` spans corners `c ..= c + 1`). Each shared boundary edge appears exactly once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelEdge {
    /// The edge's lower lattice corner.
    pub origin: IVec3,
    /// The edge's direction axis.
    pub axis: Axis,
    /// How the edge meets its neighbourhood.
    pub class: EdgeClass,
}

/// The two axes tangent to `axis` (in a fixed order, so edge enumeration is
/// deterministic).
fn tangents(axis: Axis) -> (Axis, Axis) {
    match axis {
        Axis::X => (Axis::Y, Axis::Z),
        Axis::Y => (Axis::X, Axis::Z),
        Axis::Z => (Axis::X, Axis::Y),
    }
}

/// The four lattice edges bounding panel `p`'s face, as `(origin, axis)` pairs.
/// The face lies in the plane one step along `p.axis` from `p.cell`'s corner and
/// spans the two tangent axes.
fn panel_edges(p: &FacePanel) -> [(IVec3, Axis); 4] {
    let corner = p.cell + p.axis.unit(); // the face's lower lattice corner
    let (u, v) = tangents(p.axis);
    [
        (corner, u),
        (corner + v.unit(), u),
        (corner, v),
        (corner + u.unit(), v),
    ]
}

/// The up-to-four cells sharing lattice edge `(origin, axis)`: fixed edge-axis
/// coordinate, and each combination of `-1/0` offsets along the two other axes.
fn edge_cells(origin: IVec3, axis: Axis) -> [IVec3; 4] {
    let (w1, w2) = tangents(axis);
    [
        origin - w1.unit() - w2.unit(),
        origin - w1.unit(),
        origin - w2.unit(),
        origin,
    ]
}

/// Classify every lattice edge touched by at least one of `craft`'s face panels
/// (deduplicated — a boundary edge shared by several panels appears once), in
/// canonical `(origin, axis)` order. This is the testable core of the trim language;
/// [`panel_trim_mesh`] emits one strip per returned edge.
pub fn classify_panel_edges(craft: &VoxelCraft) -> Vec<PanelEdge> {
    let mut counts: BTreeMap<(i32, i32, i32, u8), usize> = BTreeMap::new();
    for p in &craft.face_panels {
        for (origin, axis) in panel_edges(p) {
            *counts
                .entry((origin.x, origin.y, origin.z, axis as u8))
                .or_default() += 1;
        }
    }
    let occupied: HashSet<IVec3> = craft.voxels.iter().map(|v| v.cell).collect();
    counts
        .into_iter()
        .map(|((x, y, z, a), panels)| {
            let origin = IVec3::new(x, y, z);
            let axis = match a {
                0 => Axis::X,
                1 => Axis::Y,
                _ => Axis::Z,
            };
            let seated = edge_cells(origin, axis)
                .iter()
                .any(|c| occupied.contains(c));
            let class = if seated {
                EdgeClass::Rebate
            } else if panels >= 2 {
                EdgeClass::Mullion
            } else {
                EdgeClass::Rim
            };
            PanelEdge {
                origin,
                axis,
                class,
            }
        })
        .collect()
}

/// The distinct panel materials present, in first-seen order over the sorted store
/// (deterministic; mirrors the solid skin's per-material split).
fn panel_materials(craft: &VoxelCraft) -> Vec<Material> {
    let mut out: Vec<Material> = Vec::new();
    for p in &craft.face_panels {
        if !out.contains(&p.material) {
            out.push(p.material);
        }
    }
    out
}

/// One merged plate mesh per distinct panel material: a thin axis-aligned box per
/// [`FacePanel`] record — full cell face, `PANEL_FILL × cell_size` thick, centred on
/// the record's boundary. Exactly one box per record (the workitem's
/// plate-per-record contract); empty when the craft has no panels.
pub fn panel_submeshes(craft: &VoxelCraft) -> Vec<(Material, SkinMesh)> {
    let s = craft.cell_size as f32;
    let t = (craft.cell_size * PANEL_FILL) as f32;
    panel_materials(craft)
        .into_iter()
        .map(|m| {
            let mut mesh = SkinMesh::default();
            for p in craft.face_panels.iter().filter(|p| p.material == m) {
                let half = match p.axis {
                    Axis::X => Vec3::new(0.5 * t, 0.5 * s, 0.5 * s),
                    Axis::Y => Vec3::new(0.5 * s, 0.5 * t, 0.5 * s),
                    Axis::Z => Vec3::new(0.5 * s, 0.5 * s, 0.5 * t),
                };
                emit_box(&mut mesh, craft.face_center(p).as_vec3(), half);
            }
            (m, mesh)
        })
        .collect()
}

/// The trim mesh: one square-section strip per classified panel edge (all classes in
/// one mesh — one trim appearance), cell-length along the edge, cross-section set by
/// the class ([`TRIM_MULLION_SCALE`] / [`TRIM_REBATE_SCALE`] × plate thickness),
/// centred on the lattice edge line. Empty when the craft has no panels.
pub fn panel_trim_mesh(craft: &VoxelCraft) -> SkinMesh {
    let s = craft.cell_size as f32;
    let t = craft.cell_size * PANEL_FILL;
    let mut mesh = SkinMesh::default();
    for e in classify_panel_edges(craft) {
        let w = match e.class {
            EdgeClass::Rebate => (t * TRIM_REBATE_SCALE) as f32,
            EdgeClass::Mullion | EdgeClass::Rim => (t * TRIM_MULLION_SCALE) as f32,
        };
        let half = match e.axis {
            Axis::X => Vec3::new(0.5 * s, 0.5 * w, 0.5 * w),
            Axis::Y => Vec3::new(0.5 * w, 0.5 * s, 0.5 * w),
            Axis::Z => Vec3::new(0.5 * w, 0.5 * w, 0.5 * s),
        };
        let center = e.origin.as_vec3() * s + e.axis.unit().as_vec3() * (0.5 * s);
        emit_box(&mut mesh, center, half);
    }
    mesh
}

/// The six axis-aligned box faces: outward normal + four corner signs wound
/// counter-clockwise as seen from outside (same convention as `voxel_mesh::FACES`,
/// but over `±half` extents instead of unit cells).
const BOX_FACES: [([f32; 3], [[f32; 3]; 4]); 6] = [
    (
        [1.0, 0.0, 0.0],
        [
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
        ],
    ),
    (
        [-1.0, 0.0, 0.0],
        [
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
    ),
    (
        [0.0, 1.0, 0.0],
        [
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ],
    ),
    (
        [0.0, -1.0, 0.0],
        [
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
        ],
    ),
    (
        [0.0, 0.0, 1.0],
        [
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ],
    ),
    (
        [0.0, 0.0, -1.0],
        [
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
    ),
];

/// Unit-square texture coordinates matching `BOX_FACES` corner order.
const BOX_UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// Append an axis-aligned box (`center ± half`) to `mesh`: six quads with outward
/// flat normals and unit-square texture coordinates (valid for tangent generation).
fn emit_box(mesh: &mut SkinMesh, center: Vec3, half: Vec3) {
    for (normal, corners) in BOX_FACES.iter() {
        let base = mesh.positions.len() as u32;
        for (corner, uv) in corners.iter().zip(BOX_UVS.iter()) {
            mesh.positions.push([
                center.x + corner[0] * half.x,
                center.y + corner[1] * half.y,
                center.z + corner[2] * half.z,
            ]);
            mesh.normals.push(*normal);
            mesh.uvs.push(*uv);
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn craft() -> VoxelCraft {
        VoxelCraft::new(0.5)
    }

    /// A single X-normal plate with no neighbours: one submesh, one box, all four
    /// edges free rims, and the box is thin along X only.
    #[test]
    fn lone_plate_is_one_thin_box_with_four_rim_edges() {
        let mut c = craft();
        c.set_face_panel(IVec3::ZERO, IVec3::X, Some(Material::STEEL));

        let subs = panel_submeshes(&c);
        assert_eq!(subs.len(), 1);
        let (m, mesh) = &subs[0];
        assert_eq!(*m, Material::STEEL);
        assert_eq!(mesh.face_count(), 6, "one box per record");

        // Thin along the panel axis: X extent = plate thickness, Y/Z = full cell.
        let ext = |i: usize| {
            let vals: Vec<f32> = mesh.positions.iter().map(|p| p[i]).collect();
            vals.iter().cloned().fold(f32::MIN, f32::max)
                - vals.iter().cloned().fold(f32::MAX, f32::min)
        };
        let t = (c.cell_size * PANEL_FILL) as f32;
        assert!((ext(0) - t).abs() < 1e-6, "thin axis = plate thickness");
        assert!((ext(1) - 0.5).abs() < 1e-6, "full cell face");
        assert!((ext(2) - 0.5).abs() < 1e-6, "full cell face");

        let edges = classify_panel_edges(&c);
        assert_eq!(edges.len(), 4);
        assert!(edges.iter().all(|e| e.class == EdgeClass::Rim));
    }

    /// Two coplanar adjacent panels share exactly one edge (deduplicated): the seam
    /// classifies mullion, the outer boundary rims.
    #[test]
    fn coplanar_pair_shares_one_mullion_edge() {
        let mut c = craft();
        c.set_face_panel(IVec3::ZERO, IVec3::X, Some(Material::STEEL));
        c.set_face_panel(IVec3::new(0, 1, 0), IVec3::X, Some(Material::STEEL));

        let edges = classify_panel_edges(&c);
        assert_eq!(edges.len(), 7, "4 + 4 edges minus the shared one");
        let mullions: Vec<_> = edges
            .iter()
            .filter(|e| e.class == EdgeClass::Mullion)
            .collect();
        assert_eq!(mullions.len(), 1, "exactly the shared seam");
        assert_eq!(
            edges.iter().filter(|e| e.class == EdgeClass::Rim).count(),
            6
        );
    }

    /// A plate laminated onto a solid cube's face seats all four edges (rebate takes
    /// precedence); a wall standing on a solid deck seats only its bottom edge.
    #[test]
    fn edges_against_solids_classify_rebate() {
        let mut c = craft();
        c.voxels.push(crate::voxel::Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        c.set_face_panel(IVec3::ZERO, IVec3::X, Some(Material::STEEL));
        let edges = classify_panel_edges(&c);
        assert_eq!(edges.len(), 4);
        assert!(
            edges.iter().all(|e| e.class == EdgeClass::Rebate),
            "laminated plate seats on every edge"
        );

        // A free-standing X-normal wall one cell above a solid deck cell: only the
        // edge touching the deck seats.
        let mut c = craft();
        c.voxels.push(crate::voxel::Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        c.set_face_panel(IVec3::new(0, 1, 0), IVec3::X, Some(Material::STEEL));
        let edges = classify_panel_edges(&c);
        assert_eq!(edges.len(), 4);
        assert_eq!(
            edges
                .iter()
                .filter(|e| e.class == EdgeClass::Rebate)
                .count(),
            1,
            "only the deck-touching edge seats"
        );
    }

    /// The framed-window case: a glass pane surrounded by four coplanar structural
    /// panels gets mullions on all four of its edges.
    #[test]
    fn glass_pane_in_a_plate_wall_is_framed_by_mullions() {
        let mut c = craft();
        for (y, z) in [(1, 0), (1, 2), (0, 1), (2, 1)] {
            c.set_face_panel(IVec3::new(0, y, z), IVec3::X, Some(Material::STEEL));
        }
        c.set_face_panel(IVec3::new(0, 1, 1), IVec3::X, Some(Material::GLASS));

        let edges = classify_panel_edges(&c);
        let pane_corner = IVec3::new(1, 1, 1); // the glass face's lower lattice corner
        let pane_edges: Vec<_> = panel_edges(&FacePanel {
            cell: IVec3::new(0, 1, 1),
            axis: Axis::X,
            material: Material::GLASS,
        })
        .to_vec();
        assert_eq!(pane_edges[0].0, pane_corner, "fixture sanity");
        for (origin, axis) in pane_edges {
            let e = edges
                .iter()
                .find(|e| e.origin == origin && e.axis == axis)
                .expect("pane edge classified");
            assert_eq!(e.class, EdgeClass::Mullion, "window frame edge");
        }
    }

    /// Per-material grouping partitions the records: one submesh per distinct
    /// material, box counts summing to the record count; glass groups separately.
    #[test]
    fn submeshes_partition_records_by_material() {
        let mut c = craft();
        c.set_face_panel(IVec3::ZERO, IVec3::X, Some(Material::STEEL));
        c.set_face_panel(IVec3::ZERO, IVec3::Y, Some(Material::STEEL));
        c.set_face_panel(IVec3::ZERO, IVec3::Z, Some(Material::GLASS));

        let subs = panel_submeshes(&c);
        assert_eq!(subs.len(), 2, "two distinct panel materials");
        let boxes: usize = subs.iter().map(|(_, m)| m.face_count() / 6).sum();
        assert_eq!(boxes, c.face_panels.len(), "one box per record");
    }

    /// One trim strip per classified edge, and generation is deterministic
    /// (identical craft ⇒ identical vectors).
    #[test]
    fn trim_mesh_emits_one_strip_per_edge_deterministically() {
        let mut c = craft();
        c.set_face_panel(IVec3::ZERO, IVec3::X, Some(Material::STEEL));
        c.set_face_panel(IVec3::new(0, 1, 0), IVec3::X, Some(Material::STEEL));

        let edges = classify_panel_edges(&c);
        let trim = panel_trim_mesh(&c);
        assert_eq!(trim.face_count(), edges.len() * 6, "one box per edge");

        assert_eq!(trim, panel_trim_mesh(&c), "deterministic trim");
        assert_eq!(
            panel_submeshes(&c),
            panel_submeshes(&c),
            "deterministic plates"
        );
    }

    /// No panels ⇒ empty output everywhere (panel-less scenes stay untouched).
    #[test]
    fn no_panels_no_geometry() {
        let c = craft();
        assert!(panel_submeshes(&c).is_empty());
        assert!(classify_panel_edges(&c).is_empty());
        assert_eq!(panel_trim_mesh(&c).face_count(), 0);
    }
}

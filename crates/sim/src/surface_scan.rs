//! Headless **seam scan** (WI 785): measures how far apart adjacent surface chunks'
//! rendered surfaces are at their shared boundaries, turning the visual LOD "cliff"
//! (WI 783) into a deterministic, testable number — no GPU, no rendering.
//!
//! The analytic [`SurfaceField`](crate::surface_field::SurfaceField) is continuous; the
//! streamed mesh is a per-chunk approximation drawn with CDLOD geomorph (WI 780) +
//! skirts (WI 779). A *seam* is where two adjacent chunks render **different** surface
//! positions at the boundary they share. Because the renderer's geometry is fully
//! determined by pure functions in [`crate::surface_mesh`] (`build_chunk`, `morph_range`,
//! `should_split`), this disagreement is computable here.
//!
//! The scan reproduces exactly what the WI 780 shader draws: a vertex at base-relative
//! position `b` with morph target `t` renders at `b + f·(t − b)`, where
//! `f = smoothstep(start, end, |camera − b_world|)` and
//! `(start, end) = morph_range(level, radius)`. Skirts are a *cover* for
//! residual gaps, not surface, so the primary metric is the surface gap; the report also
//! notes the finer chunk's skirt depth for context.

use crate::surface_field::SurfaceField;
use crate::surface_mesh::{
    build_chunk, chunk_relief, direction, morph_range, should_split, skirt_depth_for, CubeFace,
    QuadNode, DEFAULT_MAX_LEVEL, DEFAULT_RESOLUTION,
};
use glam::{DVec3, Vec3};

/// Where the worst seam was found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeamLocation {
    /// The leaf whose boundary vertex had the worst gap.
    pub node: QuadNode,
    /// The neighbour leaf's level on the far side (equal, coarser, or finer).
    pub neighbour_level: u32,
    /// The finer of the two chunks' skirt depth (metres) — the cover available for the
    /// gap. `gap > skirt_depth` means a see-through crack; `gap ≤ skirt_depth` means the
    /// crack is hidden but a step/skirt-teeth may still show.
    pub skirt_depth: f64,
}

/// The result of a seam scan for one camera pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeamReport {
    /// Worst boundary gap found (metres) between a chunk's rendered boundary vertex and
    /// the neighbouring chunk's rendered surface at the same direction.
    pub worst_gap: f64,
    /// Where the worst gap occurred (`None` if there were no interior boundary samples).
    pub worst_location: Option<SeamLocation>,
    /// Number of resident leaves scanned.
    pub leaf_count: usize,
    /// Number of boundary vertices compared against a neighbour.
    pub boundary_samples: usize,
}

/// The resident **leaf** set for a camera pose — the headless equivalent of the renderer's
/// leaf traversal (six roots, split via [`should_split`], collect leaves). `camera_world`
/// is body-centred (same convention as [`should_split`]).
pub fn resident_leaves(camera_world: DVec3, radius: f64, max_level: u32) -> Vec<QuadNode> {
    let mut leaves = Vec::new();
    let mut stack: Vec<QuadNode> = QuadNode::roots().to_vec();
    while let Some(node) = stack.pop() {
        if should_split(node, camera_world, radius, max_level) {
            stack.extend_from_slice(&node.children());
        } else {
            leaves.push(node);
        }
    }
    leaves
}

/// The cube face a unit direction belongs to (its dominant axis). Consistent with
/// [`CubeFace::cube_point`], where the face's fixed axis is the largest component.
fn face_of(d: DVec3) -> CubeFace {
    let (ax, ay, az) = (d.x.abs(), d.y.abs(), d.z.abs());
    if ax >= ay && ax >= az {
        if d.x >= 0.0 {
            CubeFace::PosX
        } else {
            CubeFace::NegX
        }
    } else if ay >= az {
        if d.y >= 0.0 {
            CubeFace::PosY
        } else {
            CubeFace::NegY
        }
    } else if d.z >= 0.0 {
        CubeFace::PosZ
    } else {
        CubeFace::NegZ
    }
}

/// Whether `d` lies within (or on the boundary of) a node's spherical quad — tested from
/// the four corner directions alone (a direction is inside iff it is on the same side of
/// every bounding great circle as the node centre). No inverse of the spherify map is
/// needed, so this works identically within a face and across cube-face seams.
fn node_contains_dir(node: QuadNode, d: DVec3) -> bool {
    let c = node.corner_dirs(); // [c00, c10, c11, c01], CCW from outside
    let center = node.center_dir();
    for k in 0..4 {
        let n = c[k].cross(c[(k + 1) % 4]);
        let sd = d.dot(n);
        let sc = center.dot(n);
        // Opposite side of an edge from the centre (beyond a small boundary tolerance)
        // ⇒ outside. Points on the boundary (sd ≈ 0) count as inside.
        if sd * sc < 0.0 && sd.abs() > 1e-12 {
            return false;
        }
    }
    true
}

/// The resident leaf whose spherical quad contains `d`. Filters to `d`'s cube face first
/// (a boundary point is shared by adjacent faces, so any containing leaf is acceptable),
/// then falls back to all leaves.
pub fn leaf_containing(d: DVec3, leaves: &[QuadNode]) -> Option<QuadNode> {
    let f = face_of(d);
    leaves
        .iter()
        .copied()
        .find(|l| l.face == f && node_contains_dir(*l, d))
        .or_else(|| leaves.iter().copied().find(|l| node_contains_dir(*l, d)))
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    if e1 <= e0 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn angle(a: DVec3, b: DVec3) -> f64 {
    a.dot(b).clamp(-1.0, 1.0).acos()
}

/// One boundary vertex: its sphere direction and its morphed world position.
#[derive(Clone, Copy)]
struct EdgeVert {
    dir: DVec3,
    world: DVec3,
}

/// A built leaf's four boundary edges (each ordered corner→corner) with morph applied,
/// ready for gap comparison. Edges: 0 = v0 (a varies), 1 = u1 (b varies), 2 = v1, 3 = u0.
struct BuiltLeaf {
    node: QuadNode,
    edges: [Vec<EdgeVert>; 4],
}

/// Build a leaf's boundary edges with the CDLOD morph applied for the camera pose.
fn build_leaf(
    field: &SurfaceField,
    node: QuadNode,
    camera_world: DVec3,
    radius: f64,
    res: u32,
) -> BuiltLeaf {
    let chunk = build_chunk(field, node, res);
    let res = res.max(1);
    let res = res + (res & 1); // build_chunk forces even; mirror it for indexing
    let (u0, u1, v0, v1) = node.uv_rect();
    let (start, end) = morph_range(node.level, radius);
    let (start, end) = (start as f64, end as f64);

    let vert = |a: u32, b: u32| -> EdgeVert {
        let idx = (b * (res + 1) + a) as usize;
        let base = chunk.center + Vec3::from_array(chunk.positions[idx]).as_dvec3();
        let target = chunk.center + Vec3::from_array(chunk.morph_targets[idx]).as_dvec3();
        let dist = (camera_world - base).length();
        let f = smoothstep(start, end, dist);
        let world = base + (target - base) * f;
        let uu = lerp(u0, u1, a as f64 / res as f64);
        let vv = lerp(v0, v1, b as f64 / res as f64);
        EdgeVert {
            dir: direction(node.face, uu, vv),
            world,
        }
    };

    let edge_v0: Vec<EdgeVert> = (0..=res).map(|a| vert(a, 0)).collect();
    let edge_u1: Vec<EdgeVert> = (0..=res).map(|b| vert(res, b)).collect();
    let edge_v1: Vec<EdgeVert> = (0..=res).map(|a| vert(a, res)).collect();
    let edge_u0: Vec<EdgeVert> = (0..=res).map(|b| vert(0, b)).collect();
    BuiltLeaf {
        node,
        edges: [edge_v0, edge_u1, edge_v1, edge_u0],
    }
}

/// Evaluate a built leaf's rendered surface at direction `d`, which lies on one of its
/// boundary edges: pick the edge `d` is closest to (perpendicular distance) within span,
/// then linearly interpolate that edge's morphed vertices by angular parameter.
fn surface_on_boundary(built: &BuiltLeaf, d: DVec3) -> Option<DVec3> {
    let mut best: Option<(f64, usize)> = None;
    for (k, edge) in built.edges.iter().enumerate() {
        let a = edge[0].dir;
        let b = edge[edge.len() - 1].dir;
        let total = angle(a, b);
        if total <= 0.0 {
            continue;
        }
        let td = angle(a, d) / total;
        if !(-1e-6..=1.0 + 1e-6).contains(&td) {
            continue;
        }
        let n = a.cross(b);
        if n.length_squared() == 0.0 {
            continue;
        }
        let pd = d.dot(n.normalize()).abs();
        if best.map(|(p, _)| pd < p).unwrap_or(true) {
            best = Some((pd, k));
        }
    }
    let (_, k) = best?;
    let edge = &built.edges[k];
    let a = edge[0].dir;
    let total = angle(a, edge[edge.len() - 1].dir);
    let td = if total > 0.0 {
        angle(a, d) / total
    } else {
        0.0
    };
    // Bracket by per-vertex angular parameter (monotonic along the arc).
    for w in edge.windows(2) {
        let ti = angle(a, w[0].dir) / total;
        let tj = angle(a, w[1].dir) / total;
        if td >= ti - 1e-9 && td <= tj + 1e-9 && tj > ti {
            let f = ((td - ti) / (tj - ti)).clamp(0.0, 1.0);
            return Some(w[0].world.lerp(w[1].world, f));
        }
    }
    // Fallback: nearest endpoint.
    Some(if td <= 0.5 {
        edge[0].world
    } else {
        edge[edge.len() - 1].world
    })
}

/// Scan the resident chunk set for the worst boundary seam at a camera pose.
/// `camera_world` is body-centred. Reproduces the renderer's leaf selection, tessellation
/// (`res`), and CDLOD morph, so the returned gap is what the surface actually renders.
pub fn scan_seams(
    field: &SurfaceField,
    camera_world: DVec3,
    res: u32,
    max_level: u32,
) -> SeamReport {
    let radius = field.radius();
    let leaves = resident_leaves(camera_world, radius, max_level);
    let built: Vec<BuiltLeaf> = leaves
        .iter()
        .map(|&n| build_leaf(field, n, camera_world, radius, res))
        .collect();

    let mut worst_gap = 0.0f64;
    let mut worst_location: Option<SeamLocation> = None;
    let mut boundary_samples = 0usize;

    for bl in &built {
        for edge in &bl.edges {
            // Interior boundary vertices only (skip the 4 corners, where 3–4 leaves meet).
            for ev in &edge[1..edge.len().saturating_sub(1)] {
                let Some(other) = leaves
                    .iter()
                    .copied()
                    .find(|l| *l != bl.node && node_contains_dir(*l, ev.dir))
                else {
                    continue;
                };
                let Some(other_built) = built.iter().find(|b| b.node == other) else {
                    continue;
                };
                let Some(other_world) = surface_on_boundary(other_built, ev.dir) else {
                    continue;
                };
                boundary_samples += 1;
                let gap = (ev.world - other_world).length();
                if gap > worst_gap {
                    worst_gap = gap;
                    let finer = if bl.node.level >= other.level {
                        bl.node
                    } else {
                        other
                    };
                    let skirt_depth = skirt_depth_for(
                        chunk_relief(field, finer, res),
                        finer.edge_len(radius),
                        radius,
                    );
                    worst_location = Some(SeamLocation {
                        node: bl.node,
                        neighbour_level: other.level,
                        skirt_depth,
                    });
                }
            }
        }
    }

    SeamReport {
        worst_gap,
        worst_location,
        leaf_count: leaves.len(),
        boundary_samples,
    }
}

/// [`scan_seams`] with the renderer's default resolution and max level.
pub fn scan_seams_default(field: &SurfaceField, camera_world: DVec3) -> SeamReport {
    scan_seams(field, camera_world, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVEL)
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: f64 = 730_000.0;

    fn field() -> SurfaceField {
        SurfaceField::new(7, R)
    }

    // Camera at radial distance `alt` above the surface point in direction `d`.
    fn cam(d: DVec3, alt: f64) -> DVec3 {
        d.normalize() * (R + alt)
    }

    #[test]
    fn resident_leaves_split_with_proximity() {
        let f = field();
        // Far away: the six face roots.
        let far = resident_leaves(cam(DVec3::Z, R * 10.0), f.radius(), DEFAULT_MAX_LEVEL);
        assert_eq!(far.len(), 6, "far camera keeps the six roots");
        // Close: many more leaves, and they partition (no duplicates).
        let near = resident_leaves(cam(DVec3::Z, 20_000.0), f.radius(), 8);
        assert!(near.len() > 6, "near camera subdivides");
    }

    #[test]
    fn point_location_finds_the_owning_leaf() {
        let f = field();
        let leaves = resident_leaves(cam(DVec3::Z, 50_000.0), f.radius(), 6);
        for &l in &leaves {
            // A node's own centre locates back to itself.
            assert_eq!(leaf_containing(l.center_dir(), &leaves), Some(l));
        }
    }

    #[test]
    fn far_view_is_watertight_including_cross_face() {
        // Six root leaves meeting at cube edges: same level, continuous field ⇒ the
        // cross-face boundaries must agree (this exercises point-location + neighbour
        // edge evaluation across faces).
        let f = field();
        let report = scan_seams(
            &f,
            cam(DVec3::new(1.0, 1.0, 1.0), R * 8.0),
            DEFAULT_RESOLUTION,
            3,
        );
        assert_eq!(report.leaf_count, 6);
        assert!(
            report.boundary_samples > 0,
            "cross-face boundaries were compared"
        );
        // Same-level neighbours share vertices; gaps are only f32 + interpolation noise.
        assert!(
            report.worst_gap < 5.0,
            "far same-level view should be ~watertight, got {} m",
            report.worst_gap
        );
    }

    #[test]
    fn scan_is_finite_and_deterministic() {
        let f = field();
        let c = cam(DVec3::new(0.2, 0.1, 1.0), 30_000.0);
        let a = scan_seams(&f, c, DEFAULT_RESOLUTION, 9);
        let b = scan_seams(&f, c, DEFAULT_RESOLUTION, 9);
        assert!(a.worst_gap.is_finite());
        assert_eq!(a.worst_gap, b.worst_gap, "scan is deterministic");
        assert!(a.leaf_count > 6 && a.boundary_samples > 0);
    }

    #[test]
    fn lod_boundary_is_seamless_at_low_altitude() {
        // Regression guard (WI 783): this low pose with deep T-junctions had a ~3248 m
        // boundary tear before the quiet-zone morph ramp; it is now down to the coarsest
        // level's scan floor. The residual is scan *interpolation* error at level-1 edges
        // (~500 km): O(1/res²) — ~150 m at res 24, ~40 m at res 48 — not a render step, so
        // the tolerance sits above that floor and far below the pre-fix tear. If this ever
        // fails, the CDLOD morph ramp regressed.
        const SEAM_TOL: f64 = 250.0; // metres; > ~150 m coarse-level scan floor, ≪ 3248 m tear
        let f = field();
        let report = scan_seams(
            &f,
            cam(DVec3::new(0.15, 0.05, 1.0), 8_000.0),
            DEFAULT_RESOLUTION,
            12,
        );
        assert!(report.leaf_count > 20, "low pose subdivides deeply");
        assert!(
            report.worst_gap < SEAM_TOL,
            "LOD boundary should be seamless (WI 783); worst_gap = {} m exceeds {} m",
            report.worst_gap,
            SEAM_TOL
        );
    }
}

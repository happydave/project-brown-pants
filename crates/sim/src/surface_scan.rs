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
//! The scan reproduces exactly what the WI 780/795 shader draws: a vertex at base-relative
//! position `b` with morph target `t` renders at `b + f·(t − b)`, where `f` is the
//! distance smoothstep over `morph_range(level, radius)` passed through the WI 795
//! **edge weld** ([`weld_factor`](crate::surface_mesh::weld_factor), driven by
//! [`boundary_config`]). Skirts are a *cover* for residual gaps, not surface, so the
//! primary metric is the surface gap; the report also notes the finer chunk's skirt
//! depth for context.
//!
//! WI 795 adds the exact **T-junction oracle** ([`scan_tjunctions`]): the WI 791
//! post-mortem proved this fuzzy scan under-measures the visible artifact (its angular
//! interpolation has a ~150 m floor at coarse levels) and that a co-located-vertex
//! metric is blind to it (even vertices morph to themselves), so the oracle samples
//! the fine chunk's **odd** boundary vertices against the coarser neighbour's rendered
//! chord, in metres *and* grazing screen-space pixels, plus the rendered-normal
//! mismatch and 2:1 violations; [`scan_same_level_exact`] guards that the weld leaves
//! same-level edges bit-near-identical.

use crate::surface_field::SurfaceField;
use crate::surface_mesh::{
    build_chunk, chunk_relief, direction, morph_range, should_split, skirt_depth_for, weld_factor,
    CubeFace, QuadNode, DEFAULT_MAX_LEVEL, DEFAULT_RESOLUTION,
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
pub fn resident_leaves(field: &SurfaceField, camera_world: DVec3, max_level: u32) -> Vec<QuadNode> {
    let mut leaves = Vec::new();
    let mut stack: Vec<QuadNode> = QuadNode::roots().to_vec();
    while let Some(node) = stack.pop() {
        if should_split(field, node, camera_world, max_level) {
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

/// Build a leaf's boundary edges with the CDLOD morph applied for the camera pose —
/// including the WI 795 edge weld from `config`, since this models what the shader
/// actually draws.
fn build_leaf(
    field: &SurfaceField,
    node: QuadNode,
    camera_world: DVec3,
    radius: f64,
    config: (u8, u8),
    res: u32,
) -> BuiltLeaf {
    let chunk = build_chunk(field, node, res);
    let res = res.max(1);
    let res = res + (res & 1); // build_chunk forces even; mirror it for indexing
    let (u0, u1, v0, v1) = node.uv_rect();
    let (start, end) = morph_range(node.level, radius);
    let (start, end) = (start as f64, end as f64);
    let (mask_coarser, mask_finer) = config;

    let vert = |a: u32, b: u32| -> EdgeVert {
        let idx = (b * (res + 1) + a) as usize;
        let base = chunk.center + Vec3::from_array(chunk.positions[idx]).as_dvec3();
        let target = chunk.center + Vec3::from_array(chunk.morph_targets[idx]).as_dvec3();
        let dist = (camera_world - base).length();
        let f = weld_factor(
            smoothstep(start, end, dist),
            a,
            b,
            res,
            mask_coarser,
            mask_finer,
        );
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
    let leaves = resident_leaves(field, camera_world, max_level);
    let leaf_set: std::collections::HashSet<QuadNode> = leaves.iter().copied().collect();
    let built: Vec<BuiltLeaf> = leaves
        .iter()
        .map(|&n| {
            let config = boundary_config(n, &leaf_set, max_level);
            build_leaf(field, n, camera_world, radius, config, res)
        })
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

// ---------------------------------------------------------------------------
// T-junction oracle (WI 795)
// ---------------------------------------------------------------------------
//
// The WI 791 post-mortem established that the visible LOD "zippering" lives at the
// fine chunk's ODD boundary vertices (the ones between a coarse neighbour's
// vertices) and that both prior metrics missed it: `scan_seams` interpolates by
// angular parameter (resolution-noisy, ~150 m floor at coarse levels) and WI 791's
// co-located-vertex metric compared only even vertices, which morph to themselves
// and match trivially. This oracle measures the artifact itself: each odd boundary
// vertex adjacent to a COARSER leaf, as rendered (the shader's per-vertex morph),
// against the coarse neighbour's rendered edge chord at the same direction — exact,
// because the rendered mesh is linear between vertices — plus the grazing
// screen-space error (Ulrich chunked-LOD pixels) and the shading mismatch (the
// angle between the two sides' rendered normals, since the position morph does not
// morph normals).

/// Scan parameters for [`scan_tjunctions`]. The projection defaults mirror the
/// `-- surface` camera (Bevy `PerspectiveProjection::default()`: vertical fov π/4)
/// at a 1080 px reference viewport, so `pixels` reads as "on a 1080p screen".
#[derive(Clone, Copy, Debug)]
pub struct TJunctionScanParams {
    /// Per-chunk grid resolution (quads per side); the renderer's default.
    pub res: u32,
    /// Maximum quadtree depth; the renderer's default.
    pub max_level: u32,
    /// When set, every vertex's morph factor is forced to this value instead of the
    /// distance smoothstep — the oracle-validation hook (0 ⇒ un-morphed surface
    /// grid, 1 ⇒ fully morphed onto the parent grid). `None` ⇒ render behaviour.
    pub factor_override: Option<f64>,
    /// Reference viewport height, pixels, for the screen-space error.
    pub viewport_h_px: f64,
    /// Vertical field of view, radians, for the screen-space error.
    pub fov_y_rad: f64,
}

impl Default for TJunctionScanParams {
    fn default() -> Self {
        Self {
            res: DEFAULT_RESOLUTION,
            max_level: DEFAULT_MAX_LEVEL,
            factor_override: None,
            viewport_h_px: 1080.0,
            fov_y_rad: std::f64::consts::FRAC_PI_4,
        }
    }
}

/// One measured odd-vertex sample at a fine/coarser boundary.
#[derive(Clone, Copy, Debug)]
pub struct TJunctionSample {
    /// The finer leaf whose odd boundary vertex was measured.
    pub fine: QuadNode,
    /// The coarser neighbour on the far side.
    pub coarse: QuadNode,
    /// The sample's sphere direction (the odd vertex's direction).
    pub dir: DVec3,
    /// Rendered gap between the fine odd vertex and the coarse rendered chord, metres.
    pub gap_m: f64,
    /// The gap as screen-space pixels at the scan camera (Ulrich chunked-LOD error).
    pub pixels: f64,
    /// The fine vertex's morph factor as rendered (1.0 = fully morphed). A gap with
    /// factor < 1 is a ramp-not-completed violation; a gap at factor 1 is structural
    /// (e.g. a >1 level jump, whose parent-grid target cannot match the coarser chord).
    pub fine_factor: f64,
    /// Angle between the fine vertex's rendered normal and the coarse side's
    /// interpolated rendered normal, degrees — the shading-seam metric.
    pub normal_mismatch_deg: f64,
    /// Level difference `fine.level − coarse.level` (2:1-balanced ⇒ always 1).
    pub level_delta: u32,
}

/// The result of a T-junction scan. `samples` empty means the pose realized no
/// fine/coarser boundaries (e.g. far orbit, six equal roots) — "no samples", which is
/// distinct from "no gap".
#[derive(Clone, Debug, Default)]
pub struct TJunctionReport {
    /// Every odd-vertex sample at a fine/coarser boundary, unordered.
    pub samples: Vec<TJunctionSample>,
    /// Resident leaves at the pose (context for chunk-density accounting).
    pub leaf_count: usize,
}

impl TJunctionReport {
    /// The sample with the largest metric gap, if any.
    pub fn worst_gap(&self) -> Option<&TJunctionSample> {
        self.samples
            .iter()
            .max_by(|a, b| a.gap_m.total_cmp(&b.gap_m))
    }

    /// The sample with the largest screen-space error, if any.
    pub fn worst_pixels(&self) -> Option<&TJunctionSample> {
        self.samples
            .iter()
            .max_by(|a, b| a.pixels.total_cmp(&b.pixels))
    }

    /// The sample with the largest shading mismatch, if any.
    pub fn worst_normal_mismatch(&self) -> Option<&TJunctionSample> {
        self.samples
            .iter()
            .max_by(|a, b| a.normal_mismatch_deg.total_cmp(&b.normal_mismatch_deg))
    }

    /// The largest fine/coarse level jump seen (2 or more = 2:1 balance violation,
    /// a violation class of its own: morph-to-parent only targets one level up).
    pub fn max_level_delta(&self) -> u32 {
        self.samples
            .iter()
            .map(|s| s.level_delta)
            .max()
            .unwrap_or(0)
    }

    /// Number of samples with a level jump of 2 or more.
    pub fn unbalanced_samples(&self) -> usize {
        self.samples.iter().filter(|s| s.level_delta > 1).count()
    }
}

/// The realized neighbour relation of a leaf against a resident leaf set: bits 0–3
/// of `coarser`/`finer` set when edge `k` borders a coarser/finer leaf (edge order
/// v0, u1, v1, u0), bits 4–7 when the **diagonal corner** neighbour is coarser/finer
/// (corner order c00, c10, c11, c01) — matching
/// [`weld_factor`](crate::surface_mesh::weld_factor)'s distance convention. The
/// neighbour is found by point-locating a probe direction just outside each edge's
/// midpoint (or just beyond each corner, diagonally) — face-agnostic (a probe past a
/// cube edge simply lands on the adjacent face's leaf), and the same source of truth
/// for the renderer and the scans.
pub fn boundary_config(
    node: QuadNode,
    leaves: &std::collections::HashSet<QuadNode>,
    max_level: u32,
) -> (u8, u8) {
    let (u0, u1, v0, v1) = node.uv_rect();
    let span = u1 - u0;
    let off = 0.25 * span;
    let (mu, mv) = (0.5 * (u0 + u1), 0.5 * (v0 + v1));
    let probes = [
        (mu, v0 - off),
        (u1 + off, mv),
        (mu, v1 + off),
        (u0 - off, mv),
        (u0 - off, v0 - off),
        (u1 + off, v0 - off),
        (u1 + off, v1 + off),
        (u0 - off, v1 + off),
    ];
    let (mut coarser, mut finer) = (0u8, 0u8);
    for (k, &(pu, pv)) in probes.iter().enumerate() {
        let d = direction(node.face, pu, pv);
        let Some(n) = leaf_containing_in(d, leaves, max_level) else {
            continue;
        };
        if n == node {
            continue; // probe fell short of the boundary (shouldn't happen)
        }
        if n.level < node.level {
            coarser |= 1 << k;
        } else if n.level > node.level {
            finer |= 1 << k;
        }
    }
    (coarser, finer)
}

/// The leaf in `leaves` containing direction `d`, found by quadtree **descent**
/// (root → child containing `d` → … until a member of `leaves` is reached) — O(depth)
/// instead of a linear scan, cheap enough for the renderer's per-frame weld-config
/// refresh. Returns `None` if the descent exits `max_level` without hitting a leaf
/// (a hole in the set — e.g. a stale live set mid-stream; callers treat it as
/// "no relation").
pub fn leaf_containing_in(
    d: DVec3,
    leaves: &std::collections::HashSet<QuadNode>,
    max_level: u32,
) -> Option<QuadNode> {
    let mut node = QuadNode::roots()
        .into_iter()
        .find(|r| node_contains_dir(*r, d))?;
    loop {
        if leaves.contains(&node) {
            return Some(node);
        }
        if node.level >= max_level {
            return None;
        }
        node = node
            .children()
            .into_iter()
            .find(|c| node_contains_dir(*c, d))?;
    }
}

/// A built leaf's boundary, with rendered positions *and* normals per edge vertex,
/// in edge order 0 = v0 (a varies), 1 = u1 (b varies), 2 = v1, 3 = u0.
struct RenderedBoundary {
    edges: [Vec<RenderedVert>; 4],
}

#[derive(Clone, Copy)]
struct RenderedVert {
    dir: DVec3,
    world: DVec3,
    normal: DVec3,
    factor: f64,
}

/// Build a leaf's rendered boundary (morph applied per vertex, as the shader does,
/// including the WI 795 edge weld from the leaf's realized neighbour config).
fn rendered_boundary(
    field: &SurfaceField,
    node: QuadNode,
    camera_world: DVec3,
    radius: f64,
    config: (u8, u8),
    params: &TJunctionScanParams,
) -> RenderedBoundary {
    let chunk = build_chunk(field, node, params.res);
    let res = params.res.max(1);
    let res = res + (res & 1); // mirror build_chunk's forced-even resolution
    let (u0, u1, v0, v1) = node.uv_rect();
    let (start, end) = morph_range(node.level, radius);
    let (start, end) = (start as f64, end as f64);
    let (mask_coarser, mask_finer) = config;

    let vert = |a: u32, b: u32| -> RenderedVert {
        let idx = (b * (res + 1) + a) as usize;
        let base = chunk.center + Vec3::from_array(chunk.positions[idx]).as_dvec3();
        let target = chunk.center + Vec3::from_array(chunk.morph_targets[idx]).as_dvec3();
        // The override bypasses the weld too: it exists to validate the oracle
        // against raw (un-welded) geometry — forcing 0 must expose the bare
        // odd-vertex deviation regardless of the neighbour config.
        let factor = match params.factor_override {
            Some(f) => f,
            None => weld_factor(
                smoothstep(start, end, (camera_world - base).length()),
                a,
                b,
                res,
                mask_coarser,
                mask_finer,
            ),
        };
        RenderedVert {
            dir: direction(
                node.face,
                lerp(u0, u1, a as f64 / res as f64),
                lerp(v0, v1, b as f64 / res as f64),
            ),
            world: base + (target - base) * factor,
            normal: Vec3::from_array(chunk.normals[idx]).as_dvec3(),
            factor,
        }
    };

    RenderedBoundary {
        edges: [
            (0..=res).map(|a| vert(a, 0)).collect(),
            (0..=res).map(|b| vert(res, b)).collect(),
            (0..=res).map(|a| vert(a, res)).collect(),
            (0..=res).map(|b| vert(0, b)).collect(),
        ],
    }
}

/// Where the body-centre ray through `dir` pierces the rendered boundary of a leaf:
/// the exact point on the leaf's boundary segment chain (the rendered mesh is linear
/// between vertices), plus the rasterizer's interpolated normal there. Returns `None`
/// if `dir` does not lie on this leaf's boundary.
fn rendered_boundary_point(boundary: &RenderedBoundary, dir: DVec3) -> Option<(DVec3, DVec3)> {
    // Pick the boundary edge whose great circle `dir` lies on (smallest perpendicular
    // distance, within the edge's angular span) — same selection as `scan_seams`.
    let mut best: Option<(f64, usize)> = None;
    for (k, edge) in boundary.edges.iter().enumerate() {
        let a = edge[0].dir;
        let b = edge[edge.len() - 1].dir;
        let total = angle(a, b);
        if total <= 0.0 {
            continue;
        }
        let td = angle(a, dir) / total;
        if !(-1e-6..=1.0 + 1e-6).contains(&td) {
            continue;
        }
        let n = a.cross(b);
        if n.length_squared() == 0.0 {
            continue;
        }
        let pd = dir.dot(n.normalize()).abs();
        if best.map(|(p, _)| pd < p).unwrap_or(true) {
            best = Some((pd, k));
        }
    }
    let (_, k) = best?;
    let edge = &boundary.edges[k];

    // Exact ray–segment intersection per consecutive rendered pair: the point
    // `P_i + s·(P_j − P_i)` that is radial along `dir` (least-squares s from the
    // cross-product condition), accepted when s ∈ [0, 1] with a small tolerance.
    // Among accepted candidates keep the most collinear one (belt and braces at
    // shared vertices, where two segments both accept with s≈1/s≈0).
    let mut hit: Option<(f64, DVec3, DVec3)> = None; // (residual, point, normal)
    for w in edge.windows(2) {
        let (p_i, p_j) = (w[0].world, w[1].world);
        let d = p_j - p_i;
        let a_c = p_i.cross(dir);
        let b_c = d.cross(dir);
        let denom = b_c.length_squared();
        let s = if denom > 0.0 {
            -a_c.dot(b_c) / denom
        } else {
            0.0
        };
        if !(-1e-3..=1.0 + 1e-3).contains(&s) {
            continue;
        }
        let s = s.clamp(0.0, 1.0);
        let q = p_i + d * s;
        let residual = q.normalize_or_zero().cross(dir).length();
        if hit.map(|(r, _, _)| residual < r).unwrap_or(true) {
            let n = (w[0].normal.lerp(w[1].normal, s)).normalize_or_zero();
            hit = Some((residual, q, n));
        }
    }
    hit.map(|(_, q, n)| (q, n))
}

/// Scan every fine-chunk **odd** boundary vertex that borders a **coarser** resident
/// leaf, comparing its rendered position/normal against the coarse neighbour's
/// rendered edge chord at the same direction. `camera_world` is body-centred.
pub fn scan_tjunctions(
    field: &SurfaceField,
    camera_world: DVec3,
    params: TJunctionScanParams,
) -> TJunctionReport {
    let radius = field.radius();
    let leaves = resident_leaves(field, camera_world, params.max_level);
    let res = params.res.max(1);
    let res = res + (res & 1);

    let leaf_set: std::collections::HashSet<QuadNode> = leaves.iter().copied().collect();

    // Rendered boundaries built lazily, only for leaves that participate.
    let mut boundaries: std::collections::HashMap<QuadNode, RenderedBoundary> =
        std::collections::HashMap::new();
    let build = |boundaries: &mut std::collections::HashMap<QuadNode, RenderedBoundary>,
                 node: QuadNode| {
        boundaries.entry(node).or_insert_with(|| {
            let config = boundary_config(node, &leaf_set, params.max_level);
            rendered_boundary(field, node, camera_world, radius, config, &params)
        });
    };

    let pixel_scale = params.viewport_h_px / (2.0 * (params.fov_y_rad * 0.5).tan());

    let mut samples = Vec::new();
    for &fine in &leaves {
        // A leaf with no coarser neighbour anywhere can be skipped cheaply only after
        // probing its edges, so probe odd vertices directly (they are the metric).
        let mut fine_boundary_built = false;
        for edge_k in 0..4usize {
            for k in (1..res).step_by(2) {
                // Odd vertex (a, b) grid index on this edge.
                let (a, b) = match edge_k {
                    0 => (k, 0),
                    1 => (res, k),
                    2 => (k, res),
                    _ => (0, k),
                };
                let (u0, u1, v0, v1) = fine.uv_rect();
                let dir = direction(
                    fine.face,
                    lerp(u0, u1, a as f64 / res as f64),
                    lerp(v0, v1, b as f64 / res as f64),
                );
                // The neighbour on the far side of this boundary direction.
                let Some(coarse) = leaves
                    .iter()
                    .copied()
                    .find(|l| *l != fine && node_contains_dir(*l, dir))
                else {
                    continue;
                };
                if coarse.level >= fine.level {
                    continue; // same-level (matched by construction) or finer (their side)
                }
                if !fine_boundary_built {
                    build(&mut boundaries, fine);
                    fine_boundary_built = true;
                }
                build(&mut boundaries, coarse);
                let fine_b = &boundaries[&fine];
                let fv = fine_b.edges[edge_k][k as usize];
                let Some((coarse_point, coarse_normal)) =
                    rendered_boundary_point(&boundaries[&coarse], dir)
                else {
                    continue;
                };
                let gap_m = (fv.world - coarse_point).length();
                let dist = (camera_world - fv.world).length().max(1.0);
                let pixels = gap_m / dist * pixel_scale;
                let normal_mismatch_deg = angle(
                    fv.normal.normalize_or_zero(),
                    coarse_normal.normalize_or_zero(),
                )
                .to_degrees();
                samples.push(TJunctionSample {
                    fine,
                    coarse,
                    dir,
                    gap_m,
                    pixels,
                    fine_factor: fv.factor,
                    normal_mismatch_deg,
                    level_delta: fine.level - coarse.level,
                });
            }
        }
    }

    TJunctionReport {
        samples,
        leaf_count: leaves.len(),
    }
}

/// Exact same-level boundary check: for every pair of **same-level** neighbouring
/// leaves, compare their co-located rendered boundary vertices (they sample the
/// same directions, so the comparison is exact — no interpolation). Guards the
/// WI 795 edge weld: the weld must leave same-level edges matched (the falloff
/// band of a flagged edge crossing a corner is the one place the two chunks'
/// factors can legitimately differ; this measures how much that costs). Returns
/// the worst rendered mismatch in metres, with the number of vertex pairs compared.
pub fn scan_same_level_exact(
    field: &SurfaceField,
    camera_world: DVec3,
    params: TJunctionScanParams,
) -> (f64, usize) {
    let radius = field.radius();
    let leaves = resident_leaves(field, camera_world, params.max_level);
    let res = params.res.max(1);
    let res = res + (res & 1);

    let leaf_set: std::collections::HashSet<QuadNode> = leaves.iter().copied().collect();

    let mut boundaries: std::collections::HashMap<QuadNode, RenderedBoundary> =
        std::collections::HashMap::new();
    let build = |boundaries: &mut std::collections::HashMap<QuadNode, RenderedBoundary>,
                 node: QuadNode| {
        boundaries.entry(node).or_insert_with(|| {
            let config = boundary_config(node, &leaf_set, params.max_level);
            rendered_boundary(field, node, camera_world, radius, config, &params)
        });
    };

    let mut worst = 0.0f64;
    let mut pairs = 0usize;
    for &node in &leaves {
        for edge_k in 0..4usize {
            // Interior boundary vertices (corners belong to 3–4 leaves; skip them).
            for k in 1..res {
                let (a, b) = match edge_k {
                    0 => (k, 0),
                    1 => (res, k),
                    2 => (k, res),
                    _ => (0, k),
                };
                let (u0, u1, v0, v1) = node.uv_rect();
                let dir = direction(
                    node.face,
                    lerp(u0, u1, a as f64 / res as f64),
                    lerp(v0, v1, b as f64 / res as f64),
                );
                let Some(other) = leaves
                    .iter()
                    .copied()
                    .find(|l| *l != node && node_contains_dir(*l, dir))
                else {
                    continue;
                };
                if other.level != node.level {
                    continue; // cross-level boundaries are the T-junction scan's job
                }
                // (Each pair is visited from both sides; the worst is unaffected.)
                build(&mut boundaries, node);
                build(&mut boundaries, other);
                let mine = boundaries[&node].edges[edge_k][k as usize];
                // The co-located vertex on the neighbour: same direction (ULP-scale
                // difference at most), found by nearest angle on its edges.
                let theirs = boundaries[&other]
                    .edges
                    .iter()
                    .flatten()
                    .min_by(|x, y| angle(x.dir, dir).total_cmp(&angle(y.dir, dir)))
                    .copied();
                let Some(theirs) = theirs else { continue };
                if angle(theirs.dir, dir) > 1e-9 {
                    continue; // not co-located (e.g. probe crossed a face corner)
                }
                pairs += 1;
                worst = worst.max((mine.world - theirs.world).length());
            }
        }
    }
    (worst, pairs)
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
        let far = resident_leaves(&f, cam(DVec3::Z, R * 10.0), DEFAULT_MAX_LEVEL);
        assert_eq!(far.len(), 6, "far camera keeps the six roots");
        // Close: many more leaves, and they partition (no duplicates).
        let near = resident_leaves(&f, cam(DVec3::Z, 20_000.0), 8);
        assert!(near.len() > 6, "near camera subdivides");
    }

    #[test]
    fn point_location_finds_the_owning_leaf() {
        let f = field();
        let leaves = resident_leaves(&f, cam(DVec3::Z, 50_000.0), 6);
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

    /// WI 791's known-bad grazing pose: the camera where the zippering rings stood
    /// while the blind (even-vertex) oracle read ~2 m. Every WI 795 validation
    /// anchors here.
    fn known_bad_camera() -> DVec3 {
        cam(DVec3::new(0.3, -0.9, 0.2), 1_500.0)
    }

    #[test]
    fn tjunction_known_bad_pose_regression_guard() {
        // WI 795 AC-1 + the post-fix regression guard. Pre-fix (0.1.182 geometry)
        // this pose read **833.34 m / 185.08 px** worst with 4 samples above 1 px
        // (recorded in the WI 795 code.md/test.md); the blind WI 791 oracle read
        // ~2 m at the same pose. With the surface-consistent nearest-point
        // selection + the corner-aware edge weld, every fine/coarser boundary
        // sample must render **sub-visible** on the 1080p reference projection.
        let f = field();
        let report = scan_tjunctions(&f, known_bad_camera(), TJunctionScanParams::default());
        assert!(
            !report.samples.is_empty(),
            "known-bad pose must realize fine/coarse boundaries (no samples = blind oracle)"
        );
        assert_eq!(
            report.unbalanced_samples(),
            0,
            "selection stays 2:1-balanced at the known-bad pose"
        );
        let worst = report.worst_pixels().expect("samples exist");
        println!(
            "known-bad pose: {} samples, worst {:.2} px ({:.2} m, level {}->{})",
            report.samples.len(),
            worst.pixels,
            worst.gap_m,
            worst.fine.level,
            worst.coarse.level
        );
        assert!(
            worst.pixels < 1.0,
            "worst boundary step {:.2} px must stay sub-visible (pre-fix: 185.08 px)",
            worst.pixels
        );
    }

    #[test]
    fn tjunction_oracle_is_sensitive_to_odd_vertices() {
        // Oracle validation (WI 795 Phase A, the WI 791 blind-spot proof): with the
        // morph forced OFF (factor 0, bypassing the weld), the raw geometry has a
        // real odd-vertex step that the oracle must read — and the reading must
        // match an independent computation of that step (direct field sampling:
        // odd-direction surface point vs the midpoint of its two even-neighbour
        // surface points, which for a one-level jump is the coarse rendered chord).
        let f = field();
        let raw = TJunctionScanParams {
            factor_override: Some(0.0),
            max_level: 12,
            ..Default::default()
        };
        let report = scan_tjunctions(&f, known_bad_camera(), raw);
        // Restrict to fine levels, where the chord midpoint and the ray-intersection
        // point agree to well under a percent (long coarse edges bend more).
        let w = report
            .samples
            .iter()
            .filter(|s| s.fine.level >= 8 && s.level_delta == 1)
            .max_by(|a, b| a.gap_m.total_cmp(&b.gap_m))
            .expect("fine-level fine/coarse samples exist at the known-bad pose");
        // Magnitude floor re-anchored by WI 866: the original 10 m gate rode on a
        // ~1 360 m raw step that was really a crater-cliff *field discontinuity*
        // (WI 866's defect), not LOD geometry. On the continuous field the raw
        // fine-level odd-vertex step at this pose is ~1.6 m — still well above the
        // oracle's numeric noise, and the relative-agreement check below is the
        // real sensitivity assertion.
        assert!(
            w.gap_m > 1.0,
            "un-morphed odd-vertex step must be non-trivial; got {:.2} m",
            w.gap_m
        );
        // Independent recompute: find the odd vertex's grid position on its edge,
        // then measure |surface(dir) − midpoint(even-neighbour surface points)|.
        let res = raw.res + (raw.res & 1);
        let (u0, u1, v0, v1) = w.fine.uv_rect();
        let mut expected = None;
        for edge_k in 0..4u32 {
            for k in (1..res).step_by(2) {
                let (a, b) = match edge_k {
                    0 => (k, 0),
                    1 => (res, k),
                    2 => (k, res),
                    _ => (0, k),
                };
                let at = |a: u32, b: u32| {
                    direction(
                        w.fine.face,
                        lerp(u0, u1, a as f64 / res as f64),
                        lerp(v0, v1, b as f64 / res as f64),
                    )
                };
                if angle(at(a, b), w.dir) > 1e-12 {
                    continue;
                }
                // Even neighbours along the edge direction.
                let (pa, pb, na, nb) = match edge_k {
                    0 => (a - 1, 0, a + 1, 0),
                    1 => (res, b - 1, res, b + 1),
                    2 => (a - 1, res, a + 1, res),
                    _ => (0, b - 1, 0, b + 1),
                };
                let surf = |d: DVec3| d * (f.radius() + f.elevation(d));
                let v = surf(at(a, b));
                let chord = 0.5 * (surf(at(pa, pb)) + surf(at(na, nb)));
                expected = Some((v - chord).length());
            }
        }
        let expected = expected.expect("worst sample's odd vertex found on its edge");
        let rel = (w.gap_m - expected).abs() / expected.max(1.0);
        println!(
            "sensitivity: oracle {:.2} m vs independent {:.2} m (rel {:.4})",
            w.gap_m, expected, rel
        );
        assert!(
            rel < 0.02,
            "oracle gap {:.2} m must match the independent field computation {:.2} m",
            w.gap_m,
            expected
        );
    }

    #[test]
    fn tjunction_pose_sweep_stays_subvisible() {
        // The WI 791 6-pose sweep through the corrected oracle, welded geometry.
        // Every pose — including the two grazing/steep poses — must render every
        // fine/coarser boundary below 1 px on the 1080p reference projection.
        let f = field();
        let poses = [
            (DVec3::new(0.15, 0.05, 1.0), 8_000.0),
            (DVec3::new(1.0, 0.0, 0.0), 3_000.0),
            (DVec3::new(1.0, 1.0, 1.0), 20_000.0),
            (DVec3::new(-0.5, 0.2, 0.8), 50_000.0),
            (DVec3::new(0.7, 0.7, 0.05), 500.0),
            (DVec3::new(0.3, -0.9, 0.2), 1_500.0),
        ];
        for (dir, alt) in poses {
            let report = scan_tjunctions(&f, cam(dir, alt), TJunctionScanParams::default());
            let px = report.worst_pixels().map(|w| w.pixels).unwrap_or(0.0);
            let unbalanced = report.unbalanced_samples();
            println!(
                "pose {dir:?} @ {alt} m: {} samples, worst {px:.3} px, unbalanced {unbalanced}",
                report.samples.len()
            );
            assert!(
                px < 1.0,
                "pose {dir:?} @ {alt} m: worst step {px:.2} px must stay sub-visible"
            );
            assert_eq!(unbalanced, 0, "pose {dir:?} @ {alt} m: 2:1 balance holds");
        }
    }

    #[test]
    fn same_level_edges_match_exactly_under_weld() {
        // The weld must not break same-level edges (the WI 795 corner hazard: a
        // falloff band crossing a corner made two same-level chunks disagree by up
        // to ~1.1 km before the corner-aware masks). Exact co-located comparison.
        let f = field();
        for (dir, alt) in [
            (DVec3::new(0.3, -0.9, 0.2), 1_500.0),
            (DVec3::new(0.15, 0.05, 1.0), 8_000.0),
        ] {
            let params = TJunctionScanParams {
                max_level: 12,
                ..Default::default()
            };
            let (worst, pairs) = scan_same_level_exact(&f, cam(dir, alt), params);
            println!("same-level exact {dir:?} @ {alt} m: worst {worst:.3} m over {pairs} pairs");
            assert!(pairs > 1_000, "pose realizes plenty of same-level pairs");
            assert!(
                worst < 0.1,
                "same-level shared edges must render identically (worst {worst:.3} m)"
            );
        }
    }

    #[test]
    fn descent_lookup_agrees_with_linear_point_location() {
        // `leaf_containing_in` (quadtree descent, the renderer's per-frame path)
        // must find the same leaf the linear scan finds.
        let f = field();
        let leaves = resident_leaves(&f, cam(DVec3::new(0.2, 0.5, 0.8), 40_000.0), 9);
        let set: std::collections::HashSet<QuadNode> = leaves.iter().copied().collect();
        for &l in &leaves {
            assert_eq!(
                leaf_containing_in(l.center_dir(), &set, 9),
                Some(l),
                "descent finds each leaf from its own centre"
            );
        }
    }

    #[test]
    fn no_samples_is_distinct_from_no_gap() {
        // Far orbit: six equal roots, no cross-level boundary anywhere — the report
        // must say "no samples" (empty), not a misleading zero-gap pass.
        let f = field();
        let report = scan_tjunctions(&f, cam(DVec3::Z, R * 10.0), TJunctionScanParams::default());
        assert_eq!(report.leaf_count, 6);
        assert!(report.samples.is_empty());
        assert!(report.worst_gap().is_none());
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

//! Headless surface **geometry** for the streaming renderer (WI 764).
//!
//! The renderer (app-side, WI 764) tessellates WI 763's analytic
//! [`SurfaceField`](crate::surface_field::SurfaceField) on a **spherified-cube
//! quadtree**. This module is the *headless* half of that split (mirroring
//! `voxel_mesh` headless / `voxel_skin` app): pure, engine-free geometry —
//! spherified-cube mapping, quadtree nodes, per-chunk vertex/normal/UV/index
//! **buffers** (plain `Vec`s, glam only), crack-hiding skirts, and the LOD split
//! criterion. The app converts the buffers into a Bevy `Mesh` and owns the task
//! pool, entities, camera, and gizmos.
//!
//! **Seamless + crack-free.**
//! - The spherified-cube map is continuous across cube-face edges (a shared edge
//!   maps to the same locus of directions from either face), and the field is
//!   sampled at the resulting 3D direction — so there is no parameterization seam.
//! - Chunks at differing LOD meet without holes because each chunk carries a
//!   **skirt**: its border ring is duplicated and pushed radially inward by a depth
//!   sized to the chunk's own **relief** (max−min elevation). A boundary gap is
//!   bounded by the relief the chunks span, so a relief-sized skirt covers it while
//!   staying buried under the terrain — sizing it to the node's *width* instead grew
//!   kilometre-tall walls on coarse chunks that showed as a "waffle" at grazing
//!   angles (WI 773). The traversal need not keep the quadtree 2:1-balanced.
//!
//! **Precision.** A chunk's vertex positions are `f32` **relative to the node's
//! centre world point** (returned separately as `f64`), so per-vertex values stay
//! small regardless of body radius; absolute placement is the floating origin's job.

use crate::surface_field::SurfaceField;
use glam::{DVec3, Vec2, Vec3};

/// What a chunk's per-vertex colors show (WI 869). `Biome` is the shipping
/// look — the weight-blended biome tint. The other views are the **debug
/// overlay family**: the discrete dominant-biome view (the one visual consumer
/// allowed the dominant accessor — its job is to show the classification,
/// artifacts and all) and the raw climate-field ramps. All color math lives
/// here, headless — the render shader only passes the attribute through (the
/// WI 795 one-side-only lockstep rule).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SurfaceView {
    /// Weight-blended biome tint (the shipping phase-1 look).
    #[default]
    Biome,
    /// Discrete dominant-biome coloring on a categorical palette.
    DominantBiome,
    /// Temperature-field ramp (cold blue → temperate pale → hot red).
    Temperature,
    /// Moisture-field ramp (dry ochre → wet blue).
    Moisture,
}

impl SurfaceView {
    /// Cycle order for the debug key / HUD.
    pub const ALL: [SurfaceView; 4] = [
        SurfaceView::Biome,
        SurfaceView::DominantBiome,
        SurfaceView::Temperature,
        SurfaceView::Moisture,
    ];

    /// The next view in the cycle (wraps).
    pub fn next(self) -> SurfaceView {
        let i = Self::ALL.iter().position(|&v| v == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// A short HUD/debug label.
    pub fn label(self) -> &'static str {
        match self {
            SurfaceView::Biome => "biome tint",
            SurfaceView::DominantBiome => "dominant biome",
            SurfaceView::Temperature => "temperature",
            SurfaceView::Moisture => "moisture",
        }
    }

    /// Parses a bus/debug-command name (lenient; unknown ⇒ `None`).
    pub fn parse(name: &str) -> Option<SurfaceView> {
        Some(match name.to_ascii_lowercase().as_str() {
            "biome" | "tint" => SurfaceView::Biome,
            "dominant" | "dominant_biome" => SurfaceView::DominantBiome,
            "temperature" | "temp" => SurfaceView::Temperature,
            "moisture" => SurfaceView::Moisture,
            _ => return None,
        })
    }
}

/// The categorical color for dominant-biome debug row `index`: golden-ratio hue
/// stepping so adjacent indices land far apart on the wheel (deliberately
/// discrete — this view exists to show the classification boundaries).
pub fn dominant_palette(index: usize) -> [f32; 4] {
    let h = (index as f64 * 0.618_033_988_75).fract();
    let (r, g, b) = hsv_to_rgb(h, 0.65, 0.95);
    [r as f32, g as f32, b as f32, 1.0]
}

/// Temperature debug ramp: ≤180 K deep blue → 250 K pale → ≥320 K red.
/// Bounded for any input; monotone in hue position over the physical range.
pub fn temperature_ramp(kelvin: f64) -> [f32; 4] {
    let t = ((kelvin - 180.0) / (320.0 - 180.0)).clamp(0.0, 1.0);
    // Stops are per-channel monotone (red rises, blue falls) so the ramp is an
    // honest single-variable read, not just a pretty gradient.
    ramp3(
        t,
        [0.10, 0.25, 0.85],
        [0.90, 0.90, 0.70],
        [0.95, 0.15, 0.10],
    )
}

/// Moisture debug ramp: 0 dry ochre → 0.5 sage → 1 wet blue. Bounded, monotone.
pub fn moisture_ramp(moisture: f64) -> [f32; 4] {
    let t = moisture.clamp(0.0, 1.0);
    ramp3(
        t,
        [0.60, 0.45, 0.22],
        [0.45, 0.60, 0.35],
        [0.10, 0.35, 0.80],
    )
}

/// sRGB → linear conversion (exact piecewise curve). Biome-table tints and the
/// debug ramps/palette are **authored as sRGB** (perceptual values, like every
/// `Color::srgb` in the app); the GPU consumes vertex colors linearly, so the
/// conversion happens here — the one place color math lives (WI 795 lockstep
/// rule). Feeding authored values straight through as linear washed the whole
/// body out toward white (linear 0.5 displays like sRGB ~0.74).
fn srgb_to_linear(c: [f32; 4]) -> [f32; 4] {
    let conv = |v: f32| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    [conv(c[0]), conv(c[1]), conv(c[2]), c[3]]
}

/// Piecewise-linear 3-stop color ramp over `t ∈ [0, 1]`.
fn ramp3(t: f64, a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f32; 4] {
    let lerp = |x: [f64; 3], y: [f64; 3], s: f64| {
        [
            x[0] + (y[0] - x[0]) * s,
            x[1] + (y[1] - x[1]) * s,
            x[2] + (y[2] - x[2]) * s,
        ]
    };
    let rgb = if t < 0.5 {
        lerp(a, b, t * 2.0)
    } else {
        lerp(b, c, (t - 0.5) * 2.0)
    };
    [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 1.0]
}

/// HSV → RGB (h, s, v ∈ [0, 1]).
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match (i as i64).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// Skirt depth as a multiple of a chunk's own **relief** (max−min elevation). The
/// terrain gap against a finer/coarser neighbour is bounded by the relief the chunks
/// span, so a relief-sized skirt covers the *terrain* part of the gap. (Sizing the
/// skirt to the node's edge length instead grows kilometre-tall walls on coarse
/// chunks that read as a "waffle" at grazing angles — WI 773.)
const SKIRT_RELIEF_FACTOR: f64 = 2.0;
/// Skirt depth as a multiple of the chunk's own **curvature sagitta** (how far a
/// straight edge chord sinks below the spherical surface). A one-level-coarser
/// neighbour spans twice the edge, so its chord sinks ~4× as far below the true
/// surface; the finer chunk (whose edge is on the surface) must reach down past it,
/// or the seam cracks open — the concentric LOD "cliffs" seen on a large body from
/// altitude (WI 779). ≥4 covers a coarser neighbour; 6 leaves margin. This term is
/// ~0 at max LOD (tiny edges near the surface), so it never revives the WI 773 waffle.
const SKIRT_CURVATURE_FACTOR: f64 = 6.0;
/// A small floor (metres) so a near-flat chunk still has a non-degenerate skirt.
const SKIRT_FLOOR: f64 = 2.0;
/// Split when the camera is within this many node-edge-lengths of a node.
pub const SPLIT_RANGE_FACTOR: f64 = 2.5;
/// Default per-chunk grid resolution (quads per side; vertices = res+1).
pub const DEFAULT_RESOLUTION: u32 = 24;
/// Default maximum quadtree depth (subdivisions per cube face).
pub const DEFAULT_MAX_LEVEL: u32 = 18;

/// One of the six faces of the base cube.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CubeFace {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl CubeFace {
    /// All six faces.
    pub const ALL: [CubeFace; 6] = [
        CubeFace::PosX,
        CubeFace::NegX,
        CubeFace::PosY,
        CubeFace::NegY,
        CubeFace::PosZ,
        CubeFace::NegZ,
    ];

    /// The pre-spherify cube-surface point for face coordinates `(u, v)`, each in
    /// `[-1, 1]`. The `(u, v)` axes are chosen so `u_axis × v_axis` is the outward
    /// face normal (front faces wind counter-clockwise from outside).
    pub fn cube_point(self, u: f64, v: f64) -> DVec3 {
        match self {
            // +X: u=+Y, v=+Z
            CubeFace::PosX => DVec3::new(1.0, u, v),
            // -X: u=+Z, v=+Y
            CubeFace::NegX => DVec3::new(-1.0, v, u),
            // +Y: u=+Z, v=+X
            CubeFace::PosY => DVec3::new(v, 1.0, u),
            // -Y: u=+X, v=+Z
            CubeFace::NegY => DVec3::new(u, -1.0, v),
            // +Z: u=+X, v=+Y
            CubeFace::PosZ => DVec3::new(u, v, 1.0),
            // -Z: u=+Y, v=+X
            CubeFace::NegZ => DVec3::new(v, u, -1.0),
        }
    }
}

/// Maps a pre-spherify cube point (each component in `[-1, 1]`) onto the unit
/// sphere with the area-equalizing spherified-cube transform (better vertex/sample
/// density than naive normalization; pole-free). The result is normalized to
/// guarantee a unit direction.
pub fn spherify(cube: DVec3) -> DVec3 {
    let (x, y, z) = (cube.x, cube.y, cube.z);
    let (x2, y2, z2) = (x * x, y * y, z * z);
    let s = DVec3::new(
        x * (1.0 - y2 / 2.0 - z2 / 2.0 + y2 * z2 / 3.0).max(0.0).sqrt(),
        y * (1.0 - z2 / 2.0 - x2 / 2.0 + z2 * x2 / 3.0).max(0.0).sqrt(),
        z * (1.0 - x2 / 2.0 - y2 / 2.0 + x2 * y2 / 3.0).max(0.0).sqrt(),
    );
    let n = s.normalize_or_zero();
    if n == DVec3::ZERO {
        DVec3::X
    } else {
        n
    }
}

/// The unit direction for face coordinates `(u, v)` — `spherify ∘ cube_point`.
pub fn direction(face: CubeFace, u: f64, v: f64) -> DVec3 {
    spherify(face.cube_point(u, v))
}

/// A node of a per-cube-face quadtree. At `level`, a face is divided into
/// `2^level × 2^level` nodes indexed by `(i, j)` over `u` and `v` respectively.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QuadNode {
    pub face: CubeFace,
    pub level: u32,
    pub i: u32,
    pub j: u32,
}

impl QuadNode {
    /// The root node covering a whole face.
    pub fn root(face: CubeFace) -> Self {
        Self {
            face,
            level: 0,
            i: 0,
            j: 0,
        }
    }

    /// The six face roots.
    pub fn roots() -> [QuadNode; 6] {
        CubeFace::ALL.map(QuadNode::root)
    }

    /// The `(u0, u1, v0, v1)` sub-rectangle of the face this node covers, in
    /// `[-1, 1]` face coordinates.
    pub fn uv_rect(self) -> (f64, f64, f64, f64) {
        let span = 2.0 / (1u32 << self.level) as f64;
        let u0 = -1.0 + span * self.i as f64;
        let v0 = -1.0 + span * self.j as f64;
        (u0, u0 + span, v0, v0 + span)
    }

    /// The four corner directions of this node (u0v0, u1v0, u1v1, u0v1).
    pub fn corner_dirs(self) -> [DVec3; 4] {
        let (u0, u1, v0, v1) = self.uv_rect();
        [
            direction(self.face, u0, v0),
            direction(self.face, u1, v0),
            direction(self.face, u1, v1),
            direction(self.face, u0, v1),
        ]
    }

    /// The direction through this node's centre.
    pub fn center_dir(self) -> DVec3 {
        let (u0, u1, v0, v1) = self.uv_rect();
        direction(self.face, 0.5 * (u0 + u1), 0.5 * (v0 + v1))
    }

    /// The node's approximate world edge length at the reference `radius`, metres
    /// (mean of its two edge chord lengths). Used for LOD ranging and skirt depth.
    pub fn edge_len(self, radius: f64) -> f64 {
        let [c00, c10, _c11, c01] = self.corner_dirs();
        0.5 * ((c00 - c10).length() + (c00 - c01).length()) * radius
    }

    /// The four child nodes one level finer.
    pub fn children(self) -> [QuadNode; 4] {
        let (l, i, j) = (self.level + 1, self.i * 2, self.j * 2);
        [
            QuadNode {
                face: self.face,
                level: l,
                i,
                j,
            },
            QuadNode {
                face: self.face,
                level: l,
                i: i + 1,
                j,
            },
            QuadNode {
                face: self.face,
                level: l,
                i,
                j: j + 1,
            },
            QuadNode {
                face: self.face,
                level: l,
                i: i + 1,
                j: j + 1,
            },
        ]
    }

    /// Whether this node **contains** `other` — same face, this node at an equal
    /// or coarser level, and `other` lies within this node's `(u, v)` sub-rect.
    /// (A node contains itself.) Used to gate a chunk's despawn on its replacement
    /// coverage being resident.
    pub fn contains(self, other: QuadNode) -> bool {
        if self.face != other.face || self.level > other.level {
            return false;
        }
        let shift = other.level - self.level;
        (other.i >> shift) == self.i && (other.j >> shift) == self.j
    }

    /// Whether this node and `other` cover any common area — i.e. one contains the
    /// other (they are the same node, ancestor/descendant, or disjoint).
    pub fn overlaps(self, other: QuadNode) -> bool {
        self.contains(other) || other.contains(self)
    }
}

/// A built chunk's render buffers. Positions/normals/UVs/morph-targets are parallel
/// arrays; `indices` triangulates them. Positions are `f32` **relative to `center`**.
#[derive(Clone, Debug)]
pub struct ChunkMesh {
    /// The node's centre world point (metres, body-centred), the placement anchor.
    pub center: DVec3,
    /// Vertex positions relative to `center`, metres.
    pub positions: Vec<[f32; 3]>,
    /// Outward unit vertex normals.
    pub normals: Vec<[f32; 3]>,
    /// Vertex texture coordinates.
    pub uvs: Vec<[f32; 2]>,
    /// CDLOD morph targets: each vertex's position on the parent (one-level-coarser)
    /// grid, `f32` relative to `center`. Blending `positions → morph_targets` by a
    /// distance-driven factor collapses the chunk onto the coarse geometry, so a
    /// fully-morphed fine chunk matches its coarse neighbour (no seam) and level
    /// changes are continuous (no pop). Skirt vertices carry their own position
    /// (skirts do not morph).
    pub morph_targets: Vec<[f32; 3]>,
    /// Per-vertex colors (linear RGBA, alpha 1) — the biome-blended tint, or a
    /// debug view's color, per the [`SurfaceView`] the chunk was built with
    /// (WI 869). Skirt vertices inherit their source border vertex's color
    /// (the WI 786 pattern). Colors are an *additional* buffer: geometry is
    /// identical across views, and colors do not morph (positions do) — the
    /// biome fields are low-frequency relative to vertex spacing, which the
    /// LOD-agreement tests verify rather than assume.
    pub colors: Vec<[f32; 4]>,
    /// Triangle indices (three per triangle).
    pub indices: Vec<u32>,
}

/// The `LOD` split decision: split `node` (subdivide) when the camera is close
/// enough that the node's world edge subtends more than the range factor allows,
/// bounded by `max_level`. Pure function of node + camera world position.
///
/// Two WI 795 properties make the LOD boundaries land inside the CDLOD morph
/// windows (the "zippering" fix — see the module doc of `surface_scan` and the
/// WI 795 record):
///
/// 1. **Surface-consistent distances.** Distances are measured to points **on the
///    surface** (`radius + elevation`, the same anchor `build_chunk` places vertices
///    at) — not to the bare sphere. The morph factor is a per-vertex function of
///    camera-to-surface-vertex distance; measuring selection against the sphere let
///    the two metrics diverge by the local elevation (kilometres at low altitude
///    over a −5.6 km basin at the WI 791 known-bad pose), realizing boundaries far
///    outside every morph window.
/// 2. **Nearest-point ranging.** The distance is the *minimum* over the node's
///    surface centre and four surface corners — a node's shared boundary can sit
///    ~0.7–0.8 edge lengths nearer than its centre (plus intra-node elevation
///    variation), and the per-level morph anchors (`morph_range`) require realized
///    fine/coarse boundaries no nearer than `(SPLIT_RANGE_FACTOR − EDGE_OFFSET) ·
///    min_edge_len(level)`; ranging on the centre alone let a kept coarse node
///    expose a boundary *inside* the finer neighbour's ramp (measured 634 m /
///    174 px at the known-bad pose).
///
/// A conservative pre-test (bounding the surface between `±relief_bound`) resolves
/// clearly-far and clearly-near nodes without sampling the field, so the five
/// elevation samples are paid only near the split threshold.
pub fn should_split(
    field: &SurfaceField,
    node: QuadNode,
    camera_world: DVec3,
    max_level: u32,
) -> bool {
    if node.level >= max_level {
        return false;
    }
    let radius = field.radius();
    let size = node.edge_len(radius);
    let threshold = size * SPLIT_RANGE_FACTOR;

    // Conservative pre-test on the sphere-distance to the node centre: the nearest
    // surface point of the node is at least `center_sphere_dist − reach` and at most
    // `center_sphere_dist + reach`, with `reach` = half-diagonal + relief bound.
    let cdir = node.center_dir();
    let center_sphere = cdir * radius;
    let center_dist = (camera_world - center_sphere).length();
    let reach = 0.75 * size + field.relief_bound();
    if center_dist - reach >= threshold {
        return false; // even the nearest possible surface point is beyond range
    }
    if center_dist + reach < threshold {
        return true; // even the farthest possible surface point is within range
    }

    // Near the threshold: exact nearest-point test over centre + corners on the
    // actual surface.
    let mut nearest = f64::INFINITY;
    let corner_dirs = node.corner_dirs();
    for dir in std::iter::once(cdir).chain(corner_dirs) {
        let p = dir * (radius + field.elevation(dir));
        nearest = nearest.min((camera_world - p).length());
    }
    nearest < threshold
}

/// How far a chunk's shared boundary can sit from its node **centre**, as a fraction of
/// the node's edge — half an edge. A node's LOD split test uses its centre distance, but
/// the edge it shares with a neighbour spans centre ± half-edge; the morph ramp must
/// account for that offset (WI 783). See [`morph_range`].
const EDGE_OFFSET_FACTOR: f64 = 0.5;

/// The smallest world edge length (metres) among all nodes at `level` — the face-corner
/// node `(i=0, j=0)`, since the spherified-cube map compresses toward face corners
/// (verified exact against a full-face scan, WI 783). A level-`level` node stops
/// splitting (and can thus expose a coarser neighbour to a finer chunk) at the earliest
/// when its edge is this small.
pub fn min_edge_len(level: u32, radius: f64) -> f64 {
    QuadNode {
        face: CubeFace::PosZ,
        level,
        i: 0,
        j: 0,
    }
    .edge_len(radius)
}

/// The largest world edge length (metres) among all nodes at `level`. The maximum lies in
/// the face's last column/row (the spherified-cube stretches most away from face centres;
/// verified against a full-face scan, WI 783), so it is found by scanning that 1-D border
/// — exactly for shallow levels, and by a bounded subsample for deep ones where the edge
/// length varies smoothly (the tiny sampling error is immaterial: deep-level morph windows
/// are wide). A level-`level` node stops splitting at the latest when its edge is this
/// large.
pub fn max_edge_len(level: u32, radius: f64) -> f64 {
    let side = 1u32 << level;
    // Cap the number of samples so deep levels stay cheap; the border edge length is
    // smooth there, so subsampling misses the true max by a negligible amount.
    let step = (side / 2048).max(1);
    let mut mx = 0.0f64;
    let mut k = 0u32;
    while k < side {
        // Last column (i = side−1) and last row (j = side−1).
        mx = mx.max(
            QuadNode {
                face: CubeFace::PosZ,
                level,
                i: side - 1,
                j: k,
            }
            .edge_len(radius),
        );
        mx = mx.max(
            QuadNode {
                face: CubeFace::PosZ,
                level,
                i: k,
                j: side - 1,
            }
            .edge_len(radius),
        );
        k += step;
    }
    mx
}

/// The camera-distance `(start, end)` (metres) of the CDLOD morph ramp for chunks at
/// `level` on a body of `radius`. Two constraints make an LOD boundary seamless (WI 783):
/// where a chunk borders a **coarser** neighbour it must be **fully** morphed to the
/// parent shape (factor 1), and where it borders a **finer** neighbour it must be
/// **un**-morphed (factor 0) — because the finer side morphs to *this* chunk's un-morphed
/// geometry. Those two boundary families are separated in distance, so the ramp is placed
/// in the quiet zone between them:
/// - `end` = `(SPLIT_RANGE_FACTOR − EDGE_OFFSET) · min_edge_len(level−1)`: the nearest a
///   one-level-coarser neighbour can appear (its smallest node stops splitting at
///   `SPLIT_RANGE_FACTOR · min_edge`, and the shared edge is up to half a node nearer).
///   Fully morphed by here ⇒ no step against a coarser neighbour.
/// - `start` = `(SPLIT_RANGE_FACTOR + EDGE_OFFSET) · max_edge_len(level)`: the farthest a
///   finer-neighbour boundary of *this* level can sit. Un-morphed until here ⇒ a finer
///   neighbour (morphing to this chunk) still matches.
///
/// The range depends only on `level`, so same-level neighbours share an identical factor
/// and their shared edges stay matched. The render vertex shader applies
/// `smoothstep(start, end, per_vertex_distance)`, continuous in space. Level 0 never
/// borders a coarser neighbour, so it never morphs (a disjoint range beyond any real
/// distance yields factor 0).
pub fn morph_range(level: u32, radius: f64) -> (f32, f32) {
    if level == 0 {
        // No coarser neighbour ever borders a root; keep it un-morphed with a
        // well-formed (start < end) range beyond any real camera distance.
        return (1.0e12, 2.0e12);
    }
    let start = (SPLIT_RANGE_FACTOR + EDGE_OFFSET_FACTOR) * max_edge_len(level, radius);
    let end = (SPLIT_RANGE_FACTOR - EDGE_OFFSET_FACTOR) * min_edge_len(level - 1, radius);
    (start as f32, end as f32)
}

/// Rows over which an edge-weld forcing fades back to the distance-driven morph
/// factor (WI 795). Wide enough that a forced edge does not fold against its
/// interior rows (the per-vertex morph displacement can be hundreds of metres in
/// craters), narrow enough that the falloff rarely reaches the opposite edge.
pub const WELD_BAND_ROWS: f64 = 6.0;

/// The per-vertex CDLOD morph factor with the WI 795 **edge weld** applied — the
/// single formula the render shader (WGSL) and the headless oracle both implement;
/// keep them in lockstep.
///
/// Per-level global morph ramps cannot make cross-level boundaries seamless on this
/// body: a node splits for its *near* side, so far-side children realize boundaries
/// up to `(SPLIT_RANGE_FACTOR + diagonal)·edge` plus intra-node relief away — past
/// the level's ramp start — while widening the ramp start beyond the next level's
/// ramp end inverts the window (relief ~15 km bound, ~1.3× cube-distortion spread).
/// So the realized neighbour relation is welded in directly, per edge:
///
/// - On an edge bordering a **coarser** neighbour the factor forces to **1**: a
///   fully-morphed boundary vertex lies on the parent grid, which for a one-level
///   jump *is* the coarser chunk's un-morphed surface chord (odd verts land on the
///   chord midpoint between two of the coarser chunk's own vertices, bit-near-exact).
/// - On an edge bordering a **finer** neighbour the factor forces to **0**: the
///   finer side's welded (fully-morphed) vertices equal this chunk's surface grid.
///
/// Each forcing fades inward over [`WELD_BAND_ROWS`] so the edge never folds against
/// the chunk interior. Masks carry **8 relations**: bits 0–3 the four edges (v0, u1,
/// v1, u0), bits 4–7 the four **diagonal corners** (c00, c10, c11, c01). Corner bits
/// exist for cross-chunk consistency: when a coarser region touches a chunk only at
/// a corner, the two same-level chunks flanking that corner must still render their
/// shared edge identically, so the corner-adjacent chunk applies the same falloff
/// profile (by Chebyshev distance to the corner) that its neighbour applies from its
/// flagged edge. The per-source falloffs combine by **max** (order-independent,
/// idempotent — an edge and its corner never double-apply), then finer forcings pull
/// toward 0 and coarser forcings toward 1, coarser last (a vertex constrained by
/// both welds to the coarser side, the one that cannot move). Both sides of a
/// cross-level edge therefore render the **coarser level's surface grid**, and
/// same-level edges match exactly (verified by `scan_same_level_exact`, WI 795).
pub fn weld_factor(
    dist_factor: f64,
    a: u32,
    b: u32,
    res: u32,
    mask_coarser: u8,
    mask_finer: u8,
) -> f64 {
    // Rows from each edge, in edge order 0 = v0 (b = 0), 1 = u1 (a = res),
    // 2 = v1 (b = res), 3 = u0 (a = 0), then Chebyshev distance to each corner
    // (c00, c10, c11, c01).
    let dist = [
        b,
        res - a,
        res - b,
        a,
        a.max(b),
        (res - a).max(b),
        (res - a).max(res - b),
        a.max(res - b),
    ];
    let falloff = |r: u32| (1.0 - r as f64 / WELD_BAND_ROWS).clamp(0.0, 1.0);
    let weight = |mask: u8| {
        let mut w = 0.0f64;
        for (k, &r) in dist.iter().enumerate() {
            if mask & (1 << k) != 0 {
                w = w.max(falloff(r));
            }
        }
        w
    };
    let w_finer = weight(mask_finer);
    let w_coarser = weight(mask_coarser);
    let f = dist_factor + (0.0 - dist_factor) * w_finer;
    f + (1.0 - f) * w_coarser
}

/// A representative world edge length (metres) for all chunks at `level`, sampled at a
/// mid-face node — a per-level scale roughly midway between [`min_edge_len`] and
/// [`max_edge_len`]. (The morph ramp is anchored on the min/max extremes, not this; kept
/// as a convenient level scale and used in tests.)
pub fn nominal_edge_len(level: u32, radius: f64) -> f64 {
    let c = (1u32 << level) / 2;
    QuadNode {
        face: CubeFace::PosZ,
        level,
        i: c,
        j: c,
    }
    .edge_len(radius)
}

/// How far the straight chord of an edge of chord-length `edge_len` sinks below the
/// spherical surface of the given `radius` — the sagitta `R·(1 − cos(θ/2))` for an
/// edge subtending angle `θ ≈ edge_len / radius`. This is the geometric part of an
/// LOD-boundary gap that terrain relief does not account for.
pub fn edge_sagitta(edge_len: f64, radius: f64) -> f64 {
    let radius = radius.max(1.0);
    let half_angle = 0.5 * edge_len / radius;
    radius * (1.0 - half_angle.cos())
}

/// The skirt depth (metres) for a chunk spanning `relief` metres of elevation with a
/// chord edge of `edge_len` on a body of `radius` — enough to cover the worst
/// LOD-boundary gap against a one-level-coarser neighbour (its terrain relief *plus*
/// its curvature sagitta, ~4× this chunk's own), plus a small floor, and no more.
pub fn skirt_depth_for(relief: f64, edge_len: f64, radius: f64) -> f64 {
    SKIRT_RELIEF_FACTOR * relief.max(0.0)
        + SKIRT_CURVATURE_FACTOR * edge_sagitta(edge_len, radius)
        + SKIRT_FLOOR
}

/// The relief (max−min elevation, metres) a chunk spans at resolution `res` — the
/// basis for its skirt depth. Matches the range `build_chunk` computes internally.
pub fn chunk_relief(field: &SurfaceField, node: QuadNode, res: u32) -> f64 {
    let res = res.max(1);
    let (u0, u1, v0, v1) = node.uv_rect();
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let (mut min_e, mut max_e) = (f64::INFINITY, f64::NEG_INFINITY);
    for b in 0..=res {
        let vv = lerp(v0, v1, b as f64 / res as f64);
        for a in 0..=res {
            let uu = lerp(u0, u1, a as f64 / res as f64);
            let e = field.elevation(direction(node.face, uu, vv));
            min_e = min_e.min(e);
            max_e = max_e.max(e);
        }
    }
    (max_e - min_e).max(0.0)
}

/// Builds the chunk mesh for `node` at grid resolution `res` (quads per side),
/// sampling `field`. Deterministic and pure: identical inputs → identical buffers.
///
/// `res` is forced **even** (odd rounded up): CDLOD morph targets need the parent
/// grid to sample every other vertex, so the border indices (0 and `res`) must be
/// even and each interior odd vertex must sit between two even neighbours.
pub fn build_chunk(field: &SurfaceField, node: QuadNode, res: u32) -> ChunkMesh {
    build_chunk_with(field, node, res, true)
}

/// [`build_chunk`] with the skirt optional. `with_skirt: false` is the WI 795
/// diagnostic build (the `SOUNDING_NO_SKIRT=1` visual cross-check: with skirts off,
/// any remaining dark boundary geometry is the LOD step itself, not skirt walls).
pub fn build_chunk_with(
    field: &SurfaceField,
    node: QuadNode,
    res: u32,
    with_skirt: bool,
) -> ChunkMesh {
    build_chunk_view(field, node, res, with_skirt, SurfaceView::Biome)
}

/// [`build_chunk_with`] with an explicit color view (WI 869): geometry is
/// identical for every view; only the `colors` buffer differs.
pub fn build_chunk_view(
    field: &SurfaceField,
    node: QuadNode,
    res: u32,
    with_skirt: bool,
    view: SurfaceView,
) -> ChunkMesh {
    let res = res.max(1);
    let res = res + (res & 1); // even: parent grid uses every other vertex
    let (u0, u1, v0, v1) = node.uv_rect();
    let radius = field.radius();
    let n = (res + 1) as usize;

    // Placement anchor: the node centre, on the surface.
    let cdir = node.center_dir();
    let center = cdir * (radius + field.elevation(cdir));

    let vert_count = n * n;
    let mut positions = Vec::with_capacity(vert_count);
    let mut normals = Vec::with_capacity(vert_count);
    let mut uvs = Vec::with_capacity(vert_count);
    let mut colors = Vec::with_capacity(vert_count);

    // Surface grid: sample the field at each (u, v) direction. Track the elevation
    // range so the skirt can be sized to this chunk's relief.
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let mut world_of = Vec::with_capacity(vert_count); // f64 world points, for skirt reuse
    let (mut min_elev, mut max_elev) = (f64::INFINITY, f64::NEG_INFINITY);
    for b in 0..=res {
        let tv = b as f64 / res as f64;
        let vv = lerp(v0, v1, tv);
        for a in 0..=res {
            let tu = a as f64 / res as f64;
            let uu = lerp(u0, u1, tu);
            let dir = direction(node.face, uu, vv);
            let elev = field.elevation(dir);
            min_elev = min_elev.min(elev);
            max_elev = max_elev.max(elev);
            let world = dir * (radius + elev);
            world_of.push((dir, world));
            positions.push((world - center).as_vec3().to_array());
            let normal = field.normal(dir);
            normals.push(normal.as_vec3().to_array());
            uvs.push(Vec2::new(tu as f32, tv as f32).to_array());
            colors.push(vertex_color(field, dir, elev, normal, view));
        }
    }

    let idx = |a: u32, b: u32| b * (res + 1) + a;
    let mut indices = Vec::with_capacity((res * res * 6) as usize);
    for b in 0..res {
        for a in 0..res {
            let (v00, v10, v11, v01) = (idx(a, b), idx(a + 1, b), idx(a + 1, b + 1), idx(a, b + 1));
            // CCW from outside (u_axis × v_axis = outward normal).
            indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
        }
    }

    // CDLOD morph targets: for each surface vertex, the position it would occupy on
    // the parent (coarser) grid — the bilinear interpolation of the surrounding
    // even-index vertices (itself when both indices are even; the midpoint of two
    // even neighbours on an edge; the average of the four even corners in a cell).
    // The parent samples the even vertices, which coincide with these even surface
    // vertices, so this equals the coarse neighbour's geometry. Computed from the
    // `world_of` grid already sampled — no extra field evaluation.
    let world_at = |a: u32, b: u32| world_of[(b * (res + 1) + a) as usize].1;
    let mut morph_targets = Vec::with_capacity(vert_count);
    for b in 0..=res {
        for a in 0..=res {
            let target = match (a & 1, b & 1) {
                (0, 0) => world_at(a, b),
                (1, 0) => 0.5 * (world_at(a - 1, b) + world_at(a + 1, b)),
                (0, 1) => 0.5 * (world_at(a, b - 1) + world_at(a, b + 1)),
                _ => {
                    0.25 * (world_at(a - 1, b - 1)
                        + world_at(a + 1, b - 1)
                        + world_at(a - 1, b + 1)
                        + world_at(a + 1, b + 1))
                }
            };
            morph_targets.push((target - center).as_vec3().to_array());
        }
    }

    // Skirt: a wall hanging inward from each border edge, deep enough to cover the
    // LOD-boundary gap against a one-level-coarser neighbour — its terrain relief
    // (bounded by this chunk's relief) *and* its curvature sagitta (the chord of a
    // coarse edge sinks ~4× this chunk's sagitta below the true surface) — but no
    // deeper, so it stays buried instead of standing up as a wall.
    if with_skirt {
        let relief = (max_elev - min_elev).max(0.0);
        let skirt_depth = skirt_depth_for(relief, node.edge_len(radius), radius);
        add_skirt(
            res,
            &world_of,
            center,
            skirt_depth,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut colors,
            &mut indices,
        );

        // Skirt vertices do not morph: their target is their own position, so the
        // morph blend is a no-op on the skirt regardless of the chunk's morph factor.
        morph_targets.extend_from_slice(&positions[vert_count..]);
    }

    ChunkMesh {
        center,
        positions,
        normals,
        uvs,
        morph_targets,
        colors,
        indices,
    }
}

/// The per-vertex color for `view` (WI 869). `elev`/`normal` are the grid
/// pass's already-computed samples at `dir` — the `biome_weights_at` seam, so
/// the biome query here costs the climate fields + classification only.
fn vertex_color(
    field: &SurfaceField,
    dir: DVec3,
    elev: f64,
    normal: DVec3,
    view: SurfaceView,
) -> [f32; 4] {
    let srgb = match view {
        SurfaceView::Biome => {
            let t = field.biome_weights_at(dir, elev, normal).tint();
            [t[0] as f32, t[1] as f32, t[2] as f32, 1.0]
        }
        SurfaceView::DominantBiome => {
            dominant_palette(field.biome_weights_at(dir, elev, normal).dominant_index())
        }
        SurfaceView::Temperature => temperature_ramp(field.temperature(dir)),
        SurfaceView::Moisture => moisture_ramp(field.moisture(dir)),
    };
    srgb_to_linear(srgb)
}

/// Appends the border skirt: for each border grid vertex, a duplicate pushed
/// radially inward by `skirt_depth`; walls connect consecutive border vertices to
/// their skirt duplicates. The wall's `-radial` drop hides LOD-boundary gaps.
///
/// Skirt vertices **inherit the border vertex's normal and UV** (WI 786): the exposed
/// portion of a wall then shades exactly like the adjacent surface (invisible) instead of
/// as a dark inward-facing band — the residual LOD-boundary "cliff" was mostly this shading.
/// (Cesium's technique; the wall geometry/depth is unchanged, so crack coverage is preserved.)
#[allow(clippy::too_many_arguments)]
fn add_skirt(
    res: u32,
    world_of: &[(DVec3, DVec3)],
    center: DVec3,
    skirt_depth: f64,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let grid_idx = |a: u32, b: u32| (b * (res + 1) + a) as usize;

    // The border ring in order (bottom → right → top → left), as grid indices.
    let mut ring: Vec<usize> = Vec::new();
    for a in 0..=res {
        ring.push(grid_idx(a, 0));
    }
    for b in 1..=res {
        ring.push(grid_idx(res, b));
    }
    for a in (0..res).rev() {
        ring.push(grid_idx(a, res));
    }
    for b in (1..res).rev() {
        ring.push(grid_idx(0, b));
    }

    // A skirt duplicate for each ring vertex, pushed inward along its direction. The
    // duplicate inherits the border vertex's normal + UV (WI 786) so the wall shades like
    // the surface, not as a dark inward-facing band.
    let skirt_base = positions.len() as u32;
    for &gi in &ring {
        let (dir, world) = world_of[gi];
        let dropped = world - dir * skirt_depth;
        let border_normal = normals[gi];
        let border_uv = uvs[gi];
        let border_color = colors[gi];
        positions.push((dropped - center).as_vec3().to_array());
        normals.push(border_normal);
        uvs.push(border_uv);
        colors.push(border_color);
    }

    // Walls: quad (top_k, top_k+1, skirt_k+1, skirt_k) as two triangles.
    let ring_len = ring.len();
    for k in 0..ring_len {
        let k1 = (k + 1) % ring_len;
        let top0 = ring[k] as u32;
        let top1 = ring[k1] as u32;
        let sk0 = skirt_base + k as u32;
        let sk1 = skirt_base + k1 as u32;
        indices.extend_from_slice(&[top0, sk0, sk1, top0, sk1, top1]);
    }
}

/// Per-body atmosphere render parameters, derived from a body's intrinsics — the
/// data-driven resolution of designreview R5. Purely numeric (no engine types) so
/// it is unit-testable headless; the app maps it onto Bevy's `Atmosphere`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphereParams {
    /// Planet (sea-level) radius, metres — Bevy `Atmosphere::bottom_radius`.
    pub bottom_radius: f32,
    /// Radius at which the atmosphere is considered to end, metres — `top_radius`.
    pub top_radius: f32,
    /// Approximate ground albedo/colour, linear RGB — `ground_albedo`.
    pub ground_albedo: [f32; 3],
}

/// The number of scale heights at which the atmosphere is treated as ended.
const ATMO_TOP_SCALE_HEIGHTS: f64 = 12.0;

impl AtmosphereParams {
    /// Derives per-body atmosphere parameters, or `None` for an **airless** body
    /// (no atmospheric density) — which then renders with no atmosphere component.
    pub fn from_asset(asset: &crate::body_asset::BodyAsset) -> Option<Self> {
        let m = &asset.fluid_medium;
        if m.atmosphere_surface_density <= 0.0 {
            return None;
        }
        let bottom = asset.radius;
        let thickness = (m.atmosphere_scale_height * ATMO_TOP_SCALE_HEIGHTS).max(1.0);
        let ground_albedo = if m.ocean_surface_density > 0.0 {
            [0.10, 0.20, 0.40]
        } else {
            [0.30, 0.26, 0.20]
        };
        Some(Self {
            bottom_radius: bottom as f32,
            top_radius: (bottom + thickness) as f32,
            ground_albedo,
        })
    }
}

/// Converts an `f32` position array to a [`Vec3`] (small app-side convenience kept
/// headless-side so the app layer stays a thin adapter).
pub fn to_vec3(p: [f32; 3]) -> Vec3 {
    Vec3::from_array(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biome::{BiomeFamily, BodyClimate};
    use crate::body_asset::BodyAsset;
    use crate::surface_field::CraterParams;

    const R: f64 = 1_000_000.0;

    fn field() -> SurfaceField {
        SurfaceField::new(1234, R)
    }

    /// A temperate ocean-bearing atmospheric body — real tint variation for the
    /// color tests (the default airless climate is mostly regolith-grey).
    fn temperate_field() -> SurfaceField {
        SurfaceField::with_params(
            21,
            2_000_000.0,
            CraterParams::default(),
            BodyClimate {
                family: BiomeFamily::Atmospheric,
                base_temperature: 288.0,
                sea_level: Some(0.0),
                axis: DVec3::Z,
            },
        )
    }

    #[test]
    fn chunk_colors_match_the_field_tint_and_skirts_inherit() {
        // WI 869: the Biome view's per-vertex color is exactly the field's
        // blended tint at that vertex's direction (all color math lives here,
        // headless — the shader only passes it through), and skirt duplicates
        // inherit their source border vertex's color (the WI 786 pattern).
        for f in [field(), temperate_field()] {
            let node = QuadNode {
                face: CubeFace::PosX,
                level: 3,
                i: 3,
                j: 5,
            };
            let res = 8u32;
            let chunk = build_chunk(&f, node, res);
            assert_eq!(chunk.colors.len(), chunk.positions.len());
            let n = (res + 1) as usize;
            let (u0, u1, v0, v1) = node.uv_rect();
            let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
            for b in 0..=res {
                for a in 0..=res {
                    let dir = direction(
                        node.face,
                        lerp(u0, u1, a as f64 / res as f64),
                        lerp(v0, v1, b as f64 / res as f64),
                    );
                    let t = f.biome_weights(dir).tint();
                    let expected = srgb_to_linear([t[0] as f32, t[1] as f32, t[2] as f32, 1.0]);
                    let c = chunk.colors[b as usize * n + a as usize];
                    for k in 0..3 {
                        assert_eq!(
                            c[k], expected[k],
                            "grid color ≠ linearized field tint at ({a},{b})"
                        );
                        assert!((0.0..=1.0).contains(&c[k]));
                    }
                    assert_eq!(c[3], 1.0);
                }
            }
            // Skirt ring duplicates (same ring order as add_skirt).
            let grid_idx = |a: u32, b: u32| (b * (res + 1) + a) as usize;
            let mut ring: Vec<usize> = Vec::new();
            for a in 0..=res {
                ring.push(grid_idx(a, 0));
            }
            for b in 1..=res {
                ring.push(grid_idx(res, b));
            }
            for a in (0..res).rev() {
                ring.push(grid_idx(a, res));
            }
            for b in (1..res).rev() {
                ring.push(grid_idx(0, b));
            }
            let skirt_base = n * n;
            assert_eq!(chunk.colors.len(), skirt_base + ring.len());
            for (k, &gi) in ring.iter().enumerate() {
                assert_eq!(
                    chunk.colors[skirt_base + k],
                    chunk.colors[gi],
                    "skirt vertex {k} must inherit its border color"
                );
            }
            // Purity: a rebuild is bit-identical.
            let again = build_chunk(&f, node, res);
            assert_eq!(chunk.colors, again.colors);
        }
    }

    #[test]
    fn geometry_is_identical_across_color_views() {
        // WI 869 invariant: colors are an additional buffer — the view changes
        // nothing about positions/normals/uvs/morph targets/indices, so the
        // WI 795 seam/weld guarantees are untouched by construction.
        let f = temperate_field();
        let node = QuadNode {
            face: CubeFace::NegZ,
            level: 2,
            i: 1,
            j: 2,
        };
        let base = build_chunk_with(&f, node, 8, true);
        for view in SurfaceView::ALL {
            let c = build_chunk_view(&f, node, 8, true, view);
            assert_eq!(c.positions, base.positions, "{view:?}");
            assert_eq!(c.normals, base.normals, "{view:?}");
            assert_eq!(c.uvs, base.uvs, "{view:?}");
            assert_eq!(c.morph_targets, base.morph_targets, "{view:?}");
            assert_eq!(c.indices, base.indices, "{view:?}");
            assert_eq!(c.colors.len(), base.colors.len(), "{view:?}");
        }
        // And the default build (build_chunk_with) is the Biome view.
        let biome = build_chunk_view(&f, node, 8, true, SurfaceView::Biome);
        assert_eq!(base.colors, biome.colors);
    }

    #[test]
    fn debug_ramps_are_bounded_monotone_and_palette_distinct() {
        // Temperature: bounded for any input, red rises / blue falls with heat.
        let mut prev = temperature_ramp(100.0);
        for k in 0..=60 {
            let t = 100.0 + k as f64 * 5.0; // 100 → 400 K
            let c = temperature_ramp(t);
            for v in c {
                assert!((0.0..=1.0).contains(&v));
            }
            assert!(c[0] >= prev[0] - 1e-6, "red must not decrease with heat");
            assert!(c[2] <= prev[2] + 1e-6, "blue must not increase with heat");
            prev = c;
        }
        // Moisture: bounded, blue rises with wetness.
        let mut prev = moisture_ramp(-1.0);
        for k in 0..=40 {
            let m = -1.0 + k as f64 * 0.1; // clamps outside [0,1]
            let c = moisture_ramp(m);
            for v in c {
                assert!((0.0..=1.0).contains(&v));
            }
            assert!(
                c[2] >= prev[2] - 1e-6,
                "blue must not decrease with moisture"
            );
            prev = c;
        }
        // Dominant palette: distinct colors across any table-sized index range.
        for i in 0..16 {
            for j in (i + 1)..16 {
                let (a, b) = (dominant_palette(i), dominant_palette(j));
                let d: f32 = (0..3).map(|k| (a[k] - b[k]).abs()).sum();
                assert!(d > 0.05, "palette indices {i}/{j} too similar");
            }
        }
        // View cycle covers all views and wraps.
        let mut v = SurfaceView::Biome;
        for _ in 0..SurfaceView::ALL.len() {
            v = v.next();
        }
        assert_eq!(v, SurfaceView::Biome);
    }

    #[test]
    fn lod_vertex_colors_agree_across_levels() {
        // WI 869 §LOD: verify the design's claim instead of assuming it. A
        // parent chunk and its child sample coincident directions at the
        // child's even grid points — colors there must agree to float-path
        // tolerance. At the child's odd points the parent's triangles show the
        // linear interpolation of its adjacent vertex colors; the biome fields
        // are low-frequency relative to vertex spacing, so the deviation is
        // small — measured: even = 0.0 (bit-identical), odd max ≈ 0.018 on a
        // chunk whose own tint spread is 0.13; asserted at 0.05 (≈3× headroom,
        // still far below the spread a vertex-scale tint field would show).
        let f = temperate_field();
        let parent = QuadNode {
            face: CubeFace::PosX,
            level: 4,
            i: 5,
            j: 9,
        };
        let child = parent.children()[0]; // low-u/low-v quadrant
        let res = DEFAULT_RESOLUTION;
        let n = (res + 1) as usize;
        let p = build_chunk_view(&f, parent, res, false, SurfaceView::Biome);
        let c = build_chunk_view(&f, child, res, false, SurfaceView::Biome);

        // Coverage guard: the probed chunk must actually span tint variation,
        // or the agreement asserts are vacuous.
        let mut spread = 0.0_f32;
        for k in 0..3 {
            let lo = c.colors.iter().map(|v| v[k]).fold(f32::MAX, f32::min);
            let hi = c.colors.iter().map(|v| v[k]).fold(f32::MIN, f32::max);
            spread = spread.max(hi - lo);
        }
        assert!(
            spread > 0.05,
            "probe chunk shows no tint variation ({spread}) — re-site it"
        );

        let mut max_even = 0.0_f32;
        let mut max_odd = 0.0_f32;
        let pc = |a: usize, b: usize| p.colors[b * n + a];
        for b in 0..=res as usize {
            for a in 0..=res as usize {
                let cc = c.colors[b * n + a];
                let expected = match (a % 2, b % 2) {
                    (0, 0) => pc(a / 2, b / 2),
                    (1, 0) => avg2(pc(a / 2, b / 2), pc(a / 2 + 1, b / 2)),
                    (0, 1) => avg2(pc(a / 2, b / 2), pc(a / 2, b / 2 + 1)),
                    _ => avg4(
                        pc(a / 2, b / 2),
                        pc(a / 2 + 1, b / 2),
                        pc(a / 2, b / 2 + 1),
                        pc(a / 2 + 1, b / 2 + 1),
                    ),
                };
                let d = (0..3)
                    .map(|k| (cc[k] - expected[k]).abs())
                    .fold(0.0, f32::max);
                if a % 2 == 0 && b % 2 == 0 {
                    max_even = max_even.max(d);
                } else {
                    max_odd = max_odd.max(d);
                }
            }
        }
        // Even (coincident-direction) vertices: identical up to the uv float
        // path (the parent lerps its own rect).
        assert!(
            max_even < 1e-4,
            "coincident LOD vertices disagree on tint: {max_even}"
        );
        // Odd vertices vs the parent's on-screen interpolation: the level-pop
        // bound. Calibrated: measured max recorded below; a tint field varying
        // at vertex scale (the claim's failure mode) would push this toward the
        // chunk's own spread (>0.05).
        assert!(
            max_odd < 0.05,
            "LOD tint deviation at odd vertices too large: {max_odd}"
        );
    }

    fn avg2(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        std::array::from_fn(|k| 0.5 * (a[k] + b[k]))
    }

    fn avg4(a: [f32; 4], b: [f32; 4], c: [f32; 4], d: [f32; 4]) -> [f32; 4] {
        std::array::from_fn(|k| 0.25 * (a[k] + b[k] + c[k] + d[k]))
    }

    #[test]
    fn spherify_returns_unit_directions() {
        for face in CubeFace::ALL {
            for &(u, v) in &[(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0), (0.3, -0.7)] {
                let d = direction(face, u, v);
                assert!((d.length() - 1.0).abs() < 1e-12, "not unit: {d:?}");
            }
        }
    }

    #[test]
    fn faces_cover_all_octants() {
        // Every face-centre points along its own axis; together the six span the
        // sphere (one direction per axis sign).
        let centers: Vec<DVec3> = CubeFace::ALL
            .iter()
            .map(|&f| direction(f, 0.0, 0.0))
            .collect();
        for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
            assert!(centers.iter().any(|c| c.dot(axis) > 0.99));
            assert!(centers.iter().any(|c| c.dot(axis) < -0.99));
        }
    }

    #[test]
    fn cross_face_edges_are_continuous() {
        // The +X/+Z shared edge: +X at v=1 and +Z at u=1 must map to the same
        // locus for a matching parameter t (no parameterization seam).
        for k in 0..=10 {
            let t = -1.0 + 2.0 * k as f64 / 10.0;
            let a = direction(CubeFace::PosX, t, 1.0);
            let b = direction(CubeFace::PosZ, 1.0, t);
            assert!((a - b).length() < 1e-12, "seam at t={t}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn child_nodes_tile_the_parent() {
        let parent = QuadNode::root(CubeFace::PosZ);
        let (pu0, pu1, pv0, pv1) = parent.uv_rect();
        for child in parent.children() {
            let (cu0, cu1, cv0, cv1) = child.uv_rect();
            assert!(cu0 >= pu0 - 1e-12 && cu1 <= pu1 + 1e-12);
            assert!(cv0 >= pv0 - 1e-12 && cv1 <= pv1 + 1e-12);
        }
        // The four children exactly partition the parent's area (each is a quarter).
        let child = parent.children()[0];
        let (cu0, cu1, _, _) = child.uv_rect();
        assert!(((cu1 - cu0) - (pu1 - pu0) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn node_containment_and_overlap() {
        let parent = QuadNode {
            face: CubeFace::PosZ,
            level: 2,
            i: 1,
            j: 2,
        };
        // A node contains itself and each of its descendants.
        assert!(parent.contains(parent));
        for child in parent.children() {
            assert!(parent.contains(child), "parent must contain its child");
            assert!(child.overlaps(parent) && parent.overlaps(child));
            // The child does not contain the parent (finer can't cover coarser).
            assert!(!child.contains(parent));
        }
        // A grandchild is still contained.
        let grandchild = parent.children()[3].children()[0];
        assert!(parent.contains(grandchild));
        // Different face never overlaps.
        let other_face = QuadNode {
            face: CubeFace::NegZ,
            level: 2,
            i: 1,
            j: 2,
        };
        assert!(!parent.overlaps(other_face));
        // A sibling (same level, different index) does not overlap.
        let sibling = QuadNode {
            face: CubeFace::PosZ,
            level: 2,
            i: 0,
            j: 2,
        };
        assert!(!parent.overlaps(sibling));
    }

    #[test]
    fn chunk_build_is_deterministic() {
        let f = field();
        let node = QuadNode {
            face: CubeFace::PosY,
            level: 3,
            i: 2,
            j: 5,
        };
        let a = build_chunk(&f, node, 8);
        let b = build_chunk(&f, node, 8);
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.normals, b.normals);
        assert_eq!(a.morph_targets, b.morph_targets);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.center, b.center);
    }

    #[test]
    fn morph_ramp_sits_in_the_quiet_zone_between_transitions() {
        // WI 783: the ramp must complete before the nearest coarser-neighbour boundary
        // (`end`) and not start until past the farthest finer-neighbour boundary of this
        // level (`start`), so a chunk is fully morphed against a coarser neighbour and
        // un-morphed against a finer one. Verify the ramp is well-formed (start < end) at
        // every non-root level and that the anchors match the geometric definition.
        for level in 1..18u32 {
            let (start, end) = morph_range(level, R);
            let (start, end) = (start as f64, end as f64);
            assert!(
                start < end,
                "level {level}: start ({start}) precedes end ({end})"
            );
            let expect_start = (SPLIT_RANGE_FACTOR + EDGE_OFFSET_FACTOR) * max_edge_len(level, R);
            let expect_end = (SPLIT_RANGE_FACTOR - EDGE_OFFSET_FACTOR) * min_edge_len(level - 1, R);
            assert!(
                (start - expect_start).abs() < 1.0,
                "level {level}: start anchor"
            );
            assert!((end - expect_end).abs() < 1.0, "level {level}: end anchor");
        }
        // Level 0 never morphs: the range is beyond any real distance, so factor is 0.
        let (s0, e0) = morph_range(0, R);
        assert!(
            s0 < e0 && s0 as f64 > 1.0e9,
            "root is effectively un-morphed"
        );
    }

    #[test]
    fn max_and_min_edge_len_bracket_the_face() {
        // min = face-corner node; max in the last column/row. Both bracket every node.
        for level in 1..7u32 {
            let side = 1u32 << level;
            let (mn, mx) = (min_edge_len(level, R), max_edge_len(level, R));
            for i in 0..side {
                for j in 0..side {
                    let e = QuadNode {
                        face: CubeFace::PosZ,
                        level,
                        i,
                        j,
                    }
                    .edge_len(R);
                    assert!(
                        e >= mn - 1.0 && e <= mx + 1.0,
                        "edge {e} outside [{mn},{mx}]"
                    );
                }
            }
        }
    }

    #[test]
    fn nominal_edge_len_matches_a_mid_face_node() {
        for level in 0..6u32 {
            let e = nominal_edge_len(level, R);
            assert!(e > 0.0 && e.is_finite());
            // Coarser levels have longer edges.
            if level > 0 {
                assert!(e < nominal_edge_len(level - 1, R));
            }
        }
    }

    #[test]
    fn morph_targets_are_the_parent_grid_interpolation() {
        // Even/even vertices morph to themselves (they coincide with parent samples);
        // odd vertices morph to the average of their even neighbours (the coarse
        // chord). On a sphere that average sits below the surface, so an odd vertex
        // actually moves inward under morph.
        let f = field();
        let node = QuadNode {
            face: CubeFace::PosX,
            level: 3,
            i: 3,
            j: 2,
        };
        let res = 8u32;
        let m = build_chunk(&f, node, res);
        assert_eq!(m.morph_targets.len(), m.positions.len());
        let gi = |a: u32, b: u32| (b * (res + 1) + a) as usize;
        let pos = |a: u32, b: u32| Vec3::from_array(m.positions[gi(a, b)]).as_dvec3();
        let tgt = |a: u32, b: u32| Vec3::from_array(m.morph_targets[gi(a, b)]).as_dvec3();
        let mut moved = false;
        for b in 0..=res {
            for a in 0..=res {
                let expected = match (a & 1, b & 1) {
                    (0, 0) => pos(a, b),
                    (1, 0) => 0.5 * (pos(a - 1, b) + pos(a + 1, b)),
                    (0, 1) => 0.5 * (pos(a, b - 1) + pos(a, b + 1)),
                    _ => {
                        0.25 * (pos(a - 1, b - 1)
                            + pos(a + 1, b - 1)
                            + pos(a - 1, b + 1)
                            + pos(a + 1, b + 1))
                    }
                };
                assert!(
                    (tgt(a, b) - expected).length() < 0.5,
                    "morph target ({a},{b}) off parent-grid interpolation"
                );
                if (a & 1, b & 1) == (0, 0) {
                    assert_eq!(m.morph_targets[gi(a, b)], m.positions[gi(a, b)]);
                } else if (tgt(a, b) - pos(a, b)).length() > 1.0 {
                    moved = true;
                }
            }
        }
        assert!(
            moved,
            "odd vertices should morph away from the true surface"
        );
    }

    #[test]
    fn fully_morphed_child_edge_matches_the_parent_edge() {
        // The seam-matching property: a child chunk fully morphed to its parent grid
        // reproduces the parent's rendered edge exactly, so at a LOD boundary the
        // fine (fully-morphed) edge coincides with the coarse neighbour — no step.
        let f = field();
        let parent = QuadNode {
            face: CubeFace::PosZ,
            level: 3,
            i: 2,
            j: 5,
        };
        let child = parent.children()[0]; // shares the parent's u0/v0 corner + edges
        let res = 8u32;
        let p = build_chunk(&f, parent, res);
        let c = build_chunk(&f, child, res);
        let gi = |a: u32, b: u32| (b * (res + 1) + a) as usize;
        // World point of a parent surface vertex on its b=0 edge.
        let parent_edge = |ci: u32| p.center + Vec3::from_array(p.positions[gi(ci, 0)]).as_dvec3();
        // World point of the child's fully-morphed b=0 edge vertex.
        let child_morphed =
            |a: u32| c.center + Vec3::from_array(c.morph_targets[gi(a, 0)]).as_dvec3();
        for a in 0..=res {
            let expected = if a & 1 == 0 {
                parent_edge(a / 2)
            } else {
                0.5 * (parent_edge((a - 1) / 2) + parent_edge(a.div_ceil(2)))
            };
            assert!(
                (child_morphed(a) - expected).length() < 1.0,
                "child morphed edge vertex {a} does not lie on the parent edge"
            );
        }
    }

    #[test]
    fn lod_independent_shared_vertices_are_bit_identical() {
        // A coarse node's corner direction equals the same corner of one of its
        // children; the field is a pure function of direction, so the world point
        // at that shared direction must be bit-identical whichever LOD produced it.
        let f = field();
        let parent = QuadNode {
            face: CubeFace::NegX,
            level: 2,
            i: 1,
            j: 1,
        };
        let child = parent.children()[0]; // shares the parent's u0v0 corner
        let pc = parent.corner_dirs()[0];
        let cc = child.corner_dirs()[0];
        assert_eq!(pc, cc, "shared corner directions must be identical");
        let wp = pc * (f.radius() + f.elevation(pc));
        let wc = cc * (f.radius() + f.elevation(cc));
        assert_eq!(wp, wc, "shared-direction world point must be bit-identical");
    }

    #[test]
    fn skirt_covers_the_coarse_neighbour_gap() {
        // A coarse node's skirt must be deep enough to cover the worst deviation of
        // the true surface from the straight line between its edge endpoints — the
        // gap its own coarseness can create against a finer neighbour.
        let f = field();
        let coarse = QuadNode {
            face: CubeFace::PosZ,
            level: 3,
            i: 3,
            j: 4,
        };
        let skirt_depth = skirt_depth_for(chunk_relief(&f, coarse, 24), coarse.edge_len(R), R);
        // Sample the surface along one edge; measure max radial dip below the chord
        // between the edge endpoints (a fine neighbour would resolve this dip).
        let (u0, u1, v0, _v1) = coarse.uv_rect();
        let p0 = {
            let d = direction(coarse.face, u0, v0);
            d * (R + f.elevation(d))
        };
        let p1 = {
            let d = direction(coarse.face, u1, v0);
            d * (R + f.elevation(d))
        };
        let mut max_dip = 0.0f64;
        for k in 1..32 {
            let t = k as f64 / 32.0;
            let uu = u0 + (u1 - u0) * t;
            let d = direction(coarse.face, uu, v0);
            let surf = d * (R + f.elevation(d));
            let chord = p0.lerp(p1, t);
            // Radial (below-chord) component of the deviation.
            let dip = (chord.length() - surf.length()).max(0.0);
            max_dip = max_dip.max(dip);
        }
        assert!(
            skirt_depth >= max_dip,
            "skirt {skirt_depth} must cover edge dip {max_dip}"
        );
    }

    #[test]
    fn skirt_covers_a_coarser_neighbours_curvature_sagitta() {
        // On a large body the dominant LOD seam is curvature, not relief: a coarse
        // neighbour renders a shared edge as a chord that sinks below the sphere by
        // its sagitta, so the FINE chunk (edge on the true surface) must reach down
        // past that chord. A relief-only skirt (WI 773) misses this and cracks open
        // into the concentric "cliffs" of WI 779. Verify the fine chunk's skirt
        // covers the sagitta of a one-level-coarser neighbour (twice the edge span).
        let radius = 730_000.0;
        let f = SurfaceField::new(7, radius);
        let fine = QuadNode {
            face: CubeFace::PosZ,
            level: 6,
            i: 20,
            j: 20,
        };
        let edge_len = fine.edge_len(radius);
        let relief = chunk_relief(&f, fine, DEFAULT_RESOLUTION);
        let depth = skirt_depth_for(relief, edge_len, radius);
        let coarse_sagitta = edge_sagitta(2.0 * edge_len, radius);
        assert!(
            depth >= coarse_sagitta,
            "skirt {depth} must cover coarse-neighbour sagitta {coarse_sagitta}"
        );
        // And the term is negligible at max LOD (tiny edges near the surface) so it
        // cannot revive the WI 773 waffle: a metre-scale edge yields a sub-metre
        // curvature contribution.
        assert!(
            edge_sagitta(4.0, radius) < 0.01,
            "curvature term must vanish for tiny edges (no WI 773 waffle regression)"
        );
    }

    #[test]
    fn skirt_vertices_inherit_border_normal_and_uv() {
        // WI 786: each skirt (dropped-ring) vertex takes its border vertex's normal + UV,
        // so an exposed wall shades like the surface (not a dark inward `-radial` band).
        let f = field();
        let node = QuadNode {
            face: CubeFace::PosZ,
            level: 4,
            i: 5,
            j: 6,
        };
        let res = 8u32; // even (build_chunk forces even; already even here)
        let m = build_chunk(&f, node, res);
        let n = res + 1;
        let vert_count = (n * n) as usize;
        let grid_idx = |a: u32, b: u32| (b * n + a) as usize;
        // Rebuild the border ring in add_skirt's order (bottom → right → top → left).
        let mut ring: Vec<usize> = Vec::new();
        for a in 0..=res {
            ring.push(grid_idx(a, 0));
        }
        for b in 1..=res {
            ring.push(grid_idx(res, b));
        }
        for a in (0..res).rev() {
            ring.push(grid_idx(a, res));
        }
        for b in (1..res).rev() {
            ring.push(grid_idx(0, b));
        }
        assert_eq!(
            m.normals.len(),
            vert_count + ring.len(),
            "grid + one skirt vertex per ring vertex"
        );
        for (k, &gi) in ring.iter().enumerate() {
            let sk = vert_count + k;
            assert_eq!(m.normals[sk], m.normals[gi], "skirt normal inherits border");
            assert_eq!(m.uvs[sk], m.uvs[gi], "skirt uv inherits border");
            // Sanity: border normals point outward (not the old inward `-dir`).
            let dir = direction(
                node.face,
                {
                    let (u0, u1, _, _) = node.uv_rect();
                    u0 + (u1 - u0) * (gi as u32 % n) as f64 / res as f64
                },
                {
                    let (_, _, v0, v1) = node.uv_rect();
                    v0 + (v1 - v0) * (gi as u32 / n) as f64 / res as f64
                },
            );
            let normal = Vec3::from_array(m.normals[sk]).as_dvec3();
            assert!(
                normal.dot(dir) > 0.0,
                "skirt normal faces outward like the surface"
            );
        }
    }

    #[test]
    fn chunk_has_no_degenerate_triangles_and_finite_verts() {
        let f = field();
        let node = QuadNode {
            face: CubeFace::PosX,
            level: 4,
            i: 3,
            j: 7,
        };
        let m = build_chunk(&f, node, 12);
        assert!(m.center.is_finite());
        for p in &m.positions {
            assert!(p.iter().all(|c| c.is_finite()));
        }
        for n in &m.normals {
            let v = Vec3::from_array(*n);
            assert!((v.length() - 1.0).abs() < 1e-3, "normal not unit: {v:?}");
        }
        assert_eq!(m.indices.len() % 3, 0);
        for tri in m.indices.chunks(3) {
            let a = Vec3::from_array(m.positions[tri[0] as usize]);
            let b = Vec3::from_array(m.positions[tri[1] as usize]);
            let c = Vec3::from_array(m.positions[tri[2] as usize]);
            let area2 = (b - a).cross(c - a).length();
            assert!(area2 > 0.0, "degenerate triangle: {a:?} {b:?} {c:?}");
        }
    }

    #[test]
    fn positions_are_small_relative_to_center() {
        // Node-centre-relative positions stay f32-safe even at planetary radius.
        let f = SurfaceField::new(9, 6_360_000.0);
        let node = QuadNode {
            face: CubeFace::NegY,
            level: 10,
            i: 200,
            j: 511,
        };
        let m = build_chunk(&f, node, 16);
        for p in &m.positions {
            let mag = Vec3::from_array(*p).length();
            assert!(mag < 1.0e5, "position too large for f32 precision: {mag}");
        }
    }

    #[test]
    fn split_near_camera_true_far_camera_false() {
        let f = field();
        let root = QuadNode::root(CubeFace::PosX);
        // Camera hovering just above the node centre → split.
        let near = root.center_dir() * (R + 1_000.0);
        assert!(should_split(&f, root, near, DEFAULT_MAX_LEVEL));
        // Camera far out in orbit → keep (root).
        let far = root.center_dir() * (R * 20.0);
        assert!(!should_split(&f, root, far, DEFAULT_MAX_LEVEL));
        // At the max level, never split.
        let leaf = QuadNode {
            face: CubeFace::PosX,
            level: 2,
            i: 1,
            j: 1,
        };
        assert!(!should_split(&f, leaf, leaf.center_dir() * (R + 1.0), 2));
    }

    #[test]
    fn weld_factor_forces_edges_and_fades_over_the_band() {
        let res = 24u32;
        // No masks: the distance factor passes through untouched.
        assert_eq!(weld_factor(0.37, 5, 7, res, 0, 0), 0.37);
        // A coarser edge (bit 0 = v0, b = 0) forces its boundary vertices to 1
        // regardless of the distance factor, fading out over the band.
        assert_eq!(weld_factor(0.37, 5, 0, res, 1, 0), 1.0);
        let mid = weld_factor(0.0, 5, 3, res, 1, 0);
        assert!(mid > 0.0 && mid < 1.0, "inside the band: partial ({mid})");
        assert_eq!(
            weld_factor(0.37, 5, WELD_BAND_ROWS as u32 + 1, res, 1, 0),
            0.37,
            "beyond the band: untouched"
        );
        // A finer edge forces to 0 on the boundary.
        assert_eq!(weld_factor(0.83, 5, 0, res, 0, 1), 0.0);
        // Coarser wins where both constrain the same vertex.
        assert_eq!(weld_factor(0.5, 0, 0, res, 0b1000, 0b0001), 1.0);
    }

    #[test]
    fn weld_corner_profile_matches_the_edge_profile_on_a_shared_edge() {
        // The cross-chunk consistency property behind the corner bits: for vertices
        // on the v0 edge (b = 0), a chunk whose u0 edge (bit 3) is flagged and its
        // same-level neighbour whose only knowledge is the diagonal corner c00
        // (bit 4) must compute identical factors — otherwise the weld would break
        // the shared edge it exists to protect.
        let res = 24u32;
        for a in 0..=res {
            let via_edge = weld_factor(0.42, a, 0, res, 0b0000_1000, 0);
            let via_corner = weld_factor(0.42, a, 0, res, 0b0001_0000, 0);
            assert!(
                (via_edge - via_corner).abs() < 1e-12,
                "a={a}: edge profile {via_edge} != corner profile {via_corner}"
            );
        }
    }

    #[test]
    fn atmosphere_params_airless_is_none_atmo_maps_radius() {
        // Earth-like (has atmosphere) → Some with bottom_radius = body radius.
        let earth = BodyAsset::earthlike();
        let p = AtmosphereParams::from_asset(&earth).expect("earthlike has atmosphere");
        assert_eq!(p.bottom_radius, earth.radius as f32);
        assert!(p.top_radius > p.bottom_radius);
        // An airless body → None.
        let mut airless = BodyAsset::earthlike();
        airless.fluid_medium.atmosphere_surface_density = 0.0;
        assert!(AtmosphereParams::from_asset(&airless).is_none());
    }
}

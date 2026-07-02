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
    /// Triangle indices (three per triangle).
    pub indices: Vec<u32>,
}

/// The `LOD` split decision: split `node` (subdivide) when the camera is close
/// enough that the node's world edge subtends more than the range factor allows,
/// bounded by `max_level`. Pure function of node + camera world position.
pub fn should_split(node: QuadNode, camera_world: DVec3, radius: f64, max_level: u32) -> bool {
    if node.level >= max_level {
        return false;
    }
    let center_world = node.center_dir() * radius;
    let dist = (camera_world - center_world).length();
    let size = node.edge_len(radius);
    dist < size * SPLIT_RANGE_FACTOR
}

/// Fraction of a chunk's resident distance band (nearest → merge) over which it
/// morphs toward its parent. The factor is 0 up to `(1 − MORPH_REGION)` of the band
/// and ramps to 1 at the far (merge) edge — so a chunk is fully morphed to the coarse
/// shape exactly where it borders a one-level-coarser neighbour (seamless seam) and
/// where it merges into its parent (no pop), and its finer children spawn fully
/// morphed at their far edge (no pop on split).
pub const MORPH_REGION: f64 = 0.35;

/// The camera-distance `(start, end)` (metres) of the CDLOD morph ramp for a chunk of
/// world edge `edge_len`. The chunk's resident band is `[R·edge, 2·R·edge]`
/// (`R = SPLIT_RANGE_FACTOR`); morph is 0 up to `start` and ramps to 1 at `end` (the far
/// / merge edge), covering the top `MORPH_REGION` of the band. The render vertex shader
/// applies `smoothstep(start, end, per_vertex_distance)` so the factor is continuous in
/// space — shared edge vertices (same distance, same range) get the same factor and
/// stay matched, and a chunk reaches full morph exactly where it borders a coarser
/// neighbour or merges.
pub fn morph_range(edge_len: f64) -> (f32, f32) {
    let near = SPLIT_RANGE_FACTOR * edge_len;
    let far = 2.0 * near; // = SPLIT_RANGE_FACTOR * parent edge_len
    let start = far - MORPH_REGION * (far - near);
    (start as f32, far as f32)
}

/// A representative world edge length (metres) for all chunks at `level`, sampled at a
/// mid-face node so the per-level morph ramp is shared by every chunk of that level —
/// so same-level neighbours use an identical ramp and their shared edges match exactly.
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
            normals.push(field.normal(dir).as_vec3().to_array());
            uvs.push(Vec2::new(tu as f32, tv as f32).to_array());
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
        &mut indices,
    );

    // Skirt vertices do not morph: their target is their own position, so the morph
    // blend is a no-op on the skirt regardless of the chunk's morph factor.
    morph_targets.extend_from_slice(&positions[vert_count..]);

    ChunkMesh {
        center,
        positions,
        normals,
        uvs,
        morph_targets,
        indices,
    }
}

/// Appends the border skirt: for each border grid vertex, a duplicate pushed
/// radially inward by `skirt_depth`; walls connect consecutive border vertices to
/// their skirt duplicates. The wall's `-radial` drop hides LOD-boundary gaps.
#[allow(clippy::too_many_arguments)]
fn add_skirt(
    res: u32,
    world_of: &[(DVec3, DVec3)],
    center: DVec3,
    skirt_depth: f64,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
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

    // A skirt duplicate for each ring vertex, pushed inward along its direction.
    let skirt_base = positions.len() as u32;
    for &gi in &ring {
        let (dir, world) = world_of[gi];
        let dropped = world - dir * skirt_depth;
        positions.push((dropped - center).as_vec3().to_array());
        normals.push((-dir).as_vec3().to_array());
        uvs.push([0.0, 0.0]);
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
    use crate::body_asset::BodyAsset;

    const R: f64 = 1_000_000.0;

    fn field() -> SurfaceField {
        SurfaceField::new(1234, R)
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
    fn morph_range_brackets_the_top_of_the_resident_band() {
        // Resident band is [2.5·edge, 5·edge]; the ramp ends at the far/merge edge and
        // starts MORPH_REGION back from it, entirely inside the band.
        let edge = 1000.0;
        let near = SPLIT_RANGE_FACTOR * edge;
        let far = 2.0 * near;
        let (start, end) = morph_range(edge);
        assert!(
            (end as f64 - far).abs() < 1e-3,
            "ramp ends at the merge edge"
        );
        assert!(
            start as f64 > near,
            "ramp starts inside the band (after near)"
        );
        assert!(start < end, "start precedes end");
        let expected_start = far - MORPH_REGION * (far - near);
        assert!((start as f64 - expected_start).abs() < 1e-3);
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
        let root = QuadNode::root(CubeFace::PosX);
        // Camera hovering just above the node centre → split.
        let near = root.center_dir() * (R + 1_000.0);
        assert!(should_split(root, near, R, DEFAULT_MAX_LEVEL));
        // Camera far out in orbit → keep (root).
        let far = root.center_dir() * (R * 20.0);
        assert!(!should_split(root, far, R, DEFAULT_MAX_LEVEL));
        // At the max level, never split.
        let leaf = QuadNode {
            face: CubeFace::PosX,
            level: 2,
            i: 1,
            j: 1,
        };
        assert!(!should_split(leaf, leaf.center_dir() * (R + 1.0), R, 2));
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

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
//!   **skirt**: its border ring is duplicated and pushed radially inward by a
//!   depth proportional to the node's world edge length, so the coarser side of a
//!   boundary (the longer edge, the larger deviation) always grows a proportionally
//!   deeper wall that covers the gap. The traversal therefore need not keep the
//!   quadtree 2:1-balanced.
//!
//! **Precision.** A chunk's vertex positions are `f32` **relative to the node's
//! centre world point** (returned separately as `f64`), so per-vertex values stay
//! small regardless of body radius; absolute placement is the floating origin's job.

use crate::surface_field::SurfaceField;
use glam::{DVec3, Vec2, Vec3};

/// Skirt depth as a fraction of a node's world edge length.
const SKIRT_FACTOR: f64 = 0.5;
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

/// A built chunk's render buffers. Positions/normals/UVs are parallel arrays;
/// `indices` triangulates them. Positions are `f32` **relative to `center`**.
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

/// Builds the chunk mesh for `node` at grid resolution `res` (quads per side),
/// sampling `field`. Deterministic and pure: identical inputs → identical buffers.
pub fn build_chunk(field: &SurfaceField, node: QuadNode, res: u32) -> ChunkMesh {
    let res = res.max(1);
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

    // Surface grid: sample the field at each (u, v) direction.
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let mut world_of = Vec::with_capacity(vert_count); // f64 world points, for skirt reuse
    for b in 0..=res {
        let tv = b as f64 / res as f64;
        let vv = lerp(v0, v1, tv);
        for a in 0..=res {
            let tu = a as f64 / res as f64;
            let uu = lerp(u0, u1, tu);
            let dir = direction(node.face, uu, vv);
            let elev = field.elevation(dir);
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

    // Skirt: a wall hanging inward from each border edge, deep enough to cover the
    // coarser-neighbour gap (depth scales with the node's world edge length).
    let skirt_depth = SKIRT_FACTOR * node.edge_len(radius);
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

    ChunkMesh {
        center,
        positions,
        normals,
        uvs,
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
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.center, b.center);
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
        let skirt_depth = SKIRT_FACTOR * coarse.edge_len(R);
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

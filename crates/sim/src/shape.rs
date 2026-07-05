//! The shape catalog (WI 831, shaped-blocks design stage 1): **geometry-first
//! forms**. Each catalog entry is one canonical closed triangle mesh inside the
//! unit cell — the *only* authored content — and every physical constant a
//! derivation needs (volume fraction, centroid, inertia, per-face coverage,
//! shell area) is **derived from that mesh** by one shared routine
//! ([`derive_form`]). Hand-authored physics constants are prohibited: the thing
//! you see is provably the thing that weighs, floats, and (in later stages)
//! seals and breaks, because they are the same polyhedron.
//!
//! **Admission invariants** (design review R1), verified at derivation: the mesh
//! is a closed 2-manifold of positive volume; the solid is a single connected
//! region; the air remainder is at most a single connected region that reaches
//! every uncovered face portion; and the derived face coverage agrees with the
//! sampled geometry (the catalog cannot disagree with its own shape). A mesh
//! violating any of these is rejected with the named violation.
//!
//! **Orientation** is one of the 24 proper cube rotations, applied about the
//! cell centre. The [`rotations`] table order is **frozen** — orientation
//! indices are persisted data (index 0 = identity, pinned by test). Each form's
//! distinct-orientation set is derived by congruence deduplication, never
//! authored.
//!
//! Staged rollout: WI 831 made shapes exist, weigh, and float (mass / inertia /
//! displacement / breakage-load folds); WI 832 made them **seal** — per-face
//! coverage masks flow through the one `boundary_sealed` predicate and the
//! compartment flood walks air fractions. The aero envelope treats shaped
//! cells as full cubes until WI 834, breakage *bonds* until WI 835, shell
//! physics until WI 836, and collision stays the full cell box by design
//! (review R3).

use glam::{DMat3, DVec3, IVec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// A catalog form: which sub-cube polyhedron occupies a shaped cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Form {
    /// The full cell — the catalog's trivial entry (an unshaped cell *is* this).
    Cube,
    /// Half cube: the ramp `y ≤ z` (rises along +Z, hypotenuse facing +Y/−Z).
    Wedge,
    /// The corner tetrahedron at the cell origin (`x + y + z ≤ 1`), volume 1/6.
    OuterCorner,
    /// The outer corner's complement (`x + y + z ≥ 1`), volume 5/6.
    InnerCorner,
    /// The 1×2 shallow-slope pair, low half: `y ≤ z/2`, volume 1/4.
    SlopeLow,
    /// The 1×2 shallow-slope pair, high half: `y ≤ (1 + z)/2`, volume 3/4.
    /// Its `z = 0` profile mates [`Form::SlopeLow`]'s `z = 1` profile.
    SlopeHigh,
}

/// Every catalog form, in catalog order.
pub const FORMS: [Form; 6] = [
    Form::Cube,
    Form::Wedge,
    Form::OuterCorner,
    Form::InnerCorner,
    Form::SlopeLow,
    Form::SlopeHigh,
];

/// A shaped cell's fill mode. `Shell` is schema-present per the approved
/// decomposition but physically consumed in WI 836 — until then a shell derives
/// as its solid form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FillMode {
    #[default]
    Solid,
    Shell,
}

/// One shaped cell: the occupied cell it shapes, its form, its orientation (an
/// index into [`rotations`] — a **frozen** serialization contract), and its fill
/// mode. Stored on the craft as a sorted sidecar keyed by cell (the
/// `face_panels` pattern); an occupied cell without an entry is a full solid
/// cube.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapedCell {
    /// The occupied cell this shapes.
    pub cell: IVec3,
    /// The catalog form.
    pub form: Form,
    /// Orientation: index into the frozen [`rotations`] table (0 = identity).
    #[serde(default)]
    pub orientation: u8,
    /// Solid or (WI 836) shell.
    #[serde(default)]
    pub fill: FillMode,
}

impl ShapedCell {
    /// Deterministic sort key (the WI 820 ordered-encode discipline).
    pub(crate) fn key(&self) -> (i32, i32, i32) {
        (self.cell.x, self.cell.y, self.cell.z)
    }
}

/// The derived constants of one form — **all** computed from its canonical mesh
/// by [`derive_form`]; nothing here is authored.
#[derive(Clone, Debug)]
pub struct FormConstants {
    /// Which form these constants describe.
    pub form: Form,
    /// Solid volume as a fraction of the unit cell, (0, 1].
    pub volume: f64,
    /// Centroid in unit-cell coordinates.
    pub centroid: DVec3,
    /// Inertia tensor about the centroid for unit density and unit cell size
    /// (mass = `volume`); scale by `ρ·s⁵` (and rotate) at use.
    pub unit_inertia: DMat3,
    /// Covered area fraction of each unit-cube face, in the order
    /// `[x=0, x=1, y=0, y=1, z=0, z=1]`.
    pub face_coverage: [f64; 6],
    /// Total boundary surface area (unit cell) — the shell substrate (WI 836).
    pub shell_area: f64,
    /// Derived distinct orientations: the first rotation index of each
    /// congruence class (cube = `[0]`; wedge has 12).
    pub distinct_orientations: Vec<u8>,
}

impl FormConstants {
    /// The centroid under `orientation`, unit-cell coordinates (rotation about
    /// the cell centre).
    pub fn centroid_oriented(&self, orientation: u8) -> DVec3 {
        let r = rotations()[orientation as usize];
        r * (self.centroid - DVec3::splat(0.5)) + DVec3::splat(0.5)
    }

    /// The unit inertia tensor under `orientation` (`R·I·Rᵀ`).
    pub fn unit_inertia_oriented(&self, orientation: u8) -> DMat3 {
        let r = rotations()[orientation as usize];
        r * self.unit_inertia * r.transpose()
    }
}

/// The derived constants of `form`, from the process-wide catalog.
pub fn constants(form: Form) -> &'static FormConstants {
    let idx = FORMS.iter().position(|f| *f == form).expect("catalog form");
    &catalog()[idx]
}

/// The full catalog, derived once per process (and logged at debug level so a
/// mis-derived form is visible as one line, not a physics mystery). Panics if a
/// *shipped* form fails admission — that is a programmer error in the mesh, not
/// a runtime condition.
pub fn catalog() -> &'static [FormConstants; 6] {
    static CATALOG: OnceLock<[FormConstants; 6]> = OnceLock::new();
    CATALOG.get_or_init(|| {
        FORMS.map(|form| {
            let c = derive_form(form, &form_mesh(form))
                .unwrap_or_else(|e| panic!("shipped form {form:?} failed admission: {e}"));
            bevy_log::debug!(
                "shape catalog: {:?} volume {:.6} centroid ({:.4},{:.4},{:.4}) \
                 coverage {:?} shell {:.4} distinct {}",
                c.form,
                c.volume,
                c.centroid.x,
                c.centroid.y,
                c.centroid.z,
                c.face_coverage.map(|f| (f * 1e4).round() / 1e4),
                c.shell_area,
                c.distinct_orientations.len(),
            );
            c
        })
    })
}

/// The 24 proper rotations of the cube, as matrices, in a **frozen** order
/// (index 0 = identity): axis permutations in a fixed nested order × sign
/// combinations `+` before `−`, keeping determinant +1. The order is a
/// serialization contract — [`ShapedCell::orientation`] indexes this table.
pub fn rotations() -> &'static [DMat3; 24] {
    static ROTATIONS: OnceLock<[DMat3; 24]> = OnceLock::new();
    ROTATIONS.get_or_init(|| {
        let perms: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut out = Vec::with_capacity(24);
        for perm in perms {
            for signs in [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ] {
                // Column j maps e_j -> signs[j] · e_{perm[j]}.
                let mut cols = [DVec3::ZERO; 3];
                for (j, col) in cols.iter_mut().enumerate() {
                    let mut v = DVec3::ZERO;
                    v[perm[j]] = signs[j];
                    *col = v;
                }
                let m = DMat3::from_cols(cols[0], cols[1], cols[2]);
                if (m.determinant() - 1.0).abs() < 1e-9 {
                    out.push(m);
                }
            }
        }
        out.try_into().expect("exactly 24 proper rotations")
    })
}

/// One form's canonical mesh: vertices in the unit cell and outward-wound
/// triangles. The only authored content in the catalog.
pub(crate) struct FormMesh {
    pub vertices: Vec<DVec3>,
    pub triangles: Vec<[usize; 3]>,
}

/// The canonical mesh of `form` (quads authored as two triangles).
pub(crate) fn form_mesh(form: Form) -> FormMesh {
    let v = |x: f64, y: f64, z: f64| DVec3::new(x, y, z);
    match form {
        Form::Cube => FormMesh {
            vertices: vec![
                v(0., 0., 0.),
                v(1., 0., 0.),
                v(1., 1., 0.),
                v(0., 1., 0.),
                v(0., 0., 1.),
                v(1., 0., 1.),
                v(1., 1., 1.),
                v(0., 1., 1.),
            ],
            triangles: vec![
                [0, 3, 2],
                [0, 2, 1], // z = 0 (−Z out)
                [4, 5, 6],
                [4, 6, 7], // z = 1 (+Z out)
                [0, 1, 5],
                [0, 5, 4], // y = 0 (−Y out)
                [3, 7, 6],
                [3, 6, 2], // y = 1 (+Y out)
                [0, 4, 7],
                [0, 7, 3], // x = 0 (−X out)
                [1, 2, 6],
                [1, 6, 5], // x = 1 (+X out)
            ],
        },
        // The ramp y ≤ z: bottom + back full, sides half, hypotenuse y = z.
        Form::Wedge => FormMesh {
            vertices: vec![
                v(0., 0., 0.), // 0
                v(1., 0., 0.), // 1
                v(1., 0., 1.), // 2
                v(0., 0., 1.), // 3
                v(0., 1., 1.), // 4
                v(1., 1., 1.), // 5
            ],
            triangles: vec![
                [0, 1, 2],
                [0, 2, 3], // y = 0 (−Y out)
                [3, 2, 5],
                [3, 5, 4], // z = 1 (+Z out)
                [0, 4, 5],
                [0, 5, 1], // hypotenuse y = z (normal (0,1,−1))
                [0, 3, 4], // x = 0 (−X out)
                [1, 5, 2], // x = 1 (+X out)
            ],
        },
        // The corner tetrahedron x + y + z ≤ 1.
        Form::OuterCorner => FormMesh {
            vertices: vec![v(0., 0., 0.), v(1., 0., 0.), v(0., 1., 0.), v(0., 0., 1.)],
            triangles: vec![
                [0, 2, 1], // z = 0 (−Z out)
                [0, 1, 3], // y = 0 (−Y out)
                [0, 3, 2], // x = 0 (−X out)
                [1, 2, 3], // diagonal x + y + z = 1 (outward (1,1,1))
            ],
        },
        // The complement x + y + z ≥ 1.
        Form::InnerCorner => FormMesh {
            vertices: vec![
                v(1., 0., 0.), // 0
                v(0., 1., 0.), // 1
                v(0., 0., 1.), // 2
                v(1., 1., 0.), // 3
                v(1., 0., 1.), // 4
                v(0., 1., 1.), // 5
                v(1., 1., 1.), // 6
            ],
            triangles: vec![
                [0, 2, 1], // diagonal (outward −(1,1,1): toward the removed corner)
                [0, 1, 3], // z = 0 remainder (−Z out)
                [0, 4, 2], // y = 0 remainder (−Y out)
                [1, 2, 5], // x = 0 remainder (−X out)
                [0, 6, 4],
                [0, 3, 6], // x = 1 (+X out)
                [1, 6, 3],
                [1, 5, 6], // y = 1 (+Y out)
                [2, 6, 5],
                [2, 4, 6], // z = 1 (+Z out)
            ],
        },
        // The shallow pair, low half: y ≤ z/2.
        Form::SlopeLow => FormMesh {
            vertices: vec![
                v(0., 0., 0.),  // 0
                v(1., 0., 0.),  // 1
                v(1., 0., 1.),  // 2
                v(0., 0., 1.),  // 3
                v(0., 0.5, 1.), // 4
                v(1., 0.5, 1.), // 5
            ],
            triangles: vec![
                [0, 1, 2],
                [0, 2, 3], // y = 0 (−Y out)
                [3, 2, 5],
                [3, 5, 4], // z = 1 (+Z out)
                [0, 4, 5],
                [0, 5, 1], // slope y = z/2 (outward (0,2,−1)/√5)
                [0, 3, 4], // x = 0 (−X out)
                [1, 5, 2], // x = 1 (+X out)
            ],
        },
        // The shallow pair, high half: y ≤ (1 + z)/2.
        Form::SlopeHigh => FormMesh {
            vertices: vec![
                v(0., 0., 0.),  // 0
                v(1., 0., 0.),  // 1
                v(1., 0., 1.),  // 2
                v(0., 0., 1.),  // 3
                v(0., 0.5, 0.), // 4
                v(1., 0.5, 0.), // 5
                v(0., 1., 1.),  // 6
                v(1., 1., 1.),  // 7
            ],
            triangles: vec![
                [0, 1, 2],
                [0, 2, 3], // y = 0 (−Y out)
                [0, 4, 5],
                [0, 5, 1], // z = 0, y ≤ 1/2 (−Z out)
                [3, 2, 7],
                [3, 7, 6], // z = 1 (+Z out)
                [4, 6, 7],
                [4, 7, 5], // slope y = (1+z)/2 (outward (0,2,−1)/√5)
                [0, 3, 6],
                [0, 6, 4], // x = 0 (−X out)
                [1, 5, 7],
                [1, 7, 2], // x = 1 (+X out)
            ],
        },
    }
}

/// A face's coverage as a raster mask (WI 832): 16×16 jittered samples over the
/// face, one bit each — **derived** from the form's face-plane triangles, never
/// authored. The vocabulary's contract (design R4) is totality under the two
/// consumer operations: **seal composition** (`masks_seal` — do the two sides of
/// a boundary jointly cover it?) and, for WI 835, overlap area (bitwise AND
/// popcount). Bit `i·16 + j` samples in-face point `((i+½)/16 + JU, (j+½)/16 +
/// JV)` in the face's tangent axes (the fixed X→(Y,Z), Y→(X,Z), Z→(X,Y) order),
/// so the two cells facing one boundary index the **same physical point with the
/// same bit** — exactly-complementary coverages OR to full (the jitter
/// tie-breaks the shared diagonal to exactly one side).
pub type FaceMask = [u64; 4];
/// No coverage (an unoccupied side).
pub const MASK_EMPTY: FaceMask = [0; 4];
/// Full coverage (an unshaped solid side).
pub const MASK_FULL: FaceMask = [u64::MAX; 4];

/// Seal composition: the boundary is sealed when the two sides' coverages
/// jointly cover every sample.
pub fn masks_seal(a: &FaceMask, b: &FaceMask) -> bool {
    (0..4).all(|w| a[w] | b[w] == u64::MAX)
}

/// The number of covered samples in a mask (of 256).
pub fn mask_popcount(m: &FaceMask) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
}

/// The vocabulary's second consumer operation (design R4; WI 835): the
/// **overlap** of two coverages across a boundary — the number of samples
/// (of 256) both sides cover. Bond strength scales by this; exact complements
/// overlap zero (they partition the samples), so a mated complementary pair
/// seals air yet carries no structural bond.
pub fn masks_overlap(a: &FaceMask, b: &FaceMask) -> u32 {
    (0..4).map(|w| (a[w] & b[w]).count_ones()).sum()
}

/// In-face jitter for the mask sample grid (fixed; shared by every face so
/// boundary bits align — invariant, do not vary per face).
const MASK_JITTER_U: f64 = 7.548776662e-5;
const MASK_JITTER_V: f64 = 5.6984029e-5;
/// Mask resolution per axis.
const MASK_N: usize = 16;

/// The face masks of `form` under `orientation`, derived once per process for
/// all (form, orientation) pairs by rotating the canonical mesh and
/// re-rasterizing — one code path, so the table is rotation-closed by
/// construction. Face order `[x0, x1, y0, y1, z0, z1]`.
pub fn face_masks(form: Form, orientation: u8) -> &'static [FaceMask; 6] {
    static TABLE: OnceLock<Vec<[FaceMask; 6]>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut out = Vec::with_capacity(FORMS.len() * 24);
        for form in FORMS {
            let mesh = form_mesh(form);
            for r in rotations().iter() {
                let rotated: Vec<DVec3> = mesh
                    .vertices
                    .iter()
                    .map(|&p| *r * (p - DVec3::splat(0.5)) + DVec3::splat(0.5))
                    .collect();
                out.push(rasterize_face_masks(&rotated, &mesh.triangles));
            }
        }
        out
    });
    let fi = FORMS.iter().position(|f| *f == form).expect("catalog form");
    &table[fi * 24 + orientation as usize]
}

/// The oriented **crease-edge outline** of a form (WI 833): the canonical mesh's
/// undirected edges whose two adjacent triangles are non-coplanar (triangulation
/// diagonals inside flat faces are dropped), rotated about the cell centre —
/// the wireframe the editor's placement ghost draws, in unit-cell coordinates.
/// Derived from the same mesh the skins emit, so the preview cannot drift from
/// the render. Deterministic order (sorted by vertex index pair).
pub fn form_outline(form: Form, orientation: u8) -> Vec<(DVec3, DVec3)> {
    let mesh = form_mesh(form);
    let mut edge_normals: HashMap<(usize, usize), Vec<DVec3>> = HashMap::new();
    for t in &mesh.triangles {
        let (a, b, c) = (
            mesh.vertices[t[0]],
            mesh.vertices[t[1]],
            mesh.vertices[t[2]],
        );
        let n = (b - a).cross(c - a).normalize();
        for (i, j) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edge_normals
                .entry((i.min(j), i.max(j)))
                .or_default()
                .push(n);
        }
    }
    let mut keys: Vec<(usize, usize)> = edge_normals
        .iter()
        // A manifold edge has exactly two triangles; keep it when they crease.
        .filter(|(_, ns)| ns.len() != 2 || ns[0].dot(ns[1]) < 1.0 - 1e-6)
        .map(|(k, _)| *k)
        .collect();
    keys.sort_unstable();
    let r = rotations()[orientation as usize];
    let orient = |p: DVec3| r * (p - DVec3::splat(0.5)) + DVec3::splat(0.5);
    keys.into_iter()
        .map(|(i, j)| (orient(mesh.vertices[i]), orient(mesh.vertices[j])))
        .collect()
}

/// Rasterize the six face masks of a (possibly rotated) mesh: for each unit-cube
/// face, collect the triangles lying wholly in its plane, project to the face's
/// tangent axes, and sample the jittered grid.
fn rasterize_face_masks(vertices: &[DVec3], triangles: &[[usize; 3]]) -> [FaceMask; 6] {
    let mut masks = [MASK_EMPTY; 6];
    for (face, axis, side) in [
        (0usize, 0usize, 0.0),
        (1, 0, 1.0),
        (2, 1, 0.0),
        (3, 1, 1.0),
        (4, 2, 0.0),
        (5, 2, 1.0),
    ] {
        let (t1, t2) = match axis {
            0 => (1, 2), // X face: (y, z)
            1 => (0, 2), // Y face: (x, z)
            _ => (0, 1), // Z face: (x, y)
        };
        // Triangles wholly in this face plane, projected to (t1, t2).
        let tris: Vec<[(f64, f64); 3]> = triangles
            .iter()
            .filter(|t| {
                t.iter()
                    .all(|&i| (vertices[i][axis] - side).abs() < GEOM_EPS)
            })
            .map(|t| t.map(|i| (vertices[i][t1], vertices[i][t2])))
            .collect();
        if tris.is_empty() {
            continue;
        }
        let mut mask = MASK_EMPTY;
        for i in 0..MASK_N {
            for j in 0..MASK_N {
                let u = (i as f64 + 0.5) / MASK_N as f64 + MASK_JITTER_U;
                let v = (j as f64 + 0.5) / MASK_N as f64 + MASK_JITTER_V;
                if tris.iter().any(|t| point_in_triangle_2d(u, v, t)) {
                    let bit = i * MASK_N + j;
                    mask[bit / 64] |= 1u64 << (bit % 64);
                }
            }
        }
        masks[face] = mask;
    }
    masks
}

/// 2D point-in-triangle by consistent edge signs (winding-agnostic).
fn point_in_triangle_2d(u: f64, v: f64, t: &[(f64, f64); 3]) -> bool {
    let sign = |a: (f64, f64), b: (f64, f64)| (b.0 - a.0) * (v - a.1) - (b.1 - a.1) * (u - a.0);
    let s0 = sign(t[0], t[1]);
    let s1 = sign(t[1], t[2]);
    let s2 = sign(t[2], t[0]);
    (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0) || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0)
}

/// Derivation-time sampling resolution (per axis) for the admission checks.
const SAMPLE_N: usize = 16;
/// Fixed sub-grid jitter so no sample lies exactly on a form plane
/// (deterministic; irrational-ish per axis).
const SAMPLE_JITTER: DVec3 = DVec3::new(7.548776662e-5, 5.6984029e-5, 3.6630769e-5);
/// Tolerance for the coverage-vs-sampling cross-check (a boundary sample layer
/// sits half a sample step from the face, so sloped forms shift the estimate).
const COVERAGE_TOLERANCE: f64 = 0.08;
/// Vertex-position tolerance for face-plane membership and congruence hashing.
const GEOM_EPS: f64 = 1e-9;

/// Derive one form's constants from its canonical mesh, verifying the admission
/// invariants (design R1). Errors name the violation; the shipped catalog treats
/// an error as a programmer bug, tests exercise the rejection path directly.
pub(crate) fn derive_form(form: Form, mesh: &FormMesh) -> Result<FormConstants, String> {
    // --- Closed 2-manifold: every directed edge appears exactly once, paired
    // with its reverse.
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    for t in &mesh.triangles {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *directed.entry((a, b)).or_default() += 1;
        }
    }
    for (&(a, b), &n) in &directed {
        if n != 1 {
            return Err(format!(
                "directed edge ({a},{b}) appears {n} times (not a manifold)"
            ));
        }
        if !directed.contains_key(&(b, a)) {
            return Err(format!("edge ({a},{b}) has no paired reverse (open mesh)"));
        }
    }

    // --- Volume, centroid, second moments via signed origin-tetrahedra
    // (divergence theorem; exact for polyhedra).
    let mut volume = 0.0;
    let mut first = DVec3::ZERO; // ∫ x dV
    let mut s = [[0.0_f64; 3]; 3]; // S_ij = ∫ x_i x_j dV
    for t in &mesh.triangles {
        let (a, b, c) = (
            mesh.vertices[t[0]],
            mesh.vertices[t[1]],
            mesh.vertices[t[2]],
        );
        let vt = a.dot(b.cross(c)) / 6.0; // signed tet (0,a,b,c) volume
        volume += vt;
        first += vt * (a + b + c) / 4.0;
        // S_ij over the tet: (V/20)·(Σ_k v_ki·v_kj + Σ_k v_ki · Σ_k v_kj),
        // vertices (0, a, b, c).
        let verts = [DVec3::ZERO, a, b, c];
        let sum: DVec3 = a + b + c;
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = verts.iter().map(|v| v[i] * v[j]).sum();
                s[i][j] += vt / 20.0 * (dot + sum[i] * sum[j]);
            }
        }
    }
    if volume <= GEOM_EPS {
        return Err(format!(
            "non-positive volume {volume} (inward winding or degenerate)"
        ));
    }
    if volume > 1.0 + 1e-9 {
        return Err(format!("volume {volume} exceeds the unit cell"));
    }
    let centroid = first / volume;
    if !(0.0..=1.0).contains(&centroid.x)
        || !(0.0..=1.0).contains(&centroid.y)
        || !(0.0..=1.0).contains(&centroid.z)
    {
        return Err(format!("centroid {centroid} outside the unit cell"));
    }
    // Inertia about the centroid (unit density; tensor convention, products
    // negative), from the second moments by the parallel-axis shift.
    let m = volume;
    let (cx, cy, cz) = (centroid.x, centroid.y, centroid.z);
    let ixx = (s[1][1] + s[2][2]) - m * (cy * cy + cz * cz);
    let iyy = (s[0][0] + s[2][2]) - m * (cx * cx + cz * cz);
    let izz = (s[0][0] + s[1][1]) - m * (cx * cx + cy * cy);
    let ixy = -(s[0][1] - m * cx * cy);
    let ixz = -(s[0][2] - m * cx * cz);
    let iyz = -(s[1][2] - m * cy * cz);
    let unit_inertia = DMat3::from_cols(
        DVec3::new(ixx, ixy, ixz),
        DVec3::new(ixy, iyy, iyz),
        DVec3::new(ixz, iyz, izz),
    );

    // --- Per-face coverage fractions + shell area.
    let mut face_coverage = [0.0_f64; 6];
    let mut shell_area = 0.0;
    for t in &mesh.triangles {
        let (a, b, c) = (
            mesh.vertices[t[0]],
            mesh.vertices[t[1]],
            mesh.vertices[t[2]],
        );
        let area = 0.5 * (b - a).cross(c - a).length();
        shell_area += area;
        for (face, axis, side) in [
            (0usize, 0usize, 0.0),
            (1, 0, 1.0),
            (2, 1, 0.0),
            (3, 1, 1.0),
            (4, 2, 0.0),
            (5, 2, 1.0),
        ] {
            if (a[axis] - side).abs() < GEOM_EPS
                && (b[axis] - side).abs() < GEOM_EPS
                && (c[axis] - side).abs() < GEOM_EPS
            {
                face_coverage[face] += area;
            }
        }
    }

    // --- Mask cross-check (WI 832): the raster face masks must agree with the
    // analytically-derived coverage fractions — the seal vocabulary cannot
    // disagree with the geometry it was derived from.
    let masks = rasterize_face_masks(&mesh.vertices, &mesh.triangles);
    for (face, mask) in masks.iter().enumerate() {
        let sampled = mask_popcount(mask) as f64 / (MASK_N * MASK_N) as f64;
        if (sampled - face_coverage[face]).abs() > 0.05 {
            return Err(format!(
                "face {face}: raster mask covers {sampled:.3} but derived coverage is {:.3}",
                face_coverage[face]
            ));
        }
    }

    // --- Sampling verification: solid connected, air connected, air reaches
    // every open face, and sampled openness agrees with derived coverage.
    let n = SAMPLE_N;
    let sample_point = |i: usize, j: usize, k: usize| {
        DVec3::new(
            (i as f64 + 0.5) / n as f64,
            (j as f64 + 0.5) / n as f64,
            (k as f64 + 0.5) / n as f64,
        ) + SAMPLE_JITTER
    };
    let mut inside = vec![false; n * n * n];
    let idx = |i: usize, j: usize, k: usize| (i * n + j) * n + k;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                inside[idx(i, j, k)] = point_inside(mesh, sample_point(i, j, k));
            }
        }
    }
    let components = |want_inside: bool| -> usize {
        let mut seen = vec![false; n * n * n];
        let mut count = 0;
        for start in 0..n * n * n {
            if seen[start] || inside[start] != want_inside {
                continue;
            }
            count += 1;
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(p) = stack.pop() {
                let (i, j, k) = (p / (n * n), (p / n) % n, p % n);
                let mut push = |q: usize| {
                    if !seen[q] && inside[q] == want_inside {
                        seen[q] = true;
                        stack.push(q);
                    }
                };
                if i > 0 {
                    push(idx(i - 1, j, k));
                }
                if i + 1 < n {
                    push(idx(i + 1, j, k));
                }
                if j > 0 {
                    push(idx(i, j - 1, k));
                }
                if j + 1 < n {
                    push(idx(i, j + 1, k));
                }
                if k > 0 {
                    push(idx(i, j, k - 1));
                }
                if k + 1 < n {
                    push(idx(i, j, k + 1));
                }
            }
        }
        count
    };
    if components(true) != 1 {
        return Err("solid is not a single connected region".into());
    }
    if components(false) > 1 {
        return Err("air remainder is not a single connected region".into());
    }
    // Boundary layers: face order [x0, x1, y0, y1, z0, z1].
    for (face, &coverage) in face_coverage.iter().enumerate() {
        let mut air_in_layer = 0usize;
        for a in 0..n {
            for b in 0..n {
                let (i, j, k) = match face {
                    0 => (0, a, b),
                    1 => (n - 1, a, b),
                    2 => (a, 0, b),
                    3 => (a, n - 1, b),
                    4 => (a, b, 0),
                    5 => (a, b, n - 1),
                    _ => unreachable!(),
                };
                if !inside[idx(i, j, k)] {
                    air_in_layer += 1;
                }
            }
        }
        let sampled_open = air_in_layer as f64 / (n * n) as f64;
        let derived_open = 1.0 - coverage;
        if (sampled_open - derived_open).abs() > COVERAGE_TOLERANCE {
            return Err(format!(
                "face {face}: sampled open fraction {sampled_open:.3} disagrees with \
                 derived coverage {coverage:.3}"
            ));
        }
        if derived_open > COVERAGE_TOLERANCE && air_in_layer == 0 {
            return Err(format!("face {face} is open but no air sample reaches it"));
        }
    }

    // --- Distinct orientations by congruence dedup over the frozen table.
    let mut seen: HashSet<Vec<(i64, i64, i64)>> = HashSet::new();
    let mut distinct = Vec::new();
    for (o, r) in rotations().iter().enumerate() {
        let mut sig: Vec<(i64, i64, i64)> = mesh
            .vertices
            .iter()
            .map(|&p| {
                let q = *r * (p - DVec3::splat(0.5)) + DVec3::splat(0.5);
                (
                    (q.x * 1e6).round() as i64,
                    (q.y * 1e6).round() as i64,
                    (q.z * 1e6).round() as i64,
                )
            })
            .collect();
        sig.sort_unstable();
        if seen.insert(sig) {
            distinct.push(o as u8);
        }
    }

    Ok(FormConstants {
        form,
        volume,
        centroid,
        unit_inertia,
        face_coverage,
        shell_area,
        distinct_orientations: distinct,
    })
}

/// Point-in-polyhedron by ray parity along +X (Möller–Trumbore per triangle).
/// The caller jitters sample points off the form's planes, so degenerate
/// grazing hits do not arise for catalog meshes.
fn point_inside(mesh: &FormMesh, p: DVec3) -> bool {
    let dir = DVec3::X;
    let mut crossings = 0;
    for t in &mesh.triangles {
        let (a, b, c) = (
            mesh.vertices[t[0]],
            mesh.vertices[t[1]],
            mesh.vertices[t[2]],
        );
        let e1 = b - a;
        let e2 = c - a;
        let h = dir.cross(e2);
        let det = e1.dot(h);
        if det.abs() < 1e-14 {
            continue; // ray parallel to the triangle plane
        }
        let inv = 1.0 / det;
        let s = p - a;
        let u = s.dot(h) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = dir.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t_hit = e2.dot(q) * inv;
        if t_hit > 1e-12 {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rotation_table_is_a_frozen_contract() {
        let rots = rotations();
        assert_eq!(rots.len(), 24);
        // Index 0 = identity — persisted orientation indices depend on this.
        assert!((rots[0] - DMat3::IDENTITY).abs_diff_eq(DMat3::ZERO, 1e-12));
        // Every entry is a proper rotation with integer entries.
        for r in rots.iter() {
            assert!((r.determinant() - 1.0).abs() < 1e-12);
            let should_be_identity = *r * r.transpose();
            assert!(should_be_identity.abs_diff_eq(DMat3::IDENTITY, 1e-12));
        }
        // All distinct.
        for i in 0..24 {
            for j in (i + 1)..24 {
                assert!(!rots[i].abs_diff_eq(rots[j], 1e-9), "{i} == {j}");
            }
        }
    }

    #[test]
    fn cube_constants_match_hand_values() {
        let c = constants(Form::Cube);
        assert!((c.volume - 1.0).abs() < 1e-12);
        assert!((c.centroid - DVec3::splat(0.5)).length() < 1e-12);
        // Unit cube about its centre: diag 1/6, products zero.
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 / 6.0 } else { 0.0 };
                assert!(
                    (c.unit_inertia.col(i)[j] - want).abs() < 1e-12,
                    "I[{i}][{j}]"
                );
            }
        }
        assert_eq!(c.face_coverage, [1.0; 6]);
        assert!((c.shell_area - 6.0).abs() < 1e-12);
        assert_eq!(
            c.distinct_orientations,
            vec![0],
            "a cube has one orientation"
        );
    }

    #[test]
    fn wedge_constants_match_hand_values() {
        // The ramp y ≤ z: volume 1/2; centroid (1/2, 1/3, 2/3); faces — bottom
        // (y=0) and back (z=1) full, sides (x=0, x=1) half, top and front empty.
        let c = constants(Form::Wedge);
        assert!((c.volume - 0.5).abs() < 1e-12);
        assert!((c.centroid - DVec3::new(0.5, 1.0 / 3.0, 2.0 / 3.0)).length() < 1e-12);
        let cov = c.face_coverage;
        assert!((cov[0] - 0.5).abs() < 1e-12 && (cov[1] - 0.5).abs() < 1e-12); // x sides
        assert!((cov[2] - 1.0).abs() < 1e-12, "bottom full");
        assert!(cov[3].abs() < 1e-12, "top empty");
        assert!(cov[4].abs() < 1e-12, "front (z=0) empty");
        assert!((cov[5] - 1.0).abs() < 1e-12, "back full");
        // Analytic prism inertia about the centroid (unit density, mass 1/2):
        // for the right prism over triangle {(y,z): y ≤ z} with unit x-extent:
        //   ∫x² = 1/6 (about centroid x: m/12 = 1/24? — computed from moments)
        // Cross-check numerically instead against independent second moments:
        //   S over the wedge: Sxx = m/3? Use direct integrals:
        //   ∫∫∫_{y≤z} x² = 1/2·1/3 = 1/6; ∫ y² = ∫_z z³/3 = 1/12; ∫ z² = 1/4;
        //   ∫ xy = 1/2·... (x independent: mean x = 1/2): ∫xy = 1/2·∫y = 1/2·(1/6)=1/12;
        //   ∫ y over region = ∫_z z²/2 = 1/6; ∫ z = ∫ z·z = 1/3; ∫ x = 1/2·1/2 = 1/4.
        //   ∫ yz = ∫_z z·z²/2 = 1/8; ∫ xz = 1/2·1/3 = 1/6.
        let m = 0.5;
        let (cx, cy, cz) = (0.5, 1.0 / 3.0, 2.0 / 3.0);
        let (sxx, syy, szz) = (1.0 / 6.0, 1.0 / 12.0, 1.0 / 4.0);
        let (sxy, sxz, syz) = (1.0 / 12.0, 1.0 / 6.0, 1.0 / 8.0);
        let want_ixx = (syy + szz) - m * (cy * cy + cz * cz);
        let want_iyy = (sxx + szz) - m * (cx * cx + cz * cz);
        let want_izz = (sxx + syy) - m * (cx * cx + cy * cy);
        let want_ixy = -(sxy - m * cx * cy);
        let want_ixz = -(sxz - m * cx * cz);
        let want_iyz = -(syz - m * cy * cz);
        let i = c.unit_inertia;
        assert!((i.col(0).x - want_ixx).abs() < 1e-12);
        assert!((i.col(1).y - want_iyy).abs() < 1e-12);
        assert!((i.col(2).z - want_izz).abs() < 1e-12);
        assert!((i.col(1).x - want_ixy).abs() < 1e-12);
        assert!((i.col(2).x - want_ixz).abs() < 1e-12);
        assert!((i.col(2).y - want_iyz).abs() < 1e-12);
        assert_eq!(c.distinct_orientations.len(), 12, "wedge symmetry order 2");
    }

    #[test]
    fn the_whole_catalog_is_admitted_with_expected_volumes() {
        let want = [
            (Form::Cube, 1.0),
            (Form::Wedge, 0.5),
            (Form::OuterCorner, 1.0 / 6.0),
            (Form::InnerCorner, 5.0 / 6.0),
            (Form::SlopeLow, 0.25),
            (Form::SlopeHigh, 0.75),
        ];
        for (form, volume) in want {
            let c = constants(form);
            assert!(
                (c.volume - volume).abs() < 1e-12,
                "{form:?}: {} vs {volume}",
                c.volume
            );
        }
        // The shallow pair mates: low's back profile equals high's front profile.
        let low = constants(Form::SlopeLow);
        let high = constants(Form::SlopeHigh);
        assert!(
            (low.face_coverage[5] - 0.5).abs() < 1e-12,
            "low back half-covered"
        );
        assert!(
            (high.face_coverage[4] - 0.5).abs() < 1e-12,
            "high front half-covered"
        );
    }

    #[test]
    fn derivation_rejects_invariant_violations() {
        // Open mesh: a cube missing a face.
        let mut open = form_mesh(Form::Cube);
        open.triangles.truncate(10);
        assert!(derive_form(Form::Cube, &open)
            .unwrap_err()
            .contains("open mesh"));

        // Disconnected solid: two separated slabs in one cell.
        let slab = |x0: f64, x1: f64| -> (Vec<DVec3>, Vec<[usize; 3]>) {
            let v = vec![
                DVec3::new(x0, 0., 0.),
                DVec3::new(x1, 0., 0.),
                DVec3::new(x1, 1., 0.),
                DVec3::new(x0, 1., 0.),
                DVec3::new(x0, 0., 1.),
                DVec3::new(x1, 0., 1.),
                DVec3::new(x1, 1., 1.),
                DVec3::new(x0, 1., 1.),
            ];
            let t = form_mesh(Form::Cube).triangles;
            (v, t)
        };
        let (va, ta) = slab(0.0, 0.2);
        let (vb, tb) = slab(0.8, 1.0);
        let mut vertices = va;
        let offset = vertices.len();
        vertices.extend(vb);
        let mut triangles = ta;
        triangles.extend(tb.iter().map(|t| t.map(|i| i + offset)));
        let err = derive_form(
            Form::Cube,
            &FormMesh {
                vertices,
                triangles,
            },
        )
        .unwrap_err();
        assert!(
            err.contains("solid is not a single connected region"),
            "{err}"
        );

        // Inward winding: negative volume.
        let mut flipped = form_mesh(Form::Wedge);
        for t in &mut flipped.triangles {
            t.swap(1, 2);
        }
        assert!(derive_form(Form::Wedge, &flipped)
            .unwrap_err()
            .contains("volume"));
    }

    #[test]
    fn face_masks_seal_complements_and_vent_duplicates() {
        // WI 832 mask level: a cube's faces are all FULL; the canonical wedge's
        // x1 face ({y ≤ z}) plus the diag(1,−1,−1)-rotated wedge's x0 face
        // ({y ≥ z}) jointly cover the boundary; two identical halves do not.
        assert!(face_masks(Form::Cube, 0).iter().all(|m| *m == MASK_FULL));
        let complement = rotations()
            .iter()
            .position(|r| r.abs_diff_eq(DMat3::from_diagonal(DVec3::new(1.0, -1.0, -1.0)), 1e-12))
            .unwrap() as u8;
        let a = face_masks(Form::Wedge, 0)[1];
        let b = face_masks(Form::Wedge, complement)[0];
        assert!(masks_seal(&a, &b), "complements cover the boundary");
        let same = face_masks(Form::Wedge, 0)[0];
        assert!(!masks_seal(&a, &same), "duplicate halves leave a gap");
        // Half coverage reads as ~half the samples (the jitter tie-breaks the
        // diagonal to one side).
        let n = mask_popcount(&a);
        assert!((100..156).contains(&n), "half face ≈ half the samples: {n}");
        assert_eq!(
            mask_popcount(&a) + mask_popcount(&b),
            256,
            "exact complements partition the samples"
        );
    }

    #[test]
    fn overlap_and_seal_answer_consistently_on_one_pair() {
        // WI 835: the vocabulary's two closure operations, mutually pinned on
        // the mated-complement fixture — AND overlaps zero exactly where OR
        // seals (the partition property, from both sides).
        let a = face_masks(Form::Wedge, 0)[1];
        let complement = constants(Form::Wedge)
            .distinct_orientations
            .iter()
            .copied()
            .find(|&o| {
                let m = face_masks(Form::Wedge, o)[0];
                masks_seal(&a, &m) && mask_popcount(&m) < 200
            })
            .expect("a complementary orientation exists");
        let b = face_masks(Form::Wedge, complement)[0];
        assert!(masks_seal(&a, &b), "complements seal");
        assert_eq!(masks_overlap(&a, &b), 0, "and overlap nothing");
        // Degenerate anchors: full-vs-full is total; anything vs full is its
        // own popcount.
        assert_eq!(masks_overlap(&MASK_FULL, &MASK_FULL), 256);
        assert_eq!(masks_overlap(&a, &MASK_FULL), mask_popcount(&a));
        assert_eq!(masks_overlap(&a, &MASK_EMPTY), 0);
    }

    #[test]
    fn the_outline_is_the_crease_edges_of_each_form() {
        // WI 833: the ghost wireframe drops triangulation diagonals and keeps the
        // polyhedron's real edges — cube 12, wedge (triangular prism) 9,
        // outer-corner tetrahedron 6.
        assert_eq!(form_outline(Form::Cube, 0).len(), 12);
        assert_eq!(form_outline(Form::Wedge, 0).len(), 9);
        assert_eq!(form_outline(Form::OuterCorner, 0).len(), 6);
    }

    #[test]
    fn the_outline_rotates_with_the_orientation() {
        // Every oriented outline endpoint is the rotation of a canonical endpoint,
        // and stays inside the unit cell (rotation is about the cell centre).
        for &o in &constants(Form::Wedge).distinct_orientations {
            let base = form_outline(Form::Wedge, 0);
            let oriented = form_outline(Form::Wedge, o);
            assert_eq!(base.len(), oriented.len());
            let r = rotations()[o as usize];
            let map = |p: DVec3| r * (p - DVec3::splat(0.5)) + DVec3::splat(0.5);
            for ((a0, b0), (a, b)) in base.iter().zip(oriented.iter()) {
                assert!((map(*a0) - *a).length() < 1e-12);
                assert!((map(*b0) - *b).length() < 1e-12);
                for p in [a, b] {
                    assert!((-1e-9..=1.0 + 1e-9).contains(&p.x));
                    assert!((-1e-9..=1.0 + 1e-9).contains(&p.y));
                    assert!((-1e-9..=1.0 + 1e-9).contains(&p.z));
                }
            }
        }
    }

    #[test]
    fn oriented_centroid_and_inertia_transform_as_the_rotation() {
        let c = constants(Form::Wedge);
        for &o in &c.distinct_orientations {
            let r = rotations()[o as usize];
            let want = r * (c.centroid - DVec3::splat(0.5)) + DVec3::splat(0.5);
            assert!((c.centroid_oriented(o) - want).length() < 1e-12);
            let i = c.unit_inertia_oriented(o);
            // Still symmetric, same trace (rotation preserves it).
            assert!((i.col(0).y - i.col(1).x).abs() < 1e-12);
            let trace = |m: DMat3| m.col(0).x + m.col(1).y + m.col(2).z;
            assert!((trace(i) - trace(c.unit_inertia)).abs() < 1e-12);
        }
    }
}

//! Procedural surface field (WI 763).
//!
//! The spherical analog of [`crate::terrain`]: a **pure, deterministic** function
//! from a [`BodyAsset`](crate::body_asset::BodyAsset) and a direction on the body
//! to a surface **elevation** (metres, relative to the reference radius) and a
//! **surface material**. This is the single analytic surface both the renderer
//! (WI 764, which tessellates it on a spherified-cube quadtree) and physics /
//! contact (WI 765, which queries it directly) will read — never a mesh, so the
//! contact point can't "pop" under LOD or floating-origin rebasing.
//!
//! **Seamless by construction.** Noise is sampled at the **3D unit-sphere
//! position** (the direction), not a 2D per-face parameterization, so the field
//! is continuous over the whole sphere with no cube-face/chunk seams. The noise
//! input is a unit vector, so it stays small regardless of body radius — radius
//! only scales the elevation amplitude, so there is no planetary-precision issue
//! in the field itself.
//!
//! **Bounded crater cost (R3).** Craters are a deterministic seed-derived
//! population addressed by 3D spatial cell hashes — since WI 782 a small fixed
//! set of **octaves** ([`CRATER_OCTAVES`]: large/sparse/deep, medium, small/
//! dense/shallow) on **pairwise-incommensurate lattice frequencies**, so no two
//! octaves' cell periods align and the population reads as a natural
//! distribution instead of a single lattice's rows/chains. An elevation sample
//! sums, per octave, only the query's own + adjacent cells (a fixed 3×3×3 =
//! 27-cell neighbourhood). Each neighbourhood is **exact by construction**
//! (WI 866, held per octave): a crater is a *lattice-space* ball around its
//! unnormalized jittered centre, its reach is strictly below one cell per axis,
//! and its centre lies inside its own cell — so any crater able to influence a
//! query lives in one of the 27 cells, and nothing pops in or out as the query
//! crosses a lattice plane. Per-sample cost is therefore a constant (octaves ×
//! 27 cells), **independent of the total crater count**, and the crater term is
//! continuous (every profile exactly zero at reach, rim included). Per-body
//! crater **density/depth multipliers** ([`CraterParams`]) ride the reserved
//! `SurfaceRecipe.crater` area (lenient parse, defaults on anything absent or
//! malformed — no persistence-format change).

use crate::body_asset::BodyAsset;
use crate::surface::SurfaceMaterial;
use glam::DVec3;

/// Domain-warp strength (fraction of a base-frequency cell).
const WARP_STRENGTH: f64 = 0.35;

/// One crater population: a lattice of jittered-centre bowls at its own frequency,
/// evaluated by the shared WI 866 mechanism (see [`SurfaceField::crater_delta`]).
#[derive(Clone, Copy, Debug)]
struct CraterOctave {
    /// Lattice frequency (cells across the direction sphere). Frequencies are
    /// **pairwise incommensurate** across the table (non-integer ratios) so no two
    /// octaves' lattice periods or plane families align — the WI 782 anti-grid
    /// property (a single lattice reads as rows/chains of same-sized craters).
    freq: f64,
    /// Per-cell crater probability, before the per-body density multiplier.
    density: f64,
    /// Reach span in lattice units. `reach_max` strictly below 1 keeps the
    /// 27-cell window exact per axis (the WI 866 invariant, per octave).
    reach_min: f64,
    reach_max: f64,
    /// Depth scale as a fraction of the body's relief amplitude.
    depth_frac: f64,
}

/// The default crater populations (WI 782): large/sparse/deep basins, the
/// familiar medium scale (WI 763's original ~14), and small/dense/shallow
/// texture craters. The summed `depth_frac` is [`CRATER_TOTAL_DEPTH_FRAC`],
/// the crater half of [`SurfaceField::relief_bound`]'s budget.
const CRATER_OCTAVES: [CraterOctave; 3] = [
    CraterOctave {
        freq: 6.3,
        density: 0.30,
        reach_min: 0.35,
        reach_max: 0.9,
        depth_frac: 0.50,
    },
    CraterOctave {
        freq: 14.0,
        density: 0.45,
        reach_min: 0.3,
        reach_max: 0.9,
        depth_frac: 0.40,
    },
    CraterOctave {
        freq: 29.7,
        density: 0.55,
        reach_min: 0.3,
        reach_max: 0.9,
        depth_frac: 0.15,
    },
];

/// Σ `depth_frac` over [`CRATER_OCTAVES`] — the crater relief budget.
const CRATER_TOTAL_DEPTH_FRAC: f64 = 0.50 + 0.40 + 0.15;

/// Per-body crater parameters (WI 782), read from the **reserved**
/// `SurfaceRecipe.crater` area (a defaulted `serde_json::Value`, so no
/// persistence-format change). Lenient: absent / null / non-object values and
/// missing or non-numeric keys all fall back to defaults; recognized keys are
/// `"density"` and `"depth"` — global multipliers over the octave table, each
/// clamped to `[0, 4]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CraterParams {
    /// Multiplier on every octave's per-cell crater probability (default 1).
    pub density: f64,
    /// Multiplier on every octave's depth scale (default 1).
    pub depth: f64,
}

impl Default for CraterParams {
    fn default() -> Self {
        Self {
            density: 1.0,
            depth: 1.0,
        }
    }
}

impl CraterParams {
    /// Parses the reserved recipe area. Never fails; anything unrecognized
    /// yields the default for that key.
    pub fn from_value(v: &serde_json::Value) -> Self {
        let get = |key: &str| {
            v.get(key)
                .and_then(|x| x.as_f64())
                .map(|x| x.clamp(0.0, 4.0))
        };
        Self {
            density: get("density").unwrap_or(1.0),
            depth: get("depth").unwrap_or(1.0),
        }
    }
}

/// A deterministic surface field for one body, derived from its seed and radius.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceField {
    seed: u64,
    /// Reference (sea-level) radius, metres.
    radius: f64,
    /// Peak relief amplitude, metres.
    amplitude: f64,
    /// Base terrain noise frequency (feature count across the sphere).
    base_freq: f64,
    /// Per-body crater multipliers over [`CRATER_OCTAVES`] (WI 782).
    crater: CraterParams,
}

/// A surface sample: elevation (metres, relative to the reference radius) and the
/// surface material at a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSample {
    /// Elevation above (or below) the reference radius, metres.
    pub elevation: f64,
    /// Surface material (friction / rolling resistance).
    pub material: SurfaceMaterial,
}

impl SurfaceField {
    /// Builds the field for `asset` from its `surface.seed`, `radius`, and the
    /// (reserved-area) crater parameters (WI 782).
    pub fn from_asset(asset: &BodyAsset) -> Self {
        Self::with_crater_params(
            asset.surface.seed,
            asset.radius,
            CraterParams::from_value(&asset.surface.crater),
        )
    }

    /// Builds a field from an explicit seed and reference radius (metres), with
    /// the default crater configuration.
    pub fn new(seed: u64, radius: f64) -> Self {
        Self::with_crater_params(seed, radius, CraterParams::default())
    }

    /// Builds a field with explicit per-body crater multipliers (WI 782).
    pub fn with_crater_params(seed: u64, radius: f64, crater: CraterParams) -> Self {
        let amplitude = (radius.abs() * 0.015).clamp(300.0, 9_000.0);
        Self {
            seed,
            radius: radius.abs().max(1.0),
            amplitude,
            base_freq: 2.5,
            crater,
        }
    }

    /// Reference radius, metres.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// A conservative bound on `|elevation|` anywhere on the body, metres — the sum
    /// of the terrain amplitude and the crater relief budget (every octave's depth
    /// scale, times the per-body depth multiplier). Lets distance tests bracket the
    /// surface without sampling the field (used by the LOD split pre-test, WI 795).
    pub fn relief_bound(&self) -> f64 {
        self.amplitude * (1.0 + CRATER_TOTAL_DEPTH_FRAC * self.crater.depth)
    }

    /// Elevation (metres, relative to the reference radius) at a direction from the
    /// body centre. `dir` need not be normalized; a zero vector falls back to +X.
    pub fn elevation(&self, dir: DVec3) -> f64 {
        let d = normalize_or_x(dir);
        let p = d * self.base_freq;
        // Domain warp: offset the sample point by a low-octave vector field.
        let warp = DVec3::new(
            fbm(p + DVec3::splat(11.5), self.seed ^ 0xA1, 2),
            fbm(p + DVec3::splat(53.2), self.seed ^ 0xB2, 2),
            fbm(p + DVec3::splat(97.1), self.seed ^ 0xC3, 2),
        ) * WARP_STRENGTH;
        let pw = p + warp;
        // Rolling terrain (fBm, [-1,1]) + ridged mountains ([0,1]) masked by a
        // low-frequency field so ranges cluster instead of covering the globe.
        let rolling = fbm(pw, self.seed, 5);
        let mountains = ridged(pw * 2.0, self.seed ^ 0xD4, 4);
        let mask = (fbm(pw * 0.5, self.seed ^ 0xE5, 2) * 0.5 + 0.5).clamp(0.0, 1.0);
        let relief = 0.6 * rolling + 0.8 * (mountains - 0.5) * mask;
        relief * self.amplitude + self.crater_delta(d)
    }

    /// Surface material at a direction: steep slopes read as bedrock, cold/high
    /// (polar or noise-masked) bands as ice, otherwise regolith.
    pub fn material(&self, dir: DVec3) -> SurfaceMaterial {
        let d = normalize_or_x(dir);
        let n = self.normal(d);
        // On flat ground the normal is ~radial (n·d ≈ 1); a slope tilts it away.
        let slope = 1.0 - n.dot(d).clamp(-1.0, 1.0);
        if slope > 0.03 {
            return SurfaceMaterial::BEDROCK;
        }
        let polar = d.y.abs();
        let ice_mask = fbm(d * 4.0, self.seed ^ 0x1CE, 3);
        if polar > 0.85 || ice_mask > 0.6 {
            return SurfaceMaterial::ICE;
        }
        SurfaceMaterial::REGOLITH
    }

    /// Elevation + material at a direction.
    pub fn sample(&self, dir: DVec3) -> SurfaceSample {
        SurfaceSample {
            elevation: self.elevation(dir),
            material: self.material(dir),
        }
    }

    /// Outward unit surface normal at a direction, from the analytic surface via a
    /// finite difference of three nearby surface points. Deterministic; used by the
    /// renderer (764) and contact (765).
    pub fn normal(&self, dir: DVec3) -> DVec3 {
        let d = normalize_or_x(dir);
        let (t1, t2) = tangent_basis(d);
        let eps = 1e-3;
        let point = |u: DVec3| {
            let un = normalize_or_x(u);
            un * (self.radius + self.elevation(un))
        };
        let pc = point(d);
        let pu = point(d + t1 * eps);
        let pv = point(d + t2 * eps);
        let n = (pu - pc).cross(pv - pc);
        let n = if n.length_squared() > 0.0 {
            n.normalize()
        } else {
            d
        };
        if n.dot(d) < 0.0 {
            -n
        } else {
            n
        }
    }

    /// The crater-population elevation delta at a direction (usually a depression):
    /// the sum of the [`CRATER_OCTAVES`] populations (WI 782). Per-sample cost is a
    /// constant — octave count × 27 cells — independent of the total crater count.
    fn crater_delta(&self, d: DVec3) -> f64 {
        let mut delta = 0.0;
        for (i, o) in CRATER_OCTAVES.iter().enumerate() {
            delta += self.octave_delta(d, i as u64, o);
        }
        delta
    }

    /// One octave's crater delta — the WI 866 mechanism at the octave's own
    /// lattice frequency with an octave-salted hash stream.
    ///
    /// Continuity (WI 866, held per octave): the whole per-crater profile is
    /// exactly zero at and beyond its reach, and influence is measured **in
    /// lattice space against the unnormalized centre** — a crater's reach
    /// (< 1 cell per axis) plus its centre lying inside its own cell means every
    /// crater that can influence `q` has its cell within `floor(q) ± 1`, so the
    /// 27-cell window is exact by construction and nothing can pop when the query
    /// crosses a lattice plane. (Projecting centres onto the sphere — the pre-866
    /// behaviour — displaced them radially by up to ~1.7 cells and broke exactly
    /// that argument.)
    fn octave_delta(&self, d: DVec3, index: u64, o: &CraterOctave) -> f64 {
        // Independent population per octave: salt the seed before the existing
        // per-purpose XORs so octave hash streams don't correlate.
        let oseed = self
            .seed
            .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index + 1));
        let q = d * o.freq;
        let base = q.floor();
        let density = (o.density * self.crater.density).min(1.0);
        let amp = self.amplitude * o.depth_frac * self.crater.depth;
        // The rim Gaussian's value at reach (t = 1); subtracted so the rim term is
        // exactly zero where the crater's influence ends. The coefficient below is
        // rescaled (0.25 → 0.3163) so the rim's peak height at t ≈ 0.85 is
        // unchanged by the subtraction.
        let rim_residual = (-((1.0_f64 - 0.85) / 0.12).powi(2)).exp();
        let mut delta = 0.0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let c = base + DVec3::new(dx as f64, dy as f64, dz as f64);
                    let (cx, cy, cz) = (c.x as i64, c.y as i64, c.z as i64);
                    if hash_unit(cx, cy, cz, oseed ^ 0x00C0_FFEE) >= density {
                        continue;
                    }
                    let jitter = DVec3::new(
                        hash_unit(cx, cy, cz, oseed ^ 0x11),
                        hash_unit(cx, cy, cz, oseed ^ 0x22),
                        hash_unit(cx, cy, cz, oseed ^ 0x33),
                    );
                    // Lattice-space centre, inside cell `c`. Never projected onto
                    // the sphere: the crater is a lattice-space ball and the
                    // surface trace is its intersection with the |q| = freq shell
                    // (cells whose ball misses the shell contribute nowhere).
                    let center = c + jitter;
                    let chord = (q - center).length();
                    let ar = o.reach_min
                        + (o.reach_max - o.reach_min) * hash_unit(cx, cy, cz, oseed ^ 0x44);
                    if chord >= ar {
                        continue;
                    }
                    let t = chord / ar;
                    let depth = amp
                        * (0.3 + 0.7 * hash_unit(cx, cy, cz, oseed ^ 0x55))
                        * (ar / o.reach_max);
                    // Bowl: −depth at the centre rising to exactly 0 at reach; plus
                    // a raised rim bump near t ≈ 0.85, windowed to exactly 0 at
                    // reach (the `.max(0.0)` also clips the Gaussian's inner tail,
                    // where the bowl dominates anyway).
                    let bowl = -depth * (1.0 - t * t);
                    let rim = 0.3163
                        * depth
                        * ((-((t - 0.85) / 0.12).powi(2)).exp() - rim_residual).max(0.0);
                    delta += bowl + rim;
                }
            }
        }
        delta
    }
}

/// Normalizes `v`, falling back to +X for a (near-)zero vector.
fn normalize_or_x(v: DVec3) -> DVec3 {
    let n = v.normalize_or_zero();
    if n == DVec3::ZERO {
        DVec3::X
    } else {
        n
    }
}

/// Two orthonormal tangents at a unit direction `d`.
fn tangent_basis(d: DVec3) -> (DVec3, DVec3) {
    let a = if d.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let t1 = (a - d * a.dot(d)).normalize();
    let t2 = d.cross(t1);
    (t1, t2)
}

/// A 64-bit hash of integer lattice coordinates + seed (splitmix-style mixing).
fn hash3(x: i64, y: i64, z: i64, seed: u64) -> u64 {
    let mut h = seed ^ 0x9E37_79B9_7F4A_7C15;
    for v in [x as u64, y as u64, z as u64] {
        h ^= v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 31;
    }
    h
}

/// A hash of lattice coordinates + seed in `[0, 1)`.
fn hash_unit(x: i64, y: i64, z: i64, seed: u64) -> f64 {
    (hash3(x, y, z, seed) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// 3D value noise in `[-1, 1]`: trilinear-interpolated corner hashes with a
/// smoothstep fade.
fn value_noise(p: DVec3, seed: u64) -> f64 {
    let pf = p.floor();
    let f = p - pf;
    let (ix, iy, iz) = (pf.x as i64, pf.y as i64, pf.z as i64);
    // Smoothstep fade per component.
    let w = f * f * (DVec3::splat(3.0) - 2.0 * f);
    let corner = |dx: i64, dy: i64, dz: i64| hash_unit(ix + dx, iy + dy, iz + dz, seed);
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), w.x);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), w.x);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), w.x);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), w.x);
    let y0 = lerp(x00, x10, w.y);
    let y1 = lerp(x01, x11, w.y);
    2.0 * lerp(y0, y1, w.z) - 1.0
}

/// Fractional Brownian motion (sum of octaves), normalized to `[-1, 1]`.
fn fbm(p: DVec3, seed: u64, octaves: u32) -> f64 {
    let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
    for o in 0..octaves {
        sum += amp * value_noise(p * freq, seed.wrapping_add(o as u64 * 0x1000));
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

/// Ridged multifractal (sharpened `1 − |noise|`), in `[0, 1]`.
fn ridged(p: DVec3, seed: u64, octaves: u32) -> f64 {
    let (mut amp, mut freq, mut sum, mut norm) = (1.0, 1.0, 0.0, 0.0);
    for o in 0..octaves {
        let n = 1.0 - value_noise(p * freq, seed.wrapping_add(o as u64 * 0x2000)).abs();
        sum += amp * n * n;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dirs() -> Vec<DVec3> {
        let mut v = Vec::new();
        for i in 0..60 {
            let a = i as f64 * 0.618_033_988_75 * std::f64::consts::TAU;
            let z = -1.0 + 2.0 * (i as f64 + 0.5) / 60.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            v.push(DVec3::new(r * a.cos(), z, r * a.sin()));
        }
        v
    }

    #[test]
    fn elevation_and_material_are_pure() {
        let f = SurfaceField::new(42, 1_000_000.0);
        for d in dirs() {
            assert_eq!(f.elevation(d), f.elevation(d));
            assert_eq!(f.material(d), f.material(d));
        }
    }

    #[test]
    fn different_seeds_produce_different_fields() {
        let a = SurfaceField::new(1, 1_000_000.0);
        let b = SurfaceField::new(2, 1_000_000.0);
        let differ = dirs().iter().any(|&d| a.elevation(d) != b.elevation(d));
        assert!(differ, "different seeds must give different terrain");
    }

    #[test]
    fn all_outputs_are_finite_including_degenerate_direction() {
        for &(seed, radius) in &[(0u64, 2.0e5), (7, 6.36e6), (u64::MAX, 9.0e6)] {
            let f = SurfaceField::new(seed, radius);
            assert!(f.elevation(DVec3::ZERO).is_finite(), "zero dir → finite");
            for d in dirs() {
                assert!(f.elevation(d).is_finite());
                let n = f.normal(d);
                assert!((n.length() - 1.0).abs() < 1e-9 && n.dot(d.normalize()) > 0.0);
            }
        }
    }

    #[test]
    fn field_is_lod_independent_at_planetary_radius() {
        // The field is a pure function of direction, so a value is identical no
        // matter the sampling density that produced the direction. Sample a great
        // circle at a fine step and a coarse subset; shared directions must be
        // bit-identical (proving no hidden grid/order/state dependence).
        let f = SurfaceField::new(99, 6_360_000.0);
        let fine: Vec<(DVec3, f64)> = (0..360)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 360.0;
                let d = DVec3::new(a.cos(), 0.3, a.sin());
                (d, f.elevation(d))
            })
            .collect();
        for i in (0..360).step_by(9) {
            let a = i as f64 * std::f64::consts::TAU / 360.0;
            let d = DVec3::new(a.cos(), 0.3, a.sin());
            assert_eq!(f.elevation(d), fine[i].1, "coarse must match fine exactly");
        }
    }

    #[test]
    fn field_is_seamless_continuous_across_the_sphere() {
        // A small angular step anywhere (including across axis boundaries where a
        // cube parameterization would seam) changes elevation only a little.
        let f = SurfaceField::new(5, 3_000_000.0);
        let eps = 1e-4;
        // Lipschitz-ish bound: an eps step can't move elevation more than a modest
        // multiple of amplitude·eps (generous constant to allow ridges).
        let bound = f.amplitude * eps * 500.0;
        for d in dirs() {
            let (t1, _) = tangent_basis(normalize_or_x(d));
            let d2 = normalize_or_x(d + t1 * eps);
            let delta = (f.elevation(d) - f.elevation(d2)).abs();
            assert!(delta < bound, "seam/discontinuity: Δ={delta} bound={bound}");
        }
    }

    #[test]
    fn elevation_is_continuous_across_lattice_planes_and_crater_perimeters() {
        // WI 866: the field must contain no step anywhere. Historically two defects
        // in `crater_delta` broke this — craters popping in/out of the 27-cell
        // window on lattice-plane crossings (whole-bowl steps, km-scale), and the
        // rim term hard-truncated at reach (~5% of depth around every perimeter).
        // March a great-circle arc at ~1.5 m spacing through a window known to
        // cross several lattice planes and many crater perimeters; a continuous
        // field changes < 10 m per step (measured smooth-field rate is < ~2 m),
        // while either defect class exceeded 80 m.
        fn rotate(v: DVec3, axis: DVec3, ang: f64) -> DVec3 {
            v * ang.cos() + axis.cross(v) * ang.sin() + axis * axis.dot(v) * (1.0 - ang.cos())
        }
        // Count lattice-plane crossings against the finest octave (most planes);
        // any octave's planes are the pop habitat the fix must keep closed.
        let finest = CRATER_OCTAVES[CRATER_OCTAVES.len() - 1].freq;
        for seed in [7u64, 11] {
            let f = SurfaceField::new(seed, 730_000.0);
            let start = DVec3::new(0.3, -0.9, 0.2).normalize();
            let axis = start.cross(DVec3::Y).normalize();
            let step = 2.0e-6; // rad ≈ 1.5 m of arc on this body
            let (a0, a1) = (0.55_f64, 0.75_f64);
            let n = ((a1 - a0) / step) as usize;
            let mut prev_d = rotate(start, axis, a0);
            let mut prev_e = f.elevation(prev_d);
            let mut planes_crossed = 0u32;
            let mut crater_samples = 0u32;
            let mut violations: Vec<String> = Vec::new();
            let (mut within_cell, mut on_plane) = (0u32, 0u32);
            for i in 1..=n {
                let d = rotate(start, axis, a0 + i as f64 * step);
                let e = f.elevation(d);
                assert!(
                    e.abs() <= f.relief_bound(),
                    "relief_bound violated: |{e}| > {} (seed {seed})",
                    f.relief_bound()
                );
                let cell_a = (prev_d * finest).floor();
                let cell_b = (d * finest).floor();
                if cell_a != cell_b {
                    planes_crossed += 1;
                }
                if f.crater_delta(d) < 0.0 {
                    crater_samples += 1;
                }
                let de = (e - prev_e).abs();
                if de > 10.0 {
                    // Report up to 4 per class so both defect habitats stay visible
                    // in a failure (within-cell hits are far denser than pops).
                    let (class, count) = if cell_a == cell_b {
                        ("within-cell (rim truncation)", &mut within_cell)
                    } else {
                        ("lattice-plane (membership pop)", &mut on_plane)
                    };
                    *count += 1;
                    if *count <= 4 {
                        violations.push(format!(
                            "seed {seed} ang {:.6}: Δ={de:.1} m — {class}",
                            a0 + i as f64 * step
                        ));
                    }
                }
                prev_d = d;
                prev_e = e;
            }
            // Coverage guards: the march must actually exercise both defect
            // habitats, or a constant change could quietly hollow the test out.
            assert!(
                planes_crossed >= 3,
                "arc no longer crosses lattice planes (crossed {planes_crossed}); re-site it"
            );
            assert!(
                crater_samples > 0,
                "arc no longer passes through craters; re-site it"
            );
            assert!(
                violations.is_empty(),
                "elevation steps detected ({within_cell} within-cell, {on_plane} on-plane):\n{}",
                violations.join("\n")
            );
        }
    }

    /// Finds (near +X) the projected centre of a crater from octave `index` whose
    /// lattice ball reaches the sphere shell — the shared search for the
    /// perimeter/bowl tests, re-anchored from the single-lattice WI 866 form to
    /// the octave table (WI 782). The scan range covers the octave's shell radius.
    fn find_octave_crater(f: &SurfaceField, index: usize) -> Option<DVec3> {
        let o = &CRATER_OCTAVES[index];
        let oseed = f
            .seed
            .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index as u64 + 1));
        let shell = o.freq.round() as i64;
        for ix in (shell - 4)..(shell + 4) {
            for iy in -3..=3 {
                for iz in -3..=3 {
                    if hash_unit(ix, iy, iz, oseed ^ 0x00C0_FFEE)
                        >= (o.density * f.crater.density).min(1.0)
                    {
                        continue;
                    }
                    let jitter = DVec3::new(
                        hash_unit(ix, iy, iz, oseed ^ 0x11),
                        hash_unit(ix, iy, iz, oseed ^ 0x22),
                        hash_unit(ix, iy, iz, oseed ^ 0x33),
                    );
                    let c = normalize_or_x(DVec3::new(ix as f64, iy as f64, iz as f64) + jitter);
                    if c.x > 0.5 && f.octave_delta(c, index as u64, o) < 0.0 {
                        return Some(c);
                    }
                }
            }
        }
        None
    }

    #[test]
    fn a_crater_profile_has_no_step_at_its_perimeter() {
        // WI 866 fix B: the rim term is windowed to exactly zero at reach, so
        // marching out of a crater must show no step — only smooth slope. Find a
        // medium-octave crater, then march a tangent ray from its centre through
        // the perimeter at ~7 cm arc steps; the whole crater term (all octaves —
        // the march inevitably crosses other octaves' perimeters too) may not
        // change faster than a smooth-slope bound per step. Pre-866 the truncated
        // rim Gaussian left an ~5%-of-depth cliff exactly at the perimeter.
        let f = SurfaceField::new(77, 2_000_000.0);
        let center = find_octave_crater(&f, 1).expect("expected to find a medium crater near +X");
        let (t1, _) = tangent_basis(center);
        // 0.1 rad ≫ the medium octave's maximum reach (0.9 lattice units / 14
        // ≈ 0.064 rad), so the march provably crosses that crater's perimeter
        // (and, incidentally, other octaves' perimeters along the way).
        let step = 1.0e-6;
        let mut prev = f.crater_delta(center);
        let mut max_step = 0.0_f64;
        for i in 1..=100_000 {
            let d = normalize_or_x(center + t1 * (i as f64 * step));
            let cur = f.crater_delta(d);
            max_step = max_step.max((cur - prev).abs());
            prev = cur;
        }
        // Smooth-slope budget: the steepest crater wall moves the term well under
        // 1 m per 1e-6 rad; the pre-fix perimeter cliff was 80–330 m in one step.
        assert!(
            max_step < 5.0,
            "crater term stepped {max_step} m in one ~7 cm sample — perimeter discontinuity"
        );
    }

    #[test]
    fn craters_are_sparse_local_and_stable() {
        let f = SurfaceField::new(2024, 1_000_000.0);
        // Purity of the crater pass.
        for d in dirs() {
            assert_eq!(f.crater_delta(d), f.crater_delta(d));
        }
        // Sparse + local: some directions sit in a crater (delta < 0), others sit
        // in none (delta == 0). If every sample summed all craters this couldn't
        // hold — it demonstrates the bounded local query. Checked per octave since
        // WI 782: three overlapping populations legitimately leave few *totally*
        // crater-free directions, but each octave must still be sparse and local.
        for (i, o) in CRATER_OCTAVES.iter().enumerate() {
            let has_crater = dirs().iter().any(|&d| f.octave_delta(d, i as u64, o) < 0.0);
            let no_crater = dirs()
                .iter()
                .any(|&d| f.octave_delta(d, i as u64, o) == 0.0);
            assert!(
                has_crater,
                "octave {i}: expected at least one crater among the samples"
            );
            assert!(
                no_crater,
                "octave {i}: expected at least one crater-free sample (locality)"
            );
        }
    }

    #[test]
    fn octave_table_upholds_the_bounded_cost_and_exactness_invariants() {
        // WI 782 config-validity guard: the numeric preconditions of the WI 866
        // exactness argument (reach < 1 cell per axis; centres in-cell comes from
        // hash_unit ∈ [0,1)) and the R3 bounded cost (small fixed octave count)
        // must hold for the octave table, or the lattice-plane pops return.
        assert!(
            (1..=4).contains(&CRATER_OCTAVES.len()),
            "octave count must stay small and fixed (R3)"
        );
        let mut depth_total = 0.0;
        for o in &CRATER_OCTAVES {
            assert!(o.reach_max < 1.0, "reach must stay under one lattice cell");
            assert!(0.0 < o.reach_min && o.reach_min < o.reach_max);
            assert!(0.0 < o.density && o.density <= 1.0);
            assert!(o.depth_frac > 0.0);
            depth_total += o.depth_frac;
        }
        assert!(
            (depth_total - CRATER_TOTAL_DEPTH_FRAC).abs() < 1e-12,
            "relief budget constant must match the table"
        );
        // Anti-grid: frequencies pairwise incommensurate (no near-integer ratio),
        // and the finest octave's craters stay km-scale (rim ≫ render grid — the
        // WI 781 faceting-regression floor).
        for (i, a) in CRATER_OCTAVES.iter().enumerate() {
            for (j, b) in CRATER_OCTAVES.iter().enumerate().skip(i + 1) {
                let r = b.freq / a.freq;
                assert!(
                    (r - r.round()).abs() > 0.05,
                    "octave frequencies {i}/{j} must have a non-integer ratio (got {r})"
                );
            }
        }
        assert!(CRATER_OCTAVES[CRATER_OCTAVES.len() - 1].freq <= 32.0);
    }

    #[test]
    fn crater_params_parse_leniently_and_apply() {
        // Parse: defaults for anything absent/null/garbage; clamped multipliers.
        assert_eq!(
            CraterParams::from_value(&serde_json::Value::Null),
            CraterParams::default()
        );
        assert_eq!(
            CraterParams::from_value(&serde_json::json!({"bogus": 3, "density": "x"})),
            CraterParams::default()
        );
        assert_eq!(
            CraterParams::from_value(&serde_json::json!({"density": 0.5, "depth": 2.0})),
            CraterParams {
                density: 0.5,
                depth: 2.0
            }
        );
        assert_eq!(
            CraterParams::from_value(&serde_json::json!({"density": -5.0, "depth": 99.0})),
            CraterParams {
                density: 0.0,
                depth: 4.0
            }
        );

        // An asset with an untouched (reserved/empty) crater area builds a field
        // identical to the default constructor.
        let mut asset = BodyAsset::earthlike();
        asset.surface.seed = 42;
        asset.radius = 1_500_000.0;
        let from_asset = SurfaceField::from_asset(&asset);
        let plain = SurfaceField::new(42, 1_500_000.0);
        for d in dirs() {
            assert_eq!(from_asset.elevation(d), plain.elevation(d));
        }

        // Density multiplier 0 ⇒ a craterless (but still continuous) body.
        asset.surface.crater = serde_json::json!({"density": 0.0});
        let craterless = SurfaceField::from_asset(&asset);
        for d in dirs() {
            assert_eq!(craterless.crater_delta(d), 0.0);
        }

        // Depth multiplier scales the crater term linearly where it is nonzero.
        asset.surface.crater = serde_json::json!({"depth": 2.0});
        let deep = SurfaceField::from_asset(&asset);
        let mut checked = 0;
        for d in dirs() {
            let base = plain.crater_delta(d);
            if base != 0.0 {
                let scaled = deep.crater_delta(d);
                assert!(
                    (scaled - 2.0 * base).abs() <= 1e-9 * base.abs(),
                    "depth multiplier must scale the crater term ({base} -> {scaled})"
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "expected in-crater samples for the depth check"
        );
        // And the relief bound follows the depth multiplier.
        assert!(deep.relief_bound() > plain.relief_bound());
    }

    #[test]
    fn a_crater_centre_is_depressed_below_its_rim() {
        // Find a medium-octave crater near +X and confirm its centre is lower than
        // a point out near the rim (a bowl). Compared on the octave's own delta so
        // another octave's overlapping slope can't mask the shape (WI 782
        // re-anchor of the single-lattice original).
        let f = SurfaceField::new(77, 2_000_000.0);
        let o = &CRATER_OCTAVES[1];
        let center = find_octave_crater(&f, 1).expect("expected to find a medium crater near +X");
        let center_delta = f.octave_delta(center, 1, o);
        // A ring point ~0.6 of the max radius away, along a tangent.
        let (t1, _) = tangent_basis(center);
        let ring = normalize_or_x(center + t1 * (0.6 * o.reach_max / o.freq));
        assert!(
            center_delta < f.octave_delta(ring, 1, o),
            "centre {center_delta} must be below ring {}",
            f.octave_delta(ring, 1, o)
        );
    }

    #[test]
    fn materials_are_valid_and_vary() {
        let f = SurfaceField::new(313, 6_360_000.0);
        let mut kinds = std::collections::HashSet::new();
        for d in dirs() {
            let m = f.material(d);
            assert!(m.friction.is_finite() && m.rolling_resistance.is_finite());
            kinds.insert(format!("{m:?}"));
        }
        // At least two distinct materials appear across the sphere.
        assert!(kinds.len() >= 2, "material field should vary: {kinds:?}");
    }

    #[test]
    fn from_asset_uses_the_surface_seed() {
        let mut asset = BodyAsset::earthlike();
        asset.surface.seed = 12345;
        asset.radius = 3_000_000.0;
        let f = SurfaceField::from_asset(&asset);
        let g = SurfaceField::new(12345, 3_000_000.0);
        for d in dirs() {
            assert_eq!(f.elevation(d), g.elevation(d));
        }
    }
}

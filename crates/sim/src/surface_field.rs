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
//! population addressed by a 3D spatial cell hash. An elevation sample sums only
//! the craters in the query's own + adjacent cells (a fixed 3×3×3 = 27-cell
//! neighbourhood); a crater's angular reach is kept below one cell so that
//! neighbourhood is exact. Per-sample cost is therefore **independent of the
//! total crater count**.

use crate::body_asset::BodyAsset;
use crate::surface::SurfaceMaterial;
use glam::DVec3;

/// Domain-warp strength (fraction of a base-frequency cell).
const WARP_STRENGTH: f64 = 0.35;
/// Probability that a crater cell contains a crater.
const CRATER_DENSITY: f64 = 0.45;

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
    /// Crater lattice frequency (cells across the direction sphere).
    crater_freq: f64,
    /// Crater depth scale, metres.
    crater_amp: f64,
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
    /// Builds the field for `asset` from its `surface.seed` and `radius`.
    pub fn from_asset(asset: &BodyAsset) -> Self {
        Self::new(asset.surface.seed, asset.radius)
    }

    /// Builds a field from an explicit seed and reference radius (metres).
    pub fn new(seed: u64, radius: f64) -> Self {
        let amplitude = (radius.abs() * 0.015).clamp(300.0, 9_000.0);
        Self {
            seed,
            radius: radius.abs().max(1.0),
            amplitude,
            base_freq: 2.5,
            crater_freq: 14.0,
            crater_amp: amplitude * 0.7,
        }
    }

    /// Reference radius, metres.
    pub fn radius(&self) -> f64 {
        self.radius
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

    /// The crater-population elevation delta at a direction (usually a depression).
    /// Sums only craters in the 3×3×3 cell neighbourhood of the query — a fixed,
    /// bounded amount of work regardless of the total crater count.
    fn crater_delta(&self, d: DVec3) -> f64 {
        let cf = self.crater_freq;
        let q = d * cf;
        let base = q.floor();
        // A crater's reach is kept under one cell so the 27-cell neighbourhood is
        // exact (in direction-chord units, a cell spans ~1/cf).
        let ar_max = 0.9 / cf;
        let ar_min = 0.3 / cf;
        let mut delta = 0.0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let c = base + DVec3::new(dx as f64, dy as f64, dz as f64);
                    let (cx, cy, cz) = (c.x as i64, c.y as i64, c.z as i64);
                    if hash_unit(cx, cy, cz, self.seed ^ 0x00C0_FFEE) >= CRATER_DENSITY {
                        continue;
                    }
                    let jitter = DVec3::new(
                        hash_unit(cx, cy, cz, self.seed ^ 0x11),
                        hash_unit(cx, cy, cz, self.seed ^ 0x22),
                        hash_unit(cx, cy, cz, self.seed ^ 0x33),
                    );
                    let center = normalize_or_x(c + jitter);
                    let chord = (d - center).length();
                    let ar = ar_min + (ar_max - ar_min) * hash_unit(cx, cy, cz, self.seed ^ 0x44);
                    if chord >= ar {
                        continue;
                    }
                    let t = chord / ar;
                    let depth = self.crater_amp
                        * (0.3 + 0.7 * hash_unit(cx, cy, cz, self.seed ^ 0x55))
                        * (ar / ar_max);
                    // Bowl: −depth at the centre rising to 0 at the rim; plus a
                    // raised rim bump near t ≈ 0.85.
                    let bowl = -depth * (1.0 - t * t);
                    let rim = 0.25 * depth * (-(((t - 0.85) / 0.12).powi(2))).exp();
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
    fn craters_are_sparse_local_and_stable() {
        let f = SurfaceField::new(2024, 1_000_000.0);
        // Purity of the crater pass.
        for d in dirs() {
            assert_eq!(f.crater_delta(d), f.crater_delta(d));
        }
        // Sparse + local: some directions sit in a crater (delta < 0), others sit
        // in none (delta == 0). If every sample summed all craters this couldn't
        // hold — it demonstrates the bounded local query.
        let has_crater = dirs().iter().any(|&d| f.crater_delta(d) < 0.0);
        let no_crater = dirs().iter().any(|&d| f.crater_delta(d) == 0.0);
        assert!(has_crater, "expected at least one crater among the samples");
        assert!(
            no_crater,
            "expected at least one crater-free sample (locality)"
        );
    }

    #[test]
    fn a_crater_centre_is_depressed_below_its_rim() {
        // Find a cell with a crater near +X, take its centre, and confirm the centre
        // is lower than a point out near the rim (a bowl).
        let f = SurfaceField::new(77, 2_000_000.0);
        let cf = f.crater_freq;
        let mut found = false;
        'search: for ix in 10..18 {
            for iy in -3..=3 {
                for iz in -3..=3 {
                    if hash_unit(ix, iy, iz, f.seed ^ 0x00C0_FFEE) >= CRATER_DENSITY {
                        continue;
                    }
                    let jitter = DVec3::new(
                        hash_unit(ix, iy, iz, f.seed ^ 0x11),
                        hash_unit(ix, iy, iz, f.seed ^ 0x22),
                        hash_unit(ix, iy, iz, f.seed ^ 0x33),
                    );
                    let center =
                        normalize_or_x(DVec3::new(ix as f64, iy as f64, iz as f64) + jitter);
                    // Only accept centres actually near +X so `cf` scaling lines up.
                    if center.x < 0.5 {
                        continue;
                    }
                    let center_delta = f.crater_delta(center);
                    if center_delta < 0.0 {
                        // A ring point ~0.6 of the max radius away, along a tangent.
                        let (t1, _) = tangent_basis(center);
                        let ring = normalize_or_x(center + t1 * (0.6 * 0.9 / cf));
                        assert!(
                            center_delta < f.crater_delta(ring),
                            "centre {center_delta} must be below ring {}",
                            f.crater_delta(ring)
                        );
                        found = true;
                        break 'search;
                    }
                }
            }
        }
        assert!(found, "expected to find a crater near +X for this seed");
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

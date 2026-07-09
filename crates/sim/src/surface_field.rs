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

use crate::biome::{classify, BiomeFamily, BiomeWeights, BodyClimate, ClimateSample};
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

/// How many leading [`CRATER_OCTAVES`] carry ejecta rays (WI 873): the large
/// and medium populations only — the small octave's craters are sub-visible at
/// the orbital altitudes rays are for, and skipping it bounds the query cost.
const EJECTA_OCTAVES: usize = 2;

/// Fraction of craters that are "young" (carry a ray system): a fresh per-cell
/// hash against this, so ray systems are a clear minority of craters. (Gate
/// iteration: 0.2 → 0.12 — the real Moon shows a handful of prominent systems
/// per hemisphere, and per-pixel ray cost scales with how many systems overlap
/// a pixel.)
const EJECTA_YOUNG_FRACTION: f64 = 0.12;

/// Full ray extent as a multiple of the crater's own reach (the shader draws
/// rays out to this)…
const EJECTA_EXTENT_FACTOR: f64 = 2.5;

/// …capped **strictly below one lattice cell**, so the WI 866 exactness
/// argument (influence reach < 1 cell per axis + centre inside its own cell ⇒
/// the 27-cell window is exact) holds for the ejecta reach verbatim.
const EJECTA_EXTENT_MAX: f64 = 0.95;

/// The classifier halo's extent as a fraction of the full ray extent (gate
/// iteration): the sim field carries only the **round bright blanket** hugging
/// the crater — thin rays run much farther, per-pixel, in the render shader
/// (`ejecta_systems` is that seam). Splitting them this way removed the
/// vertex-resolution constraint that forced the rejected few-fat-straight-arm
/// look.
const EJECTA_HALO_FRACTION: f64 = 0.55;

/// Per-chunk budget of ray systems passed to the render shader — the gather
/// ranks by visual contribution (intensity × extent²) and truncates; typical
/// coarse-chunk overlap is ~3, and anything truncated is the faintest and
/// resolves in as chunks split.
pub const MAX_EJECTA_SYSTEMS: usize = 8;

/// One young crater's ray system, as the render shader consumes it (WI 873
/// gate iteration): where the system is and its per-crater hash parameters.
/// The shader owns the ray *pattern*; this owns the ray *placement* — derived
/// from the same occupancy/jitter/reach hash streams as the crater bowls and
/// the halo, so all three always coincide.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EjectaSystem {
    /// Unit direction of the crater centre (body frame).
    pub center_dir: DVec3,
    /// First tangent of the centre's frame (the second is
    /// `center_dir × tangent`) — precomputed so the shader pays no per-pixel
    /// frame construction.
    pub tangent: DVec3,
    /// Full ray extent, radians of arc from the centre.
    pub extent: f64,
    /// Per-crater ray-pattern seed in `[0, 1)`.
    pub ray_seed: f64,
    /// Per-crater brightness in `[0.55, 1.0)` (the halo's intensity hash).
    pub intensity: f64,
}

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
    /// Per-body climate inputs for the biome layer (WI 868).
    climate: BodyClimate,
}

/// Equator-to-pole temperature drop, Kelvin (applied as `drop · lat²`).
const LATITUDE_TEMPERATURE_DROP: f64 = 75.0;

/// Atmospheric lapse rate, Kelvin per metre above sea level.
const LAPSE_PER_M: f64 = 6.5e-3;

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
    /// Builds the field for `asset` from its `surface.seed`, `radius`, the
    /// (reserved-area) crater parameters (WI 782), and the climate inputs the
    /// biome layer reads from the asset's existing fields (WI 868).
    pub fn from_asset(asset: &BodyAsset) -> Self {
        Self::with_params(
            asset.surface.seed,
            asset.radius,
            CraterParams::from_value(&asset.surface.crater),
            BodyClimate::from_asset(asset),
        )
    }

    /// Builds a field from an explicit seed and reference radius (metres), with
    /// the default crater configuration and climate (an airless 200 K body).
    pub fn new(seed: u64, radius: f64) -> Self {
        Self::with_crater_params(seed, radius, CraterParams::default())
    }

    /// Builds a field with explicit per-body crater multipliers (WI 782) and
    /// the default climate.
    pub fn with_crater_params(seed: u64, radius: f64, crater: CraterParams) -> Self {
        Self::with_params(seed, radius, crater, BodyClimate::default())
    }

    /// Builds a field with explicit crater and climate configuration (WI 868).
    pub fn with_params(seed: u64, radius: f64, crater: CraterParams, climate: BodyClimate) -> Self {
        let amplitude = (radius.abs() * 0.015).clamp(300.0, 9_000.0);
        Self {
            seed,
            radius: radius.abs().max(1.0),
            amplitude,
            base_freq: 2.5,
            crater,
            climate,
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

    /// Surface material at a direction: since WI 868 the **weight-blended**
    /// friction/rolling of the biome table (see [`Self::biome_weights`]) — same
    /// signature and query pattern as before, blended in value, so a consumer
    /// crossing a biome frontier sees a ramp, never a step. The pre-868 rules
    /// (steep → bedrock, polar/cold → ice, else regolith) survive as table rows.
    pub fn material(&self, dir: DVec3) -> SurfaceMaterial {
        self.biome_weights(dir).material()
    }

    /// The biome blend at a direction (WI 868): climate-field evaluation +
    /// weight-blended classification against the body family's biome table.
    /// Pure and deterministic; no allocation. See `crate::biome` for the
    /// weights-not-ids contract and the continuity argument.
    pub fn biome_weights(&self, dir: DVec3) -> BiomeWeights {
        let d = normalize_or_x(dir);
        let (n, elevation) = self.normal_and_elevation(d);
        self.biome_weights_at(d, elevation, n)
    }

    /// [`Self::biome_weights`] with the elevation and normal supplied by the
    /// caller — the mesh builder's seam (WI 869): chunk generation already
    /// samples both per vertex, so its per-vertex biome query pays only the
    /// climate fields + classification, not a second finite-difference normal.
    /// `elevation`/`normal` **must** come from this same field at `dir`
    /// (`elevation(dir)` / `normal(dir)`), or the classification is
    /// inconsistent with the terrain.
    pub fn biome_weights_at(&self, dir: DVec3, elevation: f64, normal: DVec3) -> BiomeWeights {
        let d = normalize_or_x(dir);
        let slope = 1.0 - normal.dot(d).clamp(-1.0, 1.0);
        let sea = self.climate.sea_level.unwrap_or(0.0);
        // Family-irrelevant channels stay neutral and unevaluated (their rows
        // mark them "don't care"): the atmospheric path never pays for the
        // crater-bowl or ejecta queries and the airless path never pays for
        // moisture.
        let (moisture, albedo, roughness, bowl, ejecta) = match self.climate.family {
            BiomeFamily::Atmospheric => (self.moisture(d), 0.5, 0.5, 0.0, 0.0),
            BiomeFamily::Airless => (
                0.5,
                self.albedo_field(d),
                self.roughness_field(d),
                self.bowl(d),
                self.ejecta(d),
            ),
        };
        // A dry world has no marine biomes (WI 869 finding): without an ocean,
        // the elevation channel floors just above the marine bands (ocean /
        // shallows / beach all close by 0.03), so basins read as low plains and
        // classify by climate instead. `max` keeps the channel continuous; the
        // floored region is constant, so no new frontier appears.
        let e = (elevation - sea) / self.amplitude;
        let elevation_channel = if self.climate.sea_level.is_some() {
            e
        } else {
            e.max(0.03)
        };
        let sample = ClimateSample {
            temperature: self.temperature_at_elevation(d, elevation),
            moisture,
            elevation: elevation_channel,
            slope,
            latitude: self.warped_latitude(d),
            albedo,
            roughness,
            bowl,
            ejecta,
        };
        classify(self.climate.family, &sample)
    }

    /// Temperature (Kelvin-anchored classifier scale) at a direction: body base
    /// temperature − latitude gradient − altitude lapse (WI 868). Latitude is
    /// measured against the body's **rotation axis**; the lapse applies above
    /// sea level on atmospheric bodies only (airless bodies have no atmosphere
    /// to cool through).
    pub fn temperature(&self, dir: DVec3) -> f64 {
        let d = normalize_or_x(dir);
        self.temperature_at_elevation(d, self.elevation(d))
    }

    /// The temperature model with elevation as an explicit input — the internal
    /// seam `temperature` composes with `elevation`, and the tests exercise
    /// directly (the lapse is unobservable from directions alone because
    /// elevation is itself a function of direction).
    fn temperature_at_elevation(&self, d: DVec3, elevation: f64) -> f64 {
        let lat = self.warped_latitude(d);
        // The WI 870 per-body offset shifts the classifier's base only — the
        // physics medium keeps its own temperature (WI 875 owns that side).
        let base = self.climate.base_temperature + self.climate.params.temperature;
        let mut t = base - LATITUDE_TEMPERATURE_DROP * lat * lat;
        if self.climate.family == BiomeFamily::Atmospheric {
            let sea = self.climate.sea_level.unwrap_or(0.0);
            t -= LAPSE_PER_M * (elevation - sea).max(0.0);
        }
        t
    }

    /// Moisture in `[0, 1]` at a direction: an independent low-frequency
    /// domain-warped fbm (WI 868). A real hydrology model is explicitly out of
    /// scope; ocean-adjacent wetness emerges from the classifier's elevation
    /// gates instead.
    pub fn moisture(&self, dir: DVec3) -> f64 {
        let d = normalize_or_x(dir);
        let p = d * 1.4;
        let warp = DVec3::new(
            fbm(p + DVec3::splat(7.3), self.seed ^ 0x30B5, 2),
            fbm(p + DVec3::splat(41.9), self.seed ^ 0x30B6, 2),
            fbm(p + DVec3::splat(83.7), self.seed ^ 0x30B7, 2),
        ) * 0.5;
        // WI 870 knobs: `moisture` shifts the midpoint, `moisture_scale`
        // widens/narrows the deviation around the shifted midpoint — both
        // constant per body (continuity preserved), result still in [0, 1].
        let deviation = 0.8 * fbm(p + warp, self.seed ^ 0x30B0, 3);
        (0.5 + self.climate.params.moisture + self.climate.params.moisture_scale * deviation)
            .clamp(0.0, 1.0)
    }

    /// Warped absolute latitude in `[0, 1]` against the rotation axis: the raw
    /// `|d·axis|` plus a low-frequency wander so no biome edge traces a perfect
    /// parallel (the design's domain-warped-inputs rule).
    fn warped_latitude(&self, d: DVec3) -> f64 {
        let raw = d.dot(self.climate.axis).abs();
        (raw + 0.06 * fbm(d * 2.5, self.seed ^ 0x1A71, 2)).clamp(0.0, 1.0)
    }

    /// Airless-family albedo field in `[0, 1]` (maria vs bright highlands).
    fn albedo_field(&self, d: DVec3) -> f64 {
        (0.5 + 0.85 * fbm(d * 1.8, self.seed ^ 0xA1BED0, 3)).clamp(0.0, 1.0)
    }

    /// Airless-family roughness field in `[0, 1]` (boulder fields).
    fn roughness_field(&self, d: DVec3) -> f64 {
        (0.5 + 0.85 * fbm(d * 5.0, self.seed ^ 0x40C4, 2)).clamp(0.0, 1.0)
    }

    /// Crater-interior factor in `[0, 1]`: a smooth ramp over the (continuous,
    /// WI 866) crater term, →1 in deep bowls — the cold-trap gate input.
    fn bowl(&self, d: DVec3) -> f64 {
        let t = (-self.crater_delta(d) / (0.08 * self.amplitude)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
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
        self.normal_and_elevation(normalize_or_x(dir)).0
    }

    /// The normal plus the centre elevation from the same three-point finite
    /// difference — shared so the biome query pays exactly the pre-868
    /// `material()` cost (three elevation evaluations), not a fourth.
    fn normal_and_elevation(&self, d: DVec3) -> (DVec3, f64) {
        let (t1, t2) = tangent_basis(d);
        let eps = 1e-3;
        let point = |u: DVec3| {
            let un = normalize_or_x(u);
            un * (self.radius + self.elevation(un))
        };
        let e0 = self.elevation(d);
        let pc = d * (self.radius + e0);
        let pu = point(d + t1 * eps);
        let pv = point(d + t2 * eps);
        let n = (pu - pc).cross(pv - pc);
        let n = if n.length_squared() > 0.0 {
            n.normalize()
        } else {
            d
        };
        (if n.dot(d) < 0.0 { -n } else { n }, e0)
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

    /// Ejecta **halo** intensity in `[0, 1]` at a direction (WI 873, gate
    /// iteration): the round bright blanket around the deterministic "young"
    /// subset of the large and medium crater populations, feeding the airless
    /// classifier's `ejecta` channel — **albedo only**, never elevation. The
    /// pass reuses each octave's own occupancy/jitter/reach hash streams, so
    /// every halo sits on a real crater; overlapping halos combine by a
    /// saturating product, so the result stays bounded and continuous. The
    /// thin ray streaks are **not here**: they are drawn per-pixel by the
    /// render shader from [`Self::ejecta_systems`] (same hashes, same centres),
    /// because ray-width detail is below vertex-tint resolution at orbital LOD.
    ///
    /// Continuity (the WI 866 rules, applied to the new reach): each halo is
    /// exactly zero at and beyond its extent (zero slope there — no perimeter
    /// ring); influence is measured in lattice space against the unnormalized
    /// centre and stays strictly below one cell ([`EJECTA_EXTENT_MAX`]), so
    /// the 27-cell window is exact and nothing pops at lattice planes. A pure
    /// radial profile has no angular term at all, so the pre-iteration azimuth
    /// hazards are gone by construction.
    pub fn ejecta(&self, dir: DVec3) -> f64 {
        let d = normalize_or_x(dir);
        // Product of (1 − contribution): saturating combine of overlaps.
        let mut clear = 1.0;
        for (i, o) in CRATER_OCTAVES.iter().take(EJECTA_OCTAVES).enumerate() {
            let oseed = self
                .seed
                .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1));
            let q = d * o.freq;
            let base = q.floor();
            let density = (o.density * self.crater.density).min(1.0);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let c = base + DVec3::new(dx as f64, dy as f64, dz as f64);
                        let (cx, cy, cz) = (c.x as i64, c.y as i64, c.z as i64);
                        // Same occupancy gate as the crater term — a ray system
                        // exists only where its crater does (and the per-body
                        // density multiplier zeroes both together).
                        if hash_unit(cx, cy, cz, oseed ^ 0x00C0_FFEE) >= density {
                            continue;
                        }
                        if hash_unit(cx, cy, cz, oseed ^ 0x66) >= EJECTA_YOUNG_FRACTION {
                            continue;
                        }
                        let jitter = DVec3::new(
                            hash_unit(cx, cy, cz, oseed ^ 0x11),
                            hash_unit(cx, cy, cz, oseed ^ 0x22),
                            hash_unit(cx, cy, cz, oseed ^ 0x33),
                        );
                        let center = c + jitter;
                        let ar = o.reach_min
                            + (o.reach_max - o.reach_min) * hash_unit(cx, cy, cz, oseed ^ 0x44);
                        let er = (EJECTA_EXTENT_FACTOR * ar).min(EJECTA_EXTENT_MAX);
                        let eh = EJECTA_HALO_FRACTION * er;
                        let chord = (q - center).length();
                        if chord >= eh {
                            continue;
                        }
                        let t = chord / eh;
                        // Radial halo: 1 at the centre, smoothly exactly 0 at
                        // the halo extent (zero slope there — no perimeter
                        // ring). Round on purpose: small/distant real ray
                        // systems read as round bright splotches.
                        let envelope = {
                            let u = 1.0 - t * t;
                            u * u
                        };
                        let intensity = 0.55 + 0.45 * hash_unit(cx, cy, cz, oseed ^ 0x99);
                        clear *= 1.0 - (intensity * envelope).clamp(0.0, 1.0);
                    }
                }
            }
        }
        1.0 - clear
    }

    /// The young-crater ray systems overlapping a query cone (WI 873 gate
    /// iteration) — the render shader's placement seam. `dir` is the cone axis
    /// (a chunk's centre direction), `angular_radius` its half-angle; a system
    /// overlaps when its centre is within `angular_radius + extent` of the
    /// axis. Systems come from the same occupancy/jitter/reach hash streams as
    /// the crater bowls and the halo (invariant: all three coincide), are
    /// ranked by visual contribution (intensity × extent²), and truncate to
    /// [`MAX_EJECTA_SYSTEMS`] — anything cut is the faintest overlap and
    /// resolves in as chunks split. Pure and deterministic; returns the array
    /// plus the valid count. Atmospheric bodies have no ray systems (same
    /// family rule as the classifier's ejecta channel).
    pub fn ejecta_systems(
        &self,
        dir: DVec3,
        angular_radius: f64,
    ) -> ([EjectaSystem; MAX_EJECTA_SYSTEMS], usize) {
        let mut out = [EjectaSystem::default(); MAX_EJECTA_SYSTEMS];
        if self.climate.family == BiomeFamily::Atmospheric {
            return (out, 0);
        }
        let d = normalize_or_x(dir);
        // (system, ranking key); kept sorted descending by key, ties broken by
        // the deterministic iteration order (octave, then cell scan order).
        let mut found: Vec<(EjectaSystem, f64)> = Vec::new();
        for (i, o) in CRATER_OCTAVES.iter().take(EJECTA_OCTAVES).enumerate() {
            let oseed = self
                .seed
                .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1));
            let q = d * o.freq;
            let density = (o.density * self.crater.density).min(1.0);
            // Lattice-space window covering the cone plus the maximum reach
            // (+1 for centre-in-cell), clamped to the lattice's shell range so
            // a whole-face cone degrades to a bounded full scan.
            let w = angular_radius * o.freq + EJECTA_EXTENT_MAX + 1.0;
            let lo = |x: f64| ((x - w).floor().max(-o.freq - 2.0)) as i64;
            let hi = |x: f64| ((x + w).floor().min(o.freq + 2.0)) as i64;
            for cx in lo(q.x)..=hi(q.x) {
                for cy in lo(q.y)..=hi(q.y) {
                    for cz in lo(q.z)..=hi(q.z) {
                        if hash_unit(cx, cy, cz, oseed ^ 0x00C0_FFEE) >= density {
                            continue;
                        }
                        if hash_unit(cx, cy, cz, oseed ^ 0x66) >= EJECTA_YOUNG_FRACTION {
                            continue;
                        }
                        let jitter = DVec3::new(
                            hash_unit(cx, cy, cz, oseed ^ 0x11),
                            hash_unit(cx, cy, cz, oseed ^ 0x22),
                            hash_unit(cx, cy, cz, oseed ^ 0x33),
                        );
                        let center = DVec3::new(cx as f64, cy as f64, cz as f64) + jitter;
                        let ar = o.reach_min
                            + (o.reach_max - o.reach_min) * hash_unit(cx, cy, cz, oseed ^ 0x44);
                        let er = (EJECTA_EXTENT_FACTOR * ar).min(EJECTA_EXTENT_MAX);
                        // A lattice ball only traces onto the sphere if it
                        // reaches the |q| = freq shell.
                        if (center.length() - o.freq).abs() >= er {
                            continue;
                        }
                        let cdir = normalize_or_x(center);
                        let extent = er / o.freq;
                        let sep = cdir.dot(d).clamp(-1.0, 1.0).acos();
                        if sep >= angular_radius + extent {
                            continue;
                        }
                        let sys = EjectaSystem {
                            center_dir: cdir,
                            tangent: tangent_basis(cdir).0,
                            extent,
                            ray_seed: hash_unit(cx, cy, cz, oseed ^ 0x77),
                            intensity: 0.55 + 0.45 * hash_unit(cx, cy, cz, oseed ^ 0x99),
                        };
                        found.push((sys, sys.intensity * extent * extent));
                    }
                }
            }
        }
        found.sort_by(|a, b| b.1.total_cmp(&a.1));
        let n = found.len().min(MAX_EJECTA_SYSTEMS);
        for (slot, (sys, _)) in out.iter_mut().zip(found.into_iter().take(n)) {
            *slot = sys;
        }
        (out, n)
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
    use crate::biome::BiomeParams;

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

    /// Finds a "young" (ray-carrying) crater of octave `index` near +X whose
    /// projected centre sits well inside its own ejecta extent (so the halo
    /// core is guaranteed sampled on the sphere). Returns the projected centre
    /// direction and the extent in radians — the WI 873 sibling of
    /// [`find_octave_crater`], selecting on the same hash streams plus the
    /// young gate.
    fn find_young_crater(f: &SurfaceField, index: usize) -> Option<(DVec3, f64)> {
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
                    if hash_unit(ix, iy, iz, oseed ^ 0x66) >= EJECTA_YOUNG_FRACTION {
                        continue;
                    }
                    let jitter = DVec3::new(
                        hash_unit(ix, iy, iz, oseed ^ 0x11),
                        hash_unit(ix, iy, iz, oseed ^ 0x22),
                        hash_unit(ix, iy, iz, oseed ^ 0x33),
                    );
                    let center = DVec3::new(ix as f64, iy as f64, iz as f64) + jitter;
                    let ar = o.reach_min
                        + (o.reach_max - o.reach_min) * hash_unit(ix, iy, iz, oseed ^ 0x44);
                    let er = (EJECTA_EXTENT_FACTOR * ar).min(EJECTA_EXTENT_MAX);
                    let chord_at_centre = (center.length() - o.freq).abs();
                    let c = normalize_or_x(center);
                    // Near-shell so the (tighter, halo-fraction) core is
                    // guaranteed sampled on the sphere trace.
                    if c.x > 0.5 && chord_at_centre < 0.25 * er {
                        return Some((c, er / o.freq));
                    }
                }
            }
        }
        None
    }

    #[test]
    fn ejecta_is_pure_bounded_local_young_minority_and_density_gated() {
        let f = SurfaceField::new(2024, 1_000_000.0);
        // Purity + bounds on the standard direction set.
        for d in dirs() {
            let e = f.ejecta(d);
            assert!(e.is_finite() && (0.0..=1.0).contains(&e));
            assert_eq!(e, f.ejecta(d), "ejecta must be pure");
        }
        assert!(f.ejecta(DVec3::ZERO).is_finite(), "zero dir → finite");
        // Locality: on a dense scan, rayed and exactly-ray-free directions both
        // exist — each per-crater profile is exactly zero at and beyond its
        // extent, so the field has genuine zero support away from young craters.
        let (mut rayed, mut clear) = (0u32, 0u32);
        for i in 0..4_000 {
            let a = i as f64 * 0.618_033_988_75 * std::f64::consts::TAU;
            let z = -1.0 + 2.0 * (i as f64 + 0.5) / 4_000.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let e = f.ejecta(DVec3::new(r * a.cos(), z, r * a.sin()));
            if e > 0.05 {
                rayed += 1;
            }
            if e == 0.0 {
                clear += 1;
            }
        }
        assert!(rayed > 0, "no rayed directions found — field inert");
        assert!(clear > rayed, "ejecta must be sparse (local support)");

        // Young craters are a deterministic minority: scan octave 1's occupied
        // cells near the +X shell and count the young gate.
        let o = &CRATER_OCTAVES[1];
        let oseed = f
            .seed
            .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(2));
        let shell = o.freq.round() as i64;
        let (mut craters, mut young) = (0u32, 0u32);
        for ix in (shell - 4)..(shell + 4) {
            for iy in -4..=4 {
                for iz in -4..=4 {
                    if hash_unit(ix, iy, iz, oseed ^ 0x00C0_FFEE) >= o.density {
                        continue;
                    }
                    craters += 1;
                    if hash_unit(ix, iy, iz, oseed ^ 0x66) < EJECTA_YOUNG_FRACTION {
                        young += 1;
                    }
                }
            }
        }
        assert!(craters > 20, "scan window too small to judge the fraction");
        assert!(young > 0, "expected some young craters");
        assert!(
            (young as f64) < 0.5 * craters as f64,
            "young craters must be a minority ({young}/{craters})"
        );

        // Per-body density multiplier gates ejecta exactly as it gates craters…
        let mut asset = BodyAsset::earthlike();
        asset.surface.seed = 2024;
        asset.radius = 1_000_000.0;
        asset.surface.crater = serde_json::json!({"density": 0.0});
        let craterless = SurfaceField::from_asset(&asset);
        for d in dirs() {
            assert_eq!(craterless.ejecta(d), 0.0, "no craters ⇒ no rays");
        }
        // …while the depth multiplier does not apply (rays are albedo, not
        // relief).
        asset.surface.crater = serde_json::json!({"depth": 2.0});
        let deep = SurfaceField::from_asset(&asset);
        for d in dirs() {
            assert_eq!(deep.ejecta(d), f.ejecta(d));
        }
    }

    #[test]
    fn ejecta_is_continuous_and_locally_supported_along_an_arc() {
        // The WI 866 arc-march pattern on the new field: march straight through
        // a young crater's halo at ~1.5 m steps; the intensity and the blended
        // tint must move at a smooth rate everywhere — across the halo's
        // extent, its core, and the lattice planes the arc crosses (the
        // membership-pop habitat). (Gate iteration: the field is halo-only;
        // the thin rays are per-pixel shader work with no field footprint.)
        fn rotate(v: DVec3, axis: DVec3, ang: f64) -> DVec3 {
            v * ang.cos() + axis.cross(v) * ang.sin() + axis * axis.dot(v) * (1.0 - ang.cos())
        }
        let f = SurfaceField::new(2024, 1_000_000.0);
        let (center, ext) = find_young_crater(&f, 1)
            .or_else(|| find_young_crater(&f, 0))
            .expect("no young crater near +X — re-seed the probe");
        let (t1, _) = tangent_basis(center);
        let axis = center.cross(t1).normalize();
        let half = 1.6 * ext; // start/end outside the system
        let step = 2.0e-6;
        let n = (2.0 * half / step) as usize;
        let finest = CRATER_OCTAVES[EJECTA_OCTAVES - 1].freq;
        let mut prev_d = rotate(center, axis, -half);
        let mut prev_e = f.ejecta(prev_d);
        let mut prev_t = f.biome_weights(prev_d).tint();
        let (mut max_de, mut max_dt) = (0.0_f64, 0.0_f64);
        let (mut peak, mut zeros, mut planes) = (0.0_f64, 0u32, 0u32);
        for i in 1..=n {
            let d = rotate(center, axis, -half + i as f64 * step);
            let e = f.ejecta(d);
            let t = f.biome_weights(d).tint();
            max_de = max_de.max((e - prev_e).abs());
            for (a, b) in t.iter().zip(prev_t) {
                max_dt = max_dt.max((a - b).abs());
            }
            peak = peak.max(e);
            if e == 0.0 {
                zeros += 1;
            }
            if (prev_d * finest).floor() != (d * finest).floor() {
                planes += 1;
            }
            (prev_d, prev_e, prev_t) = (d, e, t);
        }
        // Coverage guards: the march must actually cross the system, exit it,
        // and cross lattice planes, or the rate asserts go vacuous.
        assert!(peak > 0.2, "arc never entered a ray system (peak {peak})");
        assert!(zeros > 0, "arc never left the system — extend it");
        assert!(planes >= 2, "arc crossed only {planes} lattice planes");
        // Smooth-rate bounds, calibrated: measured smooth maxima are
        // max_de ≈ 8e-5 and max_dt ≈ 1.3e-4 per step — >100× headroom below
        // the bounds, while a pop (whole-profile step, ≥ ~0.5 intensity) or an
        // extent cliff would exceed them by orders of magnitude.
        assert!(
            max_de < 0.02,
            "ejecta stepped {max_de} in one ~1.5 m sample — discontinuity"
        );
        assert!(
            max_dt < 0.02,
            "blended tint stepped {max_dt} in one ~1.5 m sample"
        );
    }

    #[test]
    fn ejecta_systems_gather_is_deterministic_covering_and_bounded() {
        let f = SurfaceField::new(2024, 1_000_000.0);
        // Site the probe cone on a known young crater (same hash streams).
        let (center, ext) = find_young_crater(&f, 1)
            .or_else(|| find_young_crater(&f, 0))
            .expect("no young crater near +X — re-seed the probe");
        let radius = 3.0 * ext;
        let (systems, n) = f.ejecta_systems(center, radius);
        // Deterministic: bit-identical on re-query.
        let (systems2, n2) = f.ejecta_systems(center, radius);
        assert_eq!(n, n2);
        assert_eq!(systems[..n], systems2[..n2]);
        assert!(n >= 1, "the sited young crater must be gathered");
        assert!(n <= MAX_EJECTA_SYSTEMS);
        // The sited crater itself is in the list (a system centred within its
        // own extent of the probe centre), parameters in range, and the
        // ranking key is descending.
        assert!(
            systems[..n]
                .iter()
                .any(|s| s.center_dir.dot(center).clamp(-1.0, 1.0).acos() < s.extent),
            "gathered systems miss the sited crater"
        );
        let mut prev = f64::INFINITY;
        for s in &systems[..n] {
            assert!((s.center_dir.length() - 1.0).abs() < 1e-9);
            assert!(s.extent > 0.0 && s.extent < 1.0);
            assert!((0.0..1.0).contains(&s.ray_seed));
            assert!((0.55..1.0).contains(&s.intensity));
            let key = s.intensity * s.extent * s.extent;
            assert!(key <= prev + 1e-15, "ranking must be descending");
            prev = key;
        }
        // Coverage: every direction with positive halo inside the cone lies
        // within some gathered system's extent (the halo is a subset of the
        // ray extent, and the gather over-reaches the cone) — unless the list
        // truncated, which this probe must not (guarded above by n ≤ MAX).
        if n < MAX_EJECTA_SYSTEMS {
            let (t1, t2) = tangent_basis(center);
            for i in 0..500 {
                let a = i as f64 * 0.618_033_988_75 * std::f64::consts::TAU;
                let r = radius * ((i as f64 + 0.5) / 500.0).sqrt();
                let d = normalize_or_x(center + (t1 * a.cos() + t2 * a.sin()) * r);
                if f.ejecta(d) > 0.0 {
                    let covered = systems[..n]
                        .iter()
                        .any(|s| s.center_dir.dot(d).clamp(-1.0, 1.0).acos() < s.extent);
                    assert!(covered, "halo-positive direction not covered by any system");
                }
            }
        }
        // Family rule: atmospheric bodies gather nothing.
        let atm = SurfaceField::with_params(
            2024,
            1_000_000.0,
            CraterParams::default(),
            temperate_climate(),
        );
        assert_eq!(atm.ejecta_systems(center, radius).1, 0);
        // Density 0 ⇒ no systems anywhere.
        let quiet = SurfaceField::with_crater_params(
            2024,
            1_000_000.0,
            CraterParams {
                density: 0.0,
                depth: 1.0,
            },
        );
        assert_eq!(quiet.ejecta_systems(center, radius).1, 0);
    }

    /// A temperate (288 K) ocean-bearing atmospheric climate rotating about +Z —
    /// the classifier-scale "earthlike" the biome tests probe. (The canonical
    /// `FluidMedium::EARTHLIKE` carries a representative *atmospheric* 250 K,
    /// which classifies as an ice-age world — fine physically, but the variety
    /// scenarios need the temperate regime.)
    fn temperate_climate() -> BodyClimate {
        BodyClimate {
            family: BiomeFamily::Atmospheric,
            base_temperature: 288.0,
            sea_level: Some(0.0),
            axis: DVec3::Z,
            params: BiomeParams::default(),
        }
    }

    #[test]
    fn climate_fields_are_pure_finite_and_bounded() {
        let fields = [
            SurfaceField::with_params(5, 2_000_000.0, CraterParams::default(), temperate_climate()),
            SurfaceField::new(5, 2_000_000.0), // default airless climate
        ];
        for f in fields {
            for d in dirs() {
                let t = f.temperature(d);
                assert!(t.is_finite());
                assert_eq!(t, f.temperature(d), "temperature must be pure");
                let m = f.moisture(d);
                assert!((0.0..=1.0).contains(&m));
                assert_eq!(m, f.moisture(d), "moisture must be pure");
                let w = f.biome_weights(d);
                let table = crate::biome::biome_table(w.family());
                let mut sum = 0.0;
                for (_, wi) in w.iter() {
                    assert!(wi.is_finite() && wi >= 0.0);
                    sum += wi;
                }
                assert!((sum - 1.0).abs() < 1e-12, "weights must normalize: {sum}");
                // Bit-identical on a second evaluation (purity of the whole query).
                let w2 = f.biome_weights(d);
                assert_eq!(w.dominant_index(), w2.dominant_index());
                for i in 0..table.len() {
                    assert_eq!(w.weight_of(i), w2.weight_of(i));
                }
            }
        }
    }

    #[test]
    fn the_latitude_gradient_and_lapse_cool_temperature() {
        let f =
            SurfaceField::with_params(9, 3_000_000.0, CraterParams::default(), temperate_climate());
        let d = DVec3::new(0.8, 0.5, 0.1).normalize();
        // The lapse is exact on the internal seam: 6.5 K per km above sea level,
        // nothing below it.
        let drop = f.temperature_at_elevation(d, 0.0) - f.temperature_at_elevation(d, 5_000.0);
        assert!((drop - 32.5).abs() < 1e-9, "lapse drop {drop} ≠ 32.5");
        assert_eq!(
            f.temperature_at_elevation(d, -3_000.0),
            f.temperature_at_elevation(d, 0.0),
            "no lapse below sea level"
        );
        // Airless bodies skip the lapse entirely.
        let moon = SurfaceField::new(9, 700_000.0);
        assert_eq!(
            moon.temperature_at_elevation(d, 0.0),
            moon.temperature_at_elevation(d, 5_000.0)
        );
        // Latitude cools: polar directions are far colder than equatorial ones
        // (75 K gradient dwarfs the ±few-K latitude warp).
        for f in [f, moon] {
            let polar = f.temperature_at_elevation(DVec3::Z, 0.0);
            let equatorial = f.temperature_at_elevation(DVec3::X, 0.0);
            assert!(
                equatorial - polar > 40.0,
                "expected a strong equator→pole drop, got {equatorial} → {polar}"
            );
        }
    }

    #[test]
    fn polar_cold_sits_on_the_rotation_axis_not_y() {
        // The WI 868 polar-axis fix, characterized: bodygen rotates about +Z and
        // the pre-868 material() iced ±Y. Cold must follow the axis.
        let z_axis =
            SurfaceField::with_params(3, 2_000_000.0, CraterParams::default(), temperate_climate());
        for f in [z_axis] {
            let pole = f.temperature_at_elevation(DVec3::Z, 0.0);
            for equatorial in [DVec3::X, DVec3::Y, DVec3::NEG_Y] {
                assert!(
                    f.temperature_at_elevation(equatorial, 0.0) > pole + 40.0,
                    "±Y must be equatorial (warm) on a Z-rotator"
                );
            }
        }
        // And the axis is respected when it is not +Z.
        let mut tilted_climate = temperate_climate();
        tilted_climate.axis = DVec3::X;
        let x_axis =
            SurfaceField::with_params(3, 2_000_000.0, CraterParams::default(), tilted_climate);
        assert!(
            x_axis.temperature_at_elevation(DVec3::Z, 0.0)
                > x_axis.temperature_at_elevation(DVec3::X, 0.0) + 40.0,
            "cold pole must follow the rotation axis"
        );
    }

    #[test]
    fn families_follow_the_medium_and_biomes_vary() {
        // Atmospheric family via the asset path (BodyAsset::earthlike has an
        // atmosphere), airless via bodygen's Moon.
        let mut asset = BodyAsset::earthlike();
        asset.surface.seed = 21;
        assert_eq!(
            SurfaceField::from_asset(&asset)
                .biome_weights(DVec3::X)
                .family(),
            BiomeFamily::Atmospheric
        );
        let moon_asset = crate::bodygen::generate(918_273_645, crate::bodygen::Archetype::Moon);
        let moon = SurfaceField::from_asset(&moon_asset);
        assert_eq!(moon.biome_weights(DVec3::X).family(), BiomeFamily::Airless);

        // A temperate world shows a real taxonomy: several distinct dominant
        // biomes, ice caps at the rotation poles, ocean-family rows on deep
        // low-latitude seafloor. (Dominant use here is the discrete debug view —
        // asserting the classification, not consuming it physically.)
        let f = SurfaceField::with_params(
            21,
            2_000_000.0,
            CraterParams::default(),
            temperate_climate(),
        );
        let mut names = std::collections::HashSet::new();
        for d in dirs() {
            names.insert(f.biome_weights(d).dominant().name);
        }
        assert!(
            names.len() >= 4,
            "a temperate world should show ≥ 4 dominant biomes, got {names:?}"
        );
        assert_eq!(f.biome_weights(DVec3::Z).dominant().name, "ice cap");
        // Deep + equatorial + gentle ⇒ ocean family. Scan a denser direction set
        // for qualifying sites (coverage-guarded so the scenario can't hollow).
        let mut checked = 0;
        for i in 0..1_500 {
            let a = i as f64 * 0.618_033_988_75 * std::f64::consts::TAU;
            let z = -0.4 + 0.8 * (i as f64 + 0.5) / 1_500.0; // low latitudes only
            let r = (1.0 - z * z).max(0.0).sqrt();
            let d = DVec3::new(r * a.cos(), r * a.sin(), z); // z is the axis
            if f.elevation(d) < -0.08 * f.amplitude {
                let n = f.normal(d);
                if 1.0 - n.dot(d) < 0.015 {
                    let name = f.biome_weights(d).dominant().name;
                    assert!(
                        name == "ocean" || name == "shallows",
                        "deep gentle low-latitude floor should be ocean-family, got {name}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 0,
            "no qualifying seafloor sites — re-site the scan"
        );

        // The moon varies too (albedo/roughness classes), entirely airless-table.
        let mut moon_names = std::collections::HashSet::new();
        for d in dirs() {
            moon_names.insert(moon.biome_weights(d).dominant().name);
        }
        assert!(moon_names.len() >= 2, "airless variety: {moon_names:?}");
    }

    #[test]
    fn biome_params_apply_and_defaults_are_equivalent() {
        // WI 870: the per-body knobs shift the classifier exactly as documented,
        // and an untouched recipe is bit-identical to the default constructor.
        let base =
            SurfaceField::with_params(9, 2_000_000.0, CraterParams::default(), temperate_climate());
        let mut warm_climate = temperate_climate();
        warm_climate.params.temperature = 20.0;
        let warm = SurfaceField::with_params(9, 2_000_000.0, CraterParams::default(), warm_climate);
        let mut wet_climate = temperate_climate();
        wet_climate.params.moisture = 0.25;
        wet_climate.params.moisture_scale = 2.0;
        let wet = SurfaceField::with_params(9, 2_000_000.0, CraterParams::default(), wet_climate);

        for d in dirs() {
            // Temperature: the offset moves the seam by exactly its value.
            let dt = warm.temperature_at_elevation(d, 0.0) - base.temperature_at_elevation(d, 0.0);
            assert!((dt - 20.0).abs() < 1e-12, "offset must apply exactly: {dt}");
            // Moisture: midpoint + scaled deviation, still clamped to [0, 1].
            let m0 = base.moisture(d);
            let m1 = wet.moisture(d);
            assert!((0.0..=1.0).contains(&m1));
            let expected = (0.5 + 0.25 + 2.0 * (m0 - 0.5)).clamp(0.0, 1.0);
            assert!(
                (m1 - expected).abs() < 1e-12,
                "moisture knobs must follow the documented formula ({m0} -> {m1})"
            );
        }
        // The warm body is consumer-visibly warmer at the pole.
        let pole = DVec3::Z;
        assert!(warm.temperature(pole) > base.temperature(pole));

        // Default-equivalence: an asset with an untouched material area
        // classifies bit-identically to the explicit-defaults constructor.
        let mut asset = BodyAsset::earthlike_ice_age();
        asset.surface.seed = 9;
        asset.radius = 2_000_000.0;
        let from_asset = SurfaceField::from_asset(&asset);
        let explicit = SurfaceField::with_params(
            9,
            2_000_000.0,
            CraterParams::default(),
            crate::biome::BodyClimate::from_asset(&asset),
        );
        let table = crate::biome::biome_table(from_asset.biome_weights(DVec3::X).family());
        for d in dirs() {
            let a = from_asset.biome_weights(d);
            let b = explicit.biome_weights(d);
            for i in 0..table.len() {
                assert_eq!(a.weight_of(i), b.weight_of(i));
            }
        }
    }

    #[test]
    fn earthlike_reads_temperate_and_its_ice_age_sibling_reads_cold() {
        // WI 875: the earthlike medium's own surface ambient now equals the ISA
        // sea-level anchor, so the canonical earthlike reads temperate with **no**
        // per-asset offset; the ice-age sibling carries an explicit cold offset
        // derived from the ocean-freeze band, pushing its surface below freezing.
        // Physics (the shared medium) is identical between the two.
        use crate::biome::{OCEAN_FREEZE_RAMP_K, OCEAN_FREEZE_THRESHOLD_K};
        use crate::body_asset::EARTHLIKE_ICE_AGE_OFFSET;
        use crate::fluid::ISA_SEA_LEVEL_TEMPERATURE;
        let temperate = SurfaceField::from_asset(&BodyAsset::earthlike());
        let ice_age = SurfaceField::from_asset(&BodyAsset::earthlike_ice_age());

        // The temperate asset reads its medium's own value; the medium equals ISA.
        let medium_t = BodyAsset::earthlike().fluid_medium.atmosphere_temperature;
        assert!((medium_t - ISA_SEA_LEVEL_TEMPERATURE).abs() < 1e-12);
        // The ice-age surface sits at/below the point where the ocean kernel is
        // zero (guaranteed frozen), derived from the classifier's own band.
        let ice_age_surface = ISA_SEA_LEVEL_TEMPERATURE + EARTHLIKE_ICE_AGE_OFFSET;
        assert!(ice_age_surface <= OCEAN_FREEZE_THRESHOLD_K - OCEAN_FREEZE_RAMP_K + 1e-9);

        // Body-equatorial sea level reads ≈ ISA on the temperate asset and the
        // derived frozen target on the sibling (± the small latitude warp).
        for equator in [DVec3::X, DVec3::Y, DVec3::new(0.7, -0.7, 0.0).normalize()] {
            let t = temperate.temperature_at_elevation(equator, 0.0);
            assert!(
                (t - ISA_SEA_LEVEL_TEMPERATURE).abs() < 0.5,
                "temperate equator should read ≈ ISA, got {t}"
            );
            let t = ice_age.temperature_at_elevation(equator, 0.0);
            assert!(
                (t - ice_age_surface).abs() < 0.5,
                "ice-age equator should read the derived frozen target, got {t}"
            );
        }
        // The temperate world keeps polar ice but is not ice at the equator;
        // the ice-age world is ice cap even at the equator.
        assert_eq!(temperate.biome_weights(DVec3::Z).dominant().name, "ice cap");
        assert_ne!(temperate.biome_weights(DVec3::X).dominant().name, "ice cap");
        assert_eq!(ice_age.biome_weights(DVec3::X).dominant().name, "ice cap");

        // Same physics: identical derived central body and medium.
        let (a, b) = (BodyAsset::earthlike(), BodyAsset::earthlike_ice_age());
        assert_eq!(a.central_body().mu, b.central_body().mu);
        assert_eq!(a.central_body().radius, b.central_body().radius);
        assert_eq!(a.fluid_medium(), b.fluid_medium());
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn dry_worlds_have_no_marine_biomes() {
        // WI 869 finding: a rocky planet (atmosphere, no ocean) classified its
        // basins as "ocean" — the marine rows fire on the raw elevation channel.
        // Without a sea level the channel floors above the marine bands, so dry
        // basins read as low plains (climate decides), never ocean/shallows/beach.
        let mut climate = temperate_climate();
        climate.sea_level = None;
        let f = SurfaceField::with_params(12, 6_588_000.0, CraterParams::default(), climate);
        let mut deep = 0;
        for i in 0..800 {
            let a = i as f64 * 0.618_033_988_75 * std::f64::consts::TAU;
            let z = -1.0 + 2.0 * (i as f64 + 0.5) / 800.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let d = DVec3::new(r * a.cos(), r * a.sin(), z);
            if f.elevation(d) < -900.0 {
                deep += 1; // the marine rows' old habitat — must be exercised
            }
            let name = f.biome_weights(d).dominant().name;
            assert!(
                name != "ocean" && name != "shallows" && name != "beach",
                "dry world classified {name} at a basin"
            );
        }
        assert!(deep > 0, "no deep basins sampled — re-site the scan");
        // And an ocean-bearing body still gets its ocean (the flooring is
        // strictly a no-ocean behavior).
        let wet = SurfaceField::with_params(
            12,
            6_588_000.0,
            CraterParams::default(),
            temperate_climate(),
        );
        let has_ocean = (0..800).any(|i| {
            let a = i as f64 * 0.618_033_988_75 * std::f64::consts::TAU;
            let z = -1.0 + 2.0 * (i as f64 + 0.5) / 800.0;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let d = DVec3::new(r * a.cos(), r * a.sin(), z);
            wet.biome_weights(d).dominant().name == "ocean"
        });
        assert!(has_ocean, "ocean-bearing body lost its ocean biome");
    }

    #[test]
    fn blended_material_lies_in_the_family_hull() {
        for f in [
            SurfaceField::with_params(
                11,
                1_500_000.0,
                CraterParams::default(),
                temperate_climate(),
            ),
            SurfaceField::new(11, 700_000.0),
        ] {
            let table = crate::biome::biome_table(f.biome_weights(DVec3::X).family());
            let fmin = table
                .iter()
                .map(|r| r.material.friction)
                .fold(f64::MAX, f64::min);
            let fmax = table
                .iter()
                .map(|r| r.material.friction)
                .fold(f64::MIN, f64::max);
            let rmin = table
                .iter()
                .map(|r| r.material.rolling_resistance)
                .fold(f64::MAX, f64::min);
            let rmax = table
                .iter()
                .map(|r| r.material.rolling_resistance)
                .fold(f64::MIN, f64::max);
            for d in dirs() {
                let m = f.material(d);
                assert!(m.friction >= fmin - 1e-12 && m.friction <= fmax + 1e-12);
                assert!(
                    m.rolling_resistance >= rmin - 1e-12 && m.rolling_resistance <= rmax + 1e-12
                );
            }
        }
    }

    #[test]
    fn biome_frontiers_are_continuous() {
        // The WI 866 continuity pattern one layer up: march great-circle arcs at
        // ~3 m steps and assert every per-id biome weight and the blended
        // material move smoothly — this is the check on the whole layered
        // continuity argument (smooth kernels of continuous inputs; equal raw
        // weights at any top-k ranking swap; constant floor). A hard cut
        // anywhere (a box edge, an override threshold, the truncation) shows as
        // an O(0.1..1) jump in one step, orders above the smooth rate.
        fn rotate(v: DVec3, axis: DVec3, ang: f64) -> DVec3 {
            v * ang.cos() + axis.cross(v) * ang.sin() + axis * axis.dot(v) * (1.0 - ang.cos())
        }
        let bodies = [
            (
                "atmospheric",
                SurfaceField::with_params(
                    7,
                    730_000.0,
                    CraterParams::default(),
                    temperate_climate(),
                ),
            ),
            ("airless", SurfaceField::new(7, 730_000.0)),
        ];
        for (label, f) in bodies {
            let table = crate::biome::biome_table(f.biome_weights(DVec3::X).family());
            let start = DVec3::new(0.3, -0.9, 0.2).normalize();
            let axis = start.cross(DVec3::Y).normalize();
            let step = 4.0e-6; // rad ≈ 3 m of arc on this body
            let (a0, a1) = (0.35_f64, 0.60_f64);
            let n = ((a1 - a0) / step) as usize;
            let weights_at = |d: DVec3| {
                let w = f.biome_weights(d);
                let mut v = [0.0_f64; 16];
                for (i, slot) in v.iter_mut().enumerate().take(table.len()) {
                    *slot = w.weight_of(i);
                }
                (v, w.dominant_index(), w.material())
            };
            let (mut prev_w, mut prev_dom, mut prev_m) = weights_at(rotate(start, axis, a0));
            let mut dom_changes = 0u32;
            let mut max_dw = 0.0_f64;
            let mut max_df = 0.0_f64;
            for i in 1..=n {
                let d = rotate(start, axis, a0 + i as f64 * step);
                let (w, dom, m) = weights_at(d);
                for (a, b) in w.iter().zip(prev_w) {
                    max_dw = max_dw.max((a - b).abs());
                }
                max_df = max_df
                    .max((m.friction - prev_m.friction).abs())
                    .max((m.rolling_resistance - prev_m.rolling_resistance).abs());
                if dom != prev_dom {
                    dom_changes += 1;
                }
                (prev_w, prev_dom, prev_m) = (w, dom, m);
            }
            // Coverage guard: the arc must actually cross biome frontiers, or
            // the assertions above stop exercising anything.
            assert!(
                dom_changes >= 3,
                "{label}: arc crossed only {dom_changes} frontiers — re-site it"
            );
            // Smooth-rate bounds, calibrated: measured smooth maxima are
            // max_dw ≈ 0.0032 / max_df ≈ 0.0018 (atmospheric) and 0.0018 /
            // 0.0005 (airless) — >10× headroom below the bounds, while a hard
            // cut anywhere (verified by temporarily making Band::kernel a step)
            // exceeds them by an order of magnitude.
            assert!(
                max_dw < 0.05,
                "{label}: a biome weight stepped {max_dw} in one ~3 m sample"
            );
            assert!(
                max_df < 0.02,
                "{label}: blended material stepped {max_df} in one ~3 m sample"
            );
        }
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

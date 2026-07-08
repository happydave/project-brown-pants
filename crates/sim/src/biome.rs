//! The biome layer's pure data + classification half (WI 868).
//!
//! A **biome is a data row, not code**: each entry in a per-family table carries
//! its climate box and gates (as smooth [`Band`] kernels over a shared channel
//! vector), a tint (plain RGB data for the render work item), a reserved
//! texture-set name (for splatting), and the contact coefficients the material
//! bridge blends. Adding or tuning a biome is a table edit — no new control flow.
//!
//! **Weights, never a hard id.** [`classify`] evaluates every row's kernel
//! product over a precomputed [`ClimateSample`], takes the top
//! [`BIOME_BLEND_K`] rows, and normalizes — so every consumer interpolates and
//! no threshold draws an iso-line on the sphere (the crater-arc lesson,
//! WI 866). A dominant-biome accessor exists for *discrete* consumers (naming,
//! missions, the debug overlay); nothing physical or per-pixel-visible may
//! branch on it.
//!
//! Continuity argument (load-bearing): every kernel is a smooth ramp of a
//! continuous input, so raw weights are continuous; at any point where the
//! ranking between the k-th and (k+1)-th rows swaps, the two raw weights are
//! equal — so the truncated, renormalized blend is continuous across selection
//! changes too. The per-family fallback row carries a small constant additive
//! `floor`, keeping the pre-normalization sum strictly positive everywhere
//! (normalization never divides by ~0). The field-level arc-march test
//! (`surface_field`) is the check on this whole argument.
//!
//! The other half — evaluating the climate fields themselves (temperature,
//! moisture, albedo/roughness, crater-bowl) — lives in
//! [`crate::surface_field`], which owns the seed and the noise internals.

use crate::body_asset::BodyAsset;
use crate::surface::SurfaceMaterial;
use glam::DVec3;

/// Which classifier family a body uses — chosen by its medium, exactly like the
/// renderer's `body_tint` rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiomeFamily {
    /// Bodies with an atmosphere: the full temperature × moisture climate grid.
    Atmospheric,
    /// Airless bodies: albedo/roughness/cold-trap classification, no moisture.
    Airless,
}

/// Per-body biome/climate knobs (WI 870), read from the **reserved**
/// `SurfaceRecipe.material` area — the WI 782 `CraterParams` pattern verbatim
/// (a defaulted `serde_json::Value`, so no persistence-format change; lenient:
/// absent / null / non-object values and missing or non-numeric keys all fall
/// back to defaults). Recognized keys:
/// - `"temperature"` — additive offset (Kelvin) on the **classifier's** base
///   temperature, clamped to ±100. The physics medium is untouched (WI 875
///   owns that side).
/// - `"moisture"` — additive offset on the moisture field's midpoint,
///   clamped to ±1.
/// - `"moisture_scale"` — multiplier on the moisture field's deviation from
///   its (shifted) midpoint — widens/narrows the wet–dry contrast — clamped
///   to [0, 4].
///
/// A palette/variant selector key stays **reserved** (unread) for the render
/// work items. Constant per body, so every derived field stays continuous.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiomeParams {
    /// Classifier base-temperature offset, Kelvin (default 0).
    pub temperature: f64,
    /// Moisture midpoint offset (default 0).
    pub moisture: f64,
    /// Moisture deviation multiplier (default 1).
    pub moisture_scale: f64,
}

impl Default for BiomeParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            moisture: 0.0,
            moisture_scale: 1.0,
        }
    }
}

impl BiomeParams {
    /// Parses the reserved recipe area. Never fails; anything unrecognized
    /// yields the default for that key.
    pub fn from_value(v: &serde_json::Value) -> Self {
        let get = |key: &str| v.get(key).and_then(|x| x.as_f64());
        Self {
            temperature: get("temperature").map_or(0.0, |x| x.clamp(-100.0, 100.0)),
            moisture: get("moisture").map_or(0.0, |x| x.clamp(-1.0, 1.0)),
            moisture_scale: get("moisture_scale").map_or(1.0, |x| x.clamp(0.0, 4.0)),
        }
    }
}

/// The per-body climate inputs the biome layer needs — all read from fields
/// [`BodyAsset`] already has (plus the [`BiomeParams`] knobs on the reserved
/// recipe area), so the layer adds **no persistence change**.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyClimate {
    /// The classifier family (atmosphere present or not).
    pub family: BiomeFamily,
    /// Body base temperature, Kelvin (`FluidMedium.atmosphere_temperature`).
    pub base_temperature: f64,
    /// Sea level as an elevation (metres relative to the reference radius) on
    /// ocean-bearing bodies; `None` when the medium has no ocean. Elevation 0 is
    /// the working definition WI 766 may move — it is a parameter here so the
    /// classifier never has to change.
    pub sea_level: Option<f64>,
    /// Rotation axis (unit vector) — latitude is measured against **this**, not
    /// a hardcoded coordinate (bodygen rotates about +Z; the pre-868
    /// `material()` used `d.y`, an acknowledged quirk this fixes).
    pub axis: DVec3,
    /// Per-body biome knobs from the reserved recipe area (WI 870).
    pub params: BiomeParams,
}

impl Default for BodyClimate {
    /// Defaults match `SurfaceField::new`'s "defaults" meaning: an airless
    /// 200 K body rotating about +Z (`Rotation::NONE`'s axis) with no ocean.
    fn default() -> Self {
        Self {
            family: BiomeFamily::Airless,
            base_temperature: 200.0,
            sea_level: None,
            axis: DVec3::Z,
            params: BiomeParams {
                temperature: 0.0,
                moisture: 0.0,
                moisture_scale: 1.0,
            },
        }
    }
}

impl BodyClimate {
    /// Reads the climate inputs from an asset: family from atmosphere presence,
    /// sea level from ocean presence, latitude axis from the rotation axis
    /// (normalized; a degenerate axis falls back to +Z), biome knobs from the
    /// reserved `surface.material` area (WI 870).
    pub fn from_asset(asset: &BodyAsset) -> Self {
        let m = &asset.fluid_medium;
        let axis = asset.rotation.axis.normalize_or_zero();
        Self {
            family: if m.atmosphere_surface_density > 0.0 {
                BiomeFamily::Atmospheric
            } else {
                BiomeFamily::Airless
            },
            base_temperature: m.atmosphere_temperature,
            sea_level: (m.ocean_surface_density > 0.0).then_some(0.0),
            axis: if axis == DVec3::ZERO { DVec3::Z } else { axis },
            params: BiomeParams::from_value(&asset.surface.material),
        }
    }
}

/// The classifier's input channels, one scalar each — a single vector shape
/// shared by both families (rows mark irrelevant channels [`Band::ANY`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClimateSample {
    /// Temperature, Kelvin-anchored classifier scale.
    pub temperature: f64,
    /// Moisture in `[0, 1]` (atmospheric family; neutral 0.5 for airless).
    pub moisture: f64,
    /// Elevation above sea level, normalized by the body's relief amplitude.
    pub elevation: f64,
    /// Slope measure `1 − n·d` (0 flat, larger is steeper) — the same scale the
    /// pre-868 `material()` used.
    pub slope: f64,
    /// Warped absolute latitude in `[0, 1]` against the rotation axis.
    pub latitude: f64,
    /// Surface albedo field in `[0, 1]` (airless family; neutral 0.5 otherwise).
    pub albedo: f64,
    /// Surface roughness field in `[0, 1]` (airless family; neutral 0.5 otherwise).
    pub roughness: f64,
    /// Crater-interior factor in `[0, 1]` (0 outside craters, →1 in deep bowls);
    /// a smooth ramp of the continuous crater term (airless family).
    pub bowl: f64,
}

/// Channel count/order for [`BiomeRow::bands`]. Order: temperature, moisture,
/// elevation, slope, latitude, albedo, roughness, bowl.
pub const CHANNELS: usize = 8;

/// A smooth membership band over one channel: kernel 0 at/below `lo` and
/// at/above `hi`, 1 on the plateau `[lo + ramp, hi − ramp]`, smoothstep ramps
/// between — never a hard cut (every classification threshold over a smooth
/// field draws an iso-line; WI 866).
#[derive(Clone, Copy, Debug)]
pub struct Band {
    pub lo: f64,
    pub hi: f64,
    pub ramp: f64,
}

impl Band {
    /// "Don't care": kernel 1 over any physically reachable input.
    pub const ANY: Band = Band {
        lo: -1.0e9,
        hi: 1.0e9,
        ramp: 1.0,
    };

    /// The smooth membership kernel in `[0, 1]`.
    pub fn kernel(&self, x: f64) -> f64 {
        smooth01((x - self.lo) / self.ramp) * smooth01((self.hi - x) / self.ramp)
    }
}

/// Smoothstep of `t` clamped to `[0, 1]`.
fn smooth01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One biome: identity, smooth climate box + gates, render data, contact data.
#[derive(Clone, Copy, Debug)]
pub struct BiomeRow {
    /// Human/debug name (also the discrete consumers' identity).
    pub name: &'static str,
    /// Membership bands, one per channel (see [`CHANNELS`] for the order).
    pub bands: [Band; CHANNELS],
    /// Constant additive weight floor — nonzero only on a family's fallback row,
    /// guaranteeing a strictly positive weight sum everywhere (and, being
    /// constant, changing nothing about continuity).
    pub floor: f64,
    /// Kernel gain (specificity): scales the row's kernel product so a narrow,
    /// fully-fired row (e.g. the polar cold trap) outweighs an always-on
    /// fallback whose kernel is also ~1. A constant multiplier of a continuous
    /// kernel — continuity unaffected.
    pub gain: f64,
    /// Base tint, **sRGB** in `[0, 1]` (plain data, authored perceptually like
    /// every `Color::srgb` in the app; the mesh builder converts to linear for
    /// the GPU — WI 869). Consumed by render B2.
    pub tint: [f64; 3],
    /// Terrain texture-set name under `assets/materials/` (reserved for
    /// splatting, B5); `None` ⇒ tint-only.
    pub texture_set: Option<&'static str>,
    /// Contact coefficients the material bridge blends.
    pub material: SurfaceMaterial,
}

/// Slope cap shared by every non-bedrock row: closes (smoothly) exactly where
/// the bedrock rows' slope band opens, so steep ground converges to bedrock
/// weight 1 — the "override as a smooth ramp" mechanism.
const SLOPE_CAP: Band = Band {
    lo: -1.0e9,
    hi: 0.045,
    ramp: 0.02,
};

/// The bedrock rows' opening slope band (complement of [`SLOPE_CAP`]).
const SLOPE_OPEN: Band = Band {
    lo: 0.025,
    hi: 1.0e9,
    ramp: 0.02,
};

/// Shorthand for a row's band vector:
/// `[temperature, moisture, elevation, slope, latitude, albedo, roughness, bowl]`.
/// One positional argument per channel is the point — the table rows read as
/// aligned columns.
#[expect(clippy::too_many_arguments)]
const fn bands(
    t: Band,
    m: Band,
    e: Band,
    s: Band,
    l: Band,
    a: Band,
    r: Band,
    b: Band,
) -> [Band; CHANNELS] {
    [t, m, e, s, l, a, r, b]
}

const fn band(lo: f64, hi: f64, ramp: f64) -> Band {
    Band { lo, hi, ramp }
}

/// Loose sediment/soil-family coefficients used by several rows.
const SOIL: SurfaceMaterial = SurfaceMaterial {
    friction: 0.6,
    rolling_resistance: 0.06,
};
const SAND: SurfaceMaterial = SurfaceMaterial {
    friction: 0.5,
    rolling_resistance: 0.12,
};
const MUD: SurfaceMaterial = SurfaceMaterial {
    friction: 0.35,
    rolling_resistance: 0.25,
};

/// The atmospheric-family table. Boxes overlap by at least a ramp width between
/// climatic neighbours so (with the fallback floor) coverage has no holes.
/// All numbers are tunable data; the tests pin structural properties only.
pub const ATMOSPHERIC_BIOMES: &[BiomeRow] = &[
    BiomeRow {
        name: "ocean",
        bands: bands(
            band(254.0, 1.0e9, 6.0), // warm enough not to be ice-capped
            Band::ANY,
            band(-1.0e9, -0.04, 0.04), // well below sea level
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.05, 0.15, 0.35],
        texture_set: None,
        material: MUD,
    },
    BiomeRow {
        name: "shallows",
        bands: bands(
            band(254.0, 1.0e9, 6.0),
            Band::ANY,
            band(-0.09, -0.005, 0.03),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.10, 0.30, 0.45],
        texture_set: None,
        material: MUD,
    },
    BiomeRow {
        name: "beach",
        bands: bands(
            band(268.0, 1.0e9, 8.0),
            Band::ANY,
            band(-0.015, 0.025, 0.015),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.76, 0.70, 0.50],
        texture_set: Some("sand"),
        material: SAND,
    },
    BiomeRow {
        name: "swamp",
        bands: bands(
            band(281.0, 315.0, 8.0),
            band(0.6, 1.0e9, 0.12),
            band(-0.01, 0.07, 0.03),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.20, 0.28, 0.16],
        texture_set: Some("mud"),
        material: MUD,
    },
    BiomeRow {
        // The atmospheric fallback: its floor keeps the weight sum positive
        // everywhere (an off-grid climate reads as steppe, which is sane).
        name: "grassland",
        bands: bands(
            band(268.0, 302.0, 8.0),
            band(0.20, 0.75, 0.12),
            band(-0.01, 0.55, 0.04),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.01,
        gain: 1.0,
        tint: [0.35, 0.48, 0.22],
        texture_set: Some("grass"),
        material: SOIL,
    },
    BiomeRow {
        name: "savanna",
        bands: bands(
            band(293.0, 322.0, 8.0),
            band(0.15, 0.55, 0.12),
            band(-0.01, 0.7, 0.05),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.55, 0.52, 0.25],
        texture_set: Some("steppe"),
        material: SOIL,
    },
    BiomeRow {
        name: "desert",
        bands: bands(
            band(291.0, 1.0e9, 8.0),
            band(-1.0e9, 0.32, 0.10),
            band(-0.01, 0.7, 0.05),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.72, 0.60, 0.38],
        texture_set: Some("sand"),
        material: SAND,
    },
    BiomeRow {
        name: "forest",
        bands: bands(
            band(274.0, 302.0, 8.0),
            band(0.55, 1.0e9, 0.12),
            band(-0.01, 0.5, 0.04),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.13, 0.30, 0.12],
        texture_set: Some("forest_floor"),
        material: SurfaceMaterial {
            friction: 0.65,
            rolling_resistance: 0.09,
        },
    },
    BiomeRow {
        name: "taiga",
        bands: bands(
            band(252.0, 281.0, 7.0),
            band(0.45, 1.0e9, 0.12),
            band(-0.01, 0.7, 0.05),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.16, 0.28, 0.22],
        texture_set: Some("forest_floor"),
        material: SurfaceMaterial {
            friction: 0.6,
            rolling_resistance: 0.08,
        },
    },
    BiomeRow {
        name: "tundra",
        bands: bands(
            band(248.0, 271.0, 7.0),
            Band::ANY,
            band(-0.01, 0.7, 0.05),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.45, 0.42, 0.36],
        texture_set: Some("steppe"),
        material: SurfaceMaterial {
            friction: 0.55,
            rolling_resistance: 0.10,
        },
    },
    BiomeRow {
        name: "highland",
        bands: bands(
            Band::ANY,
            Band::ANY,
            band(0.40, 0.85, 0.08),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.42, 0.36, 0.28],
        texture_set: Some("rock"),
        material: SurfaceMaterial {
            friction: 0.7,
            rolling_resistance: 0.05,
        },
    },
    BiomeRow {
        name: "alpine",
        bands: bands(
            Band::ANY,
            Band::ANY,
            band(0.70, 1.0e9, 0.10),
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.55, 0.55, 0.56],
        texture_set: Some("rock"),
        material: SurfaceMaterial {
            friction: 0.8,
            rolling_resistance: 0.04,
        },
    },
    BiomeRow {
        name: "bedrock",
        bands: bands(
            Band::ANY,
            Band::ANY,
            Band::ANY,
            SLOPE_OPEN,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.33, 0.32, 0.33],
        texture_set: Some("rock"),
        material: SurfaceMaterial::BEDROCK,
    },
    BiomeRow {
        name: "ice cap",
        bands: bands(
            band(-1.0e9, 258.0, 8.0),
            Band::ANY,
            Band::ANY,
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.88, 0.92, 0.96],
        texture_set: Some("snow"),
        material: SurfaceMaterial::ICE,
    },
];

/// The airless-family table. Classification runs on albedo/roughness noise,
/// elevation, slope, and the polar cold trap (latitude ∧ crater interior) —
/// no moisture. Ejecta rays are deferred (WI 873/B6).
pub const AIRLESS_BIOMES: &[BiomeRow] = &[
    BiomeRow {
        // The airless fallback row (floor: see the grassland note).
        name: "regolith plains",
        bands: bands(
            Band::ANY,
            Band::ANY,
            Band::ANY,
            SLOPE_CAP,
            Band::ANY,
            band(0.30, 0.75, 0.10),
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.02,
        gain: 1.0,
        tint: [0.42, 0.41, 0.40],
        texture_set: Some("regolith_fine"),
        material: SurfaceMaterial::REGOLITH,
    },
    BiomeRow {
        name: "maria",
        bands: bands(
            Band::ANY,
            Band::ANY,
            band(-1.0e9, 0.12, 0.06),
            SLOPE_CAP,
            Band::ANY,
            band(-1.0e9, 0.38, 0.08),
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.22, 0.22, 0.23],
        texture_set: Some("basalt"),
        material: SurfaceMaterial {
            friction: 0.55,
            rolling_resistance: 0.06,
        },
    },
    BiomeRow {
        name: "bright highlands",
        bands: bands(
            Band::ANY,
            Band::ANY,
            band(0.0, 1.0e9, 0.08),
            SLOPE_CAP,
            Band::ANY,
            band(0.62, 1.0e9, 0.08),
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.60, 0.59, 0.57],
        texture_set: Some("regolith_fine"),
        material: SurfaceMaterial {
            friction: 0.65,
            rolling_resistance: 0.06,
        },
    },
    BiomeRow {
        name: "boulder fields",
        bands: bands(
            Band::ANY,
            Band::ANY,
            Band::ANY,
            SLOPE_CAP,
            Band::ANY,
            Band::ANY,
            band(0.68, 1.0e9, 0.08),
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.35, 0.35, 0.36],
        texture_set: Some("regolith_coarse"),
        material: SurfaceMaterial {
            friction: 0.7,
            rolling_resistance: 0.10,
        },
    },
    BiomeRow {
        name: "cold-trap ice",
        bands: bands(
            Band::ANY,
            Band::ANY,
            Band::ANY,
            SLOPE_CAP,
            band(0.88, 1.0e9, 0.05), // polar…
            Band::ANY,
            Band::ANY,
            band(0.45, 1.0e9, 0.20), // …∧ inside a crater bowl
        ),
        floor: 0.0,
        // A fully-fired cold trap must outweigh the always-on plains fallback
        // (both kernels ≈ 1 in a polar bowl) — ice is the notable feature there.
        gain: 3.0,
        tint: [0.82, 0.88, 0.95],
        texture_set: Some("snow"),
        material: SurfaceMaterial::ICE,
    },
    BiomeRow {
        name: "bedrock outcrop",
        bands: bands(
            Band::ANY,
            Band::ANY,
            Band::ANY,
            SLOPE_OPEN,
            Band::ANY,
            Band::ANY,
            Band::ANY,
            Band::ANY,
        ),
        floor: 0.0,
        gain: 1.0,
        tint: [0.30, 0.30, 0.32],
        texture_set: Some("rock"),
        material: SurfaceMaterial::BEDROCK,
    },
];

/// A family's biome table.
pub fn biome_table(family: BiomeFamily) -> &'static [BiomeRow] {
    match family {
        BiomeFamily::Atmospheric => ATMOSPHERIC_BIOMES,
        BiomeFamily::Airless => AIRLESS_BIOMES,
    }
}

/// Canonical terrain-texture array layer order (WI 872): the union of both
/// families' `texture_set` names, **alphabetical**. The asset-harness KTX2
/// packer derives the identical order independently (a plain sort), so no
/// manifest is shared between the repos; a test pins this list against the
/// tables.
pub const TERRAIN_TEXTURE_LAYERS: [&str; 10] = [
    "basalt",
    "forest_floor",
    "grass",
    "mud",
    "regolith_coarse",
    "regolith_fine",
    "rock",
    "sand",
    "snow",
    "steppe",
];

/// Splat slot budget (WI 872): per-vertex weights ride two vec4 attributes, so a
/// family may use at most 8 distinct textured sets. Body-wide fixed slot
/// semantics (not per-chunk palettes) is what makes weight interpolation
/// well-defined everywhere and removes the chunk-border palette-mismatch seam
/// class — see the WI 872 plan. Guarded by a table test, not runtime logic.
pub const MAX_TEXTURE_SLOTS: usize = 8;

/// A family's texture slots: its distinct `texture_set` names in
/// first-appearance (table) order. Const so the query path stays
/// allocation-free; a test derives the same lists from the tables.
pub fn texture_slot_names(family: BiomeFamily) -> &'static [&'static str] {
    match family {
        BiomeFamily::Atmospheric => &[
            "sand",
            "mud",
            "grass",
            "steppe",
            "forest_floor",
            "rock",
            "snow",
        ],
        BiomeFamily::Airless => &["regolith_fine", "basalt", "regolith_coarse", "snow", "rock"],
    }
}

/// Each slot's layer index in the [`TERRAIN_TEXTURE_LAYERS`] arrays (unused
/// slots report 0 — harmless: their weight is always exactly 0).
pub fn slot_layer_indices(family: BiomeFamily) -> [u32; MAX_TEXTURE_SLOTS] {
    let mut out = [0u32; MAX_TEXTURE_SLOTS];
    for (slot, name) in texture_slot_names(family).iter().enumerate() {
        let layer = TERRAIN_TEXTURE_LAYERS
            .iter()
            .position(|l| l == name)
            .expect("slot set present in the canonical layer list (table-tested)");
        out[slot] = layer as u32;
    }
    out
}

/// Each slot's anchor tint (sRGB, like row tints): the mean tint of the family
/// rows consuming that slot's texture set — the same derivation that produced
/// the harness textures' tone anchors, so the shader's macro modulation
/// (`vertex tint / Σ w·anchor`) is ≈ 1 where the texture already matches the
/// biome look and re-expresses the tint difference where several rows share one
/// texture (highland vs alpine rock). Computed from the table: no second copy
/// of the harness constants to drift.
pub fn slot_anchor_tints(family: BiomeFamily) -> [[f64; 3]; MAX_TEXTURE_SLOTS] {
    let mut out = [[0.0; 3]; MAX_TEXTURE_SLOTS];
    for (slot, name) in texture_slot_names(family).iter().enumerate() {
        let (mut sum, mut count) = ([0.0; 3], 0.0);
        for row in biome_table(family) {
            if row.texture_set == Some(*name) {
                for (acc, c) in sum.iter_mut().zip(row.tint) {
                    *acc += c;
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            out[slot] = sum.map(|c| c / count);
        }
    }
    out
}

/// How many rows a blend keeps (the design's k ≈ 4).
pub const BIOME_BLEND_K: usize = 4;

/// A classification result: up to [`BIOME_BLEND_K`] rows of one family with
/// normalized blend weights (descending; weights sum to 1). Fixed-size — no
/// allocation in the query path.
#[derive(Clone, Copy, Debug)]
pub struct BiomeWeights {
    family: BiomeFamily,
    /// (row index into the family table, normalized weight), weight-descending.
    entries: [(usize, f64); BIOME_BLEND_K],
    len: usize,
}

impl BiomeWeights {
    /// The family whose table the indices refer to.
    pub fn family(&self) -> BiomeFamily {
        self.family
    }

    /// The winning rows and their normalized weights, descending.
    pub fn iter(&self) -> impl Iterator<Item = (&'static BiomeRow, f64)> + '_ {
        let table = biome_table(self.family);
        self.entries[..self.len]
            .iter()
            .map(move |&(i, w)| (&table[i], w))
    }

    /// This point's weight for table row `index` (0 when the row is not among
    /// the kept top-k) — the per-id view the continuity tests compare.
    pub fn weight_of(&self, index: usize) -> f64 {
        self.entries[..self.len]
            .iter()
            .find(|&&(i, _)| i == index)
            .map_or(0.0, |&(_, w)| w)
    }

    /// Index (into the family table) of the highest-weight row.
    ///
    /// **Discrete consumers only** (naming, missions, the debug overlay — whose
    /// job is to show the classification, artifacts and all). Nothing physical
    /// or per-pixel-visible may branch on this: use the weights.
    pub fn dominant_index(&self) -> usize {
        self.entries[0].0
    }

    /// The highest-weight row (see [`Self::dominant_index`]'s consumer rule).
    pub fn dominant(&self) -> &'static BiomeRow {
        &biome_table(self.family)[self.dominant_index()]
    }

    /// The weight-blended contact material — the material bridge's value. A
    /// convex combination of row coefficients (weights are normalized), so the
    /// blend always lies within the family's row-value hull.
    pub fn material(&self) -> SurfaceMaterial {
        let mut friction = 0.0;
        let mut rolling = 0.0;
        for (row, w) in self.iter() {
            friction += w * row.material.friction;
            rolling += w * row.material.rolling_resistance;
        }
        SurfaceMaterial {
            friction,
            rolling_resistance: rolling,
        }
    }

    /// The weights folded onto the family's texture slots (WI 872): each kept
    /// row's weight accumulates into its `texture_set`'s slot; rows without a
    /// texture (marine) contribute nothing, so the slot sum is 1 minus the
    /// untextured weight — the shader mixes toward the pure tint by exactly
    /// that deficit, which is what renders seabeds tint-only and feathers a
    /// beach→ocean frontier. Continuous wherever the weights are (a fixed
    /// linear fold).
    pub fn slot_weights(&self) -> [f32; MAX_TEXTURE_SLOTS] {
        let slots = texture_slot_names(self.family);
        let mut out = [0.0f32; MAX_TEXTURE_SLOTS];
        for (row, w) in self.iter() {
            if let Some(set) = row.texture_set {
                let slot = slots
                    .iter()
                    .position(|s| *s == set)
                    .expect("every table texture_set has a slot (table-tested)");
                out[slot] += w as f32;
            }
        }
        out
    }

    /// The weight-blended tint (linear RGB) — the render work item's phase-1
    /// per-vertex value, exposed here so B2 stays a pure consumer.
    pub fn tint(&self) -> [f64; 3] {
        let mut t = [0.0; 3];
        for (row, w) in self.iter() {
            for (acc, c) in t.iter_mut().zip(row.tint) {
                *acc += w * c;
            }
        }
        t
    }
}

/// Classifies a sample against a family table: every row's kernel product
/// (+ floor), top-k by weight, normalized. Pure; no allocation.
pub fn classify(family: BiomeFamily, sample: &ClimateSample) -> BiomeWeights {
    let table = biome_table(family);
    let x = [
        sample.temperature,
        sample.moisture,
        sample.elevation,
        sample.slope,
        sample.latitude,
        sample.albedo,
        sample.roughness,
        sample.bowl,
    ];
    // Top-k selection by insertion (tables are ≤ ~14 rows; k = 4).
    let mut top: [(usize, f64); BIOME_BLEND_K] = [(0, -1.0); BIOME_BLEND_K];
    for (i, row) in table.iter().enumerate() {
        let mut kernel = 1.0;
        for (b, v) in row.bands.iter().zip(x) {
            kernel *= b.kernel(v);
        }
        let w = row.floor + row.gain * kernel;
        // Insert (i, w) if it beats the current tail; ties keep the earlier row
        // (stable, deterministic).
        let mut cand = (i, w);
        for slot in &mut top {
            if cand.1 > slot.1 {
                std::mem::swap(&mut cand, slot);
            }
        }
    }
    let mut len = 0;
    while len < BIOME_BLEND_K && top[len].1 > 0.0 {
        len += 1;
    }
    // The fallback row's floor guarantees sum > 0 (table-validity test pins a
    // floored row per family), so normalization is always well-defined.
    let sum: f64 = top[..len].iter().map(|&(_, w)| w).sum();
    for slot in &mut top[..len] {
        slot.1 /= sum;
    }
    for slot in &mut top[len..] {
        slot.1 = 0.0;
    }
    BiomeWeights {
        family,
        entries: top,
        len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral() -> ClimateSample {
        ClimateSample {
            temperature: 288.0,
            moisture: 0.5,
            elevation: 0.1,
            slope: 0.0,
            latitude: 0.3,
            albedo: 0.5,
            roughness: 0.5,
            bowl: 0.0,
        }
    }

    #[test]
    fn band_kernel_is_bounded_smooth_and_any_is_one() {
        let b = Band {
            lo: 10.0,
            hi: 20.0,
            ramp: 2.0,
        };
        let mut prev = b.kernel(0.0);
        for i in 1..=3000 {
            let x = i as f64 * 0.01; // 0 → 30 in steps of 0.01
            let k = b.kernel(x);
            assert!((0.0..=1.0).contains(&k));
            // Smooth: a step of 0.01 over a ramp of 2.0 moves the kernel < 1%.
            assert!((k - prev).abs() < 0.01, "kernel stepped at x={x}");
            prev = k;
        }
        assert_eq!(b.kernel(9.9), 0.0);
        assert_eq!(b.kernel(15.0), 1.0);
        assert_eq!(b.kernel(20.1), 0.0);
        for x in [-1.0e6, -3.0, 0.0, 0.5, 288.0, 1.0e6] {
            assert_eq!(Band::ANY.kernel(x), 1.0);
        }
    }

    #[test]
    fn tables_are_valid() {
        for family in [BiomeFamily::Atmospheric, BiomeFamily::Airless] {
            let table = biome_table(family);
            assert!(
                table.len() >= BIOME_BLEND_K && table.len() <= 16,
                "table size out of range"
            );
            let mut floored = 0;
            for row in table {
                assert!(!row.name.is_empty());
                for b in &row.bands {
                    assert!(b.lo < b.hi, "{}: malformed band", row.name);
                    assert!(b.ramp > 0.0, "{}: non-positive ramp", row.name);
                }
                assert!(
                    row.material.friction.is_finite()
                        && (0.0..=2.0).contains(&row.material.friction),
                    "{}: friction out of range",
                    row.name
                );
                assert!(
                    row.material.rolling_resistance.is_finite()
                        && row.material.rolling_resistance >= 0.0,
                    "{}: rolling resistance out of range",
                    row.name
                );
                for c in row.tint {
                    assert!((0.0..=1.0).contains(&c), "{}: tint out of range", row.name);
                }
                assert!(row.floor >= 0.0);
                assert!(row.gain > 0.0, "{}: non-positive gain", row.name);
                if row.floor > 0.0 {
                    floored += 1;
                    // The floor row must be reachable everywhere it matters:
                    // its own kernels may close, but the floor is unconditional.
                }
            }
            assert!(
                floored >= 1,
                "{family:?}: a fallback row with a positive floor is required (zero-sum guard)"
            );
            // The existing three named materials survive as row values.
            let has = |m: SurfaceMaterial| table.iter().any(|r| r.material == m);
            assert!(has(SurfaceMaterial::BEDROCK), "{family:?}: bedrock row");
            assert!(has(SurfaceMaterial::ICE), "{family:?}: ice row");
        }
        assert!(AIRLESS_BIOMES
            .iter()
            .any(|r| r.material == SurfaceMaterial::REGOLITH));
    }

    /// WI 872: the texture-slot machinery is table-derived and budget-bound.
    /// The canonical layer list is exactly the alphabetically-sorted union of
    /// both families' `texture_set` names (the KTX2 packer derives the same
    /// order by sorting — this test is the cross-repo contract's Rust half);
    /// each family's slot list is its table's first-appearance order, fits the
    /// two-attribute budget, and maps into the canonical list; anchors are
    /// means of in-range tints, so they are in range.
    #[test]
    fn texture_slots_match_the_tables_and_fit_the_budget() {
        let mut union: Vec<&str> = [BiomeFamily::Atmospheric, BiomeFamily::Airless]
            .iter()
            .flat_map(|&f| biome_table(f).iter().filter_map(|r| r.texture_set))
            .collect();
        union.sort_unstable();
        union.dedup();
        assert_eq!(union, TERRAIN_TEXTURE_LAYERS, "canonical layer list drift");

        for family in [BiomeFamily::Atmospheric, BiomeFamily::Airless] {
            let mut first_appearance: Vec<&str> = Vec::new();
            for row in biome_table(family) {
                if let Some(set) = row.texture_set {
                    if !first_appearance.contains(&set) {
                        first_appearance.push(set);
                    }
                }
            }
            assert_eq!(
                first_appearance,
                texture_slot_names(family),
                "{family:?}: slot list drift"
            );
            assert!(
                first_appearance.len() <= MAX_TEXTURE_SLOTS,
                "{family:?}: more textured sets than slot budget"
            );
            let layers = slot_layer_indices(family);
            for (slot, name) in texture_slot_names(family).iter().enumerate() {
                assert_eq!(
                    TERRAIN_TEXTURE_LAYERS[layers[slot] as usize], *name,
                    "{family:?}: slot {slot} layer mapping"
                );
            }
            let anchors = slot_anchor_tints(family);
            for (slot, _) in texture_slot_names(family).iter().enumerate() {
                for c in anchors[slot] {
                    assert!((0.0..=1.0).contains(&c), "{family:?}: anchor out of range");
                }
                assert!(
                    anchors[slot].iter().any(|&c| c > 0.0),
                    "{family:?}: slot {slot} anchor is black (no consuming row?)"
                );
            }
        }
    }

    /// WI 872: the slot fold is a fixed linear map of the kept weights — the
    /// slot sum equals 1 minus the untextured-row weight, and a marine-only
    /// blend folds to all-zero slots (the shader's pure-tint case).
    #[test]
    fn slot_weights_fold_conserves_textured_weight() {
        let sample = ClimateSample {
            temperature: 288.0,
            moisture: 0.6,
            elevation: 0.1,
            slope: 0.005,
            latitude: 0.3,
            albedo: 0.5,
            roughness: 0.5,
            bowl: 0.0,
        };
        for family in [BiomeFamily::Atmospheric, BiomeFamily::Airless] {
            let w = classify(family, &sample);
            let untextured: f64 = w
                .iter()
                .filter(|(row, _)| row.texture_set.is_none())
                .map(|(_, wt)| wt)
                .sum();
            let slot_sum: f32 = w.slot_weights().iter().sum();
            assert!(
                (slot_sum as f64 - (1.0 - untextured)).abs() < 1e-6,
                "{family:?}: slot sum {slot_sum} vs textured weight {}",
                1.0 - untextured
            );
        }
        // Deep ocean: the marine rows dominate and carry no texture.
        let deep = ClimateSample {
            temperature: 290.0,
            moisture: 0.7,
            elevation: -0.5,
            slope: 0.001,
            latitude: 0.1,
            albedo: 0.5,
            roughness: 0.5,
            bowl: 0.0,
        };
        let w = classify(BiomeFamily::Atmospheric, &deep);
        let slot_sum: f32 = w.slot_weights().iter().sum();
        assert!(
            slot_sum < 0.15,
            "deep ocean should be nearly untextured (got {slot_sum})"
        );
    }

    #[test]
    fn classify_returns_normalized_descending_topk() {
        // Sweep a grid of samples (both families) far wider than any climate
        // box: weights are finite, non-negative, ≤ k entries, descending, and
        // sum to 1 — including in "off-grid" corners where only a floor row
        // fires (the zero-sum guard).
        for family in [BiomeFamily::Atmospheric, BiomeFamily::Airless] {
            for t in [80.0, 250.0, 288.0, 320.0, 500.0] {
                for m in [0.0, 0.5, 1.0] {
                    for e in [-1.5, -0.05, 0.1, 0.8, 1.5] {
                        for s in [0.0, 0.03, 0.2] {
                            let sample = ClimateSample {
                                temperature: t,
                                moisture: m,
                                elevation: e,
                                slope: s,
                                latitude: 0.4,
                                albedo: 0.5,
                                roughness: 0.5,
                                bowl: 0.0,
                            };
                            let w = classify(family, &sample);
                            let mut sum = 0.0;
                            let mut prev = f64::INFINITY;
                            let mut n = 0;
                            for (_, wi) in w.iter() {
                                assert!(wi.is_finite() && wi >= 0.0);
                                assert!(wi <= prev, "weights must be descending");
                                prev = wi;
                                sum += wi;
                                n += 1;
                            }
                            assert!((1..=BIOME_BLEND_K).contains(&n));
                            assert!(
                                (sum - 1.0).abs() < 1e-12,
                                "weights must normalize (family {family:?}, T={t}, M={m}, E={e}, S={s}): sum {sum}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn classification_matches_the_intended_climate_boxes() {
        // Spot checks that the tables encode the design's taxonomy (dominant
        // biome only — this is the discrete debug view of the tables, not a
        // physical consumer).
        let cases = [
            (
                "ocean",
                ClimateSample {
                    elevation: -0.5,
                    ..neutral()
                },
            ),
            (
                "beach",
                ClimateSample {
                    elevation: 0.005,
                    ..neutral()
                },
            ),
            (
                "desert",
                ClimateSample {
                    temperature: 310.0,
                    moisture: 0.05,
                    ..neutral()
                },
            ),
            (
                "forest",
                ClimateSample {
                    temperature: 288.0,
                    moisture: 0.8,
                    ..neutral()
                },
            ),
            (
                "ice cap",
                ClimateSample {
                    temperature: 210.0,
                    ..neutral()
                },
            ),
            (
                "bedrock",
                ClimateSample {
                    slope: 0.2,
                    ..neutral()
                },
            ),
            (
                "alpine",
                ClimateSample {
                    elevation: 0.95,
                    ..neutral()
                },
            ),
        ];
        for (name, sample) in cases {
            let w = classify(BiomeFamily::Atmospheric, &sample);
            assert_eq!(w.dominant().name, name, "sample {sample:?}");
        }
        let cold_trap = ClimateSample {
            temperature: 120.0,
            latitude: 0.97,
            bowl: 0.9,
            ..neutral()
        };
        assert_eq!(
            classify(BiomeFamily::Airless, &cold_trap).dominant().name,
            "cold-trap ice"
        );
        let maria = ClimateSample {
            temperature: 200.0,
            elevation: -0.4,
            albedo: 0.1,
            ..neutral()
        };
        assert_eq!(
            classify(BiomeFamily::Airless, &maria).dominant().name,
            "maria"
        );
    }

    #[test]
    fn blended_outputs_stay_in_the_row_hull() {
        for family in [BiomeFamily::Atmospheric, BiomeFamily::Airless] {
            let table = biome_table(family);
            let fmin = table
                .iter()
                .map(|r| r.material.friction)
                .fold(f64::MAX, f64::min);
            let fmax = table
                .iter()
                .map(|r| r.material.friction)
                .fold(f64::MIN, f64::max);
            for e in [-0.8, -0.02, 0.0, 0.3, 0.9] {
                for s in [0.0, 0.03, 0.1] {
                    let sample = ClimateSample {
                        elevation: e,
                        slope: s,
                        ..neutral()
                    };
                    let m = classify(family, &sample).material();
                    assert!(
                        m.friction >= fmin - 1e-12 && m.friction <= fmax + 1e-12,
                        "blend must stay in the hull"
                    );
                    assert!(m.rolling_resistance >= 0.0);
                    let tint = classify(family, &sample).tint();
                    for c in tint {
                        assert!((0.0..=1.0).contains(&c));
                    }
                }
            }
        }
    }

    #[test]
    fn biome_params_parse_leniently_and_clamp() {
        // The CraterParams contract (WI 782/870): absent / null / garbage /
        // partial all yield per-key defaults; recognized keys clamp.
        assert_eq!(
            BiomeParams::from_value(&serde_json::Value::Null),
            BiomeParams::default()
        );
        assert_eq!(
            BiomeParams::from_value(&serde_json::json!("weather")),
            BiomeParams::default()
        );
        assert_eq!(
            BiomeParams::from_value(&serde_json::json!({"bogus": 1, "temperature": "x"})),
            BiomeParams::default()
        );
        assert_eq!(
            BiomeParams::from_value(&serde_json::json!({"temperature": 38.15})),
            BiomeParams {
                temperature: 38.15,
                ..BiomeParams::default()
            }
        );
        assert_eq!(
            BiomeParams::from_value(&serde_json::json!({
                "temperature": 500.0, "moisture": -9.0, "moisture_scale": 99.0
            })),
            BiomeParams {
                temperature: 100.0,
                moisture: -1.0,
                moisture_scale: 4.0,
            }
        );
        // Unknown keys coexist with known ones (the reserved palette selector).
        assert_eq!(
            BiomeParams::from_value(&serde_json::json!({"palette": "lush", "moisture": 0.2})),
            BiomeParams {
                moisture: 0.2,
                ..BiomeParams::default()
            }
        );
    }

    #[test]
    fn body_climate_reads_the_asset_family_rules() {
        use crate::bodygen::{generate, Archetype};
        let moon = BodyClimate::from_asset(&generate(7, Archetype::Moon));
        assert_eq!(moon.family, BiomeFamily::Airless);
        assert_eq!(moon.sea_level, None);

        let rocky = BodyClimate::from_asset(&generate(7, Archetype::RockyPlanet));
        assert_eq!(rocky.family, BiomeFamily::Atmospheric);
        assert_eq!(rocky.sea_level, None);

        let ocean = BodyClimate::from_asset(&generate(7, Archetype::OceanWorld));
        assert_eq!(ocean.family, BiomeFamily::Atmospheric);
        assert_eq!(ocean.sea_level, Some(0.0));
        // bodygen rotates about +Z; the axis must flow through (the polar fix).
        assert_eq!(ocean.axis, DVec3::Z);

        // Degenerate axis falls back to +Z, never NaN.
        let mut broken = generate(7, Archetype::Moon);
        broken.rotation.axis = DVec3::ZERO;
        assert_eq!(BodyClimate::from_asset(&broken).axis, DVec3::Z);
    }
}

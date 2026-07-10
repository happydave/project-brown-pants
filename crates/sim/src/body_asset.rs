//! Celestial body assets (WI 760).
//!
//! A [`BodyAsset`] is the **intrinsic** definition of a celestial body — the
//! reusable "planet/moon as data" primitive of the world-building aspect. It
//! unifies what today is scattered across [`crate::sim::CentralBody`] (gravity +
//! radius), [`crate::fluid::FluidMedium`] (atmosphere/ocean), and the intrinsic
//! half of [`crate::universe::Body`] (`mu`/`radius`), and adds rotation and a
//! surface recipe.
//!
//! **Asset ⊕ placement split (load-bearing).** A `BodyAsset` carries *only*
//! intrinsic properties — never placement. A body's parent, orbit, and
//! sphere-of-influence are supplied by a *System* (WI 761) that references the
//! asset, so the same asset can be dropped into different systems and orbit
//! differently. This is what makes bodies reusable across scenes.
//!
//! The asset is plain serde data (no rendering), so it round-trips through the
//! versioned document format ([`crate::persist`]) and is unit-testable headless.
//! Detailed terrain/crater/material parameters (WI 763) and render parameters
//! (WI 764) are *reserved, extensible* areas here — defaulted on load — so those
//! work items fill them without a format-version change.

use crate::biome::{OCEAN_FREEZE_RAMP_K, OCEAN_FREEZE_THRESHOLD_K};
use crate::fluid::FluidMedium;
use crate::sim::CentralBody;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// A body's rotation about its own axis.
///
/// `sidereal_period` is the time (seconds) for one full rotation in the inertial
/// frame. A period of `0.0` means **non-rotating** (zero angular velocity) — the
/// sentinel is `0.0` rather than an infinite period because infinities are not
/// representable in JSON. Consumers (the rotating-frame handling in WI 765) treat
/// `0.0` as "no rotation".
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rotation {
    /// Spin axis (unit vector) in the body's inertial frame.
    pub axis: DVec3,
    /// Sidereal rotation period, seconds. `0.0` ⇒ non-rotating.
    pub sidereal_period: f64,
}

impl Rotation {
    /// A non-rotating body (zero angular velocity).
    pub const NONE: Rotation = Rotation {
        axis: DVec3::Z,
        sidereal_period: 0.0,
    };

    /// Earth-like rotation: one sidereal day about +Z.
    pub const EARTHLIKE: Rotation = Rotation {
        axis: DVec3::Z,
        sidereal_period: 86_164.090_5, // one sidereal day, seconds
    };
}

/// The recipe a body's procedural surface is generated from.
///
/// At WI 760 only the master `seed` was load-bearing; the terrain/crater/material
/// parameter areas were **reserved** (opaque, defaulted on load), following the
/// reserved-container idiom of [`crate::persist`], so surface work items could
/// populate them without a format-version bump. WI 782 defined the `crater` area
/// and WI 870 the `material` area (see the field docs); `terrain` remains
/// reserved.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SurfaceRecipe {
    /// Master seed — the deterministic source of the whole surface (same seed ⇒
    /// same surface, every visit and every new game).
    pub seed: u64,
    /// The **ordered surface-layer stack** (WI 892), replacing the flat
    /// `terrain`/`crater`/`material` areas (v2 documents migrate at load —
    /// `persist`'s v2 arm). Application order IS list order (an explicit sort
    /// key, never map iteration — design I1); the ladder addresses elements
    /// by stable id, never index. Empty is the common case and means
    /// all-defaults, exactly like the null areas it replaces.
    #[serde(default)]
    pub layers: Vec<SurfaceLayer>,
}

impl SurfaceRecipe {
    /// A recipe seeded with `seed` and an empty layer stack.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            seed,
            layers: Vec::new(),
        }
    }

    /// The parameters of the **first enabled** layer of `layer_type` — the one
    /// consumer contract (pinned by WI 892's plan): a disabled layer reads as
    /// absent, an absent layer means defaults (`Null`, the lenient readers'
    /// existing posture). Blending multiple enabled layers of one type is
    /// future generation work; until then first-enabled wins.
    pub fn params_of(&self, layer_type: SurfaceLayerType) -> &serde_json::Value {
        const NULL: serde_json::Value = serde_json::Value::Null;
        self.layers
            .iter()
            .find(|l| l.enabled && l.layer_type == layer_type)
            .map(|l| &l.params)
            .unwrap_or(&NULL)
    }
}

/// One element of the surface-layer stack (WI 892): a stable-id, typed,
/// switchable parameter carrier. The machine-created layers (the
/// `surface_temperature_offset` sugar, the persist v2 migration) use their
/// type's slug as their id — well-known, deterministic, digest-stable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceLayer {
    /// Stable identifier — the ladder's addressing key (by id, never index).
    pub id: String,
    /// Which well-known type this layer is (decides which consumer reads it
    /// and what its params mean).
    pub layer_type: SurfaceLayerType,
    /// A disabled layer is carried (and digested) but reads as absent — the
    /// design's `disable` semantics, expressed as data rather than a new op.
    pub enabled: bool,
    /// Type-specific parameters, read leniently by the type's consumer
    /// (`crater`: `density`/`depth` — `surface_field::CraterParams`;
    /// `material`: `temperature`/`moisture`/`moisture_scale` —
    /// `biome::BiomeParams`; `terrain`: reserved, opaque).
    #[serde(default)]
    pub params: serde_json::Value,
}

/// The well-known surface-layer types (WI 892) — the flat reserved areas of
/// the pre-stack `SurfaceRecipe`, promoted to layer types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceLayerType {
    /// Base-terrain noise parameters (reserved since WI 763 — no reader yet).
    Terrain,
    /// Crater-population parameters (WI 782).
    Crater,
    /// Biome/climate parameters (WI 870/875).
    Material,
}

impl SurfaceLayer {
    /// A machine-created (well-known) layer: enabled, with its type's slug as
    /// its id — the shape the `surface_temperature_offset` sugar and the
    /// persist v2 migration produce.
    pub fn well_known(layer_type: SurfaceLayerType, params: serde_json::Value) -> SurfaceLayer {
        SurfaceLayer {
            id: layer_type.slug().to_string(),
            layer_type,
            enabled: true,
            params,
        }
    }
}

impl SurfaceLayerType {
    /// All types, in the canonical (migration) order.
    pub const ALL: [SurfaceLayerType; 3] = [
        SurfaceLayerType::Terrain,
        SurfaceLayerType::Crater,
        SurfaceLayerType::Material,
    ];

    /// The type's stable slug — its serde spelling, the machine-created
    /// layer id, and the digest input.
    pub fn slug(self) -> &'static str {
        match self {
            SurfaceLayerType::Terrain => "terrain",
            SurfaceLayerType::Crater => "crater",
            SurfaceLayerType::Material => "material",
        }
    }

    /// Inverse of [`slug`](Self::slug); `None` for an unknown slug (persisted
    /// or authored input is reported, never panicked on).
    pub fn from_slug(slug: &str) -> Option<SurfaceLayerType> {
        SurfaceLayerType::ALL.into_iter().find(|t| t.slug() == slug)
    }
}

/// The intrinsic definition of a celestial body — reusable, serializable, and
/// free of placement (see the module docs for the asset ⊕ placement split).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyAsset {
    /// Stable identifier.
    pub id: String,
    /// Human-facing display name.
    pub name: String,
    /// Gravitational parameter (μ = G·M), m³/s².
    pub mu: f64,
    /// Surface (sea-level) radius, metres.
    pub radius: f64,
    /// Rotation about the body's own axis.
    pub rotation: Rotation,
    /// The surrounding-medium profile (atmosphere and/or ocean).
    pub fluid_medium: FluidMedium,
    /// The procedural-surface recipe.
    pub surface: SurfaceRecipe,
    /// Reserved: render/scattering parameters (WI 764). Opaque and defaulted on
    /// load so the render work item can fill it without a format-version change.
    #[serde(default)]
    pub render: serde_json::Value,
}

/// The ice-age sibling's target surface temperature, K (WI 875): at/below the
/// point where the ocean kernel contributes zero weight, so open water — hence
/// any non-frozen surface class — is impossible and the surface is guaranteed to
/// classify as ice. Derived from the classifier's own ocean-freeze band, never a
/// bare number.
const EARTHLIKE_ICE_AGE_SURFACE_TEMPERATURE: f64 = OCEAN_FREEZE_THRESHOLD_K - OCEAN_FREEZE_RAMP_K;

/// The classifier temperature offset the ice-age Earth-like sibling carries
/// (WI 875): its guaranteed-frozen target minus the medium's own
/// `atmosphere_temperature` (now [`crate::fluid::ISA_SEA_LEVEL_TEMPERATURE`]). Derived,
/// so it tracks any future change to the medium constant. The *temperate*
/// earthlike needs no offset — its medium already equals the ISA surface anchor,
/// so it reads temperate with no per-asset override (WI 875 un-magicked the
/// physics constant that WI 870 had to bridge with a +38 K classifier offset).
pub const EARTHLIKE_ICE_AGE_OFFSET: f64 =
    EARTHLIKE_ICE_AGE_SURFACE_TEMPERATURE - FluidMedium::EARTHLIKE.atmosphere_temperature;

impl BodyAsset {
    /// The canonical Earth-like body. Since WI 884 this **resolves the shipped
    /// recipe** (the embedded canonical-bodies pack, `content::canonical_body`)
    /// rather than assembling constants in code — the recipe is the single
    /// authored source of body configuration. Its derived [`CentralBody`] still
    /// equals [`CentralBody::EARTHLIKE`] and its fluid medium still equals
    /// [`FluidMedium::EARTHLIKE`] — now guaranteed by drift-guard tests welding
    /// the recipe to the physics constants, rather than by construction. Since
    /// WI 875 its medium's own surface ambient equals the ISA sea-level anchor,
    /// so the surface **reads temperate** to the biome layer with **no per-asset
    /// override** (the ice-age look lives on in [`Self::earthlike_ice_age`]).
    pub fn earthlike() -> Self {
        crate::content::canonical_body("earthlike")
    }

    /// The ice-age sibling of [`Self::earthlike`]: the identical body — same
    /// physics, same terrain seed — carrying an explicit **cold** classifier
    /// offset (the WI-875 derivation, authored in the shipped recipe and pinned
    /// to [`EARTHLIKE_ICE_AGE_OFFSET`] by a drift-guard test) that pushes its
    /// surface below the ocean-freeze point, so the biome layer classifies it as
    /// an ice-age world while physics stays identical to its temperate twin.
    /// Resolves the shipped recipe (WI 884), where it is authored as
    /// `parent: earthlike` + the one cold override.
    pub fn earthlike_ice_age() -> Self {
        crate::content::canonical_body("earthlike-ice-age")
    }

    /// The [`CentralBody`] this asset defines (gravity + radius) — the intrinsic
    /// half of a `universe::Body`, derived rather than stored separately.
    pub fn central_body(&self) -> CentralBody {
        CentralBody {
            mu: self.mu,
            radius: self.radius,
        }
    }

    /// The fluid-medium profile this asset carries.
    pub fn fluid_medium(&self) -> FluidMedium {
        self.fluid_medium
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative-tolerance float comparison for round-trip assertions (serde_json
    /// is not guaranteed bit-identical for all f64; assert structure + tolerance).
    fn approx(a: f64, b: f64) {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= 1e-9 * scale,
            "expected {a} ≈ {b} (rel 1e-9)"
        );
    }

    fn assert_medium_approx(a: &FluidMedium, b: &FluidMedium) {
        approx(a.atmosphere_surface_density, b.atmosphere_surface_density);
        approx(a.atmosphere_surface_pressure, b.atmosphere_surface_pressure);
        approx(a.atmosphere_scale_height, b.atmosphere_scale_height);
        approx(a.ocean_surface_density, b.ocean_surface_density);
        approx(a.ocean_surface_pressure, b.ocean_surface_pressure);
        approx(a.ocean_density_gradient, b.ocean_density_gradient);
        approx(a.gravity, b.gravity);
        approx(a.atmosphere_temperature, b.atmosphere_temperature);
        approx(a.ocean_temperature, b.ocean_temperature);
    }

    #[test]
    fn earthlike_asset_reproduces_the_central_body_constant() {
        // Characterization test (designreview R6): the Earth-like asset's derived
        // central body must match the canonical constant field-for-field.
        let cb = BodyAsset::earthlike().central_body();
        assert_eq!(cb.mu, CentralBody::EARTHLIKE.mu);
        assert_eq!(cb.radius, CentralBody::EARTHLIKE.radius);
    }

    #[test]
    fn earthlike_asset_reproduces_the_fluid_medium_constant() {
        assert_eq!(
            BodyAsset::earthlike().fluid_medium(),
            FluidMedium::EARTHLIKE
        );
    }

    #[test]
    fn asset_carries_no_placement_only_intrinsics() {
        // The type has no parent/orbit/soi fields — a compile-time guarantee of the
        // asset ⊕ placement split. This test documents the intent and exercises the
        // intrinsic accessors.
        let a = BodyAsset::earthlike();
        assert_eq!(a.central_body().mu, a.mu);
        assert_eq!(a.central_body().radius, a.radius);
    }

    #[test]
    fn body_asset_serde_round_trips_within_tolerance() {
        let asset = BodyAsset::earthlike();
        let json = serde_json::to_string(&asset).unwrap();
        let back: BodyAsset = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, asset.id);
        assert_eq!(back.name, asset.name);
        approx(back.mu, asset.mu);
        approx(back.radius, asset.radius);
        approx(
            back.rotation.sidereal_period,
            asset.rotation.sidereal_period,
        );
        assert_eq!(back.rotation.axis, asset.rotation.axis);
        assert_medium_approx(&back.fluid_medium, &asset.fluid_medium);
        assert_eq!(back.surface.seed, asset.surface.seed);
    }

    #[test]
    fn non_rotating_sentinel_is_zero_period_not_infinity() {
        // JSON cannot represent infinities; a non-rotating body must round-trip.
        let json = serde_json::to_string(&Rotation::NONE).unwrap();
        let back: Rotation = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sidereal_period, 0.0);
    }

    #[test]
    fn reserved_recipe_and_render_areas_default_when_absent() {
        // A document without the reserved areas still loads (forward extensibility
        // for WI 763/764): only `seed` is required in the recipe.
        let minimal = r#"{
            "id": "m", "name": "Minimal", "mu": 1.0, "radius": 2.0,
            "rotation": { "axis": [0.0, 0.0, 1.0], "sidereal_period": 0.0 },
            "fluid_medium": {
                "atmosphere_surface_density": 0.0, "atmosphere_surface_pressure": 0.0,
                "atmosphere_scale_height": 1.0, "ocean_surface_density": 0.0,
                "ocean_surface_pressure": 0.0, "ocean_density_gradient": 0.0, "gravity": 0.0
            },
            "surface": { "seed": 42 }
        }"#;
        let back: BodyAsset = serde_json::from_str(minimal).unwrap();
        assert_eq!(back.surface.seed, 42);
        assert!(back.surface.layers.is_empty(), "absent stack ⇒ empty");
        assert!(back.render.is_null());
    }

    /// WI 892: the one consumer contract — `params_of` returns the first
    /// **enabled** layer of the type; disabled layers read as absent, an
    /// absent type means `Null` (the lenient readers' defaults).
    #[test]
    fn params_of_is_first_enabled_of_type() {
        use serde_json::json;
        let mut s = SurfaceRecipe::from_seed(1);
        assert!(s.params_of(SurfaceLayerType::Crater).is_null(), "empty");
        s.layers = vec![
            SurfaceLayer {
                id: "dead".into(),
                layer_type: SurfaceLayerType::Crater,
                enabled: false,
                params: json!({"density": 9.0}),
            },
            SurfaceLayer::well_known(SurfaceLayerType::Crater, json!({"density": 2.0})),
            SurfaceLayer {
                id: "later".into(),
                layer_type: SurfaceLayerType::Crater,
                enabled: true,
                params: json!({"density": 3.0}),
            },
        ];
        assert_eq!(
            s.params_of(SurfaceLayerType::Crater)["density"].as_f64(),
            Some(2.0),
            "first ENABLED of type wins; disabled skipped"
        );
        assert!(
            s.params_of(SurfaceLayerType::Material).is_null(),
            "absent type ⇒ Null"
        );
    }
}

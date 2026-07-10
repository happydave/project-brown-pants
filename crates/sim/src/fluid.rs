//! The fluid-medium field abstraction (WI 497).
//!
//! Embodies the governing discipline: *do not hardcode atmosphere — model the
//! surrounding medium as a field.* Vacuum, air, and ocean are the **same shape
//! with different constants**, so a single [`FluidMedium`] of numeric parameters
//! describes any of them. Adding a new medium (an exotic methane ocean, a thicker
//! atmosphere) is new constant data, never new control flow.
//!
//! The field is defined over a **signed altitude** `h` relative to a body's
//! reference surface: `h > 0` is altitude (outward), `h < 0` is depth (inward).
//! The atmosphere law governs `h >= 0`, the ocean (liquid) law governs `h < 0` —
//! one field, opposite signs about the surface. This is the "decompression =
//! implosion, sign-flipped" collapse made concrete.
//!
//! [`FluidMedium`] is decoupled from body geometry: it is a pure function of
//! altitude. Use [`FluidMedium::sample_altitude`] directly, or
//! [`FluidMedium::sample_at`] to derive altitude from a world position and the
//! body's surface radius. Sampling is allocation-free and deterministic.

use crate::frame::WorldPos;
use serde::{Deserialize, Serialize};

/// Which medium a sample fell in, derived from the query position. This is a
/// medium *identity* discriminant only — scattering and rendering parameters are
/// out of scope here (they belong to the renderer, WI 504).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediumKind {
    /// No appreciable medium (zero density).
    #[default]
    Vacuum,
    /// Compressible gas above the surface.
    Atmosphere,
    /// Liquid below the surface.
    Liquid,
}

/// The local state of the surrounding medium at a queried position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluidSample {
    /// Mass density, kg/m³. Always finite and non-negative.
    pub density: f64,
    /// Pressure, Pa. Always finite and non-negative.
    pub pressure: f64,
    /// Which medium this sample fell in.
    pub medium: MediumKind,
    /// Medium temperature, K (WI 694) — the target a part's surface exchanges heat
    /// toward by convection (large coupling in water, small in air, none in vacuum).
    /// Defaulted on load so pre-temperature samples stay backward-loadable.
    #[serde(default)]
    pub temperature: f64,
}

/// A data-driven fluid-medium profile about a body's reference surface.
///
/// Carries an atmosphere layer (governing `h >= 0`) and an ocean layer (governing
/// `h < 0`). A layer is "absent" when its surface density is zero, in which case
/// that side reads as [`MediumKind::Vacuum`]. Every field is plain data, so any
/// medium — including a body with both an atmosphere and an ocean — is expressed
/// by choosing constants.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluidMedium {
    /// Atmosphere density at the surface (`h = 0`), kg/m³. Zero ⇒ no atmosphere.
    pub atmosphere_surface_density: f64,
    /// Atmosphere pressure at the surface, Pa.
    pub atmosphere_surface_pressure: f64,
    /// Atmosphere scale height, metres. Must be `> 0`.
    pub atmosphere_scale_height: f64,
    /// Ocean density at the surface (`h = 0`), kg/m³. Zero ⇒ no ocean.
    pub ocean_surface_density: f64,
    /// Ocean pressure at the surface, Pa (the medium pressing down on the liquid;
    /// set equal to the atmosphere surface pressure for continuity).
    pub ocean_surface_pressure: f64,
    /// Linear density increase per metre of depth, kg/m³ per m. `>= 0`. Zero for
    /// the incompressible case (constant density with depth).
    pub ocean_density_gradient: f64,
    /// Gravitational acceleration used for ocean hydrostatic pressure, m/s². `>= 0`.
    pub gravity: f64,
    /// Atmosphere temperature, K — the **surface (sea-level) ambient** temperature
    /// of this body. The medium is *isothermal* (`sample_altitude` returns this at
    /// every altitude above the surface), so this is a single quantity, not a
    /// separate "surface" vs "bulk/effective" pair: a lapse-aware vertical profile
    /// is a possible future physics item, not modelled here. Two consumers read it,
    /// both wanting a surface ambient — the re-entry recovery/quench target in
    /// `thermal.rs` (WI 694; at rest the skin cools toward this) and the biome
    /// classifier's base temperature (`biome::BodyClimate::from_asset`, WI 868/870).
    /// Defaulted on load.
    #[serde(default)]
    pub atmosphere_temperature: f64,
    /// Ocean temperature, K (WI 694) — the convective-exchange target below the
    /// surface (a hot part quenches toward this). Defaulted on load.
    #[serde(default)]
    pub ocean_temperature: f64,
}

/// ISA sea-level standard temperature: **288.15 K = exactly 15 °C**, the
/// sea-level datum of the International Standard Atmosphere (ISO 2533). The named
/// physical anchor for "an Earth-like surface is temperate" — used as the
/// canonical [`FluidMedium::EARTHLIKE`] surface ambient so no bare number stands
/// in for it (WI 875; the anchor itself was introduced by WI 870).
pub const ISA_SEA_LEVEL_TEMPERATURE: f64 = 288.15;

/// Mean molar mass of dry air, kg/mol — the earthlike atmosphere's composition
/// intent (WI 887). The digits **must** match the shipped recipe's
/// `mean_molar_mass` (`crates/sim/content/bodies.ron`): both parse to the same
/// `f64`, which is what keeps [`FluidMedium::EARTHLIKE`]'s derived scale height
/// bit-identical to the resolved canonical body (weld-tested).
pub const EARTHLIKE_MEAN_MOLAR_MASS: f64 = 0.028_964_4;

impl FluidMedium {
    /// Empty space: zero density and pressure at every altitude.
    pub const VACUUM: FluidMedium = FluidMedium {
        atmosphere_surface_density: 0.0,
        atmosphere_surface_pressure: 0.0,
        atmosphere_scale_height: 1.0, // any positive value; density is zero regardless
        ocean_surface_density: 0.0,
        ocean_surface_pressure: 0.0,
        ocean_density_gradient: 0.0,
        gravity: 0.0,
        atmosphere_temperature: 3.0, // immaterial (no medium); a cold-space value
        ocean_temperature: 3.0,
    };

    /// An Earth-like body carrying **both** an atmosphere (above the surface) and
    /// an ocean (below it) — the canonical medium for the vacuum→atmosphere→ocean
    /// descent (the dive, WI 509). Surface pressures match across the boundary so
    /// pressure is continuous; density jumps (air → water) as physics requires.
    ///
    /// Since **WI 887** the two formerly-authored incoherent values are
    /// **derived**: `gravity` = μ/R² (≈ 9.8542 m/s², not the legacy 9.81) and
    /// `atmosphere_scale_height` = R·T/(M·g) (≈ 8394.6 m, not the legacy 8500) —
    /// computed at const-eval by the *same* `body_derive` relations the recipe
    /// resolver runs, so this constant is bit-identical to the shipped
    /// `bodies.ron` earthlike by construction (the exact-equality weld tests
    /// enforce it; a transcribed literal here is exactly the drift they exist
    /// to catch).
    pub const EARTHLIKE: FluidMedium = FluidMedium {
        atmosphere_surface_density: 1.225, // kg/m³ at sea level (ISA; authored anchor)
        atmosphere_surface_pressure: 101_325.0, // 1 atm
        // Derived (WI 887): isothermal hydrostatic H at the ISA anchor.
        atmosphere_scale_height: crate::body_derive::scale_height(
            ISA_SEA_LEVEL_TEMPERATURE,
            EARTHLIKE_MEAN_MOLAR_MASS,
            crate::body_derive::surface_gravity(
                crate::sim::CentralBody::EARTHLIKE.mu,
                crate::sim::CentralBody::EARTHLIKE.radius,
            ),
        ),
        ocean_surface_density: 1_025.0,    // kg/m³ seawater
        ocean_surface_pressure: 101_325.0, // continuous with the atmosphere
        ocean_density_gradient: 0.0,       // near-incompressible: constant density
        // Derived (WI 887): g = μ/R² from the canonical central body.
        gravity: crate::body_derive::surface_gravity(
            crate::sim::CentralBody::EARTHLIKE.mu,
            crate::sim::CentralBody::EARTHLIKE.radius,
        ),
        atmosphere_temperature: ISA_SEA_LEVEL_TEMPERATURE, // surface ambient, 15 °C (authored anchor)
        ocean_temperature: 290.0, // K — surface seawater (~17 °C, ISA-consistent)
    };

    /// Samples the medium at signed altitude `h` (metres): `h > 0` above the
    /// surface, `h < 0` below it. Pure, allocation-free, and guaranteed to return
    /// finite, non-negative density and pressure for any finite `h`.
    pub fn sample_altitude(&self, h: f64) -> FluidSample {
        if h >= 0.0 {
            // Atmosphere: isothermal exponential falloff.
            let falloff = (-h / self.atmosphere_scale_height).exp();
            let medium = if self.atmosphere_surface_density > 0.0 {
                MediumKind::Atmosphere
            } else {
                MediumKind::Vacuum
            };
            FluidSample {
                density: self.atmosphere_surface_density * falloff,
                pressure: self.atmosphere_surface_pressure * falloff,
                medium,
                temperature: self.atmosphere_temperature,
            }
        } else {
            // Ocean: linear density in depth; hydrostatic pressure is its integral.
            let depth = -h;
            let density = self.ocean_surface_density + self.ocean_density_gradient * depth;
            // P(d) = P0 + g · ∫₀ᵈ ρ(d') dd' = P0 + g·(ρ0·d + ½·grad·d²)
            let pressure = self.ocean_surface_pressure
                + self.gravity
                    * (self.ocean_surface_density * depth
                        + 0.5 * self.ocean_density_gradient * depth * depth);
            let medium = if self.ocean_surface_density > 0.0 {
                MediumKind::Liquid
            } else {
                MediumKind::Vacuum
            };
            FluidSample {
                density,
                pressure,
                medium,
                temperature: self.ocean_temperature,
            }
        }
    }

    /// Samples the medium at a world position, deriving altitude from the body's
    /// `surface_radius`: `h = |pos| - surface_radius`.
    pub fn sample_at(&self, pos: &WorldPos, surface_radius: f64) -> FluidSample {
        self.sample_altitude(pos.radius() - surface_radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameId;
    use glam::DVec3;
    use std::f64::consts::LN_2;

    const TOL: f64 = 1e-9;

    #[test]
    fn vacuum_is_zero_everywhere() {
        for &h in &[-10_000.0, -1.0, 0.0, 1.0, 1.0e6] {
            let s = FluidMedium::VACUUM.sample_altitude(h);
            assert_eq!(s.density, 0.0);
            assert_eq!(s.pressure, 0.0);
            assert_eq!(s.medium, MediumKind::Vacuum);
        }
    }

    #[test]
    fn atmosphere_surface_values_at_zero_altitude() {
        let s = FluidMedium::EARTHLIKE.sample_altitude(0.0);
        assert!((s.density - 1.225).abs() < TOL);
        assert!((s.pressure - 101_325.0).abs() < 1e-3);
        assert_eq!(s.medium, MediumKind::Atmosphere);
    }

    #[test]
    fn atmosphere_halves_at_scale_height_times_ln2() {
        let m = FluidMedium::EARTHLIKE;
        let h = m.atmosphere_scale_height * LN_2;
        let s = m.sample_altitude(h);
        assert!((s.density - m.atmosphere_surface_density * 0.5).abs() < 1e-6);
    }

    #[test]
    fn atmosphere_decreases_monotonically_and_approaches_zero() {
        let m = FluidMedium::EARTHLIKE;
        let mut prev = f64::INFINITY;
        for k in 0..50 {
            let s = m.sample_altitude(k as f64 * 2_000.0);
            assert!(
                s.density < prev,
                "density must strictly decrease with altitude"
            );
            prev = s.density;
        }
        // Far above the surface, density is essentially zero.
        assert!(m.sample_altitude(200_000.0).density < 1e-3);
    }

    #[test]
    fn ocean_surface_density_and_rising_pressure() {
        let m = FluidMedium::EARTHLIKE;
        let surf = m.sample_altitude(-1e-9); // just below the surface
                                             // Just below the surface we are in the liquid.
        assert_eq!(surf.medium, MediumKind::Liquid);
        assert!((surf.density - 1_025.0).abs() < TOL);

        let deep = m.sample_altitude(-100.0); // 100 m depth
        assert_eq!(deep.medium, MediumKind::Liquid);
        // Pressure rises ~ ρ·g·d above the surface pressure — with the medium's
        // own g, which since WI 887 is the derived μ/R² (≈ 9.854), not 9.81.
        let expected = 101_325.0 + 1_025.0 * m.gravity * 100.0;
        assert!((deep.pressure - expected).abs() < 1e-3);
        assert!(deep.pressure > surf.pressure);
    }

    #[test]
    fn ocean_density_non_decreasing_with_depth() {
        let m = FluidMedium::EARTHLIKE;
        let shallow = m.sample_altitude(-10.0);
        let deep = m.sample_altitude(-1_000.0);
        assert!(deep.density >= shallow.density);
    }

    #[test]
    fn pressure_is_continuous_across_the_surface() {
        let m = FluidMedium::EARTHLIKE;
        let above = m.sample_altitude(1e-9);
        let below = m.sample_altitude(-1e-9);
        // Pressure continuous (both ≈ 1 atm); density jumps air → water.
        assert!((above.pressure - below.pressure).abs() < 1e-2);
        assert!(below.density > above.density);
        assert_eq!(above.medium, MediumKind::Atmosphere);
        assert_eq!(below.medium, MediumKind::Liquid);
    }

    #[test]
    fn all_samples_finite_and_non_negative() {
        for m in [FluidMedium::VACUUM, FluidMedium::EARTHLIKE] {
            for k in -200..200 {
                let s = m.sample_altitude(k as f64 * 1_000.0);
                assert!(s.density.is_finite() && s.density >= 0.0);
                assert!(s.pressure.is_finite() && s.pressure >= 0.0);
            }
        }
    }

    #[test]
    fn sample_at_derives_altitude_from_position() {
        let surface_radius = 600_000.0;
        let pos = WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(surface_radius + 8_500.0, 0.0, 0.0),
        );
        let by_pos = FluidMedium::EARTHLIKE.sample_at(&pos, surface_radius);
        let by_alt = FluidMedium::EARTHLIKE.sample_altitude(8_500.0);
        assert_eq!(by_pos, by_alt);
    }

    #[test]
    fn ad_hoc_medium_from_data_is_data_driven() {
        // A new medium is new constants, no new code path (I3). An "exotic ocean":
        // denser, compressible (nonzero gradient), no atmosphere above.
        let exotic = FluidMedium {
            atmosphere_surface_density: 0.0,
            atmosphere_surface_pressure: 0.0,
            atmosphere_scale_height: 1.0,
            ocean_surface_density: 1_300.0,
            ocean_surface_pressure: 50_000.0,
            ocean_density_gradient: 0.01,
            gravity: 12.0,
            atmosphere_temperature: 200.0,
            ocean_temperature: 320.0,
        };
        assert_eq!(exotic.sample_altitude(10.0).medium, MediumKind::Vacuum); // vacuum above
        let d = exotic.sample_altitude(-500.0);
        assert_eq!(d.medium, MediumKind::Liquid);
        assert!((d.density - (1_300.0 + 0.01 * 500.0)).abs() < TOL);
        assert!(d.density.is_finite() && d.pressure > 50_000.0);
    }
}

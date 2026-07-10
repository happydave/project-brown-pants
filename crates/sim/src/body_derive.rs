//! Phase-2 body derivation: the named physics relations (WI 886).
//!
//! The independent → derived half of the bodies-as-recipes design: a fixed
//! `BodyRecipe` authors a small **independent** set and these pure relations
//! compute the rest at resolution (`content::validate` orchestrates the
//! pin-or-derive choice per field; this module owns only the physics). Game
//! fidelity: physically-standard approximations — equilibrium temperature,
//! ideal gas at the surface, isothermal hydrostatic atmosphere — not a
//! simulation.
//!
//! **Determinism:** `f64` throughout, and the one would-be transcendental (the
//! fourth root in [`equilibrium_temperature`]) is computed as
//! `sqrt(sqrt(x))` — IEEE-correctly-rounded hardware square roots — so no
//! libm-variant function enters body derivation and the same recipe resolves
//! bit-identically on every platform (stronger than the design's minimum,
//! which only forbids libm on the noise hot path).

/// Stefan–Boltzmann constant, W·m⁻²·K⁻⁴ (2019 SI exact-derived value).
pub(crate) const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Molar gas constant, J·mol⁻¹·K⁻¹ (2019 SI exact value).
pub(crate) const GAS_CONSTANT: f64 = 8.314462618;

/// Surface gravity `g = μ / R²`, m/s². The single spelling of gravitational
/// strength is `μ` (review N3); the medium's hydrostatic gravity derives here.
/// `const fn` (WI 887) so [`crate::fluid::FluidMedium::EARTHLIKE`] computes its
/// gravity with these *same operations* at const-eval — bit-identical to the
/// recipe resolver by construction, never a transcribed literal.
pub(crate) const fn surface_gravity(mu: f64, radius: f64) -> f64 {
    mu / (radius * radius)
}

/// Equilibrium temperature from intent-level nominal insolation `S` (W/m²) and
/// bond albedo `A`: `T_eq = (S·(1−A) / 4σ)^¼` (design C1 — "as if at nominal
/// orbit"; placement-time flux modulation is deferred by design). Fourth root
/// via nested `sqrt` for cross-platform bit-stability.
pub(crate) fn equilibrium_temperature(nominal_insolation: f64, bond_albedo: f64) -> f64 {
    let absorbed = nominal_insolation * (1.0 - bond_albedo) / (4.0 * STEFAN_BOLTZMANN);
    absorbed.sqrt().sqrt()
}

/// Surface temperature `T_surf = T_eq + ΔT_greenhouse`, K (C1: the WI-875 ISA
/// anchor generalized — the greenhouse offset is authored intent).
pub(crate) fn surface_temperature(t_eq: f64, greenhouse_delta_t: f64) -> f64 {
    t_eq + greenhouse_delta_t
}

/// Ideal-gas surface density `ρ₀ = P₀·M / (R·T_surf)`, kg/m³ (mean molar mass
/// `M` in kg/mol). Earth check: (101325, 0.0289644, 288.15) → 1.2250 — the ISA
/// value the earthlike medium has always carried is *exactly* this relation.
pub(crate) fn atmosphere_surface_density(
    surface_pressure: f64,
    mean_molar_mass: f64,
    t_surf: f64,
) -> f64 {
    surface_pressure * mean_molar_mass / (GAS_CONSTANT * t_surf)
}

/// Isothermal hydrostatic scale height `H = R·T_surf / (M·g)`, metres.
/// `const fn` (WI 887) — see [`surface_gravity`].
pub(crate) const fn scale_height(t_surf: f64, mean_molar_mass: f64, gravity: f64) -> f64 {
    GAS_CONSTANT * t_surf / (mean_molar_mass * gravity)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(x: f64, y: f64, tol: f64) {
        assert!((x - y).abs() <= tol, "{x} !~ {y} (tol {tol})");
    }

    #[test]
    fn earth_isa_density_reproduces_the_canonical_value() {
        // The earthlike medium's 1.225 kg/m³ is real ISA: this relation at ISA
        // inputs must land on it (the one derived field that is exactly honest).
        close(
            atmosphere_surface_density(101_325.0, 0.028_964_4, 288.15),
            1.2250,
            1e-4,
        );
    }

    #[test]
    fn earth_equilibrium_and_surface_temperatures_are_physical() {
        // S=1361 W/m², A=0.306 → T_eq ≈ 254 K; +34 K greenhouse ≈ 288 K.
        let t_eq = equilibrium_temperature(1361.0, 0.306);
        close(t_eq, 254.0, 1.5);
        close(surface_temperature(t_eq, 34.0), 288.0, 1.5);
    }

    #[test]
    fn earth_scale_height_and_gravity_expose_the_legacy_incoherence() {
        // The derived values earthlike **now carries** (WI 887 removed its
        // gravity/scale-height pins): g = μ/R² ≈ 9.854 (the legacy authored
        // value was 9.81) and H ≈ 8395 m at that g (legacy 8500; 8433 is what
        // the legacy 9.81 would have given). Kept as the record of the
        // deliberate one-time value change.
        let g = surface_gravity(3.986e14, 6.36e6);
        close(g, 9.854, 2e-3);
        close(scale_height(288.15, 0.028_964_4, g), 8395.0, 15.0);
        close(scale_height(288.15, 0.028_964_4, 9.81), 8433.0, 15.0);
    }

    #[test]
    fn fourth_root_matches_powf_to_ulp_scale_but_is_sqrt_based() {
        // sqrt(sqrt(x)) is the deterministic spelling; sanity that it agrees
        // with the mathematical fourth root on a known case.
        let t = equilibrium_temperature(1361.0, 0.0);
        close(t, (1361.0f64 / (4.0 * STEFAN_BOLTZMANN)).powf(0.25), 1e-9);
    }
}

//! Deterministic celestial-body generator (WI 762).
//!
//! Turns a `seed` (+ an [`Archetype`]) into a [`BodyAsset`] — the "generate some,
//! keep some" engine of the world-building aspect. Generation is a **pure,
//! deterministic** function: the same `(seed, archetype)` yields a bit-identical
//! body on every run and platform, so a body you like is reproducible forever
//! (and the same `surface.seed` will drive the procedural surface in WI 763).
//!
//! Randomness comes from a small self-contained **splitmix64** stream (integer
//! ops only) rather than an external RNG crate: gameplay randomness needs to be
//! deterministic and portable, not cryptographic, and this keeps the crate
//! dependency-free and the output reproducible across builds.
//!
//! A body is kept internally consistent: surface gravity `g` is drawn per
//! archetype and `mu = g · radius²`, with the medium's `gravity` set to the same
//! `g`, so orbits, weight, and ocean pressure all agree.

use crate::body_asset::{BodyAsset, Rotation, SurfaceRecipe};
use crate::body_derive;
use crate::fluid::FluidMedium;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// A family of body the generator can produce. Determines the medium (atmosphere
/// and/or ocean) and the size/gravity ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Archetype {
    /// An airless rocky body: no atmosphere, no ocean.
    Moon,
    /// A rocky planet: an atmosphere, no ocean.
    RockyPlanet,
    /// An ocean world: an atmosphere over a global ocean.
    OceanWorld,
}

impl Archetype {
    /// All archetypes, in a stable order (for a UI to cycle through).
    pub const ALL: [Archetype; 3] = [
        Archetype::Moon,
        Archetype::RockyPlanet,
        Archetype::OceanWorld,
    ];

    /// A short human label.
    pub fn label(self) -> &'static str {
        match self {
            Archetype::Moon => "Moon",
            Archetype::RockyPlanet => "Rocky Planet",
            Archetype::OceanWorld => "Ocean World",
        }
    }

    /// A filesystem/id-friendly slug.
    pub fn slug(self) -> &'static str {
        match self {
            Archetype::Moon => "moon",
            Archetype::RockyPlanet => "rocky",
            Archetype::OceanWorld => "ocean",
        }
    }

    /// Inverse of [`slug`](Self::slug), for persisted refs (WI 891). `None`
    /// for an unknown slug — persisted input is reported, never panicked on.
    pub fn from_slug(slug: &str) -> Option<Archetype> {
        Archetype::ALL.into_iter().find(|a| a.slug() == slug)
    }
}

// ---------------------------------------------------------------------------
// Hash-derived child seeds (WI 889).
//
// `child_seed = hash(parent_seed, domain_tag)` — the design's seeding contract
// (I1, the Elite/Qud discipline), replacing the old `seed ^ archetype-salt`
// single stream. Every drawn field gets its OWN single-purpose stream keyed by
// a stable tag (`"<archetype-slug>/<field>"`), so adding, removing, or
// reordering a generation step can never shift another field's draw, and the
// old struct-literal-source-order stream layout is retired outright.
//
// The tag strings are **output-contract** inputs: changing a tag (or the hash
// composition below) is itself a deliberate stream break requiring a
// `BODY_OUTPUT_VERSION` bump. Pure integer ops throughout — bit-identical on
// every platform and Rust release.
// ---------------------------------------------------------------------------

/// FNV-1a 64 offset basis (the same primitive `body_digest` uses).
const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
/// FNV-1a 64 prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// One FNV-1a update pass over `bytes` (byte-serial, so hashing concatenated
/// parts equals hashing the concatenation).
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// Folds the parent seed into a tag hash and finishes with the splitmix64
/// avalanche, so low bits are well mixed even for short tags and tiny seeds.
fn finish_child_seed(tag_hash: u64, parent_seed: u64) -> u64 {
    let mut z = tag_hash ^ parent_seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `child_seed = hash(parent_seed, domain_tag)`: FNV-1a 64 over the tag bytes,
/// folded with the parent seed, splitmix64-finished. Allocation-free, integer
/// only, no special-casing of any seed value (0 and `u64::MAX` included).
///
/// The design's seeding primitive (its literals are test-pinned); production
/// body draws go through [`field_seed`], which hashes the same composite tag
/// without allocating — future subsystems (rings, satellites, surface
/// generation steps) enter here with their own domain tags.
#[allow(dead_code)] // the contract primitive; exercised/pinned from tests
pub(crate) fn child_seed(parent_seed: u64, domain_tag: &str) -> u64 {
    finish_child_seed(fnv1a(FNV_OFFSET, domain_tag.as_bytes()), parent_seed)
}

/// The child seed for one drawn body field: the domain tag is
/// `"<archetype-slug>/<field>"` (hashed serially, no allocation). Equivalent to
/// [`child_seed`] over the concatenated tag — pinned by a test.
fn field_seed(parent_seed: u64, archetype: Archetype, field: &str) -> u64 {
    let h = fnv1a(FNV_OFFSET, archetype.slug().as_bytes());
    let h = fnv1a(h, b"/");
    let h = fnv1a(h, field.as_bytes());
    finish_child_seed(h, parent_seed)
}

/// A tiny deterministic value source (splitmix64). Integer-only, so its stream is
/// identical on every platform for a given seed.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64-bit value (splitmix64).
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[0, 1)` with 53 bits of resolution.
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// A value in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_unit() * (hi - lo)
    }
}

/// The per-archetype parameter **bands** — the `[min, max)` range of every
/// independently-drawn field (WI 883). The seeded [`sample`] draws each drawn
/// field uniformly from its band; the archetype's *structure* (which fields are
/// drawn vs literal, the draw order, the ocean-pressure coupling, the μ/gravity
/// derivations) stays in `sample` as the deterministic spine. Fields a shape
/// never draws are left at their [`Default`] (`(0.0, 0.0)`) and ignored.
///
/// The band **numbers** live in authored body **recipes** (WI 883/884): a
/// `BodyRecipe` carrying a `shape` supplies its bands as ladder-tunable fields,
/// and the canonical archetype bands ship in the embedded bodies pack
/// (`crates/sim/content/bodies.ron`) — the single source since WI 884 deleted
/// the in-code defaults. [`generate`] reads them back through
/// `content::canonical_bands`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ArchetypeBands {
    pub radius: (f64, f64),
    pub gravity: (f64, f64),
    pub sidereal_period: (f64, f64),
    pub atmosphere_surface_pressure: (f64, f64),
    // The drawn independent set (WI 889): the medium is DERIVED from these
    // via the `body_derive` relations, never drawn directly.
    pub nominal_insolation: (f64, f64),
    pub bond_albedo: (f64, f64),
    pub greenhouse_delta_t: (f64, f64),
    pub mean_molar_mass: (f64, f64),
    pub ocean_surface_density: (f64, f64),
    pub ocean_temperature: (f64, f64),
}

/// Samples a [`BodyAsset`] deterministically from `seed`, an `archetype`
/// (structure), and its parameter `bands` (numbers).
///
/// Pure: the same inputs always produce a bit-identical body. Every drawn field
/// is the **first draw of its own hash-derived stream** (WI 889 — see
/// [`child_seed`]), keyed `"<archetype-slug>/<field>"`, so the set of drawn
/// fields — not any draw *order* — defines the output. The body's `mu` is
/// derived as `g · radius²` and its medium's `gravity` set to the same `g`;
/// the ocean's surface pressure is continuous with the atmosphere.
/// Only `surface.seed` is set (to `seed`, **verbatim** — the persisted
/// `BodyRef` round-trip regenerates from it); detailed surface/render stay
/// reserved.
pub(crate) fn sample(seed: u64, archetype: Archetype, bands: &ArchetypeBands) -> BodyAsset {
    // One single-purpose stream per drawn field.
    let draw = |field: &str, band: (f64, f64)| -> f64 {
        Rng::new(field_seed(seed, archetype, field)).range(band.0, band.1)
    };

    // Size + surface gravity.
    let radius = draw("radius", bands.radius);
    let g = draw("gravity", bands.gravity);
    let mu = g * radius * radius;

    // Rotation about +Z.
    let rotation = Rotation {
        axis: DVec3::Z,
        sidereal_period: draw("sidereal_period", bands.sidereal_period),
    };

    // Medium: presence of atmosphere/ocean is the archetype's defining trait,
    // and since WI 889 the medium is **derived, never drawn** — the sampler
    // draws the independent set (insolation / albedo / greenhouse / molar
    // mass) and the same `body_derive` relations the fixed arm uses compute
    // temperature, density, and scale height (design I2: one physics).
    let t_surf_from = |draw: &dyn Fn(&str, (f64, f64)) -> f64, greenhouse: f64| {
        body_derive::surface_temperature(
            body_derive::equilibrium_temperature(
                draw("nominal_insolation", bands.nominal_insolation),
                draw("bond_albedo", bands.bond_albedo),
            ),
            greenhouse,
        )
    };
    let fluid_medium = match archetype {
        Archetype::Moon => {
            // Airless: the equilibrium temperature of the drawn independents
            // (no atmosphere ⇒ zero greenhouse), one vocabulary across shapes.
            let t_surf = t_surf_from(&draw, 0.0);
            FluidMedium {
                atmosphere_surface_density: 0.0,
                atmosphere_surface_pressure: 0.0,
                atmosphere_scale_height: 1.0, // positive placeholder; density is zero
                ocean_surface_density: 0.0,
                ocean_surface_pressure: 0.0,
                ocean_density_gradient: 0.0,
                gravity: g,
                atmosphere_temperature: t_surf,
                // No ocean; the inert value follows the surface ambient.
                ocean_temperature: t_surf,
            }
        }
        Archetype::RockyPlanet => {
            let surface_pressure = draw(
                "atmosphere_surface_pressure",
                bands.atmosphere_surface_pressure,
            );
            let molar_mass = draw("mean_molar_mass", bands.mean_molar_mass);
            let t_surf = t_surf_from(&draw, draw("greenhouse_delta_t", bands.greenhouse_delta_t));
            FluidMedium {
                atmosphere_surface_density: body_derive::atmosphere_surface_density(
                    surface_pressure,
                    molar_mass,
                    t_surf,
                ),
                atmosphere_surface_pressure: surface_pressure,
                atmosphere_scale_height: body_derive::scale_height(t_surf, molar_mass, g),
                ocean_surface_density: 0.0,
                ocean_surface_pressure: 0.0,
                ocean_density_gradient: 0.0,
                gravity: g,
                atmosphere_temperature: t_surf,
                // No ocean; the inert value follows the surface ambient.
                ocean_temperature: t_surf,
            }
        }
        Archetype::OceanWorld => {
            let surface_pressure = draw(
                "atmosphere_surface_pressure",
                bands.atmosphere_surface_pressure,
            );
            let molar_mass = draw("mean_molar_mass", bands.mean_molar_mass);
            let t_surf = t_surf_from(&draw, draw("greenhouse_delta_t", bands.greenhouse_delta_t));
            // Drawn ocean; pressure is continuous with the atmosphere at the
            // surface; gating (frozen/airless ⇒ no liquid) is the same shared
            // decision the fixed arm applies.
            let (ocean_surface_density, ocean_surface_pressure, ocean_density_gradient) =
                body_derive::gate_ocean(
                    t_surf,
                    surface_pressure,
                    draw("ocean_surface_density", bands.ocean_surface_density),
                    surface_pressure,
                    0.0,
                );
            FluidMedium {
                atmosphere_surface_density: body_derive::atmosphere_surface_density(
                    surface_pressure,
                    molar_mass,
                    t_surf,
                ),
                atmosphere_surface_pressure: surface_pressure,
                atmosphere_scale_height: body_derive::scale_height(t_surf, molar_mass, g),
                ocean_surface_density,
                ocean_surface_pressure,
                ocean_density_gradient,
                gravity: g,
                atmosphere_temperature: t_surf,
                ocean_temperature: draw("ocean_temperature", bands.ocean_temperature),
            }
        }
    };

    BodyAsset {
        id: format!("gen-{}-{:016x}", archetype.slug(), seed),
        name: format!("{} {:04X}", archetype.label(), (seed & 0xFFFF) as u16),
        mu,
        radius,
        rotation,
        fluid_medium,
        surface: SurfaceRecipe::from_seed(seed),
        render: serde_json::Value::Null,
    }
}

/// Generates a [`BodyAsset`] deterministically from `seed` and `archetype`.
///
/// Deterministic: the same inputs always produce a bit-identical body. Since
/// WI 884 the archetype's parameter bands come from the **shipped canonical
/// recipes** (the embedded bodies pack, via `content::canonical_bands`) — the
/// single authored source — and are fed to [`sample`], which owns the draw
/// structure. The historical output is pinned by the golden-stream test below
/// plus the content-side characterization against independent literal fixtures.
/// The body's `mu` is derived as `g · radius²` and its medium's `gravity` set to
/// the same `g`, so it is internally consistent. Detailed surface/render
/// parameters stay reserved (WI 763/764); only `surface.seed` is set (to `seed`).
pub fn generate(seed: u64, archetype: Archetype) -> BodyAsset {
    sample(seed, archetype, &crate::content::canonical_bands(archetype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_and_archetype_is_bit_identical() {
        for arch in Archetype::ALL {
            let a = generate(12345, arch);
            let b = generate(12345, arch);
            assert_eq!(a, b, "generation must be deterministic for {arch:?}");
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = generate(1, Archetype::RockyPlanet);
        let b = generate(2, Archetype::RockyPlanet);
        assert!(
            a.radius != b.radius || a.mu != b.mu,
            "distinct seeds → distinct bodies"
        );
    }

    #[test]
    fn same_seed_different_archetype_differs() {
        let m = generate(7, Archetype::Moon);
        let r = generate(7, Archetype::RockyPlanet);
        assert_ne!(m.id, r.id);
        assert!(m.radius != r.radius || m.mu != r.mu);
    }

    #[test]
    fn archetype_medium_invariants_hold() {
        for seed in [0u64, 1, 42, u64::MAX] {
            let moon = generate(seed, Archetype::Moon).fluid_medium;
            assert_eq!(moon.atmosphere_surface_density, 0.0);
            assert_eq!(moon.ocean_surface_density, 0.0);

            let rocky = generate(seed, Archetype::RockyPlanet).fluid_medium;
            assert!(rocky.atmosphere_surface_density > 0.0);
            assert_eq!(rocky.ocean_surface_density, 0.0);

            let ocean = generate(seed, Archetype::OceanWorld).fluid_medium;
            assert!(ocean.atmosphere_surface_density > 0.0);
            assert!(ocean.ocean_surface_density > 0.0);
        }
    }

    #[test]
    fn body_is_physically_coherent() {
        for arch in Archetype::ALL {
            for seed in [0u64, 3, 99, u64::MAX] {
                let a = generate(seed, arch);
                assert!(a.radius.is_finite() && a.radius > 0.0);
                assert!(a.mu.is_finite() && a.mu > 0.0);
                let g_from_mu = a.mu / (a.radius * a.radius);
                assert!(
                    (g_from_mu - a.fluid_medium.gravity).abs() <= 1e-9 * a.fluid_medium.gravity,
                    "mu/r^2 must equal the medium gravity"
                );
                assert!(a.surface.seed == seed);
            }
        }
    }

    #[test]
    fn generated_body_keeps_and_reloads_unchanged() {
        use crate::body_library::{load_body, save_body};
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("snd-gen-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let asset = generate(2024, Archetype::OceanWorld);
        let path = save_body(&dir, &asset).unwrap();
        let back = load_body(&path).unwrap();
        assert_eq!(back.id, asset.id);
        assert!((back.mu - asset.mu).abs() <= 1e-9 * asset.mu);
        assert!((back.radius - asset.radius).abs() <= 1e-9 * asset.radius);
        assert_eq!(back.fluid_medium, asset.fluid_medium);
        assert_eq!(back.surface.seed, asset.surface.seed);
        let _ = std::fs::remove_dir_all(&dir);
    }
    /// Golden pin of the draw stream (WI 884). Once `generate` and the recipe
    /// path share the shipped RON bands, they can no longer characterize each
    /// other against a stream change — this test is the independent oracle that
    /// keeps every previously kept/saved body reproducible. Values captured at
    /// seed 42; they must never change without a deliberate `output_version`
    /// decision. **Re-anchored at WI 889 (BODY_OUTPUT_VERSION 3)** — the
    /// batched deliberate stream break (hash-derived per-field seeds +
    /// sampled-path derivation); the pre-889 literals died with version 2.
    #[test]
    fn golden_stream_values_are_pinned() {
        let cases = [
            // (archetype, radius, mu, sidereal period, atmosphere temperature)
            (
                Archetype::Moon,
                1_048_595.223_693_523_3,
                2_703_287_108_081.333,
                52_367.689_175_934_574,
                279.727_426_022_034_5,
            ),
            (
                Archetype::RockyPlanet,
                4_465_240.707_701_108,
                204_615_912_304_141.84,
                190_985.025_035_038_65,
                291.192_474_025_837_56,
            ),
            (
                Archetype::OceanWorld,
                4_578_588.072_473_212,
                148_846_395_598_255.13,
                137_076.276_019_381_82,
                295.713_280_544_132_1,
            ),
        ];
        for (arch, radius, mu, period, atm_t) in cases {
            let b = generate(42, arch);
            assert_eq!(b.radius, radius, "{arch:?} radius drifted");
            assert_eq!(b.mu, mu, "{arch:?} mu drifted");
            assert_eq!(
                b.rotation.sidereal_period, period,
                "{arch:?} period drifted"
            );
            assert_eq!(
                b.fluid_medium.atmosphere_temperature, atm_t,
                "{arch:?} atmosphere temperature drifted"
            );
        }
    }

    // --- WI 889: hash-derived child seeds ---

    #[test]
    fn child_seed_is_deterministic_and_tag_and_parent_sensitive() {
        for parent in [0u64, 1, 42, u64::MAX] {
            assert_eq!(
                child_seed(parent, "moon/radius"),
                child_seed(parent, "moon/radius"),
                "deterministic"
            );
            assert_ne!(
                child_seed(parent, "moon/radius"),
                child_seed(parent, "moon/gravity"),
                "tag-sensitive at parent {parent}"
            );
        }
        assert_ne!(
            child_seed(1, "moon/radius"),
            child_seed(2, "moon/radius"),
            "parent-sensitive"
        );
    }

    /// Every (archetype, field) tag yields a distinct child seed at
    /// representative parents (incl. the golden extremes 0 and `u64::MAX`) —
    /// the per-field streams never alias.
    #[test]
    fn child_seeds_are_collision_distinct_across_the_tag_set() {
        let fields = [
            "radius",
            "gravity",
            "sidereal_period",
            "atmosphere_surface_pressure",
            "nominal_insolation",
            "bond_albedo",
            "greenhouse_delta_t",
            "mean_molar_mass",
            "ocean_surface_density",
            "ocean_temperature",
        ];
        for parent in [0u64, 42, u64::MAX] {
            let mut seen = std::collections::HashSet::new();
            for arch in Archetype::ALL {
                for field in fields {
                    assert!(
                        seen.insert(field_seed(parent, arch, field)),
                        "collision at parent {parent}, {arch:?}/{field}"
                    );
                }
            }
        }
    }

    /// `field_seed` is exactly `child_seed` over the documented concatenated
    /// tag `"<slug>/<field>"` — the serial FNV parts are not a second scheme.
    #[test]
    fn field_seed_equals_child_seed_over_the_concatenated_tag() {
        for arch in Archetype::ALL {
            let tag = format!("{}/radius", arch.slug());
            assert_eq!(field_seed(97, arch, "radius"), child_seed(97, &tag));
        }
    }

    /// Cross-release stability canary: the hash composition is an output
    /// contract (a changed literal here = a stream break = a deliberate
    /// `BODY_OUTPUT_VERSION` decision).
    #[test]
    fn child_seed_literals_are_pinned() {
        assert_eq!(child_seed(0, "moon/radius"), 0x7B6D_1EA0_083C_C967);
        assert_eq!(child_seed(42, "rocky/gravity"), 0xA096_87FD_E26D_55D6);
        assert_eq!(
            child_seed(u64::MAX, "ocean/ocean_temperature"),
            0x912E_81AA_0F0A_BE0F
        );
    }

    /// Draw isolation (the WI 889 point): each drawn field's value is the
    /// first draw of its OWN tagged stream, independently reconstructed here —
    /// so adding or removing any other draw cannot shift it.
    #[test]
    fn each_drawn_field_is_the_first_draw_of_its_own_stream() {
        let first = |seed: u64, arch: Archetype, field: &str, band: (f64, f64)| -> f64 {
            Rng::new(field_seed(seed, arch, field)).range(band.0, band.1)
        };
        for seed in [0u64, 7, 42, u64::MAX] {
            for arch in Archetype::ALL {
                let bands = crate::content::canonical_bands(arch);
                let b = generate(seed, arch);
                assert_eq!(b.radius, first(seed, arch, "radius", bands.radius));
                let g = first(seed, arch, "gravity", bands.gravity);
                assert_eq!(b.mu, g * b.radius * b.radius);
                assert_eq!(
                    b.rotation.sidereal_period,
                    first(seed, arch, "sidereal_period", bands.sidereal_period)
                );
                if arch != Archetype::Moon {
                    assert_eq!(
                        b.fluid_medium.atmosphere_surface_pressure,
                        first(
                            seed,
                            arch,
                            "atmosphere_surface_pressure",
                            bands.atmosphere_surface_pressure
                        )
                    );
                }
                if arch == Archetype::OceanWorld {
                    // The canonical ocean bands never trip the freeze gate
                    // (corner-asserted in content tests), so the drawn ocean
                    // values pass through and stay assertable here.
                    assert_eq!(
                        b.fluid_medium.ocean_surface_density,
                        first(
                            seed,
                            arch,
                            "ocean_surface_density",
                            bands.ocean_surface_density
                        )
                    );
                    assert_eq!(
                        b.fluid_medium.ocean_temperature,
                        first(seed, arch, "ocean_temperature", bands.ocean_temperature)
                    );
                }
            }
        }
    }

    /// Derivation coherence (WI 889, the workitem's AC 2): a sampled body's
    /// medium satisfies the `body_derive` relations **bit-exactly** against
    /// the drawn independents — the medium is derived, never drawn. This is
    /// the generator-filled content seam suppress (WI 880) will consume.
    /// Also pins `surface.seed == body seed` verbatim (the `BodyRef`
    /// round-trip invariant).
    #[test]
    fn sampled_medium_is_derived_not_drawn() {
        let first = |seed: u64, arch: Archetype, field: &str, band: (f64, f64)| -> f64 {
            Rng::new(field_seed(seed, arch, field)).range(band.0, band.1)
        };
        for seed in [0u64, 7, 42, 1234, u64::MAX] {
            for arch in Archetype::ALL {
                let bands = crate::content::canonical_bands(arch);
                let b = generate(seed, arch);
                let m = &b.fluid_medium;
                assert_eq!(b.surface.seed, seed, "surface seed = body seed, verbatim");

                let s = first(seed, arch, "nominal_insolation", bands.nominal_insolation);
                let a = first(seed, arch, "bond_albedo", bands.bond_albedo);
                let greenhouse = match arch {
                    Archetype::Moon => 0.0, // airless: no greenhouse draw
                    _ => first(seed, arch, "greenhouse_delta_t", bands.greenhouse_delta_t),
                };
                let t_surf = body_derive::surface_temperature(
                    body_derive::equilibrium_temperature(s, a),
                    greenhouse,
                );
                assert_eq!(
                    m.atmosphere_temperature, t_surf,
                    "{arch:?}@{seed}: T_surf is the derived value"
                );

                match arch {
                    Archetype::Moon => {
                        assert_eq!(m.atmosphere_surface_density, 0.0);
                        assert_eq!(m.atmosphere_scale_height, 1.0);
                        assert_eq!(m.ocean_temperature, t_surf);
                    }
                    _ => {
                        let p = first(
                            seed,
                            arch,
                            "atmosphere_surface_pressure",
                            bands.atmosphere_surface_pressure,
                        );
                        let mm = first(seed, arch, "mean_molar_mass", bands.mean_molar_mass);
                        let g = first(seed, arch, "gravity", bands.gravity);
                        assert_eq!(
                            m.atmosphere_surface_density,
                            body_derive::atmosphere_surface_density(p, mm, t_surf),
                            "{arch:?}@{seed}: density is the ideal-gas relation"
                        );
                        assert_eq!(
                            m.atmosphere_scale_height,
                            body_derive::scale_height(t_surf, mm, g),
                            "{arch:?}@{seed}: scale height is the hydrostatic relation"
                        );
                    }
                }
                if arch == Archetype::OceanWorld && m.ocean_surface_density > 0.0 {
                    assert_eq!(
                        m.ocean_surface_pressure, m.atmosphere_surface_pressure,
                        "ocean pressure continuous with the atmosphere"
                    );
                }
            }
        }
    }
}

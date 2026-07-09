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

    /// A per-archetype salt so the same seed yields distinct bodies per archetype.
    fn salt(self) -> u64 {
        match self {
            Archetype::Moon => 0x1111_1111_1111_1111,
            Archetype::RockyPlanet => 0x2222_2222_2222_2222,
            Archetype::OceanWorld => 0x3333_3333_3333_3333,
        }
    }
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
/// These bands are the numbers that move into the authored body **recipes**
/// (WI 883): a `BodyRecipe` carrying a `shape` supplies its bands as
/// ladder-tunable fields, and content resolution samples them exactly as
/// [`generate`] samples [`default_bands`]. [`generate`]'s in-code defaults are
/// the transitional copy of the bands; WI 884 collapses the two onto the RON
/// recipes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ArchetypeBands {
    pub radius: (f64, f64),
    pub gravity: (f64, f64),
    pub sidereal_period: (f64, f64),
    pub atmosphere_surface_pressure: (f64, f64),
    pub atmosphere_surface_density: (f64, f64),
    pub atmosphere_scale_height: (f64, f64),
    pub atmosphere_temperature: (f64, f64),
    pub ocean_surface_density: (f64, f64),
    pub ocean_temperature: (f64, f64),
}

/// The canonical bands for an archetype — the ranges the generator has always
/// used, now named. `sample(seed, arch, &default_bands(arch))` reproduces the
/// historical [`generate`] output exactly.
pub(crate) fn default_bands(archetype: Archetype) -> ArchetypeBands {
    // A sidereal period drawn from one broad range for every archetype.
    let sidereal_period = (20_000.0, 200_000.0);
    match archetype {
        Archetype::Moon => ArchetypeBands {
            radius: (2.0e5, 2.0e6),
            gravity: (0.5, 3.0),
            sidereal_period,
            ..ArchetypeBands::default()
        },
        Archetype::RockyPlanet => ArchetypeBands {
            radius: (2.5e6, 8.0e6),
            gravity: (3.0, 12.0),
            sidereal_period,
            atmosphere_surface_pressure: (50_000.0, 150_000.0),
            atmosphere_surface_density: (0.2, 2.0),
            atmosphere_scale_height: (5_000.0, 12_000.0),
            atmosphere_temperature: (220.0, 300.0),
            ..ArchetypeBands::default()
        },
        Archetype::OceanWorld => ArchetypeBands {
            radius: (3.0e6, 9.0e6),
            gravity: (5.0, 12.0),
            sidereal_period,
            atmosphere_surface_pressure: (80_000.0, 180_000.0),
            atmosphere_surface_density: (0.5, 2.5),
            atmosphere_scale_height: (6_000.0, 12_000.0),
            atmosphere_temperature: (250.0, 300.0),
            ocean_surface_density: (950.0, 1_100.0),
            ocean_temperature: (275.0, 300.0),
        },
    }
}

/// Samples a [`BodyAsset`] deterministically from `seed`, an `archetype`
/// (structure), and its parameter `bands` (numbers).
///
/// Pure: the same inputs always produce a bit-identical body. The **draw order**
/// mirrors the archetype's field-evaluation order exactly (radius, gravity,
/// sidereal period, then the shape's medium draws), so sourcing the numbers from
/// `bands` rather than inline literals changes nothing about the stream. The
/// body's `mu` is derived as `g · radius²` and its medium's `gravity` set to the
/// same `g`; the ocean's surface pressure is continuous with the atmosphere.
/// Only `surface.seed` is set (to `seed`); detailed surface/render stay reserved.
pub(crate) fn sample(seed: u64, archetype: Archetype, bands: &ArchetypeBands) -> BodyAsset {
    let mut rng = Rng::new(seed ^ archetype.salt());

    // Size + surface gravity, drawn in a fixed order so the stream is stable.
    let radius = rng.range(bands.radius.0, bands.radius.1);
    let g = rng.range(bands.gravity.0, bands.gravity.1);
    let mu = g * radius * radius;

    // Rotation about +Z.
    let sidereal_period = rng.range(bands.sidereal_period.0, bands.sidereal_period.1);
    let rotation = Rotation {
        axis: DVec3::Z,
        sidereal_period,
    };

    // Medium: presence of atmosphere/ocean is the archetype's defining trait.
    let fluid_medium = match archetype {
        Archetype::Moon => FluidMedium {
            atmosphere_surface_density: 0.0,
            atmosphere_surface_pressure: 0.0,
            atmosphere_scale_height: 1.0, // positive placeholder; density is zero
            ocean_surface_density: 0.0,
            ocean_surface_pressure: 0.0,
            ocean_density_gradient: 0.0,
            gravity: g,
            atmosphere_temperature: 200.0,
            ocean_temperature: 200.0,
        },
        Archetype::RockyPlanet => {
            let surface_pressure = rng.range(
                bands.atmosphere_surface_pressure.0,
                bands.atmosphere_surface_pressure.1,
            );
            FluidMedium {
                atmosphere_surface_density: rng.range(
                    bands.atmosphere_surface_density.0,
                    bands.atmosphere_surface_density.1,
                ),
                atmosphere_surface_pressure: surface_pressure,
                atmosphere_scale_height: rng.range(
                    bands.atmosphere_scale_height.0,
                    bands.atmosphere_scale_height.1,
                ),
                ocean_surface_density: 0.0,
                ocean_surface_pressure: 0.0,
                ocean_density_gradient: 0.0,
                gravity: g,
                atmosphere_temperature: rng.range(
                    bands.atmosphere_temperature.0,
                    bands.atmosphere_temperature.1,
                ),
                ocean_temperature: 280.0,
            }
        }
        Archetype::OceanWorld => {
            let surface_pressure = rng.range(
                bands.atmosphere_surface_pressure.0,
                bands.atmosphere_surface_pressure.1,
            );
            FluidMedium {
                atmosphere_surface_density: rng.range(
                    bands.atmosphere_surface_density.0,
                    bands.atmosphere_surface_density.1,
                ),
                atmosphere_surface_pressure: surface_pressure,
                atmosphere_scale_height: rng.range(
                    bands.atmosphere_scale_height.0,
                    bands.atmosphere_scale_height.1,
                ),
                ocean_surface_density: rng
                    .range(bands.ocean_surface_density.0, bands.ocean_surface_density.1),
                // Continuous with the atmosphere at the surface.
                ocean_surface_pressure: surface_pressure,
                ocean_density_gradient: 0.0,
                gravity: g,
                atmosphere_temperature: rng.range(
                    bands.atmosphere_temperature.0,
                    bands.atmosphere_temperature.1,
                ),
                ocean_temperature: rng.range(bands.ocean_temperature.0, bands.ocean_temperature.1),
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
/// Pure: the same inputs always produce a bit-identical body. Delegates to
/// [`sample`] with the archetype's [`default_bands`], so the single sampler owns
/// the draw structure and this function's output is unchanged by construction.
/// The body's `mu` is derived as `g · radius²` and its medium's `gravity` set to
/// the same `g`, so it is internally consistent. Detailed surface/render
/// parameters stay reserved (WI 763/764); only `surface.seed` is set (to `seed`).
pub fn generate(seed: u64, archetype: Archetype) -> BodyAsset {
    sample(seed, archetype, &default_bands(archetype))
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
}

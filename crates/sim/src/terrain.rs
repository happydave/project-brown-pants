//! Analytic procedural terrain (WI 506).
//!
//! A deterministic f64 height function over the local tangent plane (XZ), with
//! surface normal (from the analytic gradient) and surface material (WI 497).
//!
//! **This function is the collision surface.** The wheel contact model queries it
//! in f64 world coordinates — never the rendered, LOD-tessellated mesh — so the
//! contact point is a deterministic function of world position and cannot "pop"
//! under level-of-detail switching or floating-origin rebasing. That is the
//! design's prescribed fix for the "kraken"; the rendered mesh is merely a
//! tessellation of this same function.

use crate::surface::SurfaceMaterial;
use glam::DVec3;
use std::f64::consts::TAU;

/// An analytic terrain: gentle sinusoidal bumps over a flat tangent plane, with a
/// uniform surface material. Height is measured along world +Y.
#[derive(Clone, Copy, Debug)]
pub struct Terrain {
    /// Bump amplitude (metres).
    pub amplitude: f64,
    /// Bump wavelength (metres).
    pub wavelength: f64,
    /// Surface material (friction / rolling resistance), WI 497.
    pub material: SurfaceMaterial,
}

impl Default for Terrain {
    fn default() -> Self {
        Self {
            amplitude: 1.5,
            wavelength: 24.0,
            material: SurfaceMaterial::REGOLITH,
        }
    }
}

impl Terrain {
    /// Surface height (world Y) at horizontal world position `(x, z)`.
    pub fn height(&self, x: f64, z: f64) -> f64 {
        let k = TAU / self.wavelength;
        0.5 * self.amplitude * ((k * x).sin() + (k * z).cos())
    }

    /// Outward unit surface normal at `(x, z)`, from the analytic height gradient.
    pub fn normal(&self, x: f64, z: f64) -> DVec3 {
        let k = TAU / self.wavelength;
        let dhdx = 0.5 * self.amplitude * k * (k * x).cos();
        let dhdz = -0.5 * self.amplitude * k * (k * z).sin();
        DVec3::new(-dhdx, 1.0, -dhdz).normalize()
    }

    /// Surface material at `(x, z)` (uniform for this toy).
    pub fn material_at(&self, _x: f64, _z: f64) -> SurfaceMaterial {
        self.material
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn height_is_deterministic_and_finite_at_planetary_offset() {
        let t = Terrain::default();
        // The same world position always yields the same height (pure f64),
        // even at a large absolute offset where f32 would lose precision.
        let x = 6_378_000.5;
        let z = -1_234_567.25;
        assert_eq!(t.height(x, z), t.height(x, z));
        assert!(t.height(x, z).is_finite());
    }

    #[test]
    fn normal_is_unit_and_points_up() {
        let t = Terrain::default();
        for (x, z) in [(0.0, 0.0), (3.7, -8.2), (1_000.0, 500.0)] {
            let n = t.normal(x, z);
            assert!((n.length() - 1.0).abs() < 1e-12);
            assert!(n.y > 0.0, "surface normal must point outward (up)");
        }
    }

    #[test]
    fn flat_terrain_is_planar() {
        let t = Terrain {
            amplitude: 0.0,
            ..Default::default()
        };
        assert_eq!(t.height(10.0, -20.0), 0.0);
        assert_eq!(t.normal(10.0, -20.0), DVec3::Y);
    }
}

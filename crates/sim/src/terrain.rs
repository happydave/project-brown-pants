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

/// A driveable wedge ramp on the terrain (WI 630 test affordance): a planar incline rising along +Z
/// within a rectangular footprint, ending in a lip (the surface drops back to the base terrain beyond
/// the run) so a rover drives up and **launches off the top** — for testing air time and tumbling.
#[derive(Clone, Copy, Debug)]
pub struct Ramp {
    /// Footprint centre on the X axis (m).
    pub center_x: f64,
    /// Footprint half-width on the X axis (m).
    pub half_width: f64,
    /// Where the incline begins on the Z axis (m).
    pub start_z: f64,
    /// Incline run along Z (m); the lip (peak) is at `start_z + run`.
    pub run: f64,
    /// Incline angle (radians); peak height is `run · tan(angle)`.
    pub angle: f64,
}

impl Ramp {
    /// Whether `(x, z)` is on the inclined part of the ramp (inside the footprint, before the lip).
    fn on_incline(&self, x: f64, z: f64) -> bool {
        (x - self.center_x).abs() <= self.half_width
            && z > self.start_z
            && z < self.start_z + self.run
    }

    /// The ramp's height contribution at `(x, z)` (added to the base terrain): the incline rises to the
    /// lip, then drops to zero beyond it (the launch edge).
    fn height(&self, x: f64, z: f64) -> f64 {
        if self.on_incline(x, z) {
            (z - self.start_z) * self.angle.tan()
        } else {
            0.0
        }
    }

    /// The ramp's `dh/dz` contribution (the incline slope; zero off the incline).
    fn dhdz(&self, x: f64, z: f64) -> f64 {
        if self.on_incline(x, z) {
            self.angle.tan()
        } else {
            0.0
        }
    }
}

/// An analytic terrain: gentle sinusoidal bumps over a flat tangent plane, with a
/// uniform surface material, and an optional driveable [`Ramp`]. Height is measured along world +Y.
#[derive(Clone, Copy, Debug)]
pub struct Terrain {
    /// Bump amplitude (metres).
    pub amplitude: f64,
    /// Bump wavelength (metres).
    pub wavelength: f64,
    /// Surface material (friction / rolling resistance), WI 497.
    pub material: SurfaceMaterial,
    /// An optional wedge ramp to drive off (WI 630 test affordance).
    pub ramp: Option<Ramp>,
}

impl Default for Terrain {
    fn default() -> Self {
        Self {
            amplitude: 1.5,
            wavelength: 24.0,
            material: SurfaceMaterial::REGOLITH,
            ramp: None,
        }
    }
}

impl Terrain {
    /// Surface height (world Y) at horizontal world position `(x, z)`: the sinusoidal base plus any
    /// ramp contribution.
    pub fn height(&self, x: f64, z: f64) -> f64 {
        let k = TAU / self.wavelength;
        let base = 0.5 * self.amplitude * ((k * x).sin() + (k * z).cos());
        base + self.ramp.map_or(0.0, |r| r.height(x, z))
    }

    /// Outward unit surface normal at `(x, z)`, from the analytic height gradient (base + ramp slope).
    pub fn normal(&self, x: f64, z: f64) -> DVec3 {
        let k = TAU / self.wavelength;
        let dhdx = 0.5 * self.amplitude * k * (k * x).cos();
        let dhdz =
            -0.5 * self.amplitude * k * (k * z).sin() + self.ramp.map_or(0.0, |r| r.dhdz(x, z));
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
    fn ramp_rises_to_the_lip_then_drops_for_launch() {
        // A 30° wedge on flat terrain: zero before the start, rising on the incline, and back to base
        // beyond the lip (the launch edge) and outside the footprint (WI 630 test ramp).
        let angle = 30.0_f64.to_radians();
        let t = Terrain {
            amplitude: 0.0,
            ramp: Some(Ramp {
                center_x: 0.0,
                half_width: 2.0,
                start_z: 5.0,
                run: 3.0,
                angle,
            }),
            ..Default::default()
        };
        assert_eq!(t.height(0.0, 4.9), 0.0, "before the ramp");
        let mid = t.height(0.0, 6.5); // 1.5 m up the incline
        assert!((mid - 1.5 * angle.tan()).abs() < 1e-9, "incline height");
        assert_eq!(
            t.height(0.0, 8.1),
            0.0,
            "past the lip → launch (drops to base)"
        );
        assert_eq!(t.height(3.0, 6.5), 0.0, "outside the footprint");
        // The incline normal tilts back (negative Z component), opposing the climb.
        let n = t.normal(0.0, 6.5);
        assert!(n.z < 0.0 && n.y > 0.0 && (n.length() - 1.0).abs() < 1e-12);
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

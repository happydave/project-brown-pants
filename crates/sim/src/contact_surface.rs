//! The contact-surface abstraction and the spherical adapter (WI 765).
//!
//! Sounding's contact thesis is that **physics queries an analytic surface**, not a
//! rendered mesh (design R1). The rover model ([`crate::rover`]) has always done
//! this against the flat [`Terrain`](crate::terrain::Terrain), working in a local
//! **+Y-up** frame over three queries: `height(x, z)`, `normal(x, z)`,
//! `material_at(x, z)`. This module lifts those three queries into a
//! [`ContactSurface`] trait so the rover can contact **any** analytic surface — and
//! adds [`SurfacePatch`], which presents the spherical procedural
//! [`SurfaceField`](crate::surface_field::SurfaceField) (WI 763) as a local
//! tangent plane at a landing site.
//!
//! The rebind is a **surface-source swap, not a new force law**: `Terrain` still
//! implements the trait with identical numbers, so the entire rover force model and
//! its stability suite are unchanged. Because the field is a *total pure function*
//! of position, a contact query always returns a surface — the rover can never meet
//! "ungenerated" ground, and contact is independent of the renderer's streaming/LOD
//! state (the WI 764 mesh is render-only).

use crate::surface::SurfaceMaterial;
use crate::surface_field::SurfaceField;
use crate::terrain::Terrain;
use glam::DVec3;

/// An analytic surface the rover contact model queries in its local **+Y-up**
/// frame: `height` is the ground's local Y under a horizontal `(x, z)`, `normal`
/// the outward unit surface normal there, and `material_at` the surface material.
pub trait ContactSurface {
    /// Ground height (local +Y) under horizontal local position `(x, z)`.
    fn height(&self, x: f64, z: f64) -> f64;
    /// Outward unit surface normal at `(x, z)`.
    fn normal(&self, x: f64, z: f64) -> DVec3;
    /// Surface material at `(x, z)`.
    fn material_at(&self, x: f64, z: f64) -> SurfaceMaterial;
}

impl ContactSurface for Terrain {
    fn height(&self, x: f64, z: f64) -> f64 {
        Terrain::height(self, x, z)
    }
    fn normal(&self, x: f64, z: f64) -> DVec3 {
        Terrain::normal(self, x, z)
    }
    fn material_at(&self, x: f64, z: f64) -> SurfaceMaterial {
        Terrain::material_at(self, x, z)
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

/// Presents a spherical [`SurfaceField`] as a local tangent-plane
/// [`ContactSurface`] at a landing site — the bridge that lets the flat-frame rover
/// contact model drive on a planet.
///
/// The local frame at landing direction `up0` (a unit vector from the body centre):
/// local **+Y** = `up0` (radial up), local **+X** = an `east` tangent, local **+Z**
/// = a `north` tangent, origin `O = up0·radius` (sea level under the landing site).
/// A local horizontal `(x, z)` maps to the sphere direction
/// `normalize(O + east·x + north·z)`; the field is evaluated there and the surface
/// point's displacement along `up0` is the local height (so the ground **curves
/// away** with distance from the origin — physically correct over the patch).
#[derive(Clone, Copy, Debug)]
pub struct SurfacePatch {
    field: SurfaceField,
    up: DVec3,
    east: DVec3,
    north: DVec3,
    radius: f64,
}

impl SurfacePatch {
    /// A patch on `field` with local up along `up0` (need not be normalized).
    pub fn new(field: SurfaceField, up0: DVec3) -> Self {
        let up = normalize_or_x(up0);
        // A **right-handed** orthonormal tangent basis at `up` (stable away from the
        // chosen seed axis): `east × up = north`, so at `up = +Y` the frame is
        // `east = +X, up = +Y, north = +Z` and `local_to_world` is the identity (no
        // reflection) — which keeps a rover rendered in the patch frame un-mirrored.
        let seed = if up.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let east = (seed - up * seed.dot(up)).normalize();
        let north = east.cross(up);
        let radius = field.radius();
        Self {
            field,
            up,
            east,
            north,
            radius,
        }
    }

    /// Local up (radial) direction.
    pub fn up(&self) -> DVec3 {
        self.up
    }

    /// The local-frame origin in **body-centred** coordinates (`up·radius`).
    pub fn origin(&self) -> DVec3 {
        self.up * self.radius
    }

    /// The sphere direction under local horizontal `(x, z)`.
    fn dir(&self, x: f64, z: f64) -> DVec3 {
        normalize_or_x(self.up * self.radius + self.east * x + self.north * z)
    }

    /// Map a local-frame point (`x`=east, `y`=up, `z`=north) to **body-centred**
    /// world coordinates — for placing/rendering a rover that lives in the patch.
    pub fn local_to_world(&self, local: DVec3) -> DVec3 {
        self.origin() + self.east * local.x + self.up * local.y + self.north * local.z
    }

    /// Map a body-centred world point into the local frame (the inverse of
    /// [`local_to_world`](Self::local_to_world)).
    pub fn world_to_local(&self, world: DVec3) -> DVec3 {
        let rel = world - self.origin();
        DVec3::new(rel.dot(self.east), rel.dot(self.up), rel.dot(self.north))
    }
}

impl ContactSurface for SurfacePatch {
    fn height(&self, x: f64, z: f64) -> f64 {
        let dir = self.dir(x, z);
        let surface = dir * (self.radius + self.field.elevation(dir));
        (surface - self.origin()).dot(self.up)
    }

    fn normal(&self, x: f64, z: f64) -> DVec3 {
        let n = self.field.normal(self.dir(x, z));
        // Express the world (radial-ish) normal in the local tangent frame.
        normalize_or_x(DVec3::new(
            n.dot(self.east),
            n.dot(self.up),
            n.dot(self.north),
        ))
    }

    fn material_at(&self, x: f64, z: f64) -> SurfaceMaterial {
        self.field.material(self.dir(x, z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_delegation_is_exact() {
        let t = Terrain::default();
        for (x, z) in [(0.0, 0.0), (3.7, -8.2), (120.0, 55.0)] {
            assert_eq!(ContactSurface::height(&t, x, z), t.height(x, z));
            assert_eq!(ContactSurface::normal(&t, x, z), t.normal(x, z));
            assert_eq!(
                ContactSurface::material_at(&t, x, z).friction,
                t.material_at(x, z).friction
            );
        }
    }

    #[test]
    fn patch_agrees_with_the_field_at_the_origin() {
        let field = SurfaceField::new(4242, 800_000.0);
        let up0 = DVec3::new(0.3, 0.8, -0.5);
        let patch = SurfacePatch::new(field, up0);
        let up = patch.up();
        // Height at the origin equals the field elevation along `up`.
        assert!((patch.height(0.0, 0.0) - field.elevation(up)).abs() < 1e-6);
        // Material at the origin equals the field material along `up` (within
        // float tolerance: the patch renormalizes `up · radius`, which can move
        // the direction by an ulp, and the WI 868 blended material is
        // value-continuous rather than piecewise-constant, so bit-equality no
        // longer holds by construction).
        assert!((patch.material_at(0.0, 0.0).friction - field.material(up).friction).abs() < 1e-9);
        // Local normal at the origin is ~+Y (radial), tilted only by local slope.
        let n = patch.normal(0.0, 0.0);
        assert!(n.y > 0.9, "origin normal should be near local up: {n:?}");
        assert!((n.length() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn patch_curves_away_from_the_origin() {
        // The ground drops away with horizontal distance d — the sphere curving
        // under the tangent plane (a flat-plane patch would show no such drop).
        // Asserted in the regime where curvature *provably* dominates terrain:
        // with the exact spherical drop D(d) = R·(1 − 1/√(1+(d/R)²)) and
        // |elevation| ≤ relief_bound everywhere, observed ≥ D − 2·relief_bound
        // holds for any relief whatsoever, and the in-test regime guard
        // D > 4·relief_bound makes that bound strictly beyond anything a flat
        // patch could show (≤ 2·relief_bound). (The original small-d fixture rode
        // on seed-0 luck — 1 km of curvature on this body is 0.5 m, far below
        // ordinary relief slopes — and died when WI 782 reshaped the terrain.
        // The small-angle d²/2R form overstates D by ~10% at these offsets, so
        // the exact form is used.)
        let field = SurfaceField::new(0, 1_000_000.0);
        let patch = SurfacePatch::new(field, DVec3::Y);
        let r = field.radius();
        let rb = field.relief_bound();
        let d_regime = 1.25 * (2.0 * r * 4.0 * rb).sqrt();
        for d in [d_regime, 2.0 * d_regime] {
            let exact_drop = r * (1.0 - 1.0 / (1.0 + (d / r).powi(2)).sqrt());
            assert!(
                exact_drop > 4.0 * rb,
                "d={d} is below the provable regime (drop {exact_drop} ≤ 4·{rb})"
            );
            let observed_drop = patch.height(0.0, 0.0) - patch.height(d, 0.0);
            assert!(
                observed_drop > exact_drop - 2.0 * rb,
                "d={d}: expected ≥ {} drop, got {observed_drop}",
                exact_drop - 2.0 * rb
            );
        }
    }

    #[test]
    fn local_world_round_trips() {
        let field = SurfaceField::new(9, 500_000.0);
        let patch = SurfacePatch::new(field, DVec3::new(-0.2, 0.9, 0.3));
        for p in [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(120.0, 3.0, -45.0),
            DVec3::new(-800.0, -12.0, 600.0),
        ] {
            let back = patch.world_to_local(patch.local_to_world(p));
            assert!(
                (back - p).length() < 1e-6,
                "round-trip failed: {p:?} → {back:?}"
            );
        }
    }

    #[test]
    fn all_queries_finite_over_a_patch() {
        let field = SurfaceField::new(7, 300_000.0);
        let patch = SurfacePatch::new(field, DVec3::new(0.5, 0.5, 0.5));
        for x in [-2_000.0, 0.0, 2_000.0] {
            for z in [-2_000.0, 0.0, 2_000.0] {
                assert!(patch.height(x, z).is_finite());
                let n = patch.normal(x, z);
                assert!(n.is_finite() && (n.length() - 1.0).abs() < 1e-9);
                assert!(patch.material_at(x, z).friction.is_finite());
            }
        }
    }
}

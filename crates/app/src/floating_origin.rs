//! Floating-origin rebasing for planetary-scale rendering (WI 504).
//!
//! Authoritative positions are f64 (WI 497 [`WorldPos`]); the GPU works in f32,
//! which cannot represent a planet radius (~6.4×10⁶ m) and centimetre detail at
//! the same time. Each frame the world is translated around a near-origin anchor
//! and each entity's f32 [`Transform`] is derived relative to it, so the focus
//! keeps full precision regardless of absolute world position.
//!
//! Bevy's atmosphere treats +Y as up with sea level at world Y = 0 and derives
//! viewing altitude from the camera's Y. So the anchor rebases the **horizontal
//! plane (X, Z) only** and leaves Y as the true altitude — the camera is pinned
//! to the render origin in X/Z while the world moves around it.

use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::transform::TransformSystems;
use sounding_sim::frame::WorldPos;

/// The f64 world placement of a rendered entity (metres), wrapping WI 497's
/// [`WorldPos`]. Its f32 [`Transform`] translation is derived each frame by
/// rebasing relative to the [`FloatingOrigin`].
#[derive(Component, Clone, Copy, Debug)]
pub struct WorldPlacement(pub WorldPos);

/// Marks the entity (the camera) whose horizontal position the floating origin
/// tracks, keeping it pinned to the render origin in X/Z.
#[derive(Component, Clone, Copy, Debug)]
pub struct AnchorCamera;

/// The floating-origin anchor: the f64 world point mapped to the render origin.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct FloatingOrigin(pub DVec3);

/// Pure rebasing: the f32 render translation of a world point relative to the
/// anchor. Computed independently per entity, so a distant entity never degrades
/// a near one's precision. This is the precision-preserving core of the toy.
pub fn render_translation(world: DVec3, anchor: DVec3) -> Vec3 {
    (world - anchor).as_vec3()
}

/// Registers the floating-origin systems: anchor tracking then rebasing, ahead of
/// Bevy's transform propagation.
pub struct FloatingOriginPlugin;

impl Plugin for FloatingOriginPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FloatingOrigin>().add_systems(
            PostUpdate,
            (track_anchor, rebase)
                .chain()
                .before(TransformSystems::Propagate),
        );
    }
}

/// Tracks the anchor to the camera's horizontal position (X, Z); Y stays at sea
/// level (0) so the atmosphere's altitude (the camera's render-space Y) stays
/// correct.
fn track_anchor(
    mut origin: ResMut<FloatingOrigin>,
    camera: Query<&WorldPlacement, With<AnchorCamera>>,
) {
    if let Ok(placement) = camera.single() {
        let p = placement.0.pos;
        origin.0 = DVec3::new(p.x, 0.0, p.z);
    }
}

/// Derives each placed entity's f32 [`Transform`] translation from its f64 world
/// position relative to the anchor. Rotation is left untouched (camera aim and
/// sun orientation are owned elsewhere).
fn rebase(origin: Res<FloatingOrigin>, mut placed: Query<(&WorldPlacement, &mut Transform)>) {
    for (placement, mut tf) in &mut placed {
        tf.translation = render_translation(placement.0.pos, origin.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebasing_is_the_translation_difference() {
        let world = DVec3::new(6_378_000.0, 1.5, -2.0);
        let anchor = DVec3::new(6_378_000.0, 0.0, 0.0);
        let r = render_translation(world, anchor);
        assert!((r - Vec3::new(0.0, 1.5, -2.0)).length() < 1e-3);
    }

    #[test]
    fn floating_origin_recovers_precision_naive_f32_loses() {
        // A craft 1 cm from the anchor, the anchor at planetary distance.
        let anchor = DVec3::new(6_360_000.0, 0.0, 0.0);
        let world = anchor + DVec3::new(0.01, 0.0, 0.0);
        // Naive: casting the absolute world position to f32 loses the centimetre.
        assert_eq!(world.as_vec3().x, 6_360_000.0_f32);
        // Floating origin preserves it (sub-millimetre) and keeps it near origin.
        let r = render_translation(world, anchor);
        assert!((r.x - 0.01).abs() < 1e-4);
        assert!(r.length() < 1.0);
    }

    #[test]
    fn distant_entity_does_not_degrade_a_near_one() {
        let anchor = DVec3::ZERO;
        let near = render_translation(DVec3::new(2.0, 0.5, 0.0), anchor);
        let far = render_translation(DVec3::new(6.36e6, 0.0, 0.0), anchor);
        // The near entity is exact regardless of the far entity's magnitude.
        assert!((near - Vec3::new(2.0, 0.5, 0.0)).length() < 1e-4);
        assert!(far.x > 1.0e6);
    }
}

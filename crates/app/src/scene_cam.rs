//! Shared **follow orbit camera** for craft-tracking scenes (WI 714).
//!
//! The dive (`dive_scene`) and harbor (`harbor_scene`) both orbit a floating-origin camera around a
//! tracked craft/hull — middle-drag orbits, the wheel zooms, and the eye tracks the target every
//! frame. They had near-identical copies; this module owns one implementation. Per-scene differences
//! (orbit direction, pitch/zoom limits, the tracked entity) are **config on [`OrbitFollowCam`]** plus
//! the [`CameraTarget`] marker, so each scene reproduces its exact feel. Distinct from the *build*
//! orbit camera (`editor::OrbitCam`), which free-orbits the CoM rather than following a craft.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use sounding_sim::frame::{FrameId, WorldPos};

use crate::floating_origin::{AnchorCamera, WorldPlacement};

/// Marks the entity the follow camera tracks (the craft, the hull). Each scene tags its target.
#[derive(Component)]
pub struct CameraTarget;

/// Mouse-driven orbit/zoom state for a follow camera: the eye sits at
/// `target + orbit_offset(yaw, pitch, dist)`. Carries the per-scene feel (orbit direction, limits).
#[derive(Resource)]
pub struct OrbitFollowCam {
    /// Orbit yaw about the target (radians).
    pub yaw: f32,
    /// Orbit pitch above the horizon (radians), clamped away from the poles.
    pub pitch: f32,
    /// Eye distance from the target (metres).
    pub dist: f32,
    /// Orbit direction for horizontal drag (+1 or −1).
    pub yaw_sign: f32,
    /// Pitch magnitude clamp (radians).
    pub pitch_limit: f32,
    /// Zoom distance clamp (metres).
    pub dist_min: f32,
    pub dist_max: f32,
}

/// The camera eye offset from the orbit target for a yaw/pitch/distance — the spherical-to-cartesian
/// the gallery/editor orbit cameras use. Pure (unit-tested).
pub fn orbit_offset(yaw: f32, pitch: f32, dist: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(sy * cp, sp, cy * cp) * dist
}

/// Reads mouse input into the orbit camera state: middle-drag orbits (yaw/pitch), the wheel zooms
/// (distance) — the editor/gallery convention, leaving left/right free.
pub fn orbit_follow_input(
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut cam: ResMut<OrbitFollowCam>,
) {
    if buttons.pressed(MouseButton::Middle) {
        cam.yaw += cam.yaw_sign * motion.delta.x * 0.01;
        let lim = cam.pitch_limit;
        cam.pitch = (cam.pitch + motion.delta.y * 0.01).clamp(-lim, lim);
    }
    if scroll.delta.y != 0.0 {
        // Zoom step scales with distance so it stays usable close in and far out.
        cam.dist = (cam.dist - scroll.delta.y * cam.dist * 0.1).clamp(cam.dist_min, cam.dist_max);
    }
}

/// Keeps the anchor camera orbiting/zooming the [`CameraTarget`]'s render position: the eye is
/// `target + orbit_offset(cam)`, tracking the target every frame.
#[allow(clippy::type_complexity)] // disjoint Bevy queries (target vs. camera)
pub fn orbit_follow_camera(
    cam: Res<OrbitFollowCam>,
    target: Query<&WorldPlacement, (With<CameraTarget>, Without<AnchorCamera>)>,
    mut camera: Query<
        (&mut Transform, &mut WorldPlacement),
        (With<AnchorCamera>, Without<CameraTarget>),
    >,
) {
    let Ok(target_wp) = target.single() else {
        return;
    };
    let Ok((mut tf, mut placement)) = camera.single_mut() else {
        return;
    };
    let target = target_wp.0.pos;
    let eye = target + orbit_offset(cam.yaw, cam.pitch, cam.dist).as_dvec3();
    placement.0 = WorldPos::new(FrameId::CENTRAL_BODY, eye);
    let look_dir = (target - eye).as_vec3().normalize_or_zero();
    if look_dir != Vec3::ZERO {
        tf.rotation = Transform::default().looking_to(look_dir, Vec3::Y).rotation;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    #[test]
    fn orbit_offset_default_reproduces_the_legacy_dive_view() {
        // Dive's default (yaw = π/4, pitch ≈ 0.305, dist ≈ 26.683) ≈ the old fixed (18, 8, 18).
        let off = orbit_offset(FRAC_PI_4, 0.305, 26.683);
        assert!((off.x - 18.0).abs() < 0.5, "x ≈ 18, got {}", off.x);
        assert!((off.y - 8.0).abs() < 0.5, "y ≈ 8, got {}", off.y);
        assert!((off.z - 18.0).abs() < 0.5, "z ≈ 18, got {}", off.z);
    }

    #[test]
    fn orbit_offset_pitch_raises_and_zoom_shortens() {
        let base = orbit_offset(FRAC_PI_4, 0.305, 26.683);
        let up = orbit_offset(FRAC_PI_4, 0.305 + 0.3, 26.683);
        assert!(up.y > base.y, "more pitch raises the eye");
        let near = orbit_offset(FRAC_PI_4, 0.305, 26.683 * 0.5);
        assert!(near.length() < base.length(), "less distance is closer");
    }

    #[test]
    fn orbit_offset_yaw_spins_around() {
        let a = orbit_offset(FRAC_PI_4, 0.305, 26.683);
        let b = orbit_offset(FRAC_PI_4 + 1.0, 0.305, 26.683);
        assert!(
            (a - b).length() > 1.0,
            "yaw moves the eye around the target"
        );
        assert!((a.length() - b.length()).abs() < 1e-3, "distance preserved");
    }
}

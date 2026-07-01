//! Shared **animated water patch** for water scenes (WI 714).
//!
//! The dive (`dive_scene`) and harbor (`harbor_scene`) both float a subdivided patch at sea level
//! that follows the camera and ripples via a bounded sum of travelling sines (WI 703). They had
//! near-identical copies differing only in the wave tuning; this module owns one implementation, with
//! the tuning a per-patch [`WaveSpec`] on [`WaterPatch`] (open-ocean vs calm-harbor presets), so each
//! scene reproduces its exact surface.

use bevy::math::DVec3;
use bevy::mesh::VertexAttributeValues;
use bevy::prelude::*;
use sounding_sim::frame::{FrameId, WorldPos};

use crate::floating_origin::{AnchorCamera, WorldPlacement};

/// The tuning of a water surface (WI 703/714): the peak amplitude (m) and the three travelling-sine
/// components' spatial frequencies and temporal rates. The component weights (0.45/0.35/0.20) sum to
/// 1, so the surface is bounded by `amplitude`.
#[derive(Clone, Copy, Debug)]
pub struct WaveSpec {
    /// Peak wave amplitude, metres: the surface oscillates within ±this.
    pub amplitude: f32,
    /// Spatial frequencies of the three components.
    pub space: [f32; 3],
    /// Temporal rates of the three components.
    pub time: [f32; 3],
}

impl WaveSpec {
    /// The open-ocean dive surface (WI 703): larger swell, faster motion.
    pub const OPEN_OCEAN: Self = Self {
        amplitude: 0.55,
        space: [0.08, 0.11, 0.05],
        time: [1.1, 0.9, 0.7],
    };
    /// The calm-harbor surface: a much smaller amplitude and gentler ripple than the open ocean.
    pub const CALM_HARBOR: Self = Self {
        amplitude: 0.12,
        space: [0.10, 0.13, 0.06],
        time: [0.7, 0.6, 0.5],
    };
}

/// The animated near-surface water patch (WI 703), carrying its [`WaveSpec`].
#[derive(Component)]
pub struct WaterPatch {
    pub wave: WaveSpec,
}

/// Height of the water surface at local patch coordinate `(x, z)` and time `t` (WI 703) — a bounded
/// sum of travelling sine waves per the [`WaveSpec`]. Pure (unit-tested); computed in the patch's
/// local frame so the surface ripples in place rather than scrolling as the camera moves.
pub fn wave_height(x: f32, z: f32, t: f32, wave: &WaveSpec) -> f32 {
    let w1 = (x * wave.space[0] + t * wave.time[0]).sin();
    let w2 = (z * wave.space[1] - t * wave.time[1]).sin();
    let w3 = ((x + z) * wave.space[2] + t * wave.time[2]).sin();
    wave.amplitude * (0.45 * w1 + 0.35 * w2 + 0.20 * w3)
}

/// Keeps the water patch under the view (follows the camera's X/Z at sea level) and ripples its
/// surface each frame by [`wave_height`], recomputing normals (WI 703).
#[allow(clippy::type_complexity)] // disjoint Bevy queries (camera vs. patch)
pub fn animate_water(
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera: Query<&WorldPlacement, (With<AnchorCamera>, Without<WaterPatch>)>,
    mut patch: Query<(&Mesh3d, &mut WorldPlacement, &WaterPatch), Without<AnchorCamera>>,
) {
    let Ok(cam_wp) = camera.single() else {
        return;
    };
    let Ok((mesh3d, mut wp, patch)) = patch.single_mut() else {
        return;
    };
    // Follow the camera horizontally at sea level (render Y = 0).
    let c = cam_wp.0.pos;
    wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, DVec3::new(c.x, 0.0, c.z));
    // Ripple the surface in the patch's local frame.
    let t = time.elapsed_secs();
    let Some(mesh) = meshes.get_mut(&mesh3d.0) else {
        return;
    };
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for p in positions.iter_mut() {
            p[1] = wave_height(p[0], p[2], t, &patch.wave);
        }
    }
    mesh.compute_normals();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_height_is_bounded_by_amplitude() {
        for wave in [WaveSpec::OPEN_OCEAN, WaveSpec::CALM_HARBOR] {
            for &t in &[0.0_f32, 1.7, 42.0] {
                for x in [-160.0_f32, 0.0, 160.0] {
                    for z in [-160.0_f32, 0.0, 160.0] {
                        assert!(
                            wave_height(x, z, t, &wave).abs() <= wave.amplitude + 1e-5,
                            "|wave| ≤ amplitude {}",
                            wave.amplitude
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn wave_height_animates_over_time() {
        let a = wave_height(3.0, 5.0, 0.0, &WaveSpec::OPEN_OCEAN);
        let b = wave_height(3.0, 5.0, 1.0, &WaveSpec::OPEN_OCEAN);
        assert!((a - b).abs() > 1e-6, "the surface moves with time");
    }

    // The calm harbor is quieter than the open ocean (compile-time).
    const _: () = assert!(WaveSpec::CALM_HARBOR.amplitude < WaveSpec::OPEN_OCEAN.amplitude);
}

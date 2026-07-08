//! App-level **debug camera + overlay control** (WI 784): lets an agent drive a scene's
//! debug camera and toggle its debug overlays over the runtime bus, so visual artifacts can
//! be reproduced and inspected with no human at the keyboard and no live display.
//!
//! Mirrors the [`crate::replay`] pattern: a [`DebugCommand`] is the unified envelope for the
//! bus (`POST /camera`, `POST /debug`) and is applied by a global system that no-ops in
//! scenes without a controllable camera. The headless `sounding_sim` crate is untouched — a
//! camera is a purely app/render concept. The current camera pose is published back
//! ([`DebugCameraState`]) so an agent can read where it aimed (`GET /camera`) — closed-loop
//! framing.
//!
//! Camera placement is **body-relative**: a scene that supports control sets
//! [`DebugCameraContext`] (its body centre + radius) and tags its camera [`DebugControllable`].
//! Positions are metres relative to the body centre; `SetOrbit` frames a point by
//! altitude + latitude/longitude. Same-frame aim uses the outward radial as "up".

use bevy::math::DVec3;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::floating_origin::WorldPlacement;
use sounding_sim::surface_mesh::SurfaceView;

/// How a placed camera should aim.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LookMode {
    /// Straight down at the surface under the camera.
    #[default]
    Nadir,
    /// Along the local horizon (tangent to the sphere, default azimuth = body "north").
    Horizon,
    /// Along a fixed body-relative direction.
    Direction([f64; 3]),
}

/// A debug camera/overlay control action — the envelope shared by `POST /camera` and
/// `POST /debug` (and drained into Bevy messages by the bus).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Message)]
#[serde(rename_all = "snake_case")]
pub enum DebugCommand {
    /// Place the camera at a body-relative orbit position and aim it.
    SetOrbit {
        altitude_m: f64,
        lat_deg: f64,
        lon_deg: f64,
        #[serde(default)]
        look: LookMode,
    },
    /// Place at a body-relative position (metres from the body centre) looking at a
    /// body-relative target (also metres from the body centre).
    SetPose {
        position: [f64; 3],
        look_at: [f64; 3],
    },
    /// Incremental move in the camera's own local frame + look deltas (degrees).
    Nudge {
        #[serde(default)]
        forward_m: f64,
        #[serde(default)]
        right_m: f64,
        #[serde(default)]
        up_m: f64,
        #[serde(default)]
        yaw_deg: f64,
        #[serde(default)]
        pitch_deg: f64,
    },
    /// Resolve a named reproducible framing (see [`resolve_named`]).
    NamedPose(String),
    /// Set debug overlays; `None` fields are left unchanged.
    SetOverlay {
        #[serde(default)]
        lod: Option<bool>,
        /// Surface color view (WI 869): `"biome"` (the shipping tint),
        /// `"dominant"`, `"temperature"`, or `"moisture"`. Unknown names are
        /// ignored (lenient like the rest of this surface).
        #[serde(default)]
        biome_view: Option<String>,
    },
}

/// The active scene's body frame for body-relative camera placement: `(center_world, radius)`
/// in metres. `None` ⇒ no controllable camera this scene (camera commands no-op).
#[derive(Resource, Default)]
pub struct DebugCameraContext(pub Option<(DVec3, f64)>);

/// Shared LOD/debug overlay state (was scene-local): toggled by the `F3` key **and** by
/// `DebugCommand::SetOverlay`.
#[derive(Resource, Default)]
pub struct DebugOverlayState(pub bool);

/// The surface color-view state (WI 869): which [`SurfaceView`] streamed chunks are
/// built with. Cycled by `F6` in `-- surface` and settable over the bus via
/// `DebugCommand::SetOverlay { biome_view }`; the streamer rebuilds chunks when it
/// changes. Default = the shipping biome tint.
#[derive(Resource, Default)]
pub struct BiomeViewState(pub SurfaceView);

/// The current debug-camera pose, published each frame and served by `GET /camera` so an
/// agent can read where it framed. `available` is false when no controllable camera exists.
#[derive(Resource, Default, Serialize, Clone)]
pub struct DebugCameraState {
    pub available: bool,
    /// Camera position, metres relative to the body centre.
    pub position: [f64; 3],
    /// Altitude above the sphere (metres).
    pub altitude_m: f64,
    /// Unit look direction (world/body frame).
    pub forward: [f64; 3],
}

/// Marks the camera a [`DebugCommand`] drives (the scene tags its controllable camera).
#[derive(Component)]
pub struct DebugControllable;

/// Body-relative unit direction for a latitude/longitude (degrees). Convention: `+Y` is the
/// body "north pole", longitude 0 along `+X`, increasing toward `+Z` (documented so an agent
/// can predict framings; the body's terrain has no intrinsic lat/long, this is just a frame).
pub fn dir_from_latlon(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

/// A horizontal (tangent) reference direction at outward radial `radial` — the body-"north"
/// tangent, or `+X` at the poles where north is undefined.
fn horizon_ref(radial: DVec3) -> DVec3 {
    let north = DVec3::Y - radial * radial.dot(DVec3::Y);
    if north.length() < 1e-6 {
        DVec3::X
    } else {
        north.normalize()
    }
}

/// Named reproducible poses → `(altitude_m, lat_deg, lon_deg, look)`.
pub fn resolve_named(name: &str) -> Option<(f64, f64, f64, LookMode)> {
    Some(match name {
        "nadir_200km" => (200_000.0, 0.0, 0.0, LookMode::Nadir),
        "nadir_20km" => (20_000.0, 0.0, 0.0, LookMode::Nadir),
        "grazing_horizon_6km" => (6_000.0, 0.0, 0.0, LookMode::Horizon),
        "grazing_horizon_60km" => (60_000.0, 20.0, 0.0, LookMode::Horizon),
        "high_orbit" => (400_000.0, 30.0, 20.0, LookMode::Nadir),
        _ => return None,
    })
}

/// The rotation that looks from `cam_world` toward `target_world`, using the outward radial
/// as up (a horizontal reference when the view is near-vertical, avoiding a degenerate basis).
fn aim_rotation(cam_world: DVec3, target_world: DVec3, center: DVec3) -> Quat {
    let dir = (target_world - cam_world).normalize_or_zero();
    let radial = (cam_world - center).normalize_or_zero();
    let up = if dir.cross(radial).length() < 1e-3 {
        horizon_ref(radial)
    } else {
        radial
    };
    Transform::default()
        .looking_to(dir.as_vec3(), up.as_vec3())
        .rotation
}

/// Places/aims the camera per one `SetOrbit`-style spec. Returns the new `(world_pos,
/// rotation)`.
fn orbit_pose(
    center: DVec3,
    radius: f64,
    altitude_m: f64,
    lat_deg: f64,
    lon_deg: f64,
    look: LookMode,
) -> (DVec3, Quat) {
    let d = dir_from_latlon(lat_deg, lon_deg);
    let alt = altitude_m.max(1.0);
    let cam_world = center + d * (radius + alt);
    let target = match look {
        LookMode::Nadir => center + d * radius, // the surface point below
        LookMode::Horizon => cam_world + horizon_ref(d) * radius,
        LookMode::Direction(dir) => cam_world + DVec3::from_array(dir).normalize_or_zero() * radius,
    };
    (cam_world, aim_rotation(cam_world, target, center))
}

/// Applies queued [`DebugCommand`]s to the controllable camera + overlay state. No-ops in
/// scenes without a [`DebugControllable`] camera or [`DebugCameraContext`] (overlay-only
/// commands still apply).
pub fn apply_debug_commands(
    mut reader: MessageReader<DebugCommand>,
    ctx: Res<DebugCameraContext>,
    mut overlay: ResMut<DebugOverlayState>,
    mut view: ResMut<BiomeViewState>,
    mut q: Query<(&mut Transform, &mut WorldPlacement), With<DebugControllable>>,
) {
    let mut camera = q.single_mut().ok();
    let frame = ctx.0; // Copy (DVec3, f64)
    for cmd in reader.read() {
        // Overlay commands apply regardless of camera availability.
        if let DebugCommand::SetOverlay { lod, biome_view } = cmd {
            if let Some(v) = lod {
                overlay.0 = *v;
            }
            // Lenient: unknown view names are ignored, never an error.
            if let Some(v) = biome_view.as_deref().and_then(SurfaceView::parse) {
                view.0 = v;
            }
            continue;
        }
        // Camera commands need both a controllable camera and a body frame; else no-op.
        let (Some((tf, placement)), Some((center, radius))) = (camera.as_mut(), frame) else {
            continue;
        };
        match cmd {
            DebugCommand::SetOrbit {
                altitude_m,
                lat_deg,
                lon_deg,
                look,
            } => {
                let (pos, rot) = orbit_pose(center, radius, *altitude_m, *lat_deg, *lon_deg, *look);
                placement.0.pos = pos;
                tf.rotation = rot;
            }
            DebugCommand::SetPose { position, look_at } => {
                let pos = center + DVec3::from_array(*position);
                let target = center + DVec3::from_array(*look_at);
                placement.0.pos = pos;
                tf.rotation = aim_rotation(pos, target, center);
            }
            DebugCommand::Nudge {
                forward_m,
                right_m,
                up_m,
                yaw_deg,
                pitch_deg,
            } => {
                let (fwd, right, up) = (*tf.forward(), *tf.right(), *tf.up());
                placement.0.pos += (fwd.as_dvec3() * *forward_m)
                    + (right.as_dvec3() * *right_m)
                    + (up.as_dvec3() * *up_m);
                tf.rotate_y(yaw_deg.to_radians() as f32);
                tf.rotate_local_x(pitch_deg.to_radians() as f32);
            }
            DebugCommand::NamedPose(name) => {
                if let Some((alt, lat, lon, look)) = resolve_named(name) {
                    let (pos, rot) = orbit_pose(center, radius, alt, lat, lon, look);
                    placement.0.pos = pos;
                    tf.rotation = rot;
                }
            }
            DebugCommand::SetOverlay { .. } => unreachable!("handled above"),
        }
    }
}

/// Publishes the controllable camera's body-relative pose into [`DebugCameraState`] each
/// frame (served by `GET /camera`).
pub fn publish_camera_pose(
    ctx: Res<DebugCameraContext>,
    mut state: ResMut<DebugCameraState>,
    q: Query<(&Transform, &WorldPlacement), With<DebugControllable>>,
) {
    match (ctx.0, q.single().ok()) {
        (Some((center, radius)), Some((tf, placement))) => {
            let body = placement.0.pos - center;
            state.available = true;
            state.position = body.to_array();
            state.altitude_m = body.length() - radius;
            state.forward = tf.forward().as_dvec3().to_array();
        }
        _ => state.available = false,
    }
}

/// Registers the debug-control envelope + resources + apply/publish systems. Added globally
/// (like [`crate::replay::ReplayPlugin`]); no-ops where no camera is [`DebugControllable`].
pub struct DebugControlPlugin;

impl Plugin for DebugControlPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<DebugCommand>()
            .init_resource::<DebugCameraContext>()
            .init_resource::<DebugOverlayState>()
            .init_resource::<BiomeViewState>()
            .init_resource::<DebugCameraState>()
            .add_systems(Update, (apply_debug_commands, publish_camera_pose).chain());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: f64 = 730_000.0;
    const CENTER: DVec3 = DVec3::new(0.0, -R, 0.0);

    #[test]
    fn command_json_round_trips() {
        let cmds = vec![
            DebugCommand::SetOrbit {
                altitude_m: 8000.0,
                lat_deg: 12.0,
                lon_deg: -30.0,
                look: LookMode::Nadir,
            },
            DebugCommand::SetPose {
                position: [1.0, 2.0, 3.0],
                look_at: [0.0, 0.0, 0.0],
            },
            DebugCommand::Nudge {
                forward_m: 100.0,
                right_m: 0.0,
                up_m: -5.0,
                yaw_deg: 10.0,
                pitch_deg: -2.0,
            },
            DebugCommand::NamedPose("nadir_200km".into()),
            DebugCommand::SetOverlay {
                lod: Some(true),
                biome_view: Some("dominant".into()),
            },
        ];
        for c in cmds {
            let j = serde_json::to_string(&c).unwrap();
            assert_eq!(c, serde_json::from_str(&j).unwrap(), "{j}");
        }
    }

    #[test]
    fn overlay_command_parses_from_partial_json() {
        // Fields default to None when omitted (additive/backward-compatible —
        // pre-869 clients send only `lod`); present values parse.
        let c: DebugCommand = serde_json::from_str(r#"{"set_overlay":{"lod":true}}"#).unwrap();
        assert_eq!(
            c,
            DebugCommand::SetOverlay {
                lod: Some(true),
                biome_view: None
            }
        );
        let c: DebugCommand = serde_json::from_str(r#"{"set_overlay":{}}"#).unwrap();
        assert_eq!(
            c,
            DebugCommand::SetOverlay {
                lod: None,
                biome_view: None
            }
        );
        let c: DebugCommand =
            serde_json::from_str(r#"{"set_overlay":{"biome_view":"temperature"}}"#).unwrap();
        assert_eq!(
            c,
            DebugCommand::SetOverlay {
                lod: None,
                biome_view: Some("temperature".into())
            }
        );
        // Unknown view names parse as data and are ignored at apply time.
        assert_eq!(SurfaceView::parse("bogus"), None);
        assert_eq!(
            SurfaceView::parse("dominant"),
            Some(SurfaceView::DominantBiome)
        );
    }

    #[test]
    fn latlon_directions_are_unit_and_oriented() {
        assert!((dir_from_latlon(0.0, 0.0) - DVec3::X).length() < 1e-9);
        assert!((dir_from_latlon(90.0, 0.0) - DVec3::Y).length() < 1e-9);
        assert!((dir_from_latlon(0.0, 90.0) - DVec3::Z).length() < 1e-9);
        for (lat, lon) in [(12.0, -30.0), (-80.0, 170.0), (45.0, 45.0)] {
            assert!((dir_from_latlon(lat, lon).length() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn orbit_pose_places_at_altitude_and_aims_down_for_nadir() {
        let (pos, rot) = orbit_pose(CENTER, R, 10_000.0, 0.0, 0.0, LookMode::Nadir);
        // Position is `altitude` above the surface along the lat/long direction.
        let body = pos - CENTER;
        assert!((body.length() - (R + 10_000.0)).abs() < 1e-3);
        // Nadir look points inward (toward the body centre): forward · (−radial) ≈ 1.
        let fwd = (rot * Vec3::NEG_Z).as_dvec3();
        let radial = body.normalize();
        assert!(fwd.dot(-radial) > 0.99, "nadir should look straight down");
    }

    #[test]
    fn horizon_pose_aims_tangent() {
        let (pos, rot) = orbit_pose(CENTER, R, 6_000.0, 0.0, 0.0, LookMode::Horizon);
        let body = pos - CENTER;
        let radial = body.normalize();
        let fwd = (rot * Vec3::NEG_Z).as_dvec3();
        // Tangent view: forward ⟂ radial (near zero dot).
        assert!(fwd.dot(radial).abs() < 0.05, "horizon should look tangent");
    }

    #[test]
    fn named_poses_resolve_and_unknown_is_none() {
        assert!(resolve_named("nadir_200km").is_some());
        assert!(resolve_named("grazing_horizon_6km").is_some());
        assert!(resolve_named("does_not_exist").is_none());
    }
}

//! The multiplayer net-adapter plugin (WI 857): connects the scenario scene to
//! a universe server and materializes remote vessels as **rails ghosts**.
//!
//! Core logic lives render-free in `sounding_netclient` (session lifecycle,
//! publish/fetch worker, ghost store — reused by the WI 858 headless
//! participant); this module owns only the app seams: configuration, the
//! event→record mapping, ghost presentation, and the `peers` telemetry bridge.
//!
//! **Configuration (per instance, no recompile):** environment variables
//! `SOUNDING_SERVER` (base URL), `SOUNDING_INVITE` (invite token),
//! `SOUNDING_PLAYER` (display name). Any missing ⇒ the plugin is fully
//! dormant and every scene behaves exactly as single-player.
//!
//! **Event → record mapping (today's real state-worthy events):** scenario
//! spawn ⇒ a `SurfaceFix` record at the pad (`live: false`); Launch/Flight ⇒
//! the fix republished `live: true` (the record then deliberately goes stale —
//! no motion streaming, per design); Recovery ⇒ landed fix (`live: false`) or
//! a `Fate::Destroyed` tombstone (R2) at the impact point; warp/pause change ⇒
//! an R1 anchor report (`universe_time = flight.elapsed`). A scenario spawn
//! opens a **fresh net session** (new session + vessel id): `elapsed` restarts
//! at zero per attempt, and a new attempt is genuinely a new subspace (the
//! plan-review coherence rule).
//!
//! **Ghosts are arithmetic, never physics**: the peer's real craft geometry
//! (its record's subgraph through the voxel skin), tinted translucent-teal,
//! positioned by `record.position_at(local time)` each frame — no collision,
//! no command targeting, no integration. Labels are camera-projected UI text.
//! Ghost orientation is a recorded M1 limitation (records carry no attitude).

use crate::voxel_skin::{pbr_material_tinted, skin_submeshes, VoxelSkin};
use bevy::prelude::*;
use sounding_netclient::{NetConfig, PeerView, SyncCommand, SyncHandle, SyncStatus};
use sounding_sim::director::ScenarioFlight;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::persist::CraftSubgraph;
use sounding_sim::session::{Outcome, Phase};
use sounding_sim::sim::SimClock;
use sounding_sim::telemetry::PeerTelemetry;
use sounding_sim::vessel::{mint_vessel_id, Fate, VesselRecord};
use std::sync::Mutex;

/// Ghost tint: an unmistakably-not-solid translucent teal.
const GHOST_TINT: Color = Color::srgba(0.45, 0.95, 0.9, 0.45);

/// Net configuration read from the environment at startup.
#[derive(Resource, Clone)]
struct NetSettings {
    server_url: String,
    invite_token: String,
    player: String,
}

/// The live adapter: the render-free sync worker + this attempt's identity.
/// (`SyncHandle` holds an mpsc `Sender`, which is `Send` but not `Sync`, so the
/// resource wraps it in a `Mutex` — locked only for brief sends/snapshots.)
#[derive(Resource)]
struct NetLink {
    handle: Mutex<SyncHandle>,
    /// This attempt's vessel id (a fresh instance per spawn, WI 855 semantics).
    vessel_id: String,
    /// Last session phase we published for (transition edge detection).
    last_phase: Phase,
    /// Last anchor state we reported (warp, paused).
    last_rate: f64,
}

/// Bus bridge: the peers block `publish_telemetry` attaches (WI 857), written
/// by the adapter each frame while connected.
#[derive(Resource, Default)]
pub struct NetPeers(pub Option<Vec<PeerTelemetry>>);

/// Marker + identity for a spawned ghost root (teardown tags the root only).
#[derive(Component)]
struct NetGhost {
    vessel_id: String,
    /// Stamp of the record the mesh was built from (rebuild on change).
    stamp: f64,
}

/// A ghost's floating UI label (screen-projected).
#[derive(Component)]
struct NetGhostLabel {
    vessel_id: String,
}

pub struct NetPlugin;

impl Plugin for NetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NetPeers>();
        let (server_url, invite_token, player) = match (
            std::env::var("SOUNDING_SERVER"),
            std::env::var("SOUNDING_INVITE"),
            std::env::var("SOUNDING_PLAYER"),
        ) {
            (Ok(s), Ok(i), Ok(p)) => (s, i, p),
            _ => return, // dormant: no resources beyond the empty bridge, no systems
        };
        info!("net: multiplayer adapter configured for {server_url} as {player}");
        app.insert_resource(NetSettings {
            server_url,
            invite_token,
            player,
        })
        .add_systems(
            Update,
            (
                net_start.run_if(resource_added::<ScenarioFlight>),
                net_track
                    .run_if(resource_exists::<NetLink>)
                    .run_if(resource_exists::<ScenarioFlight>),
                net_ghosts.run_if(resource_exists::<NetLink>),
                net_labels.run_if(resource_exists::<NetLink>),
            )
                .chain(),
        );
    }
}

/// The canonical content-identity string for the loaded scenario: id + name.
/// (Pack ids/versions are not carried on `ScenarioFlight`; enriching this to
/// the full pack list is a recorded follow-up — the handshake treats it as an
/// opaque equality token either way.)
fn content_identity(flight: &ScenarioFlight) -> String {
    format!("scenario:{}:{}", flight.id, flight.name)
}

/// Assembles a record from the primitive flight parts (pure — the unit-tested
/// B3 mapping; `assemble_record` adapts the live resource onto it).
#[allow(clippy::too_many_arguments)]
fn assemble_record_parts(
    scenario_id: &str,
    scenario_name: &str,
    position: bevy::math::DVec3,
    voxels: &sounding_sim::voxel::VoxelCraft,
    elapsed: f64,
    vessel_id: &str,
    player: &str,
    live: bool,
    fate: Option<Fate>,
) -> VesselRecord {
    let position = WorldPos::new(FrameId::CENTRAL_BODY, position);
    let mut structure = CraftSubgraph::new(scenario_id, scenario_name, position, voxels.clone());
    structure.vessel_id = Some(vessel_id.to_string());
    let mut record = VesselRecord::from_surface(
        vessel_id,
        scenario_name,
        player,
        elapsed,
        position,
        structure,
    );
    record.live = live;
    record.fate = fate;
    record
}

/// Assembles this attempt's record from the live flight state.
fn assemble_record(
    flight: &ScenarioFlight,
    vessel_id: &str,
    player: &str,
    live: bool,
    fate: Option<Fate>,
) -> VesselRecord {
    assemble_record_parts(
        &flight.id,
        &flight.name,
        flight.body.position,
        &flight.craft.voxels,
        flight.elapsed,
        vessel_id,
        player,
        live,
        fate,
    )
}

/// A scenario spawn: open a fresh session (or start the worker) and publish
/// the parked craft. Runs on every `ScenarioFlight` insertion — a new attempt
/// is a new subspace + a new vessel instance.
fn net_start(
    mut commands: Commands,
    settings: Res<NetSettings>,
    flight: Res<ScenarioFlight>,
    clock: Res<SimClock>,
    link: Option<ResMut<NetLink>>,
) {
    let vessel_id = mint_vessel_id();
    let rate = if clock.paused { 0.0 } else { clock.warp };
    let record = assemble_record(&flight, &vessel_id, &settings.player, false, None);

    match link {
        Some(mut link) => {
            // A later attempt: fresh session, fresh vessel; ghosts persist.
            let handle = link.handle.lock().expect("net worker");
            handle.send(SyncCommand::NewSession);
            handle
                .shared
                .lock()
                .expect("net shared")
                .ghosts
                .mark_own(&vessel_id);
            handle.send(SyncCommand::Anchor {
                universe_time: flight.elapsed,
                rate,
            });
            handle.send(SyncCommand::Publish(Box::new(record)));
            drop(handle);
            link.vessel_id = vessel_id;
            link.last_phase = flight.session.phase;
            link.last_rate = rate;
        }
        None => {
            let config = NetConfig::new(
                &settings.server_url,
                &settings.invite_token,
                &settings.player,
                content_identity(&flight),
            );
            let handle = SyncHandle::start(config);
            handle
                .shared
                .lock()
                .expect("net shared")
                .ghosts
                .mark_own(&vessel_id);
            handle.send(SyncCommand::Anchor {
                universe_time: flight.elapsed,
                rate,
            });
            handle.send(SyncCommand::Publish(Box::new(record)));
            commands.insert_resource(NetLink {
                handle: Mutex::new(handle),
                vessel_id,
                last_phase: flight.session.phase,
                last_rate: rate,
            });
        }
    }
}

/// Per-frame tracking: local time + ghost advancement, phase-transition
/// publishes, warp/pause anchor reports, and the peers bridge.
fn net_track(
    mut link: ResMut<NetLink>,
    flight: Res<ScenarioFlight>,
    clock: Res<SimClock>,
    settings: Res<NetSettings>,
    mut peers: ResMut<NetPeers>,
) {
    let t = flight.elapsed;
    let link = &mut *link;
    let handle = link.handle.lock().expect("net worker");

    // Local (subspace) time + materialization advance, every frame.
    let (views, connected) = {
        let mut shared = handle.shared.lock().expect("net shared");
        shared.local_time = t;
        shared.ghosts.advance(t);
        (
            shared.ghosts.visible(t),
            matches!(shared.status, SyncStatus::Connected { .. }),
        )
    };

    // Phase transitions → state-worthy publishes (the B3 mapping).
    let phase = flight.session.phase;
    if phase != link.last_phase {
        let record = match (phase, flight.session.outcome) {
            (Phase::Launch | Phase::Flight, _) => Some(assemble_record(
                &flight,
                &link.vessel_id,
                &settings.player,
                true,
                None,
            )),
            (Phase::Recovery, Outcome::Crashed) => Some(assemble_record(
                &flight,
                &link.vessel_id,
                &settings.player,
                false,
                Some(Fate::Destroyed),
            )),
            (Phase::Recovery, _) => Some(assemble_record(
                &flight,
                &link.vessel_id,
                &settings.player,
                false,
                None,
            )),
            (Phase::Build, _) => None, // handled by net_start
        };
        if let Some(record) = record {
            handle.send(SyncCommand::Publish(Box::new(record)));
        }
        link.last_phase = phase;
    }

    // Warp/pause changes → anchor re-report (R1).
    let rate = if clock.paused { 0.0 } else { clock.warp };
    if rate != link.last_rate {
        handle.send(SyncCommand::Anchor {
            universe_time: t,
            rate,
        });
        link.last_rate = rate;
    }
    drop(handle);

    // The peers bridge (data-side ghosts; absent while disconnected).
    peers.0 = connected.then(|| {
        views
            .iter()
            .map(|p| PeerTelemetry {
                vessel_id: p.record.vessel_id.clone(),
                name: p.record.name.clone(),
                player: p.record.owner.clone(),
                live: p.record.live,
                stamp: p.record.stamp,
                stale: p.stale,
            })
            .collect()
    });
}

/// The ghost's render-frame translation (the scene's pad-local mapping: world
/// minus the surface radius on +Y), at local time `t` — arithmetic only.
fn ghost_translation(view: &PeerView, t: f64, surface_radius: f64) -> Vec3 {
    let world = view.record.position_at(t).pos;
    let local = world - bevy::math::DVec3::new(0.0, surface_radius, 0.0);
    Vec3::new(local.x as f32, local.y as f32, local.z as f32)
}

/// Spawns/updates/despawns ghost roots to mirror the visible peer set.
#[allow(clippy::too_many_arguments)] // a Bevy system's parameter list (the bus.rs precedent)
fn net_ghosts(
    mut commands: Commands,
    link: Res<NetLink>,
    flight: Option<Res<ScenarioFlight>>,
    mut ghosts: Query<(Entity, &mut NetGhost, &mut Transform)>,
    labels: Query<(Entity, &NetGhostLabel)>,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(flight) = flight else { return };
    let t = flight.elapsed;
    let surface_radius = flight.params.surface_radius;
    let views = {
        let handle = link.handle.lock().expect("net worker");
        let shared = handle.shared.lock().expect("net shared");
        shared.ghosts.visible(t)
    };

    // Update or despawn existing ghosts (stamp change ⇒ rebuild mesh).
    let mut alive: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (entity, mut ghost, mut transform) in &mut ghosts {
        match views.iter().find(|v| v.record.vessel_id == ghost.vessel_id) {
            Some(view) if view.record.stamp == ghost.stamp => {
                transform.translation = ghost_translation(view, t, surface_radius);
                alive.insert(ghost.vessel_id.clone());
            }
            Some(view) => {
                // The shown record changed: rebuild by respawn (rare — a
                // state-worthy event on the peer's side).
                ghost.stamp = view.record.stamp;
                alive.insert(ghost.vessel_id.clone());
                commands.entity(entity).despawn();
                spawn_ghost(
                    &mut commands,
                    view,
                    t,
                    surface_radius,
                    &asset_server,
                    &mut meshes,
                    &mut materials,
                );
            }
            None => {
                commands.entity(entity).despawn();
                if let Some((label_entity, _)) =
                    labels.iter().find(|(_, l)| l.vessel_id == ghost.vessel_id)
                {
                    commands.entity(label_entity).despawn();
                }
            }
        }
    }

    // Spawn new ghosts (+ their labels).
    for view in views
        .iter()
        .filter(|v| !alive.contains(&v.record.vessel_id))
    {
        spawn_ghost(
            &mut commands,
            view,
            t,
            surface_radius,
            &asset_server,
            &mut meshes,
            &mut materials,
        );
        if !labels
            .iter()
            .any(|(_, l)| l.vessel_id == view.record.vessel_id)
        {
            commands.spawn((
                Text::new(String::new()),
                TextFont {
                    font_size: 13.0,
                    ..default()
                },
                TextColor(GHOST_TINT),
                Node {
                    position_type: PositionType::Absolute,
                    ..default()
                },
                NetGhostLabel {
                    vessel_id: view.record.vessel_id.clone(),
                },
            ));
        }
    }
}

/// Spawns one ghost root: the peer's real geometry, tinted, at its rails
/// position. No collision, no physics, no command targeting — a marker + a
/// transform + meshes, nothing else.
fn spawn_ghost(
    commands: &mut Commands,
    view: &PeerView,
    t: f64,
    surface_radius: f64,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let craft = &view.record.structure.craft;
    let translation = ghost_translation(view, t, surface_radius);
    let root = commands
        .spawn((
            NetGhost {
                vessel_id: view.record.vessel_id.clone(),
                stamp: view.record.stamp,
            },
            Transform::from_translation(translation),
            Visibility::default(),
        ))
        .id();
    for (material, mesh) in skin_submeshes(craft, VoxelSkin::Hull) {
        let handle = meshes.add(mesh);
        let mat = pbr_material_tinted(material, GHOST_TINT, asset_server, materials);
        let child = commands
            .spawn((Mesh3d(handle), MeshMaterial3d(mat), Transform::IDENTITY))
            .id();
        commands.entity(root).add_child(child);
    }
}

/// Projects each ghost's label to its screen position + refreshes its text
/// (name · player · live/parked · staleness).
fn net_labels(
    link: Res<NetLink>,
    flight: Option<Res<ScenarioFlight>>,
    camera: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    ghosts: Query<(&NetGhost, &GlobalTransform)>,
    mut labels: Query<(&NetGhostLabel, &mut Node, &mut Text, &mut Visibility)>,
) {
    let Some(flight) = flight else { return };
    let Ok((camera, cam_transform)) = camera.single() else {
        return;
    };
    let t = flight.elapsed;
    let views = {
        let handle = link.handle.lock().expect("net worker");
        let shared = handle.shared.lock().expect("net shared");
        shared.ghosts.visible(t)
    };
    for (label, mut node, mut text, mut visibility) in &mut labels {
        let Some(view) = views.iter().find(|v| v.record.vessel_id == label.vessel_id) else {
            *visibility = Visibility::Hidden;
            continue;
        };
        let Some((_, ghost_pos)) = ghosts
            .iter()
            .find(|(g, _)| g.vessel_id == label.vessel_id)
            .map(|(g, tf)| (g, tf.translation()))
        else {
            *visibility = Visibility::Hidden;
            continue;
        };
        let above = ghost_pos + Vec3::Y * 3.0;
        match camera.world_to_viewport(cam_transform, above) {
            Ok(screen) => {
                *visibility = Visibility::Visible;
                node.left = Val::Px(screen.x);
                node.top = Val::Px(screen.y);
                let state = if view.record.live {
                    format!("flying · last seen t+{:.0}s", view.stale)
                } else {
                    "parked".to_string()
                };
                text.0 = format!("{} · {} · {state}", view.record.name, view.record.owner);
            }
            Err(_) => *visibility = Visibility::Hidden,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::DVec3;
    use sounding_sim::voxel::VoxelCraft;

    // The pure B3 mapping: flight parts → the record we publish.
    #[test]
    fn assemble_record_maps_state_and_identity() {
        let pos = DVec3::new(1.0, 6.36e6, -2.0);
        let voxels = VoxelCraft::new(1.0);
        let record = assemble_record_parts(
            "first-flight",
            "First Flight",
            pos,
            &voxels,
            42.5,
            "vessel-1",
            "dave",
            false,
            None,
        );
        assert_eq!(record.vessel_id, "vessel-1");
        assert_eq!(record.owner, "dave");
        assert_eq!(record.stamp, 42.5);
        assert!(!record.live && record.fate.is_none());
        assert_eq!(
            record.structure.vessel_id.as_deref(),
            Some("vessel-1"),
            "instance id rides the subgraph"
        );
        match record.motion {
            sounding_sim::vessel::MotionState::SurfaceFix { position } => {
                assert_eq!(position.pos, pos);
                assert_eq!(position.frame, FrameId::CENTRAL_BODY);
            }
            _ => panic!("a scenario craft records a surface fix"),
        }

        let live =
            assemble_record_parts("s", "S", pos, &voxels, 50.0, "vessel-1", "dave", true, None);
        assert!(live.live);
        let tomb = assemble_record_parts(
            "s",
            "S",
            pos,
            &voxels,
            60.0,
            "vessel-1",
            "dave",
            false,
            Some(Fate::Destroyed),
        );
        assert!(tomb.is_tombstone());
    }
}

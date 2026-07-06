//! Resident-participant integration (WI 858): server + participant + a client
//! core in one process, no display — the CI shape, and the R5 exit semantics.

use sounding_netclient::participant::{ParticipantConfig, ParticipantHandle};
use sounding_netclient::{NetClient, NetConfig, SyncHandle, SyncStatus};
use sounding_server::store::StoreConfig;
use sounding_server::{start, ServerOptions};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::orbit::Orbit;
use sounding_sim::sim::CentralBody;
use sounding_sim::vessel::MotionState;
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};
use std::time::{Duration, Instant};

fn small_craft() -> VoxelCraft {
    let mut craft = VoxelCraft::new(1.0);
    craft.voxels.push(Voxel {
        cell: glam::IVec3::ZERO,
        material: Material::ALUMINIUM,
    });
    craft
}

fn server_with_ttl(ttl: f64) -> sounding_server::ServerHandle {
    start(ServerOptions {
        addr: "127.0.0.1:0".into(),
        store: StoreConfig {
            invite_token: "invite".into(),
            content_identity: None,
            lease_ttl: ttl,
        },
        ..Default::default()
    })
    .expect("server starts")
}

fn fast_net(url: &str, player: &str) -> NetConfig {
    let mut c = NetConfig::new(url, "invite", player, "content-a");
    c.heartbeat_period = 0.1;
    c.fetch_period = 0.05;
    c
}

fn parked(dx: f64) -> MotionState {
    MotionState::SurfaceFix {
        position: WorldPos::new(
            FrameId::CENTRAL_BODY,
            glam::DVec3::new(dx, CentralBody::EARTHLIKE.radius, 0.0),
        ),
    }
}

fn start_participant(url: &str, motion: MotionState) -> ParticipantHandle {
    ParticipantHandle::start(ParticipantConfig {
        net: fast_net(url, "bot"),
        vessel_name: "Presence".into(),
        craft: small_craft(),
        motion,
        tick: Duration::from_millis(20),
        log: false,
    })
}

fn wait(what: &str, mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !predicate() {
        assert!(Instant::now() < deadline, "timeout waiting for: {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn participant_presents_an_ordinary_vessel_and_sees_peers_too() {
    let server = server_with_ttl(300.0);
    let url = format!("http://{}", server.addr);
    let participant = start_participant(&url, parked(15.0));

    // A game-side client core sees the participant's vessel as an ordinary
    // parked record (protocol-indistinguishable from a human peer's).
    let game = SyncHandle::start(fast_net(&url, "dave"));
    game.shared.lock().unwrap().local_time = 100.0;
    wait("game sees the participant's vessel", || {
        let s = game.shared.lock().unwrap();
        s.ghosts.visible(s.local_time).iter().any(|p| {
            p.record.vessel_id == participant.vessel_id
                && p.record.name == "Presence"
                && p.record.owner == "bot"
                && !p.record.live
                && !p.record.is_tombstone()
                && matches!(p.record.motion, MotionState::SurfaceFix { position }
                    if position.pos.x == 15.0)
        })
    });

    // The participant is an observer too: a peer's record reaches its store
    // once the participant's wall-clock local time passes the record's stamp.
    let client = NetClient::new(&url);
    let session = client
        .handshake(&NetConfig::new(&url, "invite", "dave2", "content-a"))
        .expect("handshake");
    let peer_vessel = sounding_sim::vessel::mint_vessel_id();
    let record = sounding_sim::vessel::VesselRecord::from_surface(
        &peer_vessel,
        "Skiff",
        "dave2",
        0.05, // stamped near the participant's local zero
        WorldPos::new(
            FrameId::CENTRAL_BODY,
            glam::DVec3::new(-20.0, CentralBody::EARTHLIKE.radius, 0.0),
        ),
        sounding_sim::persist::CraftSubgraph::new(
            "skiff",
            "Skiff",
            WorldPos::new(FrameId::CENTRAL_BODY, glam::DVec3::ZERO),
            small_craft(),
        ),
    );
    client
        .publish(&session.session_token, &record)
        .expect("peer publish");
    wait("participant sees the peer", || {
        let s = participant.shared.lock().unwrap();
        let t = s.local_time;
        s.ghosts
            .visible(t)
            .iter()
            .any(|p| p.record.vessel_id == peer_vessel)
    });

    participant.shutdown();
    server.shutdown();
}

#[test]
fn orbit_mode_propagates_on_the_conic_for_observers() {
    let server = server_with_ttl(300.0);
    let url = format!("http://{}", server.addr);
    let mu = CentralBody::EARTHLIKE.mu;
    let r = CentralBody::EARTHLIKE.radius + 400_000.0;
    let orbit = Orbit::from_state(
        mu,
        glam::DVec2::new(r, 0.0),
        glam::DVec2::new(0.0, (mu / r).sqrt()),
        0.0,
    )
    .expect("bound");
    let participant = start_participant(
        &url,
        MotionState::Conic {
            frame: FrameId::CENTRAL_BODY,
            orbit,
        },
    );

    let game = SyncHandle::start(fast_net(&url, "dave"));
    game.shared.lock().unwrap().local_time = 10.0;
    wait("game sees the orbiting vessel", || {
        let s = game.shared.lock().unwrap();
        s.ghosts
            .visible(s.local_time)
            .iter()
            .any(|p| p.record.vessel_id == participant.vessel_id)
    });
    // The ghost's position is the closed-form conic at the observer's time —
    // two observer times give two positions, both matching the orbit exactly.
    let s = game.shared.lock().unwrap();
    let view = s
        .ghosts
        .visible(10.0)
        .into_iter()
        .find(|p| p.record.vessel_id == participant.vessel_id)
        .expect("visible");
    let quarter = orbit.period() / 4.0;
    let p1 = view.record.position_at(10.0);
    let p2 = view.record.position_at(10.0 + quarter);
    assert_ne!(p1.pos, p2.pos, "an orbiting ghost moves");
    let (expect1, _) = orbit.position_velocity(10.0);
    assert_eq!(p1.pos.x, expect1.x);
    assert_eq!(p1.pos.y, expect1.y);
    drop(s);

    participant.shutdown();
    server.shutdown();
}

#[test]
fn shutdown_leaves_the_vessel_stale_but_claimable_r5() {
    // Tiny TTL so the lease lapses promptly after shutdown (deliberate
    // per-test lease choices — the arc's standing discipline).
    let server = server_with_ttl(0.3);
    let url = format!("http://{}", server.addr);
    let mut net = fast_net(&url, "bot");
    // Cadences below the TTL: the participant survives unattended while alive.
    net.heartbeat_period = 0.05;
    net.fetch_period = 0.05;
    let participant = ParticipantHandle::start(ParticipantConfig {
        net,
        vessel_name: "Presence".into(),
        craft: small_craft(),
        motion: parked(15.0),
        tick: Duration::from_millis(20),
        log: false,
    });
    let vessel_id = participant.vessel_id.clone();

    // Alive across several TTL windows: still connected, vessel held.
    // (The observer handshakes *after* the wait — its own lease is subject to
    // the same tiny TTL.)
    std::thread::sleep(Duration::from_millis(900)); // 3× TTL
    let observer = NetClient::new(&url);
    let obs = observer
        .handshake(&NetConfig::new(&url, "invite", "watcher", "content-a"))
        .expect("observer handshake");
    let obs_keepalive = || {
        let _ = observer.heartbeat(&obs.session_token);
    };
    let (records, _) = observer.fetch_since(&obs.session_token, 0).expect("fetch");
    let rec = records
        .iter()
        .find(|r| r.vessel_id == vessel_id)
        .expect("participant's vessel present");
    assert!(
        rec.authority.is_some(),
        "unattended participant still holds its vessel across TTL windows"
    );
    {
        let shared = participant.shared.lock().unwrap();
        assert!(matches!(shared.status, SyncStatus::Connected { .. }));
    }

    // Shutdown = stop heartbeating (no logout op by design). The lease lapses:
    // the record stands, authority releases (stale-but-claimable).
    participant.shutdown();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        obs_keepalive();
        let (records, _) = observer.fetch_since(&obs.session_token, 0).expect("fetch");
        let rec = records
            .iter()
            .find(|r| r.vessel_id == vessel_id)
            .expect("the last record stands");
        if rec.authority.is_none() {
            assert!(!rec.is_tombstone(), "stale, not destroyed");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the lease to release the vessel"
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    // ...and actually claimable: an anchored observer takes it.
    observer
        .anchor(&obs.session_token, 1000.0, 1.0)
        .expect("observer anchors ahead of the stamp");
    observer
        .claim(&obs.session_token, &vessel_id)
        .expect("the released vessel is claimable");

    server.shutdown();
}

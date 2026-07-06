//! The headless M1 integration proof (plan B6): a real universe server + two
//! (then three) client cores in one test process — no display, no app. This is
//! the fixture shape WI 858's participant reuses.

use sounding_netclient::{NetClient, NetConfig, NetError, SyncCommand, SyncHandle, SyncStatus};
use sounding_server::store::StoreConfig;
use sounding_server::{start, ServerOptions};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::persist::CraftSubgraph;
use sounding_sim::vessel::{mint_vessel_id, Fate, VesselRecord};
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};
use std::time::{Duration, Instant};

fn fix_record(vessel_id: &str, name: &str, owner: &str, stamp: f64) -> VesselRecord {
    let mut craft = VoxelCraft::new(1.0);
    craft.voxels.push(Voxel {
        cell: glam::IVec3::ZERO,
        material: Material::ALUMINIUM,
    });
    VesselRecord::from_surface(
        vessel_id,
        name,
        owner,
        stamp,
        WorldPos::new(FrameId::CENTRAL_BODY, glam::DVec3::new(6.36e6, 0.0, 0.0)),
        CraftSubgraph::new(
            "starter",
            name,
            WorldPos::new(FrameId::CENTRAL_BODY, glam::DVec3::ZERO),
            craft,
        ),
    )
}

fn test_server() -> sounding_server::ServerHandle {
    start(ServerOptions {
        addr: "127.0.0.1:0".into(),
        store: StoreConfig {
            invite_token: "invite".into(),
            content_identity: None,
            // Generous: leases are not this test's subject (the WI 856 lesson).
            lease_ttl: 300.0,
        },
        ..Default::default()
    })
    .expect("server starts")
}

fn fast_config(url: &str, player: &str) -> NetConfig {
    let mut c = NetConfig::new(url, "invite", player, "content-a");
    c.heartbeat_period = 0.2;
    c.fetch_period = 0.05;
    c
}

/// Polls `predicate` against the handle's shared state until it holds or the
/// timeout expires; panics with `what` on timeout.
fn wait_for(
    handle: &SyncHandle,
    what: &str,
    predicate: impl Fn(&sounding_netclient::sync::SyncShared) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        {
            let shared = handle.shared.lock().unwrap();
            if predicate(&shared) {
                return;
            }
        }
        assert!(Instant::now() < deadline, "timeout waiting for: {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn m1_two_clients_see_each_other_and_a_late_third_converges() {
    let server = test_server();
    let url = format!("http://{}", server.addr);

    // Alice and Bob: sync workers with their own local times (subspaces).
    let alice = SyncHandle::start(fast_config(&url, "alice"));
    let bob = SyncHandle::start(fast_config(&url, "bob"));
    alice.shared.lock().unwrap().local_time = 100.0;
    bob.shared.lock().unwrap().local_time = 40.0;

    wait_for(&alice, "alice connected", |s| {
        matches!(s.status, SyncStatus::Connected { .. })
    });
    wait_for(&bob, "bob connected", |s| {
        matches!(s.status, SyncStatus::Connected { .. })
    });

    // Each publishes a parked craft (stamped in their own subspace time) and
    // marks it own so the fetched-back copy never self-ghosts.
    let a_vessel = mint_vessel_id();
    let b_vessel = mint_vessel_id();
    alice.shared.lock().unwrap().ghosts.mark_own(&a_vessel);
    bob.shared.lock().unwrap().ghosts.mark_own(&b_vessel);
    alice.send(SyncCommand::Anchor {
        universe_time: 100.0,
        rate: 1.0,
    });
    bob.send(SyncCommand::Anchor {
        universe_time: 40.0,
        rate: 1.0,
    });
    alice.send(SyncCommand::Publish(Box::new(fix_record(
        &a_vessel, "Ranger", "alice", 100.0,
    ))));
    bob.send(SyncCommand::Publish(Box::new(fix_record(
        &b_vessel, "Skiff", "bob", 40.0,
    ))));

    // Bob's record (stamp 40) is in Alice's past: visible to her.
    wait_for(&alice, "alice sees bob's skiff", |s| {
        let t = s.local_time;
        s.ghosts
            .visible(t)
            .iter()
            .any(|p| p.record.vessel_id == b_vessel && p.record.owner == "bob")
    });
    // Alice's record (stamp 100) is in Bob's future: pending, not visible —
    // until Bob's subspace time passes it (the causality rule end to end).
    {
        let shared = bob.shared.lock().unwrap();
        assert!(
            shared.ghosts.visible(shared.local_time).is_empty(),
            "the past must not see the future"
        );
    }
    {
        let mut shared = bob.shared.lock().unwrap();
        shared.local_time = 120.0;
        shared.ghosts.advance(120.0);
    }
    wait_for(&bob, "bob sees alice's ranger after syncing forward", |s| {
        s.ghosts
            .visible(s.local_time)
            .iter()
            .any(|p| p.record.vessel_id == a_vessel && p.record.owner == "alice")
    });

    // A late-joining Carol converges to the same two-vessel picture.
    let carol = SyncHandle::start(fast_config(&url, "carol"));
    carol.shared.lock().unwrap().local_time = 500.0;
    wait_for(&carol, "carol converges to both vessels", |s| {
        s.ghosts.visible(s.local_time).len() == 2
    });

    // Alice crashes: a tombstone stamped in her time. Carol (ahead) despawns
    // it once ingested + advanced; her view drops to one vessel.
    let mut tomb = fix_record(&a_vessel, "Ranger", "alice", 130.0);
    tomb.fate = Some(Fate::Destroyed);
    alice.send(SyncCommand::Publish(Box::new(tomb)));
    wait_for(&carol, "carol despawns the tombstoned ranger", |s| {
        let t = s.local_time;
        // advance() runs on the caller side each tick in the app; emulate it.
        // (visible() alone never shows tombstones; len drops when promoted.)
        s.ghosts.visible(t).len() == 1
    });

    alice.shutdown();
    bob.shutdown();
    carol.shutdown();
    server.shutdown();
}

#[test]
fn content_mismatch_and_bad_invite_reject_legibly_and_keep_retrying() {
    let server = test_server();
    let url = format!("http://{}", server.addr);

    // Pin the identity with a first good client.
    let good = SyncHandle::start(fast_config(&url, "alice"));
    wait_for(&good, "good client connected", |s| {
        matches!(s.status, SyncStatus::Connected { .. })
    });

    // A wrong-content client never connects; the server's legible message
    // reaches its status.
    let mut bad = fast_config(&url, "mallory");
    bad.content_identity = "content-b".into();
    let mallory = SyncHandle::start(bad);
    wait_for(&mallory, "mallory rejected with the content message", |s| {
        matches!(&s.status, SyncStatus::Connecting(Some(msg))
            if msg.contains("content identity mismatch") && msg.contains("content-b"))
    });

    // A wrong-invite client likewise.
    let mut bad2 = fast_config(&url, "eve");
    bad2.invite_token = "wrong".into();
    let eve = SyncHandle::start(bad2);
    wait_for(
        &eve,
        "eve rejected on the invite",
        |s| matches!(&s.status, SyncStatus::Connecting(Some(msg)) if msg.contains("invite")),
    );

    mallory.shutdown();
    eve.shutdown();
    good.shutdown();
    server.shutdown();
}

#[test]
fn direct_client_calls_and_the_non_finite_guard() {
    let server = test_server();
    let url = format!("http://{}", server.addr);
    let client = NetClient::new(&url);
    let config = NetConfig::new(&url, "invite", "alice", "content-a");

    let session = client.handshake(&config).expect("handshake");
    client
        .anchor(&session.session_token, 10.0, 1.0)
        .expect("anchor");
    client.heartbeat(&session.session_token).expect("heartbeat");

    // Publish + fetch round-trip.
    let record = fix_record(&mint_vessel_id(), "Ranger", "alice", 5.0);
    let cursor = client
        .publish(&session.session_token, &record)
        .expect("publish");
    let (records, new_cursor) = client
        .fetch_since(&session.session_token, 0)
        .expect("fetch");
    assert_eq!(new_cursor, cursor);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].vessel_id, record.vessel_id);

    // The pre-encode guard fires locally: nothing non-finite reaches the wire.
    let mut bad = fix_record(&mint_vessel_id(), "Bad", "alice", f64::NAN);
    bad.stamp = f64::NAN;
    match client.publish(&session.session_token, &bad) {
        Err(NetError::NonFiniteRecord) => {}
        other => panic!("expected the local non-finite guard, got {other:?}"),
    }

    // A server rejection surfaces its legible message (backward anchor).
    match client.anchor(&session.session_token, 1.0, 1.0) {
        Err(NetError::Server {
            status: 409,
            message,
        }) => {
            assert!(message.contains("backward sync"), "{message}");
        }
        other => panic!("expected the 409 backward-sync rejection, got {other:?}"),
    }

    server.shutdown();
}

#[test]
fn lease_expiry_reconnect_reclaims_and_republishes_the_own_vessel() {
    // The code-review fix: a lease expiry releases our vessel's authority; the
    // worker's re-handshake must CLAIM it back before republishing, or the
    // vessel goes permanently stale for peers. Tiny TTL + a heartbeat period
    // longer than it forces the expiry path (deliberate per-test lease
    // choices — the WI 856 lesson).
    let server = start(ServerOptions {
        addr: "127.0.0.1:0".into(),
        store: StoreConfig {
            invite_token: "invite".into(),
            content_identity: None,
            lease_ttl: 0.3,
        },
        ..Default::default()
    })
    .expect("server starts");
    let url = format!("http://{}", server.addr);

    let mut config = fast_config(&url, "alice");
    // ALL traffic renews the lease (any authenticated request is activity),
    // so both cadences must exceed the TTL for the lease to lapse between ops.
    config.heartbeat_period = 1.0;
    config.fetch_period = 1.0;
    let alice = SyncHandle::start(config);
    alice.shared.lock().unwrap().local_time = 10.0;
    wait_for(&alice, "alice connected", |s| {
        matches!(s.status, SyncStatus::Connected { .. })
    });
    let first_session = match &alice.shared.lock().unwrap().status {
        SyncStatus::Connected { session_id } => session_id.clone(),
        _ => unreachable!(),
    };
    let vessel = mint_vessel_id();
    alice.shared.lock().unwrap().ghosts.mark_own(&vessel);
    alice.send(SyncCommand::Anchor {
        universe_time: 10.0,
        rate: 1.0,
    });
    alice.send(SyncCommand::Publish(Box::new(fix_record(
        &vessel, "Ranger", "alice", 10.0,
    ))));

    // Wait for the lease to lapse and the worker to come back on a NEW session.
    wait_for(
        &alice,
        "alice re-handshakes after expiry",
        |s| matches!(&s.status, SyncStatus::Connected { session_id } if *session_id != first_session),
    );

    // An independent observer sees the vessel HELD by the new session — the
    // reconnect claimed it back and republished (no silent NotAuthority).
    let observer = NetClient::new(&url);
    let obs = observer
        .handshake(&NetConfig::new(&url, "invite", "bob", "content-a"))
        .expect("observer handshake");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let (records, _) = observer
            .fetch_since(&obs.session_token, 0)
            .expect("observer fetch");
        let held = records.iter().any(|r| {
            r.vessel_id == vessel
                && r.authority.is_some()
                && r.authority.as_deref() != Some(first_session.as_str())
        });
        if held {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out: vessel not re-claimed/republished after reconnect"
        );
        std::thread::sleep(Duration::from_millis(30));
    }
    assert!(
        alice.shared.lock().unwrap().last_error.is_none(),
        "the republish succeeded (no lingering op error)"
    );

    alice.shutdown();
    server.shutdown();
}

//! Integration tests: the pure router (protocol behaviors without a socket),
//! the world-save restart path, and one in-process socket round-trip — the
//! "server + clients in one test process" fixture shape WIs 857/858 reuse.

use glam::{DVec2, DVec3, IVec3};
use sounding_server::router::{handle_request, DEFAULT_SIZE_CAP};
use sounding_server::store::{StoreConfig, UniverseStore, PROTOCOL_VERSION};
use sounding_server::{load_store, start, world_save_json, ServerOptions};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::orbit::Orbit;
use sounding_sim::persist::CraftSubgraph;
use sounding_sim::sim::CentralBody;
use sounding_sim::vessel::{mint_vessel_id, VesselRecord};
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

fn sample_record(stamp: f64) -> VesselRecord {
    let mu = CentralBody::EARTHLIKE.mu;
    let orbit = Orbit::from_state(
        mu,
        DVec2::new(7.0e6, 0.0),
        DVec2::new(0.0, (mu / 7.0e6).sqrt()),
        stamp,
    )
    .expect("bound");
    let mut craft = VoxelCraft::new(1.0);
    craft.voxels.push(Voxel {
        cell: IVec3::ZERO,
        material: Material::ALUMINIUM,
    });
    VesselRecord::from_rails(
        mint_vessel_id(),
        "Ranger",
        "dave",
        stamp,
        FrameId::CENTRAL_BODY,
        orbit,
        CraftSubgraph::new(
            "starter",
            "Starter",
            WorldPos::new(FrameId::CENTRAL_BODY, DVec3::ZERO),
            craft,
        ),
    )
}

fn test_store() -> UniverseStore {
    UniverseStore::new(StoreConfig {
        invite_token: "invite".into(),
        content_identity: None,
        lease_ttl: 30.0,
    })
}

fn handshake_body() -> String {
    serde_json::json!({
        "protocol_version": PROTOCOL_VERSION,
        "invite_token": "invite",
        "player": "alice",
        "content_identity": "content-a",
    })
    .to_string()
}

/// Routes a request with defaults; returns (status, parsed JSON body).
fn route(
    store: &mut UniverseStore,
    now: f64,
    method: &str,
    url: &str,
    token: Option<&str>,
    body: &str,
) -> (u16, serde_json::Value) {
    let (status, out) = handle_request(store, now, method, url, token, body, DEFAULT_SIZE_CAP);
    let json = serde_json::from_str(&out).unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[test]
fn router_flow_handshake_anchor_publish_fetch_locks_status() {
    let mut store = test_store();

    // Handshake.
    let (status, hs) = route(
        &mut store,
        0.0,
        "POST",
        "/handshake",
        None,
        &handshake_body(),
    );
    assert_eq!(status, 200, "{hs}");
    let token = hs["session_token"].as_str().expect("token").to_string();
    let session_id = hs["session_id"].as_str().expect("id").to_string();
    assert_eq!(hs["protocol_version"], PROTOCOL_VERSION);

    // Anchor + registry with derived time.
    let (status, _) = route(
        &mut store,
        1.0,
        "POST",
        "/anchor",
        Some(&token),
        &serde_json::json!({"universe_time": 100.0, "rate": 2.0}).to_string(),
    );
    assert_eq!(status, 200);
    let (status, reg) = route(&mut store, 6.0, "GET", "/registry", Some(&token), "");
    assert_eq!(status, 200);
    assert_eq!(reg["sessions"][0]["id"], session_id.as_str());
    assert_eq!(reg["sessions"][0]["universe_time"], 110.0); // 100 + 2·5

    // Publish a record; fetch-since sees exactly it; cursor advances.
    let record = sample_record(50.0);
    let vessel_id = record.vessel_id.clone();
    let body = serde_json::to_string(&record).unwrap();
    let (status, pub_out) = route(&mut store, 7.0, "POST", "/records", Some(&token), &body);
    assert_eq!(status, 200, "{pub_out}");
    assert_eq!(pub_out["cursor"], 1);
    let (status, fetched) = route(&mut store, 8.0, "GET", "/records?since=0", Some(&token), "");
    assert_eq!(status, 200);
    assert_eq!(fetched["cursor"], 1);
    assert_eq!(fetched["records"][0]["vessel_id"], vessel_id.as_str());
    assert_eq!(
        fetched["records"][0]["authority"],
        session_id.as_str(),
        "authority is the public session id, never the token"
    );
    let (_, none) = route(&mut store, 9.0, "GET", "/records?since=1", Some(&token), "");
    assert_eq!(none["records"].as_array().unwrap().len(), 0);

    // Locks over the wire: release then claim back (anchor is ahead of stamp).
    let lock = |op: &str| serde_json::json!({"op": op, "vessel_id": vessel_id}).to_string();
    let (status, _) = route(
        &mut store,
        10.0,
        "POST",
        "/locks",
        Some(&token),
        &lock("release"),
    );
    assert_eq!(status, 200);
    let (status, _) = route(
        &mut store,
        11.0,
        "POST",
        "/locks",
        Some(&token),
        &lock("claim"),
    );
    assert_eq!(status, 200);

    // Status (authenticated) reports the universe.
    let (status, st) = route(&mut store, 12.0, "GET", "/status", Some(&token), "");
    assert_eq!(status, 200);
    assert_eq!(st["protocol_version"], PROTOCOL_VERSION);
    assert_eq!(st["sessions"], 1);
    assert_eq!(st["vessels"], 1);
    assert_eq!(st["content_identity"], "content-a");
}

#[test]
fn router_rejections_are_legible_and_correctly_coded() {
    let mut store = test_store();

    // Unauthenticated access to every non-handshake route.
    for (method, url) in [
        ("POST", "/anchor"),
        ("POST", "/heartbeat"),
        ("GET", "/registry"),
        ("POST", "/records"),
        ("GET", "/records?since=0"),
        ("POST", "/locks"),
        ("GET", "/status"),
    ] {
        let (status, body) = route(&mut store, 0.0, method, url, None, "{}");
        assert_eq!(status, 401, "{method} {url}");
        assert!(body["error"].is_string(), "{method} {url}");
    }

    // Version mismatch names both versions.
    let bad_version = serde_json::json!({
        "protocol_version": PROTOCOL_VERSION + 7,
        "invite_token": "invite",
        "player": "p",
        "content_identity": "c",
    })
    .to_string();
    let (status, body) = route(&mut store, 0.0, "POST", "/handshake", None, &bad_version);
    assert_eq!(status, 400);
    let msg = body["error"].as_str().unwrap();
    assert!(msg.contains(&PROTOCOL_VERSION.to_string()));
    assert!(msg.contains(&(PROTOCOL_VERSION + 7).to_string()));

    // Malformed JSON is a 400, never a panic; unknown route 404.
    let (status, _) = route(&mut store, 0.0, "POST", "/handshake", None, "{ not json");
    assert_eq!(status, 400);
    let (_, hs) = route(
        &mut store,
        0.0,
        "POST",
        "/handshake",
        None,
        &handshake_body(),
    );
    let token = hs["session_token"].as_str().unwrap().to_string();
    let (status, _) = route(&mut store, 1.0, "POST", "/records", Some(&token), "[]");
    assert_eq!(status, 400);
    let (status, _) = route(&mut store, 1.0, "GET", "/nope", Some(&token), "");
    assert_eq!(status, 404);

    // R4 size cap: an oversized body is a 413 before any parse.
    let huge = "x".repeat(DEFAULT_SIZE_CAP + 1);
    let (status, body) = route(&mut store, 2.0, "POST", "/records", Some(&token), &huge);
    assert_eq!(status, 413);
    assert!(body["error"].as_str().unwrap().contains("cap"));

    // R4 non-finite motion, layered: JSON's grammar cannot carry a non-finite
    // float at all (serde_json encodes it as `null`), so over this transport
    // the record fails the parse — a 400 at the boundary. The store's own
    // 422-class NonFinite check is pinned at the store level (unit test) as
    // defense in depth for any future non-JSON transport.
    let mut bad = sample_record(10.0);
    if let sounding_sim::vessel::MotionState::Conic { orbit, .. } = &mut bad.motion {
        orbit.eccentricity = f64::INFINITY;
    }
    let encoded = serde_json::to_string(&bad).unwrap();
    assert!(encoded.contains("null"), "JSON encodes non-finite as null");
    let (status, body) = route(&mut store, 3.0, "POST", "/records", Some(&token), &encoded);
    assert_eq!(status, 400);
    assert!(body["error"].as_str().unwrap().contains("malformed"));

    // Bad lock op is a 400 naming the vocabulary.
    let (status, body) = route(
        &mut store,
        4.0,
        "POST",
        "/locks",
        Some(&token),
        &serde_json::json!({"op": "steal", "vessel_id": "v"}).to_string(),
    );
    assert_eq!(status, 400);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("claim|release|transfer"));
}

#[test]
fn world_save_round_trips_through_disk_and_a_corrupt_save_fails_loudly() {
    let dir = std::env::temp_dir().join(format!("sounding-server-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let save = dir.join("universe.json");

    // Build a store with two records via the store API, save it.
    let mut store = test_store();
    let (_, token) = {
        let (status, hs) = {
            let (s, b) = handle_request(
                &mut store,
                0.0,
                "POST",
                "/handshake",
                None,
                &handshake_body(),
                DEFAULT_SIZE_CAP,
            );
            (s, serde_json::from_str::<serde_json::Value>(&b).unwrap())
        };
        assert_eq!(status, 200);
        (
            hs["session_id"].as_str().unwrap().to_string(),
            hs["session_token"].as_str().unwrap().to_string(),
        )
    };
    let r1 = sample_record(10.0);
    let r2 = sample_record(20.0);
    store.publish(1.0, &token, r1.clone()).unwrap();
    store.publish(2.0, &token, r2.clone()).unwrap();
    std::fs::write(&save, world_save_json(&store).unwrap()).unwrap();

    // Restart path: records restored, authority cleared, cursor reset-fresh.
    let options = ServerOptions {
        save_path: Some(save.clone()),
        store: StoreConfig {
            invite_token: "invite".into(),
            ..Default::default()
        },
        ..Default::default()
    };
    let restored = load_store(&options).expect("loads");
    let vessels = restored.vessels();
    assert_eq!(vessels.len(), 2);
    assert!(vessels.iter().all(|r| r.authority.is_none()));
    let ids: Vec<&str> = vessels.iter().map(|r| r.vessel_id.as_str()).collect();
    assert!(ids.contains(&r1.vessel_id.as_str()) && ids.contains(&r2.vessel_id.as_str()));

    // A missing save starts empty; a corrupt one fails loudly (never silently
    // starts an empty universe over a real file).
    let empty = load_store(&ServerOptions {
        save_path: Some(dir.join("absent.json")),
        ..Default::default()
    })
    .expect("missing save is a fresh universe");
    assert_eq!(empty.vessels().len(), 0);
    std::fs::write(dir.join("corrupt.json"), "{ not a save").unwrap();
    assert!(load_store(&ServerOptions {
        save_path: Some(dir.join("corrupt.json")),
        ..Default::default()
    })
    .is_err());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn socket_round_trip_two_clients_share_a_universe() {
    // The in-process fixture shape for WIs 857/858: real HTTP, ephemeral port.
    let handle = start(ServerOptions {
        addr: "127.0.0.1:0".into(),
        store: StoreConfig {
            invite_token: "invite".into(),
            content_identity: None,
            lease_ttl: 30.0,
        },
        ..Default::default()
    })
    .expect("starts");
    let base = format!("http://{}", handle.addr);

    let handshake = |player: &str| -> (String, String) {
        let body = ureq::post(format!("{base}/handshake"))
            .send(
                serde_json::json!({
                    "protocol_version": PROTOCOL_VERSION,
                    "invite_token": "invite",
                    "player": player,
                    "content_identity": "content-a",
                })
                .to_string(),
            )
            .expect("handshake")
            .into_body()
            .read_to_string()
            .expect("body");
        let body: serde_json::Value = serde_json::from_str(&body).expect("json");
        (
            body["session_id"].as_str().unwrap().to_string(),
            body["session_token"].as_str().unwrap().to_string(),
        )
    };
    let (_alice_id, alice) = handshake("alice");
    let (_bob_id, bob) = handshake("bob");

    // A wrong-content third client is rejected at the door (409).
    let err = ureq::post(format!("{base}/handshake"))
        .send(
            serde_json::json!({
                "protocol_version": PROTOCOL_VERSION,
                "invite_token": "invite",
                "player": "mallory",
                "content_identity": "content-b",
            })
            .to_string(),
        )
        .expect_err("content mismatch");
    match err {
        ureq::Error::StatusCode(code) => assert_eq!(code, 409),
        other => panic!("expected a status error, got {other}"),
    }

    // Alice publishes; Bob fetches and sees her vessel.
    let record = sample_record(10.0);
    let vessel_id = record.vessel_id.clone();
    ureq::post(format!("{base}/records"))
        .header("x-session-token", &alice)
        .send(serde_json::to_string(&record).unwrap())
        .expect("publish");
    let fetched = ureq::get(format!("{base}/records?since=0"))
        .header("x-session-token", &bob)
        .call()
        .expect("fetch")
        .into_body()
        .read_to_string()
        .expect("body");
    let fetched: serde_json::Value = serde_json::from_str(&fetched).expect("json");
    assert_eq!(fetched["records"][0]["vessel_id"], vessel_id.as_str());

    // Unauthenticated fetch is rejected.
    let err = ureq::get(format!("{base}/records?since=0"))
        .call()
        .expect_err("no token");
    match err {
        ureq::Error::StatusCode(code) => assert_eq!(code, 401),
        other => panic!("expected a status error, got {other}"),
    }

    // Status over the wire.
    let status = ureq::get(format!("{base}/status"))
        .header("x-session-token", &alice)
        .call()
        .expect("status")
        .into_body()
        .read_to_string()
        .expect("body");
    let status: serde_json::Value = serde_json::from_str(&status).expect("json");
    assert_eq!(status["vessels"], 1);
    assert_eq!(status["sessions"], 2);

    handle.shutdown();
}

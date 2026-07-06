//! The protocol router: pure `(method, url, token, body) → (status, body)` over
//! the store — testable without a socket (the app bus's router pattern). The
//! transport layer owns only sockets, header extraction, and the clock.
//!
//! # Wire contract (protocol version 1 — WI 856; the reference for WI 857/858)
//!
//! Auth: every route except `POST /handshake` requires the session token in the
//! `x-session-token` header. All bodies are JSON. Errors are
//! `{ "error": "<legible message>" }` with a 4xx status.
//!
//! - `POST /handshake` `{protocol_version, invite_token, player,
//!   content_identity}` → `{session_id, session_token, protocol_version}`.
//!   Rejections: 400 version mismatch (names both), 401 bad invite, 409 content
//!   mismatch (names both).
//! - `POST /anchor` `{universe_time, rate}` — R1 anchor report (rate 0 =
//!   paused). 409 on backward sync (reported time behind previously reported).
//! - `POST /heartbeat` `{}` — explicit lease renewal (any authenticated request
//!   also renews).
//! - `GET /registry` → `{sessions: [{id, player, universe_time?, rate?}]}` with
//!   `universe_time` derived from the anchor at the server's "now".
//! - `POST /records` (body: a persist-line `VesselRecord`) → `{cursor}`.
//!   Rejections: 413 over the size cap, 400 parse, 422 non-finite (defense in
//!   depth — JSON's grammar cannot carry a non-finite float, so over this
//!   transport such a record already fails the parse as a 400), 403 not the
//!   authority, 409 stale stamp / tombstoned.
//! - `GET /records?since=N` → `{records: [...], cursor}` — exactly the records
//!   (tombstones included) changed past the cursor, oldest change first.
//! - `POST /locks` `{op: "claim"|"release"|"transfer", vessel_id, to_session?}`
//!   → `{}`. Rejections: 404 unknown vessel/target, 403 not the authority,
//!   409 already held / tombstoned / behind the record stamp (sync forward) /
//!   no anchor.
//! - `GET /status` → `{protocol_version, sessions, vessels, cursor,
//!   content_identity}` (authenticated — it enumerates the universe).

use crate::store::{StoreError, UniverseStore, PROTOCOL_VERSION};
use serde::{Deserialize, Serialize};
use sounding_sim::vessel::VesselRecord;

/// Default request-body byte cap (R4): a hostile or buggy client cannot park a
/// multi-gigabyte "craft" in every peer's fetch path.
pub const DEFAULT_SIZE_CAP: usize = 1024 * 1024;

#[derive(Deserialize)]
struct HandshakeReq {
    protocol_version: u32,
    invite_token: String,
    player: String,
    content_identity: String,
}

#[derive(Serialize)]
struct HandshakeOk<'a> {
    session_id: &'a str,
    session_token: &'a str,
    protocol_version: u32,
}

#[derive(Deserialize)]
struct AnchorReq {
    universe_time: f64,
    rate: f64,
}

#[derive(Serialize)]
struct RegistrySession<'a> {
    id: &'a str,
    player: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    universe_time: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rate: Option<f64>,
}

#[derive(Deserialize)]
struct LockReq {
    op: String,
    vessel_id: String,
    #[serde(default)]
    to_session: Option<String>,
}

fn error_body(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

fn status_for(err: &StoreError) -> u16 {
    match err {
        StoreError::ProtocolMismatch(..) => 400,
        StoreError::BadInvite | StoreError::Unauthorized => 401,
        StoreError::NotAuthority => 403,
        StoreError::UnknownVessel | StoreError::UnknownTarget => 404,
        StoreError::ContentMismatch(..)
        | StoreError::BackwardSync { .. }
        | StoreError::StaleStamp { .. }
        | StoreError::Tombstoned
        | StoreError::AlreadyHeld
        | StoreError::ClaimBehindRecord { .. }
        | StoreError::NoAnchor => 409,
        StoreError::NonFinite => 422,
    }
}

fn reject(err: StoreError) -> (u16, String) {
    (status_for(&err), error_body(&err.msg()))
}

/// Routes one request against the store. `token` is the extracted
/// `x-session-token` header value, if any; `now` is server-monotonic seconds.
pub fn handle_request(
    store: &mut UniverseStore,
    now: f64,
    method: &str,
    url: &str,
    token: Option<&str>,
    body: &str,
    size_cap: usize,
) -> (u16, String) {
    // R5: expired leases release their locks on any traffic, not only their own.
    store.sweep(now);
    // R4: the byte cap guards every parse below.
    if body.len() > size_cap {
        return (
            413,
            error_body(&format!(
                "request body of {} bytes exceeds the {size_cap}-byte cap",
                body.len()
            )),
        );
    }
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p, q),
        None => (url, ""),
    };
    match (method, path) {
        ("POST", "/handshake") => {
            let req: HandshakeReq = match serde_json::from_str(body) {
                Ok(r) => r,
                Err(e) => return (400, error_body(&format!("malformed handshake: {e}"))),
            };
            match store.handshake(
                now,
                req.protocol_version,
                &req.invite_token,
                &req.player,
                &req.content_identity,
            ) {
                Ok((id, tok)) => (
                    200,
                    serde_json::to_string(&HandshakeOk {
                        session_id: &id,
                        session_token: &tok,
                        protocol_version: PROTOCOL_VERSION,
                    })
                    .expect("serializable"),
                ),
                Err(e) => reject(e),
            }
        }
        _ => {
            // Everything else is authenticated.
            let token = match token {
                Some(t) => t,
                None => return reject(StoreError::Unauthorized),
            };
            match (method, path) {
                ("POST", "/anchor") => {
                    let req: AnchorReq = match serde_json::from_str(body) {
                        Ok(r) => r,
                        Err(e) => return (400, error_body(&format!("malformed anchor: {e}"))),
                    };
                    match store.report_anchor(now, token, req.universe_time, req.rate) {
                        Ok(()) => (200, "{}".to_string()),
                        Err(e) => reject(e),
                    }
                }
                ("POST", "/heartbeat") => match store.heartbeat(now, token) {
                    Ok(()) => (200, "{}".to_string()),
                    Err(e) => reject(e),
                },
                ("GET", "/registry") => match store.auth(now, token) {
                    Ok(_) => {
                        let sessions = store.registry(now);
                        let rows: Vec<RegistrySession> = sessions
                            .iter()
                            .map(|(s, t)| RegistrySession {
                                id: &s.id,
                                player: &s.player,
                                universe_time: *t,
                                rate: s.anchor.map(|a| a.rate),
                            })
                            .collect();
                        (200, serde_json::json!({ "sessions": rows }).to_string())
                    }
                    Err(e) => reject(e),
                },
                ("POST", "/records") => {
                    let record: VesselRecord = match serde_json::from_str(body) {
                        Ok(r) => r,
                        Err(e) => {
                            return (400, error_body(&format!("malformed vessel record: {e}")))
                        }
                    };
                    match store.publish(now, token, record) {
                        Ok(cursor) => (200, serde_json::json!({ "cursor": cursor }).to_string()),
                        Err(e) => reject(e),
                    }
                }
                ("GET", "/records") => {
                    let since: u64 = query
                        .split('&')
                        .find_map(|kv| kv.strip_prefix("since="))
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    match store.fetch_since(now, token, since) {
                        Ok((records, cursor)) => (
                            200,
                            serde_json::json!({ "records": records, "cursor": cursor }).to_string(),
                        ),
                        Err(e) => reject(e),
                    }
                }
                ("POST", "/locks") => {
                    let req: LockReq = match serde_json::from_str(body) {
                        Ok(r) => r,
                        Err(e) => return (400, error_body(&format!("malformed lock op: {e}"))),
                    };
                    let result = match req.op.as_str() {
                        "claim" => store.claim(now, token, &req.vessel_id),
                        "release" => store.release(now, token, &req.vessel_id),
                        "transfer" => match req.to_session.as_deref() {
                            Some(to) => store.transfer(now, token, &req.vessel_id, to),
                            None => return (400, error_body("transfer requires to_session")),
                        },
                        other => {
                            return (
                                400,
                                error_body(&format!(
                                    "unknown lock op \"{other}\" (claim|release|transfer)"
                                )),
                            )
                        }
                    };
                    match result {
                        Ok(()) => (200, "{}".to_string()),
                        Err(e) => reject(e),
                    }
                }
                ("GET", "/status") => match store.auth(now, token) {
                    Ok(_) => {
                        let (sessions, vessels, cursor) = store.status_counts();
                        (
                            200,
                            serde_json::json!({
                                "protocol_version": PROTOCOL_VERSION,
                                "sessions": sessions,
                                "vessels": vessels,
                                "cursor": cursor,
                                "content_identity": store.content_identity(),
                            })
                            .to_string(),
                        )
                    }
                    Err(e) => reject(e),
                },
                _ => (404, error_body("no such route")),
            }
        }
    }
}

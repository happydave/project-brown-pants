//! The net-adapter **core** (WI 857, multiplayer arc): the render-free client
//! side of the universe-server protocol — session lifecycle, record
//! publish/fetch, and the ghost store — shared by the windowed app's adapter
//! plugin and the WI 858 headless resident participant.
//!
//! Wire contract: `sounding_server`'s `router.rs` module docs (protocol v1).
//! Design: `tickets/docs/projects/sounding/multiplayer/design.md`.
//!
//! Three layers, deliberately separable:
//! - [`NetClient`] — thin blocking protocol calls (one HTTP request each).
//! - [`GhostStore`] — the pure materialization state: per remote vessel a
//!   *shown* record (newest received with stamp ≤ local time) plus an optional
//!   *pending* future-stamped record. The pair preserves both causality (the
//!   future is never shown) and continuity (a behind-observer's view doesn't
//!   blink out when the owner updates from its future — the newest-per-vessel
//!   server store would otherwise make the vessel vanish for them).
//! - [`SyncHandle`] — the background worker owning **all** blocking I/O:
//!   handshake → anchor → heartbeat/fetch cadence → queued publishes, with
//!   backoff + automatic re-handshake on lease loss; state shared through a
//!   mutex snapshot the caller polls (frame systems / a headless loop).

pub mod ghost;
pub mod sync;

pub use ghost::{GhostStore, PeerView};
pub use sync::{SyncCommand, SyncHandle, SyncStatus};

use serde::Deserialize;
use sounding_sim::vessel::{MotionState, VesselRecord};

/// The protocol version this client speaks (must match the server's).
pub const PROTOCOL_VERSION: u32 = 1;

/// Connection + identity configuration for one session.
#[derive(Clone, Debug)]
pub struct NetConfig {
    /// Server base URL, e.g. `http://127.0.0.1:8790`.
    pub server_url: String,
    /// Pre-shared invite token (the LAN trust boundary).
    pub invite_token: String,
    /// Player display identity.
    pub player: String,
    /// Canonical content-identity string (equality-checked at handshake).
    pub content_identity: String,
    /// Heartbeat period, seconds (keep well under the server lease TTL).
    pub heartbeat_period: f64,
    /// Fetch-since poll period, seconds (rails cadence).
    pub fetch_period: f64,
}

impl NetConfig {
    /// A config with default cadences (heartbeat 5 s, fetch 1 s — for the
    /// server's default 30 s lease TTL).
    pub fn new(
        server_url: impl Into<String>,
        invite_token: impl Into<String>,
        player: impl Into<String>,
        content_identity: impl Into<String>,
    ) -> Self {
        Self {
            server_url: server_url.into(),
            invite_token: invite_token.into(),
            player: player.into(),
            content_identity: content_identity.into(),
            heartbeat_period: 5.0,
            fetch_period: 1.0,
        }
    }
}

/// A client-side failure: transport, a server rejection (with the server's
/// legible message), or the local pre-encode guard.
#[derive(Debug)]
pub enum NetError {
    /// Connection/transport failure (server unreachable, I/O error).
    Transport(String),
    /// The server rejected the request: HTTP status + its legible message.
    Server { status: u16, message: String },
    /// Local guard: a record carries a non-finite motion/stamp value. JSON
    /// would silently narrow it to `null` on the wire (and bounce as a
    /// confusing parse error), so it is refused *before* encoding (the WI 856
    /// reflect lesson).
    NonFiniteRecord,
}

impl std::fmt::Display for NetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetError::Transport(e) => write!(f, "transport: {e}"),
            NetError::Server { status, message } => write!(f, "server ({status}): {message}"),
            NetError::NonFiniteRecord => write!(
                f,
                "record refused locally: non-finite motion/stamp value (would encode as JSON null)"
            ),
        }
    }
}

impl std::error::Error for NetError {}

/// An open session's identity.
#[derive(Clone, Debug)]
pub struct SessionInfo {
    /// Public session id (names authority in records).
    pub session_id: String,
    /// Secret bearer token.
    pub session_token: String,
}

#[derive(Deserialize)]
struct HandshakeResp {
    session_id: String,
    session_token: String,
}

#[derive(Deserialize)]
struct PublishResp {
    cursor: u64,
}

#[derive(Deserialize)]
struct FetchResp {
    records: Vec<VesselRecord>,
    cursor: u64,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: String,
}

/// True iff every motion/stamp value in the record is finite (the pre-encode
/// guard — see [`NetError::NonFiniteRecord`]).
pub fn record_is_finite(record: &VesselRecord) -> bool {
    let motion = match &record.motion {
        MotionState::Conic { orbit, .. } => {
            orbit.mu.is_finite()
                && orbit.semi_major_axis.is_finite()
                && orbit.eccentricity.is_finite()
                && orbit.arg_periapsis.is_finite()
                && orbit.mean_anomaly_at_epoch.is_finite()
                && orbit.epoch.is_finite()
                && orbit.sense.is_finite()
        }
        MotionState::SurfaceFix { position } => position.pos.is_finite(),
    };
    motion && record.stamp.is_finite()
}

/// Thin blocking protocol-v1 client: one HTTP request per call. Use from a
/// worker thread ([`SyncHandle`]) — never from a frame path.
pub struct NetClient {
    agent: ureq::Agent,
    base: String,
}

impl NetClient {
    /// A client for `server_url` (no connection is made until a call).
    pub fn new(server_url: &str) -> Self {
        // Status errors are handled by us (we want the server's legible error
        // body, which ureq's default status-as-error path discards).
        let config = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build();
        Self {
            agent: config.into(),
            base: server_url.trim_end_matches('/').to_string(),
        }
    }

    fn call(
        &self,
        method: &str,
        path: &str,
        token: Option<&str>,
        body: Option<String>,
    ) -> Result<String, NetError> {
        let url = format!("{}{path}", self.base);
        let result = match (method, body) {
            ("GET", _) => {
                let mut req = self.agent.get(&url);
                if let Some(t) = token {
                    req = req.header("x-session-token", t);
                }
                req.call()
            }
            (_, body) => {
                let mut req = self.agent.post(&url);
                if let Some(t) = token {
                    req = req.header("x-session-token", t);
                }
                req.send(body.unwrap_or_else(|| "{}".to_string()))
            }
        };
        let mut resp = result.map_err(|e| NetError::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| NetError::Transport(e.to_string()))?;
        if (200..300).contains(&status) {
            Ok(text)
        } else {
            let message = serde_json::from_str::<ErrorBody>(&text)
                .map(|b| b.error)
                .unwrap_or_else(|_| text);
            Err(NetError::Server { status, message })
        }
    }

    /// `POST /handshake` — opens a session.
    pub fn handshake(&self, config: &NetConfig) -> Result<SessionInfo, NetError> {
        let body = serde_json::json!({
            "protocol_version": PROTOCOL_VERSION,
            "invite_token": config.invite_token,
            "player": config.player,
            "content_identity": config.content_identity,
        })
        .to_string();
        let text = self.call("POST", "/handshake", None, Some(body))?;
        let resp: HandshakeResp =
            serde_json::from_str(&text).map_err(|e| NetError::Transport(e.to_string()))?;
        Ok(SessionInfo {
            session_id: resp.session_id,
            session_token: resp.session_token,
        })
    }

    /// `POST /anchor` — R1 subspace anchor report.
    pub fn anchor(&self, token: &str, universe_time: f64, rate: f64) -> Result<(), NetError> {
        let body = serde_json::json!({ "universe_time": universe_time, "rate": rate }).to_string();
        self.call("POST", "/anchor", Some(token), Some(body))
            .map(|_| ())
    }

    /// `POST /heartbeat` — lease renewal.
    pub fn heartbeat(&self, token: &str) -> Result<(), NetError> {
        self.call("POST", "/heartbeat", Some(token), None)
            .map(|_| ())
    }

    /// `POST /records` — publishes a record (guarded: see
    /// [`NetError::NonFiniteRecord`]). Returns the server cursor.
    pub fn publish(&self, token: &str, record: &VesselRecord) -> Result<u64, NetError> {
        if !record_is_finite(record) {
            return Err(NetError::NonFiniteRecord);
        }
        let body = serde_json::to_string(record).map_err(|e| NetError::Transport(e.to_string()))?;
        let text = self.call("POST", "/records", Some(token), Some(body))?;
        let resp: PublishResp =
            serde_json::from_str(&text).map_err(|e| NetError::Transport(e.to_string()))?;
        Ok(resp.cursor)
    }

    /// `POST /locks` claim — takes an unheld vessel's authority (used on
    /// reconnect: a lease expiry released our own vessel's lock, and a publish
    /// requires holding it — the code-review reconnect fix).
    pub fn claim(&self, token: &str, vessel_id: &str) -> Result<(), NetError> {
        let body = serde_json::json!({ "op": "claim", "vessel_id": vessel_id }).to_string();
        self.call("POST", "/locks", Some(token), Some(body))
            .map(|_| ())
    }

    /// `GET /records?since=N` — the cursor diff (tombstones included).
    pub fn fetch_since(
        &self,
        token: &str,
        cursor: u64,
    ) -> Result<(Vec<VesselRecord>, u64), NetError> {
        let text = self.call(
            "GET",
            &format!("/records?since={cursor}"),
            Some(token),
            None,
        )?;
        let resp: FetchResp =
            serde_json::from_str(&text).map_err(|e| NetError::Transport(e.to_string()))?;
        Ok((resp.records, resp.cursor))
    }
}

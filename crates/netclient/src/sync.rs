//! The sync worker: owns all blocking I/O on a background thread.
//!
//! Lifecycle: handshake → initial anchor → cadenced heartbeat + fetch-since →
//! queued publishes/anchors, with exponential backoff and automatic
//! re-handshake on transport failure or lease loss (a 401 after expiry). On
//! every (re)handshake the cursor resets to 0 and the full record set is
//! refetched — ghost ingest is idempotent, and this is what makes a server
//! restart (runtime-only cursors) transparent.
//!
//! The caller (frame system or headless loop) communicates through
//! [`SyncCommand`]s and polls the shared state: connection status, the ghost
//! store, and its own **local time**, which the caller writes each tick (the
//! subspace time ghosts materialize against).

use crate::ghost::GhostStore;
use crate::{NetClient, NetConfig, NetError, SessionInfo};
use sounding_sim::vessel::VesselRecord;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Commands from the caller to the worker.
#[derive(Debug)]
pub enum SyncCommand {
    /// Publish (or republish) our vessel's record. The worker remembers the
    /// latest as "current" and republishes it after a re-handshake.
    Publish(Box<VesselRecord>),
    /// Report the subspace anchor (universe time + rate; pause = 0).
    Anchor { universe_time: f64, rate: f64 },
    /// Drop the session and open a fresh one (a new scenario attempt = a new
    /// subspace: new session id, fresh anchor history — the plan-review fix).
    NewSession,
    /// Stop the worker thread.
    Shutdown,
}

/// Connection status snapshot.
#[derive(Clone, Debug, PartialEq)]
pub enum SyncStatus {
    /// Not yet connected / retrying (with the last error, if any).
    Connecting(Option<String>),
    /// Live session.
    Connected {
        /// Public session id.
        session_id: String,
    },
    /// Worker stopped.
    Stopped,
}

/// State shared between the worker and the caller.
pub struct SyncShared {
    /// Connection status.
    pub status: SyncStatus,
    /// Materialization state for remote vessels.
    pub ghosts: GhostStore,
    /// The caller's local (subspace) time, written by the caller each tick;
    /// the worker stamps ingests with it.
    pub local_time: f64,
    /// The most recent non-fatal operation failure (e.g. a publish rejection),
    /// for observability — never silent (code-review fix). Cleared on success.
    pub last_error: Option<String>,
}

/// A running sync worker: command sender + shared state.
pub struct SyncHandle {
    /// Shared state (lock briefly; the worker never holds it across I/O).
    pub shared: Arc<Mutex<SyncShared>>,
    tx: Sender<SyncCommand>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SyncHandle {
    /// Starts the worker for `config`.
    pub fn start(config: NetConfig) -> Self {
        let shared = Arc::new(Mutex::new(SyncShared {
            status: SyncStatus::Connecting(None),
            ghosts: GhostStore::default(),
            local_time: 0.0,
            last_error: None,
        }));
        let (tx, rx) = std::sync::mpsc::channel();
        let shared2 = shared.clone();
        let thread = std::thread::spawn(move || worker_loop(config, shared2, rx));
        Self {
            shared,
            tx,
            thread: Some(thread),
        }
    }

    /// Sends a command to the worker (best-effort once stopped).
    pub fn send(&self, cmd: SyncCommand) {
        let _ = self.tx.send(cmd);
    }

    /// Stops the worker and joins it.
    pub fn shutdown(mut self) {
        let _ = self.tx.send(SyncCommand::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Worker-local session state.
struct Live {
    session: SessionInfo,
    cursor: u64,
    last_heartbeat: Instant,
    last_fetch: Instant,
}

fn worker_loop(config: NetConfig, shared: Arc<Mutex<SyncShared>>, rx: Receiver<SyncCommand>) {
    let client = NetClient::new(&config.server_url);
    let mut live: Option<Live> = None;
    // The latest published record + anchor, replayed after any re-handshake.
    let mut current_record: Option<VesselRecord> = None;
    let mut current_anchor: Option<(f64, f64)> = None;
    let mut backoff = Duration::from_millis(250);
    let mut next_attempt = Instant::now();
    let tick = Duration::from_millis(50);

    loop {
        // 1) Drain commands (never blocking the loop on the channel).
        loop {
            match rx.try_recv() {
                Ok(SyncCommand::Shutdown) => {
                    if let Ok(mut s) = shared.lock() {
                        s.status = SyncStatus::Stopped;
                    }
                    return;
                }
                Ok(SyncCommand::NewSession) => {
                    live = None;
                    current_record = None;
                    current_anchor = None;
                    next_attempt = Instant::now();
                    if let Ok(mut s) = shared.lock() {
                        s.status = SyncStatus::Connecting(None);
                    }
                }
                Ok(SyncCommand::Publish(record)) => {
                    current_record = Some((*record).clone());
                    if let Some(l) = &live {
                        drop_on_auth_error(
                            client
                                .publish(&l.session.session_token, &record)
                                .map(|_| ()),
                            &mut live,
                            &shared,
                        );
                    }
                }
                Ok(SyncCommand::Anchor {
                    universe_time,
                    rate,
                }) => {
                    current_anchor = Some((universe_time, rate));
                    if let Some(l) = &live {
                        drop_on_auth_error(
                            client.anchor(&l.session.session_token, universe_time, rate),
                            &mut live,
                            &shared,
                        );
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if let Ok(mut s) = shared.lock() {
                        s.status = SyncStatus::Stopped;
                    }
                    return;
                }
            }
        }

        // 2) Connect (with backoff) when down.
        if live.is_none() && Instant::now() >= next_attempt {
            match client.handshake(&config) {
                Ok(session) => {
                    let now = Instant::now();
                    // Replay identity-critical state on the fresh session:
                    // anchor first (the claim guard needs it), then — for a
                    // *reconnect* carrying an existing vessel — re-claim its
                    // authority (the old lease's expiry released it; a publish
                    // requires holding it) before republishing. A claim
                    // rejection is expected for a brand-new vessel
                    // (UnknownVessel) and harmless; publish failures surface
                    // via `last_error`, never silently (code-review fix).
                    if let Some((t, r)) = current_anchor {
                        report_op(client.anchor(&session.session_token, t, r), &shared);
                    }
                    if let Some(rec) = &current_record {
                        let _ = client.claim(&session.session_token, &rec.vessel_id);
                        report_op(
                            client.publish(&session.session_token, rec).map(|_| ()),
                            &shared,
                        );
                    }
                    if let Ok(mut s) = shared.lock() {
                        s.status = SyncStatus::Connected {
                            session_id: session.session_id.clone(),
                        };
                    }
                    live = Some(Live {
                        session,
                        cursor: 0, // full refetch; ingest is idempotent
                        last_heartbeat: now,
                        last_fetch: now - Duration::from_secs(3600), // fetch now
                    });
                    backoff = Duration::from_millis(250);
                }
                Err(e) => {
                    if let Ok(mut s) = shared.lock() {
                        s.status = SyncStatus::Connecting(Some(e.to_string()));
                    }
                    next_attempt = Instant::now() + backoff;
                    backoff = (backoff * 2).min(Duration::from_secs(10));
                }
            }
        }

        // 3) Cadenced work while live.
        if let Some(l) = &mut live {
            let now = Instant::now();
            if now.duration_since(l.last_heartbeat).as_secs_f64() >= config.heartbeat_period {
                l.last_heartbeat = now;
                let result = client.heartbeat(&l.session.session_token);
                drop_on_auth_error(result, &mut live, &shared);
            }
        }
        if let Some(l) = &mut live {
            let now = Instant::now();
            if now.duration_since(l.last_fetch).as_secs_f64() >= config.fetch_period {
                l.last_fetch = now;
                match client.fetch_since(&l.session.session_token, l.cursor) {
                    Ok((records, cursor)) => {
                        l.cursor = cursor;
                        if let Ok(mut s) = shared.lock() {
                            let t = s.local_time;
                            for record in records {
                                s.ghosts.ingest(record, t);
                            }
                        }
                    }
                    Err(e) => drop_on_auth_error(Err(e), &mut live, &shared),
                }
            }
        }

        std::thread::sleep(tick);
    }
}

/// Records a non-fatal op outcome into `last_error` (success clears it) —
/// operation failures are observable, never silent.
fn report_op(result: Result<(), NetError>, shared: &Arc<Mutex<SyncShared>>) {
    if let Ok(mut s) = shared.lock() {
        s.last_error = match result {
            Ok(()) => None,
            Err(e) => Some(e.to_string()),
        };
    }
}

/// On an auth failure (expired lease / server restart) or transport error,
/// drop the session so the loop re-handshakes with backoff; other server
/// rejections (e.g. a stale stamp) are not connection failures — they are
/// recorded via [`report_op`] and the session stays live.
fn drop_on_auth_error(
    result: Result<(), NetError>,
    live: &mut Option<Live>,
    shared: &Arc<Mutex<SyncShared>>,
) {
    match result {
        Ok(()) => report_op(Ok(()), shared),
        Err(NetError::Server {
            status: 401,
            message,
        }) => {
            *live = None;
            if let Ok(mut s) = shared.lock() {
                s.status = SyncStatus::Connecting(Some(message));
            }
        }
        Err(NetError::Transport(message)) => {
            *live = None;
            if let Ok(mut s) = shared.lock() {
                s.status = SyncStatus::Connecting(Some(message));
            }
        }
        Err(other) => report_op(Err(other), shared),
    }
}

//! Sounding universe server (WI 856, multiplayer arc): a headless
//! **store-and-relay** — vessel records, subspace registry, authority ledger,
//! session leases — behind a small versioned JSON-over-HTTP protocol. It
//! **never simulates**: times are derived arithmetically from R1 anchors, and
//! reconciliation stays state-based end to end (design:
//! `tickets/docs/projects/sounding/multiplayer/design.md`).
//!
//! Threat model (deliberate, documented): LAN/friends scale. A pre-shared
//! invite token gates the handshake; per-session bearer tokens scope authority;
//! there is no TLS — do not expose this to a hostile network.
//!
//! The crate is a library (so integration tests — here and in WIs 857/858 —
//! run server + clients in one process via [`start`] on an ephemeral port)
//! plus a small binary (`main.rs`) that runs it from flags.

pub mod router;
pub mod store;

use router::DEFAULT_SIZE_CAP;
use sounding_sim::persist::{FormatError, Payload, SavedDocument, WorldPayload};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use store::{StoreConfig, UniverseStore};
use tiny_http::{Response, Server};

/// Everything needed to run a server.
#[derive(Clone, Debug)]
pub struct ServerOptions {
    /// Bind address; port 0 binds an ephemeral port (tests).
    pub addr: String,
    /// Store rules (invite token, content pin, lease TTL).
    pub store: StoreConfig,
    /// R4 request-body byte cap.
    pub size_cap: usize,
    /// World-save path: loaded at start if present, rewritten (atomically)
    /// after every accepted record mutation. `None` ⇒ in-memory only.
    pub save_path: Option<PathBuf>,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:0".to_string(),
            store: StoreConfig::default(),
            size_cap: DEFAULT_SIZE_CAP,
            save_path: None,
        }
    }
}

/// A running server: bound address + shutdown. Dropping the handle without
/// [`ServerHandle::shutdown`] leaves the thread serving until process exit.
pub struct ServerHandle {
    /// The actually-bound address (resolves port 0).
    pub addr: SocketAddr,
    server: Arc<Server>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ServerHandle {
    /// Stops the accept loop and joins the server thread.
    pub fn shutdown(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.server.unblock();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// A startup failure: bind or world-save load.
#[derive(Debug)]
pub enum StartError {
    /// Could not bind the address.
    Bind(String),
    /// The world-save exists but does not load (wrong version/kind/shape) —
    /// loud by design; a corrupt universe must not silently start empty.
    Save(String),
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartError::Bind(e) => write!(f, "bind failed: {e}"),
            StartError::Save(e) => write!(f, "world-save load failed: {e}"),
        }
    }
}

impl std::error::Error for StartError {}

/// Loads a store from `options` (world-save if configured and present, else
/// empty). Public for tests exercising the restart path without a socket.
pub fn load_store(options: &ServerOptions) -> Result<UniverseStore, StartError> {
    match &options.save_path {
        Some(path) if path.exists() => {
            let json = std::fs::read_to_string(path)
                .map_err(|e| StartError::Save(format!("{}: {e}", path.display())))?;
            let doc = SavedDocument::from_json(&json)
                .map_err(|e| StartError::Save(format!("{}: {e}", path.display())))?;
            match doc.payload {
                Payload::WorldSave(w) => Ok(UniverseStore::from_vessels(
                    options.store.clone(),
                    w.vessels,
                )),
                other => Err(StartError::Save(format!(
                    "{}: expected a world_save document, found {:?}",
                    path.display(),
                    other.kind()
                ))),
            }
        }
        _ => Ok(UniverseStore::new(options.store.clone())),
    }
}

/// Serializes the store's record table as a world-save document.
pub fn world_save_json(store: &UniverseStore) -> Result<String, FormatError> {
    SavedDocument::new(Payload::WorldSave(WorldPayload {
        vessels: store.vessels(),
        // The server persists the shared vessel table only; scenario state
        // (WI 553) is a client-side solo-save concern.
        ..Default::default()
    }))
    .to_json()
}

fn write_save(path: &PathBuf, json: &str) {
    // Atomic-enough for LAN scale: write a sibling tmp, then rename over.
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, json).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// Starts the server: binds, loads the world-save (loudly failing on a corrupt
/// one), and serves on a background thread. `now` on the wire is monotonic
/// seconds since start.
pub fn start(options: ServerOptions) -> Result<ServerHandle, StartError> {
    let store = load_store(&options)?;
    let server =
        Arc::new(Server::http(options.addr.as_str()).map_err(|e| StartError::Bind(e.to_string()))?);
    let addr = match server.server_addr() {
        tiny_http::ListenAddr::IP(a) => a,
        #[cfg(unix)]
        tiny_http::ListenAddr::Unix(_) => {
            return Err(StartError::Bind("unix sockets unsupported".into()))
        }
    };
    let stop = Arc::new(AtomicBool::new(false));
    let store = Mutex::new(store);
    let save_path = options.save_path.clone();
    let size_cap = options.size_cap;
    let server2 = server.clone();
    let stop2 = stop.clone();
    let thread = std::thread::spawn(move || {
        let t0 = Instant::now();
        for mut request in server2.incoming_requests() {
            if stop2.load(Ordering::SeqCst) {
                break;
            }
            let now = t0.elapsed().as_secs_f64();
            let method = request.method().as_str().to_owned();
            let url = request.url().to_owned();
            let token = request
                .headers()
                .iter()
                .find(|h| h.field.equiv("x-session-token"))
                .map(|h| h.value.as_str().to_owned());
            let mut body = String::new();
            let _ = request.as_reader().read_to_string(&mut body);
            let mut guard = store.lock().expect("store poisoned");
            let (_, _, cursor_before) = guard.status_counts();
            let (status, payload) = router::handle_request(
                &mut guard,
                now,
                &method,
                &url,
                token.as_deref(),
                &body,
                size_cap,
            );
            // Persist on any accepted record mutation (records are the only
            // durable state; sessions/locks/cursor are runtime-only).
            if let Some(path) = &save_path {
                let (_, _, cursor_after) = guard.status_counts();
                if cursor_after != cursor_before {
                    if let Ok(json) = world_save_json(&guard) {
                        write_save(path, &json);
                    }
                }
            }
            drop(guard);
            let _ = request.respond(Response::from_string(payload).with_status_code(status));
        }
    });
    Ok(ServerHandle {
        addr,
        server,
        stop,
        thread: Some(thread),
    })
}

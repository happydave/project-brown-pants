//! Runtime state/command bus: a minimal HTTP transport over the simulation.
//!
//! `GET /telemetry` returns the current snapshot JSON; `GET /telemetry/history`
//! returns a bounded ring of recent snapshots as a JSON array (WI 644);
//! `POST /command` injects a JSON [`Command`] into the executor (malformed input →
//! HTTP 400). The server
//! runs on its own thread, bridged to Bevy by channels, so network I/O never
//! blocks the sim loop. This is the **runtime** surface — distinct from the
//! dev-gated Bevy Remote Protocol god-mode surface.

use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use sounding_sim::command::Command;
use sounding_sim::diagnostics::ENERGY_DRIFT;
use sounding_sim::sim::{CentralBody, Craft, SimClock};
use sounding_sim::telemetry::{ActiveFlightTelemetry, RoverTelemetry, Telemetry};
use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use tiny_http::{Response, Server};

/// Default port the bus listens on.
pub const DEFAULT_PORT: u16 = 8787;

/// Bounded telemetry history ring (WI 644): the most recent snapshots, oldest-first. `GET /telemetry`
/// serves the newest; `GET /telemetry/history` serves the whole ring as a JSON array. Capacity ×
/// decimation sets the window (≈ `HISTORY_CAP / (60 / HISTORY_DECIMATE)` seconds at 60 fps).
const HISTORY_CAP: usize = 120;
/// Publish one snapshot into the ring every Nth frame (WI 644) — decimates ~60 fps to ~20 Hz so the
/// ring spans several seconds with a bounded payload. `GET /telemetry` (the latest) is unaffected:
/// the newest published snapshot is always the back of the ring.
const HISTORY_DECIMATE: u32 = 3;

/// The telemetry history ring, written by Bevy and read by the server thread.
#[derive(Resource)]
struct BusTelemetry(Arc<Mutex<VecDeque<String>>>);

/// Commands received by the server thread, drained into the executor by Bevy.
#[derive(Resource)]
struct BusCommandRx(Mutex<Receiver<Command>>);

/// Bridge from an active scene to the bus publisher (WI 569): the latest active-craft
/// autonomy snapshot. A scene that owns a `FlightCraft` (e.g. `-- play`, `-- autopilot`)
/// writes it each frame; `publish_telemetry` attaches it. `None` ⇒ orbit-only telemetry.
#[derive(Resource, Default)]
pub struct ActiveFlight(pub Option<ActiveFlightTelemetry>);

/// Bridge from a grounded scene to the bus publisher (WI 640): the latest rover snapshot.
/// A scene that owns a `Rover` (`-- rover`, or the workshop Test driving an assembled rover)
/// writes it each frame; `publish_telemetry` attaches it. `None` ⇒ no rover block.
#[derive(Resource, Default)]
pub struct GroundedRover(pub Option<RoverTelemetry>);

/// Serves the runtime bus on `port`.
pub struct BusPlugin {
    pub port: u16,
}

impl Default for BusPlugin {
    fn default() -> Self {
        Self { port: DEFAULT_PORT }
    }
}

impl Plugin for BusPlugin {
    fn build(&self, app: &mut App) {
        let telemetry = Arc::new(Mutex::new(VecDeque::with_capacity(HISTORY_CAP)));
        let (tx, rx) = mpsc::channel::<Command>();

        match Server::http(("127.0.0.1", self.port)) {
            Ok(server) => {
                let shared = telemetry.clone();
                thread::spawn(move || serve(server, shared, tx));
                info!("bus: listening on http://127.0.0.1:{}", self.port);
            }
            Err(e) => warn!(
                "bus: failed to bind port {} ({e}); continuing without the bus",
                self.port
            ),
        }

        app.insert_resource(BusTelemetry(telemetry))
            .insert_resource(BusCommandRx(Mutex::new(rx)))
            .init_resource::<ActiveFlight>()
            .init_resource::<GroundedRover>()
            .add_systems(Update, (publish_telemetry, drain_commands));
    }
}

/// Blocking server loop (runs on its own thread).
fn serve(server: Server, telemetry: Arc<Mutex<VecDeque<String>>>, tx: Sender<Command>) {
    for mut request in server.incoming_requests() {
        let method = request.method().as_str().to_owned();
        let url = request.url().to_owned();
        let mut body = String::new();
        let _ = request.as_reader().read_to_string(&mut body);
        let (status, payload) = handle_request(&method, &url, &body, &telemetry, &tx);
        let _ = request.respond(Response::from_string(payload).with_status_code(status));
    }
}

/// Pure request router — returns `(status, body)`. Testable without a socket.
fn handle_request(
    method: &str,
    url: &str,
    body: &str,
    telemetry: &Mutex<VecDeque<String>>,
    tx: &Sender<Command>,
) -> (u16, String) {
    match (method, url) {
        ("GET", "/telemetry") => {
            // The newest snapshot (back of the ring), or `{}` when none yet.
            let ring = telemetry.lock().unwrap();
            (200, ring.back().cloned().unwrap_or_else(|| "{}".to_owned()))
        }
        ("GET", "/telemetry/history") => {
            // The whole ring as a JSON array, oldest-first (WI 644). Each element is already a
            // serialized snapshot object, so join them rather than re-serializing.
            let ring = telemetry.lock().unwrap();
            let body = format!("[{}]", ring.iter().cloned().collect::<Vec<_>>().join(","));
            (200, body)
        }
        ("POST", "/command") => match serde_json::from_str::<Command>(body) {
            Ok(cmd) => {
                let _ = tx.send(cmd);
                (200, r#"{"ok":true}"#.to_owned())
            }
            Err(e) => (400, format!(r#"{{"error":"{e}"}}"#)),
        },
        _ => (404, r#"{"error":"not found"}"#.to_owned()),
    }
}

/// Publishes the current authoritative state as telemetry JSON each frame.
#[allow(clippy::too_many_arguments)]
fn publish_telemetry(
    bus: Res<BusTelemetry>,
    clock: Res<SimClock>,
    body: Res<CentralBody>,
    craft: Query<&Craft>,
    diagnostics: Res<DiagnosticsStore>,
    active: Res<ActiveFlight>,
    rover: Res<GroundedRover>,
    mut frame: Local<u32>,
) {
    let orbit = craft.single().ok().map(|c| c.orbit);
    let energy_drift = diagnostics.get(&ENERGY_DRIFT).and_then(|d| d.value());
    let mut snapshot = Telemetry::capture(&clock, orbit.as_ref(), body.mu, energy_drift);
    // Attach the active craft's autonomy state when a scene has published one (WI 569).
    if let Some(a) = active.0 {
        snapshot = snapshot.with_active_flight(a);
    }
    // Attach the grounded rover's state when a rover scene has published one (WI 640).
    if let Some(r) = rover.0.clone() {
        snapshot = snapshot.with_rover(r);
    }
    // Decimate the per-frame publish into the bounded history ring (WI 644) so it spans several
    // seconds without an oversized payload; the newest entry is always `GET /telemetry`.
    *frame = frame.wrapping_add(1);
    if !frame.is_multiple_of(HISTORY_DECIMATE) {
        return;
    }
    if let (Ok(json), Ok(mut ring)) = (serde_json::to_string(&snapshot), bus.0.lock()) {
        if ring.len() == HISTORY_CAP {
            ring.pop_front();
        }
        ring.push_back(json);
    }
}

/// Drains commands received over the bus into the flight-control executor.
fn drain_commands(rx: Res<BusCommandRx>, mut commands: MessageWriter<Command>) {
    if let Ok(rx) = rx.0.lock() {
        while let Ok(cmd) = rx.try_recv() {
            commands.write(cmd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(items: &[&str]) -> Mutex<VecDeque<String>> {
        Mutex::new(items.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn get_telemetry_returns_the_newest_snapshot() {
        // The latest is the back of the ring (WI 644).
        let telemetry = ring(&[r#"{"t":1}"#, r#"{"t":2}"#]);
        let (tx, _rx) = mpsc::channel();
        let (status, body) = handle_request("GET", "/telemetry", "", &telemetry, &tx);
        assert_eq!(status, 200);
        assert_eq!(body, r#"{"t":2}"#);
    }

    #[test]
    fn get_telemetry_is_empty_object_when_no_snapshots() {
        let telemetry = ring(&[]);
        let (tx, _rx) = mpsc::channel();
        let (status, body) = handle_request("GET", "/telemetry", "", &telemetry, &tx);
        assert_eq!(status, 200);
        assert_eq!(body, "{}");
    }

    #[test]
    fn get_telemetry_history_is_a_json_array_oldest_first() {
        let telemetry = ring(&[r#"{"t":1}"#, r#"{"t":2}"#, r#"{"t":3}"#]);
        let (tx, _rx) = mpsc::channel();
        let (status, body) = handle_request("GET", "/telemetry/history", "", &telemetry, &tx);
        assert_eq!(status, 200);
        assert_eq!(body, r#"[{"t":1},{"t":2},{"t":3}]"#);
        // Valid JSON array of the right length.
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 3);
    }

    #[test]
    fn get_telemetry_history_empty_is_empty_array() {
        let telemetry = ring(&[]);
        let (tx, _rx) = mpsc::channel();
        let (status, body) = handle_request("GET", "/telemetry/history", "", &telemetry, &tx);
        assert_eq!(status, 200);
        assert_eq!(body, "[]");
    }

    #[test]
    fn post_valid_command_is_accepted_and_forwarded() {
        let telemetry = ring(&[]);
        let (tx, rx) = mpsc::channel();
        let (status, _) = handle_request("POST", "/command", r#"{"SetWarp":8.0}"#, &telemetry, &tx);
        assert_eq!(status, 200);
        assert_eq!(rx.try_recv().unwrap(), Command::SetWarp(8.0));
    }

    #[test]
    fn post_malformed_command_is_rejected() {
        let telemetry = ring(&[]);
        let (tx, rx) = mpsc::channel();
        let (status, _) = handle_request("POST", "/command", "not json", &telemetry, &tx);
        assert_eq!(status, 400);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn unknown_route_is_404() {
        let telemetry = ring(&[]);
        let (tx, _rx) = mpsc::channel();
        let (status, _) = handle_request("GET", "/nope", "", &telemetry, &tx);
        assert_eq!(status, 404);
    }
}

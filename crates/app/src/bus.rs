//! Runtime state/command bus: a minimal HTTP transport over the simulation.
//!
//! `GET /telemetry` returns the current snapshot JSON; `GET /telemetry/history`
//! returns a bounded ring of recent snapshots as a JSON array (WI 644);
//! `POST /command` injects a JSON [`Command`] into the executor (malformed input →
//! HTTP 400); `GET /screenshot` triggers a framebuffer capture saved to a file the
//! caller reads back (WI 647). The server
//! runs on its own thread, bridged to Bevy by channels, so network I/O never
//! blocks the sim loop. This is the **runtime** surface — distinct from the
//! dev-gated Bevy Remote Protocol god-mode surface.

use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
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

/// Screenshot capture requests received by the server thread (WI 647), drained by Bevy to spawn a
/// framebuffer capture. The image is delivered out-of-band as a **file** at [`ScreenshotPath`] — the
/// MCP reads it — so the sync HTTP thread never has to marshal pixels.
#[derive(Resource)]
struct BusScreenshotRx(Mutex<Receiver<()>>);

/// Absolute path the next screenshot is saved to (WI 647). Returned to the client by `GET /screenshot`
/// so the (same-machine) dev-MCP can read the PNG back.
#[derive(Resource, Clone)]
struct ScreenshotPath(String);

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
        let (shot_tx, shot_rx) = mpsc::channel::<()>();
        // An absolute path next to the working directory, so the (same-machine) MCP can read it back.
        let shot_path = std::env::current_dir()
            .unwrap_or_default()
            .join("sounding-screenshot.png")
            .to_string_lossy()
            .into_owned();

        match Server::http(("127.0.0.1", self.port)) {
            Ok(server) => {
                let shared = telemetry.clone();
                let path = shot_path.clone();
                thread::spawn(move || serve(server, shared, tx, shot_tx, path));
                info!("bus: listening on http://127.0.0.1:{}", self.port);
            }
            Err(e) => warn!(
                "bus: failed to bind port {} ({e}); continuing without the bus",
                self.port
            ),
        }

        app.insert_resource(BusTelemetry(telemetry))
            .insert_resource(BusCommandRx(Mutex::new(rx)))
            .insert_resource(BusScreenshotRx(Mutex::new(shot_rx)))
            .insert_resource(ScreenshotPath(shot_path))
            .init_resource::<ActiveFlight>()
            .init_resource::<GroundedRover>()
            .add_systems(
                Update,
                (publish_telemetry, drain_commands, drain_screenshots),
            );
    }
}

/// Blocking server loop (runs on its own thread).
fn serve(
    server: Server,
    telemetry: Arc<Mutex<VecDeque<String>>>,
    tx: Sender<Command>,
    shot_tx: Sender<()>,
    shot_path: String,
) {
    for mut request in server.incoming_requests() {
        let method = request.method().as_str().to_owned();
        let url = request.url().to_owned();
        let mut body = String::new();
        let _ = request.as_reader().read_to_string(&mut body);
        let (status, payload) =
            handle_request(&method, &url, &body, &telemetry, &tx, &shot_tx, &shot_path);
        let _ = request.respond(Response::from_string(payload).with_status_code(status));
    }
}

/// Pure request router — returns `(status, body)`. Testable without a socket.
#[allow(clippy::too_many_arguments)]
fn handle_request(
    method: &str,
    url: &str,
    body: &str,
    telemetry: &Mutex<VecDeque<String>>,
    tx: &Sender<Command>,
    shot_tx: &Sender<()>,
    shot_path: &str,
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
        ("GET", "/screenshot") => {
            // Best-effort delete the stale file so the client can poll for the fresh capture's
            // reappearance (WI 647), then ask Bevy to capture. The image arrives as a file at
            // `shot_path`; the response carries that path for the (same-machine) reader.
            let _ = std::fs::remove_file(shot_path);
            let _ = shot_tx.send(());
            (
                200,
                format!(r#"{{"ok":true,"path":{}}}"#, json_string(shot_path)),
            )
        }
        _ => (404, r#"{"error":"not found"}"#.to_owned()),
    }
}

/// Minimal JSON string escaping for a filesystem path (quotes + backslashes).
fn json_string(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Drains screenshot requests (WI 647): spawns a primary-window framebuffer capture that saves a PNG
/// to [`ScreenshotPath`] a frame or two later (the capture is async on the render thread).
fn drain_screenshots(rx: Res<BusScreenshotRx>, path: Res<ScreenshotPath>, mut commands: Commands) {
    if let Ok(rx) = rx.0.lock() {
        let mut requested = false;
        while rx.try_recv().is_ok() {
            requested = true; // coalesce bursts into one capture
        }
        if requested {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path.0.clone()));
        }
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

    /// Route a request with throwaway command + screenshot channels (the latter mostly unused).
    fn route(
        method: &str,
        url: &str,
        body: &str,
        telemetry: &Mutex<VecDeque<String>>,
    ) -> (u16, String) {
        let (tx, _rx) = mpsc::channel();
        let (stx, _srx) = mpsc::channel();
        handle_request(method, url, body, telemetry, &tx, &stx, "/tmp/shot.png")
    }

    #[test]
    fn get_telemetry_returns_the_newest_snapshot() {
        // The latest is the back of the ring (WI 644).
        let (status, body) = route(
            "GET",
            "/telemetry",
            "",
            &ring(&[r#"{"t":1}"#, r#"{"t":2}"#]),
        );
        assert_eq!(status, 200);
        assert_eq!(body, r#"{"t":2}"#);
    }

    #[test]
    fn get_telemetry_is_empty_object_when_no_snapshots() {
        let (status, body) = route("GET", "/telemetry", "", &ring(&[]));
        assert_eq!(status, 200);
        assert_eq!(body, "{}");
    }

    #[test]
    fn get_telemetry_history_is_a_json_array_oldest_first() {
        let (status, body) = route(
            "GET",
            "/telemetry/history",
            "",
            &ring(&[r#"{"t":1}"#, r#"{"t":2}"#, r#"{"t":3}"#]),
        );
        assert_eq!(status, 200);
        assert_eq!(body, r#"[{"t":1},{"t":2},{"t":3}]"#);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 3);
    }

    #[test]
    fn get_telemetry_history_empty_is_empty_array() {
        let (status, body) = route("GET", "/telemetry/history", "", &ring(&[]));
        assert_eq!(status, 200);
        assert_eq!(body, "[]");
    }

    #[test]
    fn get_screenshot_signals_capture_and_returns_the_path() {
        let telemetry = ring(&[]);
        let (tx, _rx) = mpsc::channel();
        let (stx, srx) = mpsc::channel();
        let (status, body) = handle_request(
            "GET",
            "/screenshot",
            "",
            &telemetry,
            &tx,
            &stx,
            "/tmp/shot.png",
        );
        assert_eq!(status, 200);
        assert!(srx.try_recv().is_ok(), "a capture request was queued");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["ok"], true);
        assert_eq!(parsed["path"], "/tmp/shot.png");
    }

    #[test]
    fn post_valid_command_is_accepted_and_forwarded() {
        let telemetry = ring(&[]);
        let (tx, rx) = mpsc::channel();
        let (stx, _srx) = mpsc::channel();
        let (status, _) = handle_request(
            "POST",
            "/command",
            r#"{"SetWarp":8.0}"#,
            &telemetry,
            &tx,
            &stx,
            "/tmp/shot.png",
        );
        assert_eq!(status, 200);
        assert_eq!(rx.try_recv().unwrap(), Command::SetWarp(8.0));
    }

    #[test]
    fn post_malformed_command_is_rejected() {
        let (status, _) = route("POST", "/command", "not json", &ring(&[]));
        assert_eq!(status, 400);
    }

    #[test]
    fn unknown_route_is_404() {
        let (status, _) = route("GET", "/nope", "", &ring(&[]));
        assert_eq!(status, 404);
    }
}

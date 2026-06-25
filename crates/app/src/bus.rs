//! Runtime state/command bus: a minimal HTTP transport over the simulation.
//!
//! `GET /telemetry` returns the current snapshot JSON; `POST /command` injects a
//! JSON [`Command`] into the executor (malformed input → HTTP 400). The server
//! runs on its own thread, bridged to Bevy by channels, so network I/O never
//! blocks the sim loop. This is the **runtime** surface — distinct from the
//! dev-gated Bevy Remote Protocol god-mode surface.

use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use sounding_sim::command::Command;
use sounding_sim::diagnostics::ENERGY_DRIFT;
use sounding_sim::sim::{CentralBody, Craft, SimClock};
use sounding_sim::telemetry::{ActiveFlightTelemetry, RoverTelemetry, Telemetry};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use tiny_http::{Response, Server};

/// Default port the bus listens on.
pub const DEFAULT_PORT: u16 = 8787;

/// The latest telemetry JSON, written by Bevy and read by the server thread.
#[derive(Resource)]
struct BusTelemetry(Arc<Mutex<String>>);

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
        let telemetry = Arc::new(Mutex::new(String::from("{}")));
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
fn serve(server: Server, telemetry: Arc<Mutex<String>>, tx: Sender<Command>) {
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
    telemetry: &Mutex<String>,
    tx: &Sender<Command>,
) -> (u16, String) {
    match (method, url) {
        ("GET", "/telemetry") => (200, telemetry.lock().unwrap().clone()),
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
fn publish_telemetry(
    bus: Res<BusTelemetry>,
    clock: Res<SimClock>,
    body: Res<CentralBody>,
    craft: Query<&Craft>,
    diagnostics: Res<DiagnosticsStore>,
    active: Res<ActiveFlight>,
    rover: Res<GroundedRover>,
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
    if let (Ok(json), Ok(mut slot)) = (serde_json::to_string(&snapshot), bus.0.lock()) {
        *slot = json;
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

    #[test]
    fn get_telemetry_returns_the_snapshot() {
        let telemetry = Mutex::new(r#"{"warp":4.0}"#.to_owned());
        let (tx, _rx) = mpsc::channel();
        let (status, body) = handle_request("GET", "/telemetry", "", &telemetry, &tx);
        assert_eq!(status, 200);
        assert!(body.contains("warp"));
    }

    #[test]
    fn post_valid_command_is_accepted_and_forwarded() {
        let telemetry = Mutex::new(String::new());
        let (tx, rx) = mpsc::channel();
        let (status, _) = handle_request("POST", "/command", r#"{"SetWarp":8.0}"#, &telemetry, &tx);
        assert_eq!(status, 200);
        assert_eq!(rx.try_recv().unwrap(), Command::SetWarp(8.0));
    }

    #[test]
    fn post_malformed_command_is_rejected() {
        let telemetry = Mutex::new(String::new());
        let (tx, rx) = mpsc::channel();
        let (status, _) = handle_request("POST", "/command", "not json", &telemetry, &tx);
        assert_eq!(status, 400);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn unknown_route_is_404() {
        let telemetry = Mutex::new(String::new());
        let (tx, _rx) = mpsc::channel();
        let (status, _) = handle_request("GET", "/nope", "", &telemetry, &tx);
        assert_eq!(status, 404);
    }
}

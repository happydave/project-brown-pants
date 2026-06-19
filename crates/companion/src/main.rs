//! Sounding AI companion: an external agent that flies the craft through the bus.
//!
//! It reads `GET /telemetry`, decides via a pluggable [`Brain`], and issues
//! `POST /command` — never touching the simulation directly, reasoning only from
//! telemetry. This increment ships a deterministic navigator (goal: circularize);
//! an LLM-backed brain is a future swap behind the same `Brain` trait.

mod brain;

use brain::{Brain, Decision, NavigatorBrain};
use sounding_sim::command::Command;
use sounding_sim::telemetry::Telemetry;
use std::error::Error;
use std::thread;
use std::time::Duration;

const BUS: &str = "http://127.0.0.1:8787";
const POLL: Duration = Duration::from_millis(250);

fn main() {
    println!("[companion] Online. Watching the ship. Ctrl-C to disconnect.");
    let mut brain = NavigatorBrain;
    let mut last_narration = String::new();

    loop {
        match get_telemetry() {
            Ok(telemetry) => {
                let (narration, command) = match brain.decide(&telemetry) {
                    Decision::Idle(msg) => (msg, None),
                    Decision::Act(cmd, msg) => (msg, Some(cmd)),
                };
                if narration != last_narration {
                    println!("[companion] {narration}");
                    last_narration = narration;
                }
                if let Some(cmd) = command {
                    if let Err(e) = post_command(&cmd) {
                        eprintln!("[companion] command failed: {e}");
                    }
                }
            }
            Err(e) => eprintln!("[companion] no telemetry ({e}); retrying..."),
        }
        thread::sleep(POLL);
    }
}

fn get_telemetry() -> Result<Telemetry, Box<dyn Error>> {
    let body = ureq::get(format!("{BUS}/telemetry"))
        .call()?
        .into_body()
        .read_to_string()?;
    Ok(serde_json::from_str(&body)?)
}

fn post_command(cmd: &Command) -> Result<(), Box<dyn Error>> {
    let json = serde_json::to_string(cmd)?;
    ureq::post(format!("{BUS}/command")).send(json.as_str())?;
    Ok(())
}

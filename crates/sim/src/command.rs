//! Flight control: the deterministic inner loop.
//!
//! A single [`Command`] message is the only way to act on the simulation. Every
//! source — player input, the AI companion, a remote/bus client — emits commands,
//! and one executor applies them; no source mutates simulation state directly.
//! This is what lets the AI remain "a player": it issues the same commands a human
//! does. The command type is also WI 502's bus envelope.
//!
//! Impulsive/on-rails actions only for now; continuous actuators (thrust, gimbal)
//! and attitude/SAS control arrive with the active gear (WI 508).

use crate::orbit::Orbit;
use crate::sim::{Craft, SimClock};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{info, warn};
use glam::DVec2;
use serde::{Deserialize, Serialize};

/// Safe time-warp bounds enforced by the executor — a command source cannot set
/// a warp outside this range.
pub const MIN_WARP: f64 = 0.25;
pub const MAX_WARP: f64 = 256.0;

/// A command to the simulation. Serializable: it is both the in-process message
/// and the bus's wire envelope (WI 502), and the basis for future replay.
#[derive(Message, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Command {
    /// Set the time-warp factor (clamped to `[MIN_WARP, MAX_WARP]`).
    SetWarp(f64),
    /// Pause or resume simulated time.
    SetPaused(bool),
    /// Execute an impulsive maneuver now: a world-frame delta-v applied at the
    /// current instant and craft state (changes velocity, not position).
    ExecuteManeuver { delta_v: DVec2 },
}

/// Applies a single command to the simulation. Pure and deterministic — the only
/// place command semantics live. Returns `true` if applied, `false` if rejected
/// (an unbound maneuver, or a maneuver with no craft).
pub fn apply_command(cmd: &Command, clock: &mut SimClock, orbit: Option<&mut Orbit>) -> bool {
    match *cmd {
        Command::SetWarp(w) => {
            clock.warp = w.clamp(MIN_WARP, MAX_WARP);
            true
        }
        Command::SetPaused(p) => {
            clock.paused = p;
            true
        }
        Command::ExecuteManeuver { delta_v } => {
            match orbit.and_then(|o| o.with_maneuver(clock.time, delta_v).map(|new| (o, new))) {
                Some((o, new)) => {
                    *o = new;
                    true
                }
                None => false,
            }
        }
    }
}

/// Registers the command message and the executor system.
pub struct FlightControlPlugin;

impl Plugin for FlightControlPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Command>()
            .add_systems(Update, execute_commands);
    }
}

fn execute_commands(
    mut reader: MessageReader<Command>,
    mut clock: ResMut<SimClock>,
    mut craft: Query<&mut Craft>,
) {
    let mut craft = craft.single_mut().ok();
    for cmd in reader.read() {
        let applied = apply_command(cmd, &mut clock, craft.as_deref_mut().map(|c| &mut c.orbit));
        match cmd {
            Command::SetWarp(_) => info!("warp: {}x", clock.warp),
            Command::SetPaused(_) => info!("paused: {}", clock.paused),
            Command::ExecuteManeuver { .. } if applied => info!("maneuver executed"),
            Command::ExecuteManeuver { .. } => warn!("maneuver rejected (unbound or no craft)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circular() -> Orbit {
        Orbit::from_state(1.0, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0), 0.0).unwrap()
    }

    #[test]
    fn set_warp_clamps_to_bounds() {
        let mut clock = SimClock::default();
        apply_command(&Command::SetWarp(1000.0), &mut clock, None);
        assert_eq!(clock.warp, MAX_WARP);
        apply_command(&Command::SetWarp(0.001), &mut clock, None);
        assert_eq!(clock.warp, MIN_WARP);
    }

    #[test]
    fn set_paused_applies() {
        let mut clock = SimClock::default();
        apply_command(&Command::SetPaused(true), &mut clock, None);
        assert!(clock.paused);
    }

    #[test]
    fn maneuver_applies_and_unbound_is_rejected() {
        let mut clock = SimClock::default();

        let mut orbit = circular();
        let apo_before = orbit.apoapsis_radius();
        let ok = apply_command(
            &Command::ExecuteManeuver {
                delta_v: DVec2::new(0.0, 0.1),
            },
            &mut clock,
            Some(&mut orbit),
        );
        assert!(ok);
        assert!(orbit.apoapsis_radius() > apo_before + 1e-3);

        let mut orbit2 = circular();
        let snapshot = orbit2;
        let rejected = apply_command(
            &Command::ExecuteManeuver {
                delta_v: DVec2::new(0.0, 5.0),
            },
            &mut clock,
            Some(&mut orbit2),
        );
        assert!(!rejected);
        assert_eq!(
            orbit2, snapshot,
            "rejected maneuver must leave the orbit unchanged"
        );
    }

    #[test]
    fn maneuver_keeps_position_continuous() {
        // Applied at the current instant, an impulsive burn changes velocity, not
        // position — so the craft does not jump.
        let mut clock = SimClock {
            time: 2.3,
            warp: 1.0,
            paused: false,
        };
        let mut orbit = circular();
        let before = orbit.position(clock.time);
        apply_command(
            &Command::ExecuteManeuver {
                delta_v: DVec2::new(0.05, 0.1),
            },
            &mut clock,
            Some(&mut orbit),
        );
        let after = orbit.position(clock.time);
        assert!(
            (after - before).length() < 1e-9,
            "position must be continuous across a burn"
        );
    }

    #[test]
    fn maneuver_without_craft_is_noop() {
        let mut clock = SimClock::default();
        let applied = apply_command(
            &Command::ExecuteManeuver {
                delta_v: DVec2::ZERO,
            },
            &mut clock,
            None,
        );
        assert!(!applied);
    }

    #[test]
    fn command_json_round_trips() {
        for cmd in [
            Command::SetWarp(4.0),
            Command::SetPaused(true),
            Command::ExecuteManeuver {
                delta_v: DVec2::new(0.1, -0.2),
            },
        ] {
            let json = serde_json::to_string(&cmd).unwrap();
            let back: Command = serde_json::from_str(&json).unwrap();
            assert_eq!(cmd, back, "round-trip failed via {json}");
        }
    }

    #[test]
    fn executor_applies_a_queued_command() {
        let mut app = App::new();
        app.insert_resource(SimClock::default());
        app.add_plugins(FlightControlPlugin);
        app.world_mut().write_message(Command::SetWarp(8.0));
        app.update();
        assert_eq!(app.world().resource::<SimClock>().warp, 8.0);
    }
}

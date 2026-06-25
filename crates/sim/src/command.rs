//! Flight control: the deterministic inner loop.
//!
//! A single [`Command`] message is the only way to act on the simulation. Every
//! source — player input, the AI companion, a remote/bus client — emits commands,
//! and one executor applies them; no source mutates simulation state directly.
//! This is what lets the AI remain "a player": it issues the same commands a human
//! does. The command type is also WI 502's bus envelope.
//!
//! WI 508 adds the gear-switch command (`SetGear`, on-rails ↔ active hand-off).
//! Continuous actuators (thrust, gimbal) and attitude/SAS control during active
//! flight remain a later Flight Control concern.

use crate::autopilot::Autopilot;
use crate::control::ControlTier;
use crate::handoff::GearKind;
use crate::orbit::Orbit;
use crate::sim::{Craft, SimClock};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{info, warn};
use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

/// A stability-assist (SAS) mode — the attitude-hold autopilot's intent (WI 533).
/// Defined here (with [`Command`]) so the attitude controller can depend on the
/// command envelope without a cycle.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SasMode {
    /// No assist — only manual attitude intent acts.
    Off,
    /// Damp all rotation toward zero angular rate.
    KillRotation,
    /// Hold the attitude captured when this mode was engaged.
    Hold,
    /// Point the craft's nose axis along a world-frame target direction.
    Point(DVec3),
}

/// Safe time-warp bounds enforced by the executor — a command source cannot set
/// a warp outside this range.
pub const MIN_WARP: f64 = 0.25;
pub const MAX_WARP: f64 = 256.0;

/// Maximum step budget (sim-seconds) the executor will hold at once (WI 643): a single
/// `Command::Step` is clamped into `[0, MAX_STEP_BUDGET]` and queued steps accumulate up to it, so a
/// stray large request cannot run a frozen scene away.
pub const MAX_STEP_BUDGET: f64 = 5.0;

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
    /// Switch the craft into the given gear (on-rails ↔ active) at the current
    /// instant. Handled by the gear-switch system (WI 508), not [`apply_command`]:
    /// the swap is structural (component insert/remove), outside this function's
    /// `(clock, orbit)` reach.
    SetGear(GearKind),
    /// Set the engine throttle (clamped to `[0, 1]`). Applied to propulsion command
    /// state (`crate::propulsion::Propulsion::apply_command`), not [`apply_command`]
    /// — it is propulsion state, outside this function's `(clock, orbit)` reach
    /// (like `SetGear`). WI 531.
    SetThrottle(f64),
    /// Set the engine gimbal deflection (radians, clamped per engine). Applied by
    /// the propulsion system, not [`apply_command`]. WI 531.
    SetGimbal(DVec2),
    /// Manual attitude intent: pitch/yaw/roll about the body axes, each in
    /// `[-1, 1]` (scaled by actuator authority). Applied by the attitude system. WI 533.
    SetAttitude(DVec3),
    /// Set the stability-assist mode. Applied by the attitude system (it captures
    /// the current attitude when entering `Hold`). WI 533.
    SetSas(SasMode),
    /// Set the SAS hold-target re-capture policy (WI 564): `true` re-captures the
    /// attitude when a manual nudge releases (the nudge sticks); `false` returns to
    /// the prior hold target. Applied by the attitude system.
    SetSasRecapture(bool),
    /// Engage (`Some`) or disengage (`None`) a Tier-1 canned autopilot (WI 565).
    /// Engaging requires a powered Tier-1 (`Canned`) computer; applied by the flight
    /// layer (`FlightCraft::apply_command`), not [`apply_command`].
    SetAutopilot(Option<Autopilot>),
    /// Live-tune the SAS PD gains `(kp, kd)` (WI 566). Requires a powered Tier-2
    /// (`Tunable`) computer; applied by the attitude system.
    SetSasGains(f64, f64),
    /// Select a control-tier cap (WI 571): `Some(tier)` operates the craft at
    /// `min(available-given-power, tier)` (a downshift); `None` clears the cap (full
    /// available). Applied by the flight layer (`FlightCraft::apply_command`), not
    /// [`apply_command`] — it is control-system state, outside this function's
    /// `(clock, orbit)` reach.
    SetControlTier(Option<ControlTier>),
    /// Advance a **paused** scene by `seconds` of sim time (WI 643): adds to the
    /// [`SimClock::step_budget`], which the scene step loops consume (frame-bounded) so a frozen
    /// scene can be stepped a known amount for inspection. Clamped to `[0, MAX_STEP_BUDGET]`; ignored
    /// while running. Applied by [`apply_command`] (in the `clock` reach, like `SetPaused`).
    Step { seconds: f64 },
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
            // Toggling pause clears any pending step budget (WI 643) so a stale step can't auto-run
            // the scene on the next pause.
            clock.step_budget = 0.0;
            true
        }
        Command::Step { seconds } => {
            // Accrue the step budget (sim-seconds to advance while paused), bounded.
            clock.step_budget = (clock.step_budget + seconds.max(0.0)).clamp(0.0, MAX_STEP_BUDGET);
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
        // Structural gear switch and propulsion commands are applied by their own
        // systems, not here (outside the `(clock, orbit)` reach).
        Command::SetGear(_)
        | Command::SetThrottle(_)
        | Command::SetGimbal(_)
        | Command::SetAttitude(_)
        | Command::SetSas(_)
        | Command::SetSasRecapture(_)
        | Command::SetAutopilot(_)
        | Command::SetSasGains(..)
        | Command::SetControlTier(_) => false,
    }
}

/// The default keyboard step interval (sim-seconds) emitted by the `.` key (WI 643).
pub const KEY_STEP_SECONDS: f64 = 0.1;

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
            Command::SetGear(g) => info!("gear switch requested: {g:?}"),
            Command::SetThrottle(t) => info!("throttle: {t}"),
            Command::SetGimbal(g) => info!("gimbal: {g:?}"),
            Command::SetAttitude(a) => info!("attitude intent: {a:?}"),
            Command::SetSas(m) => info!("sas: {m:?}"),
            Command::SetSasRecapture(b) => info!("sas recapture-on-release: {b}"),
            Command::SetAutopilot(a) => info!("autopilot: {a:?}"),
            Command::SetSasGains(kp, kd) => info!("sas gains: kp={kp} kd={kd}"),
            Command::SetControlTier(t) => info!("control-tier selection: {t:?}"),
            Command::Step { seconds } => info!("step: +{seconds}s (budget {})", clock.step_budget),
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
    fn maneuver_applies_including_to_a_hyperbolic_escape() {
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

        // A large prograde burn now escapes onto a represented hyperbolic conic
        // (WI 528) — the maneuver applies and the orbit becomes unbound.
        let mut orbit2 = circular();
        let escaped = apply_command(
            &Command::ExecuteManeuver {
                delta_v: DVec2::new(0.0, 5.0),
            },
            &mut clock,
            Some(&mut orbit2),
        );
        assert!(escaped, "escape maneuver applies (hyperbolic conic)");
        assert!(!orbit2.is_bound(), "the resulting orbit is hyperbolic");
        assert!(orbit2.eccentricity > 1.0);
    }

    #[test]
    fn maneuver_keeps_position_continuous() {
        // Applied at the current instant, an impulsive burn changes velocity, not
        // position — so the craft does not jump.
        let mut clock = SimClock {
            time: 2.3,
            warp: 1.0,
            paused: false,
            ..Default::default()
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
    fn step_accrues_clamps_and_clears_on_pause_toggle() {
        let mut clock = SimClock::default();
        // Accrues and clamps to MAX_STEP_BUDGET.
        apply_command(&Command::Step { seconds: 0.1 }, &mut clock, None);
        apply_command(&Command::Step { seconds: 0.2 }, &mut clock, None);
        assert!((clock.step_budget - 0.3).abs() < 1e-12);
        apply_command(&Command::Step { seconds: 100.0 }, &mut clock, None);
        assert_eq!(clock.step_budget, MAX_STEP_BUDGET);
        // A negative request never reduces the budget below the accrued value.
        apply_command(&Command::Step { seconds: -10.0 }, &mut clock, None);
        assert_eq!(clock.step_budget, MAX_STEP_BUDGET);
        // Toggling pause clears the pending budget.
        apply_command(&Command::SetPaused(true), &mut clock, None);
        assert_eq!(clock.step_budget, 0.0);
    }

    #[test]
    fn command_json_round_trips() {
        for cmd in [
            Command::SetWarp(4.0),
            Command::SetPaused(true),
            Command::Step { seconds: 0.25 },
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

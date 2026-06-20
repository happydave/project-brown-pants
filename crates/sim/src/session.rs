//! Game-session state machine (WI 534).
//!
//! Sequences **one craft** through a played session — Build → Launch → Flight →
//! Recovery — with the transitions driven by *simulation state* (lift-off, surface
//! contact), not scripting. The session owns no physics; it reads telemetry-level
//! state and decides the phase. The app swaps the active camera/control-map around
//! the same craft as the phase changes. Headless and pure (testable without
//! rendering).

use serde::{Deserialize, Serialize};

/// Surface-contact speed at or below which a touchdown is a recovery (not a crash), m/s.
pub const SAFE_LANDING_SPEED: f64 = 12.0;

/// The phase of a played session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Building / configuring the craft (on the ground, pre-launch).
    Build,
    /// On the launch pad, awaiting / building thrust.
    Launch,
    /// Free active flight (ascent, coast, orbit, descent).
    Flight,
    /// Down — landed or crashed. Terminal.
    Recovery,
}

/// How a session ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Not yet recovered.
    None,
    /// Touched down at a safe speed.
    Landed,
    /// Hit the surface too fast.
    Crashed,
}

/// A played session: the current phase and its outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameSession {
    /// The current phase.
    pub phase: Phase,
    /// The outcome (set on Recovery).
    pub outcome: Outcome,
}

impl Default for GameSession {
    fn default() -> Self {
        Self {
            phase: Phase::Build,
            outcome: Outcome::None,
        }
    }
}

impl GameSession {
    /// A new session in the Build phase.
    pub fn new() -> Self {
        Self::default()
    }

    /// Move to the launch pad (from Build or already on the pad). A player/session
    /// action, not a sim transition.
    pub fn begin_launch(&mut self) {
        if matches!(self.phase, Phase::Build | Phase::Launch) {
            self.phase = Phase::Launch;
        }
    }

    /// Advance the phase from simulation state: `released` (the craft has left the
    /// pad), `altitude` (above the surface, m), `vertical_speed` (signed, + up),
    /// `speed` (magnitude, for the landing/crash threshold). Lift-off drives
    /// Launch→Flight; a **descending** surface contact drives Flight→Recovery (so a
    /// craft just leaving the pad, ascending through altitude ≈ 0, does not falsely
    /// recover). Recovery is terminal.
    pub fn update(&mut self, released: bool, altitude: f64, vertical_speed: f64, speed: f64) {
        match self.phase {
            Phase::Launch if released => self.phase = Phase::Flight,
            Phase::Flight if altitude <= 0.0 && vertical_speed < 0.0 => {
                self.phase = Phase::Recovery;
                self.outcome = if speed <= SAFE_LANDING_SPEED {
                    Outcome::Landed
                } else {
                    Outcome::Crashed
                };
            }
            _ => {}
        }
    }

    /// Whether the session has ended.
    pub fn is_terminal(&self) -> bool {
        matches!(self.phase, Phase::Recovery)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begins_in_build_then_launches() {
        let mut s = GameSession::new();
        assert_eq!(s.phase, Phase::Build);
        s.begin_launch();
        assert_eq!(s.phase, Phase::Launch);
    }

    #[test]
    fn lift_off_enters_flight_and_does_not_falsely_recover() {
        let mut s = GameSession::new();
        s.begin_launch();
        // On the pad (not released): stays in Launch even at altitude 0.
        s.update(false, 0.0, 0.0, 0.0);
        assert_eq!(s.phase, Phase::Launch);
        // Lift-off: released, ascending through altitude ≈ 0 → Flight, not Recovery.
        s.update(true, 0.0, 5.0, 5.0);
        assert_eq!(s.phase, Phase::Flight);
        // Still ascending at low altitude: stays in Flight (no false recovery).
        s.update(true, 0.5, 8.0, 8.0);
        assert_eq!(s.phase, Phase::Flight);
    }

    #[test]
    fn descending_contact_lands_softly() {
        let mut s = GameSession::new();
        s.begin_launch();
        s.update(true, 0.0, 5.0, 5.0); // Flight
        s.update(true, 1000.0, 50.0, 50.0); // climbing
                                            // Descending, slow touchdown → Landed.
        s.update(true, 0.0, -6.0, 6.0);
        assert_eq!(s.phase, Phase::Recovery);
        assert_eq!(s.outcome, Outcome::Landed);
    }

    #[test]
    fn fast_descending_contact_crashes() {
        let mut s = GameSession::new();
        s.begin_launch();
        s.update(true, 0.0, 5.0, 5.0); // Flight
        s.update(true, 0.0, -120.0, 120.0); // fast descent into the surface
        assert_eq!(s.phase, Phase::Recovery);
        assert_eq!(s.outcome, Outcome::Crashed);
    }

    #[test]
    fn recovery_is_terminal() {
        let mut s = GameSession::new();
        s.begin_launch();
        s.update(true, 0.0, 5.0, 5.0);
        s.update(true, 0.0, -5.0, 5.0); // Landed
        assert!(s.is_terminal());
        // Further updates do nothing.
        s.update(true, 100.0, 10.0, 10.0);
        assert_eq!(s.phase, Phase::Recovery);
        assert_eq!(s.outcome, Outcome::Landed);
    }

    #[test]
    fn no_flight_before_release() {
        let mut s = GameSession::new();
        s.begin_launch();
        for _ in 0..100 {
            s.update(false, 0.0, 0.0, 0.0); // never released
        }
        assert_eq!(s.phase, Phase::Launch);
    }

    #[test]
    fn session_round_trips_through_serde() {
        let s = GameSession {
            phase: Phase::Flight,
            outcome: Outcome::None,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: GameSession = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }
}

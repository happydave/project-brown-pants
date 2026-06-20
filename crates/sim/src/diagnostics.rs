//! Observability harness: records physics-invariant metrics as named Bevy
//! diagnostics. These are automated regression detectors (conservation checks)
//! and double as educational readouts. The harness is read-only over simulation
//! state and rendering-free; WI 502's bus reads these from the diagnostics store.

use crate::orbit::Orbit;
use crate::sim::{Craft, SimClock};
use bevy_app::prelude::*;
use bevy_diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic};
use bevy_ecs::prelude::*;
use glam::DVec2;

/// |specific energy of the live state − the orbit's analytic energy|.
pub const ENERGY_DRIFT: DiagnosticPath = DiagnosticPath::const_new("sim/energy_drift");
/// |specific angular momentum of the live state − the orbit's analytic value|.
pub const ANGULAR_MOMENTUM_DRIFT: DiagnosticPath =
    DiagnosticPath::const_new("sim/angular_momentum_drift");
/// Position/velocity jump injected at an on-rails ↔ active gear transition (WI 508).
/// Registered and recorded by the hand-off plugin (`handoff.rs`), which produces
/// the value at a transition rather than per frame.
pub const HANDOFF_DISCONTINUITY: DiagnosticPath =
    DiagnosticPath::const_new("sim/handoff_discontinuity");

// Future invariant metric, deferred to the system that observes it:
// - wheel contact-query jitter (the kraken detector): WI 506.

/// Absolute deviation of a state's specific orbital energy from the orbit's
/// analytic energy. ≈0 for states on the orbit, strictly positive once the state
/// drifts off it. The state is passed in so the metric is testable against both
/// on-orbit and corrupted states.
pub fn energy_drift_of(orbit: &Orbit, pos: DVec2, vel: DVec2) -> f64 {
    let live = 0.5 * vel.length_squared() - orbit.mu / pos.length();
    (live - orbit.specific_energy()).abs()
}

/// Absolute deviation of a state's specific angular momentum from the orbit's
/// analytic value.
pub fn angular_momentum_drift_of(orbit: &Orbit, pos: DVec2, vel: DVec2) -> f64 {
    let live = pos.x * vel.y - pos.y * vel.x;
    (live - orbit.specific_angular_momentum()).abs()
}

/// Registers the physics-invariant diagnostics and records them each frame.
pub struct SimDiagnosticsPlugin;

impl Plugin for SimDiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        app.register_diagnostic(Diagnostic::new(ENERGY_DRIFT))
            .register_diagnostic(Diagnostic::new(ANGULAR_MOMENTUM_DRIFT))
            .add_systems(Update, record_invariants);
    }
}

fn record_invariants(mut diagnostics: Diagnostics, clock: Res<SimClock>, craft: Query<&Craft>) {
    let Ok(craft) = craft.single() else {
        return;
    };
    let orbit = craft.orbit;
    let (pos, vel) = orbit.position_velocity(clock.time);
    diagnostics.add_measurement(&ENERGY_DRIFT, || energy_drift_of(&orbit, pos, vel));
    diagnostics.add_measurement(&ANGULAR_MOMENTUM_DRIFT, || {
        angular_momentum_drift_of(&orbit, pos, vel)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_diagnostic::DiagnosticsStore;

    fn test_orbit() -> Orbit {
        Orbit::from_state(1.0, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.1), 0.0).unwrap()
    }

    #[test]
    fn on_orbit_state_has_no_drift() {
        let orbit = test_orbit();
        for i in 0..50 {
            let t = i as f64 * 0.2;
            let (pos, vel) = orbit.position_velocity(t);
            assert!(energy_drift_of(&orbit, pos, vel) < 1e-9);
            assert!(angular_momentum_drift_of(&orbit, pos, vel) < 1e-9);
        }
    }

    #[test]
    fn corrupted_state_is_detected() {
        let orbit = test_orbit();
        let (pos, vel) = orbit.position_velocity(0.7);
        // Perturb the state off the orbit; the detector must notice.
        let drift = energy_drift_of(&orbit, pos * 1.1, vel);
        assert!(drift > 1e-3, "drift should be detected: {drift}");
    }

    #[test]
    fn harness_records_into_store() {
        let mut app = App::new();
        app.insert_resource(SimClock::default());
        app.world_mut().spawn(Craft {
            orbit: test_orbit(),
        });
        app.add_plugins(SimDiagnosticsPlugin);
        app.update();

        let store = app.world().resource::<DiagnosticsStore>();
        let value = store.get(&ENERGY_DRIFT).and_then(|d| d.value());
        assert!(
            value.is_some_and(|v| v.is_finite() && v < 1e-9),
            "energy drift not recorded or nonzero: {value:?}"
        );
    }
}

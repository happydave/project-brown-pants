//! The warp-gearbox hand-off (Toy 8, WI 508).
//!
//! The project's trickiest interface: switching a craft between the analytic
//! **on-rails** gear (a Kepler [`Orbit`]) and the numerical **active** gear (an
//! [`ActiveBody`]). Both gears describe the same physical thing as an
//! instantaneous **(position, velocity)** state in the central body's frame, and
//! the two halves of the bridge are **mutual inverses**:
//!
//! - [`wake`] evaluates the orbit's (r, v) at the switch time and builds the
//!   active body from exactly that state — zero injected jump by construction.
//! - [`sleep`] fits a bound conic *through* the active body's current (r, v) — the
//!   conic passes through that state, so again zero injected jump.
//!
//! So the hand-off is clean by construction; a discontinuity arises only from
//! representation mismatch (an out-of-plane state a 2D orbit cannot hold) or
//! non-representability (an unbound state has no bound conic). The
//! [`HandoffDiscontinuity`] metric surfaces exactly those, filling the WI 499
//! `sim/handoff_discontinuity` placeholder.
//!
//! **Planar bridge:** the orbital plane is `z = 0`, so 2D `(x, y)` ↔ 3D
//! `(x, y, 0)` (the convention the active-gear tests already use). Central
//! point-mass gravity preserves the plane, so an in-plane woken body stays
//! in-plane and round-trips losslessly. Headless.

use crate::active::{ActiveBody, Gravity};
use crate::command::Command;
use crate::diagnostics::HANDOFF_DISCONTINUITY;
use crate::orbit::Orbit;
use crate::sim::{Craft, SimClock};
use crate::voxel::MassProperties;
use bevy_app::prelude::*;
use bevy_diagnostic::{Diagnostic, Diagnostics, RegisterDiagnostic};
use bevy_ecs::prelude::*;
use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};

/// Which gear a craft should be in — the payload of a gear-switch [`Command`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GearKind {
    /// The analytic on-rails gear (a Kepler [`Orbit`], component [`Craft`]).
    OnRails,
    /// The numerical active gear (an [`ActiveBody`]).
    Active,
}

/// Per-craft configuration that persists across a gear sojourn. The mass and
/// inertia are needed to wake an active body; the orientation and angular
/// momentum are the rotational state, which the on-rails gear does not carry and
/// which is therefore parked here while on rails and restored on wake.
#[derive(Component, Clone, Copy, Debug)]
pub struct GearState {
    /// Total mass.
    pub mass: f64,
    /// Inertia tensor about the centre of mass, body frame.
    pub inertia: DMat3,
    /// Orientation parked across a rails sojourn (body → world).
    pub orientation: DQuat,
    /// Angular momentum parked across a rails sojourn (world frame).
    pub angular_momentum: DVec3,
}

impl GearState {
    /// A gear-state with the given mass/inertia, at rest orientation and no spin.
    pub fn new(mass: f64, inertia: DMat3) -> Self {
        Self {
            mass,
            inertia,
            orientation: DQuat::IDENTITY,
            angular_momentum: DVec3::ZERO,
        }
    }

    /// A gear-state from a voxel craft's derived mass properties (WI 505).
    pub fn from_mass_properties(mp: &MassProperties) -> Self {
        Self::new(mp.mass, mp.inertia)
    }
}

/// The position and velocity discontinuity injected at a gear transition: the
/// difference between the state in the gear being left and the gear being
/// entered, both evaluated at the switch time. ≈0 for a clean in-plane
/// transition; strictly positive when a real discontinuity exists (out-of-plane
/// projection loss, or a corrupted state).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HandoffDiscontinuity {
    /// Position jump (world units).
    pub position: f64,
    /// Velocity jump (world units per second).
    pub velocity: f64,
}

impl HandoffDiscontinuity {
    /// The larger of the two jumps — a single scalar to threshold/alarm on.
    pub fn magnitude(&self) -> f64 {
        self.position.max(self.velocity)
    }
}

/// Lifts a planar orbit's state at time `t` into 3D world coordinates, with the
/// orbital plane as `z = 0`.
pub fn orbit_state_3d(orbit: &Orbit, t: f64) -> (DVec3, DVec3) {
    let (p, v) = orbit.position_velocity(t);
    (DVec3::new(p.x, p.y, 0.0), DVec3::new(v.x, v.y, 0.0))
}

/// Wake (rails → active): build the active body from the orbit's (r, v) at `t`,
/// embedded in the `z = 0` plane, carrying the parked mass/inertia and rotational
/// state. Continuous by construction — its (r, v) equals the orbit's at `t`.
pub fn wake(orbit: &Orbit, t: f64, gear: &GearState) -> ActiveBody {
    let (pos, vel) = orbit_state_3d(orbit, t);
    let mut body = ActiveBody::new(pos, vel, gear.mass, gear.inertia);
    body.orientation = gear.orientation;
    body.angular_momentum = gear.angular_momentum;
    body
}

/// Sleep (active → rails): fit a bound conic through the active body's (r, v) at
/// `t`, projected to the `z = 0` plane. `None` if the state is unbound
/// (parabolic/hyperbolic) and so cannot be put on rails.
pub fn sleep(body: &ActiveBody, mu: f64, t: f64) -> Option<Orbit> {
    Orbit::from_state(mu, body.position.truncate(), body.velocity.truncate(), t)
}

/// The discontinuity between an `old` (pos, vel) state and a `new` (pos, vel)
/// state — used at a transition with both evaluated at the switch time.
pub fn discontinuity(old: (DVec3, DVec3), new: (DVec3, DVec3)) -> HandoffDiscontinuity {
    HandoffDiscontinuity {
        position: (new.0 - old.0).length(),
        velocity: (new.1 - old.1).length(),
    }
}

/// The most recent gear transition's discontinuity, for the diagnostic readout.
#[derive(Resource, Default, Debug)]
pub struct LastHandoff(pub Option<HandoffDiscontinuity>);

/// Drives the gear hand-off: a [`Command::SetGear`] switches the craft between
/// gears at the clock time (component swap), records the injected discontinuity,
/// and publishes it to the `sim/handoff_discontinuity` diagnostic. Compose
/// alongside [`crate::active::ActivePlugin`] (the woken body is advanced by it)
/// and [`crate::command::FlightControlPlugin`] (the command stream).
pub struct HandoffPlugin;

impl Plugin for HandoffPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LastHandoff>()
            .register_diagnostic(Diagnostic::new(HANDOFF_DISCONTINUITY))
            .add_systems(
                Update,
                (apply_gear_switch, record_handoff_diagnostic).chain(),
            );
    }
}

/// Applies gear-switch commands by swapping the on-rails [`Craft`] and active
/// [`ActiveBody`] components on the craft entity, recording each transition's
/// discontinuity into [`LastHandoff`].
fn apply_gear_switch(
    mut commands: Commands,
    mut reader: MessageReader<Command>,
    clock: Res<SimClock>,
    gravity: Res<Gravity>,
    mut last: ResMut<LastHandoff>,
    mut crafts: Query<(Entity, &mut GearState, Option<&Craft>, Option<&ActiveBody>)>,
) {
    for cmd in reader.read() {
        let Command::SetGear(target) = cmd else {
            continue;
        };
        let t = clock.time;
        for (entity, mut gear, craft, body) in &mut crafts {
            match (target, craft, body) {
                // Wake: rails → active.
                (GearKind::Active, Some(craft), None) => {
                    let old = orbit_state_3d(&craft.orbit, t);
                    let woken = wake(&craft.orbit, t, &gear);
                    last.0 = Some(discontinuity(old, (woken.position, woken.velocity)));
                    commands.entity(entity).remove::<Craft>().insert(woken);
                }
                // Sleep: active → rails (only if the state is bound, I3).
                (GearKind::OnRails, None, Some(body)) => {
                    if let Some(orbit) = sleep(body, gravity.mu, t) {
                        let new = orbit_state_3d(&orbit, t);
                        last.0 = Some(discontinuity((body.position, body.velocity), new));
                        // Park the rotational state the rails gear cannot hold.
                        gear.orientation = body.orientation;
                        gear.angular_momentum = body.angular_momentum;
                        commands
                            .entity(entity)
                            .remove::<ActiveBody>()
                            .insert(Craft { orbit });
                    }
                    // Unbound: stays active.
                }
                // Already in the target gear (or mid-swap this frame): no-op.
                _ => {}
            }
        }
    }
}

/// Publishes the last transition's discontinuity magnitude to the diagnostic.
fn record_handoff_diagnostic(mut diagnostics: Diagnostics, last: Res<LastHandoff>) {
    if let Some(h) = last.0 {
        diagnostics.add_measurement(&HANDOFF_DISCONTINUITY, || h.magnitude());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active::{ActivePlugin, FIXED_DT};
    use crate::command::FlightControlPlugin;
    use crate::sim::{CentralBody, OrbitPlugin};
    use glam::DVec2;

    const MU: f64 = 1.0;

    fn test_orbit() -> Orbit {
        // A mildly eccentric, in-plane, bound orbit.
        Orbit::from_state(MU, DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.1), 0.0).unwrap()
    }

    fn unit_gear() -> GearState {
        GearState::new(1.0, DMat3::IDENTITY)
    }

    // --- Pure bridge: the mutual-inverse round trips (I1) ---

    #[test]
    fn wake_reproduces_orbit_state() {
        let orbit = test_orbit();
        let t = 1.37;
        let body = wake(&orbit, t, &unit_gear());
        let (p, v) = orbit_state_3d(&orbit, t);
        assert!((body.position - p).length() < 1e-12);
        assert!((body.velocity - v).length() < 1e-12);
        // Embedded in the z = 0 plane.
        assert_eq!(body.position.z, 0.0);
        assert_eq!(body.velocity.z, 0.0);
    }

    #[test]
    fn sleep_then_wake_round_trips_state() {
        let orbit = test_orbit();
        let t = 2.5;
        let body = wake(&orbit, t, &unit_gear());
        let orbit2 = sleep(&body, MU, t).expect("bound");
        let (p1, v1) = orbit_state_3d(&orbit, t);
        let (p2, v2) = orbit_state_3d(&orbit2, t);
        assert!((p2 - p1).length() < 1e-9, "position round-trip");
        assert!((v2 - v1).length() < 1e-9, "velocity round-trip");
    }

    #[test]
    fn wake_then_sleep_reproduces_orbit() {
        let orbit = test_orbit();
        let t = 0.0;
        let body = wake(&orbit, t, &unit_gear());
        let back = sleep(&body, MU, t).expect("bound");
        // Same conic: compare a derived, sense-independent quantity set.
        assert!((back.semi_major_axis - orbit.semi_major_axis).abs() < 1e-9);
        assert!((back.eccentricity - orbit.eccentricity).abs() < 1e-9);
        assert!((back.specific_energy() - orbit.specific_energy()).abs() < 1e-9);
    }

    // --- The metric: zero for clean, positive for real discontinuities (I2) ---

    #[test]
    fn clean_transition_metric_is_negligible() {
        let orbit = test_orbit();
        let t = 3.1;
        // Wake: old = orbit state, new = woken body state -> ~0.
        let old = orbit_state_3d(&orbit, t);
        let body = wake(&orbit, t, &unit_gear());
        let d = discontinuity(old, (body.position, body.velocity));
        assert!(d.magnitude() < 1e-12, "wake jump: {d:?}");

        // Sleep of an in-plane body: round-trip residual only.
        let orbit2 = sleep(&body, MU, t).unwrap();
        let new = orbit_state_3d(&orbit2, t);
        let d2 = discontinuity((body.position, body.velocity), new);
        assert!(d2.magnitude() < 1e-9, "sleep jump: {d2:?}");
    }

    #[test]
    fn out_of_plane_state_is_surfaced_by_metric() {
        // A body with an out-of-plane component cannot be held by a planar orbit;
        // the metric must report the lost component, not hide it.
        let orbit = test_orbit();
        let t = 0.0;
        let mut body = wake(&orbit, t, &unit_gear());
        body.position.z = 0.05;
        body.velocity.z = 0.03;
        let orbit2 = sleep(&body, MU, t).expect("in-plane projection is still bound");
        let new = orbit_state_3d(&orbit2, t);
        let d = discontinuity((body.position, body.velocity), new);
        assert!(
            d.magnitude() > 1e-3,
            "out-of-plane discontinuity must be detected: {d:?}"
        );
    }

    #[test]
    fn hyperbolic_active_state_rails_as_a_hyperbolic_conic() {
        // Since WI 528 the on-rails gear represents hyperbolic conics, so a craft
        // boosted past escape speed sleeps onto a (hyperbolic) conic rather than
        // being un-railable. The parabolic knife-edge is still rejected.
        let orbit = test_orbit();
        let mut body = wake(&orbit, 0.0, &unit_gear());
        let escape = (2.0 * MU / body.position.length()).sqrt();
        body.velocity = DVec3::new(0.0, escape + 0.5, 0.0); // hyperbolic
        let railed = sleep(&body, MU, 0.0).expect("hyperbolic state rails");
        assert!(!railed.is_bound() && railed.eccentricity > 1.0);

        // Exactly escape speed (parabolic) remains un-railable.
        body.velocity = DVec3::new(0.0, escape, 0.0);
        assert!(
            sleep(&body, MU, 0.0).is_none(),
            "parabolic edge cannot rail"
        );
    }

    // --- Repeated transitions: non-accumulation (acceptance criterion) ---

    #[test]
    fn repeated_transitions_do_not_accumulate_discontinuity() {
        let mut orbit = test_orbit();
        let gear = unit_gear();
        let mut t = 0.0;
        let mut worst = 0.0_f64;
        for cycle in 0..50 {
            // Wake at t (exact), coast actively a while, sleep, repeat.
            let mut body = wake(&orbit, t, &gear);
            // The wake jump is exactly zero by construction.
            let wake_jump =
                discontinuity(orbit_state_3d(&orbit, t), (body.position, body.velocity))
                    .magnitude();
            assert!(
                wake_jump < 1e-12,
                "wake jump grew at cycle {cycle}: {wake_jump}"
            );

            let steps = 20;
            for _ in 0..steps {
                body.step(MU, FIXED_DT);
                t += FIXED_DT;
            }
            // Sleep: the conic fits the current (in-plane) state; jump ~0 and the
            // body stays in plane (central gravity preserves z = 0).
            let next = sleep(&body, MU, t).expect("still bound after a short coast");
            let sleep_jump =
                discontinuity((body.position, body.velocity), orbit_state_3d(&next, t)).magnitude();
            worst = worst.max(sleep_jump);
            orbit = next;
        }
        assert!(
            worst < 1e-6,
            "per-transition discontinuity exceeded threshold over 50 cycles: {worst}"
        );
    }

    /// WI 527: the hand-off stays clean at **SI / planetary scale** — the wake
    /// reproduces the orbit state exactly (zero injected jump) and the sleep
    /// round-trip residual is tiny *relative* to the 6.6e6 m / 8 km/s magnitudes.
    #[test]
    fn si_scale_handoff_is_clean() {
        const MU_SI: f64 = 3.986e14;
        let r0 = 6_560_000.0;
        let orbit =
            Orbit::from_state(MU_SI, DVec2::new(r0, 0.0), DVec2::new(0.0, 8_200.0), 0.0).unwrap();
        // A real-ish craft mass/inertia (the magnitudes must not matter to the bridge).
        let gear = GearState::new(50_000.0, DMat3::IDENTITY);
        let t = 1_234.5;
        let body = wake(&orbit, t, &gear);
        let d_wake = discontinuity(orbit_state_3d(&orbit, t), (body.position, body.velocity));
        assert!(
            d_wake.magnitude() < 1e-6,
            "SI wake jump must be ~0: {d_wake:?}"
        );

        let orbit2 = sleep(&body, MU_SI, t).expect("bound");
        let d_sleep = discontinuity((body.position, body.velocity), orbit_state_3d(&orbit2, t));
        assert!(
            d_sleep.position < 1e-6 * r0,
            "SI sleep position jump (relative): {d_sleep:?}"
        );
        assert!(
            d_sleep.velocity < 1e-6 * 8_200.0,
            "SI sleep velocity jump (relative): {d_sleep:?}"
        );
    }

    // --- ECS gear switch through a Bevy App (I4, the wiring) ---

    fn handoff_app() -> App {
        let mut app = App::new();
        app.add_plugins(bevy_time::TimePlugin);
        app.add_plugins(OrbitPlugin {
            central_body: CentralBody {
                mu: MU,
                radius: 0.1,
            },
            initial_orbit: test_orbit(),
        });
        app.add_plugins(ActivePlugin { mu: MU });
        app.add_plugins(FlightControlPlugin);
        app.add_plugins(HandoffPlugin);
        app
    }

    fn craft_gear(app: &mut App) -> (bool, bool) {
        let mut q = app
            .world_mut()
            .query::<(Option<&Craft>, Option<&ActiveBody>)>();
        let (c, b) = q.single(app.world()).unwrap();
        (c.is_some(), b.is_some())
    }

    #[test]
    fn set_gear_command_swaps_components_both_ways() {
        let mut app = handoff_app();
        // Starts on rails.
        assert_eq!(craft_gear(&mut app), (true, false));

        // Wake.
        app.world_mut()
            .write_message(Command::SetGear(GearKind::Active));
        app.update();
        assert_eq!(craft_gear(&mut app), (false, true), "should be active");

        // Sleep.
        app.world_mut()
            .write_message(Command::SetGear(GearKind::OnRails));
        app.update();
        assert_eq!(craft_gear(&mut app), (true, false), "should be on rails");

        // A discontinuity was recorded and is negligible.
        let last = app.world().resource::<LastHandoff>();
        assert!(last.0.is_some_and(|h| h.magnitude() < 1e-6));
    }

    #[test]
    fn set_gear_to_current_gear_is_noop() {
        let mut app = handoff_app();
        // Already on rails; requesting rails changes nothing.
        app.world_mut()
            .write_message(Command::SetGear(GearKind::OnRails));
        app.update();
        assert_eq!(craft_gear(&mut app), (true, false));
    }

    #[test]
    fn craft_always_has_exactly_one_gear_across_many_switches() {
        let mut app = handoff_app();
        for i in 0..8 {
            let target = if i % 2 == 0 {
                GearKind::Active
            } else {
                GearKind::OnRails
            };
            app.world_mut().write_message(Command::SetGear(target));
            app.update();
            let (on_rails, active) = craft_gear(&mut app);
            assert!(on_rails ^ active, "exactly one gear at step {i}");
        }
    }
}

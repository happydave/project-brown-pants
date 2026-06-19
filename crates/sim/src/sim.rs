//! Simulation-state ECS layer for the on-rails gear: the simulated-time clock
//! (with time warp) and the craft carried on its [`Orbit`]. Rendering-free; the
//! application crate draws from this state.

use crate::orbit::Orbit;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_time::prelude::*;

/// The simulated-time clock. `time` advances by real frame time scaled by
/// `warp` (unless `paused`). This is the on-rails gear: because the craft's
/// position is a closed-form function of `time`, a large warp step is exact, not
/// an approximation.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SimClock {
    /// Simulated time.
    pub time: f64,
    /// Time-warp factor (simulated seconds per real second).
    pub warp: f64,
    /// When true, simulated time does not advance.
    pub paused: bool,
}

impl Default for SimClock {
    fn default() -> Self {
        Self {
            time: 0.0,
            warp: 1.0,
            paused: false,
        }
    }
}

impl SimClock {
    /// Advances simulated time by `real_dt` real seconds, scaled by `warp`.
    pub fn advance(&mut self, real_dt: f64) {
        if !self.paused {
            self.time += real_dt * self.warp;
        }
    }
}

/// The central attracting body, at the world origin. `radius` is for display
/// and collision sizing only; gravity comes from `mu`.
#[derive(Resource, Debug, Clone, Copy)]
pub struct CentralBody {
    /// Gravitational parameter (μ = G·M).
    pub mu: f64,
    /// Display radius, world units.
    pub radius: f64,
}

/// A craft, carried on its current orbit.
#[derive(Component, Debug, Clone, Copy)]
pub struct Craft {
    pub orbit: Orbit,
}

/// Wires the on-rails simulation: inserts the clock and central body, spawns a
/// single craft on `initial_orbit`, and advances the clock each frame.
pub struct OrbitPlugin {
    pub central_body: CentralBody,
    pub initial_orbit: Orbit,
}

impl Plugin for OrbitPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SimClock::default())
            .insert_resource(self.central_body)
            .add_systems(Update, advance_clock);
        app.world_mut().spawn(Craft {
            orbit: self.initial_orbit,
        });
    }
}

fn advance_clock(time: Res<Time>, mut clock: ResMut<SimClock>) {
    clock.advance(time.delta_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_advances_scaled_by_warp() {
        let mut clock = SimClock {
            warp: 10.0,
            ..Default::default()
        };
        clock.advance(0.5);
        assert_eq!(clock.time, 5.0);
    }

    #[test]
    fn paused_clock_does_not_advance() {
        let mut clock = SimClock {
            time: 3.0,
            warp: 4.0,
            paused: true,
        };
        clock.advance(1.0);
        assert_eq!(clock.time, 3.0);
    }
}

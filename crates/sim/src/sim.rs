//! Simulation-state ECS layer for the on-rails gear: the simulated-time clock
//! (with time warp) and the craft carried on its [`Orbit`]. Rendering-free; the
//! application crate draws from this state.

use crate::handoff::GearState;
use crate::orbit::Orbit;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_time::prelude::*;
use glam::DMat3;

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

/// The central attracting body, at the world origin. `radius` is the surface
/// (sea-level) radius — used for display, altitude, and collision sizing; gravity
/// comes from `mu`. **All SI** (metres, m³/s²): this is the project's canonical
/// unit system (WI 527), paired with [`crate::fluid::FluidMedium`] for the medium.
#[derive(Resource, Debug, Clone, Copy)]
pub struct CentralBody {
    /// Gravitational parameter (μ = G·M), m³/s².
    pub mu: f64,
    /// Surface (sea-level) radius, metres.
    pub radius: f64,
}

impl CentralBody {
    /// The canonical Earth-like body in SI units: μ = G·M ≈ 3.986×10¹⁴ m³/s²
    /// (≈ g·R²) and surface radius ≈ 6.36×10⁶ m. The single source of these
    /// constants for the app, the scenes, and integration tests — paired with
    /// [`crate::fluid::FluidMedium::EARTHLIKE`] (WI 527).
    pub const EARTHLIKE: CentralBody = CentralBody {
        mu: 3.986e14,
        radius: 6_360_000.0,
    };
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
        // The craft starts on rails, carrying a persistent gear-state (a unit
        // body by default) so the WI 508 hand-off can wake it into active physics.
        app.world_mut().spawn((
            Craft {
                orbit: self.initial_orbit,
            },
            GearState::new(1.0, DMat3::IDENTITY),
        ));
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

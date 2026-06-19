//! Sounding simulation core.
//!
//! This crate is the headless, rendering-free heart of the simulation (WI 496).
//! It depends on the Bevy sub-crates (`bevy_app`, `bevy_ecs`) rather than the
//! `bevy` umbrella, so its dependency graph contains no rendering or windowing.
//! The same `SimPlugin` runs headless or inside the windowed application, because
//! `bevy::app::App` is a re-export of `bevy_app::App`.

pub mod diagnostics;
pub mod orbit;
pub mod sim;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

/// Counts simulation ticks. A placeholder proving the simulation schedule
/// advances; later work items replace it with the universe state and the warp
/// gearbox.
#[derive(Resource, Default, Debug)]
pub struct SimTick(pub u64);

/// The simulation core plugin. Rendering-agnostic: it registers only simulation
/// logic, so the same plugin runs headless or in the windowed app.
pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimTick>()
            .add_systems(Update, advance_tick);
    }
}

fn advance_tick(mut tick: ResMut<SimTick>) {
    tick.0 += 1;
}

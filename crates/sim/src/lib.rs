//! Sounding simulation core.
//!
//! This crate is the headless, rendering-free heart of the simulation (WI 496).
//! It depends on the Bevy sub-crates (`bevy_app`, `bevy_ecs`) rather than the
//! `bevy` umbrella, so its dependency graph contains no rendering or windowing.
//! The same `SimPlugin` runs headless or inside the windowed application, because
//! `bevy::app::App` is a re-export of `bevy_app::App`.

pub mod active;
pub mod aero;
pub mod afloat;
pub mod attitude;
pub mod autopilot;
pub mod ballast;
pub mod biome;
pub mod body_asset;
mod body_derive;
pub mod body_digest;
pub mod body_library;
pub mod body_ref;
pub mod bodygen;
pub mod breakage;
pub mod check;
pub mod collision;
pub mod collision_detect;
pub mod command;
pub mod compartments;
pub mod contact;
pub mod contact_surface;
pub mod content;
pub mod control;
pub mod diagnostics;
pub mod director;
pub mod export;
pub mod flight;
pub mod flooding;
pub mod fluid;
pub mod frame;
pub mod handoff;
pub mod launch;
pub mod library;
pub mod marine;
pub mod medium;
pub mod mission;
pub mod orbit;
pub mod panel_mesh;
pub mod persist;
pub mod powertrain;
pub mod propulsion;
pub mod resource;
pub mod rover;
pub mod scenario;
pub mod session;
pub mod shape;
pub mod sim;
pub mod surface;
pub mod surface_field;
pub mod surface_mesh;
pub mod surface_scan;
pub mod system;
pub mod system_library;
pub mod telemetry;
pub mod terrain;
pub mod thermal;
pub mod universe;
pub mod vessel;
pub mod voxel;
pub mod voxel_mesh;
pub mod warp;
pub mod world_save;

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

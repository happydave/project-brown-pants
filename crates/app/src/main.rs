//! Sounding application: the windowed Bevy app that wraps the rendering-free
//! simulation core (`sounding_sim`). Toy 1 renders a 2D orbit with on-rails time
//! warp and an executable maneuver node.
//!
//! Controls:
//! - `.` / `,` — increase / decrease time warp
//! - `Space`   — pause / resume
//! - `M`       — place / clear a maneuver node at the craft's current position
//! - `Up`/`Down` — adjust prograde / retrograde delta-v of the node
//! - `Enter`   — execute the maneuver

use bevy::color::palettes::css;
use bevy::math::{DVec2, Isometry2d};
use bevy::prelude::*;
use sounding_sim::command::{Command, FlightControlPlugin};
use sounding_sim::diagnostics::SimDiagnosticsPlugin;
use sounding_sim::orbit::Orbit;
use sounding_sim::sim::{CentralBody, Craft, OrbitPlugin, SimClock};

mod bus;

/// Pixels per world distance unit.
const SCALE: f32 = 220.0;
const DV_STEP: f64 = 0.02;

fn main() {
    let central_body = CentralBody {
        mu: 1.0,
        radius: 0.08,
    };
    // A mildly eccentric starting orbit so the conic is visibly an ellipse.
    let initial_orbit = Orbit::from_state(
        central_body.mu,
        DVec2::new(1.0, 0.0),
        DVec2::new(0.0, 1.15),
        0.0,
    )
    .expect("initial orbit is bound");

    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(OrbitPlugin {
            central_body,
            initial_orbit,
        })
        .add_plugins(FlightControlPlugin)
        .add_plugins(SimDiagnosticsPlugin)
        .add_plugins(bus::BusPlugin::default())
        .init_resource::<ManeuverPlan>()
        .add_systems(Startup, setup)
        .add_systems(Update, (time_warp_input, maneuver_input, draw));

    #[cfg(feature = "dev")]
    add_dev_tools(&mut app);

    app.run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// A pending maneuver: a delta-v along the velocity direction at a fixed point on
/// the current orbit. Held app-side because it is interaction state, not physics.
#[derive(Resource, Default)]
struct ManeuverPlan {
    active: bool,
    prograde_dv: f64,
}

fn world_to_screen(p: DVec2) -> Vec2 {
    Vec2::new(p.x as f32, p.y as f32) * SCALE
}

/// World-frame delta-v for a prograde (`mag > 0`) or retrograde (`mag < 0`) burn
/// at time `t`. `None` if the velocity is degenerate.
fn prograde_delta_v(orbit: &Orbit, t: f64, mag: f64) -> Option<DVec2> {
    let (_, vel) = orbit.position_velocity(t);
    let dir = vel.normalize_or_zero();
    (dir != DVec2::ZERO).then_some(dir * mag)
}

fn time_warp_input(
    keys: Res<ButtonInput<KeyCode>>,
    clock: Res<SimClock>,
    mut commands: MessageWriter<Command>,
) {
    if keys.just_pressed(KeyCode::Space) {
        commands.write(Command::SetPaused(!clock.paused));
    }
    if keys.just_pressed(KeyCode::Period) {
        commands.write(Command::SetWarp(clock.warp * 2.0));
    }
    if keys.just_pressed(KeyCode::Comma) {
        commands.write(Command::SetWarp(clock.warp / 2.0));
    }
}

fn maneuver_input(
    keys: Res<ButtonInput<KeyCode>>,
    clock: Res<SimClock>,
    mut plan: ResMut<ManeuverPlan>,
    craft: Query<&Craft>,
    mut commands: MessageWriter<Command>,
) {
    let Ok(craft) = craft.single() else {
        return;
    };

    if keys.just_pressed(KeyCode::KeyM) {
        plan.active = !plan.active;
        if plan.active {
            plan.prograde_dv = 0.0;
            info!("maneuver planned (burn now); Up/Down to size, Enter to execute");
        } else {
            info!("maneuver cleared");
        }
    }
    if !plan.active {
        return;
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        plan.prograde_dv += DV_STEP;
        info!("prograde dv: {:+.3}", plan.prograde_dv);
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        plan.prograde_dv -= DV_STEP;
        info!("prograde dv: {:+.3}", plan.prograde_dv);
    }
    if keys.just_pressed(KeyCode::Enter) {
        // Emit a command; the core executor applies it now (or rejects an unbound burn).
        if let Some(delta_v) = prograde_delta_v(&craft.orbit, clock.time, plan.prograde_dv) {
            commands.write(Command::ExecuteManeuver { delta_v });
            plan.active = false;
        }
    }
}

fn draw(
    mut gizmos: Gizmos,
    body: Res<CentralBody>,
    clock: Res<SimClock>,
    plan: Res<ManeuverPlan>,
    craft: Query<&Craft>,
) {
    // Central body.
    gizmos.circle_2d(
        Isometry2d::from_translation(Vec2::ZERO),
        body.radius as f32 * SCALE,
        css::ORANGE,
    );

    let Ok(craft) = craft.single() else {
        return;
    };
    let orbit = &craft.orbit;

    // Current orbit path, closed into a loop.
    gizmos.linestrip_2d(closed_path(orbit), css::GRAY);

    // Craft marker at the current simulated time.
    gizmos.circle_2d(
        Isometry2d::from_translation(world_to_screen(orbit.position(clock.time))),
        6.0,
        css::AQUA,
    );

    // Maneuver node and its predicted orbit.
    if plan.active {
        gizmos.circle_2d(
            Isometry2d::from_translation(world_to_screen(orbit.position(clock.time))),
            5.0,
            css::YELLOW,
        );
        if let Some(preview) = prograde_delta_v(orbit, clock.time, plan.prograde_dv)
            .and_then(|dv| orbit.with_maneuver(clock.time, dv))
        {
            gizmos.linestrip_2d(closed_path(&preview), css::LIME);
        }
    }
}

/// Screen-space points tracing the orbit once around, closed back to the start.
fn closed_path(orbit: &Orbit) -> Vec<Vec2> {
    let mut pts: Vec<Vec2> = orbit
        .sample_path(128)
        .into_iter()
        .map(world_to_screen)
        .collect();
    if let Some(&first) = pts.first() {
        pts.push(first);
    }
    pts
}

/// Registers dev-only tooling. Compiled only under the `dev` feature so that
/// the Bevy Remote Protocol is absent from default and release builds.
#[cfg(feature = "dev")]
fn add_dev_tools(app: &mut App) {
    use bevy::remote::http::RemoteHttpPlugin;
    use bevy::remote::RemotePlugin;

    app.add_plugins(RemotePlugin::default())
        .add_plugins(RemoteHttpPlugin::default());
    info!("dev: Bevy Remote Protocol enabled (HTTP transport)");
}

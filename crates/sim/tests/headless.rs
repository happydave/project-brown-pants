//! Proves the simulation core runs headless with no rendering or display server
//! (WI 496 invariant). Builds a bare app (no rendering/windowing plugins),
//! advances the schedule, and asserts the simulation ticked.

use bevy_app::prelude::*;
use sounding_sim::{SimPlugin, SimTick};

#[test]
fn sim_core_ticks_headless() {
    let mut app = App::new();
    // TaskPoolPlugin alone provides the task pools the ECS scheduler expects;
    // no rendering, windowing, or display server is involved.
    app.add_plugins(TaskPoolPlugin::default());
    app.add_plugins(SimPlugin);

    for _ in 0..5 {
        app.update();
    }

    assert_eq!(
        app.world().resource::<SimTick>().0,
        5,
        "simulation schedule should advance once per update"
    );
}

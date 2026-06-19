//! Sounding application: the windowed Bevy app that wraps the rendering-free
//! simulation core (`sounding_sim`). Rendering and (later) the bus adapters live
//! here; the simulation logic lives in the core crate.

use bevy::prelude::*;
use sounding_sim::SimPlugin;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins).add_plugins(SimPlugin);

    #[cfg(feature = "dev")]
    add_dev_tools(&mut app);

    app.run();
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

//! Shared world-pause control (WI 638).
//!
//! `P` pauses/resumes the running scene. It is **routed through the command bus** — `toggle_pause`
//! emits the inverted [`Command::SetPaused`], the same envelope an external client or the dev MCP
//! would send — rather than flipping an ad-hoc per-scene flag, so pause composes with telemetry and
//! automation. The executor ([`sounding_sim::command::FlightControlPlugin`], added globally in
//! `main`) applies it to the global [`SimClock`]; a scene's step system gates its active physics on
//! `clock.paused`, and its HUD shows [`paused_banner`]. Each scene registers `toggle_pause` in its
//! own `Update` and adds the guard + banner where it steps and draws.

use bevy::prelude::*;
use sounding_sim::command::Command;
use sounding_sim::sim::SimClock;

/// `P` emits the inverted [`Command::SetPaused`] (pause ↔ resume) onto the command bus.
pub(crate) fn toggle_pause(
    keys: Res<ButtonInput<KeyCode>>,
    clock: Res<SimClock>,
    mut commands: MessageWriter<Command>,
) {
    if keys.just_pressed(KeyCode::KeyP) {
        commands.write(Command::SetPaused(!clock.paused));
    }
}

/// A HUD suffix marking the paused state (empty when running), so the freeze is unmistakable.
pub(crate) fn paused_banner(clock: &SimClock) -> &'static str {
    if clock.paused {
        "\n⏸ PAUSED (P)"
    } else {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `P` emits the *inverted* `SetPaused` — the command, not a flag (WI 638). The executor then
    /// applying it to `SimClock` is covered in `sounding_sim::command`. Recorded via a collector so it
    /// is a single deterministic frame.
    #[test]
    fn p_key_emits_inverted_setpaused_through_the_command_bus() {
        #[derive(Resource, Default)]
        struct Seen(Vec<Command>);
        fn record(mut seen: ResMut<Seen>, mut reader: MessageReader<Command>) {
            seen.0.extend(reader.read().copied());
        }

        let emit_for = |paused: bool| {
            let mut app = App::new();
            app.add_message::<Command>()
                .insert_resource(SimClock {
                    paused,
                    ..Default::default()
                })
                .insert_resource(ButtonInput::<KeyCode>::default())
                .init_resource::<Seen>()
                .add_systems(Update, (toggle_pause, record).chain());
            app.world_mut()
                .resource_mut::<ButtonInput<KeyCode>>()
                .press(KeyCode::KeyP);
            app.update();
            app.world().resource::<Seen>().0.clone()
        };

        assert_eq!(emit_for(false), vec![Command::SetPaused(true)]);
        assert_eq!(emit_for(true), vec![Command::SetPaused(false)]);
    }

    /// The banner reflects the clock and is empty while running.
    #[test]
    fn banner_only_shows_when_paused() {
        let mut clock = SimClock::default();
        assert_eq!(paused_banner(&clock), "");
        clock.paused = true;
        assert!(paused_banner(&clock).contains("PAUSED"));
    }
}

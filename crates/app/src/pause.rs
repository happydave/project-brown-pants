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
use sounding_sim::command::{Command, KEY_STEP_SECONDS};
use sounding_sim::sim::SimClock;

/// Per-frame chunk (sim-seconds) of the step budget a paused scene consumes (WI 643). Bounded so a
/// large step plays out over several frames at roughly real-time rather than over-filling a scene's
/// substep accumulator in one frame (which the substep cap would then drop).
const STEP_CHUNK_SECONDS: f64 = 1.0 / 60.0;

/// `P` emits the inverted [`Command::SetPaused`] (pause ↔ resume) onto the command bus.
pub(crate) fn toggle_pause(
    keys: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    pad_map: Res<crate::gamepad::GamepadMap>,
    clock: Res<SimClock>,
    mut commands: MessageWriter<Command>,
    modal: Res<crate::craft_library::CraftLibraryModal>,
) {
    // Don't pause from a stray `P` typed into the craft-library naming prompt (WI 675).
    if modal.is_open() {
        return;
    }
    // P or the gamepad Start button (WI 617).
    if keys.just_pressed(KeyCode::KeyP) || pad_map.sample(&gamepads).pause {
        commands.write(Command::SetPaused(!clock.paused));
    }
}

/// `.` steps a **paused** scene forward a small fixed interval (WI 643), emitting the same
/// [`Command::Step`] a bus/MCP client would — so keyboard and automation share one path. No-op while
/// running (stepping only makes sense when frozen).
pub(crate) fn step_scene(
    keys: Res<ButtonInput<KeyCode>>,
    clock: Res<SimClock>,
    mut commands: MessageWriter<Command>,
) {
    if clock.paused && keys.just_pressed(KeyCode::Period) {
        commands.write(Command::Step {
            seconds: KEY_STEP_SECONDS,
        });
    }
}

/// The sim-time to advance this frame, honouring pause + the step budget (WI 643). `None` ⇒ the scene
/// stays frozen (paused, no pending step); `Some(dt)` ⇒ advance by `dt` — the real frame delta while
/// running, or a bounded chunk of the step budget while paused-and-stepping (consumed here). The
/// scene replaces its `if clock.paused { return }` gate + `time.delta_secs_f64()` with this.
pub(crate) fn frame_step_dt(clock: &mut SimClock, time: &Time) -> Option<f64> {
    if !clock.paused {
        return Some(time.delta_secs_f64());
    }
    if clock.step_budget <= 0.0 {
        return None;
    }
    let chunk = clock.step_budget.min(STEP_CHUNK_SECONDS);
    clock.step_budget -= chunk;
    Some(chunk)
}

/// A HUD suffix marking the paused state (empty when running), so the freeze is unmistakable.
pub(crate) fn paused_banner(clock: &SimClock) -> &'static str {
    if clock.paused {
        "\n⏸ PAUSED (P · . step)"
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
                .init_resource::<crate::gamepad::GamepadMap>()
                .init_resource::<crate::craft_library::CraftLibraryModal>()
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

    /// `frame_step_dt` (WI 643): running → real dt; paused+no budget → None; paused+budget → a bounded
    /// chunk that decrements the budget; the budget drains to zero over enough calls.
    #[test]
    fn frame_step_dt_honours_pause_and_budget() {
        // Running: returns the (zero, in a default Time) frame delta, never None.
        let time = Time::default();
        let mut running = SimClock::default();
        assert!(frame_step_dt(&mut running, &time).is_some());

        // Paused, no budget: frozen.
        let mut paused = SimClock {
            paused: true,
            ..Default::default()
        };
        assert!(frame_step_dt(&mut paused, &time).is_none());

        // Paused with a budget: consumes bounded chunks until drained.
        paused.step_budget = STEP_CHUNK_SECONDS * 2.5;
        let a = frame_step_dt(&mut paused, &time).unwrap();
        assert!(
            (a - STEP_CHUNK_SECONDS).abs() < 1e-12,
            "first chunk is capped"
        );
        let _ = frame_step_dt(&mut paused, &time).unwrap();
        let c = frame_step_dt(&mut paused, &time).unwrap();
        assert!(c <= STEP_CHUNK_SECONDS && c > 0.0, "last partial chunk");
        assert_eq!(paused.step_budget, 0.0, "budget fully drained");
        assert!(frame_step_dt(&mut paused, &time).is_none(), "frozen again");
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

//! Tier-B replay cam (WI 648): record the world transforms of **tagged** entities for the last few
//! seconds into a bounded ring, then freeze the sim and scrub them back — "what just happened?"
//! without determinism or a full-state save. Used by the workshop Test (it has solid mesh entities;
//! the gizmo-drawn `-- rover` scene has none to replay). Drivable from the keyboard (`R` toggle,
//! `[`/`]` scrub) **and** the bus (`POST /replay`), so the assistant can replay-and-screenshot a crash.
//!
//! While in playback the sim is paused and the live mesh-positioning systems are suppressed (a run
//! condition), so the recorded poses are what is shown; leaving playback resumes the sim.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Recorded frames retained (~4 s at 60 fps).
pub const REPLAY_CAP: usize = 240;

/// Marks an entity whose world transform is recorded and replayed (rover chassis/wheels/parts,
/// obstacles, ramp).
#[derive(Component)]
pub struct Replayable;

/// Whether the replay cam is recording live or scrubbing recorded frames.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReplayMode {
    Live,
    Playback,
}

/// The replay ring + scrub cursor. Records `(Entity, Transform)` per frame while live; on playback the
/// cursor selects a recorded frame to re-pose the (still-alive) entities from.
#[derive(Resource)]
pub struct ReplayCam {
    frames: VecDeque<Vec<(Entity, Transform)>>,
    mode: ReplayMode,
    cursor: usize,
}

impl Default for ReplayCam {
    fn default() -> Self {
        Self {
            frames: VecDeque::with_capacity(REPLAY_CAP),
            mode: ReplayMode::Live,
            cursor: 0,
        }
    }
}

impl ReplayCam {
    pub fn is_playback(&self) -> bool {
        self.mode == ReplayMode::Playback
    }
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Record one frame (live only), dropping the oldest past capacity.
    fn record(&mut self, frame: Vec<(Entity, Transform)>) {
        if self.frames.len() == REPLAY_CAP {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    /// Enter playback at the most recent frame (no-op if nothing recorded).
    fn enter(&mut self) {
        if self.frames.is_empty() {
            return;
        }
        self.mode = ReplayMode::Playback;
        self.cursor = self.frames.len() - 1;
    }

    /// Leave playback, resuming live recording.
    fn exit(&mut self) {
        self.mode = ReplayMode::Live;
    }

    /// Move the scrub cursor by `delta` frames, clamped to the recorded range.
    fn scrub(&mut self, delta: i32) {
        if self.frames.is_empty() {
            return;
        }
        let last = (self.frames.len() - 1) as i32;
        self.cursor = (self.cursor as i32 + delta).clamp(0, last) as usize;
    }

    /// Seek to a fraction `[0,1]` of the recorded window.
    fn seek_fraction(&mut self, f: f32) {
        if self.frames.is_empty() {
            return;
        }
        let last = (self.frames.len() - 1) as f32;
        self.cursor = (f.clamp(0.0, 1.0) * last).round() as usize;
    }

    /// The frame under the cursor (playback).
    fn current(&self) -> Option<&Vec<(Entity, Transform)>> {
        self.frames.get(self.cursor)
    }
}

/// A replay control action — the unified envelope for the keyboard and the bus (`POST /replay`), so
/// both drive the cam through one path (mirrors how pause/step share `Command`).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Message)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCommand {
    /// Enter playback at the latest frame and pause the sim.
    Enter,
    /// Leave playback and resume the sim.
    Exit,
    /// Toggle playback.
    Toggle,
    /// Scrub by a number of frames (negative = back).
    Scrub(i32),
    /// Seek to a fraction `[0,1]` of the window.
    Seek(f32),
}

/// Records the tagged entities' transforms each frame while live (WI 648).
pub fn record_replay(mut cam: ResMut<ReplayCam>, q: Query<(Entity, &Transform), With<Replayable>>) {
    if cam.mode != ReplayMode::Live {
        return;
    }
    let frame: Vec<(Entity, Transform)> = q.iter().map(|(e, t)| (e, *t)).collect();
    if !frame.is_empty() {
        cam.record(frame);
    }
}

/// Re-poses the tagged entities from the scrubbed frame while in playback (WI 648). Entities that were
/// despawned since recording are simply skipped.
pub fn apply_replay(cam: Res<ReplayCam>, mut q: Query<&mut Transform, With<Replayable>>) {
    if cam.mode != ReplayMode::Playback {
        return;
    }
    if let Some(frame) = cam.current() {
        for (entity, tf) in frame {
            if let Ok(mut t) = q.get_mut(*entity) {
                *t = *tf;
            }
        }
    }
}

/// Translates the keyboard into [`ReplayCommand`]s (WI 648): `R` toggles playback, `[`/`]` scrub.
pub fn replay_keys(keys: Res<ButtonInput<KeyCode>>, mut out: MessageWriter<ReplayCommand>) {
    if keys.just_pressed(KeyCode::KeyR) {
        out.write(ReplayCommand::Toggle);
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        out.write(ReplayCommand::Scrub(-1));
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        out.write(ReplayCommand::Scrub(1));
    }
}

/// Applies queued [`ReplayCommand`]s to the cam, pausing on enter and resuming on exit (WI 648). Pause
/// is routed through the existing `Command::SetPaused` so it composes with the rest of the bus.
pub fn apply_replay_commands(
    mut reader: MessageReader<ReplayCommand>,
    mut cam: ResMut<ReplayCam>,
    mut pause: MessageWriter<sounding_sim::command::Command>,
) {
    for cmd in reader.read() {
        let was_playback = cam.is_playback();
        match cmd {
            ReplayCommand::Enter => cam.enter(),
            ReplayCommand::Exit => cam.exit(),
            ReplayCommand::Toggle => {
                if cam.is_playback() {
                    cam.exit()
                } else {
                    cam.enter()
                }
            }
            ReplayCommand::Scrub(d) => cam.scrub(*d),
            ReplayCommand::Seek(f) => cam.seek_fraction(*f),
        }
        // Pause when we just entered playback; resume when we just left it.
        if cam.is_playback() && !was_playback {
            pause.write(sounding_sim::command::Command::SetPaused(true));
        } else if !cam.is_playback() && was_playback {
            pause.write(sounding_sim::command::Command::SetPaused(false));
        }
    }
}

/// Registers the replay cam: the ring resource, the [`ReplayCommand`] message, and the record / scrub
/// / apply systems. Global (no-op where nothing is [`Replayable`]); the workshop Test tags entities
/// and gates its mesh positioners on [`live`].
pub struct ReplayPlugin;

impl Plugin for ReplayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ReplayCam>()
            .add_message::<ReplayCommand>()
            .add_systems(
                Update,
                (
                    replay_keys,
                    apply_replay_commands,
                    record_replay,
                    apply_replay,
                )
                    .chain(),
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam_with(n: usize) -> ReplayCam {
        let mut c = ReplayCam::default();
        for _ in 0..n {
            c.record(vec![]);
        }
        c
    }

    #[test]
    fn ring_is_bounded() {
        let c = cam_with(REPLAY_CAP + 50);
        assert_eq!(c.len(), REPLAY_CAP);
    }

    #[test]
    fn enter_seeks_to_latest_then_scrub_clamps() {
        let mut c = cam_with(10);
        c.enter();
        assert!(c.is_playback());
        assert_eq!(c.cursor(), 9);
        c.scrub(-100);
        assert_eq!(c.cursor(), 0, "scrub clamps at the start");
        c.scrub(100);
        assert_eq!(c.cursor(), 9, "scrub clamps at the end");
        c.exit();
        assert!(!c.is_playback());
    }

    #[test]
    fn seek_fraction_maps_to_the_window() {
        let mut c = cam_with(11); // indices 0..=10
        c.enter();
        c.seek_fraction(0.0);
        assert_eq!(c.cursor(), 0);
        c.seek_fraction(0.5);
        assert_eq!(c.cursor(), 5);
        c.seek_fraction(1.0);
        assert_eq!(c.cursor(), 10);
    }

    #[test]
    fn enter_on_empty_is_a_noop() {
        let mut c = ReplayCam::default();
        c.enter();
        assert!(!c.is_playback(), "nothing to replay → stays live");
    }

    #[test]
    fn command_json_round_trips() {
        for cmd in [
            ReplayCommand::Enter,
            ReplayCommand::Toggle,
            ReplayCommand::Scrub(-3),
            ReplayCommand::Seek(0.25),
        ] {
            let j = serde_json::to_string(&cmd).unwrap();
            assert_eq!(cmd, serde_json::from_str(&j).unwrap(), "{j}");
        }
    }
}

//! Game controller (gamepad) input layer (WI 617).
//!
//! A small, rebindable mapping table ([`GamepadMap`]) plus a per-frame [`PadSample`] that the
//! existing keyboard input systems read **in addition** to the keyboard. Gamepad input is strictly
//! additive: with no controller connected the sample is inert (`connected == false`, all fields
//! zero/false) and the keyboard fully governs.
//!
//! All mapping is app-side and feeds the same control intents / `Command` paths the keyboard already
//! uses — there is no simulation change and `sounding_sim` gains no input dependency.
//!
//! Bevy already applies a per-axis deadzone via its gamepad `AxisSettings` before `Gamepad::get`
//! returns a value; [`apply_deadzone`] is an additional, intent-side deadzone so a resting stick
//! reads as exactly zero and the response is proportional and sign-preserving outside it.

use bevy::input::gamepad::{Gamepad, GamepadAxis, GamepadButton};
use bevy::prelude::*;

/// Rebindable assignment of physical controller inputs to logical control intents (WI 617).
///
/// Held as a [`Resource`] seeded with [`GamepadMap::default`]; rebinding is a matter of changing
/// fields in one place rather than at each call site. The physical/logical split (e.g. `steer` and
/// `roll` both default to the left stick X but are distinct bindings) lets them be rebound
/// independently later.
#[derive(Resource, Clone, Copy, Debug)]
pub struct GamepadMap {
    /// Intent-side deadzone applied on top of Bevy's own axis deadzone; in `[0.0, 1.0)`.
    pub deadzone: f32,

    // Sticks (analog, signed −1..1).
    /// Rover steering / flight roll target.
    pub steer: GamepadAxis,
    /// Flight pitch / rover lean.
    pub pitch: GamepadAxis,
    /// Flight roll (defaults to the same physical axis as `steer`).
    pub roll: GamepadAxis,
    /// Build-camera orbit yaw (right stick X).
    pub cam_yaw: GamepadAxis,
    /// Build-camera orbit pitch (right stick Y).
    pub cam_pitch: GamepadAxis,

    // Triggers (analog pressure read via the trigger buttons, 0..1).
    /// Throttle forward / increase.
    pub throttle_fwd: GamepadButton,
    /// Throttle reverse / decrease.
    pub throttle_rev: GamepadButton,

    // Bumpers + buttons (digital). The unified ground/air layout (see
    // `docs/projects/sounding/controller-mapping-research.md`) puts yaw on the bumpers so the right
    // stick is free for the camera, and the rover handbrake on a bumper too; the same physical
    // button means different things per scene, which is fine — each scene reads only its own field.
    /// Flight yaw left (bumper).
    pub yaw_left: GamepadButton,
    /// Flight yaw right (bumper).
    pub yaw_right: GamepadButton,
    /// Rover brake / handbrake (bumper).
    pub brake: GamepadButton,
    /// Toggle SAS hold.
    pub sas_toggle: GamepadButton,
    /// Throttle to maximum (rockets/flight).
    pub throttle_max: GamepadButton,
    /// Throttle to zero (rockets/flight).
    pub throttle_zero: GamepadButton,
    /// Build-camera zoom in (bumper).
    pub cam_zoom_in: GamepadButton,
    /// Build-camera zoom out (bumper).
    pub cam_zoom_out: GamepadButton,
    /// Pause / unpause the scene (Start).
    pub pause: GamepadButton,
    /// Return to the workshop build view (Select).
    pub back: GamepadButton,
}

impl Default for GamepadMap {
    fn default() -> Self {
        Self {
            deadzone: 0.12,
            steer: GamepadAxis::LeftStickX,
            pitch: GamepadAxis::LeftStickY,
            roll: GamepadAxis::LeftStickX,
            cam_yaw: GamepadAxis::RightStickX,
            cam_pitch: GamepadAxis::RightStickY,
            throttle_fwd: GamepadButton::RightTrigger2,
            throttle_rev: GamepadButton::LeftTrigger2,
            // Bumpers carry yaw (air) / handbrake (ground) / zoom (build), contextually.
            yaw_left: GamepadButton::LeftTrigger,
            yaw_right: GamepadButton::RightTrigger,
            brake: GamepadButton::LeftTrigger,
            sas_toggle: GamepadButton::North,
            throttle_max: GamepadButton::East,
            throttle_zero: GamepadButton::West,
            cam_zoom_in: GamepadButton::LeftTrigger,
            cam_zoom_out: GamepadButton::RightTrigger,
            // `Select` is the Xbox "Back/View" button; `Start` is "Menu".
            pause: GamepadButton::Start,
            back: GamepadButton::Select,
        }
    }
}

impl GamepadMap {
    /// Sample the primary (first) connected gamepad into a flat [`PadSample`]. Returns an inert
    /// sample (`connected == false`) when no controller is present, so callers can merge
    /// unconditionally. Additional controllers are ignored — only the first is read, so conflicting
    /// inputs never sum.
    pub fn sample(&self, gamepads: &Query<&Gamepad>) -> PadSample {
        let Some(pad) = gamepads.iter().next() else {
            return PadSample::default();
        };
        let axis = |a: GamepadAxis| apply_deadzone(pad.get(a).unwrap_or(0.0), self.deadzone);
        // Trigger pressure comes through the analog trigger *buttons* on common gamepads.
        let trigger = |b: GamepadButton| apply_deadzone(pad.get(b).unwrap_or(0.0), self.deadzone);

        let fwd = trigger(self.throttle_fwd);
        let rev = trigger(self.throttle_rev);
        // Yaw is digital on the bumpers (unified layout): right − left, in −1..1.
        let yaw =
            pad.pressed(self.yaw_right) as i32 as f32 - pad.pressed(self.yaw_left) as i32 as f32;
        PadSample {
            connected: true,
            steer: axis(self.steer),
            roll: axis(self.roll),
            pitch: axis(self.pitch),
            yaw,
            cam_yaw: axis(self.cam_yaw),
            cam_pitch: axis(self.cam_pitch),
            throttle_fwd: fwd,
            throttle_rev: rev,
            throttle: fwd - rev,
            brake: pad.pressed(self.brake),
            sas_toggle: pad.just_pressed(self.sas_toggle),
            throttle_max: pad.just_pressed(self.throttle_max),
            throttle_zero: pad.just_pressed(self.throttle_zero),
            zoom_in: pad.pressed(self.cam_zoom_in),
            zoom_out: pad.pressed(self.cam_zoom_out),
            pause: pad.just_pressed(self.pause),
            back: pad.just_pressed(self.back),
        }
    }
}

/// A flat, deadzoned snapshot of the primary controller for one frame. Defaults to inert (no
/// controller): every analog field `0.0`, every button `false`, `connected == false`.
#[derive(Default, Clone, Copy, Debug)]
pub struct PadSample {
    /// Whether a controller was connected this frame.
    pub connected: bool,
    /// Steer target, −1..1.
    pub steer: f32,
    /// Roll, −1..1.
    pub roll: f32,
    /// Pitch, −1..1.
    pub pitch: f32,
    /// Yaw, −1..1.
    pub yaw: f32,
    /// Camera orbit yaw rate input, −1..1.
    pub cam_yaw: f32,
    /// Camera orbit pitch rate input, −1..1.
    pub cam_pitch: f32,
    /// Forward throttle trigger, 0..1.
    pub throttle_fwd: f32,
    /// Reverse throttle trigger, 0..1.
    pub throttle_rev: f32,
    /// Combined bipolar throttle (`throttle_fwd - throttle_rev`), −1..1.
    pub throttle: f32,
    /// Brake held.
    pub brake: bool,
    /// SAS toggle pressed this frame.
    pub sas_toggle: bool,
    /// Throttle-to-max pressed this frame.
    pub throttle_max: bool,
    /// Throttle-to-zero pressed this frame.
    pub throttle_zero: bool,
    /// Camera zoom-in held.
    pub zoom_in: bool,
    /// Camera zoom-out held.
    pub zoom_out: bool,
    /// Pause / unpause pressed this frame.
    pub pause: bool,
    /// Back-to-workshop pressed this frame.
    pub back: bool,
}

impl PadSample {
    /// True when the stick deflection is past the deadzone (i.e. the gamepad is the live source for
    /// this axis and should win over the keyboard).
    pub fn active(v: f32) -> bool {
        v.abs() > f32::EPSILON
    }
}

/// Apply a radial deadzone to a single signed axis value and renormalize so the response is
/// continuous and sign-preserving: inputs with magnitude `<= deadzone` read as exactly `0.0`;
/// larger inputs are rescaled so magnitude `deadzone` maps to `0.0` and `1.0` maps to `1.0`, and the
/// result is clamped to `[-1.0, 1.0]`. A `deadzone` outside `[0.0, 1.0)` is treated as `0.0`.
pub fn apply_deadzone(value: f32, deadzone: f32) -> f32 {
    let dz = if (0.0..1.0).contains(&deadzone) {
        deadzone
    } else {
        0.0
    };
    let mag = value.abs();
    if mag <= dz {
        return 0.0;
    }
    let scaled = (mag - dz) / (1.0 - dz);
    scaled.min(1.0).copysign(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadzone_zeroes_small_inputs() {
        assert_eq!(apply_deadzone(0.0, 0.12), 0.0);
        assert_eq!(apply_deadzone(0.05, 0.12), 0.0);
        assert_eq!(apply_deadzone(-0.12, 0.12), 0.0);
    }

    #[test]
    fn deadzone_is_proportional_and_sign_preserving_outside() {
        // Just past the deadzone is near zero, mid-range is positive and < 1, sign follows input.
        let small = apply_deadzone(0.13, 0.12);
        assert!(small > 0.0 && small < 0.05, "got {small}");
        let mid = apply_deadzone(0.56, 0.12);
        assert!(mid > 0.0 && mid < 1.0, "got {mid}");
        assert!(apply_deadzone(-0.56, 0.12) < 0.0);
        // Monotonic: larger magnitude -> larger output.
        assert!(apply_deadzone(0.8, 0.12) > mid);
    }

    #[test]
    fn deadzone_saturates_at_one() {
        assert!((apply_deadzone(1.0, 0.12) - 1.0).abs() < 1e-6);
        assert!((apply_deadzone(-1.0, 0.12) + 1.0).abs() < 1e-6);
        // Overshoot (some pads can report slightly > 1) still clamps.
        assert!((apply_deadzone(1.4, 0.12) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn deadzone_zero_is_identity_clamped() {
        assert!((apply_deadzone(0.5, 0.0) - 0.5).abs() < 1e-6);
        // Out-of-range deadzone falls back to no deadzone rather than dividing by zero.
        assert!((apply_deadzone(0.5, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn default_map_is_sane_and_covers_every_intent() {
        let m = GamepadMap::default();
        // Deadzone is engaged but modest.
        assert!(m.deadzone > 0.0 && m.deadzone < 0.5);
        // Each intent named in the plan has a binding, and the ones that must be physically
        // distinct are distinct (so e.g. throttle and brake aren't the same control).
        assert_ne!(m.throttle_fwd, m.throttle_rev);
        assert_ne!(m.throttle_max, m.throttle_zero);
        assert_ne!(m.cam_zoom_in, m.cam_zoom_out);
        assert_ne!(m.yaw_left, m.yaw_right);
        assert_ne!(m.brake, m.sas_toggle);
        assert_ne!(m.steer, m.pitch);
        assert_ne!(m.pause, m.back);
    }

    #[test]
    fn no_controller_samples_inert() {
        // A PadSample with no live pad contributes nothing.
        let s = PadSample::default();
        assert!(!s.connected);
        assert_eq!(s.throttle, 0.0);
        assert!(!s.brake && !s.sas_toggle);
        assert!(!PadSample::active(s.steer));
    }
}

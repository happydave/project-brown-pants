//! Dev-bridge **input injection** (WI 830): lets a bridge script drive the game's
//! keyboard- and mouse-bound controls (`POST /input`) so scene-interaction checks —
//! mode toggles, brushes, palette clicks, place/remove — become scripted screenshot
//! anchors with no human at the keyboard.
//!
//! **Injection is event-level only** (the load-bearing invariant): the drain
//! synthesizes the same raw Bevy input events winit produces (`KeyboardInput`,
//! `MouseButtonInput`, `MouseWheel`) plus the one piece of window state winit itself
//! writes (the stored cursor position), in `PreUpdate` **before** Bevy's
//! [`InputSystems`] fold them into `ButtonInput`/`AccumulatedMouseScroll`. Nothing
//! writes `ButtonInput` or app state directly, so every consumer — `just_pressed`
//! toggles, the hover raycast, bevy_ui palette interaction, the input-isolation
//! guards (`PointerOnPalette`, the library modal) — sees injected input exactly as it
//! sees real input.
//!
//! The envelope + parser compile unconditionally (inert data, unit-testable); the
//! bus route, channel, and drain registration are **`dev`-feature-gated** in
//! [`crate::bus`] (the WI 496 BRP pattern) — a default/release build serves 404 for
//! `/input` and runs no drain.

use bevy::input::keyboard::Key;
#[cfg(any(test, feature = "dev"))]
use bevy::input::keyboard::KeyboardInput;
#[cfg(any(test, feature = "dev"))]
use bevy::input::mouse::{MouseButtonInput, MouseScrollUnit, MouseWheel};
#[cfg(any(test, feature = "dev"))]
use bevy::input::ButtonState;
use bevy::prelude::*;
#[cfg(any(test, feature = "dev"))]
use bevy::window::PrimaryWindow;
use serde::{Deserialize, Serialize};
#[cfg(any(test, feature = "dev"))]
use std::sync::mpsc::Receiver;
#[cfg(any(test, feature = "dev"))]
use std::sync::Mutex;

/// What to do with a key or mouse button.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KeyAction {
    /// Press and hold (until an explicit `release`).
    Press,
    /// Release a held key/button.
    Release,
    /// Press this frame, auto-release the next — one `just_pressed` +
    /// `just_released` cycle, the shape every toggle consumes.
    #[default]
    Tap,
}

/// One injected input action — the `POST /input` envelope (WI 830). Coordinates are
/// **logical** window pixels, origin top-left (the `Window::cursor_position`
/// convention the hover raycast and bevy_ui read).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Message)]
#[serde(rename_all = "snake_case")]
pub enum InputCommand {
    /// A named key (see [`parse_key`]): single characters (`"a"`–`"z"`, `"0"`–`"9"`)
    /// or named specials (`"enter"`, `"tab"`, `"escape"`, `"space"`, `"backspace"`,
    /// `"shift"`, `"ctrl"`, arrows, `"comma"`, `"period"`, `"minus"`, `"f1"`–`"f12"`).
    Key {
        key: String,
        #[serde(default)]
        action: KeyAction,
    },
    /// Move the cursor to logical window coordinates.
    CursorMove { x: f32, y: f32 },
    /// A mouse button (`"left"`, `"right"`, `"middle"`).
    MouseButton {
        button: String,
        #[serde(default)]
        action: KeyAction,
    },
    /// Convenience: move the cursor **and** tap a button (default left) — the
    /// "place a block here / pick this palette entry" primitive.
    Click {
        x: f32,
        y: f32,
        #[serde(default)]
        button: Option<String>,
    },
    /// Vertical wheel scroll in lines (positive = away/zoom-in, the wheel
    /// convention).
    Scroll { lines: f32 },
}

/// Validate a command's key/button names — used by the bus route so an unknown name
/// is a 400 to the caller, never a silent no-op in the drain.
pub fn validate(cmd: &InputCommand) -> Result<(), String> {
    match cmd {
        InputCommand::Key { key, .. } => parse_key(key).map(|_| ()),
        InputCommand::MouseButton { button, .. } => parse_button(button).map(|_| ()),
        InputCommand::Click {
            button: Some(b), ..
        } => parse_button(b).map(|_| ()),
        _ => Ok(()),
    }
}

/// Parse a key name into its physical [`KeyCode`] + logical [`Key`] pair (layout
/// assumed US/QWERTY — the injection convention, matching the physical bindings the
/// app documents). Pure; unknown names are an `Err` naming the offender.
pub fn parse_key(name: &str) -> Result<(KeyCode, Key), String> {
    let lower = name.to_ascii_lowercase();
    let mut chars = lower.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        if let Some(code) = char_code(c) {
            return Ok((code, Key::Character(c.to_string().into())));
        }
    }
    let named = match lower.as_str() {
        "enter" | "return" => (KeyCode::Enter, Key::Enter),
        "tab" => (KeyCode::Tab, Key::Tab),
        "escape" | "esc" => (KeyCode::Escape, Key::Escape),
        "space" => (KeyCode::Space, Key::Space),
        "backspace" => (KeyCode::Backspace, Key::Backspace),
        "shift" => (KeyCode::ShiftLeft, Key::Shift),
        "ctrl" | "control" => (KeyCode::ControlLeft, Key::Control),
        "up" => (KeyCode::ArrowUp, Key::ArrowUp),
        "down" => (KeyCode::ArrowDown, Key::ArrowDown),
        "left" => (KeyCode::ArrowLeft, Key::ArrowLeft),
        "right" => (KeyCode::ArrowRight, Key::ArrowRight),
        "comma" => (KeyCode::Comma, Key::Character(",".into())),
        "period" => (KeyCode::Period, Key::Character(".".into())),
        "minus" => (KeyCode::Minus, Key::Character("-".into())),
        "f1" => (KeyCode::F1, Key::F1),
        "f2" => (KeyCode::F2, Key::F2),
        "f3" => (KeyCode::F3, Key::F3),
        "f4" => (KeyCode::F4, Key::F4),
        "f5" => (KeyCode::F5, Key::F5),
        "f6" => (KeyCode::F6, Key::F6),
        "f7" => (KeyCode::F7, Key::F7),
        "f8" => (KeyCode::F8, Key::F8),
        "f9" => (KeyCode::F9, Key::F9),
        "f10" => (KeyCode::F10, Key::F10),
        "f11" => (KeyCode::F11, Key::F11),
        "f12" => (KeyCode::F12, Key::F12),
        _ => return Err(format!("unknown key: {name}")),
    };
    Ok(named)
}

/// The physical key code for a single character, if it is a letter or digit.
fn char_code(c: char) -> Option<KeyCode> {
    Some(match c {
        'a' => KeyCode::KeyA,
        'b' => KeyCode::KeyB,
        'c' => KeyCode::KeyC,
        'd' => KeyCode::KeyD,
        'e' => KeyCode::KeyE,
        'f' => KeyCode::KeyF,
        'g' => KeyCode::KeyG,
        'h' => KeyCode::KeyH,
        'i' => KeyCode::KeyI,
        'j' => KeyCode::KeyJ,
        'k' => KeyCode::KeyK,
        'l' => KeyCode::KeyL,
        'm' => KeyCode::KeyM,
        'n' => KeyCode::KeyN,
        'o' => KeyCode::KeyO,
        'p' => KeyCode::KeyP,
        'q' => KeyCode::KeyQ,
        'r' => KeyCode::KeyR,
        's' => KeyCode::KeyS,
        't' => KeyCode::KeyT,
        'u' => KeyCode::KeyU,
        'v' => KeyCode::KeyV,
        'w' => KeyCode::KeyW,
        'x' => KeyCode::KeyX,
        'y' => KeyCode::KeyY,
        'z' => KeyCode::KeyZ,
        '0' => KeyCode::Digit0,
        '1' => KeyCode::Digit1,
        '2' => KeyCode::Digit2,
        '3' => KeyCode::Digit3,
        '4' => KeyCode::Digit4,
        '5' => KeyCode::Digit5,
        '6' => KeyCode::Digit6,
        '7' => KeyCode::Digit7,
        '8' => KeyCode::Digit8,
        '9' => KeyCode::Digit9,
        _ => return None,
    })
}

/// Parse a mouse-button name. Pure.
pub fn parse_button(name: &str) -> Result<MouseButton, String> {
    match name.to_ascii_lowercase().as_str() {
        "left" => Ok(MouseButton::Left),
        "right" => Ok(MouseButton::Right),
        "middle" => Ok(MouseButton::Middle),
        _ => Err(format!("unknown mouse button: {name}")),
    }
}

/// Injected commands received by the bus server thread, drained by [`drain_input`].
/// The drain machinery below compiles only for dev builds (and tests) — a default or
/// release build carries no injection capability at all (the workitem's gate).
#[cfg(any(test, feature = "dev"))]
#[derive(Resource)]
pub struct InputInjectRx(pub Mutex<Receiver<InputCommand>>);

/// Auto-releases queued by `tap` actions, emitted at the start of the **next** drain
/// run so a tap yields one full `just_pressed` → `just_released` cycle.
#[cfg(any(test, feature = "dev"))]
#[derive(Resource, Default)]
pub struct PendingReleases {
    keys: Vec<(KeyCode, Key)>,
    buttons: Vec<MouseButton>,
}

/// The `KeyboardInput` event for a parsed key: on press, character keys (and space)
/// carry their produced text, exactly as winit reports them.
#[cfg(any(test, feature = "dev"))]
fn key_event(code: KeyCode, logical: Key, state: ButtonState, window: Entity) -> KeyboardInput {
    let text = match (&state, &logical) {
        (ButtonState::Pressed, Key::Character(s)) => Some(s.clone()),
        (ButtonState::Pressed, Key::Space) => Some(" ".into()),
        _ => None,
    };
    KeyboardInput {
        key_code: code,
        logical_key: logical,
        state,
        text,
        repeat: false,
        window,
    }
}

/// Drains injected commands into **real input events** on the primary window.
/// Registered (dev builds only) in `PreUpdate` **before** [`InputSystems`], so
/// Bevy's own folding systems process the events this same frame. Pending tap
/// releases flush first; a re-press of a still-pending key/button flushes its
/// release inline so back-to-back taps in one batch don't collapse.
#[cfg(any(test, feature = "dev"))]
pub fn drain_input(
    rx: Res<InputInjectRx>,
    mut pending: ResMut<PendingReleases>,
    mut keys: MessageWriter<KeyboardInput>,
    mut mouse: MessageWriter<MouseButtonInput>,
    mut wheel: MessageWriter<MouseWheel>,
    mut windows: Query<(Entity, &mut Window), With<PrimaryWindow>>,
) {
    let Ok((window, mut win)) = windows.single_mut() else {
        return; // no primary window: drop injected input safely
    };
    for (code, logical) in pending.keys.drain(..) {
        keys.write(key_event(code, logical, ButtonState::Released, window));
    }
    for button in pending.buttons.drain(..) {
        mouse.write(MouseButtonInput {
            button,
            state: ButtonState::Released,
            window,
        });
    }
    let Ok(rx) = rx.0.lock() else { return };
    while let Ok(cmd) = rx.try_recv() {
        match cmd {
            InputCommand::Key { key, action } => {
                // Route-validated, so parse cannot fail; skip defensively if it does.
                let Ok((code, logical)) = parse_key(&key) else {
                    continue;
                };
                match action {
                    KeyAction::Press => {
                        keys.write(key_event(code, logical, ButtonState::Pressed, window));
                    }
                    KeyAction::Release => {
                        keys.write(key_event(code, logical, ButtonState::Released, window));
                    }
                    KeyAction::Tap => {
                        // A same-batch re-tap flushes the pending release first.
                        if let Some(i) = pending.keys.iter().position(|(c, _)| *c == code) {
                            let (c, l) = pending.keys.remove(i);
                            keys.write(key_event(c, l, ButtonState::Released, window));
                        }
                        keys.write(key_event(
                            code,
                            logical.clone(),
                            ButtonState::Pressed,
                            window,
                        ));
                        pending.keys.push((code, logical));
                    }
                }
            }
            InputCommand::CursorMove { x, y } => {
                win.set_cursor_position(Some(Vec2::new(x, y)));
            }
            InputCommand::MouseButton { button, action } => {
                let Ok(button) = parse_button(&button) else {
                    continue;
                };
                write_button(&mut mouse, &mut pending, button, action, window);
            }
            InputCommand::Click { x, y, button } => {
                win.set_cursor_position(Some(Vec2::new(x, y)));
                let button = button
                    .as_deref()
                    .map(parse_button)
                    .transpose()
                    .unwrap_or(Some(MouseButton::Left)) // route-validated
                    .unwrap_or(MouseButton::Left);
                write_button(&mut mouse, &mut pending, button, KeyAction::Tap, window);
            }
            InputCommand::Scroll { lines } => {
                wheel.write(MouseWheel {
                    unit: MouseScrollUnit::Line,
                    x: 0.0,
                    y: lines,
                    window,
                });
            }
        }
    }
}

/// Emits a mouse-button action (shared by `MouseButton` and `Click`), with the same
/// tap bookkeeping as keys.
#[cfg(any(test, feature = "dev"))]
fn write_button(
    mouse: &mut MessageWriter<MouseButtonInput>,
    pending: &mut PendingReleases,
    button: MouseButton,
    action: KeyAction,
    window: Entity,
) {
    match action {
        KeyAction::Press => {
            mouse.write(MouseButtonInput {
                button,
                state: ButtonState::Pressed,
                window,
            });
        }
        KeyAction::Release => {
            mouse.write(MouseButtonInput {
                button,
                state: ButtonState::Released,
                window,
            });
        }
        KeyAction::Tap => {
            if let Some(i) = pending.buttons.iter().position(|b| *b == button) {
                pending.buttons.remove(i);
                mouse.write(MouseButtonInput {
                    button,
                    state: ButtonState::Released,
                    window,
                });
            }
            mouse.write(MouseButtonInput {
                button,
                state: ButtonState::Pressed,
                window,
            });
            pending.buttons.push(button);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::input::mouse::AccumulatedMouseScroll;
    use bevy::input::InputSystems;
    use std::sync::mpsc::{self, Sender};

    /// A headless app with the real input plugin, a primary window, and the drain
    /// ordered before [`InputSystems`] — the shipped wiring, minus the HTTP thread.
    fn test_app() -> (App, Sender<InputCommand>) {
        let (tx, rx) = mpsc::channel();
        let mut app = App::new();
        app.add_plugins(bevy::input::InputPlugin)
            .insert_resource(InputInjectRx(Mutex::new(rx)))
            .init_resource::<PendingReleases>()
            .add_systems(PreUpdate, drain_input.before(InputSystems));
        app.world_mut().spawn((Window::default(), PrimaryWindow));
        (app, tx)
    }

    #[test]
    fn key_names_parse_and_unknown_is_an_error() {
        assert_eq!(parse_key("t").unwrap().0, KeyCode::KeyT);
        assert_eq!(parse_key("T").unwrap().0, KeyCode::KeyT, "case-insensitive");
        assert_eq!(parse_key("7").unwrap().0, KeyCode::Digit7);
        assert_eq!(parse_key("enter").unwrap(), (KeyCode::Enter, Key::Enter));
        assert_eq!(parse_key("space").unwrap().0, KeyCode::Space);
        assert_eq!(parse_key("f3").unwrap().0, KeyCode::F3);
        assert!(parse_key("hyperspace").is_err());
        assert!(parse_button("left").is_ok());
        assert!(parse_button("fourth").is_err());
        // validate() surfaces the same errors for the route's 400.
        assert!(validate(&InputCommand::Key {
            key: "enter".into(),
            action: KeyAction::Tap
        })
        .is_ok());
        assert!(validate(&InputCommand::Key {
            key: "hyperspace".into(),
            action: KeyAction::Tap
        })
        .is_err());
        assert!(validate(&InputCommand::Click {
            x: 1.0,
            y: 1.0,
            button: Some("fourth".into())
        })
        .is_err());
    }

    #[test]
    fn command_json_round_trips_and_action_defaults_to_tap() {
        let cmds = vec![
            InputCommand::Key {
                key: "enter".into(),
                action: KeyAction::Tap,
            },
            InputCommand::CursorMove { x: 12.0, y: 34.0 },
            InputCommand::MouseButton {
                button: "left".into(),
                action: KeyAction::Press,
            },
            InputCommand::Click {
                x: 40.0,
                y: 300.0,
                button: None,
            },
            InputCommand::Scroll { lines: -2.0 },
        ];
        for c in cmds {
            let j = serde_json::to_string(&c).unwrap();
            assert_eq!(c, serde_json::from_str(&j).unwrap(), "{j}");
        }
        // Omitted action ⇒ tap (the common script form: {"key":{"key":"enter"}}).
        let c: InputCommand = serde_json::from_str(r#"{"key":{"key":"enter"}}"#).unwrap();
        assert_eq!(
            c,
            InputCommand::Key {
                key: "enter".into(),
                action: KeyAction::Tap
            }
        );
    }

    #[test]
    fn tap_is_just_pressed_then_just_released_across_frames() {
        let (mut app, tx) = test_app();
        tx.send(InputCommand::Key {
            key: "enter".into(),
            action: KeyAction::Tap,
        })
        .unwrap();
        app.update();
        let keys = app.world().resource::<ButtonInput<KeyCode>>();
        assert!(
            keys.just_pressed(KeyCode::Enter),
            "pressed on the tap frame"
        );
        app.update();
        let keys = app.world().resource::<ButtonInput<KeyCode>>();
        assert!(
            keys.just_released(KeyCode::Enter),
            "auto-released next frame"
        );
        app.update();
        let keys = app.world().resource::<ButtonInput<KeyCode>>();
        assert!(!keys.pressed(KeyCode::Enter));
    }

    #[test]
    fn press_holds_until_release() {
        let (mut app, tx) = test_app();
        tx.send(InputCommand::Key {
            key: "w".into(),
            action: KeyAction::Press,
        })
        .unwrap();
        app.update();
        app.update();
        assert!(
            app.world()
                .resource::<ButtonInput<KeyCode>>()
                .pressed(KeyCode::KeyW),
            "held across frames"
        );
        tx.send(InputCommand::Key {
            key: "w".into(),
            action: KeyAction::Release,
        })
        .unwrap();
        app.update();
        assert!(app
            .world()
            .resource::<ButtonInput<KeyCode>>()
            .just_released(KeyCode::KeyW));
    }

    #[test]
    fn character_keys_carry_logical_text_for_the_name_prompt_path() {
        // The craft-library prompt reads logical `Key::Character` values from
        // `KeyboardInput` events (WI 675) — injected characters must feed it.
        let (mut app, tx) = test_app();
        tx.send(InputCommand::Key {
            key: "k".into(),
            action: KeyAction::Tap,
        })
        .unwrap();
        app.update();
        let events = app.world().resource::<Messages<KeyboardInput>>();
        let pressed: Vec<&KeyboardInput> = events
            .iter_current_update_messages()
            .filter(|e| e.state == ButtonState::Pressed)
            .collect();
        assert_eq!(pressed.len(), 1);
        assert_eq!(pressed[0].logical_key, Key::Character("k".into()));
        assert_eq!(pressed[0].text.as_deref(), Some("k"));
        assert!(!pressed[0].repeat);
    }

    #[test]
    fn click_moves_the_cursor_and_taps_the_button() {
        let (mut app, tx) = test_app();
        tx.send(InputCommand::Click {
            x: 40.0,
            y: 300.0,
            button: None,
        })
        .unwrap();
        app.update();
        let window = app
            .world_mut()
            .query_filtered::<&Window, With<PrimaryWindow>>()
            .single(app.world())
            .unwrap();
        assert_eq!(window.cursor_position(), Some(Vec2::new(40.0, 300.0)));
        assert!(app
            .world()
            .resource::<ButtonInput<MouseButton>>()
            .just_pressed(MouseButton::Left));
        app.update();
        assert!(app
            .world()
            .resource::<ButtonInput<MouseButton>>()
            .just_released(MouseButton::Left));
    }

    #[test]
    fn scroll_accumulates_wheel_lines() {
        let (mut app, tx) = test_app();
        tx.send(InputCommand::Scroll { lines: -2.5 }).unwrap();
        app.update();
        let scroll = app.world().resource::<AccumulatedMouseScroll>();
        assert_eq!(scroll.delta.y, -2.5);
    }

    #[test]
    fn same_batch_double_tap_does_not_collapse() {
        // Two taps of one key in a single drained batch: the pending release is
        // flushed between the presses, so both taps register.
        let (mut app, tx) = test_app();
        for _ in 0..2 {
            tx.send(InputCommand::Key {
                key: "t".into(),
                action: KeyAction::Tap,
            })
            .unwrap();
        }
        app.update();
        let events = app.world().resource::<Messages<KeyboardInput>>();
        let states: Vec<ButtonState> = events
            .iter_current_update_messages()
            .map(|e| e.state)
            .collect();
        assert_eq!(
            states,
            vec![
                ButtonState::Pressed,
                ButtonState::Released,
                ButtonState::Pressed
            ],
            "press / flushed release / press"
        );
        app.update();
        assert!(app
            .world()
            .resource::<ButtonInput<KeyCode>>()
            .just_released(KeyCode::KeyT));
    }

    #[test]
    fn no_primary_window_drops_input_safely() {
        let (tx, rx) = mpsc::channel();
        let mut app = App::new();
        app.add_plugins(bevy::input::InputPlugin)
            .insert_resource(InputInjectRx(Mutex::new(rx)))
            .init_resource::<PendingReleases>()
            .add_systems(PreUpdate, drain_input.before(InputSystems));
        tx.send(InputCommand::Key {
            key: "enter".into(),
            action: KeyAction::Tap,
        })
        .unwrap();
        app.update(); // must not panic
        assert!(!app
            .world()
            .resource::<ButtonInput<KeyCode>>()
            .pressed(KeyCode::Enter));
    }
}

//! Craft save library — editor UI (WI 675).
//!
//! Turns the single fixed-file quick-save into a **named library** of vehicles. The
//! durable storage and discovery live headless in [`sounding_sim::library`]; this
//! module is the Build-mode UI over it: a modal that is either a **naming** prompt
//! (type a name, save) or a **browser** (pick a saved vehicle, load it into Build).
//!
//! While the modal is open it owns input — the build systems guard on
//! [`CraftLibraryModal::is_open`] and skip a frame so typing/selecting never also
//! edits the craft or changes mode.
//!
//! Scope: this is the editor-local *craft* library. Whole-world saves (WI 553) are a
//! separate concern and untouched here.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::ButtonState;
use bevy::prelude::*;
use sounding_sim::library::{self, CraftEntry};
use std::path::Path;

use crate::editor::EditorState;

/// Directory holding the named craft library (one document per vehicle).
const SAVES_DIR: &str = "saves/crafts";
/// Default name offered the first time a build is saved.
const DEFAULT_NAME: &str = "rover";
/// Cap on the typed name length (keeps the overlay tidy; slug is bounded separately).
const MAX_NAME_LEN: usize = 48;

/// The library modal's state. `Closed` lets the normal build input run; the other two
/// variants are mutually exclusive modal screens that suppress build input.
#[derive(Resource, Default)]
pub enum CraftLibraryModal {
    /// No modal — normal Build editing.
    #[default]
    Closed,
    /// Naming prompt before a save: the edited buffer and whether the last confirm was
    /// rejected as blank.
    Naming { buffer: String, rejected: bool },
    /// Load browser: the discovered vehicles and the highlighted index.
    Browsing {
        entries: Vec<CraftEntry>,
        selected: usize,
    },
}

impl CraftLibraryModal {
    /// Whether a modal screen is up (build input must stand down).
    pub fn is_open(&self) -> bool {
        !matches!(self, CraftLibraryModal::Closed)
    }
}

/// The name the current build was last saved/loaded under — seeds the naming prompt so
/// re-saving an existing vehicle defaults to its own name (an update, not a new slot).
#[derive(Resource)]
pub struct CurrentCraftName(pub String);

impl Default for CurrentCraftName {
    fn default() -> Self {
        Self(DEFAULT_NAME.to_string())
    }
}

/// Marker for the centered modal overlay text node (spawned in Build, despawned on exit
/// via `BuildEntity`).
#[derive(Component)]
pub struct CraftLibraryHud;

fn saves_dir() -> &'static Path {
    Path::new(SAVES_DIR)
}

/// Opens the library modal from Build: `K` → naming prompt (seeded with the current
/// name), `O` → load browser (scans the library). Only fires while `Closed`; the build
/// systems guard on `is_open` so these keys don't also act on the craft.
///
/// One system handles both opening (`K`/`O` while `Closed`) and driving the open modal,
/// so the opening frame can drain the keyboard queue before any text entry — otherwise
/// the very `k` that opens the naming prompt would leak into the seeded name buffer.
pub fn craft_library_input(
    mut modal: ResMut<CraftLibraryModal>,
    mut state: ResMut<EditorState>,
    mut current: ResMut<CurrentCraftName>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut typed: MessageReader<KeyboardInput>,
) {
    match &mut *modal {
        CraftLibraryModal::Closed => {
            // Open on K (naming, seeded with the current name) or O (load browser). Drain
            // the frame's typed events either way so the opening keystroke never lands in
            // the buffer; build input is guarded on `is_open` so K/O don't edit the craft.
            if keys.just_pressed(KeyCode::KeyK) {
                *modal = CraftLibraryModal::Naming {
                    buffer: current.0.clone(),
                    rejected: false,
                };
            } else if keys.just_pressed(KeyCode::KeyO) {
                *modal = CraftLibraryModal::Browsing {
                    entries: library::list_crafts(saves_dir()),
                    selected: 0,
                };
            }
            typed.clear();
        }
        CraftLibraryModal::Naming { buffer, rejected } => {
            // Esc cancels with no write.
            if keys.just_pressed(KeyCode::Escape) {
                typed.clear();
                *modal = CraftLibraryModal::Closed;
                return;
            }
            // Collect typed characters / backspace from logical key events.
            for ev in typed.read() {
                if ev.state != ButtonState::Pressed {
                    continue;
                }
                match &ev.logical_key {
                    Key::Character(s) => {
                        for ch in s.chars() {
                            if !ch.is_control() && buffer.chars().count() < MAX_NAME_LEN {
                                buffer.push(ch);
                            }
                        }
                        *rejected = false;
                    }
                    Key::Space => {
                        if buffer.chars().count() < MAX_NAME_LEN {
                            buffer.push(' ');
                        }
                        *rejected = false;
                    }
                    Key::Backspace => {
                        buffer.pop();
                        *rejected = false;
                    }
                    Key::Enter => {
                        // Consume the Enter so a later-running, non-ordered system (e.g.
                        // `toggle_mode`) can't also act on it once we close the modal.
                        keys.clear_just_pressed(KeyCode::Enter);
                        if library::is_valid_name(buffer) {
                            let name = buffer.trim();
                            match library::save_craft(saves_dir(), name, &state.craft) {
                                Ok(path) => {
                                    info!("saved craft \"{name}\" to {}", path.display());
                                    current.0 = name.to_string();
                                    *modal = CraftLibraryModal::Closed;
                                }
                                Err(e) => {
                                    warn!("craft save failed: {e}");
                                    *rejected = true;
                                }
                            }
                        } else {
                            *rejected = true;
                        }
                        return;
                    }
                    _ => {}
                }
            }
        }
        CraftLibraryModal::Browsing { entries, selected } => {
            if keys.just_pressed(KeyCode::Escape) {
                *modal = CraftLibraryModal::Closed;
                return;
            }
            if entries.is_empty() {
                // Nothing to pick; Enter/Esc both just close.
                if keys.just_pressed(KeyCode::Enter) {
                    keys.clear_just_pressed(KeyCode::Enter);
                    *modal = CraftLibraryModal::Closed;
                }
                return;
            }
            if keys.just_pressed(KeyCode::ArrowDown) {
                *selected = (*selected + 1) % entries.len();
            }
            if keys.just_pressed(KeyCode::ArrowUp) {
                *selected = (*selected + entries.len() - 1) % entries.len();
            }
            if keys.just_pressed(KeyCode::Enter) {
                // Consume the Enter (see the naming branch) before we close the modal.
                keys.clear_just_pressed(KeyCode::Enter);
                let entry = &entries[*selected];
                match library::load_craft(&entry.path) {
                    Ok(craft) => {
                        info!(
                            "loaded craft \"{}\" ({} voxels) into Build",
                            entry.name,
                            craft.voxels.len()
                        );
                        state.craft = craft;
                        current.0 = entry.name.clone();
                        *modal = CraftLibraryModal::Closed;
                    }
                    Err(e) => {
                        warn!("craft load failed for {}: {e}", entry.path.display());
                        *modal = CraftLibraryModal::Closed;
                    }
                }
            }
        }
    }
}

/// Renders the modal as centered text. Hidden (empty) when `Closed`.
pub fn draw_craft_library_overlay(
    modal: Res<CraftLibraryModal>,
    mut hud: Query<(&mut Text, &mut Visibility), With<CraftLibraryHud>>,
) {
    let Ok((mut text, mut vis)) = hud.single_mut() else {
        return;
    };
    match &*modal {
        CraftLibraryModal::Closed => {
            *vis = Visibility::Hidden;
        }
        CraftLibraryModal::Naming { buffer, rejected } => {
            *vis = Visibility::Visible;
            let mut s = String::from("SAVE VEHICLE\n\n");
            s.push_str(&format!("name: {buffer}_\n\n"));
            if *rejected {
                s.push_str("! enter a non-blank name\n\n");
            }
            s.push_str("[Enter] save   [Esc] cancel");
            text.0 = s;
        }
        CraftLibraryModal::Browsing { entries, selected } => {
            *vis = Visibility::Visible;
            let mut s = String::from("LOAD VEHICLE\n\n");
            if entries.is_empty() {
                s.push_str("(no saved vehicles)\n\n[Esc] close");
            } else {
                for (i, e) in entries.iter().enumerate() {
                    let marker = if i == *selected { ">" } else { " " };
                    s.push_str(&format!("{marker} {}\n", e.name));
                }
                s.push_str("\n[\u{2191}/\u{2193}] select   [Enter] load   [Esc] cancel");
            }
            text.0 = s;
        }
    }
}

/// Spawns the (initially hidden) modal overlay node. Call from Build's enter system;
/// tag with `BuildEntity` at the call site so it tears down with the mode.
pub fn spawn_craft_library_overlay(commands: &mut Commands) -> Entity {
    commands
        .spawn((
            Text::new(""),
            TextFont {
                font_size: 20.0,
                ..default()
            },
            TextColor(Color::srgb(0.95, 0.97, 1.0)),
            BackgroundColor(Color::srgba(0.05, 0.07, 0.12, 0.88)),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Percent(28.0),
                left: Val::Percent(38.0),
                padding: UiRect::all(Val::Px(16.0)),
                ..default()
            },
            Visibility::Hidden,
            CraftLibraryHud,
        ))
        .id()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modal_open_state() {
        assert!(!CraftLibraryModal::Closed.is_open());
        assert!(CraftLibraryModal::Naming {
            buffer: String::new(),
            rejected: false
        }
        .is_open());
        assert!(CraftLibraryModal::Browsing {
            entries: Vec::new(),
            selected: 0
        }
        .is_open());
    }

    #[test]
    fn current_name_defaults_nonblank() {
        assert!(library::is_valid_name(&CurrentCraftName::default().0));
    }
}

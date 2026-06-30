//! Shared Build-mode chrome (convergence Split A1, WI 738).
//!
//! Both the grounded workshop (`workshop_scene`) and the harbor (`harbor_scene`) author craft with
//! the **one** `editor` core (`EditorState`, `place_brush`, `PALETTE_GROUPS`, `mouse_build`, …). This
//! module owns the Build-mode chrome that was previously **duplicated** across those two scenes: the
//! clickable left-edge palette (WI 613), the Build marker components, the gizmo overlays, and the
//! Build teardown. Each scene keeps only what genuinely differs — its mode enum, its Build↔X toggle,
//! its environment dressing, its test/launch destination, its build-mesh rendering (which renders a
//! different *content subset* per scene), and its build-HUD readout (mass/inertia vs float/sink),
//! which writes into the shared [`BuildHud`].
//!
//! Sharing the palette here also **upgrades the harbor** to the clickable palette it previously
//! lacked. App-side UI only; no `sounding_sim` dependency.

use bevy::math::{DVec3, IVec3};
use bevy::prelude::*;

use crate::editor::{EditorState, HoverState, PaletteEntry, PointerOnPalette, PALETTE_GROUPS};

/// Tags every entity owned by Build mode (despawned on leaving Build). Roots only — children of a
/// tagged root are removed by the recursive despawn, so they must not also carry this (a double
/// despawn warns).
#[derive(Component)]
pub(crate) struct BuildEntity;

/// Tags a solid mesh entity rendering part of the Build craft (rebuilt on edit). Each scene's
/// `sync_build_meshes` owns what it spawns under this marker.
#[derive(Component)]
pub(crate) struct BuildMesh;

/// The Build status HUD text. Each scene runs its own update writing the scene's readout here.
#[derive(Component)]
pub(crate) struct BuildHud;

/// The root container of the Build palette (WI 613); carries `Interaction` so hovering its
/// background/gaps still counts as "pointer over the palette".
#[derive(Component)]
pub(crate) struct PaletteRoot;

/// A clickable Build-palette entry button (WI 613): clicking it selects that block/device/part.
#[derive(Component)]
pub(crate) struct PaletteButton(pub(crate) PaletteEntry);

/// Spawns the left-edge Build palette (WI 613): a docked column of grouped, clickable swatch+label
/// entries — Blocks, Devices, Wheels, Parts — one [`PaletteButton`] per buildable item. The root
/// carries an `Interaction` so hovering its background (between buttons) still registers as "over the
/// palette". Call from a scene's `enter_build`.
pub(crate) fn spawn_palette(commands: &mut Commands) {
    let idle = Color::srgb(0.16, 0.16, 0.18);
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(12.0),
                width: Val::Px(168.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            Interaction::default(),
            PaletteRoot,
            BuildEntity,
        ))
        .with_children(|root| {
            for (group, entries) in PALETTE_GROUPS {
                root.spawn((
                    Text::new(*group),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.55, 0.6, 0.68)),
                    Node {
                        margin: UiRect::top(Val::Px(4.0)),
                        ..default()
                    },
                ));
                for &entry in *entries {
                    root.spawn((
                        Button,
                        Node {
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(8.0),
                            padding: UiRect::all(Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(idle),
                        PaletteButton(entry),
                        // No BuildEntity here: buttons are children of the PaletteRoot, so the
                        // recursive despawn in exit_build removes them with the root (avoids a
                        // double-despawn warning on each Build→Test switch).
                    ))
                    .with_children(|btn| {
                        // Identity swatch.
                        btn.spawn((
                            Node {
                                width: Val::Px(16.0),
                                height: Val::Px(16.0),
                                ..default()
                            },
                            BackgroundColor(entry.swatch_color()),
                            BorderColor::all(Color::srgb(0.0, 0.0, 0.0)),
                        ));
                        // Label (so identity never rests on colour alone).
                        btn.spawn((
                            Text::new(entry.label()),
                            TextFont {
                                font_size: 13.0,
                                ..default()
                            },
                            TextColor(Color::srgb(0.88, 0.9, 0.94)),
                        ));
                    });
                }
            }
        });
}

/// Sets [`PointerOnPalette`] when the cursor is over the palette root or any entry (WI 613), so
/// `editor::mouse_build` skips a click that lands on the UI.
pub(crate) fn track_pointer_over_palette(
    mut flag: ResMut<PointerOnPalette>,
    roots: Query<&Interaction, With<PaletteRoot>>,
    buttons: Query<&Interaction, With<PaletteButton>>,
) {
    let over = roots.iter().any(|i| *i != Interaction::None)
        || buttons.iter().any(|i| *i != Interaction::None);
    flag.0 = over;
}

/// Applies a palette entry to the editor selection when its button is pressed (WI 613).
pub(crate) fn palette_click(
    buttons: Query<(&PaletteButton, &Interaction), Changed<Interaction>>,
    mut editor: ResMut<EditorState>,
    modal: Res<crate::craft_library::CraftLibraryModal>,
) {
    // Don't change the brush from a click while the library modal owns input (WI 675).
    if modal.is_open() {
        return;
    }
    for (button, interaction) in &buttons {
        if *interaction == Interaction::Pressed {
            button.0.apply(&mut editor);
        }
    }
}

/// Highlights the active palette entry and reflects hover (WI 613): selected reads from the editor
/// state, so keyboard shortcuts and palette clicks stay in sync through the one source of truth.
pub(crate) fn update_palette_highlight(
    editor: Res<EditorState>,
    mut buttons: Query<(&PaletteButton, &Interaction, &mut BackgroundColor)>,
) {
    for (button, interaction, mut bg) in &mut buttons {
        *bg = if button.0.is_active(&editor) {
            BackgroundColor(Color::srgb(0.20, 0.42, 0.78))
        } else if *interaction == Interaction::Hovered {
            BackgroundColor(Color::srgb(0.30, 0.30, 0.34))
        } else {
            BackgroundColor(Color::srgb(0.16, 0.16, 0.18))
        };
    }
}

/// Tears down Build mode: despawn every [`BuildEntity`] root (children removed recursively).
pub(crate) fn exit_build(mut commands: Commands, q: Query<Entity, With<BuildEntity>>) {
    for e in &q {
        commands.entity(e).despawn();
    }
}

/// Draws Build **overlays** as gizmos (WI 612): the mouse hover highlight + add-ghost, the keyboard
/// cursor, and the derived CoM / forward / principal-inertia axes. The solid geometry itself is meshes
/// (each scene's `sync_build_meshes`); gizmos are only for these overlays.
pub(crate) fn draw_build_overlays(
    mut gizmos: Gizmos,
    editor: Res<EditorState>,
    hover: Res<HoverState>,
) {
    let s = editor.craft.cell_size as f32;
    let cc = |c: IVec3| ((c.as_dvec3() + DVec3::splat(0.5)) * editor.craft.cell_size).as_vec3();

    // Keyboard cursor (faint yellow) — the precise fallback.
    gizmos.primitive_3d(
        &Cuboid::new(s * 1.04, s * 1.04, s * 1.04),
        cc(editor.cursor),
        Color::srgba(1.0, 1.0, 0.1, 0.45),
    );
    // Mouse hover: highlight the hovered cell and ghost where a click would add.
    if let Some(h) = hover.0 {
        gizmos.primitive_3d(
            &Cuboid::new(s * 1.08, s * 1.08, s * 1.08),
            cc(h.highlight),
            Color::srgb(0.2, 1.0, 0.45),
        );
        gizmos.primitive_3d(
            &Cuboid::new(s * 0.94, s * 0.94, s * 0.94),
            cc(h.add_cell),
            Color::srgba(0.2, 1.0, 0.45, 0.4),
        );
    }

    if let Some(mp) = editor.craft.mass_properties() {
        let com = mp.center_of_mass.as_vec3();
        gizmos.sphere(com, s * 0.3, Color::srgb(1.0, 0.1, 1.0));
        // Forward indicator: +Z is the assembled craft/rover's forward (cyan arrow).
        let fwd_len = (s * 5.0).max(1.5);
        gizmos.arrow(com, com + Vec3::Z * fwd_len, Color::srgb(0.1, 0.8, 1.0));
        let colors = [
            Color::srgb(1.0, 0.3, 0.3),
            Color::srgb(0.3, 1.0, 0.3),
            Color::srgb(0.4, 0.5, 1.0),
        ];
        let moments = [
            mp.principal_moments.x,
            mp.principal_moments.y,
            mp.principal_moments.z,
        ];
        let max_m = moments.iter().cloned().fold(0.0_f64, f64::max).max(1e-9);
        for i in 0..3 {
            let axis = mp.principal_axes.col(i).as_vec3().normalize_or_zero();
            let len = s * 2.5 * (moments[i] / max_m).sqrt() as f32;
            gizmos.line(com, com + axis * len, colors[i]);
            gizmos.line(com, com - axis * len, colors[i]);
        }
    }
}

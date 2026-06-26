//! Part gallery scene (`-- gallery`, WI 653): a viewer for the mechanical-kit part catalog. Every
//! shipped part glb is laid out on the ground **grouped by category** (a row per category), slowly
//! spinning on a pedestal, under an orbit/zoom camera. Click a part (Bevy mesh picking) to open a
//! read-only properties panel (name, category, material set, orientation, verts, and — for
//! device-backed parts — mass at the display cell size). Editable stats are WI 652.
//!
//! This is the first consumer of the [`crate::parts`] loader, so it doubles as the orient/scale
//! verifier for the catalog. App-side only; touches no `sounding_sim` state.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::picking::events::{Click, Pointer};
use bevy::picking::mesh_picking::MeshPickingPlugin;
use bevy::prelude::*;

use crate::parts::{part_device_mass, spawn_part_mesh, PartCategory, CATALOG, REFERENCE_CELL};

/// The cell size the gallery displays parts at (the workshop's editor-scale cell).
const DISPLAY_CELL: f64 = 0.5;
const DISPLAY_H: f32 = 0.42; // part display height — clears the tallest centred part above the disc
const PEDESTAL_H: f32 = 0.06; // a thin marker disc on the ground (not a can that swallows the part)
const ROW_GAP: f32 = 1.7; // spacing between category rows (Z)
const COL_GAP: f32 = 1.35; // spacing between parts in a row (X)

pub struct GalleryScenePlugin;

impl Plugin for GalleryScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MeshPickingPlugin)
            .init_resource::<GalleryCam>()
            .insert_resource(Selected(None))
            .insert_resource(SpinOn(true))
            .add_systems(Startup, setup)
            .add_systems(Update, (camera_input, orbit_camera, spin_parts, update_inspector));
    }
}

/// Orbit camera state for the gallery (its own — the editor's frames the build CoM).
#[derive(Resource)]
struct GalleryCam {
    yaw: f32,
    pitch: f32,
    dist: f32,
    target: Vec3,
}

impl Default for GalleryCam {
    fn default() -> Self {
        let z_center = (PartCategory::ORDER.len() as f32 - 1.0) * 0.5 * ROW_GAP;
        Self {
            yaw: 0.5,
            pitch: 0.55,
            dist: 9.0,
            target: Vec3::new(0.0, DISPLAY_H, z_center),
        }
    }
}

/// The currently selected catalog index (read-only inspect).
#[derive(Resource)]
struct Selected(Option<usize>);

#[derive(Resource)]
struct SpinOn(bool);

#[derive(Component)]
struct GallerySpin;

#[derive(Component)]
struct InspectorText;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground.
    let ground = meshes.add(Mesh::from(Plane3d::default().mesh().size(40.0, 40.0)));
    let ground_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.12, 0.13, 0.15),
        perceptual_roughness: 0.95,
        ..default()
    });
    let z_center = (PartCategory::ORDER.len() as f32 - 1.0) * 0.5 * ROW_GAP;
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(ground_mat),
        Transform::from_xyz(0.0, 0.0, z_center),
    ));

    let pedestal_mesh = meshes.add(Mesh::from(Cylinder::new(0.34, PEDESTAL_H)));
    let pedestal_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.20, 0.21, 0.24),
        perceptual_roughness: 0.8,
        ..default()
    });

    // Lay each category in a row (Z); parts within a category along X, row centred.
    for (row, cat) in PartCategory::ORDER.iter().enumerate() {
        let parts: Vec<usize> = CATALOG
            .iter()
            .enumerate()
            .filter(|(_, p)| p.category == *cat)
            .map(|(i, _)| i)
            .collect();
        let z = row as f32 * ROW_GAP;
        let n = parts.len() as f32;
        for (col, &idx) in parts.iter().enumerate() {
            let x = (col as f32 - (n - 1.0) * 0.5) * COL_GAP;
            // A thin marker disc on the ground under the part.
            commands.spawn((
                Mesh3d(pedestal_mesh.clone()),
                MeshMaterial3d(pedestal_mat.clone()),
                Transform::from_xyz(x, PEDESTAL_H * 0.5, z),
            ));
            // The part mesh, mount at the display height, slowly spinning + pickable.
            let root = spawn_part_mesh(
                &mut commands,
                &asset_server,
                CATALOG[idx].name,
                (DISPLAY_CELL / REFERENCE_CELL) as f32,
                Vec3::new(x, DISPLAY_H, z),
                Quat::IDENTITY,
            );
            commands.entity(root).insert(GallerySpin).observe(
                move |_click: On<Pointer<Click>>, mut sel: ResMut<Selected>| {
                    sel.0 = Some(idx);
                },
            );
        }
    }

    // Lights.
    commands.spawn((
        DirectionalLight {
            illuminance: 11_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(8.0, 14.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::default(),
        AmbientLight {
            brightness: 320.0,
            ..default()
        },
    ));

    // Legend / controls (left) — conveys the row grouping since per-part 3D labels are deferred.
    let rows = PartCategory::ORDER
        .iter()
        .enumerate()
        .map(|(i, c)| format!("  row {}: {}", i + 1, c.label()))
        .collect::<Vec<_>>()
        .join("\n");
    commands.spawn((
        Text::new(format!(
            "PART GALLERY ({} parts)\nclick a part to inspect\n\nrows (near \u{2192} far):\n{}\n\n\
             middle-drag orbit \u{b7} scroll zoom \u{b7} WASD pan \u{b7} Space spin",
            CATALOG.len(),
            rows
        )),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgb(0.85, 0.9, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));

    // Inspector panel (right) — filled on selection.
    commands.spawn((
        Text::new("(no part selected)"),
        InspectorText,
        TextFont {
            font_size: 15.0,
            ..default()
        },
        TextColor(Color::srgb(0.95, 0.95, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            right: Val::Px(14.0),
            ..default()
        },
    ));
}

/// Mouse/keyboard orbit + pan + zoom input → `GalleryCam`.
fn camera_input(
    mut cam: ResMut<GalleryCam>,
    mut spin: ResMut<SpinOn>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
) {
    if buttons.pressed(MouseButton::Middle) {
        cam.yaw -= motion.delta.x * 0.006;
        cam.pitch = (cam.pitch + motion.delta.y * 0.006).clamp(-1.4, 1.4);
    }
    if scroll.delta.y != 0.0 {
        cam.dist = (cam.dist * (1.0 - scroll.delta.y * 0.12)).clamp(1.5, 40.0);
    }
    // Pan the target on the ground plane (camera-relative), WASD.
    let dt = time.delta_secs();
    let (s, c) = cam.yaw.sin_cos();
    let fwd = Vec3::new(-s, 0.0, -c);
    let right = Vec3::new(c, 0.0, -s);
    let mut pan = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        pan += fwd;
    }
    if keys.pressed(KeyCode::KeyS) {
        pan -= fwd;
    }
    if keys.pressed(KeyCode::KeyD) {
        pan += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        pan -= right;
    }
    if pan != Vec3::ZERO {
        let step = dt * cam.dist * 0.6;
        cam.target += pan.normalize() * step;
    }
    if keys.just_pressed(KeyCode::Space) {
        spin.0 = !spin.0;
    }
}

fn orbit_camera(cam: Res<GalleryCam>, mut camera: Query<&mut Transform, With<Camera3d>>) {
    let Ok(mut tf) = camera.single_mut() else {
        return;
    };
    let (sy, cy) = cam.yaw.sin_cos();
    let (sp, cp) = cam.pitch.sin_cos();
    let offset = Vec3::new(sy * cp, sp, cy * cp) * cam.dist;
    *tf = Transform::from_translation(cam.target + offset).looking_at(cam.target, Vec3::Y);
}

fn spin_parts(spin: Res<SpinOn>, time: Res<Time>, mut q: Query<&mut Transform, With<GallerySpin>>) {
    if !spin.0 {
        return;
    }
    for mut t in &mut q {
        t.rotate_y(0.5 * time.delta_secs());
    }
}

fn update_inspector(
    sel: Res<Selected>,
    mut q: Query<&mut Text, With<InspectorText>>,
) {
    if !sel.is_changed() {
        return;
    }
    let Ok(mut text) = q.single_mut() else {
        return;
    };
    **text = match sel.0 {
        None => "(no part selected)".to_string(),
        Some(i) => {
            let p = &CATALOG[i];
            let mass = part_device_mass(p, DISPLAY_CELL)
                .map(|m| format!("\ndevice mass: {m:.1} kg (@ {DISPLAY_CELL} m cell)"))
                .unwrap_or_default();
            let device = p
                .device
                .map(|d| format!("\ndevice: {d:?}"))
                .unwrap_or_default();
            format!(
                "{}\ncategory: {}\nmaterial: {}\norientation: {}\nverts: {}{}{}",
                p.name,
                p.category.label(),
                p.material_set,
                p.orientation,
                p.verts,
                device,
                mass,
            )
        }
    };
}

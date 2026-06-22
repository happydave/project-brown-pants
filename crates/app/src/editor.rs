//! Toy 5 voxel editor + visualization (WI 505).
//!
//! A minimal interactive editor over a [`VoxelCraft`]: a movable cursor adds and
//! removes voxels and devices, blueprints and subassemblies are saved and loaded
//! through the WI 498 format, and the derived centre of mass, inertia (as the
//! principal axes), and cross-sectional-area curve are drawn live as gizmos. The
//! derivations themselves are headless and unit-tested in `sounding_sim::voxel`;
//! this module is the editor and the view.

use bevy::math::DVec3;
use bevy::prelude::*;
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::persist::{CraftSubgraph, Kind, Payload, SavedDocument};
use sounding_sim::voxel::{
    AttachmentPoint, Axis, Device, DeviceKind, Face, Material, Voxel, VoxelCraft,
};
use std::fs;

/// The selectable structural materials.
const PALETTE: [(&str, Material); 4] = [
    ("aluminium", Material::ALUMINIUM),
    ("steel", Material::STEEL),
    ("titanium", Material::TITANIUM),
    ("composite", Material::COMPOSITE),
];

const BLUEPRINT_PATH: &str = "blueprint.json";
const SUBASSEMBLY_PATH: &str = "subassembly.json";

/// The editor's state: the craft under construction, the build cursor, the
/// selected material, and a loaded subassembly buffer.
#[derive(Resource)]
pub struct EditorState {
    pub craft: VoxelCraft,
    pub cursor: IVec3,
    pub material: usize,
    pub subassembly: Option<VoxelCraft>,
}

impl Default for EditorState {
    fn default() -> Self {
        // A small seed craft so the derivations have something to show at startup.
        let mut craft = VoxelCraft::new(1.0);
        for x in 0..3 {
            for z in 0..2 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::ALUMINIUM,
                });
            }
        }
        Self {
            craft,
            cursor: IVec3::new(0, 1, 0),
            material: 0,
            subassembly: None,
        }
    }
}

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EditorState>()
            .init_resource::<OrbitCam>()
            .add_systems(Startup, setup_view)
            .add_systems(Update, (editor_input, draw_editor, orbit_camera));
    }
}

/// Orbit-camera state: yaw, pitch, and distance about the craft.
#[derive(Resource)]
struct OrbitCam {
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl Default for OrbitCam {
    fn default() -> Self {
        Self {
            yaw: 0.7,
            pitch: 0.5,
            dist: 14.0,
        }
    }
}

fn setup_view(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Transform::default()));
    commands.spawn((
        DirectionalLight {
            illuminance: 6_000.0,
            ..default()
        },
        Transform::from_xyz(6.0, 12.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Keyboard orbit camera, always framing the editor's build volume.
fn orbit_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut cam: ResMut<OrbitCam>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    let dt = time.delta_secs();
    if keys.pressed(KeyCode::KeyQ) {
        cam.yaw += dt;
    }
    if keys.pressed(KeyCode::KeyE) {
        cam.yaw -= dt;
    }
    if keys.pressed(KeyCode::KeyR) {
        cam.pitch = (cam.pitch + dt).clamp(0.05, 1.5);
    }
    if keys.pressed(KeyCode::KeyF) {
        cam.pitch = (cam.pitch - dt).clamp(0.05, 1.5);
    }
    if keys.pressed(KeyCode::KeyZ) {
        cam.dist = (cam.dist - dt * 12.0).max(2.0);
    }
    if keys.pressed(KeyCode::KeyC) {
        cam.dist = (cam.dist + dt * 12.0).min(80.0);
    }

    let target = Vec3::new(1.5, 0.5, 1.0);
    let dir = Vec3::new(
        cam.yaw.cos() * cam.pitch.cos(),
        cam.pitch.sin(),
        cam.yaw.sin() * cam.pitch.cos(),
    );
    if let Ok(mut tf) = camera.single_mut() {
        *tf = Transform::from_translation(target + dir * cam.dist).looking_at(target, Vec3::Y);
    }
}

fn editor_input(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<EditorState>) {
    // Move the cursor.
    let mut delta = IVec3::ZERO;
    if keys.just_pressed(KeyCode::ArrowLeft) {
        delta.x -= 1;
    }
    if keys.just_pressed(KeyCode::ArrowRight) {
        delta.x += 1;
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        delta.z -= 1;
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        delta.z += 1;
    }
    if keys.just_pressed(KeyCode::PageUp) {
        delta.y += 1;
    }
    if keys.just_pressed(KeyCode::PageDown) {
        delta.y -= 1;
    }
    if delta != IVec3::ZERO {
        state.cursor += delta;
    }

    // Add a voxel of the current material at the cursor (if empty).
    if keys.just_pressed(KeyCode::Space) {
        let cell = state.cursor;
        let material = PALETTE[state.material].1;
        if !state.craft.voxels.iter().any(|v| v.cell == cell) {
            state.craft.voxels.push(Voxel { cell, material });
        }
    }
    // Remove any voxel or device at the cursor.
    if keys.just_pressed(KeyCode::Backspace) {
        let cell = state.cursor;
        state.craft.voxels.retain(|v| v.cell != cell);
        state.craft.devices.retain(|d| d.cell != cell);
    }
    // Cycle the active material.
    if keys.just_pressed(KeyCode::Tab) {
        state.material = (state.material + 1) % PALETTE.len();
        info!("material: {}", PALETTE[state.material].0);
    }
    // Place a device at the cursor.
    if keys.just_pressed(KeyCode::KeyG) {
        let cell = state.cursor;
        state.craft.devices.retain(|d| d.cell != cell);
        state
            .craft
            .devices
            .push(Device::structural(cell, 100.0, DeviceKind::Engine));
    }

    // Save as a blueprint / a reusable subassembly.
    if keys.just_pressed(KeyCode::KeyB) {
        save(&state.craft, Kind::Blueprint, BLUEPRINT_PATH);
    }
    if keys.just_pressed(KeyCode::KeyN) {
        save_subassembly(&state.craft, SUBASSEMBLY_PATH);
    }
    // Load the subassembly into the buffer.
    if keys.just_pressed(KeyCode::KeyL) {
        match load(SUBASSEMBLY_PATH) {
            Some(c) => {
                info!("loaded subassembly ({} voxels)", c.voxels.len());
                state.subassembly = Some(c);
            }
            None => warn!("no subassembly to load at {SUBASSEMBLY_PATH}"),
        }
    }
    // Insert the loaded subassembly at the cursor (reuse).
    if keys.just_pressed(KeyCode::KeyV) {
        if let Some(sub) = state.subassembly.clone() {
            let at = state.cursor;
            state.craft.insert(&sub, at);
            info!("inserted subassembly at {at:?}");
        }
    }
    // Print mass properties to the log.
    if keys.just_pressed(KeyCode::KeyM) {
        if let Some(mp) = state.craft.mass_properties() {
            info!(
                "mass {:.1} kg, CoM ({:.2}, {:.2}, {:.2})",
                mp.mass, mp.center_of_mass.x, mp.center_of_mass.y, mp.center_of_mass.z
            );
        }
    }
}

fn craft_subgraph(craft: &VoxelCraft) -> CraftSubgraph {
    CraftSubgraph::new(
        "editor-craft",
        "Editor Craft",
        WorldPos::new(FrameId::CENTRAL_BODY, DVec3::ZERO),
        craft.clone(),
    )
}

fn save(craft: &VoxelCraft, kind: Kind, path: &str) {
    let payload = match kind {
        Kind::Subassembly => Payload::Subassembly(craft_subgraph(craft)),
        Kind::Blueprint => Payload::Blueprint(craft_subgraph(craft)),
        _ => Payload::Craft(craft_subgraph(craft)),
    };
    let result = SavedDocument::new(payload)
        .to_json()
        .map_err(|e| e.to_string())
        .and_then(|json| fs::write(path, json).map_err(|e| e.to_string()));
    match result {
        Ok(()) => info!("saved {kind:?} to {path}"),
        Err(e) => warn!("save failed: {e}"),
    }
}

fn save_subassembly(craft: &VoxelCraft, path: &str) {
    let mut c = craft.clone();
    // Give it a default attachment point at its lowest cell if it has none.
    if c.attachments.is_empty() {
        if let Some(cell) = c
            .voxels
            .iter()
            .map(|v| v.cell)
            .min_by_key(|c| (c.y, c.x, c.z))
        {
            c.attachments.push(AttachmentPoint {
                cell,
                face: Face::NegY,
            });
        }
    }
    save(&c, Kind::Subassembly, path);
}

fn load(path: &str) -> Option<VoxelCraft> {
    let json = fs::read_to_string(path).ok()?;
    let doc = SavedDocument::from_json(&json).ok()?;
    match doc.payload {
        Payload::Craft(c) | Payload::Subassembly(c) | Payload::Blueprint(c) => Some(c.craft),
        Payload::WorldSave(_) => None,
    }
}

fn cell_center(c: IVec3, cell_size: f64) -> Vec3 {
    ((c.as_dvec3() + DVec3::splat(0.5)) * cell_size).as_vec3()
}

fn material_color(m: Material) -> Color {
    match m {
        x if x == Material::ALUMINIUM => Color::srgb(0.70, 0.72, 0.78),
        x if x == Material::STEEL => Color::srgb(0.45, 0.47, 0.52),
        x if x == Material::TITANIUM => Color::srgb(0.62, 0.60, 0.52),
        x if x == Material::COMPOSITE => Color::srgb(0.25, 0.24, 0.30),
        _ => Color::srgb(0.55, 0.35, 0.55),
    }
}

fn draw_editor(mut gizmos: Gizmos, state: Res<EditorState>) {
    let s = state.craft.cell_size as f32;

    // Voxels.
    for v in &state.craft.voxels {
        gizmos.primitive_3d(
            &Cuboid::new(s, s, s),
            cell_center(v.cell, state.craft.cell_size),
            material_color(v.material),
        );
    }
    // Devices (smaller, orange).
    for d in &state.craft.devices {
        gizmos.primitive_3d(
            &Cuboid::new(s * 0.6, s * 0.6, s * 0.6),
            cell_center(d.cell, state.craft.cell_size),
            Color::srgb(1.0, 0.55, 0.0),
        );
    }
    // Build cursor (yellow, slightly oversized).
    gizmos.primitive_3d(
        &Cuboid::new(s * 1.06, s * 1.06, s * 1.06),
        cell_center(state.cursor, state.craft.cell_size),
        Color::srgb(1.0, 1.0, 0.1),
    );

    if let Some(mp) = state.craft.mass_properties() {
        let com = mp.center_of_mass.as_vec3();
        // Centre of mass (magenta).
        gizmos.sphere(com, s * 0.3, Color::srgb(1.0, 0.1, 1.0));
        // Principal inertia axes, length scaled by the moment, RGB.
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

    // Cross-sectional-area curve along X, plotted off to the side (cyan).
    let curve = state.craft.area_curve(Axis::X);
    if !curve.is_empty() {
        let max_a = curve
            .iter()
            .map(|(_, a)| *a)
            .fold(0.0_f64, f64::max)
            .max(1e-9);
        let origin = Vec3::new(-4.0, 0.0, -2.0);
        let points: Vec<Vec3> = curve
            .iter()
            .enumerate()
            .map(|(i, (_station, area))| {
                origin + Vec3::new(i as f32 * 0.6, (area / max_a) as f32 * 3.0, 0.0)
            })
            .collect();
        gizmos.linestrip(points, Color::srgb(0.2, 1.0, 1.0));
    }
}

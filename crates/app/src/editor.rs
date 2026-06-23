//! Toy 5 voxel editor + visualization (WI 505).
//!
//! A minimal interactive editor over a [`VoxelCraft`]: a movable cursor adds and
//! removes voxels and devices, blueprints and subassemblies are saved and loaded
//! through the WI 498 format, and the derived centre of mass, inertia (as the
//! principal axes), and cross-sectional-area curve are drawn live as gizmos. The
//! derivations themselves are headless and unit-tested in `sounding_sim::voxel`;
//! this module is the editor and the view.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::math::DVec3;
use bevy::prelude::*;
use sounding_sim::control::{BatterySpec, ControlComputer};
use sounding_sim::frame::{FrameId, WorldPos};
use sounding_sim::persist::{CraftSubgraph, Kind, Payload, SavedDocument};
use sounding_sim::voxel::{
    AttachmentPoint, Axis, Device, DeviceKind, Face, Material, Part, PartKind, Voxel, VoxelCraft,
    WheelPart,
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

/// The active placement tool (WI 612): what a click (or Space) places. The number keys select it;
/// `Voxel` uses the active material. This is what makes mouse building work — the tool follows the
/// cursor instead of every device piling onto the keyboard cursor cell.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Brush {
    #[default]
    Voxel,
    ControlPoint,
    Computer,
    Battery,
    Engine,
    Tank,
    Wheel {
        drive: bool,
        steer: bool,
    },
}

impl Brush {
    /// A short label for the HUD.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Brush::Voxel => "voxel",
            Brush::ControlPoint => "control point",
            Brush::Computer => "computer",
            Brush::Battery => "battery",
            Brush::Engine => "engine",
            Brush::Tank => "tank",
            Brush::Wheel { steer: false, .. } => "wheel (drive)",
            Brush::Wheel { steer: true, .. } => "wheel (drive+steer)",
        }
    }
}

/// The editor's state: the craft under construction, the build cursor, the
/// selected material, the active placement brush, and a loaded subassembly buffer.
#[derive(Resource)]
pub struct EditorState {
    pub craft: VoxelCraft,
    pub cursor: IVec3,
    pub material: usize,
    pub(crate) brush: Brush,
    pub subassembly: Option<VoxelCraft>,
}

impl Default for EditorState {
    fn default() -> Self {
        // A small seed craft so the derivations have something to show at startup. 0.1 m cells —
        // fine enough to build vehicles, not castles (WI 612 feedback).
        let mut craft = VoxelCraft::new(0.1);
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
            brush: Brush::Voxel,
            subassembly: None,
        }
    }
}

/// Places the active `brush` into `craft`. The three cells let the caller aim each kind: a voxel at
/// `voxel_cell` (the empty face cell for the mouse), a device at `device_cell` (the hovered solid
/// cell), and a wheel at `wheel_mount` (a continuous body-frame point).
fn place_brush(
    craft: &mut VoxelCraft,
    brush: Brush,
    material: Material,
    voxel_cell: IVec3,
    device_cell: IVec3,
    wheel_mount: DVec3,
) {
    match brush {
        Brush::Voxel => {
            if !craft.voxels.iter().any(|v| v.cell == voxel_cell) {
                craft.voxels.push(Voxel {
                    cell: voxel_cell,
                    material,
                });
            }
        }
        Brush::Wheel { drive, steer } => {
            let s = craft.cell_size;
            craft
                .parts
                .retain(|p| (p.mount - wheel_mount).length() > 1e-6);
            craft.parts.push(Part {
                mount: wheel_mount,
                // Wheel mass scaled to the build (a few cells' worth of material).
                mass: (15.0 * s).max(1.0),
                kind: PartKind::Wheel(WheelPart::for_cell_size(s, drive, steer)),
            });
        }
        device => {
            let d = match device {
                Brush::ControlPoint => Device::control_point(device_cell, 120.0, true),
                Brush::Computer => {
                    Device::computer(device_cell, 40.0, ControlComputer::tuning_computer(0.4))
                }
                Brush::Battery => Device::battery(device_cell, 60.0, BatterySpec::full(120.0)),
                Brush::Engine => Device::structural(device_cell, 100.0, DeviceKind::Engine),
                Brush::Tank => Device::structural(device_cell, 80.0, DeviceKind::Tank),
                Brush::Voxel | Brush::Wheel { .. } => unreachable!(),
            };
            craft.devices.retain(|x| x.cell != device_cell);
            craft.devices.push(d);
        }
    }
}

/// The human-readable name of palette material `index` (for HUD/palette display).
pub(crate) fn material_label(index: usize) -> &'static str {
    PALETTE[index % PALETTE.len()].0
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

/// Orbit-camera state: yaw, pitch, and distance about the craft. Reused by the workshop's
/// Build mode (WI 603), so it is crate-visible.
#[derive(Resource)]
pub(crate) struct OrbitCam {
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl Default for OrbitCam {
    fn default() -> Self {
        Self {
            yaw: 0.7,
            pitch: 0.4,
            dist: 3.0,
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

/// Keyboard orbit camera, always framing the editor's build volume. Crate-visible so the
/// workshop's Build mode (WI 603) can run it under a state run-condition.
pub(crate) fn orbit_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    editor: Res<EditorState>,
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
        cam.pitch = (cam.pitch + dt).clamp(-1.4, 1.4);
    }
    if keys.pressed(KeyCode::KeyF) {
        cam.pitch = (cam.pitch - dt).clamp(-1.4, 1.4);
    }
    if keys.pressed(KeyCode::KeyZ) {
        cam.dist = (cam.dist - dt * cam.dist * 1.5).max(0.2);
    }
    if keys.pressed(KeyCode::KeyC) {
        cam.dist = (cam.dist + dt * 12.0).min(80.0);
    }

    // Frame the build: target its centre of mass so any cell size / build is centred.
    let target = editor
        .craft
        .mass_properties()
        .map(|mp| mp.center_of_mass.as_vec3())
        .unwrap_or(Vec3::splat(0.5));
    let dir = Vec3::new(
        cam.yaw.cos() * cam.pitch.cos(),
        cam.pitch.sin(),
        cam.yaw.sin() * cam.pitch.cos(),
    );
    if let Ok(mut tf) = camera.single_mut() {
        *tf = Transform::from_translation(target + dir * cam.dist).looking_at(target, Vec3::Y);
    }
}

/// Editor keybindings over [`EditorState`]. Crate-visible so the workshop's Build mode (WI 603)
/// can run it under a state run-condition.
pub(crate) fn editor_input(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<EditorState>) {
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

    // Select the active brush (what a click / Space places). Tab cycles material and returns to the
    // voxel brush; the digit keys pick devices/wheels: 1 control point · 2 computer · 3 battery ·
    // 4 (or G) engine · 5 tank · 6 wheel (drive) · 7 wheel (drive+steer).
    if keys.just_pressed(KeyCode::Tab) {
        state.material = (state.material + 1) % PALETTE.len();
        state.brush = Brush::Voxel;
        info!("material: {}", PALETTE[state.material].0);
    }
    if keys.just_pressed(KeyCode::Digit1) {
        state.brush = Brush::ControlPoint;
    } else if keys.just_pressed(KeyCode::Digit2) {
        state.brush = Brush::Computer;
    } else if keys.just_pressed(KeyCode::Digit3) {
        state.brush = Brush::Battery;
    } else if keys.just_pressed(KeyCode::KeyG) || keys.just_pressed(KeyCode::Digit4) {
        state.brush = Brush::Engine;
    } else if keys.just_pressed(KeyCode::Digit5) {
        state.brush = Brush::Tank;
    } else if keys.just_pressed(KeyCode::Digit6) {
        state.brush = Brush::Wheel {
            drive: true,
            steer: false,
        };
    } else if keys.just_pressed(KeyCode::Digit7) {
        state.brush = Brush::Wheel {
            drive: true,
            steer: true,
        };
    }

    // Place the active brush at the cursor (keyboard fallback; the mouse is the primary path).
    if keys.just_pressed(KeyCode::Space) {
        let cell = state.cursor;
        let material = PALETTE[state.material].1;
        let brush = state.brush;
        let mount = (cell.as_dvec3() + DVec3::splat(0.5)) * state.craft.cell_size;
        place_brush(&mut state.craft, brush, material, cell, cell, mount);
    }
    // Remove any voxel, device, or attached part at the cursor.
    if keys.just_pressed(KeyCode::Backspace) {
        let cell = state.cursor;
        let center = (cell.as_dvec3() + DVec3::splat(0.5)) * state.craft.cell_size;
        state.craft.voxels.retain(|v| v.cell != cell);
        state.craft.devices.retain(|d| d.cell != cell);
        state
            .craft
            .parts
            .retain(|p| (p.mount - center).length() > 1e-6);
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

/// The voxel/face under the mouse cursor (WI 612): where a left-click adds, where a right-click
/// removes (None when hovering empty ground), and the cell to highlight.
#[derive(Clone, Copy)]
pub(crate) struct Hovered {
    pub add_cell: IVec3,
    pub remove_cell: Option<IVec3>,
    pub highlight: IVec3,
}

/// The current mouse hover, recomputed each frame (WI 612). `None` when the cursor is off-window or
/// the ray misses both the craft and the ground plane.
#[derive(Resource, Default)]
pub(crate) struct HoverState(pub Option<Hovered>);

/// Ray vs. one axis-aligned cell box; returns the entry distance and the entry-face normal.
fn ray_aabb(o: Vec3, d: Vec3, min: Vec3, max: Vec3) -> Option<(f32, IVec3)> {
    let inv = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    let t1 = (min - o) * inv;
    let t2 = (max - o) * inv;
    let tmin = t1.min(t2);
    let tmax = t1.max(t2);
    let tnear = tmin.x.max(tmin.y).max(tmin.z);
    let tfar = tmax.x.min(tmax.y).min(tmax.z);
    // Require entering from outside the box (tnear ≥ 0) and a valid interval.
    if !tnear.is_finite() || tnear > tfar || tnear < 0.0 {
        return None;
    }
    let normal = if tnear == tmin.x {
        IVec3::new(if d.x > 0.0 { -1 } else { 1 }, 0, 0)
    } else if tnear == tmin.y {
        IVec3::new(0, if d.y > 0.0 { -1 } else { 1 }, 0)
    } else {
        IVec3::new(0, 0, if d.z > 0.0 { -1 } else { 1 })
    };
    Some((tnear, normal))
}

/// Nearest voxel hit by the ray `(o, d)`: the hit cell and the entry-face normal. Brute-force over
/// the (sparse, modest) voxel set. Crate-visible + pure so it is unit-testable.
pub(crate) fn raycast_voxels(o: Vec3, d: Vec3, craft: &VoxelCraft) -> Option<(IVec3, IVec3)> {
    let s = craft.cell_size as f32;
    let mut best_t = f32::INFINITY;
    let mut best = None;
    for v in &craft.voxels {
        let min = v.cell.as_vec3() * s;
        let max = (v.cell + IVec3::ONE).as_vec3() * s;
        if let Some((t, n)) = ray_aabb(o, d, min, max) {
            if t < best_t {
                best_t = t;
                best = Some((v.cell, n));
            }
        }
    }
    best
}

/// Mouse orbit/zoom: middle-drag orbits, scroll zooms (WI 612). Left/right buttons stay free for
/// building. Mutates [`OrbitCam`]; `orbit_camera` then positions the camera. Crate-visible so the
/// workshop's Build mode runs it under a state run-condition.
pub(crate) fn mouse_orbit_input(
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut cam: ResMut<OrbitCam>,
) {
    if buttons.pressed(MouseButton::Middle) {
        // Horizontal drag orbits (yaw); vertical drag tilts (pitch), allowed below the horizon so
        // you can look up at the underside of the build.
        cam.yaw += motion.delta.x * 0.01;
        cam.pitch = (cam.pitch + motion.delta.y * 0.01).clamp(-1.4, 1.4);
    }
    if scroll.delta.y != 0.0 {
        // Scale the zoom step with distance so it stays usable from 0.1 m builds out to large ones.
        cam.dist = (cam.dist - scroll.delta.y * cam.dist * 0.1).clamp(0.2, 80.0);
    }
}

/// Recompute the mouse hover from the camera ray (WI 612): the hovered voxel + entry face, or a
/// ground-plane (y=0) cell when the ray misses the craft (so the first voxel can be placed).
pub(crate) fn update_hover(
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    editor: Res<EditorState>,
    mut hover: ResMut<HoverState>,
) {
    hover.0 = None;
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((cam, cam_tf)) = cameras.single() else {
        return;
    };
    let Ok(ray) = cam.viewport_to_world(cam_tf, cursor) else {
        return;
    };
    let o = ray.origin;
    let d = *ray.direction;
    let s = editor.craft.cell_size as f32;

    if let Some((cell, n)) = raycast_voxels(o, d, &editor.craft) {
        hover.0 = Some(Hovered {
            add_cell: cell + n,
            remove_cell: Some(cell),
            highlight: cell,
        });
    } else if d.y.abs() > 1e-6 {
        let t = -o.y / d.y;
        if t > 0.0 {
            let p = o + d * t;
            let cell = IVec3::new((p.x / s).floor() as i32, 0, (p.z / s).floor() as i32);
            hover.0 = Some(Hovered {
                add_cell: cell,
                remove_cell: None,
                highlight: cell,
            });
        }
    }
}

/// Mouse building (WI 612): left-click adds a voxel of the active material at the hovered face;
/// right-click removes the hovered voxel and any device/part in that cell.
pub(crate) fn mouse_build(
    buttons: Res<ButtonInput<MouseButton>>,
    hover: Res<HoverState>,
    mut state: ResMut<EditorState>,
) {
    let Some(h) = hover.0 else {
        return;
    };
    if buttons.just_pressed(MouseButton::Left) {
        let material = PALETTE[state.material].1;
        let brush = state.brush;
        // Voxel/wheel go on the clicked face (the empty adjacent cell); a device goes into the
        // hovered solid cell.
        let device_cell = h.remove_cell.unwrap_or(h.add_cell);
        let wheel_mount = (h.add_cell.as_dvec3() + DVec3::splat(0.5)) * state.craft.cell_size;
        place_brush(
            &mut state.craft,
            brush,
            material,
            h.add_cell,
            device_cell,
            wheel_mount,
        );
    }
    if buttons.just_pressed(MouseButton::Right) {
        let s = state.craft.cell_size;
        // Remove the voxel/device in the hovered solid cell.
        if let Some(cell) = h.remove_cell {
            state.craft.voxels.retain(|v| v.cell != cell);
            state.craft.devices.retain(|dd| dd.cell != cell);
        }
        // Remove a wheel/part near the hovered cells. Wheels mount on a **face** (the empty adjacent
        // cell), so they aren't in any voxel cell — check both the hovered cell and the face cell so
        // a wheel is removable by right-clicking the face it hangs off (the "can't remove wheels" gap).
        let mut centers = vec![(h.add_cell.as_dvec3() + DVec3::splat(0.5)) * s];
        if let Some(cell) = h.remove_cell {
            centers.push((cell.as_dvec3() + DVec3::splat(0.5)) * s);
        }
        state
            .craft
            .parts
            .retain(|p| centers.iter().all(|c| (p.mount - *c).length() > 0.6 * s));
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

/// Draws the editor's craft, cursor, and derived properties as gizmos. Crate-visible so the
/// workshop's Build mode (WI 603) can run it under a state run-condition.
pub(crate) fn draw_editor(mut gizmos: Gizmos, state: Res<EditorState>) {
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
    // Attached wheel parts (dark spheres at their continuous mount; steer wheels tinted).
    for p in &state.craft.parts {
        if let PartKind::Wheel(spec) = p.kind {
            let color = if spec.steer {
                Color::srgb(0.15, 0.18, 0.28)
            } else {
                Color::srgb(0.12, 0.12, 0.14)
            };
            gizmos.sphere(p.mount.as_vec3(), spec.radius as f32, color);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raycast_hits_top_face_of_a_voxel() {
        // A unit voxel at the origin cell; a ray straight down from above its centre.
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        let hit = raycast_voxels(Vec3::new(0.5, 5.0, 0.5), Vec3::NEG_Y, &craft);
        assert_eq!(hit, Some((IVec3::ZERO, IVec3::Y)), "top face, +Y normal");
    }

    #[test]
    fn raycast_misses_when_pointing_away() {
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        // Ray above the voxel pointing up — never enters the box.
        assert_eq!(
            raycast_voxels(Vec3::new(0.5, 5.0, 0.5), Vec3::Y, &craft),
            None
        );
    }

    #[test]
    fn raycast_nearest_voxel_wins() {
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        craft.voxels.push(Voxel {
            cell: IVec3::new(0, 3, 0),
            material: Material::ALUMINIUM,
        });
        // From far above looking down, the higher voxel (y=3) is hit first.
        let hit = raycast_voxels(Vec3::new(0.5, 20.0, 0.5), Vec3::NEG_Y, &craft);
        assert_eq!(hit, Some((IVec3::new(0, 3, 0), IVec3::Y)));
    }
}

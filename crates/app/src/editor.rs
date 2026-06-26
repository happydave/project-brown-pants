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
use sounding_sim::persist::{CraftSubgraph, FormatError, Kind, Payload, SavedDocument};
use sounding_sim::powertrain::MotorTier;
use sounding_sim::voxel::{
    device_mass, AttachmentPoint, Axis, Device, DeviceKind, Face, Material, Part, PartKind,
    RimSpec, SuspensionSpec, TireSpec, Voxel, VoxelCraft,
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
/// Where the whole current build is saved/loaded as a craft (WI 637).
const CRAFT_PATH: &str = "craft.json";

/// A rim+tire combination the player picks as a unit (WI 630). Each preset is a coherent
/// rim+tire character — the felt difference of "change the tires" — while the underlying suspension /
/// rim / tire are still separate components (so per-component mixing and failure modes land later).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum WheelPreset {
    /// Grippy, soft, tall — planted and bouncy on rough ground.
    OffRoad,
    /// Balanced all-rounder.
    #[default]
    Road,
    /// Very grippy, stiff, low-profile — responsive and harsh.
    Slick,
}

impl WheelPreset {
    /// The total (effective rolling) wheel radius for this preset at `cell_size`, matching the legacy
    /// wheel sizing (1.5 × cell) so builds read the same; the tire profile is a fraction of it.
    fn total_radius(self, cell_size: f64) -> f64 {
        1.5 * cell_size
    }

    /// The tire's section height as a fraction of the total radius (tall off-road … low-profile slick).
    fn profile_frac(self) -> f64 {
        match self {
            WheelPreset::OffRoad => 0.45,
            WheelPreset::Road => 0.30,
            WheelPreset::Slick => 0.15,
        }
    }

    /// The rim component for this preset, carrying the drivetrain flags (drive is always on, as before).
    pub(crate) fn rim(self, cell_size: f64, steer: bool) -> RimSpec {
        let total = self.total_radius(cell_size);
        RimSpec {
            radius: total - self.profile_frac() * total,
            drive: true,
            steer,
        }
    }

    /// The tire component for this preset: grip (compound), compliance (rubber/air spring), slip
    /// stiffness (response), and profile (with the rim → effective radius).
    pub(crate) fn tire(self, cell_size: f64) -> TireSpec {
        let total = self.total_radius(cell_size);
        let (grip_scale, stiffness, slip_long, slip_lat) = match self {
            WheelPreset::OffRoad => (1.35, 5.0e4, 4.5, 3.5),
            WheelPreset::Road => (1.0, 1.5e5, 5.0, 4.0),
            WheelPreset::Slick => (1.6, 4.0e5, 6.5, 5.5),
        };
        TireSpec {
            profile: self.profile_frac() * total,
            grip_scale,
            slip_long,
            slip_lat,
            stiffness,
        }
    }

    fn color(self) -> Color {
        match self {
            WheelPreset::OffRoad => Color::srgb(0.30, 0.22, 0.12),
            WheelPreset::Road => Color::srgb(0.12, 0.12, 0.14),
            WheelPreset::Slick => Color::srgb(0.05, 0.05, 0.07),
        }
    }
}

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
    /// A rim+tire wheel of the chosen preset (drive always on); `steer` adds it to the steer group.
    /// Placed at a wheel station; a suspension may be added to the same station, or omitted to ride on
    /// the tire (WI 630).
    Wheel {
        preset: WheelPreset,
        steer: bool,
    },
    /// An optional suspension strut for a wheel station (WI 630): adds spring travel; omit it and the
    /// wheel rides on the tire's compliance.
    Suspension,
    Seat,
    Antenna,
    SolarPanel,
    Bumper,
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
            Brush::Wheel {
                preset: WheelPreset::OffRoad,
                steer: false,
            } => "off-road wheel",
            Brush::Wheel {
                preset: WheelPreset::OffRoad,
                steer: true,
            } => "off-road wheel (steer)",
            Brush::Wheel {
                preset: WheelPreset::Road,
                steer: false,
            } => "road wheel",
            Brush::Wheel {
                preset: WheelPreset::Road,
                steer: true,
            } => "road wheel (steer)",
            Brush::Wheel {
                preset: WheelPreset::Slick,
                steer: false,
            } => "slick wheel",
            Brush::Wheel {
                preset: WheelPreset::Slick,
                steer: true,
            } => "slick wheel (steer)",
            Brush::Suspension => "suspension",
            Brush::Seat => "seat",
            Brush::Antenna => "antenna",
            Brush::SolarPanel => "solar panel",
            Brush::Bumper => "bumper",
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
    /// The motor tier a placed Engine device gets (WI 652); cycled with `M`.
    pub(crate) motor: MotorTier,
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
            motor: MotorTier::Standard,
        }
    }
}

/// The wheel-station id of any part mounted at (≈) `mount`, if one exists (WI 630): lets a suspension
/// and a rim+tire placed at the same corner join the same station.
fn station_at(craft: &VoxelCraft, mount: DVec3) -> Option<u32> {
    craft
        .parts
        .iter()
        .filter(|p| (p.mount - mount).length() <= 1e-6)
        .find_map(|p| p.station)
}

/// A station id not yet used by any part (WI 630): the next corner gets a fresh station.
fn next_station_id(craft: &VoxelCraft) -> u32 {
    craft
        .parts
        .iter()
        .filter_map(|p| p.station)
        .max()
        .map_or(0, |m| m + 1)
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
    motor: MotorTier,
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
        Brush::Wheel { preset, steer } => {
            let s = craft.cell_size;
            // Find this corner's station (reusing one a suspension already created here) or start a new
            // one; replace any rim/tire already on it, keeping a suspension that is present (WI 630).
            let id = station_at(craft, wheel_mount).unwrap_or_else(|| next_station_id(craft));
            craft.parts.retain(|p| {
                !(p.station == Some(id) && matches!(p.kind, PartKind::Rim(_) | PartKind::Tire(_)))
            });
            craft.parts.push(Part {
                mount: wheel_mount,
                mass: (8.0 * s).max(0.5),
                kind: PartKind::Rim(preset.rim(s, steer)),
                station: Some(id),
            });
            craft.parts.push(Part {
                mount: wheel_mount,
                mass: (8.0 * s).max(0.5),
                kind: PartKind::Tire(preset.tire(s)),
                station: Some(id),
            });
        }
        Brush::Suspension => {
            // Optional strut for a wheel station: reuse the corner's station (or start one), replacing
            // a strut already there. Omitting it leaves the wheel riding on the tire (WI 630).
            let s = craft.cell_size;
            let id = station_at(craft, wheel_mount).unwrap_or_else(|| next_station_id(craft));
            craft
                .parts
                .retain(|p| !(p.station == Some(id) && matches!(p.kind, PartKind::Suspension(_))));
            craft.parts.push(Part {
                mount: wheel_mount,
                mass: (4.0 * s).max(0.3),
                kind: PartKind::Suspension(SuspensionSpec::for_cell_size(s)),
                station: Some(id),
            });
        }
        Brush::Seat | Brush::Antenna | Brush::SolarPanel | Brush::Bumper => {
            let kind = match brush {
                Brush::Seat => PartKind::Seat,
                Brush::Antenna => PartKind::Antenna,
                Brush::SolarPanel => PartKind::SolarPanel,
                Brush::Bumper => PartKind::Bumper,
                _ => unreachable!(),
            };
            // Cosmetic parts mount on a face like wheels, at a small build-scaled mass.
            craft
                .parts
                .retain(|p| (p.mount - wheel_mount).length() > 1e-6);
            craft.parts.push(Part {
                mount: wheel_mount,
                mass: (10.0 * craft.cell_size).max(1.0),
                kind,
                station: None,
            });
        }
        Brush::ControlPoint | Brush::Computer | Brush::Battery | Brush::Engine | Brush::Tank => {
            // Device mass scales with the build's cell volume (WI 615) so a device is comparable to
            // the voxels around it, not a fixed 40–120 kg that dominates a small build.
            let s = craft.cell_size;
            let d = match brush {
                Brush::ControlPoint => {
                    Device::control_point(device_cell, device_mass(DeviceKind::Command, s), true)
                }
                Brush::Computer => Device::computer(
                    device_cell,
                    device_mass(DeviceKind::Computer, s),
                    ControlComputer::tuning_computer(0.4),
                ),
                Brush::Battery => Device::battery(
                    device_cell,
                    device_mass(DeviceKind::Battery, s),
                    BatterySpec::full(120.0),
                ),
                // An Engine is a selectable motor (WI 652): its mass + drivetrain come from the tier.
                Brush::Engine => Device::engine(device_cell, motor),
                Brush::Tank => Device::structural(
                    device_cell,
                    device_mass(DeviceKind::Tank, s),
                    DeviceKind::Tank,
                ),
                _ => unreachable!(),
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

/// One selectable item in the Build palette (WI 613). A `Material` entry selects a structural
/// material and switches the active brush to `Voxel` (the same semantics as cycling material with
/// Tab); a `Tool` entry selects a non-voxel brush (a device or a part). The palette is a *view* of
/// the selection held in [`EditorState`] — these are the pure mappings between an entry and that
/// state, kept Bevy/ECS-free so they are unit-testable.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PaletteEntry {
    Material(usize),
    Tool(Brush),
}

/// The palette's grouped, ordered entries: structural blocks, then devices, then attached parts.
/// Adding a buildable item is one line here; the entry↔state mappings and the round-trip test then
/// cover it automatically.
pub(crate) const PALETTE_GROUPS: &[(&str, &[PaletteEntry])] = &[
    (
        "BLOCKS",
        &[
            PaletteEntry::Material(0),
            PaletteEntry::Material(1),
            PaletteEntry::Material(2),
            PaletteEntry::Material(3),
        ],
    ),
    (
        "DEVICES",
        &[
            PaletteEntry::Tool(Brush::ControlPoint),
            PaletteEntry::Tool(Brush::Computer),
            PaletteEntry::Tool(Brush::Battery),
            PaletteEntry::Tool(Brush::Engine),
            PaletteEntry::Tool(Brush::Tank),
        ],
    ),
    (
        "WHEELS",
        &[
            PaletteEntry::Tool(Brush::Wheel {
                preset: WheelPreset::OffRoad,
                steer: false,
            }),
            PaletteEntry::Tool(Brush::Wheel {
                preset: WheelPreset::OffRoad,
                steer: true,
            }),
            PaletteEntry::Tool(Brush::Wheel {
                preset: WheelPreset::Road,
                steer: false,
            }),
            PaletteEntry::Tool(Brush::Wheel {
                preset: WheelPreset::Road,
                steer: true,
            }),
            PaletteEntry::Tool(Brush::Wheel {
                preset: WheelPreset::Slick,
                steer: false,
            }),
            PaletteEntry::Tool(Brush::Wheel {
                preset: WheelPreset::Slick,
                steer: true,
            }),
            PaletteEntry::Tool(Brush::Suspension),
        ],
    ),
    (
        "PARTS",
        &[
            PaletteEntry::Tool(Brush::Seat),
            PaletteEntry::Tool(Brush::Antenna),
            PaletteEntry::Tool(Brush::SolarPanel),
            PaletteEntry::Tool(Brush::Bumper),
        ],
    ),
];

impl PaletteEntry {
    /// Applies this entry to the editor selection: a material sets the material and the Voxel brush;
    /// a tool sets its brush.
    pub(crate) fn apply(self, state: &mut EditorState) {
        match self {
            PaletteEntry::Material(i) => {
                state.material = i % PALETTE.len();
                state.brush = Brush::Voxel;
            }
            PaletteEntry::Tool(brush) => state.brush = brush,
        }
    }

    /// Whether this entry is the one currently selected. For the Voxel brush, the active entry is the
    /// *material* being placed (so the palette shows which block), not a generic voxel entry.
    pub(crate) fn is_active(self, state: &EditorState) -> bool {
        match self {
            PaletteEntry::Material(i) => {
                state.brush == Brush::Voxel && state.material % PALETTE.len() == i % PALETTE.len()
            }
            PaletteEntry::Tool(brush) => state.brush == brush,
        }
    }

    /// The short label shown beneath/beside the swatch.
    pub(crate) fn label(self) -> &'static str {
        match self {
            PaletteEntry::Material(i) => material_label(i),
            PaletteEntry::Tool(brush) => brush.label(),
        }
    }

    /// The swatch colour: a material's own colour, or a representative tint per device/part. Identity
    /// is always paired with the text label, so colour is never the sole carrier of meaning.
    pub(crate) fn swatch_color(self) -> Color {
        match self {
            PaletteEntry::Material(i) => material_color(PALETTE[i % PALETTE.len()].1),
            PaletteEntry::Tool(brush) => match brush {
                Brush::Voxel => Color::srgb(0.70, 0.72, 0.78),
                Brush::ControlPoint => Color::srgb(0.30, 0.80, 0.90),
                Brush::Computer => Color::srgb(0.30, 0.80, 0.45),
                Brush::Battery => Color::srgb(0.90, 0.80, 0.20),
                Brush::Engine => Color::srgb(1.00, 0.45, 0.10),
                Brush::Tank => Color::srgb(0.55, 0.62, 0.82),
                Brush::Wheel {
                    preset,
                    steer: false,
                } => preset.color(),
                Brush::Wheel {
                    preset,
                    steer: true,
                } => {
                    // Steer variants tinted bluer so they read apart from the drive-only siblings.
                    let c = preset.color().to_srgba();
                    Color::srgb(c.red + 0.08, c.green + 0.10, c.blue + 0.22)
                }
                Brush::Suspension => Color::srgb(0.55, 0.45, 0.20),
                Brush::Seat => Color::srgb(0.50, 0.35, 0.20),
                Brush::Antenna => Color::srgb(0.82, 0.82, 0.86),
                Brush::SolarPanel => Color::srgb(0.12, 0.22, 0.62),
                Brush::Bumper => Color::srgb(0.72, 0.16, 0.16),
            },
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
    gamepads: Query<&Gamepad>,
    pad_map: Res<crate::gamepad::GamepadMap>,
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
    // Gamepad orbit (WI 617): right stick orbits (yaw/pitch), bumpers zoom; same clamps as the keys.
    // Yaw negated to match the Test/flight free-look convention (WI 665) — stick-right orbits right.
    let pad = pad_map.sample(&gamepads);
    cam.yaw -= pad.cam_yaw * dt * 2.0;
    cam.pitch = (cam.pitch - pad.cam_pitch * dt * 2.0).clamp(-1.4, 1.4);
    if pad.zoom_in {
        cam.dist = (cam.dist - dt * cam.dist * 1.5).max(0.2);
    }
    if pad.zoom_out {
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
            preset: WheelPreset::Road,
            steer: false,
        };
    } else if keys.just_pressed(KeyCode::Digit7) {
        state.brush = Brush::Wheel {
            preset: WheelPreset::Road,
            steer: true,
        };
    } else if keys.just_pressed(KeyCode::KeyU) {
        state.brush = Brush::Suspension;
    } else if keys.just_pressed(KeyCode::Digit8) {
        state.brush = Brush::Seat;
    } else if keys.just_pressed(KeyCode::Digit9) {
        state.brush = Brush::Antenna;
    } else if keys.just_pressed(KeyCode::Digit0) {
        state.brush = Brush::SolarPanel;
    } else if keys.just_pressed(KeyCode::Minus) {
        state.brush = Brush::Bumper;
    }

    // Cycle the motor tier a placed Engine gets (WI 652).
    if keys.just_pressed(KeyCode::KeyM) {
        let i = MotorTier::ALL
            .iter()
            .position(|&t| t == state.motor)
            .unwrap_or(0);
        state.motor = MotorTier::ALL[(i + 1) % MotorTier::ALL.len()];
    }

    // Place the active brush at the cursor (keyboard fallback; the mouse is the primary path).
    if keys.just_pressed(KeyCode::Space) {
        let cell = state.cursor;
        let material = PALETTE[state.material].1;
        let brush = state.brush;
        let mount = (cell.as_dvec3() + DVec3::splat(0.5)) * state.craft.cell_size;
        let motor = state.motor;
        place_brush(&mut state.craft, brush, material, cell, cell, mount, motor);
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

    // Save the whole build as a craft / load one back into Build (WI 637). Unlike a subassembly
    // (which loads into the insert buffer), opening a craft *replaces* the current build.
    if keys.just_pressed(KeyCode::KeyK) {
        save(&state.craft, Kind::Craft, CRAFT_PATH);
    }
    if keys.just_pressed(KeyCode::KeyO) {
        match load(CRAFT_PATH) {
            Some(c) => {
                info!("loaded craft ({} voxels) into Build", c.voxels.len());
                state.craft = c;
            }
            None => warn!("no craft to load at {CRAFT_PATH}"),
        }
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

/// True while the mouse pointer is over the Build palette UI (WI 613). Set by the workshop's palette
/// systems and read by [`mouse_build`] so a click that lands on the palette selects a brush instead
/// of also editing the world behind it (the input-isolation invariant).
#[derive(Resource, Default)]
pub(crate) struct PointerOnPalette(pub bool);

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
    pointer_on_palette: Res<PointerOnPalette>,
    mut state: ResMut<EditorState>,
) {
    // A click that lands on the palette selects a brush; it must not also edit the world (WI 613).
    if pointer_on_palette.0 {
        return;
    }
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
        let motor = state.motor;
        place_brush(
            &mut state.craft,
            brush,
            material,
            h.add_cell,
            device_cell,
            wheel_mount,
            motor,
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

/// Serializes a craft as a `kind`-tagged document. Pure (no I/O), so the save/load round-trip is
/// unit-testable without touching the filesystem (WI 637).
fn craft_to_json(craft: &VoxelCraft, kind: Kind) -> Result<String, FormatError> {
    let payload = match kind {
        Kind::Subassembly => Payload::Subassembly(craft_subgraph(craft)),
        Kind::Blueprint => Payload::Blueprint(craft_subgraph(craft)),
        _ => Payload::Craft(craft_subgraph(craft)),
    };
    SavedDocument::new(payload).to_json()
}

/// Deserializes any craft-scope document back into its craft. Pure counterpart to [`craft_to_json`].
fn craft_from_json(json: &str) -> Option<VoxelCraft> {
    match SavedDocument::from_json(json).ok()?.payload {
        Payload::Craft(c) | Payload::Subassembly(c) | Payload::Blueprint(c) => Some(c.craft),
        Payload::WorldSave(_) => None,
    }
}

fn save(craft: &VoxelCraft, kind: Kind, path: &str) {
    let result = craft_to_json(craft, kind)
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
    craft_from_json(&fs::read_to_string(path).ok()?)
}

fn cell_center(c: IVec3, cell_size: f64) -> Vec3 {
    ((c.as_dvec3() + DVec3::splat(0.5)) * cell_size).as_vec3()
}

/// The editor gizmo / palette-swatch colour for a material — the same per-material appearance the
/// skin renders with (WI 614), so the palette and the built craft agree.
fn material_color(m: Material) -> Color {
    crate::voxel_skin::material_visual(m).tint
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
    // Attached wheel parts (dark spheres at their continuous mount; steer wheels tinted). Both the
    // legacy monolithic wheel and the new rim+tire components draw the wheel; a suspension draws a
    // small amber marker so an added strut is visible (WI 630).
    for p in &state.craft.parts {
        match p.kind {
            PartKind::Wheel(spec) => {
                let color = if spec.steer {
                    Color::srgb(0.15, 0.18, 0.28)
                } else {
                    Color::srgb(0.12, 0.12, 0.14)
                };
                gizmos.sphere(p.mount.as_vec3(), spec.radius as f32, color);
            }
            PartKind::Rim(r) => {
                let color = if r.steer {
                    Color::srgb(0.15, 0.18, 0.28)
                } else {
                    Color::srgb(0.12, 0.12, 0.14)
                };
                gizmos.sphere(p.mount.as_vec3(), r.radius as f32, color);
            }
            PartKind::Suspension(_) => {
                gizmos.sphere(p.mount.as_vec3(), s * 0.4, Color::srgb(0.85, 0.65, 0.2));
            }
            _ => {}
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
    fn palette_entries_round_trip_to_selection() {
        // Applying each palette entry to a fresh state must select it, and exactly that entry must
        // then report active — for every block, device, and part (WI 613).
        for (_group, entries) in PALETTE_GROUPS {
            for &entry in *entries {
                let mut state = EditorState::default();
                entry.apply(&mut state);
                assert!(
                    entry.is_active(&state),
                    "applied entry must read back as active: {}",
                    entry.label()
                );
                // No other entry claims to be active at the same time.
                let active: Vec<&str> = PALETTE_GROUPS
                    .iter()
                    .flat_map(|(_, es)| es.iter())
                    .filter(|e| e.is_active(&state))
                    .map(|e| e.label())
                    .collect();
                assert_eq!(
                    active,
                    vec![entry.label()],
                    "exactly one entry active for {}",
                    entry.label()
                );
            }
        }
    }

    /// A small chassis to attach wheels to (gives the craft mass so assembly succeeds).
    fn chassis() -> VoxelCraft {
        let mut craft = VoxelCraft::new(0.5);
        for x in 0..3 {
            for z in 0..4 {
                craft.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::COMPOSITE,
                });
            }
        }
        craft
    }

    #[test]
    fn wheel_brush_places_a_complete_station() {
        // Placing a wheel brush drops a rim + tire sharing one station id — a complete wheel (WI 630).
        let mut craft = chassis();
        let mount = DVec3::new(0.5, -0.2, 0.5);
        place_brush(
            &mut craft,
            Brush::Wheel {
                preset: WheelPreset::OffRoad,
                steer: true,
            },
            Material::COMPOSITE,
            IVec3::ZERO,
            IVec3::ZERO,
            mount,
            MotorTier::Standard,
        );
        let rim = craft
            .parts
            .iter()
            .find(|p| matches!(p.kind, PartKind::Rim(_)))
            .unwrap();
        let tire = craft
            .parts
            .iter()
            .find(|p| matches!(p.kind, PartKind::Tire(_)))
            .unwrap();
        assert_eq!(rim.station, tire.station);
        assert!(rim.station.is_some());
        let asm = sounding_sim::rover::assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81)
            .expect("a placed wheel ⇒ a rover");
        assert_eq!(asm.rover.wheels.len(), 1);
        assert_eq!(asm.steer, vec![0]); // the steer flag carried onto the rim
    }

    #[test]
    fn suspension_joins_the_wheels_station_and_makes_it_sprung() {
        // A suspension placed at the same corner joins the wheel's station (one station, three
        // components) and the assembled wheel is sprung rather than riding on the tire (WI 630).
        let mut craft = chassis();
        let mount = DVec3::new(0.5, -0.2, 0.5);
        place_brush(
            &mut craft,
            Brush::Wheel {
                preset: WheelPreset::Road,
                steer: false,
            },
            Material::COMPOSITE,
            IVec3::ZERO,
            IVec3::ZERO,
            mount,
            MotorTier::Standard,
        );
        place_brush(
            &mut craft,
            Brush::Suspension,
            Material::COMPOSITE,
            IVec3::ZERO,
            IVec3::ZERO,
            mount,
            MotorTier::Standard,
        );
        let ids: std::collections::BTreeSet<_> =
            craft.parts.iter().filter_map(|p| p.station).collect();
        assert_eq!(ids.len(), 1, "all three components share one station");
        assert_eq!(craft.parts.len(), 3, "rim + tire + suspension");
        let asm =
            sounding_sim::rover::assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81).unwrap();
        assert!(
            !asm.rover.wheels[0].rigid_suspension,
            "a station with a suspension is sprung"
        );
    }

    #[test]
    fn reselecting_a_preset_replaces_the_rim_and_tire_in_place() {
        // Placing a different preset at an existing station swaps its rim+tire (not a second wheel),
        // and the tire character changes (WI 630).
        let mut craft = chassis();
        let mount = DVec3::new(0.5, -0.2, 0.5);
        let place = |craft: &mut VoxelCraft, preset| {
            place_brush(
                craft,
                Brush::Wheel {
                    preset,
                    steer: false,
                },
                Material::COMPOSITE,
                IVec3::ZERO,
                IVec3::ZERO,
                mount,
                MotorTier::Standard,
            );
        };
        place(&mut craft, WheelPreset::Road);
        place(&mut craft, WheelPreset::Slick);
        assert_eq!(craft.parts.len(), 2, "still one rim + one tire");
        let asm =
            sounding_sim::rover::assemble_rover(&craft, DVec3::new(0.0, 5.0, 0.0), 9.81).unwrap();
        assert_eq!(asm.rover.wheels.len(), 1);
        // Slick grip (1.6) replaced road grip (1.0).
        assert!((asm.rover.wheels[0].grip_scale - 1.6).abs() < 1e-9);
    }

    #[test]
    fn voxel_brush_activates_the_selected_material_entry() {
        // With the Voxel brush, the active palette entry is the *material* being placed, so the
        // palette shows which block lands (WI 613).
        let state = EditorState {
            brush: Brush::Voxel,
            material: 2,
            ..Default::default()
        };
        assert!(PaletteEntry::Material(2).is_active(&state));
        assert!(!PaletteEntry::Material(0).is_active(&state));
    }

    #[test]
    fn craft_save_load_round_trips_a_full_build() {
        // A full build — chassis voxels + a device + a complete wheel station — survives the craft
        // save/load path identically (WI 637). Whole-craft equality covers mass/inertia/parts/stations.
        let mut craft = chassis();
        craft.devices.push(Device::battery(
            IVec3::new(0, 0, 0),
            device_mass(DeviceKind::Battery, craft.cell_size),
            BatterySpec::full(20.0),
        ));
        place_brush(
            &mut craft,
            Brush::Wheel {
                preset: WheelPreset::Road,
                steer: true,
            },
            Material::COMPOSITE,
            IVec3::ZERO,
            IVec3::ZERO,
            DVec3::new(0.5, -0.2, 0.5),
            MotorTier::Standard,
        );

        let json = craft_to_json(&craft, Kind::Craft).expect("serialize");
        let back = craft_from_json(&json).expect("a craft document loads back");

        // Structural identity: voxels and devices survive exactly; the wheel station (rim + tire,
        // same station id) and cell size are preserved.
        assert_eq!(back.voxels, craft.voxels);
        assert_eq!(back.devices, craft.devices);
        assert_eq!(back.cell_size, craft.cell_size);
        assert_eq!(back.parts.len(), craft.parts.len());
        let stations: Vec<_> = back.parts.iter().map(|p| p.station).collect();
        assert_eq!(
            stations,
            vec![Some(0), Some(0)],
            "rim + tire share one station"
        );
        // Mass/inertia are preserved (the acceptance phrasing) to within float tolerance.
        let mp0 = craft.mass_properties().unwrap();
        let mp1 = back.mass_properties().unwrap();
        assert!((mp0.mass - mp1.mass).abs() < 1e-9);
        assert!((mp0.center_of_mass - mp1.center_of_mass).length() < 1e-9);
        // It is specifically a craft-kind document (not a blueprint/subassembly).
        assert_eq!(SavedDocument::from_json(&json).unwrap().kind(), Kind::Craft);
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

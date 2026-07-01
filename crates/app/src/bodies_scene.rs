//! Body viewer scene (`-- bodies`, WI 762): the "generate some, keep some" loop.
//!
//! Generates a [`BodyAsset`](sounding_sim::body_asset::BodyAsset) from a seed +
//! archetype ([`sounding_sim::bodygen`]) and shows it as a **coarse** tinted
//! sphere (colour derived from the asset's medium; a translucent shell stands in
//! for an atmosphere). `Space` regenerates with the next seed, `Tab` cycles the
//! archetype, and `K` **keeps** (saves) the current body to `saves/bodies` via the
//! body library. Deliberately minimal — the real procedural surface arrives in
//! WI 763/764; this makes bodies inspectable and keepable now.
//!
//! App-side only. All generation/persistence is the headless `sounding_sim`
//! (unit-tested); this scene is the view + input shell.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use std::path::PathBuf;

use sounding_sim::body_asset::BodyAsset;
use sounding_sim::body_library::save_body;
use sounding_sim::bodygen::{generate, Archetype};

/// Where kept bodies are saved (parallel to the craft library's `saves/crafts`).
const SAVES_DIR: &str = "saves/bodies";

pub struct BodiesScenePlugin;

impl Plugin for BodiesScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BodyView>()
            .init_resource::<BodiesCam>()
            .add_systems(Startup, setup)
            .add_systems(Update, (bodies_input, apply_body, orbit_camera));
    }
}

/// The current generate/keep state.
#[derive(Resource)]
struct BodyView {
    seed: u64,
    archetype_idx: usize,
    asset: BodyAsset,
    status: String,
    /// Set when seed/archetype changed and the view must be rebuilt.
    dirty: bool,
}

impl Default for BodyView {
    fn default() -> Self {
        let seed = 1;
        let archetype_idx = 0;
        Self {
            seed,
            archetype_idx,
            asset: generate(seed, Archetype::ALL[archetype_idx]),
            status: "generate: Space (next seed) \u{b7} Tab (archetype) \u{b7} K (keep)"
                .to_string(),
            dirty: false,
        }
    }
}

impl BodyView {
    fn archetype(&self) -> Archetype {
        Archetype::ALL[self.archetype_idx]
    }
    fn regenerate(&mut self) {
        self.asset = generate(self.seed, self.archetype());
        self.dirty = true;
    }
}

/// Handles to the mutable materials so regeneration can re-tint without respawning.
#[derive(Resource)]
struct BodyMaterials {
    body: Handle<StandardMaterial>,
    atmo: Handle<StandardMaterial>,
}

#[derive(Component)]
struct AtmoShell;

#[derive(Component)]
struct HudText;

/// Orbit camera state.
#[derive(Resource)]
struct BodiesCam {
    yaw: f32,
    pitch: f32,
    dist: f32,
}

impl Default for BodiesCam {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.35,
            dist: 5.0,
        }
    }
}

/// A coarse base colour for a body, derived from its medium (WI 764 will replace
/// this with real render params). Ocean ⇒ blue; atmosphere-only ⇒ tan; bare ⇒ grey.
/// A small seed-driven jitter gives variety.
fn body_tint(asset: &BodyAsset) -> Color {
    let j = ((asset.surface.seed & 0xFF) as f32 / 255.0 - 0.5) * 0.12;
    let m = &asset.fluid_medium;
    let (r, g, b) = if m.ocean_surface_density > 0.0 {
        (0.10, 0.32, 0.72)
    } else if m.atmosphere_surface_density > 0.0 {
        (0.62, 0.50, 0.36)
    } else {
        (0.52, 0.52, 0.55)
    };
    Color::srgb(
        (r + j).clamp(0.0, 1.0),
        (g + j).clamp(0.0, 1.0),
        (b + j).clamp(0.0, 1.0),
    )
}

/// The translucent atmosphere-shell colour, or `None` for an airless body.
fn atmo_tint(asset: &BodyAsset) -> Option<Color> {
    let m = &asset.fluid_medium;
    (m.atmosphere_surface_density > 0.0).then(|| {
        if m.ocean_surface_density > 0.0 {
            Color::srgba(0.45, 0.65, 0.95, 0.28)
        } else {
            Color::srgba(0.75, 0.68, 0.55, 0.22)
        }
    })
}

fn hud_text(view: &BodyView) -> String {
    let a = &view.asset;
    let m = &a.fluid_medium;
    let medium = match (
        m.atmosphere_surface_density > 0.0,
        m.ocean_surface_density > 0.0,
    ) {
        (false, _) => "airless",
        (true, false) => "atmosphere",
        (true, true) => "atmosphere + ocean",
    };
    let g = a.mu / (a.radius * a.radius);
    format!(
        "BODY GENERATOR (-- bodies)\n\
         name: {name}\narchetype: {arch}\nseed: {seed}\n\
         radius: {rkm:.0} km\nsurface gravity: {g:.2} m/s^2\nmedium: {medium}\n\n\
         Space next seed \u{b7} Tab archetype \u{b7} K keep \u{b7} middle-drag orbit \u{b7} scroll zoom\n\
         {status}",
        name = a.name,
        arch = view.archetype().label(),
        seed = view.seed,
        rkm = a.radius / 1000.0,
        g = g,
        medium = medium,
        status = view.status,
    )
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    view: Res<BodyView>,
) {
    // Body sphere (fixed display radius — real radius is shown in the HUD).
    let body_mat = materials.add(StandardMaterial {
        base_color: body_tint(&view.asset),
        perceptual_roughness: 0.85,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere::new(1.5)))),
        MeshMaterial3d(body_mat.clone()),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Translucent atmosphere shell (visible only when the body has an atmosphere).
    let atmo_color = atmo_tint(&view.asset);
    let atmo_mat = materials.add(StandardMaterial {
        base_color: atmo_color.unwrap_or(Color::NONE),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Sphere::new(1.62)))),
        MeshMaterial3d(atmo_mat.clone()),
        Transform::from_xyz(0.0, 0.0, 0.0),
        AtmoShell,
        if atmo_color.is_some() {
            Visibility::Visible
        } else {
            Visibility::Hidden
        },
    ));

    commands.insert_resource(BodyMaterials {
        body: body_mat,
        atmo: atmo_mat,
    });

    // Lights + camera.
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(6.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::default(),
        AmbientLight {
            brightness: 260.0,
            ..default()
        },
    ));

    // HUD.
    commands.spawn((
        Text::new(hud_text(&view)),
        HudText,
        TextFont {
            font_size: 15.0,
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.93, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn bodies_input(keys: Res<ButtonInput<KeyCode>>, mut view: ResMut<BodyView>) {
    if keys.just_pressed(KeyCode::Space) {
        view.seed = view.seed.wrapping_add(1);
        view.regenerate();
    }
    if keys.just_pressed(KeyCode::Tab) {
        view.archetype_idx = (view.archetype_idx + 1) % Archetype::ALL.len();
        view.regenerate();
    }
    if keys.just_pressed(KeyCode::KeyK) {
        let dir = PathBuf::from(SAVES_DIR);
        match save_body(&dir, &view.asset) {
            Ok(path) => view.status = format!("kept: {}", path.display()),
            Err(e) => view.status = format!("keep failed: {e}"),
        }
    }
}

/// Rebuilds the view (materials, shell visibility, HUD) after a regenerate.
fn apply_body(
    mut view: ResMut<BodyView>,
    mats: Option<Res<BodyMaterials>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut shell: Query<&mut Visibility, With<AtmoShell>>,
    mut hud: Query<&mut Text, With<HudText>>,
) {
    // HUD always reflects the latest status (e.g. after a keep); cheap.
    if let Ok(mut text) = hud.single_mut() {
        **text = hud_text(&view);
    }
    if !view.dirty {
        return;
    }
    if let Some(mats) = mats {
        if let Some(m) = materials.get_mut(&mats.body) {
            m.base_color = body_tint(&view.asset);
        }
        let atmo = atmo_tint(&view.asset);
        if let Some(m) = materials.get_mut(&mats.atmo) {
            m.base_color = atmo.unwrap_or(Color::NONE);
        }
        if let Ok(mut vis) = shell.single_mut() {
            *vis = if atmo.is_some() {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
    view.dirty = false;
}

fn orbit_camera(
    mut cam: ResMut<BodiesCam>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    if buttons.pressed(MouseButton::Middle) {
        cam.yaw -= motion.delta.x * 0.006;
        cam.pitch = (cam.pitch + motion.delta.y * 0.006).clamp(-1.4, 1.4);
    }
    if scroll.delta.y != 0.0 {
        cam.dist = (cam.dist * (1.0 - scroll.delta.y * 0.12)).clamp(2.5, 20.0);
    }
    let Ok(mut tf) = camera.single_mut() else {
        return;
    };
    let (sy, cy) = cam.yaw.sin_cos();
    let (sp, cp) = cam.pitch.sin_cos();
    let offset = Vec3::new(sy * cp, sp, cy * cp) * cam.dist;
    *tf = Transform::from_translation(offset).looking_at(Vec3::ZERO, Vec3::Y);
}

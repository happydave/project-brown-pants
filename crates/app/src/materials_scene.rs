//! Material preview scene (`-- materials`): renders a generated PBR material set
//! (from the asset-harness `pbr-materials` track) on lit geometry so the maps can be
//! validated in Sounding's real Bevy pipeline — colour space (sRGB albedo vs linear
//! normal/metallic-roughness/occlusion), mesh tangents (required for normal mapping),
//! and the normal-map green-channel convention.
//!
//! Not a simulation scene: it spawns its own camera + lights and a few primitives, and
//! touches no `sounding_sim` state. Assets live in `crates/app/assets/materials/`.

use bevy::image::ImageLoaderSettings;
use bevy::prelude::*;

/// The material set to preview (basename under `assets/materials/`).
const MATERIAL: &str = "hull_panel";

/// Slowly-rotating showpiece geometry.
#[derive(Component)]
struct Spin(f32);

/// The material preview scene.
pub struct MaterialsScenePlugin;

impl Plugin for MaterialsScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup).add_systems(Update, spin);
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Base colour is sRGB (default loader); normal / metallic-roughness / occlusion are
    // non-colour data and MUST load as linear (is_srgb = false), or lighting goes wrong.
    let linear = |path: String| {
        asset_server.load_with_settings(path, |s: &mut ImageLoaderSettings| s.is_srgb = false)
    };
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(asset_server.load(format!("materials/{MATERIAL}_albedo.png"))),
        normal_map_texture: Some(linear(format!("materials/{MATERIAL}_normal.png"))),
        metallic_roughness_texture: Some(linear(format!(
            "materials/{MATERIAL}_metallic_roughness.png"
        ))),
        occlusion_texture: Some(linear(format!("materials/{MATERIAL}_occlusion.png"))),
        // Textures supply the variation; keep the multipliers neutral-high.
        perceptual_roughness: 1.0,
        metallic: 1.0,
        ..default()
    });

    // Normal mapping needs tangents; primitive meshes don't generate them by default.
    let sphere = meshes.add(
        Sphere::new(1.4)
            .mesh()
            .uv(48, 32)
            .with_generated_tangents()
            .expect("sphere has normals + UVs"),
    );
    let cube = meshes.add(
        Mesh::from(Cuboid::new(2.4, 2.4, 2.4))
            .with_generated_tangents()
            .expect("cuboid has normals + UVs"),
    );
    let ground = meshes.add(
        Mesh::from(Plane3d::default().mesh().size(16.0, 16.0))
            .with_generated_tangents()
            .expect("plane has normals + UVs"),
    );

    commands.spawn((
        Mesh3d(sphere),
        MeshMaterial3d(material.clone()),
        Transform::from_xyz(-2.6, 1.5, 0.0),
        Spin(0.4),
    ));
    commands.spawn((
        Mesh3d(cube),
        MeshMaterial3d(material.clone()),
        Transform::from_xyz(2.6, 1.4, 0.0),
        Spin(-0.3),
    ));
    commands.spawn((
        Mesh3d(ground),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Sun + camera.
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(5.0, 9.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.2, 8.5).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
        // AmbientLight is a per-camera component in Bevy 0.18; fill so shadowed faces read.
        AmbientLight {
            brightness: 220.0,
            ..default()
        },
    ));

    commands.spawn((
        Text::new(format!(
            "material preview: {MATERIAL} (asset-harness, Apache/MIT)\n\
             albedo sRGB; normal / metallic-roughness / occlusion linear; tangents generated\n\
             if relief looks inverted, re-derive the normal with --flip-g"
        )),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.95, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn spin(time: Res<Time>, mut q: Query<(&mut Transform, &Spin)>) {
    for (mut t, s) in &mut q {
        t.rotate_y(s.0 * time.delta_secs());
    }
}

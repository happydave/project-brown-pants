//! Terrain-mesh preview scene (`-- terrainmesh`): loads a generated terrain chunk (asset-harness
//! `3d-static-props`, **heightmap-displacement** pipeline) as a glTF `SceneRoot` and shows it, to
//! validate glb loading + materials in Sounding's real Bevy pipeline.
//!
//! The mesh is a clean displaced grid — ~10×10 units in the XZ plane, Y-up, low relief, already
//! centered at the origin — so it loads near-identity and slow-spins in place under sun + fill
//! light. (Superseded the earlier MoGe relief, which stretched on deep scenes.) Touches no
//! `sounding_sim` state; assets in `crates/app/assets/models/`.

use bevy::gltf::GltfAssetLabel;
use bevy::prelude::*;

const TERRAIN: &str = "models/terrain.glb";

#[derive(Component)]
struct Spin(f32);

/// The terrain-mesh preview scene.
pub struct TerrainMeshScenePlugin;

impl Plugin for TerrainMeshScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup).add_systems(Update, spin);
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let scene = asset_server.load(GltfAssetLabel::Scene(0).from_asset(TERRAIN));
    // Already centered + Y-up (~10u, low relief): load near-identity and spin in place.
    commands.spawn((SceneRoot(scene), Transform::default(), Spin(0.25)));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(6.0, 10.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 8.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
        AmbientLight {
            brightness: 280.0,
            ..default()
        },
    ));
    commands.spawn((
        Text::new(
            "terrain (asset-harness heightmap-displacement) — clean grid, Z-Image albedo\n\
             ~40k tris; slow-spin",
        ),
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

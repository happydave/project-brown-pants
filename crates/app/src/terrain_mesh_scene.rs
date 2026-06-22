//! Terrain-mesh preview scene (`-- terrainmesh`): loads a generated MoGe terrain relief (from the
//! asset-harness `3d-static-props` track, Blender-decimated) as a glTF `SceneRoot` and shows it,
//! to validate glb loading, scale, and orientation in Sounding's real Bevy pipeline.
//!
//! The MoGe mesh is **camera-space 2.5D** (~544×572×385 units, reconstructed looking down −Z), so
//! it is recentered + scaled down and slow-spun about its centre (a parent pivot) so the whole
//! relief is visible. Touches no `sounding_sim` state; assets in `crates/app/assets/models/`.

use bevy::gltf::GltfAssetLabel;
use bevy::prelude::*;

const TERRAIN: &str = "models/moge_terrain.glb";
/// Mesh centre in its own (camera-space) coordinates — from the glb POSITION min/max.
const CENTER: Vec3 = Vec3::new(112.9, -98.5, -192.4);
/// Scale the ~550-unit mesh down to ~11 units across.
const SCALE: f32 = 0.02;

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
    // Parent pivot at the origin spins in place; child carries the recenter + scale.
    commands
        .spawn((Transform::default(), Visibility::default(), Spin(0.3)))
        .with_children(|p| {
            p.spawn((
                SceneRoot(scene),
                Transform::from_translation(-CENTER * SCALE).with_scale(Vec3::splat(SCALE)),
            ));
        });

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(8.0, 14.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 6.0, 16.0).looking_at(Vec3::ZERO, Vec3::Y),
        AmbientLight {
            brightness: 300.0,
            ..default()
        },
    ));
    commands.spawn((
        Text::new(
            "MoGe terrain relief (asset-harness, MIT) — decimated ~102k tris, textured\n\
             camera-space 2.5D; scale/center constants live in terrain_mesh_scene.rs",
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

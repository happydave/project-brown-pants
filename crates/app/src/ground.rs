//! Textured ground (WI 588): a reusable tiled rocky-ground surface for the flat-ground
//! scenes, replacing the flat-colour planet sphere under the craft. Uses the asset-harness
//! `rocky_ground` PBR set through the repeat-wrap loader (so the texture tiles per metre,
//! not stretched across the whole ground), with a generated tangent basis for normal
//! mapping.

use bevy::math::{Affine2, DVec3};
use bevy::prelude::*;
use sounding_sim::frame::{FrameId, WorldPos};

use crate::floating_origin::WorldPlacement;
use crate::voxel_skin::load_repeat;

/// The ground PBR set basename under `assets/materials/`.
const GROUND_SET: &str = "rocky_ground";
/// Side length of the textured ground patch, metres (finite; the far body shows beyond it).
const GROUND_SIZE: f32 = 3_000.0;
/// Texture repeat scale, metres per tile.
const TILE_METRES: f32 = 6.0;
/// Small lift above the surface (y=0) so the ground patch sits just above the planet
/// sphere's top and avoids z-fighting with it.
const GROUND_LIFT: f64 = 0.05;

/// Build the tiled rocky-ground `StandardMaterial`: repeat-wrapped maps with a UV scale so
/// the texture tiles roughly every [`TILE_METRES`] over the patch.
fn ground_material(
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
) -> Handle<StandardMaterial> {
    let tiles = GROUND_SIZE / TILE_METRES;
    materials.add(StandardMaterial {
        base_color_texture: Some(load_repeat(
            asset_server,
            format!("materials/{GROUND_SET}_albedo.png"),
            true,
        )),
        normal_map_texture: Some(load_repeat(
            asset_server,
            format!("materials/{GROUND_SET}_normal.png"),
            false,
        )),
        metallic_roughness_texture: Some(load_repeat(
            asset_server,
            format!("materials/{GROUND_SET}_metallic_roughness.png"),
            false,
        )),
        occlusion_texture: Some(load_repeat(
            asset_server,
            format!("materials/{GROUND_SET}_occlusion.png"),
            false,
        )),
        // Tile the texture across the patch (the maps are repeat-wrapped, WI 587).
        uv_transform: Affine2::from_scale(Vec2::splat(tiles)),
        perceptual_roughness: 1.0,
        metallic: 1.0,
        ..default()
    })
}

/// Spawn a tiled textured ground patch at the local surface (`y = 0`, lifted a hair above
/// the planet sphere's top), floating-origin anchored so it stays put as the craft moves.
/// Call once from a flat-ground scene's `setup`.
pub fn spawn_ground(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    asset_server: &AssetServer,
) {
    let mesh = meshes.add(
        Mesh::from(Plane3d::default().mesh().size(GROUND_SIZE, GROUND_SIZE))
            .with_generated_tangents()
            .expect("plane has normals + UVs"),
    );
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(ground_material(asset_server, materials)),
        Transform::default(),
        WorldPlacement(WorldPos::new(
            FrameId::CENTRAL_BODY,
            DVec3::new(0.0, GROUND_LIFT, 0.0),
        )),
    ));
}

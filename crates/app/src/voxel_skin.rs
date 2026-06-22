//! App-side voxel skinning (WI 582/583): turn the headless mesh data from
//! `sounding_sim::voxel_mesh` into a Bevy `Mesh` (with a generated tangent basis for
//! normal mapping) and bind the asset-harness PBR material set. The reusable
//! [`VoxelSkin`] selector lets any scene choose how a craft's render mesh is built.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageLoaderSettings;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use sounding_sim::voxel::{Material, VoxelCraft};
use sounding_sim::voxel_mesh::{blocky_mesh, greedy_mesh, SkinMesh};

/// How a craft's render mesh is built from its voxel lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelSkin {
    /// One textured cube per occupied cell — the Stormworks-style blocky skin (WI 582).
    Blocky,
    /// Coplanar faces merged into panels — the Starbase-style hull, the primary look (WI 583).
    Hull,
}

impl VoxelSkin {
    /// Generate the engine-agnostic mesh data for this skin (headless core).
    fn mesh_data(self, craft: &VoxelCraft) -> SkinMesh {
        match self {
            VoxelSkin::Blocky => blocky_mesh(craft),
            VoxelSkin::Hull => greedy_mesh(craft),
        }
    }
}

/// Build a Bevy `Mesh` for `craft` rendered with `skin`, with a generated tangent basis
/// (required for normal mapping — the `-- materials` validation flagged this).
pub fn build_skin_mesh(craft: &VoxelCraft, skin: VoxelSkin) -> Mesh {
    let data = skin.mesh_data(craft);
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs);
    mesh.insert_indices(Indices::U32(data.indices));
    mesh.with_generated_tangents()
        .expect("skin mesh has positions, normals, and UVs")
}

/// The asset material-set basename for a structural cell (the data-shaped cell-material →
/// material-set seam, WI 582). One set for all structural cells for now; a per-material /
/// asset-pack catalog is future work (ties into the content catalog, WI 547).
pub fn material_set_for(_material: Material) -> &'static str {
    "hull_panel"
}

/// Build the `StandardMaterial` for a named PBR set: albedo sRGB; normal /
/// metallic-roughness / occlusion are non-colour data and load linear (the `-- materials`
/// convention). Textures supply the variation, so the multipliers stay neutral-high.
pub fn pbr_material(
    set: &str,
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
) -> Handle<StandardMaterial> {
    let linear = |path: String| {
        asset_server.load_with_settings(path, |s: &mut ImageLoaderSettings| s.is_srgb = false)
    };
    materials.add(StandardMaterial {
        base_color_texture: Some(asset_server.load(format!("materials/{set}_albedo.png"))),
        normal_map_texture: Some(linear(format!("materials/{set}_normal.png"))),
        metallic_roughness_texture: Some(linear(format!("materials/{set}_metallic_roughness.png"))),
        occlusion_texture: Some(linear(format!("materials/{set}_occlusion.png"))),
        perceptual_roughness: 1.0,
        metallic: 1.0,
        ..default()
    })
}

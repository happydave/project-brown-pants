//! App-side voxel skinning (WI 582/583): turn the headless mesh data from
//! `sounding_sim::voxel_mesh` into a Bevy `Mesh` (with a generated tangent basis for
//! normal mapping) and bind the asset-harness PBR material set. The reusable
//! [`VoxelSkin`] selector lets any scene choose how a craft's render mesh is built.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor};
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

/// Load a texture with a **repeat-wrapped**, linear-filtered sampler so tiled UVs (UVs that
/// exceed `1.0`) repeat the texture instead of clamping/smearing the edge texel (WI 587).
/// `srgb` true for colour (albedo), false for non-colour data (normal / metallic-roughness
/// / occlusion). Shared by the craft skins and the textured ground (WI 588).
pub fn load_repeat(asset_server: &AssetServer, path: String, srgb: bool) -> Handle<Image> {
    asset_server.load_with_settings(path, move |s: &mut ImageLoaderSettings| {
        s.is_srgb = srgb;
        let mut desc = ImageSamplerDescriptor::linear();
        desc.address_mode_u = ImageAddressMode::Repeat;
        desc.address_mode_v = ImageAddressMode::Repeat;
        s.sampler = ImageSampler::Descriptor(desc);
    })
}

/// Build the `StandardMaterial` for a named PBR set (albedo sRGB; normal /
/// metallic-roughness / occlusion linear), repeat-wrapped so the greedy hull's tiled UVs
/// repeat per cell. Textures supply the variation, so multipliers stay neutral-high.
pub fn pbr_material(
    set: &str,
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color_texture: Some(load_repeat(
            asset_server,
            format!("materials/{set}_albedo.png"),
            true,
        )),
        normal_map_texture: Some(load_repeat(
            asset_server,
            format!("materials/{set}_normal.png"),
            false,
        )),
        metallic_roughness_texture: Some(load_repeat(
            asset_server,
            format!("materials/{set}_metallic_roughness.png"),
            false,
        )),
        occlusion_texture: Some(load_repeat(
            asset_server,
            format!("materials/{set}_occlusion.png"),
            false,
        )),
        perceptual_roughness: 1.0,
        metallic: 1.0,
        ..default()
    })
}

//! Reusable surface-streaming helper (WI 775) — the WI 764 spherified-cube quadtree
//! CDLOD streamer, factored out so a scene can render a planet surface without
//! copy-pasting the loop. Used by the workshop's moon Test.
//!
//! This variant renders **anchor-relative** (no floating-origin plugin): chunk
//! entities carry their body-centred [`ChunkCenter`] and a caller-run
//! [`reposition`] sets each `Transform` relative to a chosen anchor each frame — so
//! it drops straight into the workshop's existing rover-anchored render frame (rover
//! meshes + fixed camera untouched). [`stream`] advances the resident/in-flight/
//! ready chunk set given the field and the LOD anchor (in body-centred coords);
//! chunks are meshed off-thread, uploaded under a per-frame budget, and despawned
//! with coverage gating (no LOD-transition holes, WI 771). Chunk entities carry
//! [`StreamedChunk`] so a caller can despawn them all on teardown.

use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task};
use std::collections::{HashMap, HashSet};

use sounding_sim::surface_field::SurfaceField;
use sounding_sim::surface_mesh::{
    build_chunk, should_split, ChunkMesh, QuadNode, DEFAULT_MAX_LEVEL, DEFAULT_RESOLUTION,
};

/// Meshes uploaded to the world per frame (the bounded GPU-upload budget).
const UPLOAD_BUDGET: usize = 6;
/// New mesh-build tasks spawned per frame (bounds the async backlog).
const SPAWN_BUDGET: usize = 12;

/// Marks a streamed surface-chunk entity (root-only despawn marker) so a scene can
/// tear them all down on exit.
#[derive(Component)]
pub struct StreamedChunk;

/// A chunk's centre in **body-centred** world coordinates — the anchor-relative
/// render reads this each frame.
#[derive(Component)]
pub struct ChunkCenter(pub DVec3);

/// Resident (uploaded), in-flight (meshing), and built-but-not-yet-uploaded (ready)
/// chunk state, keyed by quadtree node. Plain struct — a scene owns one wherever it
/// keeps its per-run state.
#[derive(Default)]
pub struct ChunkStreamer {
    live: HashMap<QuadNode, Entity>,
    meshing: HashMap<QuadNode, Task<ChunkMesh>>,
    ready: Vec<(QuadNode, ChunkMesh)>,
}

impl ChunkStreamer {
    /// Number of resident chunks (for HUD/telemetry).
    pub fn live_count(&self) -> usize {
        self.live.len()
    }
    /// Number of in-flight + ready (pending) chunks (for HUD/telemetry).
    pub fn pending_count(&self) -> usize {
        self.meshing.len() + self.ready.len()
    }
}

/// Advance the streamer one frame. `anchor_body` is the LOD focus (the vehicle) in
/// **body-centred** coordinates; new chunk entities are spawned with a
/// [`ChunkCenter`] (their body-centred centre) and tinted with `material`, at a
/// placeholder `Transform` that [`reposition`] fixes up.
pub fn stream(
    streamer: &mut ChunkStreamer,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    field: &SurfaceField,
    anchor_body: DVec3,
    material: &Handle<StandardMaterial>,
) {
    let radius = field.radius();

    let desired: HashSet<QuadNode> = {
        let mut leaves = Vec::new();
        let mut stack: Vec<QuadNode> = QuadNode::roots().to_vec();
        while let Some(node) = stack.pop() {
            if should_split(node, anchor_body, radius, DEFAULT_MAX_LEVEL) {
                stack.extend_from_slice(&node.children());
            } else {
                leaves.push(node);
            }
        }
        leaves.into_iter().collect()
    };

    // 1. Poll in-flight builds → ready queue.
    let in_flight: Vec<QuadNode> = streamer.meshing.keys().copied().collect();
    for node in in_flight {
        let done = streamer
            .meshing
            .get_mut(&node)
            .and_then(|task| block_on(future::poll_once(task)));
        if let Some(chunk) = done {
            streamer.meshing.remove(&node);
            streamer.ready.push((node, chunk));
        }
    }

    // 2. Upload ready chunks under budget; keep the overflow for next frame.
    let ready_items = std::mem::take(&mut streamer.ready);
    let mut keep = Vec::new();
    let mut uploaded = 0;
    for (node, chunk) in ready_items {
        if !desired.contains(&node) {
            continue;
        }
        if uploaded >= UPLOAD_BUDGET {
            keep.push((node, chunk));
            continue;
        }
        let center = chunk.center;
        let mesh = to_bevy_mesh(meshes, &chunk);
        // Initial transform in the anchor-relative frame (`anchor_world = body_center + anchor_body`,
        // so `body_center + center − anchor_world = center − anchor_body`) — so a chunk is placed
        // correctly the frame it spawns, before `reposition` runs.
        let init = Transform::from_translation((center - anchor_body).as_vec3());
        let entity = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                init,
                ChunkCenter(center),
                StreamedChunk,
            ))
            .id();
        streamer.live.insert(node, entity);
        uploaded += 1;
    }
    streamer.ready = keep;

    // 3. Enqueue builds for desired nodes not live/meshing/ready.
    let pool = AsyncComputeTaskPool::get();
    let mut spawned = 0;
    for &node in &desired {
        if spawned >= SPAWN_BUDGET {
            break;
        }
        if streamer.live.contains_key(&node)
            || streamer.meshing.contains_key(&node)
            || streamer.ready.iter().any(|(n, _)| *n == node)
        {
            continue;
        }
        let field = *field; // SurfaceField is Copy
        let task = pool.spawn(async move { build_chunk(&field, node, DEFAULT_RESOLUTION) });
        streamer.meshing.insert(node, task);
        spawned += 1;
    }

    // 4. Coverage-gated despawn (no LOD-transition holes, WI 771).
    let live_nodes: HashSet<QuadNode> = streamer.live.keys().copied().collect();
    let stale: Vec<QuadNode> = streamer
        .live
        .keys()
        .copied()
        .filter(|&n| !desired.contains(&n))
        .filter(|&n| {
            desired
                .iter()
                .filter(|&&d| n.overlaps(d))
                .all(|&d| live_nodes.contains(&d))
        })
        .collect();
    for node in stale {
        if let Some(e) = streamer.live.remove(&node) {
            commands.entity(e).despawn();
        }
    }
}

/// Set each resident chunk's render transform relative to `anchor_world` (the same
/// world anchor the caller renders its vehicle around): `render = body_center +
/// center − anchor_world`, in f32. Call each frame after [`stream`].
pub fn reposition(
    body_center: DVec3,
    anchor_world: DVec3,
    chunks: &mut Query<(&ChunkCenter, &mut Transform), With<StreamedChunk>>,
) {
    for (center, mut tf) in chunks.iter_mut() {
        tf.translation = (body_center + center.0 - anchor_world).as_vec3();
    }
}

/// Build a Bevy `Mesh` from the headless chunk buffers.
fn to_bevy_mesh(meshes: &mut Assets<Mesh>, chunk: &ChunkMesh) -> Handle<Mesh> {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, chunk.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, chunk.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, chunk.uvs.clone());
    mesh.insert_indices(Indices::U32(chunk.indices.clone()));
    meshes.add(mesh)
}

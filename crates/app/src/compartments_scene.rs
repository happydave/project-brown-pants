//! Toy: airtight compartments (WI 519). A hollow craft's sealed interior volumes
//! are derived by flood-fill (`sounding_sim::compartments`) and drawn colour-coded.
//! A hatch in the internal wall toggles to merge/split the two rooms, and a hull
//! breach vents a room to the exterior — each a structural-change event that
//! recomputes the compartment set (otherwise it is cached, not recomputed).
//!
//! Controls: `H` toggle the hatch · `B` breach the hull · `R` reset.

use bevy::math::Isometry3d;
use bevy::prelude::*;
use sounding_sim::compartments::CompartmentCache;
use sounding_sim::voxel::{Door, Material, Voxel, VoxelCraft};

/// The cell whose removal breaches the +x room's hull.
const BREACH_CELL: IVec3 = IVec3::new(6, 2, 2);
/// The hatch cell (the gap in the internal wall).
const HATCH_CELL: IVec3 = IVec3::new(3, 2, 2);

/// The demo craft and its cached compartment set.
#[derive(Resource)]
struct CompartmentsWorld {
    craft: VoxelCraft,
    cache: CompartmentCache,
}

impl CompartmentsWorld {
    fn new() -> Self {
        let craft = demo_craft();
        let cache = CompartmentCache::new(&craft);
        Self { craft, cache }
    }
}

/// A 7×5×5 shell with an internal wall at x=3 (a one-cell hatch gap at the centre)
/// and a hatch door there, initially closed.
fn demo_craft() -> VoxelCraft {
    let mut c = VoxelCraft::new(1.0);
    let (nx, ny, nz) = (7, 5, 5);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let on_shell =
                    x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1;
                let on_wall = x == 3 && IVec3::new(x, y, z) != HATCH_CELL;
                if on_shell || on_wall {
                    c.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::ALUMINIUM,
                    });
                }
            }
        }
    }
    c.doors.push(Door {
        cell: HATCH_CELL,
        open: false,
    });
    c
}

/// Marks the heads-up readout.
#[derive(Component)]
struct Hud;

/// The Toy airtight-compartments scene.
pub struct CompartmentsScenePlugin;

impl Plugin for CompartmentsScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CompartmentsWorld::new())
            .add_systems(Startup, setup_view)
            .add_systems(Update, (handle_input, draw, update_hud).chain());
    }
}

fn setup_view(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, 9.0, 17.0).looking_at(Vec3::new(3.0, 2.0, 2.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 6_000.0,
            ..default()
        },
        Transform::from_xyz(6.0, 14.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Text::new("H toggle hatch · B breach hull · R reset"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.95, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        Hud,
    ));
}

fn handle_input(keys: Res<ButtonInput<KeyCode>>, mut world: ResMut<CompartmentsWorld>) {
    let mut changed = false;
    if keys.just_pressed(KeyCode::KeyH) {
        if let Some(d) = world.craft.doors.iter_mut().find(|d| d.cell == HATCH_CELL) {
            d.open = !d.open;
            changed = true;
        }
    }
    if keys.just_pressed(KeyCode::KeyB) {
        let before = world.craft.voxels.len();
        world.craft.voxels.retain(|v| v.cell != BREACH_CELL);
        changed |= world.craft.voxels.len() != before;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        world.craft = demo_craft();
        changed = true;
    }
    if changed {
        // A structural-change event: recompute the compartment set (otherwise the
        // cache is reused each frame).
        world.cache.mark_dirty();
    }
}

fn draw(mut gizmos: Gizmos, mut world: ResMut<CompartmentsWorld>) {
    let palette = [
        Color::srgb(0.45, 0.75, 0.95),
        Color::srgb(0.90, 0.55, 0.30),
        Color::srgb(0.60, 0.85, 0.45),
        Color::srgb(0.90, 0.80, 0.35),
    ];
    let size = world.craft.cell_size as f32;
    let at = |c: IVec3| Vec3::new(c.x as f32, c.y as f32, c.z as f32) + Vec3::splat(0.5) * size;
    let cube = |gizmos: &mut Gizmos, c: IVec3, s: f32, color: Color| {
        gizmos.primitive_3d(
            &Cuboid::new(s, s, s),
            Isometry3d::from_translation(at(c)),
            color,
        );
    };

    // Structure: grey wireframe cubes (solid voxels and any closed door).
    for v in &world.craft.voxels {
        cube(&mut gizmos, v.cell, size, Color::srgb(0.35, 0.35, 0.40));
    }
    for d in &world.craft.doors {
        if !d.open {
            cube(&mut gizmos, d.cell, size, Color::srgb(0.75, 0.65, 0.30));
        }
    }

    // Compartments: colour-coded inset cubes per cell.
    let craft = world.craft.clone();
    let set = world.cache.get(&craft);
    for (i, comp) in set.compartments.iter().enumerate() {
        let color = palette[i % palette.len()];
        for &cell in &comp.cells {
            cube(&mut gizmos, cell, size * 0.7, color);
        }
    }
}

fn update_hud(mut world: ResMut<CompartmentsWorld>, mut hud: Query<&mut Text, With<Hud>>) {
    let craft = world.craft.clone();
    let (count, volume) = {
        let set = world.cache.get(&craft);
        (set.count(), set.total_volume())
    };
    if let Ok(mut text) = hud.single_mut() {
        text.0 = format!(
            "sealed compartments: {count}   volume: {volume:.0} m³\nH toggle hatch · B breach hull · R reset"
        );
    }
}

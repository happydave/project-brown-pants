//! Toy: structural breakage (WI 518). A craft is spun up until the centripetal
//! load exceeds its bonds' strength, then it snaps into connected-component
//! fragments that tumble apart — the momentum-conserving fling (`v = v_cm + ω×r`)
//! made visible. Headless breakage lives in `sounding_sim::breakage`; this scene
//! drives and draws it (gizmos, origin-local — the CoM sits at the origin).

use bevy::math::{DVec3, Isometry3d};
use bevy::prelude::*;
use sounding_sim::active::ActiveBody;
use sounding_sim::breakage::fracture;
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

/// Spin-up rate while intact, rad/s² — ramps the craft toward its break point.
const SPIN_RAMP: f64 = 8.0;
/// Starting spin, rad/s.
const SPIN_START: f64 = 40.0;

/// The breakage demo: one intact craft that fractures into several fragments.
#[derive(Resource)]
struct BreakWorld {
    /// One `(craft, body)` while intact; several after the break.
    pieces: Vec<(VoxelCraft, ActiveBody)>,
    fractured: bool,
    spin: f64,
}

impl BreakWorld {
    fn new() -> Self {
        // A 9-cell aluminium bar, centre of mass at the origin, spinning about z.
        let mut craft = VoxelCraft::new(1.0);
        for x in 0..9 {
            craft.voxels.push(Voxel {
                cell: IVec3::new(x, 0, 0),
                material: Material::ALUMINIUM,
            });
        }
        let mp = craft.mass_properties().expect("non-empty");
        let body = ActiveBody::new(DVec3::ZERO, DVec3::ZERO, mp.mass, mp.inertia)
            .with_angular_velocity(DVec3::new(0.0, 0.0, SPIN_START));
        Self {
            pieces: vec![(craft, body)],
            fractured: false,
            spin: SPIN_START,
        }
    }
}

/// Marks the heads-up readout.
#[derive(Component)]
struct Hud;

/// The Toy structural-breakage scene.
pub struct BreakScenePlugin;

impl Plugin for BreakScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(BreakWorld::new())
            .add_systems(Startup, setup_view)
            .add_systems(Update, (step_break, draw_pieces, update_hud).chain());
    }
}

fn setup_view(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 6.0, 18.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            ..default()
        },
        Transform::from_xyz(6.0, 14.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Text::new("intact — spinning up…"),
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

fn step_break(time: Res<Time>, mut world: ResMut<BreakWorld>) {
    let dt = time.delta_secs_f64().min(0.05);
    if !world.fractured {
        // Ramp the spin and look for the break.
        world.spin += SPIN_RAMP * dt;
        let spin = world.spin;
        let (craft, body) = &mut world.pieces[0];
        *body = body.with_angular_velocity(DVec3::new(0.0, 0.0, spin));
        body.step(0.0, dt); // free rotation (no gravity)
        let craft = craft.clone();
        let body = *body;
        if let Some(fragments) = fracture(&craft, &body, DVec3::ZERO) {
            world.pieces = fragments;
            world.fractured = true;
        }
    } else {
        // Free flight: each fragment drifts and tumbles apart.
        for (_, body) in &mut world.pieces {
            body.step(0.0, dt);
        }
    }
}

fn draw_pieces(mut gizmos: Gizmos, world: Res<BreakWorld>) {
    // A distinct colour per piece so the split reads clearly.
    let palette = [
        Color::srgb(0.85, 0.84, 0.88),
        Color::srgb(0.90, 0.55, 0.30),
        Color::srgb(0.45, 0.75, 0.95),
        Color::srgb(0.60, 0.85, 0.45),
        Color::srgb(0.90, 0.80, 0.35),
    ];
    for (i, (craft, body)) in world.pieces.iter().enumerate() {
        let color = palette[i % palette.len()];
        let Some(mp) = craft.mass_properties() else {
            continue;
        };
        let q = body.orientation;
        let qf = q.as_quat();
        let size = craft.cell_size as f32;
        for v in &craft.voxels {
            let local =
                (v.cell.as_dvec3() + DVec3::splat(0.5)) * craft.cell_size - mp.center_of_mass;
            let world_pos = body.position + q * local;
            gizmos.primitive_3d(
                &Cuboid::new(size, size, size),
                Isometry3d::new(world_pos.as_vec3(), qf),
                color,
            );
        }
    }
}

fn update_hud(world: Res<BreakWorld>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        text.0 = if world.fractured {
            format!("fractured into {} pieces", world.pieces.len())
        } else {
            format!("intact — spin {:.0} rad/s (ramping to break)", world.spin)
        };
    }
}

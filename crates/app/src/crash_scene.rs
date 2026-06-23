//! Crash — breakage-on-impact demo (`-- crash`, WI 594).
//!
//! Couples collision (WI 592/593) to structural breakage (WI 518): a **frangible** craft you
//! ram (hold `SPACE`) into a heavy target. A *gentle* nudge bounces off intact; a *hard* impact
//! drives a large contact force through the structure, which `sounding_sim::breakage::
//! fracture_on_impact` turns into a connected-component fracture — the craft shatters and the
//! fragments (themselves collidable, WI 593) tumble and settle on the ground. `R` resets.
//!
//! A self-contained local sandbox drawn with gizmos (like `-- break`): a flat ground at `y = 0`,
//! uniform gravity, small coordinates — no floating origin. Each substep composes, per piece,
//! gravity + ground + pairwise contact, integrates, then tests the (still-intact) projectile's
//! **contact** force against its fracture threshold.

use bevy::math::{DVec3, Isometry3d};
use bevy::prelude::*;
use std::f64::consts::FRAC_PI_2;

use sounding_sim::active::ActiveBody;
use sounding_sim::breakage::fracture_on_impact;
use sounding_sim::collision::{
    craft_bounds, craft_collision_shape, ground_half_space, Bounds, CollisionShape,
};
use sounding_sim::contact::{body_contact_wrench, ground_contact_wrench, ContactParams};
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

/// Uniform gravity for the local sandbox, m/s² (Earth surface).
const G: f64 = 9.85;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
/// Forward acceleration applied to the projectile while SPACE is held, m/s².
const THRUST_ACCEL: f64 = 40.0;
/// A weak material so an achievable ram speed crosses the fracture threshold.
const FRANGIBLE: Material = Material {
    density: 2700.0,
    strength: 2.0e6,
};

const PROJECTILE_COLOR: Color = Color::srgb(0.90, 0.55, 0.30);
const TARGET_COLOR: Color = Color::srgb(0.55, 0.60, 0.70);
/// Distinct colours for the impact fragments, so the split reads clearly.
const FRAGMENT_PALETTE: [Color; 5] = [
    Color::srgb(0.85, 0.84, 0.88),
    Color::srgb(0.45, 0.75, 0.95),
    Color::srgb(0.60, 0.85, 0.45),
    Color::srgb(0.90, 0.80, 0.35),
    Color::srgb(0.95, 0.45, 0.55),
];

/// One collidable piece: its lattice, rigid state, and draw colour.
struct Piece {
    voxels: VoxelCraft,
    body: ActiveBody,
    color: Color,
}

impl Piece {
    fn shape(&self) -> CollisionShape {
        craft_collision_shape(&self.voxels)
    }
    fn bounds(&self) -> Option<Bounds> {
        craft_bounds(&self.voxels)
    }
    fn com(&self) -> DVec3 {
        self.voxels
            .mass_properties()
            .map(|mp| mp.center_of_mass)
            .unwrap_or(DVec3::ZERO)
    }
}

/// A frangible bar of `n` cells along +x.
fn frangible_bar(n: i32) -> VoxelCraft {
    let mut c = VoxelCraft::new(1.0);
    for x in 0..n {
        c.voxels.push(Voxel {
            cell: IVec3::new(x, 0, 0),
            material: FRANGIBLE,
        });
    }
    c
}

/// A solid `n³` block of one material (the heavy target).
fn block(n: i32, material: Material) -> VoxelCraft {
    let mut c = VoxelCraft::new(1.0);
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                c.voxels.push(Voxel {
                    cell: IVec3::new(x, y, z),
                    material,
                });
            }
        }
    }
    c
}

/// The two starting pieces: a frangible projectile to the left, a heavy steel target.
fn initial_pieces() -> Vec<(VoxelCraft, ActiveBody, Color)> {
    let proj = frangible_bar(6);
    let pmp = proj.mass_properties().unwrap();
    // Bar is 1 cell tall → its CoM rests 0.5 m above the ground.
    let pbody = ActiveBody::new(
        DVec3::new(-9.0, 0.5, 0.0),
        DVec3::ZERO,
        pmp.mass,
        pmp.inertia,
    );

    let target = block(3, Material::STEEL);
    let tmp = target.mass_properties().unwrap();
    let tbody = ActiveBody::new(
        DVec3::new(3.0, 1.5, 0.0),
        DVec3::ZERO,
        tmp.mass,
        tmp.inertia,
    );

    vec![
        (proj, pbody, PROJECTILE_COLOR),
        (target, tbody, TARGET_COLOR),
    ]
}

#[derive(Resource)]
struct CrashWorld {
    pieces: Vec<Piece>,
    ground: CollisionShape,
    params: ContactParams,
    /// SPACE held this frame (accelerate the projectile).
    thrust: bool,
    /// The projectile (index 0) has not yet fractured.
    intact: bool,
    accumulator: f64,
}

impl CrashWorld {
    fn new() -> Self {
        Self {
            pieces: initial_pieces()
                .into_iter()
                .map(|(voxels, body, color)| Piece {
                    voxels,
                    body,
                    color,
                })
                .collect(),
            ground: ground_half_space(0.0),
            params: ContactParams::default(),
            thrust: false,
            intact: true,
            accumulator: 0.0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    /// Advance every piece one substep, then fracture the projectile if its contact load is high
    /// enough. Returns nothing; mutates `pieces` in place (growing it on a fracture).
    fn substep(&mut self, dt: f64) {
        let n = self.pieces.len();
        let shapes: Vec<CollisionShape> = self.pieces.iter().map(Piece::shape).collect();
        let bounds: Vec<Option<Bounds>> = self.pieces.iter().map(Piece::bounds).collect();
        let coms: Vec<DVec3> = self.pieces.iter().map(Piece::com).collect();

        // Contact-only wrench per piece (gravity/thrust kept separate so the fracture test sees
        // the load actually transmitted through the structure).
        let mut contact = vec![(DVec3::ZERO, DVec3::ZERO); n];
        for i in 0..n {
            let (gf, gt) = ground_contact_wrench(
                &self.pieces[i].body,
                &shapes[i],
                bounds[i],
                coms[i],
                &self.ground,
                &self.params,
            );
            contact[i].0 += gf;
            contact[i].1 += gt;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let ((fa, ta), (fb, tb)) = body_contact_wrench(
                    &self.pieces[i].body,
                    &shapes[i],
                    bounds[i],
                    coms[i],
                    &self.pieces[j].body,
                    &shapes[j],
                    bounds[j],
                    coms[j],
                    &self.params,
                );
                contact[i].0 += fa;
                contact[i].1 += ta;
                contact[j].0 += fb;
                contact[j].1 += tb;
            }
        }

        // Integrate: gravity (uniform) + contact + projectile thrust while held & intact.
        for (i, piece) in self.pieces.iter_mut().enumerate() {
            let mut force = contact[i].0 + DVec3::new(0.0, -G * piece.body.mass, 0.0);
            if i == 0 && self.intact && self.thrust {
                force += DVec3::new(piece.body.mass * THRUST_ACCEL, 0.0, 0.0);
            }
            piece.body.integrate_wrench(force, contact[i].1, dt);
        }

        // Breakage-on-impact: the still-intact projectile fractures if its contact load exceeds
        // its bonds. Fragments replace it and become ordinary collidable pieces.
        if self.intact {
            let proj = &self.pieces[0];
            if let Some(frags) = fracture_on_impact(&proj.voxels, &proj.body, contact[0].0) {
                let rest: Vec<Piece> = self.pieces.drain(1..).collect();
                self.pieces = frags
                    .into_iter()
                    .enumerate()
                    .map(|(i, (voxels, body))| Piece {
                        voxels,
                        body,
                        color: FRAGMENT_PALETTE[i % FRAGMENT_PALETTE.len()],
                    })
                    .chain(rest)
                    .collect();
                self.intact = false;
            }
        }
    }

    /// The projectile's (or, post-break, the first fragment's) forward speed, for the HUD.
    fn projectile_speed(&self) -> f64 {
        self.pieces
            .first()
            .map(|p| p.body.velocity.x)
            .unwrap_or(0.0)
    }
}

#[derive(Component)]
struct Hud;

/// The crash demo scene.
pub struct CrashScenePlugin;

impl Plugin for CrashScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CrashWorld::new())
            .add_systems(Startup, setup_view)
            .add_systems(
                Update,
                (crash_input, step_crash, draw_scene, update_hud).chain(),
            );
    }
}

fn setup_view(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(6.0, 7.0, 22.0).looking_at(Vec3::new(-1.0, 1.0, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(6.0, 14.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Text::new("crash: ready"),
        TextFont {
            font_size: 18.0,
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
    commands.spawn((
        Text::new("hold SPACE to ram the projectile into the block · R reset"),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgb(0.7, 0.75, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

/// Holding SPACE rams the projectile; R resets.
fn crash_input(keys: Res<ButtonInput<KeyCode>>, mut world: ResMut<CrashWorld>) {
    world.thrust = keys.pressed(KeyCode::Space);
    if keys.just_pressed(KeyCode::KeyR) {
        world.reset();
    }
}

fn step_crash(time: Res<Time>, mut world: ResMut<CrashWorld>) {
    world.accumulator += time.delta_secs_f64();
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS {
        world.substep(SUBSTEP_DT);
        world.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

fn draw_scene(mut gizmos: Gizmos, world: Res<CrashWorld>) {
    // Ground grid in the XZ plane.
    gizmos.grid(
        Isometry3d::new(Vec3::ZERO, Quat::from_rotation_x(-FRAC_PI_2 as f32)),
        UVec2::new(40, 40),
        Vec2::splat(1.0),
        Color::srgb(0.30, 0.33, 0.38),
    );
    // Each piece: a coloured cube per voxel, placed by the body's rigid transform.
    for piece in &world.pieces {
        let Some(mp) = piece.voxels.mass_properties() else {
            continue;
        };
        let q = piece.body.orientation;
        let qf = q.as_quat();
        let size = piece.voxels.cell_size as f32;
        for v in &piece.voxels.voxels {
            let local = (v.cell.as_dvec3() + DVec3::splat(0.5)) * piece.voxels.cell_size
                - mp.center_of_mass;
            let world_pos = piece.body.position + q * local;
            gizmos.primitive_3d(
                &Cuboid::new(size, size, size),
                Isometry3d::new(world_pos.as_vec3(), qf),
                piece.color,
            );
        }
    }
}

fn update_hud(world: Res<CrashWorld>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let state = if world.intact { "intact" } else { "FRACTURED" };
        text.0 = format!(
            "crash: {state}\nprojectile speed: {:5.1} m/s\npieces: {}",
            world.projectile_speed(),
            world.pieces.len(),
        );
    }
}

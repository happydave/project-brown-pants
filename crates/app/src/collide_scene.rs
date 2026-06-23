//! Collide — craft↔craft and debris collision demo (`-- collide`, WI 593).
//!
//! The body↔body counterpart to `-- land`. Several movable craft share the textured ground:
//! a **projectile** craft you fire (`SPACE`) at a **target** craft, plus a small pile of
//! **debris** fragments that fall and settle on each other. Each active substep this scene
//! composes, for every body, gravity + the craft↔ground penalty wrench
//! (`sounding_sim::contact::ground_contact_wrench`) + the pairwise craft↔craft wrench
//! (`body_contact_wrench`, WI 593), then integrates each `ActiveBody` — so the same penalty
//! response resolves resting, sliding, head-on impact, and stacking. `R` resets the scene; the
//! HUD shows the closing speed, the projectile↔target separation, and a RESTING flag.

use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::{light_consts::lux, AtmosphereEnvironmentMapLight};
use bevy::math::DVec3;
use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use sounding_sim::active::ActiveBody;
use sounding_sim::collision::{
    craft_bounds, craft_collision_shape, ground_half_space, Bounds, CollisionShape,
};
use sounding_sim::contact::{body_contact_wrench, ground_contact_wrench, ContactParams};
use sounding_sim::sim::CentralBody;
use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

use crate::floating_origin::{AnchorCamera, FloatingOriginPlugin, WorldPlacement};
use crate::voxel_skin::{build_skin_mesh, material_set_for, pbr_material, VoxelSkin};
use sounding_sim::frame::{FrameId, WorldPos};

const BODY: CentralBody = CentralBody::EARTHLIKE;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
/// Speed the projectile is launched at the target, m/s.
const FIRE_SPEED: f64 = 6.0;
/// Speed below which a body counts as resting, m/s.
const REST_SPEED: f64 = 0.1;

/// One collidable body: its rigid state plus the derived, static collision data.
struct Collider {
    body: ActiveBody,
    shape: CollisionShape,
    bounds: Option<Bounds>,
    com: DVec3,
    /// The body's resting (or launch) state, restored on reset.
    home: ActiveBody,
}

impl Collider {
    /// A collider for `voxels` whose CoM starts at world `position` with `velocity`.
    fn new(voxels: &VoxelCraft, position: DVec3, velocity: DVec3) -> Self {
        let mp = voxels.mass_properties().expect("non-empty craft");
        let body = ActiveBody::new(position, velocity, mp.mass, mp.inertia);
        Self {
            shape: craft_collision_shape(voxels),
            bounds: craft_bounds(voxels),
            com: mp.center_of_mass,
            home: body,
            body,
        }
    }
}

/// The collide demo world: a fixed set of colliders. Index 0 is the projectile, 1 the target;
/// the rest are debris.
#[derive(Resource)]
struct CollideWorld {
    colliders: Vec<Collider>,
    ground: CollisionShape,
    params: ContactParams,
    accumulator: f64,
}

/// A solid cube of side `n` cells (1 m each) of the given material, centred-on-cell-corner like
/// the rest of the voxel scenes.
fn cube_craft(n: i32, material: Material) -> VoxelCraft {
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

impl CollideWorld {
    fn new() -> Self {
        let surface = BODY.radius;
        // The box half-height of a 2-cell cube is 1 m → its CoM rests 1 m above the surface.
        let rest_y = surface + 1.0;
        let target = cube_craft(2, Material::COMPOSITE);
        let projectile = cube_craft(2, Material::COMPOSITE);

        let mut colliders = vec![
            // 0: projectile, parked to the left at rest (fired with SPACE).
            Collider::new(&projectile, DVec3::new(-8.0, rest_y, 0.0), DVec3::ZERO),
            // 1: target, parked at the origin.
            Collider::new(&target, DVec3::new(0.0, rest_y, 0.0), DVec3::ZERO),
        ];

        // Debris: three 1-cell fragments dropped from staggered heights to settle into a pile,
        // off to the side so they don't interfere with the head-on shot.
        let frag = cube_craft(1, Material::ALUMINIUM);
        let frag_rest = surface + 0.5;
        for (i, dz) in [0.0, 0.6, 0.3].into_iter().enumerate() {
            colliders.push(Collider::new(
                &frag,
                DVec3::new(
                    6.0 + 0.2 * i as f64,
                    frag_rest + 2.0 + 0.8 * i as f64,
                    6.0 + dz,
                ),
                DVec3::ZERO,
            ));
        }

        Self {
            colliders,
            ground: ground_half_space(surface),
            params: ContactParams::default(),
            accumulator: 0.0,
        }
    }

    /// World position of a collider relative to the rendered (surface-centred) origin.
    fn render_pos(&self, i: usize) -> DVec3 {
        self.colliders[i].body.position - DVec3::new(0.0, BODY.radius, 0.0)
    }

    /// Restore every body to its home state (the reset button).
    fn reset(&mut self) {
        for c in &mut self.colliders {
            c.body = c.home;
        }
        self.accumulator = 0.0;
    }

    /// Launch the projectile (index 0) toward the target (+x).
    fn fire(&mut self) {
        self.colliders[0].body.velocity = DVec3::new(FIRE_SPEED, 0.0, 0.0);
    }

    /// Advance every body one substep: gravity + ground + pairwise craft↔craft, then integrate.
    fn substep(&mut self, dt: f64) {
        let n = self.colliders.len();
        let mut acc = vec![(DVec3::ZERO, DVec3::ZERO); n];
        // Gravity + craft↔ground for each body.
        for (i, c) in self.colliders.iter().enumerate() {
            acc[i].0 += gravity_force(&c.body);
            let (gf, gt) = ground_contact_wrench(
                &c.body,
                &c.shape,
                c.bounds,
                c.com,
                &self.ground,
                &self.params,
            );
            acc[i].0 += gf;
            acc[i].1 += gt;
        }
        // Pairwise craft↔craft (equal-and-opposite).
        for i in 0..n {
            for j in (i + 1)..n {
                let a = &self.colliders[i];
                let b = &self.colliders[j];
                let ((fa, ta), (fb, tb)) = body_contact_wrench(
                    &a.body,
                    &a.shape,
                    a.bounds,
                    a.com,
                    &b.body,
                    &b.shape,
                    b.bounds,
                    b.com,
                    &self.params,
                );
                acc[i].0 += fa;
                acc[i].1 += ta;
                acc[j].0 += fb;
                acc[j].1 += tb;
            }
        }
        for (c, (f, t)) in self.colliders.iter_mut().zip(acc) {
            c.body.integrate_wrench(f, t, dt);
        }
    }
}

/// Point-mass gravity **force** on a body about the central attractor.
fn gravity_force(body: &ActiveBody) -> DVec3 {
    let r = body.position;
    let r2 = r.length_squared();
    if r2 <= 0.0 {
        return DVec3::ZERO;
    }
    -BODY.mu * body.mass * r / (r2 * r2.sqrt())
}

#[derive(Component)]
struct ColliderTag(usize);
#[derive(Component)]
struct Hud;

/// The collide demo scene.
pub struct CollideScenePlugin;

impl Plugin for CollideScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FloatingOriginPlugin)
            .insert_resource(CollideWorld::new())
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (collide_input, step_collide, track_bodies, update_hud).chain(),
            );
    }
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering: ResMut<Assets<ScatteringMedium>>,
    world: Res<CollideWorld>,
) {
    crate::ground::spawn_ground(&mut commands, &mut meshes, &mut materials, &asset_server);

    // One skinned mesh per collider. Projectile + target are COMPOSITE; debris is ALUMINIUM,
    // both rebuilt from their own lattice so the cube and fragment shapes render correctly.
    let cube = cube_craft(2, Material::COMPOSITE);
    let frag = cube_craft(1, Material::ALUMINIUM);
    for (i, _) in world.colliders.iter().enumerate() {
        let (voxels, mat) = if i < 2 {
            (&cube, Material::COMPOSITE)
        } else {
            (&frag, Material::ALUMINIUM)
        };
        let mesh = meshes.add(build_skin_mesh(voxels, VoxelSkin::Hull));
        let material = pbr_material(material_set_for(mat), &asset_server, &mut materials);
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::default(),
            WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, world.render_pos(i))),
            ColliderTag(i),
        ));
    }

    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5) * Quat::from_rotation_y(0.6)),
    ));

    commands.spawn((
        Text::new("collide: ready"),
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
        Text::new("SPACE fire projectile · R reset"),
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

    // A fixed vantage looking at the action between projectile and target.
    let look = DVec3::new(-1.0, 1.0, 0.0);
    let cam = look + DVec3::new(10.0, 8.0, 18.0);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(cam.as_vec3()).looking_at(look.as_vec3(), Vec3::Y),
        Atmosphere::earthlike(scattering.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        Exposure { ev100: 13.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        WorldPlacement(WorldPos::new(FrameId::CENTRAL_BODY, cam)),
        AnchorCamera,
    ));
}

/// SPACE fires the projectile; R resets.
fn collide_input(keys: Res<ButtonInput<KeyCode>>, mut world: ResMut<CollideWorld>) {
    if keys.just_pressed(KeyCode::Space) {
        world.fire();
    }
    if keys.just_pressed(KeyCode::KeyR) {
        world.reset();
    }
}

fn step_collide(time: Res<Time>, mut world: ResMut<CollideWorld>) {
    world.accumulator += time.delta_secs_f64();
    let mut n = 0;
    while world.accumulator >= SUBSTEP_DT && n < MAX_SUBSTEPS {
        world.substep(SUBSTEP_DT);
        world.accumulator -= SUBSTEP_DT;
        n += 1;
    }
}

fn track_bodies(
    world: Res<CollideWorld>,
    mut bodies: Query<(&ColliderTag, &mut WorldPlacement, &mut Transform)>,
) {
    for (tag, mut wp, mut tf) in &mut bodies {
        wp.0 = WorldPos::new(FrameId::CENTRAL_BODY, world.render_pos(tag.0));
        tf.rotation = world.colliders[tag.0].body.orientation.as_quat();
    }
}

fn update_hud(world: Res<CollideWorld>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let proj = &world.colliders[0].body;
        let target = &world.colliders[1].body;
        let closing = (proj.velocity - target.velocity).length();
        let separation = (proj.position - target.position).length();
        let resting = world
            .colliders
            .iter()
            .all(|c| c.body.velocity.length() < REST_SPEED);
        let state = if resting { "RESTING" } else { "active" };
        text.0 = format!(
            "collide: {state}\nclosing:    {closing:6.2} m/s\nseparation: {separation:6.2} m",
        );
    }
}

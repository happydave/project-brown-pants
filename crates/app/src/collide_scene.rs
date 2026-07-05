//! Collide — craft↔craft and debris collision demo (`-- collide`, WI 593).
//!
//! The body↔body counterpart to `-- land`. Several movable craft share the textured ground:
//! a **projectile** craft you accelerate (hold `SPACE`) into a **target** craft, plus a small
//! pile of **debris** fragments that fall and settle on each other. Each active substep this scene
//! composes, for every body, gravity + the craft↔ground penalty wrench
//! (`sounding_sim::contact::ground_contact_wrench`) + the pairwise craft↔craft wrench
//! (`body_contact_wrench`, WI 593), then integrates each `ActiveBody` — so the same penalty
//! response resolves resting, sliding, head-on impact, and stacking. `R` resets the scene; the
//! HUD shows the closing speed, the projectile↔target separation, and a RESTING flag.
//!
//! Fixture loading (WI 843): `-- collide [projectile] [target]` swaps either slot's procedural
//! 2³ cube for a saved craft (see [`crate::harness_fixture`]; `-` keeps a slot's default), so
//! shaped builds reach the WI 837 form-aware contact path here. Each collider's skin is built
//! from **its own** lattice via the shape-aware submesh path (WI 833), so a wedge nose renders
//! as a wedge; the debris pile is untouched.

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
use crate::harness_fixture::{fixture_arg, rest_height, slot_name, Fixture};
use crate::voxel_skin::{panel_render_pieces, pbr_material, skin_submeshes, VoxelSkin};
use sounding_sim::frame::{FrameId, WorldPos};

const BODY: CentralBody = CentralBody::EARTHLIKE;
const SUBSTEP_DT: f64 = 0.004;
const MAX_SUBSTEPS: u32 = 250;
/// Forward acceleration applied to the projectile while SPACE is held, m/s².
const PROJECTILE_ACCEL: f64 = 25.0;
/// Speed below which a body counts as resting, m/s.
const REST_SPEED: f64 = 0.1;

/// One collidable body: its lattice (kept for skin building, WI 843), rigid state,
/// and the derived, static collision data.
struct Collider {
    voxels: VoxelCraft,
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
            voxels: voxels.clone(),
            shape: craft_collision_shape(voxels),
            bounds: craft_bounds(voxels),
            com: mp.center_of_mass,
            home: body,
            body,
        }
    }
}

/// The scene's fixture slots (WI 843): a loaded craft per slot, or `None` for the
/// procedural 2³ composite cube. Resolved once at plugin build from the CLI
/// arguments; `R` reset restores the loaded bodies via their homes as always.
#[derive(Resource, Default)]
struct CollideFixtures {
    projectile: Option<Fixture>,
    target: Option<Fixture>,
}

/// The collide demo world: a fixed set of colliders. Index 0 is the projectile, 1 the target;
/// the rest are debris.
#[derive(Resource)]
struct CollideWorld {
    colliders: Vec<Collider>,
    ground: CollisionShape,
    params: ContactParams,
    accumulator: f64,
    /// Whether the projectile is being accelerated this step (SPACE held).
    thrust: bool,
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
    fn new(fixtures: &CollideFixtures) -> Self {
        let surface = BODY.radius;
        // Each fixture rests with its AABB on the surface (`rest_height` — for the
        // default 2³ cube this is the historic `surface + 1.0`).
        let projectile = fixtures
            .projectile
            .as_ref()
            .map(|f| f.craft.clone())
            .unwrap_or_else(|| cube_craft(2, Material::COMPOSITE));
        let target = fixtures
            .target
            .as_ref()
            .map(|f| f.craft.clone())
            .unwrap_or_else(|| cube_craft(2, Material::COMPOSITE));

        let mut colliders = vec![
            // 0: projectile, parked to the left at rest (fired with SPACE).
            Collider::new(
                &projectile,
                DVec3::new(-8.0, surface + rest_height(&projectile), 0.0),
                DVec3::ZERO,
            ),
            // 1: target, parked at the origin.
            Collider::new(
                &target,
                DVec3::new(0.0, surface + rest_height(&target), 0.0),
                DVec3::ZERO,
            ),
        ];

        // Debris: three 1-cell fragments dropped straight down with >1 m vertical gaps (so they
        // do not overlap at t=0 — an initial overlap would fling them apart) and a small
        // horizontal offset, off to the side so they don't interfere with the head-on shot; they
        // fall and settle into a small pile.
        let frag = cube_craft(1, Material::ALUMINIUM);
        for (x, y, z) in [(6.0, 0.5, 6.0), (6.1, 2.0, 6.0), (6.0, 3.5, 6.1)] {
            colliders.push(Collider::new(
                &frag,
                DVec3::new(x, surface + y, z),
                DVec3::ZERO,
            ));
        }

        Self {
            colliders,
            ground: ground_half_space(surface),
            params: ContactParams::default(),
            accumulator: 0.0,
            thrust: false,
        }
    }

    /// Render origin for a collider's skin mesh: the mesh is in raw lattice coordinates while
    /// `body.position` is the CoM, so place the mesh's lattice origin at the physical lattice
    /// origin (`body.position − orientation·com`, where the collision shape sits) — the hull
    /// then coincides with the physics (no float/sink).
    fn render_pos(&self, i: usize) -> DVec3 {
        let c = &self.colliders[i];
        c.body.position - c.body.orientation * c.com - DVec3::new(0.0, BODY.radius, 0.0)
    }

    /// Restore every body to its home state (the reset button).
    fn reset(&mut self) {
        for c in &mut self.colliders {
            c.body = c.home;
        }
        self.accumulator = 0.0;
        self.thrust = false;
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
        // Projectile thrust while SPACE is held: a continuous forward (+x) acceleration so it
        // ramps up to speed toward the target the longer it's held.
        if self.thrust {
            acc[0].0 += DVec3::new(self.colliders[0].body.mass * PROJECTILE_ACCEL, 0.0, 0.0);
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
        // Fixture arguments resolve at build so a bad craft name fails fast (WI 843).
        let fixtures = CollideFixtures {
            projectile: fixture_arg(2),
            target: fixture_arg(3),
        };
        app.add_plugins(FloatingOriginPlugin)
            .insert_resource(CollideWorld::new(&fixtures))
            .insert_resource(fixtures)
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
    fixtures: Res<CollideFixtures>,
) {
    info!(
        "collide fixtures: projectile={} target={}",
        slot_name(&fixtures.projectile),
        slot_name(&fixtures.target),
    );
    crate::ground::spawn_ground(&mut commands, &mut meshes, &mut materials, &asset_server);

    // Skins per collider, built from **its own** lattice via the shape-aware submesh
    // path (one submesh per distinct material, plus panel plates) — so loaded shaped
    // crafts render their forms (WI 843). Every piece of a collider carries the same
    // `ColliderTag`, so `track_bodies` moves them together.
    for (i, c) in world.colliders.iter().enumerate() {
        let placement = WorldPos::new(FrameId::CENTRAL_BODY, world.render_pos(i));
        for (material, mesh) in skin_submeshes(&c.voxels, VoxelSkin::Hull) {
            let material = pbr_material(material, &asset_server, &mut materials);
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material),
                Transform::default(),
                WorldPlacement(placement),
                ColliderTag(i),
            ));
        }
        for (mesh, mat) in
            panel_render_pieces(&c.voxels, &asset_server, &mut materials, &mut meshes)
        {
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mat),
                Transform::default(),
                WorldPlacement(placement),
                ColliderTag(i),
            ));
        }
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
        Text::new(format!(
            "hold SPACE to accelerate projectile · R reset\nfixtures: {} / {}",
            slot_name(&fixtures.projectile),
            slot_name(&fixtures.target),
        )),
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

    // A fixed vantage pulled back to frame the projectile, the target, and the debris pile.
    let look = DVec3::new(0.0, 1.0, 2.0);
    let cam = look + DVec3::new(16.0, 12.0, 30.0);
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

/// Holding SPACE accelerates the projectile toward the target; R resets.
fn collide_input(keys: Res<ButtonInput<KeyCode>>, mut world: ResMut<CollideWorld>) {
    world.thrust = keys.pressed(KeyCode::Space);
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

#[cfg(test)]
mod tests {
    use super::*;
    use sounding_sim::shape::{FillMode, Form, ShapedCell};

    #[test]
    fn a_loaded_shaped_target_swaps_only_its_slot_and_rests_by_its_own_height() {
        let mut shaped = cube_craft(2, Material::ALUMINIUM);
        shaped.set_shape(ShapedCell {
            cell: IVec3::new(0, 1, 0),
            form: Form::Wedge,
            orientation: 0,
            fill: FillMode::Solid,
        });
        let fixtures = CollideFixtures {
            projectile: None,
            target: Some(Fixture {
                name: "shaped".into(),
                craft: shaped,
            }),
        };
        let w = CollideWorld::new(&fixtures);
        // Five bodies as always: slots 0/1 plus the untouched three-fragment pile.
        assert_eq!(w.colliders.len(), 5);
        // Slot 1 is the loaded shaped craft (mixed compound, WI 837); slot 0 keeps
        // the procedural cube at the historic `surface + 1.0` rest height.
        assert!(matches!(
            w.colliders[1].shape,
            CollisionShape::Compound { .. }
        ));
        assert!(matches!(
            w.colliders[0].shape,
            CollisionShape::CuboidCompound(_)
        ));
        assert!((w.colliders[0].body.position.y - BODY.radius - 1.0).abs() < 1e-9);
        let want = BODY.radius + rest_height(&w.colliders[1].voxels);
        assert!((w.colliders[1].body.position.y - want).abs() < 1e-9);
    }
}

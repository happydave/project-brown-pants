//! Multi-fluid medium forces and the dive descent (Toy 9, WI 509).
//!
//! The capstone's load-bearing proof: **one fluid module swapping field
//! constants**. `drag_force` and `buoyancy_force` consume only the density a
//! [`FluidMedium`] sample returns — there is **no branch on medium identity**. So
//! the same two functions yield vacuum (ρ=0 → zero), atmospheric (light), and
//! oceanic (heavy) behaviour purely from the sampled constants. That is the
//! governing discipline ("do not hardcode atmosphere") realised as a running
//! descent through vacuum → atmosphere → ocean.
//!
//! [`descent_step`] accumulates gravity + drag + buoyancy and integrates the
//! active body with [`ActiveBody::integrate_wrench`] (the dissipative path the
//! rover uses). [`DiveTriggerPlugin`] composes with WI 508's hand-off: when an
//! on-rails craft's altitude drops below the atmospheric-entry interface it emits
//! `Command::SetGear(Active)`, the design's "atmospheric entry forces a drop out
//! of warp". Headless; the rendered descent scene lives in the app.

use crate::aero;
use crate::command::Command;
use crate::fluid::{FluidMedium, FluidSample};
use crate::handoff::GearKind;
use crate::orbit::Orbit;
use crate::sim::{Craft, SimClock};
use crate::voxel::{Axis, VoxelCraft};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use glam::{DQuat, DVec3};

/// Aero/hydro drag: a force opposing the body's velocity relative to the
/// (static) medium, scaling with the sampled density, speed², a reference area,
/// and a drag coefficient. Medium-agnostic — zero when the medium has no density
/// (vacuum) or the body is at rest.
pub fn drag_force(
    sample: &FluidSample,
    velocity: DVec3,
    area: f64,
    drag_coefficient: f64,
) -> DVec3 {
    let speed = velocity.length();
    if speed <= 0.0 || sample.density <= 0.0 {
        return DVec3::ZERO;
    }
    let dir = velocity / speed;
    -0.5 * sample.density * speed * speed * drag_coefficient * area * dir
}

/// Dynamic (ram) pressure of the flow over the craft: `q = ½·ρ·v²` — the
/// aerodynamic pressure increment at the windward/stagnation (leading) face,
/// where the leading-face total is ambient + q (incompressible Bernoulli).
/// Medium-agnostic and zero in vacuum or at rest. Its peak over a descent is
/// "max-Q", the canonical re-entry stress milestone. (A resolved per-face
/// pressure *distribution* — windward high, leeward low — is the deferred
/// FAR-style aero; this is the single scalar.)
pub fn dynamic_pressure(sample: &FluidSample, velocity: DVec3) -> f64 {
    0.5 * sample.density * velocity.length_squared()
}

/// Buoyancy: the weight of displaced medium, directed `up` (radially outward).
/// Equal to `density · submerged_volume · gravity`. Medium-agnostic — the same
/// formula gives a negligible force in air and a large one in water, purely from
/// the density.
pub fn buoyancy_force(density: f64, submerged_volume: f64, gravity: f64, up: DVec3) -> DVec3 {
    density * submerged_volume * gravity * up
}

/// The volume of the craft below the local surface — the voxel lattice
/// intersected with the sub-surface half-space. A cell is submerged when its
/// world position lies inside the planet sphere (`|p| < surface_radius`). For a
/// craft ≪ planet this is the locally-flat "below sea level" test. `com` is the
/// craft's centre of mass (the active body integrates the CoM), so cell offsets
/// are taken relative to it.
pub fn submerged_volume(
    craft: &VoxelCraft,
    com: DVec3,
    body_position: DVec3,
    body_orientation: DQuat,
    surface_radius: f64,
) -> f64 {
    let mut submerged = 0usize;
    for v in &craft.voxels {
        let local = (v.cell.as_dvec3() + DVec3::splat(0.5)) * craft.cell_size - com;
        let world = body_position + body_orientation * local;
        if world.length() < surface_radius {
            submerged += 1;
        }
    }
    submerged as f64 * craft.cell_volume()
}

/// The largest voxel cross-sectional area over the three axes — a conservative
/// frontal area for drag.
pub fn max_cross_section(craft: &VoxelCraft) -> f64 {
    [Axis::X, Axis::Y, Axis::Z]
        .into_iter()
        .flat_map(|axis| craft.area_curve(axis).into_iter().map(|(_, a)| a))
        .fold(0.0_f64, f64::max)
}

/// The fixed constants of a dive: the medium field, the central body, and the
/// craft's aero reference area + drag coefficient. All SI.
#[derive(Clone, Copy, Debug)]
pub struct DescentParams {
    /// The unified fluid-medium field (atmosphere + ocean).
    pub medium: FluidMedium,
    /// Central-body gravitational parameter (μ = G·M), m³/s².
    pub mu: f64,
    /// Reference surface radius (sea level), m.
    pub surface_radius: f64,
    /// Drag reference area, m².
    pub drag_area: f64,
    /// Drag coefficient (dimensionless).
    pub drag_coefficient: f64,
}

/// Point-mass gravitational force on `mass` at `position` (toward the origin).
fn gravity_force(mass: f64, position: DVec3, mu: f64) -> DVec3 {
    let r2 = position.length_squared();
    if r2 <= 0.0 || !r2.is_finite() {
        return DVec3::ZERO;
    }
    let r = r2.sqrt();
    -mu * mass * position / (r2 * r)
}

/// Advance the active body one step under **gravity + drag + buoyancy**, all
/// drawn from the one medium field, and return the medium sample at the craft
/// (so the caller knows the medium and ambient pressure). `com` is the craft's
/// centre of mass (constant in the body frame; pass it once rather than redoing
/// the eigensolve per sub-step).
pub fn descent_step(
    body: &mut crate::active::ActiveBody,
    craft: &VoxelCraft,
    com: DVec3,
    params: &DescentParams,
    dt: f64,
) -> FluidSample {
    let r = body.position.length();
    let up = if r > 0.0 { body.position / r } else { DVec3::Y };
    let altitude = r - params.surface_radius;
    let sample = params.medium.sample_altitude(altitude);
    let g_local = if r > 0.0 { params.mu / (r * r) } else { 0.0 };

    let gravity = gravity_force(body.mass, body.position, params.mu);
    let drag = drag_force(
        &sample,
        body.velocity,
        params.drag_area,
        params.drag_coefficient,
    );
    let sub_vol = submerged_volume(
        craft,
        com,
        body.position,
        body.orientation,
        params.surface_radius,
    );
    let buoyancy = buoyancy_force(sample.density, sub_vol, g_local, up);

    body.integrate_wrench(gravity + drag + buoyancy, DVec3::ZERO, dt);
    sample
}

/// A unit vector along an [`Axis`].
fn axis_unit(axis: Axis) -> DVec3 {
    match axis {
        Axis::X => DVec3::X,
        Axis::Y => DVec3::Y,
        Axis::Z => DVec3::Z,
    }
}

/// The component of `v` along an [`Axis`].
fn axis_component(v: DVec3, axis: Axis) -> f64 {
    match axis {
        Axis::X => v.x,
        Axis::Y => v.y,
        Axis::Z => v.z,
    }
}

/// The fixed constants of a **gliding** descent: a [`DescentParams`] plus the aero
/// terms that turn a ballistic fall into a glide — lift, the transonic wave drag,
/// the static (weathervaning) pitching moment, and aerodynamic pitch damping. All
/// derived from the **same** voxel `area_curve` the drag uses; medium-agnostic
/// (the [`FluidSample`] parameterises every term), so lift/wave-drag vanish in
/// vacuum and wave drag vanishes in liquid, with no branch on medium identity.
#[derive(Clone, Copy, Debug)]
pub struct GlideParams {
    /// The ballistic descent constants (gravity + drag + buoyancy).
    pub descent: DescentParams,
    /// Lift / wave-drag reference area, m².
    pub lift_area: f64,
    /// Abruptness of the area curve along the forward axis (drives wave drag).
    pub area_ruling_factor: f64,
    /// Body-frame longitudinal ("forward") unit axis — the craft's nose direction.
    pub forward_local: DVec3,
    /// Body-frame offset from the centre of mass to the centre of pressure, m
    /// (along `forward_local`). Aft of the CoM ⇒ statically stable.
    pub cop_offset_local: DVec3,
    /// Aerodynamic pitch-damping coefficient (dimensionless).
    pub pitch_damping: f64,
    /// Characteristic length for the pitch-damping moment, m.
    pub damping_length: f64,
}

impl GlideParams {
    /// Builds glide parameters for a craft, taking `forward` as the longitudinal
    /// axis. Precomputes the lift reference area (the drag reference area), the
    /// area-ruling factor and the centre-of-pressure offset from the craft's own
    /// `area_curve` and centre of mass — so the per-sub-step `glide_step` does no
    /// geometry work. `pitch_damping` and `damping_length` default to mild,
    /// stable values the caller may override.
    pub fn for_craft(descent: DescentParams, craft: &VoxelCraft, forward: Axis) -> Self {
        let curve = craft.area_curve(forward);
        let area_ruling_factor = aero::area_ruling_factor(&curve);
        let cop = aero::center_of_pressure(&curve, craft.cell_size);
        let com = craft
            .mass_properties()
            .map(|mp| axis_component(mp.center_of_mass, forward))
            .unwrap_or(0.0);
        let unit = axis_unit(forward);
        Self {
            lift_area: descent.drag_area,
            area_ruling_factor,
            forward_local: unit,
            cop_offset_local: (cop - com) * unit,
            pitch_damping: 0.5,
            damping_length: 1.0,
            descent,
        }
    }
}

/// Advance the active body one step under **gravity + drag + buoyancy + lift +
/// transonic wave drag**, with a static (weathervaning) pitching moment and
/// aerodynamic pitch damping — the gliding counterpart to [`descent_step`]. At a
/// nonzero angle of attack the lift bends the path away from a pure fall (a glide)
/// and the restoring moment + damping trim the craft toward the flow. Every aero
/// term is drawn from the one medium sample, so all vanish appropriately in
/// vacuum/liquid with no medium-identity branch. Returns the medium sample.
pub fn glide_step(
    body: &mut crate::active::ActiveBody,
    craft: &VoxelCraft,
    com: DVec3,
    params: &GlideParams,
    dt: f64,
) -> FluidSample {
    let p = &params.descent;
    let r = body.position.length();
    let up = if r > 0.0 { body.position / r } else { DVec3::Y };
    let altitude = r - p.surface_radius;
    let sample = p.medium.sample_altitude(altitude);
    let g_local = if r > 0.0 { p.mu / (r * r) } else { 0.0 };

    // Central forces (no moment arm): gravity, drag, buoyancy.
    let gravity = gravity_force(body.mass, body.position, p.mu);
    let drag = drag_force(&sample, body.velocity, p.drag_area, p.drag_coefficient);
    let sub_vol = submerged_volume(
        craft,
        com,
        body.position,
        body.orientation,
        p.surface_radius,
    );
    let buoyancy = buoyancy_force(sample.density, sub_vol, g_local, up);

    // Aero forces from the one area curve: lift ⊥ to the flow, transonic wave drag.
    let forward_world = body.orientation * params.forward_local;
    let lift = aero::lift_force(&sample, body.velocity, forward_world, params.lift_area);
    let wave = aero::wave_drag_force(
        &sample,
        body.velocity,
        params.area_ruling_factor,
        params.lift_area,
    );

    // Moment: lift acting at the centre of pressure (static stability) + damping.
    let cop_world = body.orientation * params.cop_offset_local;
    let restoring = aero::pitching_moment(cop_world, lift);
    let omega = body.angular_velocity();
    let damping = aero::pitch_damping_moment(
        &sample,
        body.velocity.length(),
        omega,
        params.lift_area,
        params.damping_length,
        params.pitch_damping,
    );

    body.integrate_wrench(
        gravity + drag + buoyancy + lift + wave,
        restoring + damping,
        dt,
    );
    sample
}

/// The signed altitude of an on-rails craft at time `t` (planar orbit embedded in
/// the z=0 plane, per the WI 508 bridge).
pub fn rails_altitude(orbit: &Orbit, t: f64, surface_radius: f64) -> f64 {
    orbit.position(t).length() - surface_radius
}

/// The atmospheric-entry interface: the altitude below which an on-rails craft is
/// dropped into active physics.
#[derive(Resource, Clone, Copy, Debug)]
pub struct EntryInterface {
    /// Reference surface radius, m.
    pub surface_radius: f64,
    /// Entry altitude above the surface, m.
    pub altitude: f64,
}

/// Automatically drops an on-rails craft into the active gear at atmospheric
/// entry, by emitting `Command::SetGear(Active)` when its altitude falls below
/// the [`EntryInterface`]. Composes with WI 508's `HandoffPlugin`, which performs
/// the actual wake. This is the automatic trigger WI 508 deferred.
pub struct DiveTriggerPlugin {
    /// The entry interface to install.
    pub interface: EntryInterface,
}

impl Plugin for DiveTriggerPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.interface)
            .add_systems(Update, auto_drop_to_active);
    }
}

fn auto_drop_to_active(
    clock: Res<SimClock>,
    interface: Res<EntryInterface>,
    mut writer: MessageWriter<Command>,
    crafts: Query<&Craft>,
) {
    for craft in &crafts {
        if rails_altitude(&craft.orbit, clock.time, interface.surface_radius) < interface.altitude {
            writer.write(Command::SetGear(GearKind::Active));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active::ActiveBody;
    use crate::command::FlightControlPlugin;
    use crate::fluid::{FluidMedium, MediumKind};
    use crate::handoff::{GearState, HandoffPlugin};
    use crate::sim::{CentralBody, OrbitPlugin};
    use crate::voxel::{Material, Voxel, VoxelCraft};
    use glam::{DVec2, IVec3};

    // Earth-like SI constants for the dive.
    const SURFACE_R: f64 = 6_360_000.0;
    const MU: f64 = 3.986e14; // ~ g·R²

    fn test_craft() -> VoxelCraft {
        // A 2×2×2 m composite block: ~8 m³, denser than water (sinks).
        let mut c = VoxelCraft::new(1.0);
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    c.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
        }
        c
    }

    fn earthlike_params() -> DescentParams {
        let craft = test_craft();
        DescentParams {
            medium: FluidMedium::EARTHLIKE,
            mu: MU,
            surface_radius: SURFACE_R,
            drag_area: max_cross_section(&craft),
            drag_coefficient: 1.0,
        }
    }

    // --- I1: one code path, three media ---

    #[test]
    fn drag_is_zero_in_vacuum_small_in_air_large_in_water() {
        let m = FluidMedium::EARTHLIKE;
        let v = DVec3::new(0.0, -100.0, 0.0);
        let (area, cd) = (4.0, 1.0);
        // True vacuum (ρ exactly 0) → exactly zero drag, one code path.
        let vac = drag_force(&FluidMedium::VACUUM.sample_altitude(0.0), v, area, cd);
        let air = drag_force(&m.sample_altitude(0.0), v, area, cd); // sea-level air
        let water = drag_force(&m.sample_altitude(-10.0), v, area, cd); // 10 m deep
        assert_eq!(vac, DVec3::ZERO, "vacuum drag is zero");
        assert!(air.length() > 0.0);
        // Water is ~840× denser than sea-level air, so drag is far larger.
        assert!(water.length() > 100.0 * air.length());
        // All oppose the velocity (point +y, against the -y motion).
        assert!(air.y > 0.0 && water.y > 0.0);
    }

    #[test]
    fn dynamic_pressure_is_ram_pressure_and_zero_without_flow() {
        let air = FluidMedium::EARTHLIKE.sample_altitude(0.0);
        // At rest: no ram pressure.
        assert_eq!(dynamic_pressure(&air, DVec3::ZERO), 0.0);
        // Vacuum: zero regardless of speed.
        let vac = FluidMedium::VACUUM.sample_altitude(0.0);
        assert_eq!(dynamic_pressure(&vac, DVec3::new(0.0, -2_000.0, 0.0)), 0.0);
        // Sea-level air at 100 m/s: q = ½·1.225·100² = 6125 Pa.
        let q = dynamic_pressure(&air, DVec3::new(0.0, -100.0, 0.0));
        assert!((q - 6_125.0).abs() < 1.0, "q = {q}");
        // Scales with density: water at the same speed is far larger.
        let water = FluidMedium::EARTHLIKE.sample_altitude(-10.0);
        assert!(dynamic_pressure(&water, DVec3::new(0.0, -100.0, 0.0)) > 100.0 * q);
        // Scales with v²: doubling speed quadruples q.
        let q2 = dynamic_pressure(&air, DVec3::new(0.0, -200.0, 0.0));
        assert!((q2 - 4.0 * q).abs() < 1.0);
    }

    #[test]
    fn buoyancy_scales_with_density_one_formula() {
        let up = DVec3::Y;
        let air = buoyancy_force(1.225, 8.0, 9.81, up);
        let water = buoyancy_force(1025.0, 8.0, 9.81, up);
        assert!(water.length() > 100.0 * air.length());
        assert!(air.y > 0.0 && water.y > 0.0, "buoyancy acts up");
        // Vacuum / no displacement → no force.
        assert_eq!(buoyancy_force(0.0, 8.0, 9.81, up), DVec3::ZERO);
        assert_eq!(buoyancy_force(1025.0, 0.0, 9.81, up), DVec3::ZERO);
    }

    // --- I3: physical force directions and submersion ---

    #[test]
    fn drag_opposes_velocity_and_zero_at_rest() {
        let s = FluidMedium::EARTHLIKE.sample_altitude(0.0);
        assert_eq!(drag_force(&s, DVec3::ZERO, 4.0, 1.0), DVec3::ZERO);
        let v = DVec3::new(3.0, -4.0, 0.0);
        let d = drag_force(&s, v, 4.0, 1.0);
        assert!(d.dot(v) < 0.0, "drag must oppose velocity");
    }

    #[test]
    fn submerged_volume_tracks_the_surface() {
        let craft = test_craft();
        let com = craft.mass_properties().unwrap().center_of_mass;
        // Well above the surface: nothing submerged.
        let high = submerged_volume(
            &craft,
            com,
            DVec3::new(0.0, SURFACE_R + 1000.0, 0.0),
            DQuat::IDENTITY,
            SURFACE_R,
        );
        assert_eq!(high, 0.0);
        // Well below: fully submerged (≈ occupied volume).
        let deep = submerged_volume(
            &craft,
            com,
            DVec3::new(0.0, SURFACE_R - 1000.0, 0.0),
            DQuat::IDENTITY,
            SURFACE_R,
        );
        assert!((deep - craft.occupied_volume()).abs() < 1e-9);
    }

    // --- I2 / I4: the continuous descent ---

    #[test]
    fn continuous_descent_reaches_the_ocean_bounded() {
        let params = earthlike_params();
        let craft = test_craft();
        let com = craft.mass_properties().unwrap().center_of_mass;

        // Start 30 km up, moving radially inward at 1.5 km/s (a steep re-entry).
        let start_r = SURFACE_R + 30_000.0;
        let mut body = ActiveBody::new(
            DVec3::new(0.0, start_r, 0.0),
            DVec3::new(0.0, -1_500.0, 0.0),
            craft.mass_properties().unwrap().mass,
            craft.mass_properties().unwrap().inertia,
        );

        let dt = 0.01;
        let mut seen_vacuum_or_thin = false;
        let mut seen_atmosphere = false;
        let mut reached_ocean = false;
        let mut max_speed = 0.0_f64;
        for _ in 0..200_000 {
            let sample = descent_step(&mut body, &craft, com, &params, dt);
            assert!(
                body.position.is_finite() && body.velocity.is_finite(),
                "state stayed finite"
            );
            max_speed = max_speed.max(body.velocity.length());
            match sample.medium {
                MediumKind::Vacuum => seen_vacuum_or_thin = true,
                MediumKind::Atmosphere => seen_atmosphere = true,
                MediumKind::Liquid => {
                    reached_ocean = true;
                    break;
                }
            }
        }
        assert!(seen_atmosphere, "passed through the atmosphere");
        assert!(reached_ocean, "reached the ocean (submerged)");
        // Bounded: never exceeded the entry speed by a meaningful margin (drag only
        // removes energy; gravity adds a little over 30 km).
        assert!(max_speed < 2_000.0, "velocity stayed bounded: {max_speed}");
        let _ = seen_vacuum_or_thin;
    }

    #[test]
    fn descent_force_is_bounded_across_the_surface_density_jump() {
        // Sample drag+buoyancy just above and just below the surface at the same
        // speed; both finite, and the jump is large but not explosive.
        let params = earthlike_params();
        let craft = test_craft();
        let com = craft.mass_properties().unwrap().center_of_mass;
        let v = DVec3::new(0.0, -50.0, 0.0);

        let mut above = ActiveBody::new(
            DVec3::new(0.0, SURFACE_R + 1.0, 0.0),
            v,
            craft.mass_properties().unwrap().mass,
            craft.mass_properties().unwrap().inertia,
        );
        let mut below = above;
        below.position = DVec3::new(0.0, SURFACE_R - 1.0, 0.0);

        let s_above = descent_step(&mut above, &craft, com, &params, 1e-3);
        let s_below = descent_step(&mut below, &craft, com, &params, 1e-3);
        assert_eq!(s_above.medium, MediumKind::Atmosphere);
        assert_eq!(s_below.medium, MediumKind::Liquid);
        assert!(above.velocity.is_finite() && below.velocity.is_finite());
        // The submerged step decelerates much harder (water), but remains finite.
        assert!(below.velocity.length() < v.length() + 1.0);
    }

    // --- Auto-trigger (the WI 508 deferred trigger), through a Bevy App ---

    #[test]
    fn auto_drop_to_active_fires_below_the_interface() {
        // A low circular orbit whose altitude is already below the entry interface.
        let low_r = SURFACE_R + 5_000.0;
        let speed = (MU / low_r).sqrt();
        let orbit =
            Orbit::from_state(MU, DVec2::new(low_r, 0.0), DVec2::new(0.0, speed), 0.0).unwrap();

        let mut app = App::new();
        app.add_plugins(bevy_time::TimePlugin);
        app.add_plugins(OrbitPlugin {
            central_body: CentralBody {
                mu: MU,
                radius: SURFACE_R,
            },
            initial_orbit: orbit,
        });
        app.add_plugins(crate::active::ActivePlugin { mu: MU });
        app.add_plugins(FlightControlPlugin);
        app.add_plugins(HandoffPlugin);
        app.add_plugins(DiveTriggerPlugin {
            interface: EntryInterface {
                surface_radius: SURFACE_R,
                altitude: 100_000.0, // 100 km — the craft at 5 km is well below
            },
        });
        // Ensure the craft can wake (a real gear-state).
        let craft = test_craft();
        let mp = craft.mass_properties().unwrap();
        {
            let mut q = app.world_mut().query_filtered::<Entity, With<Craft>>();
            let e = q.single(app.world()).unwrap();
            app.world_mut()
                .entity_mut(e)
                .insert(GearState::new(mp.mass, mp.inertia));
        }

        // A couple of updates: the trigger fires and the hand-off wakes the craft.
        app.update();
        app.update();

        let mut q = app
            .world_mut()
            .query::<(Option<&Craft>, Option<&ActiveBody>)>();
        let (on_rails, active) = q.single(app.world()).unwrap();
        assert!(active.is_some(), "craft should have been woken to active");
        assert!(on_rails.is_none(), "craft should have left rails");
    }

    #[test]
    fn rails_altitude_is_signed_about_the_surface() {
        let orbit = Orbit::from_state(
            MU,
            DVec2::new(SURFACE_R + 10_000.0, 0.0),
            DVec2::new(0.0, (MU / (SURFACE_R + 10_000.0)).sqrt()),
            0.0,
        )
        .unwrap();
        assert!((rails_altitude(&orbit, 0.0, SURFACE_R) - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn max_cross_section_of_a_block_is_a_face() {
        // A 2×2×2 block of 1 m cells: each axis slice is 2×2 = 4 m².
        let craft = test_craft();
        assert!((max_cross_section(&craft) - 4.0).abs() < 1e-9);
    }

    // --- WI 526: gliding re-entry (lift + moment + wave drag applied) ---

    /// A symmetric, **tapered, heavy-nose** craft: a 3×3 base + body along +Z with
    /// a single centred steel nose cell at the tip. The taper gives a nonzero
    /// area-ruling factor (so wave drag exists) and, with the dense nose moving the
    /// centre of mass forward of the geometric (area) centroid, a positive static
    /// margin (centre of pressure aft of the centre of mass → weathervaning).
    /// Forward axis is +Z.
    fn glide_craft() -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        for z in 0..4 {
            for x in 0..3 {
                for y in 0..3 {
                    c.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
        }
        // A denser, centred nose tip at the +Z end — a small positive static
        // margin (gentle weathervaning, controllable; not a hard snap-to-flow).
        c.voxels.push(Voxel {
            cell: IVec3::new(1, 1, 4),
            material: Material::ALUMINIUM,
        });
        c
    }

    fn glide_params(craft: &VoxelCraft) -> (DescentParams, GlideParams) {
        let descent = DescentParams {
            medium: FluidMedium::EARTHLIKE,
            mu: MU,
            surface_radius: SURFACE_R,
            drag_area: max_cross_section(craft),
            drag_coefficient: 1.0,
        };
        let glide = GlideParams::for_craft(descent, craft, Axis::Z);
        (descent, glide)
    }

    /// Acute angle of attack between the body forward axis and the velocity.
    fn aoa(orientation: DQuat, forward_local: DVec3, velocity: DVec3) -> f64 {
        use std::f64::consts::{FRAC_PI_2, PI};
        let f = (orientation * forward_local).normalize();
        let v = velocity.normalize_or_zero();
        if v == DVec3::ZERO {
            return 0.0;
        }
        let a = f.dot(v).clamp(-1.0, 1.0).acos();
        if a > FRAC_PI_2 {
            PI - a
        } else {
            a
        }
    }

    /// An orientation putting the forward axis at angle `alpha` to a straight-down
    /// (−Y) velocity, tilted toward +X (so lift acts in +X).
    fn entry_orientation(forward_local: DVec3, alpha: f64) -> DQuat {
        let desired = DVec3::new(alpha.sin(), -alpha.cos(), 0.0);
        DQuat::from_rotation_arc(forward_local, desired)
    }

    #[test]
    fn the_static_margin_is_positive() {
        // Sanity on the fixture: the centre of pressure is aft of the centre of
        // mass along +Z (a stable craft), and the taper gives wave-drag headroom.
        let craft = glide_craft();
        let (_d, glide) = glide_params(&craft);
        assert!(
            glide.cop_offset_local.z < 0.0,
            "CoP should be aft (−Z) of CoM: {:?}",
            glide.cop_offset_local
        );
        assert!(
            glide.area_ruling_factor > 0.0,
            "tapered body has a nonzero area-ruling factor"
        );
    }

    #[test]
    fn lift_produces_a_glide_vs_ballistic() {
        let craft = glide_craft();
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let (descent, glide) = glide_params(&craft);

        let start = DVec3::new(0.0, SURFACE_R + 3_000.0, 0.0);
        let v0 = DVec3::new(0.0, -300.0, 0.0);
        let mut body = ActiveBody::new(start, v0, mp.mass, mp.inertia);
        body.orientation = entry_orientation(glide.forward_local, 0.35); // ~20° AoA
        let mut ballistic = body; // identical start, no lift

        let dt = 0.004;
        for _ in 0..2_000 {
            glide_step(&mut body, &craft, com, &glide, dt);
            descent_step(&mut ballistic, &craft, com, &descent, dt);
        }

        // The glider converted descent into downrange (+X) motion; the ballistic
        // reference fell straight down.
        assert!(
            body.velocity.x > 5.0,
            "glide should gain horizontal velocity: {}",
            body.velocity.x
        );
        assert!(
            ballistic.velocity.x.abs() < 1e-6,
            "ballistic stays vertical: {}",
            ballistic.velocity.x
        );
        assert!(
            body.position.x > ballistic.position.x + 5.0,
            "glide path deflects downrange"
        );
        assert!(body.position.is_finite() && body.velocity.is_finite());
    }

    #[test]
    fn statically_stable_craft_trims_and_does_not_tumble() {
        let craft = glide_craft();
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let (_d, glide) = glide_params(&craft);

        let mut body = ActiveBody::new(
            DVec3::new(0.0, SURFACE_R + 3_000.0, 0.0),
            DVec3::new(0.0, -300.0, 0.0),
            mp.mass,
            mp.inertia,
        );
        body.orientation = entry_orientation(glide.forward_local, 0.6); // ~34° AoA

        let dt = 0.004;
        let mut early_max = 0.0_f64;
        let mut late_max = 0.0_f64;
        let mut max_omega = 0.0_f64;
        for i in 0..4_000 {
            glide_step(&mut body, &craft, com, &glide, dt);
            let a = aoa(body.orientation, glide.forward_local, body.velocity);
            if i < 1_000 {
                early_max = early_max.max(a);
            } else if i >= 3_000 {
                late_max = late_max.max(a);
            }
            max_omega = max_omega.max(body.angular_velocity().length());
        }
        // The pitch oscillation decays toward trim (damping), and the craft never
        // tumbles (angular velocity stays bounded).
        assert!(
            late_max < early_max,
            "AoA oscillation should decay toward trim: early {early_max} late {late_max}"
        );
        assert!(max_omega < 5.0, "no tumble (bounded ω): {max_omega}");
        assert!(body.orientation.is_finite());
    }

    #[test]
    fn wave_drag_included_in_air_absent_in_water() {
        let craft = glide_craft();
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let (descent, glide) = glide_params(&craft);

        // Forward aligned with the velocity → zero AoA → zero lift/moment, so the
        // only glide/ballistic difference is wave drag.
        let aligned = DQuat::from_rotation_arc(glide.forward_local, DVec3::NEG_Y);
        let v = DVec3::new(0.0, -400.0, 0.0); // transonic in dense air

        // Atmosphere (transonic, ~1 km up): wave drag adds deceleration.
        let mut g_air = ActiveBody::new(
            DVec3::new(0.0, SURFACE_R + 1_000.0, 0.0),
            v,
            mp.mass,
            mp.inertia,
        );
        g_air.orientation = aligned;
        let mut d_air = g_air;
        let s = glide_step(&mut g_air, &craft, com, &glide, 0.004);
        descent_step(&mut d_air, &craft, com, &descent, 0.004);
        assert_eq!(s.medium, MediumKind::Atmosphere);
        assert!(
            g_air.velocity.length() < d_air.velocity.length(),
            "transonic air: wave drag decelerates the glider more than plain drag"
        );

        // Ocean (submerged): wave drag is exactly zero (incompressible) and AoA is
        // zero, so the glide step is linearly identical to the ballistic step.
        let mut g_w = ActiveBody::new(
            DVec3::new(0.0, SURFACE_R - 10.0, 0.0),
            v,
            mp.mass,
            mp.inertia,
        );
        g_w.orientation = aligned;
        let mut d_w = g_w;
        let sw = glide_step(&mut g_w, &craft, com, &glide, 0.004);
        descent_step(&mut d_w, &craft, com, &descent, 0.004);
        assert_eq!(sw.medium, MediumKind::Liquid);
        assert!(
            (g_w.velocity - d_w.velocity).length() < 1e-9,
            "ocean: no wave drag and no lift → identical to ballistic"
        );
    }
}

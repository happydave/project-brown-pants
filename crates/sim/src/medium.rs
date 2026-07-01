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
use crate::compartments::compartments;
use crate::fluid::{FluidMedium, FluidSample};
use crate::handoff::GearKind;
use crate::orbit::Orbit;
use crate::sim::{Craft, SimClock};
use crate::voxel::{Axis, VoxelCraft};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_time::prelude::*;
use glam::{DQuat, DVec3, IVec3};
use std::collections::HashSet;

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

/// A representative water-entry slamming coefficient (dimensionless, WI 700) — chosen so the
/// transient entry load exceeds the steady drag at the same closing speed (a slam peak above
/// the `½·C_d·ρ·v²` steady drag with `C_d ≈ 1`).
pub const DEFAULT_SLAM_COEFFICIENT: f64 = 3.0;

/// Water-entry **slamming** load (WI 700): the transient impact force as a craft pierces the
/// surface, *beyond* steady drag. Models the momentum flux into the water the craft entrains
/// as it penetrates (the added-mass slam):
/// `F = slam_coefficient · ρ_water · A_entry · v_in² · (1 − submerged_fraction)`, directed
/// along the outward surface normal `up` (opposing penetration), where `v_in` is the closing
/// (into-water) speed — the inward normal component of velocity.
///
/// **Gated to the entry window:** nonzero only while the craft *straddles* the surface
/// (`0 < submerged_fraction < 1`) and is *descending*, and it **decays as it submerges**
/// (peaking near first contact). Zero with no ocean (`water_density = 0`), at rest, when
/// rising or skimming, and once fully in or out of the water — where steady drag + buoyancy
/// govern alone. A `½ρv²`-family form, so it stays finite and bounded.
///
/// `water_density` is the surrounding **liquid** density (not the air the centre of mass may
/// still sit in during entry); `up` is the unit outward surface normal. `water_velocity` is the
/// local velocity of the water surface itself — **zero in calm water** (WI 705 forward hook #2):
/// the closing speed is the *relative* normal speed `(v_hull − v_water)·(−up)`, so a future wave
/// field that heaves the surface re-fires the slam as the hull re-enters, with no edit here.
pub fn entry_impact_force(
    water_density: f64,
    velocity: DVec3,
    water_velocity: DVec3,
    up: DVec3,
    entry_area: f64,
    submerged_fraction: f64,
    slam_coefficient: f64,
) -> DVec3 {
    if water_density <= 0.0
        || entry_area <= 0.0
        || slam_coefficient <= 0.0
        || submerged_fraction <= 0.0
        || submerged_fraction >= 1.0
    {
        return DVec3::ZERO;
    }
    // Closing speed into the surface: the inward (−up) component of the velocity of the hull
    // *relative to the water surface* (calm water ⇒ `water_velocity == 0`).
    let closing = -(velocity - water_velocity).dot(up);
    if closing <= 0.0 {
        return DVec3::ZERO; // rising or skimming the surface: no entry slam
    }
    let decay = 1.0 - submerged_fraction; // peak at first contact, → 0 as it fully submerges
    slam_coefficient * water_density * entry_area * closing * closing * decay * up
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

/// The effective water-surface radius at a world position and time — the single **seam** a
/// future wave field plugs into (WI 705 forward hook #1). Today it returns the flat sea-level
/// datum; a wave implementation will return `surface_radius + wave_height(world, time)` here, and
/// nothing in [`buoyancy_wrench`] or [`entry_impact_force`] changes. `time` is unused in calm
/// water but is part of the contract so a heaving surface needs no signature change later.
#[inline]
fn water_surface_radius(_world: DVec3, _time: f64, surface_radius: f64) -> f64 {
    surface_radius
}

/// The graded submerged fraction of a single cell whose centre sits at world `position`: `1`
/// when the cell is a full cell-height below the local waterline, `0` a full cell-height above,
/// ramping linearly (C0, monotone) through a band ~one cell thick across the surface. This is
/// the **surface-hydrodynamics** smoothing: it removes the binary waterline step so buoyancy is
/// continuous through the surface (and damps the `dF/dz` kink that would otherwise limit-cycle).
#[inline]
fn cell_submerged_fraction(position: DVec3, cell_size: f64, surface_radius: f64, time: f64) -> f64 {
    let depth = water_surface_radius(position, time, surface_radius) - position.length();
    (0.5 + depth / cell_size).clamp(0.0, 1.0)
}

/// The hydrostatic load on a craft: the buoyant **wrench** (force + moment about the centre of
/// mass) plus the readouts a HUD/telemetry wants. The force equals the displaced-medium weight
/// (identical to the central [`buoyancy_force`]); the **moment** — `Σ rᵢ × Fᵢ` over the graded
/// cells — is the righting couple that makes a hull self-right and trim (WI 705).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BuoyancyLoad {
    /// Resultant buoyant force in the world frame (along the local up).
    pub force: DVec3,
    /// Buoyant moment about the centre of mass, world frame — the righting/trim couple.
    pub torque: DVec3,
    /// Graded displaced volume, m³ (`Σ fractionᵢ · cell_volume`).
    pub submerged_volume: f64,
    /// Draft: the maximum depth of any submerged cell below the local waterline, m (0 if dry).
    pub draft: f64,
}

/// Accumulate the buoyant wrench over the craft's cells in **one** O(cells) pass: each cell
/// contributes its graded displaced weight along the local up, applied **at the cell's location**
/// (never lumped at the CoM), and the moment of that force about the centre of mass. Stability
/// (metacentric righting, GM > 0) therefore **emerges** from the geometry — it requires the
/// displaced cells to be spread across the beam; a centreline-only distribution yields ~zero roll
/// stiffness. Returns zero force *and* moment in vacuum (`density == 0`) or fully out of the water.
///
/// `up` is taken once as the radial out-vector at `body_position` (the craft ≪ planet), so the
/// force matches the central [`buoyancy_force`] exactly and only the moment is new. `time` feeds
/// the [`water_surface_radius`] seam (unused in calm water).
///
/// `enclosed` are the **enclosed airtight-compartment cells** (WI 711): a hollow hull floats on the
/// water its hull encloses, not just the shell it is built from. They displace exactly like solid
/// cells (graded, at their own location). Pass an empty slice for solid-only displacement (the WI 705
/// behaviour). Derive them once via [`enclosed_cells`] and cache — do not recompute per sub-step.
#[allow(clippy::too_many_arguments)]
pub fn buoyancy_wrench(
    craft: &VoxelCraft,
    com: DVec3,
    body_position: DVec3,
    body_orientation: DQuat,
    surface_radius: f64,
    time: f64,
    density: f64,
    gravity: f64,
    enclosed: &[IVec3],
) -> BuoyancyLoad {
    if density <= 0.0 || gravity <= 0.0 {
        return BuoyancyLoad::default();
    }
    let r = body_position.length();
    let up = if r > 0.0 { body_position / r } else { DVec3::Y };
    let cell_volume = craft.cell_volume();
    let mut force = DVec3::ZERO;
    let mut torque = DVec3::ZERO;
    let mut submerged_volume = 0.0;
    let mut draft = 0.0_f64;
    // Displacement = solid voxels (the hull) + enclosed compartment cells (the air it encloses). A
    // panel voxel displaces only `voxel_fill` of a cell (WI 716, a thin plate); enclosed air is a full
    // cell.
    let cells = craft
        .voxels
        .iter()
        .map(|v| (v.cell, craft.voxel_fill(v.cell)))
        .chain(enclosed.iter().map(|c| (*c, 1.0)));
    for (cell, fill) in cells {
        let local = (cell.as_dvec3() + DVec3::splat(0.5)) * craft.cell_size - com;
        let world = body_position + body_orientation * local;
        let fraction = cell_submerged_fraction(world, craft.cell_size, surface_radius, time);
        if fraction <= 0.0 {
            continue;
        }
        let displaced = cell_volume * fill * fraction; // the water this cell displaces
        let f = density * gravity * displaced * up;
        let arm = world - body_position; // offset from the centre of mass (the body integrates the CoM)
        force += f;
        torque += arm.cross(f);
        submerged_volume += displaced;
        draft = draft.max(surface_radius - world.length());
    }
    BuoyancyLoad {
        force,
        torque,
        submerged_volume,
        draft: draft.max(0.0),
    }
}

/// The enclosed airtight-compartment cells of a craft, flattened (WI 711) — the cells a hollow hull
/// displaces in addition to its shell. Computed via [`crate::compartments::compartments`]; cache the
/// result (e.g. on [`DivingCraft`]) rather than recomputing per sub-step.
pub fn enclosed_cells(craft: &VoxelCraft) -> Vec<IVec3> {
    compartments(craft)
        .compartments
        .iter()
        .flat_map(|c| c.cells.iter().copied())
        .collect()
}

/// A hull's **open** (un-sealed) cavity (WI 713): the interior air the hull holds out from the water up
/// to its open **rim**. Unlike a sealed compartment (WI 711, which displaces at *any* depth), an open
/// cavity floods when its rim submerges — an open boat floats on its held-out volume and **swamps** when
/// the gunwale goes under. `cells` are the bucket-interior air cells; `rim_cells` the topmost cavity
/// cells (the opening edge, where water spills in first). Empty for a sealed/solid hull.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenCavity {
    /// Interior air cells of the open cavity (the held-out volume).
    pub cells: Vec<IVec3>,
    /// The opening edge — the topmost cavity cells; the lowest of these in world space is where water
    /// first spills in (so a heeled hull ships water on the low side first).
    pub rim_cells: Vec<IVec3>,
}

/// Compute a hull's [`OpenCavity`] (WI 713): exterior-reachable empty cells that sit **inside** the hull
/// — a solid cell below them in their column (a floor), at or below the hull's top (under the gunwale) —
/// i.e. the bucket interior. Empty for a sealed hull (its interior is not exterior-reachable) or a solid
/// block. A first cut for **top-opening** hulls; side-hole openings below the waterline are a later item.
pub fn open_cavity(craft: &VoxelCraft) -> OpenCavity {
    let solid: HashSet<IVec3> = craft.voxels.iter().map(|v| v.cell).collect();
    if solid.is_empty() {
        return OpenCavity::default();
    }
    let (mut lo, mut hi) = (IVec3::MAX, IVec3::MIN);
    for &c in &solid {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    let max_solid_y = hi.y;
    let (elo, ehi) = (lo - IVec3::ONE, hi + IVec3::ONE);
    let in_bbox = |c: IVec3| {
        c.x >= elo.x && c.x <= ehi.x && c.y >= elo.y && c.y <= ehi.y && c.z >= elo.z && c.z <= ehi.z
    };
    let is_air = |c: IVec3| in_bbox(c) && !solid.contains(&c);
    let neighbours = [
        IVec3::X,
        IVec3::NEG_X,
        IVec3::Y,
        IVec3::NEG_Y,
        IVec3::Z,
        IVec3::NEG_Z,
    ];
    // Flood-fill the exterior (everything reachable from the border through air).
    let mut exterior = HashSet::new();
    let mut stack = Vec::new();
    for x in elo.x..=ehi.x {
        for y in elo.y..=ehi.y {
            for z in elo.z..=ehi.z {
                let c = IVec3::new(x, y, z);
                let border = x == elo.x
                    || x == ehi.x
                    || y == elo.y
                    || y == ehi.y
                    || z == elo.z
                    || z == ehi.z;
                if border && is_air(c) && exterior.insert(c) {
                    stack.push(c);
                }
            }
        }
    }
    while let Some(c) = stack.pop() {
        for off in neighbours {
            let n = c + off;
            if is_air(n) && exterior.insert(n) {
                stack.push(n);
            }
        }
    }
    // The bucket interior: exterior-reachable air with a floor below it, under the gunwale.
    let has_solid_below = |c: IVec3| (lo.y..c.y).any(|y| solid.contains(&IVec3::new(c.x, y, c.z)));
    let mut cells: Vec<IVec3> = exterior
        .iter()
        .copied()
        .filter(|&c| c.y <= max_solid_y && has_solid_below(c))
        .collect();
    if cells.is_empty() {
        return OpenCavity::default();
    }
    cells.sort_by_key(|c| (c.y, c.x, c.z));
    let rim_y = cells.iter().map(|c| c.y).max().unwrap();
    let rim_cells = cells.iter().copied().filter(|c| c.y == rim_y).collect();
    OpenCavity { cells, rim_cells }
}

/// The buoyant **wrench** from a hull's [`OpenCavity`] (WI 713): each interior cell displaces graded
/// submerged water scaled by a **rim factor** — `clamp(rim_alt / cell, 0, 1)` where `rim_alt` is the
/// lowest rim cell's height above the local waterline. So the cavity displaces fully while the gunwale
/// is a cell above water and **ramps to zero (swamps) as the rim reaches the surface**. Per-cell at its
/// own location (a wrench), so it contributes a righting moment and a heeled hull ships water low-side
/// first. Zero for a sealed/empty cavity, in vacuum, or once swamped. Add to the shell
/// The open cavity's **rim factor** at a pose (WI 713/728): `1` while the gunwale (the **lowest** rim
/// cell) is a cell above the waterline (floating — the cavity holds air), ramping to `0` as the rim
/// submerges (swamped). It scales open-cavity displacement, and `1 - factor` is the share of the cavity
/// that has shipped water — the interior-water level the WI 718 viz raises as an open boat swamps. Zero
/// when there is no rim (no open cavity).
pub fn open_cavity_rim_factor(
    craft: &VoxelCraft,
    com: DVec3,
    body_position: DVec3,
    body_orientation: DQuat,
    surface_radius: f64,
    time: f64,
    open: &OpenCavity,
) -> f64 {
    if open.rim_cells.is_empty() {
        return 0.0;
    }
    let mut rim_alt = f64::INFINITY;
    for &c in &open.rim_cells {
        let local = (c.as_dvec3() + DVec3::splat(0.5)) * craft.cell_size - com;
        let world = body_position + body_orientation * local;
        let alt = world.length() - water_surface_radius(world, time, surface_radius);
        rim_alt = rim_alt.min(alt);
    }
    (rim_alt / craft.cell_size).clamp(0.0, 1.0)
}

/// [`buoyancy_wrench`]; the band-smoothing keeps the swamp continuous (no buoyancy cliff).
#[allow(clippy::too_many_arguments)]
pub fn open_cavity_load(
    craft: &VoxelCraft,
    com: DVec3,
    body_position: DVec3,
    body_orientation: DQuat,
    surface_radius: f64,
    time: f64,
    density: f64,
    gravity: f64,
    open: &OpenCavity,
) -> BuoyancyLoad {
    if density <= 0.0 || gravity <= 0.0 || open.cells.is_empty() {
        return BuoyancyLoad::default();
    }
    let r = body_position.length();
    let up = if r > 0.0 { body_position / r } else { DVec3::Y };
    let cell_volume = craft.cell_volume();
    let rim_factor = open_cavity_rim_factor(
        craft,
        com,
        body_position,
        body_orientation,
        surface_radius,
        time,
        open,
    );
    if rim_factor <= 0.0 {
        return BuoyancyLoad::default(); // swamped: the gunwale is under, the cavity has flooded
    }
    let mut force = DVec3::ZERO;
    let mut torque = DVec3::ZERO;
    let mut submerged_volume = 0.0;
    let mut draft = 0.0_f64;
    for &c in &open.cells {
        let local = (c.as_dvec3() + DVec3::splat(0.5)) * craft.cell_size - com;
        let world = body_position + body_orientation * local;
        let fraction = cell_submerged_fraction(world, craft.cell_size, surface_radius, time);
        if fraction <= 0.0 {
            continue;
        }
        let displaced = cell_volume * rim_factor * fraction;
        let f = density * gravity * displaced * up;
        let arm = world - body_position;
        force += f;
        torque += arm.cross(f);
        submerged_volume += displaced;
        draft = draft.max(surface_radius - world.length());
    }
    BuoyancyLoad {
        force,
        torque,
        submerged_volume,
        draft: draft.max(0.0),
    }
}

/// Heel/tilt angle of a body from upright: the angle (radians) between the craft's body-up axis
/// (`orientation · +Y`) and the local up. Zero when level; `π` when fully inverted. A pure
/// function of pose — the readout a boat HUD/telemetry surfaces (WI 705).
pub fn heel_angle(orientation: DQuat, up: DVec3) -> f64 {
    let body_up = orientation * DVec3::Y;
    body_up
        .normalize_or_zero()
        .dot(up.normalize_or_zero())
        .clamp(-1.0, 1.0)
        .acos()
}

/// Free-surface damping rate (WI 705): the dimensionless coefficient on the dissipative resistance
/// a hull feels working the water. The buoyant spring is conservative and the bulk drag acts on the
/// centre-of-mass velocity only, so heave and (especially) roll about the CoM would otherwise ring;
/// this is the physical loss (radiation + viscous) that settles them. Scale-relative — the force is
/// `∝ density · cell_area`, so it sizes itself to the craft and medium rather than being an absolute.
pub const FREE_SURFACE_DAMPING: f64 = 2.0;

/// The dissipative free-surface damping **wrench** about the centre of mass (WI 705): each submerged
/// cell opposes its **own** velocity component along the local up (heave + roll motion), with a
/// resistance `∝ density · cell_area · fraction`. Kept separate from [`buoyancy_wrench`] so the
/// hydrostatic force/draft/telemetry stay exactly the displaced weight; this only removes energy.
/// Strictly dissipative (power `Σ f·v ≤ 0`) and zero at rest, in vacuum, or out of the water.
///
/// Yaw damping (WI 730): the up-component term alone leaves **yaw** undamped — yaw sweeps cells
/// horizontally in the waterplane, where the up-component is ~zero — so a steered hull would spin in
/// place after the rudder centres. Each cell additionally opposes the **horizontal** part of its
/// *rotational* velocity (`angular_velocity × arm`, up-component removed). It is rotation-only (zero
/// when `angular_velocity` is zero) so it bleeds off yaw and the horizontal sweep of roll/pitch without
/// adding per-cell surge drag that would fight the thruster's forward drive; translation stays the job
/// of the bulk drag. Same scale-relative sizing, still strictly dissipative and zero at rest.
#[allow(clippy::too_many_arguments)]
pub fn free_surface_damping(
    craft: &VoxelCraft,
    com: DVec3,
    body_position: DVec3,
    body_orientation: DQuat,
    body_velocity: DVec3,
    angular_velocity: DVec3,
    surface_radius: f64,
    time: f64,
    density: f64,
    coefficient: f64,
) -> (DVec3, DVec3) {
    if density <= 0.0 || coefficient <= 0.0 {
        return (DVec3::ZERO, DVec3::ZERO);
    }
    let r = body_position.length();
    let up = if r > 0.0 { body_position / r } else { DVec3::Y };
    let cell_area = craft.cell_size * craft.cell_size; // a cell's waterplane footprint
    let mut force = DVec3::ZERO;
    let mut torque = DVec3::ZERO;
    for v in &craft.voxels {
        let local = (v.cell.as_dvec3() + DVec3::splat(0.5)) * craft.cell_size - com;
        let world = body_position + body_orientation * local;
        let fraction = cell_submerged_fraction(world, craft.cell_size, surface_radius, time);
        if fraction <= 0.0 {
            continue;
        }
        let arm = world - body_position;
        let v_cell = body_velocity + angular_velocity.cross(arm);
        let resistance = coefficient * density * cell_area * fraction;
        // Heave/roll/pitch: oppose the cell's velocity along local up.
        let v_up = v_cell.dot(up);
        let mut f = -resistance * v_up * up;
        // Yaw (WI 730): oppose the horizontal part of the cell's rotational velocity only, so a
        // steered hull bleeds off its yaw rate instead of spinning in place — without damping surge.
        let v_rot = angular_velocity.cross(arm);
        let v_rot_h = v_rot - v_rot.dot(up) * up;
        f -= resistance * v_rot_h;
        force += f;
        torque += arm.cross(f);
    }
    (force, torque)
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
    /// Water-entry slamming coefficient (dimensionless, WI 700) — scales the transient
    /// impact load as the craft pierces the surface, above steady drag. `0.0` disables the
    /// slam (steady drag only).
    pub slam_coefficient: f64,
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

/// A net force + torque about the centre of mass — the contract currency of the active force
/// contributors (convergence Split B2, WI 737). Each contributor returns a `Wrench` (zero outside its
/// regime), and an assembly is their additive sum, so a shared force is defined once and felt in every
/// active path. `From<(force, torque)>` lets existing tuple-returning terms compose.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Wrench {
    pub force: DVec3,
    pub torque: DVec3,
}

impl Wrench {
    pub const ZERO: Self = Self {
        force: DVec3::ZERO,
        torque: DVec3::ZERO,
    };
}

impl std::ops::Add for Wrench {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            force: self.force + o.force,
            torque: self.torque + o.torque,
        }
    }
}

impl std::ops::AddAssign for Wrench {
    fn add_assign(&mut self, o: Self) {
        self.force += o.force;
        self.torque += o.torque;
    }
}

impl From<(DVec3, DVec3)> for Wrench {
    fn from((force, torque): (DVec3, DVec3)) -> Self {
        Self { force, torque }
    }
}

/// The output of the shared core contributor: the core wrench, the medium sample at the craft, and the
/// buoyancy load (callers read `submerged_volume` for the slam gate).
#[derive(Clone, Copy, Debug)]
pub struct CoreWrench {
    pub wrench: Wrench,
    pub sample: FluidSample,
    pub buoyancy: BuoyancyLoad,
}

/// The **shared, self-gating core** of every active force path (convergence Split B2, WI 737):
/// gravity + drag + distributed buoyancy (righting wrench) + free-surface damping, all from the one
/// medium field. Each term is zero outside its regime (buoyancy/damping vanish out of water, drag at
/// rest/in vacuum), so this is safe to run on any craft. Returns the core wrench, the medium sample,
/// and the buoyancy load. `descent_step`, `glide_step`, and `flight_step` compose their own terms
/// (slam / lift / thrust / attitude / …) on top of this.
#[allow(clippy::too_many_arguments)]
pub fn core_active_wrench(
    craft: &VoxelCraft,
    com: DVec3,
    body: &crate::active::ActiveBody,
    medium: &FluidMedium,
    surface_radius: f64,
    mu: f64,
    drag_area: f64,
    drag_coefficient: f64,
    enclosed: &[IVec3],
) -> CoreWrench {
    let r = body.position.length();
    let altitude = r - surface_radius;
    let sample = medium.sample_altitude(altitude);
    let g_local = if r > 0.0 { mu / (r * r) } else { 0.0 };

    let gravity = gravity_force(body.mass, body.position, mu);
    let drag = drag_force(&sample, body.velocity, drag_area, drag_coefficient);
    let buoyancy = buoyancy_wrench(
        craft,
        com,
        body.position,
        body.orientation,
        surface_radius,
        0.0, // calm water (WI 705 seam): wave-time threads sim time here
        sample.density,
        g_local,
        enclosed,
    );
    let (damp_f, damp_t) = free_surface_damping(
        craft,
        com,
        body.position,
        body.orientation,
        body.velocity,
        body.angular_velocity(),
        surface_radius,
        0.0,
        sample.density,
        FREE_SURFACE_DAMPING,
    );

    CoreWrench {
        wrench: Wrench {
            force: gravity + drag + buoyancy.force + damp_f,
            torque: buoyancy.torque + damp_t,
        },
        sample,
        buoyancy,
    }
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
    enclosed: &[IVec3],
    params: &DescentParams,
    dt: f64,
) -> FluidSample {
    let r = body.position.length();
    let up = if r > 0.0 { body.position / r } else { DVec3::Y };

    // Shared core (WI 737): gravity + drag + distributed righting buoyancy + free-surface damping.
    let core = core_active_wrench(
        craft,
        com,
        body,
        &params.medium,
        params.surface_radius,
        params.mu,
        params.drag_area,
        params.drag_coefficient,
        enclosed,
    );
    // Water-entry slam (WI 700): a transient impact while the craft straddles the surface. The gate
    // fraction is over the **total displaceable** volume (solid + enclosed, WI 711), so it stays in
    // [0,1].
    let displaceable = craft.occupied_volume() + enclosed.len() as f64 * craft.cell_volume();
    let submerged_fraction = if displaceable > 0.0 {
        core.buoyancy.submerged_volume / displaceable
    } else {
        0.0
    };
    let water_density = params.medium.sample_altitude(-1.0).density;
    let slam = entry_impact_force(
        water_density,
        body.velocity,
        DVec3::ZERO, // calm water: still surface (WI 705 forward hook #2)
        up,
        params.drag_area,
        submerged_fraction,
        params.slam_coefficient,
    );

    let w = core.wrench + Wrench::from((slam, DVec3::ZERO));
    body.integrate_wrench(w.force, w.torque, dt);
    core.sample
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
///
/// `external` is a world-frame wrench `(force, torque about the CoM)` summed with the
/// aero/hydro forces **before** the single integration step — the seam through which
/// device forces enter (marine propulsion, WI 708). Pass `(DVec3::ZERO, DVec3::ZERO)`
/// for none; the result is then identical to the pure aero/hydro descent.
#[allow(clippy::too_many_arguments)]
pub fn glide_step(
    body: &mut crate::active::ActiveBody,
    craft: &VoxelCraft,
    com: DVec3,
    enclosed: &[IVec3],
    params: &GlideParams,
    external: (DVec3, DVec3),
    dt: f64,
) -> FluidSample {
    let p = &params.descent;
    let r = body.position.length();
    let up = if r > 0.0 { body.position / r } else { DVec3::Y };

    // Shared core (WI 737): gravity + drag + distributed righting buoyancy + free-surface damping.
    let core = core_active_wrench(
        craft,
        com,
        body,
        &p.medium,
        p.surface_radius,
        p.mu,
        p.drag_area,
        p.drag_coefficient,
        enclosed,
    );
    let sample = core.sample;

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

    // Water-entry slam (WI 700): a transient impact while the craft straddles the surface,
    // added to the central forces (same gate as the ballistic descent). The fraction is over the
    // total displaceable volume (solid + enclosed, WI 711).
    let displaceable = craft.occupied_volume() + enclosed.len() as f64 * craft.cell_volume();
    let submerged_fraction = if displaceable > 0.0 {
        core.buoyancy.submerged_volume / displaceable
    } else {
        0.0
    };
    let water_density = p.medium.sample_altitude(-1.0).density;
    let slam = entry_impact_force(
        water_density,
        body.velocity,
        DVec3::ZERO, // calm water: still surface (WI 705 forward hook #2)
        up,
        p.drag_area,
        submerged_fraction,
        p.slam_coefficient,
    );

    // Compose: core + aero (lift/wave force, restoring/damping moment) + slam + the external
    // (marine/ballast/open-cavity) wrench.
    let w = core.wrench
        + Wrench::from((lift + wave + slam, restoring + damping))
        + Wrench::from(external);
    body.integrate_wrench(w.force, w.torque, dt);
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
            // Atmospheric entry forces a drop out of warp (the warp filter): reset
            // to real time so the active descent is not handed a craft deep inside
            // dense air, then switch to the active gear.
            if clock.warp != 1.0 {
                writer.write(Command::SetWarp(1.0));
            }
            writer.write(Command::SetGear(GearKind::Active));
        }
    }
}

/// The craft's physical descent description, carried on the entity **across both
/// gears** so the active-gear descent driver can advance it after a wake (WI 527).
/// Holds the voxel craft, its (constant body-frame) centre of mass, and the glide
/// parameters precomputed once.
#[derive(Component, Clone)]
pub struct DivingCraft {
    /// The voxel craft (drag/buoyancy/lift geometry source).
    pub craft: VoxelCraft,
    /// Centre of mass, body frame (metres).
    pub com: DVec3,
    /// Precomputed gliding-descent parameters.
    pub glide: GlideParams,
    /// Cached enclosed airtight-compartment cells (WI 711) — the hull's enclosed air, displaced for
    /// buoyancy alongside the solid voxels so a hollow hull floats. Computed once (geometry is fixed
    /// across a descent); empty for a solid craft.
    pub enclosed: Vec<IVec3>,
    /// Cached **open** cavity (WI 713) — the interior an *un-sealed* (open-top) hull holds out from the
    /// water up to its rim. Displaced (via [`open_cavity_load`]) until the rim submerges, then swamps.
    /// Empty for a sealed or solid hull. Computed once; geometry is fixed across a descent.
    pub open: OpenCavity,
}

impl DivingCraft {
    /// Build a diving craft, computing and caching its enclosed-compartment cells (WI 711) and open
    /// cavity (WI 713) from the voxel geometry. Prefer this over a struct literal so the caches are
    /// never forgotten.
    pub fn new(craft: VoxelCraft, com: DVec3, glide: GlideParams) -> Self {
        let enclosed = enclosed_cells(&craft);
        let open = open_cavity(&craft);
        Self {
            craft,
            com,
            glide,
            enclosed,
            open,
        }
    }
}

/// The convective heat-scale for the `-- dive` re-entry. **It is now `1.0` — the
/// fully-physical model, no balance fudge.** The journey: WI 691's tame lob needed a
/// 250× scalar to show heat; WI 693 (a genuine ~7 km/s orbital entry, heating ∝ v³)
/// dropped it to ~4; WI 692 (a realistic thin skin, depth from thermal diffusivity)
/// and WI 688 (the ablative nose that survives the thin skin by ablating) retired the
/// rest. Heating is the physical `√ρ·v³`. The calibration test brackets the working
/// window (nose ablates and survives, hull < `max_temp`, ≈[0.55, >1.5]); 1.0 sits
/// comfortably inside. Kept as a knob so a scenario can still dial difficulty.
pub const DIVE_HEAT_SCALE: f64 = 1.0;

/// Re-entry **thermal state** carried on a diving craft (WI 691). When present on
/// an entity alongside [`DivingCraft`], [`advance_descent`] steps the two-node
/// thermal model ([`crate::thermal`]) each sub-step using the same medium sample and
/// body motion the descent uses — so a craft heats through re-entry. The
/// `heat_scale` is a **scenario balance scalar** over the physical heating shape
/// (the `-- dive` uses it to make its tame entry visibly consequential without
/// changing the physical law). Opt-in: a diving craft without this component simply
/// does not heat.
#[derive(Component, Clone)]
pub struct CraftThermal {
    /// The per-voxel two-node thermal state.
    pub state: crate::thermal::ThermalState,
    /// Radiative-sink (environment) temperature, K.
    pub env_temp: f64,
    /// Convective-flux balance scalar (1.0 = pure physical).
    pub heat_scale: f64,
}

impl CraftThermal {
    /// A thermal state initialised to `ambient`, radiating to `env_temp`, with the
    /// given convective `heat_scale`.
    pub fn new(craft: &VoxelCraft, ambient: f64, env_temp: f64, heat_scale: f64) -> Self {
        Self {
            state: crate::thermal::ThermalState::new(craft, ambient),
            env_temp,
            heat_scale,
        }
    }

    /// The hottest skin temperature (K), whether any voxel has reached its material
    /// limit, and the remaining ablative-shield fraction (`None` if no ablator) — the
    /// re-entry gauges for the HUD/telemetry (WI 688).
    pub fn readout(&self, craft: &VoxelCraft) -> (f64, bool, Option<f64>) {
        (
            self.state.max_skin_temp(),
            self.state.any_over_limit(craft),
            self.state.ablator_fraction_remaining(craft),
        )
    }
}

/// Drives the **active gear's aero/descent forces** (WI 527). Each frame it
/// sub-steps [`glide_step`] on every active craft carrying a [`DivingCraft`], so a
/// craft woken into the active gear inside a fluid experiences gravity + drag +
/// buoyancy + lift + the pitching moment **in the shared schedule** — the dive runs
/// on the pipeline instead of a scene hand-driving the integrator. Compose with
/// `HandoffPlugin`/`DiveTriggerPlugin`, and use it **instead of**
/// [`crate::active::ActivePlugin`] for diving craft, since `glide_step` already
/// includes gravity (running both would double-integrate). Fixed sub-step with a
/// per-frame cap (the active-vehicle stability budget); a bounded accumulator
/// avoids a spiral of death under load.
pub struct DescentPlugin {
    /// Fixed integration sub-step, seconds.
    pub substep_dt: f64,
    /// Maximum sub-steps integrated per frame.
    pub max_substeps: u32,
}

#[derive(Resource)]
struct DescentSubstep {
    dt: f64,
    max: u32,
    accumulator: f64,
}

impl Plugin for DescentPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DescentSubstep {
            dt: self.substep_dt,
            max: self.max_substeps,
            accumulator: 0.0,
        })
        .add_systems(Update, advance_descent);
    }
}

#[allow(clippy::type_complexity)]
fn advance_descent(
    time: Res<Time>,
    clock: Res<SimClock>,
    mut sub: ResMut<DescentSubstep>,
    mut bodies: Query<(
        &mut crate::active::ActiveBody,
        &DivingCraft,
        Option<&mut CraftThermal>,
        Option<&mut crate::marine::MarinePropulsion>,
        Option<&mut crate::ballast::Ballast>,
        Option<&crate::marine::Rudder>,
    )>,
) {
    if clock.paused {
        return;
    }
    sub.accumulator += time.delta_secs_f64() * clock.warp;
    // Bound the backlog to one frame's worth of sub-steps (no spiral of death).
    let cap = sub.max as f64 * sub.dt;
    if sub.accumulator > cap {
        sub.accumulator = cap;
    }
    let dt = sub.dt;
    let mut n = 0;
    while sub.accumulator >= dt && n < sub.max {
        for (mut body, dc, thermal, marine, ballast, rudder) in &mut bodies {
            // Marine propulsion (WI 708): a screw's thrust wrench, scaled by the medium
            // density at each thruster, drawn from its tanks — the external force into
            // `glide_step`. Absent component ⇒ zero (the dive path is unaffected).
            let mut external = if let Some(mut mp) = marine {
                mp.thrust_step(
                    &dc.glide.descent.medium,
                    dc.glide.descent.surface_radius,
                    body.position,
                    body.orientation,
                    dc.com,
                    dt,
                )
            } else {
                (DVec3::ZERO, DVec3::ZERO)
            };
            // Rudder (WI 725): a hydrodynamic steering wrench from the water flow — speed-
            // dependent yaw, summed into the same external force. Works coasting/unpowered.
            if let Some(r) = rudder {
                let (rf, rt) = r.wrench(
                    &dc.glide.descent.medium,
                    dc.glide.descent.surface_radius,
                    body.position,
                    body.orientation,
                    body.velocity,
                    dc.com,
                );
                external.0 += rf;
                external.1 += rt;
            }
            // Ballast (WI 709): step fill/blow, fold the tank water mass + CoM into the
            // body on the floodwater precedent, and use the **wet** CoM for the descent
            // (so net buoyancy flips sign as it floods and it sits lower / trims). The
            // ballast water is the liquid the tanks sit in (sampled at depth, so it is
            // inert above an ocean / in vacuum). Absent component ⇒ the dry mass/CoM.
            let com = if let Some(mut b) = ballast {
                b.step(dt);
                let water_density = dc.glide.descent.medium.sample_altitude(-1.0).density;
                let wet = b.wet_mass(dc.com, water_density);
                body.mass = wet.mass;
                wet.center_of_mass
            } else {
                dc.com
            };
            // Open-boat displacement (WI 713): an un-sealed hull's held-out volume buoys it until the
            // rim submerges, then swamps. Summed into the external wrench (about the same `com` the
            // glide uses). Empty cavity (sealed/solid hull) ⇒ zero, so sealed buoyancy is unchanged.
            if !dc.open.cells.is_empty() {
                let p = &dc.glide.descent;
                let r = body.position.length();
                let g_local = if r > 0.0 { p.mu / (r * r) } else { 0.0 };
                let density = p.medium.sample_altitude(r - p.surface_radius).density;
                let open = open_cavity_load(
                    &dc.craft,
                    com,
                    body.position,
                    body.orientation,
                    p.surface_radius,
                    0.0,
                    density,
                    g_local,
                    &dc.open,
                );
                external.0 += open.force;
                external.1 += open.torque;
            }
            let sample = glide_step(
                &mut body,
                &dc.craft,
                com,
                &dc.enclosed,
                &dc.glide,
                external,
                dt,
            );
            // A craft carrying thermal state heats from the same medium sample and
            // motion the descent just used (WI 691) — a passive overlay (no force
            // feedback this slice), so the trajectory is unchanged.
            if let Some(mut th) = thermal {
                let env = th.env_temp;
                let scale = th.heat_scale;
                th.state.step_scaled(
                    &dc.craft,
                    &sample,
                    body.velocity,
                    body.orientation,
                    env,
                    dt,
                    scale,
                );
            }
        }
        sub.accumulator -= dt;
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active::ActiveBody;

    #[test]
    fn wrench_composes_additively() {
        let a = Wrench::from((DVec3::X, DVec3::Y));
        let b = Wrench::from((DVec3::new(0.0, 2.0, 0.0), DVec3::new(0.0, 0.0, 3.0)));
        let mut s = Wrench::ZERO;
        s += a;
        s += b;
        assert_eq!(s, a + b);
        assert_eq!((a + b).force, DVec3::new(1.0, 2.0, 0.0));
        assert_eq!((a + b).torque, DVec3::new(0.0, 1.0, 3.0));
        // Order-independent (additive contributors).
        assert_eq!(a + b, b + a);
    }

    #[test]
    fn core_active_wrench_is_inert_in_vacuum() {
        // A craft in vacuum, far from any body: only gravity contributes; drag/buoyancy/damping are
        // zero (no medium, nothing submerged), so the core wrench is pure central gravity, no torque.
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        let mp = craft.mass_properties().unwrap();
        let body = ActiveBody::from_mass_properties(
            DVec3::new(0.0, CentralBody::EARTHLIKE.radius + 1_000_000.0, 0.0),
            DVec3::ZERO,
            &mp,
        );
        let core = core_active_wrench(
            &craft,
            mp.center_of_mass,
            &body,
            &FluidMedium::EARTHLIKE,
            CentralBody::EARTHLIKE.radius,
            CentralBody::EARTHLIKE.mu,
            1.0,
            1.0,
            &[],
        );
        assert!(core.wrench.force.length() > 0.0, "gravity pulls it down");
        assert!(
            core.wrench.force.dot(body.position.normalize()) < 0.0,
            "gravity is toward the body (−up)"
        );
        assert!(core.wrench.torque.length() < 1e-12, "no torque in vacuum");
        assert_eq!(core.buoyancy.submerged_volume, 0.0, "nothing submerged");
    }
    use crate::command::FlightControlPlugin;
    use crate::fluid::{FluidMedium, MediumKind};
    use crate::handoff::{GearState, HandoffPlugin};
    use crate::sim::{CentralBody, OrbitPlugin};
    use crate::voxel::{Material, Voxel, VoxelCraft};
    use glam::{DVec2, IVec3};

    // Earth-like SI constants for the dive — the shared canonical body (WI 527).
    const SURFACE_R: f64 = CentralBody::EARTHLIKE.radius;
    const MU: f64 = CentralBody::EARTHLIKE.mu;

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
            slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
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

    // --- WI 700: water-entry slamming load ---

    #[test]
    fn entry_slam_fires_only_while_straddling_and_descending() {
        let up = DVec3::Y;
        let down = DVec3::new(0.0, -100.0, 0.0); // descending into the water
        let area = 4.0;
        let rho = 1_025.0;
        let cs = DEFAULT_SLAM_COEFFICIENT;

        // Straddling (half submerged) and descending: a real upward (decelerating) slam.
        let slam = entry_impact_force(rho, down, DVec3::ZERO, up, area, 0.5, cs);
        assert!(slam.y > 0.0, "slam opposes the downward entry: {slam:?}");
        assert!(slam.dot(down) < 0.0, "slam decelerates the entry");

        // No ocean (zero water density) → no slam.
        assert_eq!(
            entry_impact_force(0.0, down, DVec3::ZERO, up, area, 0.5, cs),
            DVec3::ZERO
        );
        // Fully out of the water (fraction 0) and fully submerged (fraction 1) → no slam.
        assert_eq!(
            entry_impact_force(rho, down, DVec3::ZERO, up, area, 0.0, cs),
            DVec3::ZERO
        );
        assert_eq!(
            entry_impact_force(rho, down, DVec3::ZERO, up, area, 1.0, cs),
            DVec3::ZERO
        );
        // Rising out of the water (velocity along +up) → no slam.
        let rising = DVec3::new(0.0, 100.0, 0.0);
        assert_eq!(
            entry_impact_force(rho, rising, DVec3::ZERO, up, area, 0.5, cs),
            DVec3::ZERO
        );
        // Skimming horizontally (no inward closing speed) → no slam.
        let skim = DVec3::new(100.0, 0.0, 0.0);
        assert_eq!(
            entry_impact_force(rho, skim, DVec3::ZERO, up, area, 0.5, cs),
            DVec3::ZERO
        );
        // Slam disabled (coefficient 0) → no slam.
        assert_eq!(
            entry_impact_force(rho, down, DVec3::ZERO, up, area, 0.5, 0.0),
            DVec3::ZERO
        );
    }

    #[test]
    fn entry_slam_grows_with_speed_squared_and_decays_with_submersion() {
        let up = DVec3::Y;
        let area = 4.0;
        let rho = 1_025.0;
        let cs = DEFAULT_SLAM_COEFFICIENT;
        let slam = |speed: f64, frac: f64| {
            entry_impact_force(
                rho,
                DVec3::new(0.0, -speed, 0.0),
                DVec3::ZERO,
                up,
                area,
                frac,
                cs,
            )
            .length()
        };
        // Doubling the closing speed quadruples the slam (v² scaling).
        let slow = slam(50.0, 0.5);
        let fast = slam(100.0, 0.5);
        assert!(
            (fast - 4.0 * slow).abs() < 1e-6 * fast.max(1.0),
            "v² scaling: {slow} vs {fast}"
        );
        // The slam decays as the craft submerges (peak near first contact).
        assert!(
            slam(100.0, 0.1) > slam(100.0, 0.9),
            "decays toward full submersion"
        );
    }

    #[test]
    fn fast_entry_decelerates_more_than_steady_drag_alone() {
        // Two identical craft step through the straddle window: one with the slam, one with
        // steady drag only. The slammed one loses more downward speed, and both stay finite.
        let craft = test_craft();
        let com = craft.mass_properties().unwrap().center_of_mass;
        let mp = craft.mass_properties().unwrap();
        let base = earthlike_params();
        let with_slam = DescentParams {
            slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
            ..base
        };
        let no_slam = DescentParams {
            slam_coefficient: 0.0,
            ..base
        };

        // Start with the craft half-submerged at the surface, descending fast, and take a
        // single step through the straddle window (before it fully submerges and water drag
        // dominates both equally).
        let start = DVec3::new(0.0, SURFACE_R, 0.0);
        let v0 = DVec3::new(0.0, -300.0, 0.0);
        let run = |params: &DescentParams| {
            let mut body = ActiveBody::new(start, v0, mp.mass, mp.inertia);
            descent_step(&mut body, &craft, com, &[], params, 1e-3);
            body
        };
        let slammed = run(&with_slam);
        let plain = run(&no_slam);
        assert!(
            slammed.velocity.is_finite() && plain.velocity.is_finite(),
            "both stay finite"
        );
        // Downward speed is the magnitude of the (negative-y) velocity; the slam removes more.
        assert!(
            slammed.velocity.y > plain.velocity.y,
            "the slam decelerates the entry more than steady drag alone: slam vy {} vs plain vy {}",
            slammed.velocity.y,
            plain.velocity.y
        );
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
            let sample = descent_step(&mut body, &craft, com, &[], &params, dt);
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

    /// The `-- dive` craft: a 3×3×4 composite hull with a single **ablative** nose tip
    /// at the windward front (cell (1,1,4), `Material::ABLATOR`, WI 688) — replicated
    /// from `dive_scene::dive_craft` for the calibration.
    fn dive_calibration_craft() -> VoxelCraft {
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
        c.voxels.push(Voxel {
            cell: IVec3::new(1, 1, 4),
            material: Material::ABLATOR,
        });
        c
    }

    // --- WI 691/693/688: dive thermal calibration (full orbital re-entry chain) ---

    /// The whole `-- dive` chain, headless: the dive craft on the **orbital** re-entry
    /// orbit coasts on rails, auto-drops to active at the entry interface, and descends
    /// through the atmosphere into the ocean while the `DescentPlugin` steps its
    /// `CraftThermal`. Asserts the invariants together: the entry is genuinely orbital
    /// (≥ 6 km/s, WI 693), it reaches the ocean, and under `DIVE_HEAT_SCALE` the
    /// **ablative nose ablates and survives** (WI 688) while the composite hull survives.
    #[test]
    fn dive_orbital_reentry_ablative_nose_survives_and_hull_survives() {
        use crate::active::Gravity;
        use crate::sim::CentralBody;
        use std::time::Duration;

        let body = CentralBody::EARTHLIKE;
        // The dive's orbital entry: near-circular at 120 km, periapsis in the atmosphere
        // (matches dive_scene's ENTRY_SPEED = 7000 m/s).
        let orbit = Orbit::from_state(
            body.mu,
            DVec2::new(body.radius + 120_000.0, 0.0),
            DVec2::new(0.0, 7_000.0),
            0.0,
        )
        .unwrap();
        assert!(
            orbit.periapsis_radius() < body.radius,
            "must be a re-entry trajectory"
        );

        let craft = dive_calibration_craft();
        let mp = craft.mass_properties().unwrap();
        let descent = DescentParams {
            medium: FluidMedium::EARTHLIKE,
            mu: body.mu,
            surface_radius: body.radius,
            drag_area: max_cross_section(&craft),
            drag_coefficient: 1.0,
            slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
        };
        let glide = GlideParams::for_craft(descent, &craft, Axis::Z);

        let mut app = App::new();
        app.insert_resource(Time::<()>::default());
        app.insert_resource(Gravity { mu: body.mu });
        app.add_plugins(OrbitPlugin {
            central_body: body,
            initial_orbit: orbit,
        });
        app.add_plugins(FlightControlPlugin);
        app.add_plugins(HandoffPlugin);
        app.add_plugins(DiveTriggerPlugin {
            interface: EntryInterface {
                surface_radius: body.radius,
                altitude: 100_000.0,
            },
        });
        app.add_plugins(DescentPlugin {
            substep_dt: 0.002,
            max_substeps: 4_000,
        });
        {
            let mut q = app.world_mut().query_filtered::<Entity, With<Craft>>();
            let e = q.single(app.world()).unwrap();
            app.world_mut().entity_mut(e).insert((
                GearState::new(mp.mass, mp.inertia),
                DivingCraft::new(craft.clone(), mp.center_of_mass, glide),
                CraftThermal::new(&craft, 250.0, 250.0, DIVE_HEAT_SCALE),
            ));
        }
        app.world_mut().resource_mut::<SimClock>().warp = 50.0;

        let nose = IVec3::new(1, 1, 4);
        let mut entry_speed = 0.0_f64;
        let mut reached_ocean = false;
        let mut peak_nose = 0.0_f64;
        let mut peak_hull = 0.0_f64;
        let mut ablator_frac = 1.0_f64;
        for _ in 0..20_000 {
            app.world_mut()
                .resource_mut::<Time<()>>()
                .advance_by(Duration::from_secs_f64(0.25));
            app.update();

            let mut q = app.world_mut().query::<(&ActiveBody, &CraftThermal)>();
            if let Ok((b, th)) = q.single(app.world()) {
                assert!(
                    b.position.is_finite() && b.velocity.is_finite(),
                    "active descent state must stay finite"
                );
                if entry_speed == 0.0 {
                    entry_speed = b.velocity.length();
                }
                peak_nose = peak_nose.max(th.state.skin(nose).unwrap());
                for v in &craft.voxels {
                    if v.cell != nose {
                        peak_hull = peak_hull.max(th.state.skin(v.cell).unwrap());
                    }
                }
                ablator_frac = th.state.ablator_fraction_remaining(&craft).unwrap_or(1.0);
                let altitude = b.position.length() - body.radius;
                if descent.medium.sample_altitude(altitude).medium == MediumKind::Liquid {
                    reached_ocean = true;
                    break;
                }
            }
        }

        // I2: a genuinely orbital entry speed (not the old ~1.5 km/s lob).
        assert!(
            entry_speed >= 6_000.0,
            "entry must be orbital: {entry_speed}"
        );
        // I1: it still descends all the way to the ocean.
        assert!(reached_ocean, "must reach the ocean");
        // WI 688: the ablative nose gets hot enough to ablate but **survives** (held
        // below its bare-char failure temperature), and the composite hull survives.
        assert!(
            peak_nose >= Material::ABLATOR.thermal.ablation_temp,
            "nose should reach the ablation set-point: peak_nose={peak_nose}"
        );
        assert!(
            peak_nose < Material::ABLATOR.thermal.max_temp,
            "ablative nose should survive (not reach bare-char failure): peak_nose={peak_nose}"
        );
        assert!(
            peak_hull < Material::COMPOSITE.thermal.max_temp,
            "composite hull should survive: peak_hull={peak_hull}"
        );
        // It survived by ablating: ablator consumed (the shield worked) but not spent.
        assert!(
            ablator_frac > 0.0 && ablator_frac < 1.0,
            "ablator partially consumed: {ablator_frac}"
        );
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

        let s_above = descent_step(&mut above, &craft, com, &[], &params, 1e-3);
        let s_below = descent_step(&mut below, &craft, com, &[], &params, 1e-3);
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

    /// WI 527 — the headline integration: **one craft**, on an SI Kepler orbit,
    /// coasts on rails, auto-drops to the active gear at the entry interface, and is
    /// driven by the active-gear descent forces (`DescentPlugin`) down through the
    /// atmosphere into the ocean — all in one schedule, one consistent SI system.
    #[test]
    fn one_craft_orbits_hands_off_and_descends_in_si() {
        use crate::active::Gravity;
        use crate::handoff::LastHandoff;
        use crate::sim::CentralBody;
        use std::time::Duration;

        let body = CentralBody::EARTHLIKE;
        // A steep, low-energy plunge from 120 km: a bound Kepler orbit whose
        // periapsis is deep below the surface (a re-entry trajectory) and whose
        // entry speed stays tame (~1–2 km/s), not full orbital velocity.
        let orbit = Orbit::from_state(
            body.mu,
            DVec2::new(body.radius + 120_000.0, 0.0),
            DVec2::new(0.0, 600.0),
            0.0,
        )
        .unwrap();
        assert!(
            orbit.periapsis_radius() < body.radius,
            "must be a re-entry trajectory"
        );

        let craft = test_craft(); // a dense composite block (sinks)
        let mp = craft.mass_properties().unwrap();
        let descent = DescentParams {
            medium: FluidMedium::EARTHLIKE,
            mu: body.mu,
            surface_radius: body.radius,
            drag_area: max_cross_section(&craft),
            drag_coefficient: 1.0,
            slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
        };
        let glide = GlideParams::for_craft(descent, &craft, Axis::Z);

        let mut app = App::new();
        app.insert_resource(Time::<()>::default()); // drive time manually (deterministic)
        app.insert_resource(Gravity { mu: body.mu }); // handoff sleep path reads it
        app.add_plugins(OrbitPlugin {
            central_body: body,
            initial_orbit: orbit,
        });
        app.add_plugins(FlightControlPlugin);
        app.add_plugins(HandoffPlugin);
        app.add_plugins(DiveTriggerPlugin {
            interface: EntryInterface {
                surface_radius: body.radius,
                altitude: 100_000.0,
            },
        });
        app.add_plugins(DescentPlugin {
            substep_dt: 0.002,
            max_substeps: 4_000,
        });

        // Put the real craft on the auto-spawned entity: a real gear (mass/inertia)
        // and the diving description the descent driver needs after the wake.
        {
            let mut q = app.world_mut().query_filtered::<Entity, With<Craft>>();
            let e = q.single(app.world()).unwrap();
            app.world_mut().entity_mut(e).insert((
                GearState::new(mp.mass, mp.inertia),
                DivingCraft::new(craft.clone(), mp.center_of_mass, glide),
            ));
        }
        // Coast under warp; the entry trigger drops warp to 1 and switches gears.
        app.world_mut().resource_mut::<SimClock>().warp = 50.0;

        let mut transitioned = false;
        let mut reached_ocean = false;
        for _ in 0..6_000 {
            app.world_mut()
                .resource_mut::<Time<()>>()
                .advance_by(Duration::from_secs_f64(0.5));
            app.update();

            let mut q = app
                .world_mut()
                .query::<(Option<&Craft>, Option<&ActiveBody>)>();
            let (on_rails, active) = q.single(app.world()).unwrap();
            if let Some(active_body) = active {
                transitioned = true;
                // Once active, the state must stay finite all the way down.
                let body_state = *active_body;
                assert!(
                    body_state.position.is_finite() && body_state.velocity.is_finite(),
                    "active descent state must stay finite"
                );
                let altitude = body_state.position.length() - body.radius;
                if descent.medium.sample_altitude(altitude).medium == MediumKind::Liquid {
                    reached_ocean = true;
                    break;
                }
            }
            let _ = on_rails;
        }

        assert!(
            transitioned,
            "the craft must auto-drop from rails to active in SI"
        );
        assert!(
            reached_ocean,
            "the craft must descend through the atmosphere into the ocean in SI"
        );
        // The SI hand-off was clean (sub-metre / sub-m/s injected jump).
        let last = app.world().resource::<LastHandoff>();
        assert!(
            last.0.is_some_and(|h| h.magnitude() < 1.0),
            "SI hand-off must be ~clean: {:?}",
            last.0
        );
        // Warp was dropped to real time at entry (the warp filter).
        assert_eq!(app.world().resource::<SimClock>().warp, 1.0);
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
            slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
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
            glide_step(
                &mut body,
                &craft,
                com,
                &[],
                &glide,
                (DVec3::ZERO, DVec3::ZERO),
                dt,
            );
            descent_step(&mut ballistic, &craft, com, &[], &descent, dt);
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
            glide_step(
                &mut body,
                &craft,
                com,
                &[],
                &glide,
                (DVec3::ZERO, DVec3::ZERO),
                dt,
            );
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
        let s = glide_step(
            &mut g_air,
            &craft,
            com,
            &[],
            &glide,
            (DVec3::ZERO, DVec3::ZERO),
            0.004,
        );
        descent_step(&mut d_air, &craft, com, &[], &descent, 0.004);
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
        let sw = glide_step(
            &mut g_w,
            &craft,
            com,
            &[],
            &glide,
            (DVec3::ZERO, DVec3::ZERO),
            0.004,
        );
        descent_step(&mut d_w, &craft, com, &[], &descent, 0.004);
        assert_eq!(sw.medium, MediumKind::Liquid);
        assert!(
            (g_w.velocity - d_w.velocity).length() < 1e-9,
            "ocean: no wave drag and no lift → identical to ballistic"
        );
    }

    // --- WI 705: righting buoyancy + surface hydrodynamics ---

    /// A light, beam-spread raft: wide in X (the beam), one cell tall, a few cells long in Z.
    /// Editor-scale 0.5 m cells. `width` should be odd so it is symmetric about the centreline.
    fn raft_hull(width: i32, length: i32) -> VoxelCraft {
        let mut c = VoxelCraft::new(0.5);
        for x in -(width / 2)..=(width / 2) {
            for z in 0..length {
                c.voxels.push(Voxel {
                    cell: IVec3::new(x, 0, z),
                    material: Material::ALUMINIUM,
                });
            }
        }
        c
    }

    /// Behaviour 1: the per-cell submerged fraction is graded (C0) and monotone through the
    /// waterline, reducing to fully-dry / fully-wet / half at the band edges and centre.
    #[test]
    fn cell_submersion_is_graded_and_monotone_through_the_waterline() {
        let cs = 0.5;
        let mut last = -1.0;
        for k in 0..=40 {
            // Sweep the cell centre downward: +cs above the surface (dry) → −cs below (wet).
            let height = cs - (2.0 * cs) * (k as f64 / 40.0);
            let pos = DVec3::new(0.0, SURFACE_R + height, 0.0);
            let f = cell_submerged_fraction(pos, cs, SURFACE_R, 0.0);
            assert!((0.0..=1.0).contains(&f), "fraction in [0,1]: {f}");
            assert!(f >= last - 1e-12, "monotone non-decreasing as it sinks");
            last = f;
        }
        // Band edges and centre: a full half-cell above ⇒ dry, below ⇒ wet, on the surface ⇒ half.
        assert_eq!(
            cell_submerged_fraction(DVec3::new(0.0, SURFACE_R - cs, 0.0), cs, SURFACE_R, 0.0),
            1.0
        );
        assert_eq!(
            cell_submerged_fraction(DVec3::new(0.0, SURFACE_R + cs, 0.0), cs, SURFACE_R, 0.0),
            0.0
        );
        assert!(
            (cell_submerged_fraction(DVec3::new(0.0, SURFACE_R, 0.0), cs, SURFACE_R, 0.0) - 0.5)
                .abs()
                < 1e-9
        );
    }

    /// Behaviours 2, 3, 8: the buoyant force equals the displaced-medium weight, and the wrench
    /// vanishes both fully out of the water and in vacuum.
    #[test]
    fn buoyant_force_equals_displaced_weight_and_vanishes_out_of_water_and_in_vacuum() {
        let craft = raft_hull(7, 3);
        let com = craft.mass_properties().unwrap().center_of_mass;
        let rho = 1_025.0;
        let g = 9.81;
        // Deep: every cell submerged ⇒ force = ρ·g·occupied_volume, directed up; draft positive.
        let deep = buoyancy_wrench(
            &craft,
            com,
            DVec3::new(0.0, SURFACE_R - 100.0, 0.0),
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            rho,
            g,
            &[],
        );
        let full = rho * g * craft.occupied_volume();
        assert!(
            (deep.force.length() - full).abs() < 1e-6 * full,
            "force = displaced weight"
        );
        assert!(deep.force.y > 0.0, "buoyancy points up");
        assert!(deep.draft > 0.0, "submerged ⇒ positive draft");
        // Fully out of the water ⇒ zero wrench, zero displaced volume.
        let dry = buoyancy_wrench(
            &craft,
            com,
            DVec3::new(0.0, SURFACE_R + 100.0, 0.0),
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            rho,
            g,
            &[],
        );
        assert_eq!(dry.force, DVec3::ZERO);
        assert_eq!(dry.torque, DVec3::ZERO);
        assert_eq!(dry.submerged_volume, 0.0);
        assert_eq!(dry.draft, 0.0);
        // Vacuum (zero density) ⇒ zero wrench even when geometrically submerged.
        let vac = buoyancy_wrench(
            &craft,
            com,
            DVec3::new(0.0, SURFACE_R - 100.0, 0.0),
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            0.0,
            g,
            &[],
        );
        assert_eq!(vac.force, DVec3::ZERO);
        assert_eq!(vac.torque, DVec3::ZERO);
    }

    /// Behaviours 4, 5 (the load-bearing property): a beam-spread hull produces a restoring roll
    /// moment opposing an imposed heel, while a degenerate hull whose cells lie on the roll axis
    /// has ~zero roll stiffness — righting EMERGES from beam spread, it is not a tuned term.
    #[test]
    fn righting_moment_opposes_heel_and_requires_beam_spread() {
        let theta = 0.15_f64;
        let heel = DQuat::from_rotation_z(theta);
        let rho = 1_025.0;
        let g = 9.81;
        let pos = DVec3::new(0.0, SURFACE_R, 0.0); // craft centre at the waterline

        // Beam-spread raft: a restoring moment (torque.z opposite in sign to the heel θ).
        let raft = raft_hull(7, 3);
        let com = raft.mass_properties().unwrap().center_of_mass;
        let load = buoyancy_wrench(&raft, com, pos, heel, SURFACE_R, 0.0, rho, g, &[]);
        assert!(
            load.torque.z * theta < 0.0,
            "beam hull rights (restoring): {:?}",
            load.torque
        );
        let beam_restoring = load.torque.z.abs();

        // Larger heel ⇒ larger restoring over the small (linear GM) range.
        let load2 = buoyancy_wrench(
            &raft,
            com,
            pos,
            DQuat::from_rotation_z(2.0 * theta),
            SURFACE_R,
            0.0,
            rho,
            g,
            &[],
        );
        assert!(
            load2.torque.z.abs() > beam_restoring,
            "restoring grows with heel"
        );

        // Degenerate hull on the roll axis (1×1×N along Z): heeling about Z leaves the cells on the
        // axis, so the roll moment is ~exactly zero — correct draft, no roll stiffness.
        let mut line = VoxelCraft::new(0.5);
        for z in 0..3 {
            line.voxels.push(Voxel {
                cell: IVec3::new(0, 0, z),
                material: Material::ALUMINIUM,
            });
        }
        let lcom = line.mass_properties().unwrap().center_of_mass;
        let lload = buoyancy_wrench(&line, lcom, pos, heel, SURFACE_R, 0.0, rho, g, &[]);
        assert!(
            lload.torque.z.abs() < 1e-6,
            "centreline-on-roll-axis hull has ~zero roll stiffness: {:?}",
            lload.torque
        );
        assert!(
            beam_restoring > 1e3 * lload.torque.z.abs().max(1e-12),
            "righting comes from beam spread, not incidentally"
        );
    }

    /// Behaviours 6, 7: a raft released at a heel near the surface RIGHTS itself and SETTLES — the
    /// heel decays, the late-time oscillation is small (no ring/limit-cycle), the state stays finite
    /// and bounded near the waterline. Light editor-scale fixture.
    #[test]
    fn a_heeled_raft_rights_itself_and_settles() {
        let craft = raft_hull(7, 3);
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let params = DescentParams {
            medium: FluidMedium::EARTHLIKE,
            mu: MU,
            surface_radius: SURFACE_R,
            drag_area: max_cross_section(&craft),
            drag_coefficient: 1.0,
            slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
        };
        // Float ~half-submerged: mass = half the fully-submerged displaced water mass.
        let mass = 0.5 * 1_025.0 * craft.occupied_volume();
        let inertia = mp.inertia * (mass / mp.mass);
        let mut body = ActiveBody::new(
            DVec3::new(0.0, SURFACE_R - 0.1, 0.0),
            DVec3::ZERO,
            mass,
            inertia,
        );
        body.orientation = DQuat::from_rotation_z(0.3);
        let initial_heel = heel_angle(body.orientation, DVec3::Y);

        let dt = 0.005;
        let mut max_late_heel = 0.0_f64;
        for i in 0..20_000 {
            descent_step(&mut body, &craft, com, &[], &params, dt);
            assert!(
                body.position.is_finite() && body.velocity.is_finite(),
                "state stayed finite at step {i}"
            );
            if i >= 16_000 {
                let up = body.position.normalize_or_zero();
                max_late_heel = max_late_heel.max(heel_angle(body.orientation, up));
            }
        }
        let up = body.position.normalize_or_zero();
        let final_heel = heel_angle(body.orientation, up);
        assert!(
            final_heel < 0.5 * initial_heel,
            "raft rights itself: {initial_heel:.3} → {final_heel:.3} rad"
        );
        assert!(
            max_late_heel < 0.15,
            "settles (small residual heel, no ring): {max_late_heel:.3} rad"
        );
        let altitude = body.position.length() - SURFACE_R;
        assert!(
            altitude.abs() < 5.0,
            "settles near the waterline (bounded): {altitude:.2} m"
        );
    }

    /// WI 730: a floating hull given a yaw rate with no steering input **bleeds it off** instead of
    /// spinning in place. The up-component damping alone leaves yaw undamped (it sweeps cells
    /// horizontally); the horizontal rotational term must settle it. Surge is left to the bulk drag,
    /// so this asserts the *rotational* decay specifically. Light editor-scale fixture.
    #[test]
    fn a_yawing_raft_stops_spinning() {
        let craft = raft_hull(7, 3);
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let params = DescentParams {
            medium: FluidMedium::EARTHLIKE,
            mu: MU,
            surface_radius: SURFACE_R,
            drag_area: max_cross_section(&craft),
            drag_coefficient: 1.0,
            slam_coefficient: DEFAULT_SLAM_COEFFICIENT,
        };
        let mass = 0.5 * 1_025.0 * craft.occupied_volume();
        let inertia = mp.inertia * (mass / mp.mass);
        // Yaw about local up (+Y here, since position is on the +Y axis).
        let mut body = ActiveBody::new(
            DVec3::new(0.0, SURFACE_R - 0.1, 0.0),
            DVec3::ZERO,
            mass,
            inertia,
        )
        .with_angular_velocity(DVec3::new(0.0, 1.0, 0.0));
        let up = body.position.normalize_or_zero();
        let yaw0 = body.angular_velocity().dot(up).abs();

        let dt = 0.005;
        for i in 0..20_000 {
            descent_step(&mut body, &craft, com, &[], &params, dt);
            assert!(
                body.angular_velocity().is_finite(),
                "angular velocity stayed finite at step {i}"
            );
        }
        let up = body.position.normalize_or_zero();
        let yaw1 = body.angular_velocity().dot(up).abs();
        assert!(
            yaw1 < 0.05 * yaw0,
            "yaw rate bleeds off (no continuous spin): {yaw0:.3} → {yaw1:.3} rad/s"
        );
    }

    /// WI 730: the yaw damping is **rotation-only** — it must not add per-cell surge drag that fights
    /// the thruster. A hull translating forward with no spin sees no extra horizontal damping force
    /// beyond the existing up-component term (which is zero for purely horizontal motion).
    #[test]
    fn yaw_damping_does_not_damp_surge() {
        let craft = raft_hull(7, 3);
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let body_position = DVec3::new(0.0, SURFACE_R - 0.1, 0.0);
        let up = body_position.normalize_or_zero();
        // Pure horizontal (surge) velocity, no rotation: tangent to the surface.
        let surge = up.cross(DVec3::Z).normalize() * 5.0;
        let (force, torque) = free_surface_damping(
            &craft,
            com,
            body_position,
            DQuat::IDENTITY,
            surge,
            DVec3::ZERO,
            SURFACE_R,
            0.0,
            FluidMedium::EARTHLIKE.sample_altitude(-1.0).density,
            FREE_SURFACE_DAMPING,
        );
        assert!(
            force.length() < 1e-6 && torque.length() < 1e-6,
            "no damping on pure horizontal surge: f={force:?} t={torque:?}"
        );
    }

    /// Forward hook #2 (behaviour 9): the entry slam's closing speed is the hull velocity relative
    /// to the water surface. With a still surface it is unchanged; a surface rising with the hull
    /// (no relative closing) produces no slam.
    #[test]
    fn entry_slam_uses_relative_closing_speed() {
        let up = DVec3::Y;
        let down = DVec3::new(0.0, -100.0, 0.0);
        let area = 4.0;
        let rho = 1_025.0;
        let cs = DEFAULT_SLAM_COEFFICIENT;
        // Still water ⇒ identical to the calm-water slam.
        let still = entry_impact_force(rho, down, DVec3::ZERO, up, area, 0.5, cs);
        assert!(still.y > 0.0);
        // A water surface descending with the hull at the same rate ⇒ zero relative closing ⇒ no slam.
        let following = entry_impact_force(rho, down, down, up, area, 0.5, cs);
        assert_eq!(following, DVec3::ZERO);
    }

    // --- WI 711: enclosed-volume buoyancy (hollow hulls float) ---

    /// A sealed hollow box: an `n×n×n` cube whose surface cells are solid and whose interior is
    /// empty (one sealed compartment). Editor-scale 0.5 m cells.
    fn sealed_box(n: i32, material: Material) -> VoxelCraft {
        let mut c = VoxelCraft::new(0.5);
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let on_surface =
                        x == 0 || x == n - 1 || y == 0 || y == n - 1 || z == 0 || z == n - 1;
                    if on_surface {
                        c.voxels.push(Voxel {
                            cell: IVec3::new(x, y, z),
                            material,
                        });
                    }
                }
            }
        }
        c
    }

    /// Behaviour 3: `enclosed_cells` is exactly the airtight-compartment interior — the sealed box's
    /// `(n−2)³` interior cells, and nothing for a solid block.
    #[test]
    fn enclosed_cells_are_the_sealed_interior() {
        let hull = sealed_box(5, Material::ALUMINIUM);
        assert_eq!(enclosed_cells(&hull).len(), 27, "interior 3×3×3 enclosed"); // (5−2)³
        let solid = test_craft(); // a dense 2×2×2 block — no sealed interior
        assert!(
            enclosed_cells(&solid).is_empty(),
            "a solid block encloses nothing"
        );
    }

    /// Behaviours 1, 4: the buoyant force counts the enclosed volume — a hull submerged with its
    /// enclosed cells displaces exactly the enclosed volume more than the shell alone, still zero in
    /// vacuum.
    #[test]
    fn buoyant_force_includes_enclosed_volume() {
        let hull = sealed_box(5, Material::ALUMINIUM);
        let enc = enclosed_cells(&hull);
        let com = hull.mass_properties().unwrap().center_of_mass;
        let pos = DVec3::new(0.0, SURFACE_R - 100.0, 0.0); // fully submerged
        let rho = 1_025.0;
        let g = 9.81;
        let shell = buoyancy_wrench(
            &hull,
            com,
            pos,
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            rho,
            g,
            &[],
        );
        let whole = buoyancy_wrench(
            &hull,
            com,
            pos,
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            rho,
            g,
            &enc,
        );
        assert!(
            whole.force.length() > shell.force.length(),
            "enclosed adds buoyancy"
        );
        let enclosed_vol = enc.len() as f64 * hull.cell_volume();
        assert!(
            (whole.submerged_volume - shell.submerged_volume - enclosed_vol).abs() < 1e-9,
            "extra displaced volume = the enclosed volume exactly"
        );
        // Vacuum still zero even with enclosed cells.
        let vac = buoyancy_wrench(
            &hull,
            com,
            pos,
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            0.0,
            g,
            &enc,
        );
        assert_eq!(vac.force, DVec3::ZERO);
    }

    /// Behaviour 2 (the headline): a hollow hull **floats only because of its enclosed volume** — the
    /// same hull + mass with the enclosed cells counted rises to the surface, but displacing the shell
    /// alone it sinks. Isolates the enclosed-volume effect (mass held identical).
    #[test]
    fn hollow_hull_floats_only_because_of_enclosed_volume() {
        let hull = sealed_box(5, Material::ALUMINIUM);
        let enc = enclosed_cells(&hull);
        let mp = hull.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let params = DescentParams {
            medium: FluidMedium::EARTHLIKE,
            mu: MU,
            surface_radius: SURFACE_R,
            drag_area: max_cross_section(&hull),
            drag_coefficient: 1.0,
            slam_coefficient: 0.0, // isolate buoyancy from the entry slam
        };
        let cv = hull.cell_volume();
        let shell_vol = hull.occupied_volume();
        let total_vol = shell_vol + enc.len() as f64 * cv;
        // Mass between shell displacement and total displacement: floats with enclosed, sinks without.
        let mass = 0.5 * 1_025.0 * (shell_vol + total_vol);
        assert!(
            1_025.0 * shell_vol < mass,
            "shell alone cannot float this mass"
        );
        assert!(1_025.0 * total_vol > mass, "shell + enclosed can");
        let inertia = mp.inertia * (mass / mp.mass);

        let run = |enclosed: &[IVec3]| {
            let mut body = ActiveBody::new(
                DVec3::new(0.0, SURFACE_R - 1.0, 0.0), // start 1 m under
                DVec3::ZERO,
                mass,
                inertia,
            );
            for _ in 0..40_000 {
                descent_step(&mut body, &hull, com, enclosed, &params, 0.005);
            }
            body.position.length() - SURFACE_R // final altitude relative to the surface
        };
        let floats = run(&enc);
        let sinks = run(&[]);
        assert!(
            floats > -2.0,
            "with enclosed volume the hull rises to/near the surface: {floats:.2} m"
        );
        assert!(
            sinks < floats - 1.0,
            "without it (shell only) it sinks markedly deeper: shell {sinks:.2} vs hull {floats:.2}"
        );
    }

    /// WI 716 (the panel enabler): a hull built of **thin panels** floats where the **solid-cube** hull
    /// of the same geometry sinks — because panels mass + displace a plate, not a cube, so the enclosed
    /// air wins against the light walls. Real mass (no auto-ballast).
    #[test]
    fn a_panel_hull_floats_where_a_solid_hull_sinks() {
        let solid = sealed_box(7, Material::ALUMINIUM); // hollow enough that panels can float it
        let mut panel = solid.clone();
        for v in &solid.voxels {
            panel.set_panel(v.cell, true); // every wall cell is a thin panel
        }
        let enc = enclosed_cells(&solid); // identical geometry ⇒ same enclosed set
        let g = 9.81;
        let rho = 1_025.0;
        let deep = DVec3::new(0.0, SURFACE_R - 100.0, 0.0); // fully submerged ⇒ max buoyancy

        let net = |craft: &VoxelCraft| {
            let mp = craft.mass_properties().unwrap();
            let buoy = buoyancy_wrench(
                craft,
                mp.center_of_mass,
                deep,
                DQuat::IDENTITY,
                SURFACE_R,
                0.0,
                rho,
                g,
                &enc,
            )
            .force
            .length();
            buoy - mp.mass * g // >0 ⇒ floats, <0 ⇒ sinks
        };
        assert!(
            net(&solid) < 0.0,
            "a solid aluminium hull sinks: net {}",
            net(&solid)
        );
        assert!(
            net(&panel) > 0.0,
            "the same hull in panels floats: net {}",
            net(&panel)
        );
    }

    /// Behaviour 5: a breached hull (a hole in the shell) has no sealed compartment, so it loses its
    /// enclosed-volume buoyancy — composing with flooding (WI 520) without a special case.
    #[test]
    fn a_breached_hull_loses_enclosed_buoyancy() {
        let mut hull = sealed_box(5, Material::ALUMINIUM);
        assert_eq!(enclosed_cells(&hull).len(), 27);
        // Punch a 1-cell hole in a wall: the exterior now reaches the interior ⇒ no sealed volume.
        hull.voxels.retain(|v| v.cell != IVec3::new(0, 2, 2));
        assert!(
            enclosed_cells(&hull).is_empty(),
            "a breached hull encloses nothing"
        );
    }

    // --- WI 713: open-boat displacement ---

    /// A `sealed_box` with its top face removed — an open bucket.
    fn open_box(n: i32, material: Material) -> VoxelCraft {
        let mut c = sealed_box(n, material);
        c.voxels.retain(|v| v.cell.y != n - 1);
        c
    }

    #[test]
    fn open_cavity_is_the_bucket_interior_and_empty_when_sealed() {
        // A sealed hull's interior is not exterior-reachable ⇒ no open cavity (711 owns it).
        let sealed = sealed_box(5, Material::ALUMINIUM);
        assert!(
            open_cavity(&sealed).cells.is_empty(),
            "a sealed hull has no open cavity"
        );
        // Remove the top ⇒ the 3×3×3 interior becomes the open bucket, rim = its top layer.
        let open = open_cavity(&open_box(5, Material::ALUMINIUM));
        assert_eq!(open.cells.len(), 27, "the 3×3×3 interior is the bucket");
        let rim_y = open.cells.iter().map(|c| c.y).max().unwrap();
        assert_eq!(open.rim_cells.len(), 9, "the rim is the top interior layer");
        assert!(open.rim_cells.iter().all(|c| c.y == rim_y));
    }

    #[test]
    fn open_boat_floats_until_swamped_over_the_rim() {
        let craft = open_box(5, Material::ALUMINIUM);
        let open = open_cavity(&craft);
        let mp = craft.mass_properties().unwrap();
        let com = mp.center_of_mass;
        let cs = craft.cell_size;
        let (rho, g) = (1025.0, 9.81);
        let rim_local_y = open
            .rim_cells
            .iter()
            .map(|c| (c.y as f64 + 0.5) * cs)
            .fold(f64::MIN, f64::max)
            - com.y;

        // Floating: the rim a cell above the waterline ⇒ the held-out volume buoys it (force up).
        let pos_float = DVec3::new(0.0, SURFACE_R + (cs - rim_local_y), 0.0);
        let load = open_cavity_load(
            &craft,
            com,
            pos_float,
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            rho,
            g,
            &open,
        );
        assert!(
            load.force.y > 0.0 && load.submerged_volume > 0.0,
            "an open boat floats on its held-out volume: {:?}",
            load.force
        );

        // Swamped: the rim well under the waterline ⇒ the cavity has flooded, no open buoyancy.
        let pos_deep = DVec3::new(0.0, SURFACE_R + (-rim_local_y - 2.0 * cs), 0.0);
        let load2 = open_cavity_load(
            &craft,
            com,
            pos_deep,
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            rho,
            g,
            &open,
        );
        assert_eq!(
            load2.force,
            DVec3::ZERO,
            "swamped over the rim ⇒ no open buoyancy"
        );

        // Contrast (WI 711): a *sealed* hull keeps full displacement even fully submerged.
        let sealed = sealed_box(5, Material::ALUMINIUM);
        let smp = sealed.mass_properties().unwrap();
        let sealed_deep = buoyancy_wrench(
            &sealed,
            smp.center_of_mass,
            pos_deep,
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            rho,
            g,
            &enclosed_cells(&sealed),
        );
        assert!(
            sealed_deep.force.length() > 0.0,
            "a sealed hull still displaces when deep (open ≠ sealed)"
        );
    }

    #[test]
    fn open_cavity_load_is_zero_in_vacuum_and_for_a_sealed_hull() {
        let craft = open_box(5, Material::ALUMINIUM);
        let open = open_cavity(&craft);
        let pos = DVec3::new(0.0, SURFACE_R, 0.0);
        // Vacuum (zero density) ⇒ no force.
        let vac = open_cavity_load(
            &craft,
            DVec3::ZERO,
            pos,
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            0.0,
            9.81,
            &open,
        );
        assert_eq!(vac.force, DVec3::ZERO);
        // A sealed hull has an empty cavity ⇒ no open load (sealed buoyancy unchanged).
        let sealed_open = open_cavity(&sealed_box(5, Material::ALUMINIUM));
        let load = open_cavity_load(
            &craft,
            DVec3::ZERO,
            pos,
            DQuat::IDENTITY,
            SURFACE_R,
            0.0,
            1025.0,
            9.81,
            &sealed_open,
        );
        assert_eq!(load.force, DVec3::ZERO);
    }
}

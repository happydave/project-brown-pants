//! Two-node lumped-capacitance thermal model (WI 687) — slice a (the spine) of the
//! *Thermal (Two-Node Lumped Model)* subsystem.
//!
//! The design's affordable primitive: **per voxel, a skin and a core temperature**
//! — not a spatially-resolved field, not FEA. Three heat flows move energy each
//! active-gear step:
//!
//! - **Convective exchange** (WI 696) — a single Newton relaxation of the skin
//!   toward a speed-dependent **recovery temperature**
//!   `T_recovery = T_static + r·v²/(2·c_p)` (the shock/boundary-layer recovery of
//!   the flow's kinetic energy). The conductance is the sum of two physical
//!   mechanisms sharing the *one* driving potential `(T_recovery − T_skin)`: a
//!   **forced** part on the **windward** faces (∝ `√ρ·v`, with the blunt-body
//!   softening `1/√A_w`), and a **natural/immersion** part on the whole exposed
//!   surface (∝ `ρ`). At hypersonic speed the forced part dominates and reproduces
//!   the Sutton–Graves `√ρ·v³` heating shape in the cold-skin limit, but now
//!   **saturates** at `T_recovery` instead of injecting unbounded power; at rest
//!   `T_recovery = T_static` and the natural part quenches the part toward the
//!   ambient medium (the WI 694 behaviour). The windward/exposed geometry comes
//!   from [`crate::aero::windward_faces`] — thermal consumes an aero-derived output,
//!   it does not re-derive the lattice (design resolutions T1/T3/T4).
//! - **Radiative out** — `εσA(T_skin⁴ − T_env⁴)` from each voxel's exposed skin
//!   area; the only sink in vacuum.
//! - **Conduction** — skin→core within a voxel and core→core between
//!   face-adjacent voxels, scaled by material conductivity.
//!
//! **Ablative shielding (WI 688).** An ablative material (e.g. [`crate::voxel::Thermal::ABLATOR`])
//! carries a per-voxel ablator budget; while its skin is above the material's
//! ablation set-point and ablator remains, a **thermostat** vaporises ablator to
//! hold the surface near the set-point (the latent-heat sink). When the budget is
//! spent the clamp stops and the skin rises into the failure path below.
//!
//! A voxel whose **skin** temperature reaches its material's maximum fails: the
//! caller removes it and re-partitions the lattice through the existing
//! connected-component breakage ([`sever_failed`]) — thermal adds no destruction
//! path of its own (design "thermal failure = breakage").
//!
//! **Warp & stability.** Heating exists only in the active gear (atmospheric flight
//! already drops warp), so there is no analytic-over-warp form to carry here; an
//! idle craft is simply not stepped. The integrator sub-steps internally at an
//! adaptively-bounded stable step so temperatures stay finite and bounded for any
//! `dt` (the thermal analogue of the project's numerical-stability rule). Headless;
//! the live `-- dive` wiring, HUD glow, and bus telemetry are in the dive scene (WI 691/693/688).

use crate::aero::windward_faces;
use crate::breakage::{connected_components, Severed};
use crate::fluid::{FluidSample, MediumKind};
use crate::voxel::{Thermal, VoxelCraft};
use glam::{DQuat, DVec3, IVec3};
use std::collections::{HashMap, HashSet};

/// Stefan–Boltzmann constant, W·m⁻²·K⁻⁴.
const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Convective heating coefficient (Sutton–Graves family). It fixes the magnitude of
/// the **forced** convective conductance so that, in the cold-skin limit, the Newton
/// relaxation toward `T_recovery` reproduces the historical per-area stagnation flux
/// `q = CONV_COEFF · √ρ · v³ / √A_w` (WI 696): a blunter craft (larger `A_w`) sees a
/// *lower* flux density (blunt-body softening) while absorbing more total heat over more
/// material. The absolute value is the calibration knob; the tests assert
/// ordering/relative properties, not absolute temperatures.
const CONV_COEFF: f64 = 1.74e-4;

/// Reference gas specific heat at constant pressure, J·kg⁻¹·K⁻¹ (WI 696) — maps the
/// flow's kinetic energy into the recovery (stagnation) temperature
/// `T_recovery = T_static + r·v²/(2·c_p)`. A fixed air-like value (not the per-material
/// specific heat), so the recovery temperature is a property of the *flow*.
const AIR_SPECIFIC_HEAT: f64 = 1_005.0;

/// Recovery factor `r` (WI 696): the fraction of the free-stream kinetic energy that
/// appears as the boundary-layer recovery temperature. Slightly below 1
/// (turbulent-boundary-layer typical).
const RECOVERY_FACTOR: f64 = 0.9;

/// Characteristic heating time, s (WI 692) — the timescale the surface thermal
/// layer ("skin") responds over. Sets the skin depth `δ = √(α·τ)` from the material's
/// thermal diffusivity `α = k/(ρ·c)`: a high-conductivity material (aluminium) gets a
/// deep skin, a low one (composite/ablator) a thin hot shell — the physical reason a
/// poor conductor's surface runs hotter.
const THERMAL_SKIN_TIME: f64 = 4.0;

/// Floor (and ceiling) on the skin's share of a voxel's mass (WI 692): keeps both the
/// skin and the core capacities strictly positive — so an interior (unexposed) voxel
/// still integrates and a fully-exposed thin voxel keeps a core node.
const MIN_SKIN_FRACTION: f64 = 0.01;

/// Safety factor below the explicit-Euler stability limit for the adaptive
/// sub-step (the limit is 2·C/G; staying well under it keeps the linear loss terms
/// from oscillating).
const STABILITY_SAFETY: f64 = 0.4;

/// Upper bound on internal sub-steps per [`ThermalState::step`] call — bounds cost
/// under a very large `dt`. When hit, the call integrates a stable prefix of the
/// interval (monotone toward equilibrium), never an unstable large step.
const MAX_SUBSTEPS: usize = 256;

/// Finiteness clamp on node temperatures, K — a guard so a pathological input can
/// never produce a non-finite or unbounded temperature.
const TEMP_CEILING: f64 = 100_000.0;

/// Natural / immersion convective coefficient (WI 694, unified WI 696): the
/// speed-independent part of the convective conductance, `h = CONV_EXCHANGE_COEFF · ρ`,
/// acting on the whole exposed surface toward the shared `T_recovery` potential — strong
/// in water, weak in thin air, and **zero in vacuum** (ρ=0). At rest (`T_recovery =
/// T_static`) it is the WI 694 quench; at speed it is dominated by the forced windward
/// conductance.
const CONV_EXCHANGE_COEFF: f64 = 2.0;

/// Water's boiling temperature, K (WI 698) — the set-point of the boiling clamp. A
/// submerged surface above this cannot quench through it within a sub-step; the heat it
/// sheds vaporises water at this temperature (a fixed surface value — pressure dependence
/// is out of scope).
const BOILING_TEMP: f64 = 373.0;

/// Latent heat of vaporisation of water, J·kg⁻¹ (WI 698) — converts the latent energy the
/// boiling clamp removes into a steam **mass** (the quantity that drives the WI 695 VFX).
const LATENT_HEAT_WATER: f64 = 2.26e6;

/// The two thermal nodes of one voxel (kelvin).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelTemp {
    /// Skin (surface) temperature — loaded by the medium, fast-responding.
    pub skin: f64,
    /// Core (bulk) temperature — the slow interior node.
    pub core: f64,
}

/// Per-voxel two-node thermal state for a craft, keyed by lattice cell (robust to
/// voxel removal/reordering across breakage). Runtime-only (not serialised) —
/// ephemeral active-gear state, like flooding.
#[derive(Clone, Debug, Default)]
pub struct ThermalState {
    temps: HashMap<IVec3, VoxelTemp>,
    /// Remaining ablator mass per voxel, kg (WI 688) — only present for ablative
    /// materials; drained by the ablation thermostat, never refilled.
    ablator: HashMap<IVec3, f64>,
    /// Steam mass vaporised by the boiling clamp during the most recent [`Self::step`]
    /// call, kg (WI 698) — reset at the start of each call, accumulated across its
    /// sub-steps. Drives the steam VFX; runtime-only, not serialised.
    steam_mass: f64,
}

/// The skin and core heat capacities (J/K) of a voxel (WI 692). The skin is the
/// responsive surface layer — `exposed_area × δ`, where the thermal skin depth
/// `δ = √(α·τ)` (α = `k/(ρ·c)` the diffusivity, τ = [`THERMAL_SKIN_TIME`]) — capped at
/// the cell and floored by [`MIN_SKIN_FRACTION`] so both nodes keep a positive
/// capacity. Replaces the old fixed skin fraction with a material-physical split: a
/// poor conductor gets a thin, fast-heating skin; a good one a deep, slow one.
fn skin_core_capacity(
    density: f64,
    th: Thermal,
    exposed_area: f64,
    cell_volume: f64,
) -> (f64, f64) {
    let mass = density * cell_volume;
    let diffusivity = th.conductivity / (density * th.specific_heat);
    let skin_depth = (diffusivity * THERMAL_SKIN_TIME).max(0.0).sqrt();
    let skin_volume = (exposed_area * skin_depth).clamp(
        MIN_SKIN_FRACTION * cell_volume,
        (1.0 - MIN_SKIN_FRACTION) * cell_volume,
    );
    let skin_mass = density * skin_volume;
    (
        skin_mass * th.specific_heat,
        (mass - skin_mass) * th.specific_heat,
    )
}

/// The ablator mass budget of a voxel (kg): `density · cell_volume · ablator_fraction`,
/// or 0 for a non-ablative material (WI 688).
fn ablator_budget(material: &crate::voxel::Material, cell_volume: f64) -> f64 {
    let th = material.thermal;
    if th.ablation_temp > 0.0 && th.latent_heat > 0.0 && th.ablator_fraction > 0.0 {
        material.density * cell_volume * th.ablator_fraction
    } else {
        0.0
    }
}

impl ThermalState {
    /// A fresh state with every voxel at `ambient` (both nodes) and a full ablator
    /// budget on each ablative voxel.
    pub fn new(craft: &VoxelCraft, ambient: f64) -> Self {
        let temps = craft
            .voxels
            .iter()
            .map(|v| {
                (
                    v.cell,
                    VoxelTemp {
                        skin: ambient,
                        core: ambient,
                    },
                )
            })
            .collect();
        let cell_volume = craft.cell_volume();
        let ablator = craft
            .voxels
            .iter()
            .filter_map(|v| {
                let b = ablator_budget(&v.material, cell_volume);
                (b > 0.0).then_some((v.cell, b))
            })
            .collect();
        Self {
            temps,
            ablator,
            steam_mass: 0.0,
        }
    }

    /// The skin temperature of `cell`, if present.
    pub fn skin(&self, cell: IVec3) -> Option<f64> {
        self.temps.get(&cell).map(|t| t.skin)
    }

    /// The core temperature of `cell`, if present.
    pub fn core(&self, cell: IVec3) -> Option<f64> {
        self.temps.get(&cell).map(|t| t.core)
    }

    /// The hottest skin temperature over all voxels (the educational re-entry gauge
    /// and the heating-model regression readout, design review M1). `0.0` for an
    /// empty craft.
    pub fn max_skin_temp(&self) -> f64 {
        self.temps.values().map(|t| t.skin).fold(0.0_f64, f64::max)
    }

    /// Advance the thermal state by `dt` seconds for a craft moving at `velocity`
    /// (world frame, relative to the static medium) with the given `orientation`,
    /// in the sampled medium, radiating toward `env_temp`. The physical model
    /// (convective flux scale = 1).
    pub fn step(
        &mut self,
        craft: &VoxelCraft,
        sample: &FluidSample,
        velocity: DVec3,
        orientation: DQuat,
        env_temp: f64,
        dt: f64,
    ) {
        self.step_scaled(craft, sample, velocity, orientation, env_temp, dt, 1.0);
    }

    /// As [`Self::step`], but with a `heat_scale` multiplier on the convective flux
    /// (WI 691). The multiplier is a **scenario balance scalar over the physical
    /// `√ρ·v³` shape** — it tunes magnitude, never the shape (design discipline:
    /// "balance multiplies physics, never replaces it"). `heat_scale = 1.0` is the
    /// pure physical model.
    #[allow(clippy::too_many_arguments)] // the full thermal step state is irreducible here
    pub fn step_scaled(
        &mut self,
        craft: &VoxelCraft,
        sample: &FluidSample,
        velocity: DVec3,
        orientation: DQuat,
        env_temp: f64,
        dt: f64,
        heat_scale: f64,
    ) {
        if dt <= 0.0 || craft.voxels.is_empty() {
            return;
        }

        // Fresh steam tally for this call (WI 698) — accumulated across the sub-steps.
        self.steam_mass = 0.0;

        // Per-voxel static inputs for this call: material thermal props, mass,
        // exposed area, and the convective power onto the skin.
        let cell_volume = craft.cell_volume();
        let cell_size = craft.cell_size;

        // Flow direction in the craft's local frame (windward = front faces).
        let local_flow = orientation.inverse() * velocity;
        let exposure = windward_faces(craft, local_flow);
        let speed = velocity.length();
        let density = sample.density;
        let total_windward: f64 = exposure.iter().map(|e| e.windward_area).sum();
        // Forced convective conductance per unit windward area (WI 696). Derived from
        // CONV_COEFF so the cold-skin limit of the Newton relaxation reproduces the
        // historical `√ρ·v³/√A_w` heating: the `2·c_p/r` factor converts the absolute
        // flux into a conductance against the `r·v²/(2·c_p)` recovery-temperature head.
        // Vanishes at rest (∝ v) and in vacuum (∝ √ρ).
        let forced_conductance_density = if density > 0.0 && speed > 0.0 && total_windward > 0.0 {
            heat_scale
                * CONV_COEFF
                * (2.0 * AIR_SPECIFIC_HEAT / RECOVERY_FACTOR)
                * density.sqrt()
                * speed
                / total_windward.sqrt()
        } else {
            0.0
        };
        // Recovery (stagnation) temperature of the flow — the single convective target.
        // At rest it collapses to the static medium temperature (quench, WI 694).
        let t_recovery =
            sample.temperature + RECOVERY_FACTOR * speed * speed / (2.0 * AIR_SPECIFIC_HEAT);

        // Index voxels and build the per-voxel work record.
        let exposure_by_cell: HashMap<IVec3, &crate::aero::VoxelExposure> =
            exposure.iter().map(|e| (e.cell, e)).collect();
        let conductivity_by_cell: HashMap<IVec3, f64> = craft
            .voxels
            .iter()
            .map(|v| (v.cell, v.material.thermal.conductivity))
            .collect();

        let mut nodes: Vec<Node> = Vec::with_capacity(craft.voxels.len());
        for v in &craft.voxels {
            let th = v.material.thermal;
            let exp = exposure_by_cell[&v.cell];
            // Physical skin/core split (WI 692): the responsive surface layer is the
            // exposed area × the material's thermal skin depth `δ = √(α·τ)`.
            let (c_skin, c_core) =
                skin_core_capacity(v.material.density, th, exp.exposed_area, cell_volume);
            // Occupied face-neighbours, each with a symmetric (averaged) core↔core
            // bond conductance so heat exchange across dissimilar materials conserves.
            let neighbours: Vec<(IVec3, f64)> = FACE_OFFSETS
                .iter()
                .map(|off| v.cell + *off)
                .filter_map(|n| {
                    conductivity_by_cell.get(&n).map(|&cond_nb| {
                        let bond = 0.5 * (th.conductivity + cond_nb) * cell_size;
                        (n, bond)
                    })
                })
                .collect();
            nodes.push(Node {
                cell: v.cell,
                conductivity: th.conductivity,
                c_skin,
                c_core,
                exposed_area: exp.exposed_area,
                forced_cond: forced_conductance_density * exp.windward_area,
                emissivity: th.emissivity,
                neighbours,
                ablation: (th.ablation_temp > 0.0 && th.latent_heat > 0.0)
                    .then_some((th.ablation_temp, th.latent_heat)),
            });
        }

        // Natural/immersion convective conductance with the medium (WI 694): coefficient
        // ∝ density (zero in vacuum). Shares the `t_recovery` target with the forced
        // windward conductance (WI 696), so at rest it quenches toward the static medium.
        let h_exchange = CONV_EXCHANGE_COEFF * sample.density;

        // Boiling clamp set-point (WI 698): only when submerged in a liquid; `None` in
        // atmosphere/vacuum leaves re-entry (688/693) and the air/vacuum cases untouched.
        let boiling = (sample.medium == MediumKind::Liquid).then_some(BOILING_TEMP);

        // Sub-step with an adaptively-bounded stable step.
        let mut remaining = dt;
        let mut steps = 0usize;
        while remaining > 0.0 && steps < MAX_SUBSTEPS {
            let sub_dt = self
                .stable_substep(&nodes, cell_size, h_exchange)
                .min(remaining);
            self.integrate_substep(
                &nodes, cell_size, env_temp, h_exchange, t_recovery, boiling, sub_dt,
            );
            remaining -= sub_dt;
            steps += 1;
        }
    }

    /// The largest stable explicit sub-step over all nodes, given current temps.
    fn stable_substep(&self, nodes: &[Node], cell_size: f64, h_exchange: f64) -> f64 {
        let mut min_dt = f64::INFINITY;
        for n in nodes {
            let t = self.temps[&n.cell];
            let g_sc = n.conductivity * cell_size; // skin↔core conductance
                                                   // skin loss derivative: conduction to core + linearised radiation + convective relaxation.
            let rad_deriv =
                4.0 * n.emissivity * STEFAN_BOLTZMANN * n.exposed_area * t.skin.max(0.0).powi(3);
            // WI 696: the convective term is now a linear relaxation toward `t_recovery`,
            // so its full conductance (forced windward + natural exposed) is the loss
            // derivative — including the speed-driven forced part, which can be large at
            // re-entry speed and must bound the sub-step.
            let conv_deriv = n.forced_cond + h_exchange * n.exposed_area;
            let skin_loss = (g_sc + rad_deriv + conv_deriv).max(1e-9);
            let dt_skin = STABILITY_SAFETY * n.c_skin / skin_loss;
            // core loss derivative: conduction to skin + to occupied neighbours.
            let g_cc: f64 = n.neighbours.iter().map(|&(_, bond)| bond).sum();
            let core_loss = (g_sc + g_cc).max(1e-9);
            let dt_core = STABILITY_SAFETY * n.c_core / core_loss;
            min_dt = min_dt.min(dt_skin).min(dt_core);
        }
        if min_dt.is_finite() {
            min_dt.max(1e-9)
        } else {
            // No loss terms anywhere (degenerate): take the whole remaining step.
            f64::INFINITY
        }
    }

    /// One explicit sub-step (Jacobi over voxels: neighbour temps read from the
    /// snapshot at the start of the sub-step), clamped finite.
    #[allow(clippy::too_many_arguments)] // the full thermal sub-step state is irreducible here
    fn integrate_substep(
        &mut self,
        nodes: &[Node],
        cell_size: f64,
        env_temp: f64,
        h_exchange: f64,
        t_recovery: f64,
        boiling: Option<f64>,
        dt: f64,
    ) {
        let snapshot = self.temps.clone();
        let env4 = env_temp.powi(4);
        for n in nodes {
            let t = snapshot[&n.cell];
            let g_sc = n.conductivity * cell_size;
            // Skin node (WI 696): one convective relaxation toward the recovery
            // temperature — forced (windward) + natural (exposed) conductance sharing the
            // single `(t_recovery − t.skin)` potential — plus radiation out and conduction
            // to the core. The convection saturates at `t_recovery` (no unbounded source)
            // and reverses sign to cool a part hotter than the flow.
            let p_rad = n.emissivity * STEFAN_BOLTZMANN * n.exposed_area * (t.skin.powi(4) - env4);
            let p_sc = g_sc * (t.skin - t.core); // skin → core
            let conv_cond = n.forced_cond + h_exchange * n.exposed_area;
            let p_conv = conv_cond * (t_recovery - t.skin);
            let d_skin = (p_conv - p_rad - p_sc) / n.c_skin * dt;
            // Core node: conduction from skin + exchange with occupied neighbours
            // (symmetric averaged bond conductance, computed once per call).
            let mut p_core = p_sc;
            for &(nb, g_cc) in &n.neighbours {
                let core_nb = snapshot[&nb].core;
                p_core += g_cc * (core_nb - t.core);
            }
            let d_core = p_core / n.c_core * dt;
            let mut new_skin = clamp_temp(t.skin + d_skin);
            let new_core = clamp_temp(t.core + d_core);

            // Ablation thermostat (WI 688): while this voxel is ablative, hot above the
            // set-point, and has ablator remaining, vaporise ablator to carry the
            // excess heat away — holding the skin toward the set-point (fully while the
            // budget lasts, partially once it runs low). Only removes heat; once the
            // budget hits zero the skin rises normally and can fail at `max_temp`.
            if let Some((abl_temp, latent)) = n.ablation {
                if new_skin > abl_temp {
                    if let Some(remaining) = self.ablator.get_mut(&n.cell) {
                        if *remaining > 0.0 && latent > 0.0 {
                            let excess_q = n.c_skin * (new_skin - abl_temp);
                            let consumed = (excess_q / latent).min(*remaining);
                            new_skin -= consumed * latent / n.c_skin;
                            *remaining -= consumed;
                        }
                    }
                }
            }

            // Boiling clamp (WI 698): while submerged and boiling-hot, the latent plateau
            // holds the surface at the boiling point — it cannot quench through it within a
            // sub-step. The heat its cooling sheds at the plateau vaporises water (water is
            // an unbounded consumable, so unlike ablation nothing depletes); the latent
            // energy / water's latent heat is the steam mass produced. Only arrested
            // *cooling* counts (a surface being heated underwater makes no steam).
            if let Some(t_boil) = boiling {
                if t.skin > t_boil {
                    let clamped = new_skin.max(t_boil);
                    let vaporised_q = n.c_skin * (t.skin - clamped);
                    if vaporised_q > 0.0 {
                        self.steam_mass += vaporised_q / LATENT_HEAT_WATER;
                    }
                    new_skin = clamped;
                }
            }

            let entry = self.temps.get_mut(&n.cell).expect("node cell present");
            entry.skin = new_skin;
            entry.core = new_core;
        }
    }

    /// The cells whose skin temperature has reached their material's maximum
    /// temperature — the voxels that have failed and must be removed.
    pub fn failed_cells(&self, craft: &VoxelCraft) -> Vec<IVec3> {
        craft
            .voxels
            .iter()
            .filter(|v| {
                self.skin(v.cell)
                    .map(|s| s >= v.material.thermal.max_temp)
                    .unwrap_or(false)
            })
            .map(|v| v.cell)
            .collect()
    }

    /// Whether any voxel has reached its material maximum temperature — the
    /// non-allocating "is anything overheating?" check (the HUD/telemetry path),
    /// equivalent to `!failed_cells(craft).is_empty()`.
    pub fn any_over_limit(&self, craft: &VoxelCraft) -> bool {
        craft.voxels.iter().any(|v| {
            self.skin(v.cell)
                .map(|s| s >= v.material.thermal.max_temp)
                .unwrap_or(false)
        })
    }

    /// Remaining ablator mass (kg) on `cell`, if it is an ablative voxel (WI 688).
    pub fn ablator_remaining(&self, cell: IVec3) -> Option<f64> {
        self.ablator.get(&cell).copied()
    }

    /// The craft's remaining ablator as a fraction of its initial budget (WI 688) —
    /// the shield gauge for the HUD/telemetry. `None` if the craft carries no ablative
    /// material; `0.0` once every shield is spent.
    pub fn ablator_fraction_remaining(&self, craft: &VoxelCraft) -> Option<f64> {
        let cell_volume = craft.cell_volume();
        let mut total_initial = 0.0;
        let mut total_remaining = 0.0;
        for v in &craft.voxels {
            let budget = ablator_budget(&v.material, cell_volume);
            if budget > 0.0 {
                total_initial += budget;
                total_remaining += self.ablator.get(&v.cell).copied().unwrap_or(0.0);
            }
        }
        (total_initial > 0.0).then(|| total_remaining / total_initial)
    }

    /// Steam mass (kg) vaporised by the boiling clamp during the most recent [`Self::step`]
    /// call (WI 698) — the quantity of water flashed to steam as a submerged hot surface
    /// quenched. `0.0` unless the craft was submerged and boiling. Drives the WI 695 steam
    /// VFX from energy actually vaporised rather than a skin-temperature proxy.
    pub fn steam_mass(&self) -> f64 {
        self.steam_mass
    }
}

/// Per-voxel work record for one [`ThermalState::step`] call (static across the
/// internal sub-steps).
struct Node {
    cell: IVec3,
    conductivity: f64,
    c_skin: f64,
    c_core: f64,
    exposed_area: f64,
    /// Forced (windward) convective conductance, W/K (WI 696): the speed-driven part of
    /// the skin's convective coupling, relaxing it toward `t_recovery`. Zero at rest.
    forced_cond: f64,
    emissivity: f64,
    /// Occupied face-neighbours as `(cell, bond conductance)` for core↔core
    /// conduction (the conductance is the symmetric averaged value).
    neighbours: Vec<(IVec3, f64)>,
    /// `(ablation_temp, latent_heat)` if the material is ablative (WI 688); the
    /// thermostat consumes this voxel's ablator while its skin is above the set-point.
    ablation: Option<(f64, f64)>,
}

/// The six axis-aligned face offsets of a cubic cell.
const FACE_OFFSETS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// Clamp a node temperature to a finite, bounded range (a NaN/Inf and runaway
/// guard; never reached in normal active-gear stepping).
fn clamp_temp(t: f64) -> f64 {
    if t.is_finite() {
        t.clamp(0.0, TEMP_CEILING)
    } else {
        TEMP_CEILING
    }
}

/// Remove `failed` cells from `craft` and re-partition the remaining lattice
/// through the existing connected-component breakage — the thermal→breakage
/// failure path (no new severing logic). Returns the fragment crafts (one if the
/// removal left the lattice connected, more if it disconnected it).
pub fn sever_failed(craft: &VoxelCraft, failed: &[IVec3]) -> Vec<VoxelCraft> {
    let failset: HashSet<IVec3> = failed.iter().copied().collect();
    let mut reduced = craft.clone();
    reduced.voxels.retain(|v| !failset.contains(&v.cell));
    connected_components(&reduced, &Severed::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid::{FluidMedium, FluidSample, MediumKind};
    use crate::voxel::{Material, Thermal, Voxel};

    const AMBIENT: f64 = 300.0;

    /// A dense atmospheric sample (for harsh-heating tests).
    fn dense_air() -> FluidSample {
        FluidMedium::EARTHLIKE.sample_altitude(0.0)
    }
    fn vacuum() -> FluidSample {
        FluidSample {
            density: 0.0,
            pressure: 0.0,
            medium: MediumKind::Vacuum,
            temperature: 250.0,
        }
    }

    /// A solid box of voxels of one material spanning the half-open ranges.
    fn box_craft(nx: i32, ny: i32, nz: i32, cell: f64, material: Material) -> VoxelCraft {
        let mut c = VoxelCraft::new(cell);
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    c.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material,
                    });
                }
            }
        }
        c
    }

    fn alu() -> Material {
        Material::ALUMINIUM
    }

    // I1: no heating in vacuum or at rest.
    #[test]
    fn vacuum_and_rest_do_not_heat() {
        let craft = box_craft(2, 2, 2, 1.0, alu());
        // Vacuum, moving fast: no medium → no convection.
        let mut s1 = ThermalState::new(&craft, AMBIENT);
        for _ in 0..100 {
            s1.step(
                &craft,
                &vacuum(),
                DVec3::new(8000.0, 0.0, 0.0),
                DQuat::IDENTITY,
                AMBIENT,
                0.1,
            );
        }
        assert!(
            s1.max_skin_temp() <= AMBIENT + 1e-6,
            "vacuum heated: {}",
            s1.max_skin_temp()
        );

        // Dense medium, at rest: no relative speed → no convection.
        let mut s2 = ThermalState::new(&craft, AMBIENT);
        for _ in 0..100 {
            s2.step(
                &craft,
                &dense_air(),
                DVec3::ZERO,
                DQuat::IDENTITY,
                AMBIENT,
                0.1,
            );
        }
        assert!(
            s2.max_skin_temp() <= AMBIENT + 1e-6,
            "rest heated: {}",
            s2.max_skin_temp()
        );
    }

    // WI 875: the at-rest quench floor is the medium's own surface-ambient
    // temperature. Moving the earthlike anchor 250 K → ISA sea-level (288.15 K)
    // raises this floor by ~38 K — measured here and accepted: it is a correct
    // ~15 °C surface ambient, is negligible against re-entry recovery
    // temperatures (the kinetic `v²/(2·c_p)` head at entry speed dwarfs it), and
    // sits far below any material maximum, so no voxel-failure behaviour changes.
    #[test]
    fn rest_quench_floor_is_the_medium_surface_ambient() {
        let craft = box_craft(2, 2, 2, 1.0, alu());
        let medium = dense_air(); // EARTHLIKE at the surface
        let floor = medium.temperature;
        // The floor is the ISA anchor now, and strictly above the old 250 K magic.
        assert!((floor - crate::fluid::ISA_SEA_LEVEL_TEMPERATURE).abs() < 1e-9);
        assert!(
            floor > 250.0,
            "WI 875 raised the rest floor above 250 K: {floor}"
        );

        // The floor is a **fixed point**: a craft already at the medium
        // temperature, at rest, with the radiative sink also at the medium
        // temperature, has zero convective and radiative gradient — it neither
        // heats nor cools, so the steady state is exactly the medium ambient.
        let mut at_floor = ThermalState::new(&craft, floor);
        for _ in 0..200 {
            at_floor.step(&craft, &medium, DVec3::ZERO, DQuat::IDENTITY, floor, 1.0);
        }
        assert!(
            (at_floor.max_skin_temp() - floor).abs() < 1e-6,
            "the medium ambient is a rest fixed point, got {}",
            at_floor.max_skin_temp()
        );

        // And it is an *attracting* floor: a hotter craft at rest only cools, and
        // never overshoots below the medium ambient.
        let mut hot = ThermalState::new(&craft, 500.0);
        let start = hot.max_skin_temp();
        for _ in 0..200 {
            hot.step(&craft, &medium, DVec3::ZERO, DQuat::IDENTITY, floor, 1.0);
        }
        let cooled = hot.max_skin_temp();
        assert!(
            cooled < start && cooled >= floor - 1e-6,
            "at rest a hot craft cools toward the floor {floor} without overshoot: {start} → {cooled}"
        );
    }

    // Heating rises with density and speed.
    #[test]
    fn faster_denser_descent_is_hotter() {
        let craft = box_craft(2, 2, 2, 1.0, alu());
        let run = |density: f64, speed: f64| {
            let sample = FluidSample {
                density,
                pressure: 0.0,
                medium: MediumKind::Atmosphere,
                temperature: 250.0,
            };
            let mut st = ThermalState::new(&craft, AMBIENT);
            for _ in 0..50 {
                st.step(
                    &craft,
                    &sample,
                    DVec3::new(speed, 0.0, 0.0),
                    DQuat::IDENTITY,
                    AMBIENT,
                    0.05,
                );
            }
            st.max_skin_temp()
        };
        let hot = run(1.2, 3000.0);
        let mild = run(0.3, 1500.0);
        assert!(hot > mild, "expected hotter: hot={hot} mild={mild}");
        assert!(
            hot > AMBIENT,
            "harsh profile should heat above ambient: {hot}"
        );
    }

    // T1: blunt orientation stays cooler than a slender one, and windward beats leeward.
    #[test]
    fn blunt_orientation_survives_where_slender_does_not() {
        // A 4×4×1 plate: blunt face is the 4×4 (flow along z); slender edge is 4×1 (flow along x).
        let plate = box_craft(4, 4, 1, 1.0, Material::TITANIUM);
        let sample = FluidSample {
            density: 1.0,
            pressure: 0.0,
            medium: MediumKind::Atmosphere,
            temperature: 250.0,
        };
        let run = |vel: DVec3| {
            let mut st = ThermalState::new(&plate, AMBIENT);
            for _ in 0..200 {
                st.step(&plate, &sample, vel, DQuat::IDENTITY, AMBIENT, 0.05);
            }
            st
        };
        let blunt = run(DVec3::new(0.0, 0.0, 4000.0)); // flow on the broad face
        let slender = run(DVec3::new(4000.0, 0.0, 0.0)); // flow on the thin edge
        assert!(
            blunt.max_skin_temp() < slender.max_skin_temp(),
            "blunt should be cooler: blunt={} slender={}",
            blunt.max_skin_temp(),
            slender.max_skin_temp()
        );

        // Windward beats leeward along the slender flow (+x): the leading face is +x,
        // so the x=3 voxels are windward and the x=0 voxels trail.
        let front = slender.skin(IVec3::new(3, 0, 0)).unwrap();
        let back = slender.skin(IVec3::new(0, 0, 0)).unwrap();
        assert!(
            front > back,
            "windward should beat leeward: front={front} back={back}"
        );
    }

    // I2 + warp-safety: an idle hot craft relaxes toward the environment, stays finite.
    #[test]
    fn idle_relaxes_toward_environment() {
        let craft = box_craft(2, 2, 2, 1.0, alu());
        let mut st = ThermalState::new(&craft, 1200.0); // start hot
        let env = 300.0;
        let before = st.max_skin_temp();
        for _ in 0..2000 {
            // No velocity, vacuum: only radiation + conduction → relax toward env.
            st.step(&craft, &vacuum(), DVec3::ZERO, DQuat::IDENTITY, env, 1.0);
        }
        let after = st.max_skin_temp();
        assert!(after < before, "should cool: before={before} after={after}");
        assert!(after >= env - 1.0, "should not undershoot env: {after}");
        assert!(after.is_finite());
    }

    // I2: a single very large/harsh step stays finite and bounded.
    #[test]
    fn temperatures_stay_finite_under_harsh_step() {
        let craft = box_craft(3, 3, 3, 0.1, alu()); // small build, low heat capacity
        let mut st = ThermalState::new(&craft, AMBIENT);
        st.step(
            &craft,
            &dense_air(),
            DVec3::new(12000.0, 0.0, 0.0),
            DQuat::IDENTITY,
            AMBIENT,
            10.0,
        );
        let m = st.max_skin_temp();
        assert!(m.is_finite() && m <= TEMP_CEILING, "unbounded: {m}");
        assert!(m > AMBIENT, "should have heated: {m}");
    }

    // I5: an over-temperature voxel fails and re-partitions via existing breakage.
    #[test]
    fn over_temperature_voxel_fails_into_breakage() {
        // A 1×1×3 bar; the middle voxel (z=1) is the structural bridge.
        let bar = box_craft(1, 1, 3, 1.0, alu());
        let mut st = ThermalState::new(&bar, AMBIENT);
        // Drive the middle voxel's skin past aluminium's max temperature directly.
        st.temps.get_mut(&IVec3::new(0, 0, 1)).unwrap().skin = Thermal::ALUMINIUM.max_temp + 50.0;
        let failed = st.failed_cells(&bar);
        assert_eq!(failed, vec![IVec3::new(0, 0, 1)], "the bridge voxel failed");
        // Removing the bridge disconnects the bar into two fragments.
        let fragments = sever_failed(&bar, &failed);
        assert_eq!(fragments.len(), 2, "bridge removal splits the bar");
        let total: usize = fragments.iter().map(|f| f.voxels.len()).sum();
        assert_eq!(total, 2, "two end voxels survive");
    }

    // The heating dynamics actually reach failure on a low-max material.
    #[test]
    fn harsh_heating_drives_a_voxel_to_failure() {
        let craft = box_craft(2, 2, 2, 0.2, alu()); // alu max_temp 900 K
        let sample = FluidSample {
            density: 2.0,
            pressure: 0.0,
            medium: MediumKind::Atmosphere,
            temperature: 250.0,
        };
        let mut st = ThermalState::new(&craft, AMBIENT);
        let mut failed = Vec::new();
        for _ in 0..400 {
            st.step(
                &craft,
                &sample,
                DVec3::new(6000.0, 0.0, 0.0),
                DQuat::IDENTITY,
                AMBIENT,
                0.05,
            );
            failed = st.failed_cells(&craft);
            if !failed.is_empty() {
                break;
            }
        }
        assert!(
            !failed.is_empty(),
            "harsh heating should fail a voxel; max={}",
            st.max_skin_temp()
        );
    }

    // --- WI 692: physical skin depth ---

    #[test]
    fn skin_depth_is_physical_and_thinner_for_poor_conductors() {
        let cell_volume = 1.0;
        // Both nodes keep a positive capacity, and the skin is a smaller share than the
        // core (a surface layer, not the bulk).
        let (alu_skin, alu_core) = skin_core_capacity(2700.0, Thermal::ALUMINIUM, 6.0, cell_volume);
        let (comp_skin, comp_core) =
            skin_core_capacity(1600.0, Thermal::COMPOSITE, 6.0, cell_volume);
        for c in [alu_skin, alu_core, comp_skin, comp_core] {
            assert!(c > 0.0, "capacities are strictly positive");
        }
        // The good conductor (aluminium) gets a *deeper* skin than the poor one
        // (composite) — the physical reason a poor conductor's surface runs hotter.
        let alu_frac = alu_skin / (alu_skin + alu_core);
        let comp_frac = comp_skin / (comp_skin + comp_core);
        assert!(
            alu_frac > comp_frac,
            "aluminium skin deeper than composite: {alu_frac} vs {comp_frac}"
        );
        // An interior (unexposed) voxel still keeps a positive, floored skin capacity.
        let (interior_skin, interior_core) =
            skin_core_capacity(2700.0, Thermal::ALUMINIUM, 0.0, cell_volume);
        assert!(interior_skin > 0.0 && interior_core > 0.0);
    }

    // --- WI 688: ablation ---

    fn atmosphere(density: f64) -> FluidSample {
        FluidSample {
            density,
            pressure: 0.0,
            medium: MediumKind::Atmosphere,
            temperature: 250.0,
        }
    }

    // The headline: an ablative shield survives a re-entry that destroys bare metal.
    #[test]
    fn ablative_shield_survives_where_bare_material_fails() {
        let sample = atmosphere(1.5);
        let vel = DVec3::new(4000.0, 0.0, 0.0);

        let bare = box_craft(2, 2, 2, 0.2, Material::ALUMINIUM); // max 900 K
        let shield = box_craft(2, 2, 2, 0.2, Material::ABLATOR); // ablates at 1300 K
        let mut sb = ThermalState::new(&bare, AMBIENT);
        let mut ss = ThermalState::new(&shield, AMBIENT);

        let mut bare_failed = false;
        for _ in 0..150 {
            sb.step(&bare, &sample, vel, DQuat::IDENTITY, AMBIENT, 0.05);
            ss.step(&shield, &sample, vel, DQuat::IDENTITY, AMBIENT, 0.05);
            if !sb.failed_cells(&bare).is_empty() {
                bare_failed = true;
            }
        }
        assert!(bare_failed, "bare aluminium should fail under the load");
        assert!(
            ss.failed_cells(&shield).is_empty(),
            "ablative shield should survive: max_skin={}",
            ss.max_skin_temp()
        );
        // It survived by ablating: some — but not all — ablator was consumed.
        let frac = ss.ablator_fraction_remaining(&shield).unwrap();
        assert!(
            frac > 0.0 && frac < 1.0,
            "ablator partially consumed: {frac}"
        );
        // The surface was held below the bare-char failure temperature.
        assert!(ss.max_skin_temp() < Thermal::ABLATOR.max_temp);
    }

    // Once the ablator is spent, the bare char heats normally and fails.
    #[test]
    fn ablator_depletes_then_fails() {
        // A small (0.1 m) shield voxel → a small budget → it depletes under a sustained
        // harsh load, after which it exceeds its bare max temperature and fails.
        let mut craft = VoxelCraft::new(0.1);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ABLATOR,
        });
        let sample = atmosphere(3.0);
        let vel = DVec3::new(8000.0, 0.0, 0.0);
        let mut st = ThermalState::new(&craft, AMBIENT);
        let initial = st.ablator_remaining(IVec3::ZERO).unwrap();
        assert!(initial > 0.0);

        let mut failed = false;
        for _ in 0..5000 {
            st.step(&craft, &sample, vel, DQuat::IDENTITY, AMBIENT, 0.05);
            if !st.failed_cells(&craft).is_empty() {
                failed = true;
                break;
            }
        }
        assert_eq!(
            st.ablator_remaining(IVec3::ZERO).unwrap(),
            0.0,
            "the ablator drained to empty"
        );
        assert!(failed, "after depletion the bare char fails");
    }

    // Passive shielding is just content: a high-max-temp material survives without a consumable.
    #[test]
    fn passive_shield_survives_mild_profile_without_ablator() {
        let craft = box_craft(2, 2, 2, 1.0, Material::COMPOSITE); // max 3000 K, no ablator
        let mut st = ThermalState::new(&craft, AMBIENT);
        let sample = atmosphere(0.5);
        for _ in 0..200 {
            st.step(
                &craft,
                &sample,
                DVec3::new(2000.0, 0.0, 0.0),
                DQuat::IDENTITY,
                AMBIENT,
                0.05,
            );
        }
        assert!(
            st.failed_cells(&craft).is_empty(),
            "passive composite survives a mild profile"
        );
        assert!(
            st.ablator_fraction_remaining(&craft).is_none(),
            "composite carries no ablator (passive shielding)"
        );
    }

    // --- WI 694: convective heat exchange with the medium ---

    fn liquid(temperature: f64) -> FluidSample {
        FluidSample {
            density: 1025.0,
            pressure: 0.0,
            medium: MediumKind::Liquid,
            temperature,
        }
    }

    // A hot block dropped in water quenches; the same block in vacuum only radiates.
    #[test]
    fn hot_block_quenches_in_water_not_in_vacuum() {
        let craft = box_craft(2, 2, 2, 0.5, Material::ALUMINIUM);
        let env = 290.0; // both cases radiate to the same sink; only the medium coupling differs
        let run = |sample: &FluidSample| {
            let mut st = ThermalState::new(&craft, 1500.0); // start hot
            for _ in 0..600 {
                st.step(&craft, sample, DVec3::ZERO, DQuat::IDENTITY, env, 0.1);
                // 60 s at rest
            }
            st.max_skin_temp()
        };
        let in_water = run(&liquid(290.0));
        let in_vacuum = run(&vacuum());
        assert!(
            in_water < in_vacuum,
            "water quenches faster than vacuum: water={in_water} vacuum={in_vacuum}"
        );
        assert!(
            in_water < 600.0,
            "the water-quenched surface cooled well below the 1500 K start: {in_water}"
        );
        assert!(
            in_water >= 289.0,
            "never below the water temperature: {in_water}"
        );
    }

    // Quenching is faster in dense water than in thin air (the coefficient scales with density).
    #[test]
    fn quench_faster_in_water_than_air() {
        let craft = box_craft(2, 2, 2, 0.5, Material::ALUMINIUM);
        let env = 290.0;
        let run = |sample: &FluidSample| {
            let mut st = ThermalState::new(&craft, 1500.0);
            for _ in 0..300 {
                st.step(&craft, sample, DVec3::ZERO, DQuat::IDENTITY, env, 0.1);
            }
            st.max_skin_temp()
        };
        let in_water = run(&liquid(290.0));
        let in_air = run(&atmosphere(1.2)); // thin air at 250 K
        assert!(
            in_water < in_air,
            "water cools faster than air: water={in_water} air={in_air}"
        );
    }

    // --- WI 696: convective heating saturates at the recovery temperature ---

    // The soft-spot fix: convection relaxes the skin toward a finite recovery temperature
    // instead of injecting unbounded power. The skin never exceeds it, heating slows as it
    // approaches, and a part hotter than the flow is cooled toward it.
    #[test]
    fn convective_heating_saturates_at_recovery_temperature() {
        let craft = box_craft(2, 2, 2, 0.5, Material::COMPOSITE); // high failure temp
        let sample = atmosphere(1.0); // T_static = 250 K
        let vel = DVec3::new(4_000.0, 0.0, 0.0);
        let speed = vel.length();
        // The recovery temperature the flow can heat toward (mirrors the model).
        let t_recovery =
            sample.temperature + RECOVERY_FACTOR * speed * speed / (2.0 * AIR_SPECIFIC_HEAT);

        // Heat hard for a while: the skin climbs toward, but never past, the recovery temp.
        let mut st = ThermalState::new(&craft, AMBIENT);
        for _ in 0..2_000 {
            st.step(&craft, &sample, vel, DQuat::IDENTITY, AMBIENT, 0.05);
        }
        let hot = st.max_skin_temp();
        assert!(hot > AMBIENT, "should have heated: {hot}");
        assert!(
            hot < t_recovery,
            "skin must not exceed the recovery temperature: {hot} vs {t_recovery}"
        );

        // Saturation: one step warms a cold skin more than a skin already near T_recovery.
        let one_step = |start: f64| {
            let mut s = ThermalState::new(&craft, start);
            let before = s.max_skin_temp();
            s.step(&craft, &sample, vel, DQuat::IDENTITY, AMBIENT, 0.01);
            s.max_skin_temp() - before
        };
        let d_cold = one_step(300.0);
        let d_warm = one_step(0.9 * t_recovery);
        assert!(d_cold > 0.0, "a cold skin heats: {d_cold}");
        assert!(
            d_warm < d_cold,
            "heating saturates near recovery: warm Δ {d_warm} < cold Δ {d_cold}"
        );

        // Above the recovery temperature the convective term reverses sign (cools).
        let mut above = ThermalState::new(&craft, t_recovery + 2_000.0);
        let before = above.max_skin_temp();
        above.step(&craft, &sample, vel, DQuat::IDENTITY, AMBIENT, 0.01);
        assert!(
            above.max_skin_temp() < before,
            "a skin above T_recovery cools toward it: {} !< {before}",
            above.max_skin_temp()
        );
    }

    // --- WI 697: windward shadowing protects a body behind a heat shield ---

    // The same body voxel heats far less behind a standoff shield than directly exposed:
    // the shield protects by shadowing, not by a calibrated heat scale.
    #[test]
    fn a_shadowed_body_stays_cooler_than_an_exposed_one() {
        let sample = atmosphere(1.5);
        let vel = DVec3::new(0.0, 0.0, 4_000.0); // flow +Z
        let body = IVec3::new(0, 0, 0);

        let run = |craft: &VoxelCraft| {
            let mut st = ThermalState::new(craft, AMBIENT);
            for _ in 0..150 {
                st.step(craft, &sample, vel, DQuat::IDENTITY, AMBIENT, 0.05);
            }
            st.skin(body).unwrap()
        };

        // Directly exposed.
        let mut exposed = VoxelCraft::new(0.3);
        exposed.voxels.push(Voxel {
            cell: body,
            material: Material::COMPOSITE,
        });
        let t_exposed = run(&exposed);

        // Behind a standoff shield (gap at z=1, shield at z=2) — not adjacent, so no
        // conduction bridges them; only the windward shadow protects the body.
        let mut shielded = VoxelCraft::new(0.3);
        shielded.voxels.push(Voxel {
            cell: body,
            material: Material::COMPOSITE,
        });
        shielded.voxels.push(Voxel {
            cell: IVec3::new(0, 0, 2),
            material: Material::COMPOSITE,
        });
        let t_shielded = run(&shielded);

        assert!(t_exposed > AMBIENT, "the exposed body heats: {t_exposed}");
        assert!(
            t_shielded < t_exposed,
            "the shadowed body stays cooler: shielded {t_shielded} vs exposed {t_exposed}"
        );
    }

    // --- WI 698: boiling clamp (latent-heat plateau + steam) ---

    // A hot block submerged in water boils: the surface is held at the boiling point (it
    // does not quench through it in one sub-step) and the vaporised water is reported as
    // steam mass. In air and vacuum the clamp is inert (no steam) — AC2.
    #[test]
    fn boiling_clamp_plateaus_submerged_surface_and_reports_steam() {
        let craft = box_craft(2, 2, 2, 0.5, Material::ALUMINIUM);
        let env = 290.0;

        // Submerged in water: boils — steam produced, surface plateaus at/above boiling.
        let mut water = ThermalState::new(&craft, 1_500.0);
        water.step(
            &craft,
            &liquid(290.0),
            DVec3::ZERO,
            DQuat::IDENTITY,
            env,
            0.1,
        );
        assert!(
            water.steam_mass() > 0.0,
            "boiling produced steam: {}",
            water.steam_mass()
        );
        assert!(
            water.max_skin_temp() >= BOILING_TEMP - 1.0,
            "surface held at/above boiling, not dropped to the ocean instantly: {}",
            water.max_skin_temp()
        );

        // Same hot block in air: the clamp is inert (no steam) — AC2.
        let mut air = ThermalState::new(&craft, 1_500.0);
        air.step(
            &craft,
            &atmosphere(1.2),
            DVec3::ZERO,
            DQuat::IDENTITY,
            env,
            0.1,
        );
        assert_eq!(air.steam_mass(), 0.0, "no boiling in air");

        // And in vacuum.
        let mut vac = ThermalState::new(&craft, 1_500.0);
        vac.step(&craft, &vacuum(), DVec3::ZERO, DQuat::IDENTITY, env, 0.1);
        assert_eq!(vac.steam_mass(), 0.0, "no boiling in vacuum");

        // A submerged block below the boiling point makes no steam (just quenches).
        let mut cool = ThermalState::new(&craft, 320.0); // below BOILING_TEMP
        cool.step(
            &craft,
            &liquid(290.0),
            DVec3::ZERO,
            DQuat::IDENTITY,
            env,
            0.1,
        );
        assert_eq!(
            cool.steam_mass(),
            0.0,
            "a sub-boiling submerged surface makes no steam"
        );
    }
}

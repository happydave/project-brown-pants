//! Two-node lumped-capacitance thermal model (WI 687) — slice a (the spine) of the
//! *Thermal (Two-Node Lumped Model)* subsystem.
//!
//! The design's affordable primitive: **per voxel, a skin and a core temperature**
//! — not a spatially-resolved field, not FEA. Three heat flows move energy each
//! active-gear step:
//!
//! - **Convective in** — a Sutton–Graves-shaped stagnation flux (rises with the
//!   square root of medium density and the *cube* of speed, softened for blunt
//!   bodies via the windward frontal area) distributed onto each voxel's
//!   **windward** exposed faces. The exposed/windward geometry comes from
//!   [`crate::aero::windward_faces`] — thermal consumes an aero-derived output, it
//!   does not re-derive the lattice (design resolutions T1/T3/T4).
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
use crate::fluid::FluidSample;
use crate::voxel::{Thermal, VoxelCraft};
use glam::{DQuat, DVec3, IVec3};
use std::collections::{HashMap, HashSet};

/// Stefan–Boltzmann constant, W·m⁻²·K⁻⁴.
const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Convective heating coefficient (Sutton–Graves family). The per-area stagnation
/// flux is `q = CONV_COEFF · √ρ · v³ / √A_w`, where `A_w` is the total windward
/// frontal area — so a blunter craft (larger `A_w`) sees a *lower* flux density
/// (blunt-body softening) while absorbing more total heat over more material. The
/// absolute value is the calibration knob (plan Remaining Unknown); the tests
/// assert ordering/relative properties, not absolute temperatures.
const CONV_COEFF: f64 = 1.74e-4;

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

/// Convective heat-exchange coefficient (WI 694): the skin exchanges heat with the
/// medium at `h · area · (T_medium − T_skin)`, with `h = CONV_EXCHANGE_COEFF · ρ` —
/// so coupling is strong in water, weak in thin air, and **zero in vacuum** (ρ=0).
/// Bidirectional: it cools a hot part (quench) and warms a cold one toward the medium
/// temperature. Distinct from the speed-driven aeroheating (which dominates at
/// re-entry speed); this governs at rest.
const CONV_EXCHANGE_COEFF: f64 = 2.0;

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
        Self { temps, ablator }
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
        let q_density = if density > 0.0 && speed > 0.0 && total_windward > 0.0 {
            heat_scale * CONV_COEFF * density.sqrt() * speed.powi(3) / total_windward.sqrt()
        } else {
            0.0
        };

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
                conv_power: q_density * exp.windward_area,
                emissivity: th.emissivity,
                neighbours,
                ablation: (th.ablation_temp > 0.0 && th.latent_heat > 0.0)
                    .then_some((th.ablation_temp, th.latent_heat)),
            });
        }

        // Convective heat-exchange with the medium (WI 694): coefficient ∝ density
        // (zero in vacuum), target = the medium temperature.
        let h_exchange = CONV_EXCHANGE_COEFF * sample.density;
        let medium_temp = sample.temperature;

        // Sub-step with an adaptively-bounded stable step.
        let mut remaining = dt;
        let mut steps = 0usize;
        while remaining > 0.0 && steps < MAX_SUBSTEPS {
            let sub_dt = self
                .stable_substep(&nodes, cell_size, h_exchange)
                .min(remaining);
            self.integrate_substep(&nodes, cell_size, env_temp, h_exchange, medium_temp, sub_dt);
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
                                                   // skin loss derivative: conduction to core + linearised radiation + convective exchange.
            let rad_deriv =
                4.0 * n.emissivity * STEFAN_BOLTZMANN * n.exposed_area * t.skin.max(0.0).powi(3);
            let exchange_deriv = h_exchange * n.exposed_area; // WI 694: linear loss toward T_medium
            let skin_loss = (g_sc + rad_deriv + exchange_deriv).max(1e-9);
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
        medium_temp: f64,
        dt: f64,
    ) {
        let snapshot = self.temps.clone();
        let env4 = env_temp.powi(4);
        for n in nodes {
            let t = snapshot[&n.cell];
            let g_sc = n.conductivity * cell_size;
            // Skin node: convection in, radiation out, conduction to core, and
            // convective exchange with the medium (WI 694 — cools/heats toward T_medium).
            let p_rad = n.emissivity * STEFAN_BOLTZMANN * n.exposed_area * (t.skin.powi(4) - env4);
            let p_sc = g_sc * (t.skin - t.core); // skin → core
            let p_exchange = h_exchange * n.exposed_area * (medium_temp - t.skin);
            let d_skin = (n.conv_power - p_rad - p_sc + p_exchange) / n.c_skin * dt;
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
}

/// Per-voxel work record for one [`ThermalState::step`] call (static across the
/// internal sub-steps).
struct Node {
    cell: IVec3,
    conductivity: f64,
    c_skin: f64,
    c_core: f64,
    exposed_area: f64,
    conv_power: f64,
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
}

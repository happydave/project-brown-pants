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
//! the live `-- dive` wiring + HUD/telemetry are a scene-integration follow-up.

use crate::aero::windward_faces;
use crate::breakage::{connected_components, Severed};
use crate::fluid::FluidSample;
use crate::voxel::VoxelCraft;
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

/// Fraction of a voxel's mass assigned to the fast-responding **skin** node; the
/// remainder is the slow **core**. A thin skin heats and cools quickly (the
/// re-entry surface), the core lags.
const SKIN_FRACTION: f64 = 0.1;

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
}

impl ThermalState {
    /// A fresh state with every voxel at `ambient` (both nodes).
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
        Self { temps }
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
            let mass = v.material.density * cell_volume;
            let heat_capacity = mass * th.specific_heat;
            let exp = exposure_by_cell[&v.cell];
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
                c_skin: SKIN_FRACTION * heat_capacity,
                c_core: (1.0 - SKIN_FRACTION) * heat_capacity,
                exposed_area: exp.exposed_area,
                conv_power: q_density * exp.windward_area,
                emissivity: th.emissivity,
                neighbours,
            });
        }

        // Sub-step with an adaptively-bounded stable step.
        let mut remaining = dt;
        let mut steps = 0usize;
        while remaining > 0.0 && steps < MAX_SUBSTEPS {
            let sub_dt = self.stable_substep(&nodes, cell_size).min(remaining);
            self.integrate_substep(&nodes, cell_size, env_temp, sub_dt);
            remaining -= sub_dt;
            steps += 1;
        }
    }

    /// The largest stable explicit sub-step over all nodes, given current temps.
    fn stable_substep(&self, nodes: &[Node], cell_size: f64) -> f64 {
        let mut min_dt = f64::INFINITY;
        for n in nodes {
            let t = self.temps[&n.cell];
            let g_sc = n.conductivity * cell_size; // skin↔core conductance
                                                   // skin loss derivative: conduction to core + linearised radiation.
            let rad_deriv =
                4.0 * n.emissivity * STEFAN_BOLTZMANN * n.exposed_area * t.skin.max(0.0).powi(3);
            let skin_loss = (g_sc + rad_deriv).max(1e-9);
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
    fn integrate_substep(&mut self, nodes: &[Node], cell_size: f64, env_temp: f64, dt: f64) {
        let snapshot = self.temps.clone();
        let env4 = env_temp.powi(4);
        for n in nodes {
            let t = snapshot[&n.cell];
            let g_sc = n.conductivity * cell_size;
            // Skin node: convection in, radiation out, conduction to core.
            let p_rad = n.emissivity * STEFAN_BOLTZMANN * n.exposed_area * (t.skin.powi(4) - env4);
            let p_sc = g_sc * (t.skin - t.core); // skin → core
            let d_skin = (n.conv_power - p_rad - p_sc) / n.c_skin * dt;
            // Core node: conduction from skin + exchange with occupied neighbours
            // (symmetric averaged bond conductance, computed once per call).
            let mut p_core = p_sc;
            for &(nb, g_cc) in &n.neighbours {
                let core_nb = snapshot[&nb].core;
                p_core += g_cc * (core_nb - t.core);
            }
            let d_core = p_core / n.c_core * dt;
            let entry = self.temps.get_mut(&n.cell).expect("node cell present");
            entry.skin = clamp_temp(t.skin + d_skin);
            entry.core = clamp_temp(t.core + d_core);
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
}

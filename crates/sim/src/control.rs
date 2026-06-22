//! Control devices and the autonomy-tier substrate (WI 562).
//!
//! Where autopilot lives: the inner-loop executor (`Flight Control`) is always
//! present, but what a craft can do *autonomously* is a function of its mounted
//! **control devices** and available **power** — the root of the autopilot tier
//! ladder (the Starbase chassis / MechJeb-as-a-part pattern, generalised; see
//! `tickets/docs/projects/sounding/design.md` → *Control Computers*).
//!
//! Three device functions (WI 570 — placed devices carry these as data, assembled by
//! [`assemble_control`]):
//! - A **control point** (command seat / cockpit / probe receiver) admits commands
//!   into the craft and enables the **Direct (manual, no computer)** floor. A
//!   *crewed* control point needs no electrical power; an *uncrewed* one does. A
//!   craft with no control point is **uncontrolled** (inert debris).
//! - A **control computer** carries its granted tier as data and a power draw; when
//!   powered it raises the craft to its tier (Tier 0 / `Stabilized` here; Tiers 1–2
//!   are WI 565/566).
//! - A **battery** ([`BatterySpec`]) supplies the [`ELECTRICITY`] [`Reservoir`] the
//!   computers run on.
//!
//! Available vs. effective (WI 570). The **available** tier ([`ControlSystem::
//! available_tier`]) is the installed capability — a function of mounted devices only,
//! **independent of battery charge**. The **effective** tier ([`ControlSystem::
//! effective_tier`], what the executor gates on) is the available tier while powered
//! above a small **low-power reserve**, and otherwise falls hard to the unpowered floor
//! (Direct if crewed, Uncontrolled if uncrewed) — recovering when power returns. Power
//! is thus a hard cutoff, never a gradient on the tier; [`ControlSystem::assist_offline`]
//! reports when effective is below available because of it.
//!
//! Lattice vs. flight: lattice-level controllability (used by breakage) is the
//! presence of a `DeviceKind::Command` device on the `VoxelCraft`
//! (`VoxelCraft::has_control_point`); this module's [`ControlSystem`] is the
//! richer flight-level loadout (crewed flags, computers, battery).

use crate::resource::{Consumer, Reservoir, ReservoirId, ResourceGraph, ResourceType};
use crate::voxel::VoxelCraft;
use serde::{Deserialize, Serialize};

/// Conventional resource tag for electrical power (content, like every
/// [`ResourceType`]; distinct id from propellant tags used by scenes/tests).
pub const ELECTRICITY: ResourceType = ResourceType(10);

/// Amount below which a battery reservoir is treated as empty (unpowered).
const EPS_POWER: f64 = 1e-9;

/// The autonomy tier available to a craft, ordered from least to most capable.
/// Ordering is load-bearing: capability resolution takes the **maximum** reachable
/// tier, and degradation can only move *down* this order. Extensible toward the
/// higher tiers (canned/tunable/programmable, WI 565/566/560).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlTier {
    /// No control point (or an unpowered, uncrewed one): the craft accepts no
    /// commands — inert debris under the active gear.
    Uncontrolled,
    /// Manual command only, no stabilization: raw throttle/gimbal/attitude routed
    /// through the executor. The skill floor and the degraded fallback.
    Direct,
    /// Tier 0: a powered command computer enables SAS-style stabilization on top of
    /// manual control. (SAS hardware/engagement specifics are WI 564.)
    Stabilized,
    /// Tier 1: a powered autopilot computer offers **canned autopilots** (orbital-frame
    /// attitude holds, gravity-turn ascent, …) on top of stabilization (WI 565).
    Canned,
    /// Tier 2: a powered tuning computer additionally allows **live tuning** of a
    /// controller's parameters (e.g. SAS PD gains) on top of canned autopilots (WI 566).
    Tunable,
}

impl ControlTier {
    /// Whether manual actuation commands take effect at this tier.
    pub fn allows_manual(self) -> bool {
        self >= ControlTier::Direct
    }

    /// Whether stabilization (SAS) is available at this tier.
    pub fn allows_stabilization(self) -> bool {
        self >= ControlTier::Stabilized
    }

    /// Whether canned autopilots (Tier 1) are available at this tier.
    pub fn allows_canned(self) -> bool {
        self >= ControlTier::Canned
    }

    /// Whether live controller tuning (Tier 2) is available at this tier.
    pub fn allows_tuning(self) -> bool {
        self >= ControlTier::Tunable
    }
}

/// A control point: the device that admits commands into a craft. A *crewed* point
/// (cockpit / crewed pod) needs no power; an *uncrewed* one (probe / remote
/// receiver) requires electrical power to admit commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlPoint {
    /// Crewed control points admit commands without electrical power.
    pub crewed: bool,
}

impl ControlPoint {
    /// A crewed control point (cockpit / crewed pod).
    pub fn crewed() -> Self {
        Self { crewed: true }
    }

    /// An uncrewed control point (probe core / remote receiver) — needs power.
    pub fn uncrewed() -> Self {
        Self { crewed: false }
    }
}

/// A control computer: grants an autonomy [`ControlTier`] while running, drawing
/// `power_draw` units of [`ELECTRICITY`] per unit time from the craft's battery. A
/// computer with `power_draw <= 0` is **self-powered** (always running, no battery
/// required) — the convenient model for craft where electrical power is not being
/// simulated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlComputer {
    /// The tier this computer grants while powered.
    pub grants: ControlTier,
    /// Electrical draw per unit time while running.
    pub power_draw: f64,
}

impl ControlComputer {
    /// A Tier-0 (stabilization-capable) command computer with the given draw.
    pub fn command_core(power_draw: f64) -> Self {
        Self {
            grants: ControlTier::Stabilized,
            power_draw,
        }
    }

    /// A Tier-1 (canned-autopilot-capable) computer with the given draw.
    pub fn autopilot_computer(power_draw: f64) -> Self {
        Self {
            grants: ControlTier::Canned,
            power_draw,
        }
    }

    /// A Tier-2 (live-tuning-capable) computer with the given draw.
    pub fn tuning_computer(power_draw: f64) -> Self {
        Self {
            grants: ControlTier::Tunable,
            power_draw,
        }
    }
}

/// The electrical store a battery device provides: an [`ELECTRICITY`] reservoir
/// (charge / capacity) injected into the craft's resource graph at assembly.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatterySpec {
    /// Stored charge at assembly, units of [`ELECTRICITY`].
    pub charge: f64,
    /// Reservoir capacity, units of [`ELECTRICITY`].
    pub capacity: f64,
}

impl BatterySpec {
    /// A battery filled to capacity.
    pub fn full(capacity: f64) -> Self {
        Self {
            charge: capacity,
            capacity,
        }
    }
}

/// The flight function a placed device performs (WI 570). This is the data-shaped,
/// catalog-ready bridge between a lattice [`crate::voxel::Device`] and its control
/// behaviour: assembly ([`assemble_control`]) reads these to build a [`ControlSystem`].
/// A device without a function is structural / inert mass only (the pre-570 default).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceFunction {
    /// A control point (cockpit / probe core) — admits commands; the Direct floor.
    ControlPoint(ControlPoint),
    /// A control computer — grants its tier while powered.
    Computer(ControlComputer),
    /// A battery — an electricity reservoir powering the computers.
    Battery(BatterySpec),
}

/// A craft's control loadout: its control points, control computers, and which
/// resource-graph reservoir is its electrical battery. The single source of the
/// craft's available [`ControlTier`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ControlSystem {
    /// Mounted control points.
    pub points: Vec<ControlPoint>,
    /// Mounted control computers.
    pub computers: Vec<ControlComputer>,
    /// The electrical battery reservoir in the craft's resource graph, if any.
    pub battery: Option<ReservoirId>,
    /// Critically-low charge reserve (units of [`ELECTRICITY`]): powered assistance
    /// **fails hard** when battery charge drops to or below this reserve, not only at
    /// exactly zero (WI 570). Charge-independent capability (the *available* tier) is
    /// unaffected. Defaulted to 0 (legacy ≈empty cutoff) so pre-570 systems are
    /// unchanged; a positive value models a real operating reserve. Clamped
    /// non-negative and bounded by the battery's capacity at evaluation.
    #[serde(default)]
    pub low_power_reserve: f64,
    /// Player-selected control-tier cap (WI 571): the craft may be operated **below**
    /// its available tier (e.g. fly Direct with assist off to conserve power or for
    /// skill). `resolve` returns `min(effective_tier, selected)`. `None` ⇒ no cap (full
    /// available). Downshift is always permitted; a selection above capability is
    /// harmless (the `min` ignores it). Does **not** affect `available_tier` or
    /// `assist_offline` (the latter stays a low-power indicator, not a downshift one).
    #[serde(default)]
    pub selected: Option<ControlTier>,
}

impl ControlSystem {
    /// A bare manual craft: one crewed control point, no computer, no battery →
    /// resolves to [`ControlTier::Direct`].
    pub fn crewed_manual() -> Self {
        Self {
            points: vec![ControlPoint::crewed()],
            ..Default::default()
        }
    }

    /// A crewed, stabilized craft with a **self-powered** command core (no battery
    /// needed) → resolves to [`ControlTier::Stabilized`]. The convenient default for
    /// craft where electrical power is not being modelled.
    pub fn crewed_stabilized() -> Self {
        Self {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::command_core(0.0)],
            ..Default::default()
        }
    }

    /// A crewed craft with a **self-powered** Tier-1 autopilot computer → resolves to
    /// [`ControlTier::Canned`] (canned autopilots available). Convenience for craft
    /// where power is not modelled.
    pub fn crewed_canned() -> Self {
        Self {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::autopilot_computer(0.0)],
            ..Default::default()
        }
    }

    /// A crewed craft with a **self-powered** Tier-2 tuning computer → resolves to
    /// [`ControlTier::Tunable`] (live controller tuning available). Convenience for
    /// craft where power is not modelled.
    pub fn crewed_tunable() -> Self {
        Self {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::tuning_computer(0.0)],
            ..Default::default()
        }
    }

    /// Whether the craft currently has electrical power **above its low-power
    /// reserve** (WI 570): battery present and charge strictly above the effective
    /// cutoff. The cutoff is `low_power_reserve`, clamped non-negative and bounded by
    /// the reservoir capacity, and never below `EPS_POWER` (so an unconfigured reserve
    /// keeps the pre-570 ≈empty behaviour). Crewed control points do not depend on this.
    pub fn is_powered(&self, graph: &ResourceGraph) -> bool {
        match self.battery {
            Some(id) => graph.reservoirs.get(id.0).is_some_and(|r| {
                let cutoff = self.low_power_reserve.clamp(0.0, r.capacity).max(EPS_POWER);
                r.amount > cutoff
            }),
            None => false,
        }
    }

    /// The craft's **available** (installed) control tier — a function of mounted
    /// devices only, **independent of battery charge** (WI 570). A Stabilized craft
    /// stays Stabilized whatever the charge; power gates only whether that capability
    /// is currently *running* (see [`Self::effective_tier`]). `Uncontrolled` only when
    /// there is no control point at all (no installed control hardware).
    pub fn available_tier(&self) -> ControlTier {
        if self.points.is_empty() {
            return ControlTier::Uncontrolled;
        }
        let mut tier = ControlTier::Direct;
        for c in &self.computers {
            if c.grants > tier {
                tier = c.grants;
            }
        }
        tier
    }

    /// The craft's **effective** control tier — what is currently operating given
    /// power (WI 570). Equals [`Self::available_tier`] when powered above the reserve;
    /// otherwise it falls hard to the unpowered floor: Direct for a crewed craft (the
    /// crew still flies), Uncontrolled for an uncrewed one. Pure and deterministic;
    /// degradation is the same function over reduced state, so power/structural loss
    /// can only lower the result.
    pub fn effective_tier(&self, graph: &ResourceGraph) -> ControlTier {
        if self.points.is_empty() {
            return ControlTier::Uncontrolled;
        }
        let powered = self.is_powered(graph);
        let any_crewed = self.points.iter().any(|p| p.crewed);

        // Controllable at all? A crewed point always admits commands; an uncrewed
        // point needs power. Otherwise the craft is inert.
        if !any_crewed && !powered {
            return ControlTier::Uncontrolled;
        }

        // Base controllability is Direct; running computers raise it to their tier.
        // A computer runs when self-powered (`power_draw <= 0`) or the craft is powered.
        let mut tier = ControlTier::Direct;
        for c in &self.computers {
            let running = c.power_draw <= 0.0 || powered;
            if running && c.grants > tier {
                tier = c.grants;
            }
        }
        tier
    }

    /// Resolve the craft's **operating** control tier the inner-loop executor gates on:
    /// the power-resolved [`Self::effective_tier`] capped by the player-selected tier
    /// (WI 571) — `min(effective_tier, selected)`. With no selection this is exactly the
    /// effective tier (name retained from WI 562).
    pub fn resolve(&self, graph: &ResourceGraph) -> ControlTier {
        let eff = self.effective_tier(graph);
        match self.selected {
            Some(sel) => eff.min(sel),
            None => eff,
        }
    }

    /// Whether powered assistance is **offline because of low power** (WI 570): the
    /// effective tier is below the installed/available tier due to the power gate.
    /// Lets a HUD show "assist offline (low power)" against the unchanged available
    /// tier rather than relabelling it.
    pub fn assist_offline(&self, graph: &ResourceGraph) -> bool {
        self.effective_tier(graph) < self.available_tier()
    }

    /// The standing electricity [`Consumer`] for this control system's computers —
    /// added to the craft's [`ResourceGraph`] at assembly so power drains over time.
    /// `None` if there is no battery or no draw.
    pub fn power_consumer(&self) -> Option<Consumer> {
        let battery = self.battery?;
        let rate: f64 = self.computers.iter().map(|c| c.power_draw).sum();
        (rate > 0.0).then_some(Consumer {
            from: battery,
            rate,
        })
    }
}

/// Assemble a [`ControlSystem`] from a craft's **placed devices** (WI 570): the
/// data-shaped bridge that replaces per-scene hand-assembly. Walks the craft's
/// devices, and for each carrying a [`DeviceFunction`]:
/// - **Battery** → push an [`ELECTRICITY`] [`Reservoir`] into `graph` and wire it as
///   the system battery (the first battery is the electrical source; multi-bank power
///   is out of scope, WI 570).
/// - **ControlPoint** / **Computer** → collect into the system's points / computers.
///
/// Finally pushes the standing electricity [`Consumer`] for the computers' draw into
/// `graph`, so power drains over time exactly as a hand-built system would. A device
/// with no [`DeviceFunction`] is ignored here (structural / inert mass only).
pub fn assemble_control(craft: &VoxelCraft, graph: &mut ResourceGraph) -> ControlSystem {
    let mut sys = ControlSystem::default();
    for d in &craft.devices {
        match d.function {
            Some(DeviceFunction::ControlPoint(p)) => sys.points.push(p),
            Some(DeviceFunction::Computer(c)) => sys.computers.push(c),
            Some(DeviceFunction::Battery(b)) => {
                let id = ReservoirId(graph.reservoirs.len());
                graph
                    .reservoirs
                    .push(Reservoir::new(ELECTRICITY, b.charge, b.capacity));
                if sys.battery.is_none() {
                    sys.battery = Some(id);
                }
            }
            None => {}
        }
    }
    if let Some(c) = sys.power_consumer() {
        graph.consumers.push(c);
    }
    sys
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::Reservoir;

    fn graph_with_battery(amount: f64) -> (ResourceGraph, ReservoirId) {
        let g = ResourceGraph {
            reservoirs: vec![Reservoir::new(ELECTRICITY, amount, 100.0)],
            ..Default::default()
        };
        (g, ReservoirId(0))
    }

    #[test]
    fn no_control_point_is_uncontrolled() {
        let sys = ControlSystem::default();
        let g = ResourceGraph::default();
        assert_eq!(sys.resolve(&g), ControlTier::Uncontrolled);
    }

    #[test]
    fn crewed_point_resolves_direct_without_power() {
        let sys = ControlSystem::crewed_manual();
        let g = ResourceGraph::default();
        assert_eq!(sys.resolve(&g), ControlTier::Direct);
        assert!(ControlTier::Direct.allows_manual());
        assert!(!ControlTier::Direct.allows_stabilization());
    }

    #[test]
    fn uncrewed_point_needs_power() {
        let (g_full, bat) = graph_with_battery(50.0);
        let sys = ControlSystem {
            points: vec![ControlPoint::uncrewed()],
            battery: Some(bat),
            ..Default::default()
        };
        assert_eq!(sys.resolve(&g_full), ControlTier::Direct);

        let (g_empty, _) = graph_with_battery(0.0);
        assert_eq!(sys.resolve(&g_empty), ControlTier::Uncontrolled);
    }

    #[test]
    fn self_powered_computer_needs_no_battery() {
        // A crewed craft with a self-powered (zero-draw) core resolves Stabilized
        // without any battery — the convenient default for unmodelled power.
        let sys = ControlSystem::crewed_stabilized();
        let g = ResourceGraph::default();
        assert_eq!(sys.resolve(&g), ControlTier::Stabilized);
        assert!(
            sys.power_consumer().is_none(),
            "zero-draw core adds no consumer"
        );
    }

    #[test]
    fn powered_computer_grants_stabilized() {
        let (g_full, bat) = graph_with_battery(50.0);
        let sys = ControlSystem {
            points: vec![ControlPoint::uncrewed()],
            computers: vec![ControlComputer::command_core(1.0)],
            battery: Some(bat),
            ..Default::default()
        };
        assert_eq!(sys.resolve(&g_full), ControlTier::Stabilized);
    }

    #[test]
    fn autopilot_computer_grants_canned_tier() {
        let sys = ControlSystem::crewed_canned();
        let g = ResourceGraph::default();
        let tier = sys.resolve(&g);
        assert_eq!(tier, ControlTier::Canned);
        assert!(tier.allows_canned() && tier.allows_stabilization() && tier.allows_manual());
        // A Tier-0 command core does not grant canned autopilots.
        assert!(!ControlSystem::crewed_stabilized()
            .resolve(&g)
            .allows_canned());
    }

    #[test]
    fn tuning_computer_grants_tunable_tier() {
        let g = ResourceGraph::default();
        let tier = ControlSystem::crewed_tunable().resolve(&g);
        assert_eq!(tier, ControlTier::Tunable);
        assert!(tier.allows_tuning() && tier.allows_canned() && tier.allows_stabilization());
        // A Tier-1 autopilot computer does not grant tuning.
        assert!(!ControlSystem::crewed_canned().resolve(&g).allows_tuning());
    }

    #[test]
    fn power_loss_degrades_down_the_ladder() {
        // Crewed craft with a computer: powered → Stabilized; unpowered → Direct
        // (crew still flies). Never rises with loss (monotonic).
        let (g_full, bat) = graph_with_battery(50.0);
        let sys = ControlSystem {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::command_core(1.0)],
            battery: Some(bat),
            ..Default::default()
        };
        let powered = sys.resolve(&g_full);
        let (g_empty, _) = graph_with_battery(0.0);
        let unpowered = sys.resolve(&g_empty);
        assert_eq!(powered, ControlTier::Stabilized);
        assert_eq!(unpowered, ControlTier::Direct);
        assert!(unpowered <= powered, "loss must not raise the tier");
    }

    #[test]
    fn uncrewed_computer_craft_goes_uncontrolled_on_power_loss() {
        let sys = ControlSystem {
            points: vec![ControlPoint::uncrewed()],
            computers: vec![ControlComputer::command_core(1.0)],
            battery: Some(ReservoirId(0)),
            ..Default::default()
        };
        let (g_empty, _) = graph_with_battery(0.0);
        assert_eq!(sys.resolve(&g_empty), ControlTier::Uncontrolled);
    }

    #[test]
    fn battery_drains_analytically_then_tier_drops() {
        // An uncrewed computer craft: integrate the graph to drain the battery via
        // the standing consumer; once empty the tier drops to Uncontrolled.
        let (mut g, bat) = graph_with_battery(10.0);
        let sys = ControlSystem {
            points: vec![ControlPoint::uncrewed()],
            computers: vec![ControlComputer::command_core(2.0)],
            battery: Some(bat),
            ..Default::default()
        };
        g.consumers.push(sys.power_consumer().expect("consumer"));
        assert_eq!(sys.resolve(&g), ControlTier::Stabilized);
        // 10 units at 2/s drains in 5 s; integrate past that.
        g.integrate(10.0);
        assert!(
            g.reservoirs[0].amount <= EPS_POWER,
            "battery should be empty"
        );
        assert_eq!(sys.resolve(&g), ControlTier::Uncontrolled);
    }

    // --- WI 570: assembly + available-vs-effective power model ---

    #[test]
    fn assemble_builds_control_system_from_placed_devices() {
        use crate::voxel::{Device, VoxelCraft};
        use glam::IVec3;
        let mut craft = VoxelCraft::new(1.0);
        craft
            .devices
            .push(Device::control_point(IVec3::ZERO, 80.0, true));
        craft.devices.push(Device::computer(
            IVec3::new(0, 1, 0),
            20.0,
            ControlComputer::command_core(0.5),
        ));
        craft.devices.push(Device::battery(
            IVec3::new(0, 2, 0),
            40.0,
            BatterySpec::full(100.0),
        ));
        // A structural-only device is ignored by assembly (mass only).
        craft.devices.push(Device::structural(
            IVec3::new(0, 3, 0),
            5.0,
            crate::voxel::DeviceKind::Engine,
        ));

        let mut graph = ResourceGraph::default();
        let sys = assemble_control(&craft, &mut graph);

        assert_eq!(sys.points.len(), 1);
        assert_eq!(sys.computers.len(), 1);
        assert!(sys.battery.is_some(), "battery wired");
        assert_eq!(
            graph.reservoirs.len(),
            1,
            "battery injected as an electricity reservoir"
        );
        assert_eq!(graph.reservoirs[0].resource, ELECTRICITY);
        assert_eq!(graph.consumers.len(), 1, "standing power consumer added");
        assert_eq!(sys.available_tier(), ControlTier::Stabilized);
        assert_eq!(sys.effective_tier(&graph), ControlTier::Stabilized);
        assert!(!sys.assist_offline(&graph));
    }

    #[test]
    fn assemble_with_no_functional_devices_is_uncontrolled() {
        use crate::voxel::{Device, DeviceKind, VoxelCraft};
        use glam::IVec3;
        let mut craft = VoxelCraft::new(1.0);
        craft
            .devices
            .push(Device::structural(IVec3::ZERO, 10.0, DeviceKind::Tank));
        let mut graph = ResourceGraph::default();
        let sys = assemble_control(&craft, &mut graph);
        assert!(sys.points.is_empty() && sys.computers.is_empty());
        assert!(sys.battery.is_none());
        assert!(graph.reservoirs.is_empty() && graph.consumers.is_empty());
        assert_eq!(sys.available_tier(), ControlTier::Uncontrolled);
        assert_eq!(sys.effective_tier(&graph), ControlTier::Uncontrolled);
    }

    #[test]
    fn available_tier_is_charge_independent_while_effective_drops() {
        // A crewed, stabilized craft on a battery: available stays Stabilized as the
        // battery drains; effective falls to Direct (crew still flies). Then recovers.
        let (mut g, bat) = graph_with_battery(50.0);
        let sys = ControlSystem {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::command_core(1.0)],
            battery: Some(bat),
            ..Default::default()
        };
        assert_eq!(sys.available_tier(), ControlTier::Stabilized);
        assert_eq!(sys.effective_tier(&g), ControlTier::Stabilized);
        assert!(!sys.assist_offline(&g));

        // Drain below the cutoff: available unchanged, effective drops, assist offline.
        g.reservoirs[bat.0].amount = 0.0;
        assert_eq!(
            sys.available_tier(),
            ControlTier::Stabilized,
            "installed capability is charge-independent"
        );
        assert_eq!(sys.effective_tier(&g), ControlTier::Direct);
        assert!(sys.assist_offline(&g), "assist offline due to low power");

        // Recharge: effective recovers to available.
        g.reservoirs[bat.0].amount = 50.0;
        assert_eq!(sys.effective_tier(&g), ControlTier::Stabilized);
        assert!(!sys.assist_offline(&g));
    }

    #[test]
    fn low_power_reserve_cuts_assist_before_zero() {
        // A configured reserve makes assist fail at a small reserve, not only at zero.
        let (mut g, bat) = graph_with_battery(100.0);
        let reserve = 5.0;
        let sys = ControlSystem {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::command_core(1.0)],
            battery: Some(bat),
            low_power_reserve: reserve,
            ..Default::default()
        };
        // Just above the reserve → powered.
        g.reservoirs[bat.0].amount = reserve + 0.5;
        assert_eq!(sys.effective_tier(&g), ControlTier::Stabilized);
        // At the reserve → not powered (strict cutoff), assist offline, but charge > 0.
        g.reservoirs[bat.0].amount = reserve;
        assert_eq!(sys.effective_tier(&g), ControlTier::Direct);
        assert!(g.reservoirs[bat.0].amount > 0.0, "fails before empty");
        assert!(sys.assist_offline(&g));
    }

    // --- WI 571: player-selectable tier (downshift cap) ---

    #[test]
    fn selected_tier_caps_the_operating_tier() {
        // A powered, crewed Stabilized craft. Selecting Direct caps the operating tier;
        // selecting above capability is a no-op; clearing restores full available.
        let (g, bat) = graph_with_battery(50.0);
        let mut sys = ControlSystem {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::command_core(1.0)],
            battery: Some(bat),
            ..Default::default()
        };
        assert_eq!(sys.available_tier(), ControlTier::Stabilized);
        assert_eq!(
            sys.resolve(&g),
            ControlTier::Stabilized,
            "no selection ⇒ full"
        );

        sys.selected = Some(ControlTier::Direct);
        assert_eq!(
            sys.resolve(&g),
            ControlTier::Direct,
            "downshift caps to Direct"
        );
        assert_eq!(
            sys.available_tier(),
            ControlTier::Stabilized,
            "available unchanged by selection"
        );
        assert!(
            !sys.assist_offline(&g),
            "a deliberate downshift is not low-power assist-offline"
        );

        sys.selected = Some(ControlTier::Tunable);
        assert_eq!(
            sys.resolve(&g),
            ControlTier::Stabilized,
            "selection cannot exceed capability"
        );

        sys.selected = None;
        assert_eq!(
            sys.resolve(&g),
            ControlTier::Stabilized,
            "cleared ⇒ full again"
        );
    }

    #[test]
    fn selection_composes_with_low_power_floor() {
        // Both caps apply: when power has already floored the tier, a higher selection
        // cannot lift it; a lower selection still caps further.
        let (mut g, bat) = graph_with_battery(50.0);
        let mut sys = ControlSystem {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::command_core(1.0)],
            battery: Some(bat),
            ..Default::default()
        };
        g.reservoirs[bat.0].amount = 0.0; // unpowered → effective floor Direct (crewed)
        sys.selected = Some(ControlTier::Tunable);
        assert_eq!(
            sys.resolve(&g),
            ControlTier::Direct,
            "selecting high cannot lift the power floor"
        );
    }
}

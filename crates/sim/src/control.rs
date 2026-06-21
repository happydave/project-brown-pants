//! Control devices and the autonomy-tier substrate (WI 562).
//!
//! Where autopilot lives: the inner-loop executor (`Flight Control`) is always
//! present, but what a craft can do *autonomously* is a function of its mounted
//! **control devices** and available **power** — the root of the autopilot tier
//! ladder (the Starbase chassis / MechJeb-as-a-part pattern, generalised; see
//! `tickets/docs/projects/sounding/design.md` → *Control Computers*).
//!
//! Two devices:
//! - A **control point** (command seat / cockpit / probe receiver) admits commands
//!   into the craft and enables the **Direct (manual, no computer)** floor. A
//!   *crewed* control point needs no electrical power; an *uncrewed* one does. A
//!   craft with no control point is **uncontrolled** (inert debris).
//! - A **control computer** carries its granted tier as data and a power draw; when
//!   powered it raises the craft to its tier (Tier 0 / `Stabilized` here; Tiers 1–2
//!   are WI 565/566).
//!
//! Power is a real resource-graph element: a battery [`Reservoir`] of [`ELECTRICITY`]
//! drained by a [`Consumer`]. Losing power **degrades** a craft down the ladder
//! (a crewed craft to Direct, an uncrewed one to Uncontrolled); it never raises it.
//!
//! Lattice vs. flight: lattice-level controllability (used by breakage) is the
//! presence of a `DeviceKind::Command` device on the `VoxelCraft`
//! (`VoxelCraft::has_control_point`); this module's [`ControlSystem`] is the
//! richer flight-level loadout (crewed flags, computers, battery).

use crate::resource::{Consumer, ReservoirId, ResourceGraph, ResourceType};
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
            battery: None,
        }
    }

    /// A crewed craft with a **self-powered** Tier-1 autopilot computer → resolves to
    /// [`ControlTier::Canned`] (canned autopilots available). Convenience for craft
    /// where power is not modelled.
    pub fn crewed_canned() -> Self {
        Self {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::autopilot_computer(0.0)],
            battery: None,
        }
    }

    /// A crewed craft with a **self-powered** Tier-2 tuning computer → resolves to
    /// [`ControlTier::Tunable`] (live controller tuning available). Convenience for
    /// craft where power is not modelled.
    pub fn crewed_tunable() -> Self {
        Self {
            points: vec![ControlPoint::crewed()],
            computers: vec![ControlComputer::tuning_computer(0.0)],
            battery: None,
        }
    }

    /// Whether the craft currently has electrical power (battery present and
    /// non-empty). Crewed control points do not depend on this.
    pub fn is_powered(&self, graph: &ResourceGraph) -> bool {
        match self.battery {
            Some(id) => graph
                .reservoirs
                .get(id.0)
                .is_some_and(|r| r.amount > EPS_POWER),
            None => false,
        }
    }

    /// Resolve the craft's available control tier from its mounted devices and
    /// current power. Pure and deterministic; degradation is the same function over
    /// reduced state, so power/structural loss can only lower the result.
    pub fn resolve(&self, graph: &ResourceGraph) -> ControlTier {
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
}

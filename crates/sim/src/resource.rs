//! The resource / converter system (Toy 7, WI 507).
//!
//! The design's fourth load-bearing decision: *one generic, data-driven graph*
//! of reservoirs, converters, consumers, and conduits, of which mining, ISRU,
//! life support, and gas/fluid flow are all instances. This module is that graph
//! plus its **analytic on-rails catch-up integrator**.
//!
//! ## Why this satisfies the warp filter
//!
//! Every flow in the graph (mining draw, conversion throughput, consumer draw,
//! conduit flow) is **piecewise-constant in time**: it holds a constant rate
//! until a reservoir hits empty or full, or a resource-node depletes. Between
//! such *events*, every reservoir's quantity is *linear* in time. So
//! [`ResourceGraph::integrate`] advances by event-stepping: compute the effective
//! rates, find the earliest reservoir-saturation or node-depletion breakpoint,
//! advance all quantities linearly to it in closed form, recompute, repeat. The
//! number of breakpoints is bounded by the graph's saturation transitions — **not
//! by the elapsed interval or warp factor** — so advancing a thousand years with
//! no new saturation costs the same as advancing a second. That is "advancing
//! time is arithmetic, not integration".
//!
//! Headless and data-driven: a new resource or recipe is new constants, not new
//! control flow. Standalone — the graph carries its own last-integration time;
//! embedding it (with that timestamp) into the durable world-save is the hand-off
//! work (WI 508). The `persist.rs` `resources` container stays reserved.

use serde::{Deserialize, Serialize};

/// Data-driven identity of a resource (fuel, ore, oxygen, water, food, …). The
/// meaning is content; the system treats it only as an opaque tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceType(pub u32);

/// Handle to a [`Reservoir`] within a [`ResourceGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservoirId(pub usize);

/// Handle to a [`ResourceNode`] within a [`ResourceGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeId(pub usize);

/// A bounded store of one resource type. Its quantity always stays in
/// `[0, capacity]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reservoir {
    /// What this reservoir holds.
    pub resource: ResourceType,
    /// Current quantity, in `[0, capacity]`.
    pub amount: f64,
    /// Maximum quantity.
    pub capacity: f64,
    /// Mass of one unit of `amount`, kg — the reservoir's explicit mass model
    /// (WI 810). Mass folds (`Propulsion::wet_mass`) weigh a reservoir at
    /// `amount × mass_per_unit`, so a mass-denominated store (propellant, kg)
    /// uses `1` and a non-material store (electric charge) uses `0`. Absent in
    /// pre-810 serialized graphs, so it serde-defaults to the legacy `1`.
    #[serde(default = "default_mass_per_unit")]
    pub mass_per_unit: f64,
}

fn default_mass_per_unit() -> f64 {
    1.0
}

impl Reservoir {
    /// A reservoir of `resource` with `capacity`, holding `amount`, with the
    /// default mass model (1 kg per unit — the kg-denominated convention).
    pub fn new(resource: ResourceType, amount: f64, capacity: f64) -> Self {
        Self {
            resource,
            amount,
            capacity,
            mass_per_unit: 1.0,
        }
    }

    /// A reservoir whose contents carry no mass (electric charge and other
    /// non-material stores): `mass_per_unit = 0`.
    pub fn massless(resource: ResourceType, amount: f64, capacity: f64) -> Self {
        Self {
            resource,
            amount,
            capacity,
            mass_per_unit: 0.0,
        }
    }

    /// Headroom before the reservoir is full.
    fn headroom(&self) -> f64 {
        (self.capacity - self.amount).max(0.0)
    }

    fn is_empty(&self) -> bool {
        self.amount <= EPS_Q
    }

    fn is_full(&self) -> bool {
        self.amount >= self.capacity - EPS_Q
    }
}

/// An independent, depletable source of a resource — the mining target. Distinct
/// from a [`Reservoir`]: it is not owned by any craft, has no inflow, and stops
/// supplying once depleted.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceNode {
    /// What this node yields.
    pub resource: ResourceType,
    /// Quantity remaining (`>= 0`); a depleted node yields nothing further.
    pub remaining: f64,
}

impl ResourceNode {
    /// A node of `resource` with `remaining` quantity.
    pub fn new(resource: ResourceType, remaining: f64) -> Self {
        Self {
            resource,
            remaining,
        }
    }

    fn is_depleted(&self) -> bool {
        self.remaining <= EPS_Q
    }
}

/// A source a converter draws from: either a craft reservoir or an independent
/// resource-node (mining).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    /// Draw from a reservoir.
    Reservoir(ReservoirId),
    /// Draw from an independent resource-node (an extractor / miner).
    Node(NodeId),
}

/// One input or output port of a converter: a target and a nominal rate
/// (quantity per unit time at full throttle).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Flow<T> {
    /// The endpoint this port draws from or fills.
    pub target: T,
    /// Quantity per unit time at full (unthrottled) throughput.
    pub rate: f64,
}

impl<T> Flow<T> {
    /// A port on `target` at `rate`.
    pub fn new(target: T, rate: f64) -> Self {
        Self { target, rate }
    }
}

/// Transforms inputs into outputs at defined rates. A miner/extractor is just a
/// converter whose single input is a [`Source::Node`]; a refinery draws a
/// reservoir and fills another. Throughput throttles to the scarcest input and
/// the tightest output headroom (Leontief: all inputs are required).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Converter {
    /// Inputs drawn (each from a reservoir or a node).
    pub inputs: Vec<Flow<Source>>,
    /// Outputs produced (each into a reservoir).
    pub outputs: Vec<Flow<ReservoirId>>,
}

/// Draws a resource from a reservoir at a rate and sinks it (out of the graph):
/// an engine burning fuel, crew metabolising oxygen. Life support is a set of
/// these plus a recycler [`Converter`] — content, not code.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Consumer {
    /// Reservoir drawn.
    pub from: ReservoirId,
    /// Quantity per unit time at full draw.
    pub rate: f64,
}

/// Moves a resource between two reservoirs at a flow rate (a pipe). The design
/// notes a hull breach is simply a large conduit to the outside medium; that
/// (and the flooding → mass coupling) is the dive, not this toy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Conduit {
    /// Source reservoir.
    pub from: ReservoirId,
    /// Destination reservoir.
    pub to: ReservoirId,
    /// Quantity per unit time at full flow.
    pub rate: f64,
}

/// Quantity threshold below which a reservoir/node is treated as at a boundary.
const EPS_Q: f64 = 1e-9;
/// Time threshold below which a step is treated as non-advancing.
const EPS_T: f64 = 1e-9;

/// One generic, data-driven resource graph with analytic catch-up integration.
///
/// Build it from data, then [`integrate`](Self::integrate) it forward to a target
/// time; the graph carries its own last-integration `time` (the on-rails
/// posture). Invariants held after every catch-up: every reservoir stays in
/// `[0, capacity]`, every node stays `>= 0`, and no quantity is created or
/// destroyed except by an explicit node/converter/consumer.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceGraph {
    /// Bounded resource stores.
    pub reservoirs: Vec<Reservoir>,
    /// Independent depletable sources (mining targets).
    pub nodes: Vec<ResourceNode>,
    /// Input→output transformers (extractors, refineries, recyclers).
    pub converters: Vec<Converter>,
    /// Rate sinks (engines, crew metabolism).
    pub consumers: Vec<Consumer>,
    /// Inter-reservoir flows (pipes).
    pub conduits: Vec<Conduit>,
    /// Last-integration time (the on-rails timestamp).
    pub time: f64,
}

/// The effective, throttle-resolved net rates for one event-step: how fast each
/// reservoir and node is changing while no boundary is crossed.
struct Rates {
    reservoir_net: Vec<f64>,
    node_net: Vec<f64>,
}

impl ResourceGraph {
    /// Advance the graph analytically from its current `time` to `target` by
    /// event-stepping, and return the number of event-steps taken (bounded by the
    /// graph's saturation transitions, not by `target - time` — the warp-filter
    /// property). A no-op if `target <= time`.
    pub fn integrate(&mut self, target: f64) -> usize {
        // Safety cap: the loop is bounded by saturation events, but guard against
        // pathological non-progress (degenerate/cyclic topologies) regardless.
        let max_events = 4 * (self.reservoirs.len() + self.nodes.len()) + 16;
        let mut steps = 0;
        while self.time + EPS_T < target {
            let rates = self.effective_rates();
            let remaining = target - self.time;
            let dt = match self.next_event_dt(&rates) {
                Some(e) => e.min(remaining),
                None => remaining,
            };
            if dt <= EPS_T || steps >= max_events {
                // No event reachable (or safety cap): nothing more will change
                // shape before `target`, so jump straight there.
                self.apply(remaining, &rates);
                self.time = target;
                steps += 1;
                break;
            }
            self.apply(dt, &rates);
            self.time += dt;
            steps += 1;
        }
        steps
    }

    /// Reference integrator: advance by fixed steps of at most `dt`, recomputing
    /// rates each step. Tick-by-tick rather than analytic — the baseline the
    /// catch-up [`integrate`](Self::integrate) is validated against. Test-only by
    /// intent (it does `O((target - time) / dt)` work — the cost the warp filter
    /// exists to avoid).
    #[cfg(test)]
    fn integrate_fixed(&mut self, target: f64, dt: f64) {
        while self.time + EPS_T < target {
            let rates = self.effective_rates();
            let h = dt.min(target - self.time);
            self.apply(h, &rates);
            self.time += h;
        }
    }

    /// Resolve effective net rates under the instantaneous boundary constraints:
    /// an empty reservoir's outflow cannot exceed its inflow, a full reservoir's
    /// inflow cannot exceed its outflow, and a depleted node cannot be drawn.
    /// Non-boundary reservoirs impose no instantaneous constraint. Throttles are
    /// reduced by proportional rationing until all boundaries hold (monotonic
    /// decrease → converges; linear chains settle in one or two sweeps).
    fn effective_rates(&self) -> Rates {
        let mut conv = vec![1.0_f64; self.converters.len()];
        let mut cons = vec![1.0_f64; self.consumers.len()];
        let mut cond = vec![1.0_f64; self.conduits.len()];

        let max_sweeps = 2 * (self.converters.len() + self.consumers.len() + self.conduits.len())
            + self.reservoirs.len()
            + self.nodes.len()
            + 4;

        for _ in 0..max_sweeps {
            let mut changed = false;

            // Depleted nodes: their drawing converters cannot run (Leontief —
            // a missing input stops the converter entirely).
            for (ni, node) in self.nodes.iter().enumerate() {
                if !node.is_depleted() {
                    continue;
                }
                for (ci, c) in self.converters.iter().enumerate() {
                    if conv[ci] > 0.0
                        && c.inputs
                            .iter()
                            .any(|f| f.target == Source::Node(NodeId(ni)) && f.rate > 0.0)
                    {
                        conv[ci] = 0.0;
                        changed = true;
                    }
                }
            }

            // Reservoir boundaries.
            for (ri, res) in self.reservoirs.iter().enumerate() {
                let id = ReservoirId(ri);
                let (inflow, outflow) = self.reservoir_flows(id, &conv, &cons, &cond);
                if res.is_empty() && outflow > inflow + EPS_Q {
                    // Ration the drainers down so outflow == inflow.
                    let s = if outflow > 0.0 { inflow / outflow } else { 0.0 };
                    if self.scale_drainers(id, s, &mut conv, &mut cons, &mut cond) {
                        changed = true;
                    }
                } else if res.is_full() && inflow > outflow + EPS_Q {
                    // Ration the fillers down so inflow == outflow.
                    let s = if inflow > 0.0 { outflow / inflow } else { 0.0 };
                    if self.scale_fillers(id, s, &mut conv, &mut cond) {
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        self.net_rates(&conv, &cons, &cond)
    }

    /// Total inflow and outflow rate at a reservoir under the given throttles.
    fn reservoir_flows(
        &self,
        id: ReservoirId,
        conv: &[f64],
        cons: &[f64],
        cond: &[f64],
    ) -> (f64, f64) {
        let mut inflow = 0.0;
        let mut outflow = 0.0;
        for (ci, c) in self.converters.iter().enumerate() {
            for f in &c.outputs {
                if f.target == id {
                    inflow += conv[ci] * f.rate;
                }
            }
            for f in &c.inputs {
                if f.target == Source::Reservoir(id) {
                    outflow += conv[ci] * f.rate;
                }
            }
        }
        for (di, d) in self.conduits.iter().enumerate() {
            if d.to == id {
                inflow += cond[di] * d.rate;
            }
            if d.from == id {
                outflow += cond[di] * d.rate;
            }
        }
        for (si, s) in self.consumers.iter().enumerate() {
            if s.from == id {
                outflow += cons[si] * s.rate;
            }
        }
        (inflow, outflow)
    }

    /// Scale every element drawing from `id` by `s`. Returns whether anything
    /// changed.
    fn scale_drainers(
        &self,
        id: ReservoirId,
        s: f64,
        conv: &mut [f64],
        cons: &mut [f64],
        cond: &mut [f64],
    ) -> bool {
        let mut changed = false;
        for (ci, c) in self.converters.iter().enumerate() {
            if conv[ci] > 0.0 && c.inputs.iter().any(|f| f.target == Source::Reservoir(id)) {
                conv[ci] *= s;
                changed = true;
            }
        }
        for (di, d) in self.conduits.iter().enumerate() {
            if cond[di] > 0.0 && d.from == id {
                cond[di] *= s;
                changed = true;
            }
        }
        for (si, c) in self.consumers.iter().enumerate() {
            if cons[si] > 0.0 && c.from == id {
                cons[si] *= s;
                changed = true;
            }
        }
        changed
    }

    /// Scale every element filling `id` by `s`. Returns whether anything changed.
    fn scale_fillers(&self, id: ReservoirId, s: f64, conv: &mut [f64], cond: &mut [f64]) -> bool {
        let mut changed = false;
        for (ci, c) in self.converters.iter().enumerate() {
            if conv[ci] > 0.0 && c.outputs.iter().any(|f| f.target == id) {
                conv[ci] *= s;
                changed = true;
            }
        }
        for (di, d) in self.conduits.iter().enumerate() {
            if cond[di] > 0.0 && d.to == id {
                cond[di] *= s;
                changed = true;
            }
        }
        changed
    }

    /// Assemble per-reservoir and per-node net rates from resolved throttles.
    fn net_rates(&self, conv: &[f64], cons: &[f64], cond: &[f64]) -> Rates {
        let mut reservoir_net = vec![0.0_f64; self.reservoirs.len()];
        let mut node_net = vec![0.0_f64; self.nodes.len()];
        for (ci, c) in self.converters.iter().enumerate() {
            for f in &c.inputs {
                match f.target {
                    Source::Reservoir(ReservoirId(r)) => reservoir_net[r] -= conv[ci] * f.rate,
                    Source::Node(NodeId(n)) => node_net[n] -= conv[ci] * f.rate,
                }
            }
            for f in &c.outputs {
                reservoir_net[f.target.0] += conv[ci] * f.rate;
            }
        }
        for (di, d) in self.conduits.iter().enumerate() {
            reservoir_net[d.from.0] -= cond[di] * d.rate;
            reservoir_net[d.to.0] += cond[di] * d.rate;
        }
        for (si, c) in self.consumers.iter().enumerate() {
            reservoir_net[c.from.0] -= cons[si] * c.rate;
        }
        Rates {
            reservoir_net,
            node_net,
        }
    }

    /// Time until the first reservoir reaches a boundary (0 or capacity) or a
    /// node depletes, under constant `rates`. `None` if nothing saturates.
    fn next_event_dt(&self, rates: &Rates) -> Option<f64> {
        let mut best: Option<f64> = None;
        let mut consider = |t: f64| {
            if t > EPS_T && best.is_none_or(|b| t < b) {
                best = Some(t);
            }
        };
        for (ri, res) in self.reservoirs.iter().enumerate() {
            let net = rates.reservoir_net[ri];
            if net > EPS_Q {
                consider(res.headroom() / net);
            } else if net < -EPS_Q {
                consider(res.amount / -net);
            }
        }
        for (ni, node) in self.nodes.iter().enumerate() {
            let net = rates.node_net[ni];
            if net < -EPS_Q {
                consider(node.remaining / -net);
            }
        }
        best
    }

    /// Advance all amounts by `dt` under constant `rates`, clamping to bounds.
    /// On the analytic path `dt` never overshoots a boundary, so the clamp is a
    /// no-op and conservation is exact; the fixed reference may overshoot, which
    /// the clamp absorbs (the source of its `O(dt)` error).
    fn apply(&mut self, dt: f64, rates: &Rates) {
        for (ri, res) in self.reservoirs.iter_mut().enumerate() {
            res.amount = (res.amount + rates.reservoir_net[ri] * dt).clamp(0.0, res.capacity);
        }
        for (ni, node) in self.nodes.iter_mut().enumerate() {
            node.remaining = (node.remaining + rates.node_net[ni] * dt).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Resource tags (content, not code).
    const ORE: ResourceType = ResourceType(0);
    const FUEL: ResourceType = ResourceType(1);
    const O2: ResourceType = ResourceType(2);
    const WATER: ResourceType = ResourceType(3);

    /// A reservoir at a known amount with capacity.
    fn res(r: ResourceType, amount: f64, cap: f64) -> Reservoir {
        Reservoir::new(r, amount, cap)
    }

    fn assert_bounds(g: &ResourceGraph) {
        for r in &g.reservoirs {
            assert!(
                r.amount >= -EPS_Q && r.amount <= r.capacity + EPS_Q,
                "reservoir out of bounds: {} not in [0, {}]",
                r.amount,
                r.capacity
            );
        }
        for n in &g.nodes {
            assert!(n.remaining >= -EPS_Q, "node went negative: {}", n.remaining);
        }
    }

    // --- Per-element isolation tests (the "test every DOF" discipline) ---

    #[test]
    fn miner_depletes_node_and_fills_reservoir() {
        // Node (100) --extractor (rate 2)--> ore reservoir (cap 1000).
        let mut g = ResourceGraph {
            reservoirs: vec![res(ORE, 0.0, 1000.0)],
            nodes: vec![ResourceNode::new(ORE, 100.0)],
            converters: vec![Converter {
                inputs: vec![Flow::new(Source::Node(NodeId(0)), 2.0)],
                outputs: vec![Flow::new(ReservoirId(0), 2.0)],
            }],
            ..Default::default()
        };
        // Past depletion time (100 / 2 = 50).
        g.integrate(80.0);
        assert!(g.nodes[0].is_depleted(), "node should be depleted");
        assert!(
            (g.reservoirs[0].amount - 100.0).abs() < 1e-6,
            "reservoir should hold exactly the node's yield"
        );
        assert_bounds(&g);
    }

    #[test]
    fn converter_throttles_on_empty_input() {
        // Ore reservoir (10, draining), refinery ore->fuel at rate 5, no resupply.
        let mut g = ResourceGraph {
            reservoirs: vec![res(ORE, 10.0, 100.0), res(FUEL, 0.0, 100.0)],
            converters: vec![Converter {
                inputs: vec![Flow::new(Source::Reservoir(ReservoirId(0)), 5.0)],
                outputs: vec![Flow::new(ReservoirId(1), 5.0)],
            }],
            ..Default::default()
        };
        g.integrate(100.0);
        assert!(g.reservoirs[0].is_empty(), "input drains to empty");
        assert!(
            (g.reservoirs[1].amount - 10.0).abs() < 1e-6,
            "all input converted to output"
        );
        assert_bounds(&g);
    }

    #[test]
    fn converter_throttles_on_full_output() {
        // Plenty of input, output reservoir small (cap 4) -> converter stalls at full.
        let mut g = ResourceGraph {
            reservoirs: vec![res(ORE, 1000.0, 1000.0), res(FUEL, 0.0, 4.0)],
            converters: vec![Converter {
                inputs: vec![Flow::new(Source::Reservoir(ReservoirId(0)), 5.0)],
                outputs: vec![Flow::new(ReservoirId(1), 5.0)],
            }],
            ..Default::default()
        };
        g.integrate(100.0);
        assert!(g.reservoirs[1].is_full(), "output fills to capacity");
        // Exactly 4 drawn from input (recipe-conservation), not 500.
        assert!(
            (g.reservoirs[0].amount - 996.0).abs() < 1e-6,
            "only what fit the output was drawn"
        );
        assert_bounds(&g);
    }

    #[test]
    fn conduit_moves_between_reservoirs_conserving_total() {
        // A (100) --conduit rate 3--> B (cap 100), total conserved.
        let mut g = ResourceGraph {
            reservoirs: vec![res(WATER, 100.0, 100.0), res(WATER, 0.0, 100.0)],
            conduits: vec![Conduit {
                from: ReservoirId(0),
                to: ReservoirId(1),
                rate: 3.0,
            }],
            ..Default::default()
        };
        g.integrate(10.0);
        let total = g.reservoirs[0].amount + g.reservoirs[1].amount;
        assert!(
            (total - 100.0).abs() < 1e-9,
            "conduit conserves total: {total}"
        );
        assert!((g.reservoirs[1].amount - 30.0).abs() < 1e-6);
        // Past full transfer the destination cannot exceed capacity.
        g.integrate(1000.0);
        assert!((g.reservoirs[1].amount - 100.0).abs() < 1e-6);
        assert!(g.reservoirs[0].is_empty());
        assert_bounds(&g);
    }

    #[test]
    fn consumer_drains_then_throttles() {
        // Oxygen reservoir (20), crew consumer rate 1, no resupply.
        let mut g = ResourceGraph {
            reservoirs: vec![res(O2, 20.0, 100.0)],
            consumers: vec![Consumer {
                from: ReservoirId(0),
                rate: 1.0,
            }],
            ..Default::default()
        };
        g.integrate(15.0);
        assert!((g.reservoirs[0].amount - 5.0).abs() < 1e-6);
        g.integrate(100.0);
        assert!(g.reservoirs[0].is_empty(), "drains to empty, no negative");
        assert_bounds(&g);
    }

    #[test]
    fn life_support_is_content_on_the_same_system() {
        // O2 + Water consumers plus a recycler converting water->O2: no bespoke
        // life-support code, just graph content. Recycler keeps O2 from emptying
        // as fast as raw consumption would.
        let mut g = ResourceGraph {
            reservoirs: vec![res(O2, 50.0, 100.0), res(WATER, 50.0, 100.0)],
            converters: vec![Converter {
                inputs: vec![Flow::new(Source::Reservoir(ReservoirId(1)), 0.5)],
                outputs: vec![Flow::new(ReservoirId(0), 0.5)],
            }],
            consumers: vec![
                Consumer {
                    from: ReservoirId(0),
                    rate: 1.0,
                },
                Consumer {
                    from: ReservoirId(1),
                    rate: 0.2,
                },
            ],
            ..Default::default()
        };
        g.integrate(40.0);
        assert_bounds(&g);
        // Nothing exploded; O2 declined (net consumption) but recycler slowed it.
        assert!(g.reservoirs[0].amount < 50.0 && g.reservoirs[0].amount > 0.0);
    }

    // --- The defining tests: analytic catch-up ---

    /// Build the canonical mining→refining chain used by the catch-up tests.
    fn mining_chain() -> ResourceGraph {
        // Node(500) -extractor r2-> ore(cap 100) -refinery r3-> fuel(cap 200)
        ResourceGraph {
            reservoirs: vec![res(ORE, 0.0, 100.0), res(FUEL, 0.0, 200.0)],
            nodes: vec![ResourceNode::new(ORE, 500.0)],
            converters: vec![
                Converter {
                    inputs: vec![Flow::new(Source::Node(NodeId(0)), 2.0)],
                    outputs: vec![Flow::new(ReservoirId(0), 2.0)],
                },
                Converter {
                    inputs: vec![Flow::new(Source::Reservoir(ReservoirId(0)), 3.0)],
                    outputs: vec![Flow::new(ReservoirId(1), 3.0)],
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn catch_up_equals_tick_by_tick() {
        // One analytic jump must match many small fixed ticks (acceptance #2).
        let target = 300.0;
        let mut analytic = mining_chain();
        analytic.integrate(target);

        let mut reference = mining_chain();
        reference.integrate_fixed(target, 0.01);

        for (a, r) in analytic.reservoirs.iter().zip(&reference.reservoirs) {
            assert!(
                (a.amount - r.amount).abs() < 1e-2,
                "reservoir mismatch: analytic {} vs tick {}",
                a.amount,
                r.amount
            );
        }
        for (a, r) in analytic.nodes.iter().zip(&reference.nodes) {
            assert!((a.remaining - r.remaining).abs() < 1e-2);
        }
        assert_bounds(&analytic);
    }

    #[test]
    fn catch_up_cost_is_event_bounded_not_interval_bounded() {
        // The warp-filter property: a huge interval with no new saturation costs
        // the same as a small one. The step count is bounded by graph events,
        // independent of the interval / warp factor.
        let mut short = mining_chain();
        let short_steps = short.integrate(50.0);

        let mut long = mining_chain();
        let long_steps = long.integrate(1_000_000.0);

        let bound = 4 * (long.reservoirs.len() + long.nodes.len()) + 16;
        assert!(
            long_steps <= bound,
            "event count {long_steps} exceeded bound {bound}"
        );
        // A million-second catch-up is not meaningfully costlier than a short one.
        assert!(
            long_steps <= short_steps + 4,
            "long {long_steps} vs short {short_steps}"
        );
    }

    #[test]
    fn empty_graph_and_no_saturation_take_one_step() {
        // No elements: a single trivial step, regardless of interval.
        let mut g = ResourceGraph::default();
        assert_eq!(g.integrate(1e9), 1);
        assert_eq!(g.time, 1e9);

        // A reservoir nowhere near a boundary with a slow conduit: still O(1).
        let mut g2 = ResourceGraph {
            reservoirs: vec![res(WATER, 50.0, 1e12), res(WATER, 50.0, 1e12)],
            conduits: vec![Conduit {
                from: ReservoirId(0),
                to: ReservoirId(1),
                rate: 1e-6,
            }],
            ..Default::default()
        };
        let steps = g2.integrate(1000.0);
        assert!(steps <= 2, "no saturation should be ~1 step, got {steps}");
        assert_bounds(&g2);
    }

    // --- Edge cases ---

    #[test]
    fn integrate_is_noop_when_target_in_past() {
        let mut g = mining_chain();
        g.time = 100.0;
        let steps = g.integrate(50.0);
        assert_eq!(steps, 0);
        assert_eq!(g.time, 100.0);
        assert_eq!(g.reservoirs[0].amount, 0.0);
    }

    #[test]
    fn contention_two_consumers_on_one_empty_reservoir_stays_bounded() {
        // Two consumers draining one small reservoir faster than refill; the
        // rationing keeps it non-negative and bounded (no equivalence claim here,
        // contention is supported-not-required).
        let mut g = ResourceGraph {
            reservoirs: vec![res(WATER, 5.0, 100.0)],
            conduits: vec![],
            consumers: vec![
                Consumer {
                    from: ReservoirId(0),
                    rate: 3.0,
                },
                Consumer {
                    from: ReservoirId(0),
                    rate: 4.0,
                },
            ],
            ..Default::default()
        };
        g.integrate(100.0);
        assert!(g.reservoirs[0].is_empty());
        assert_bounds(&g);
    }

    #[test]
    fn degenerate_zero_rate_and_zero_capacity_are_inert() {
        let mut g = ResourceGraph {
            reservoirs: vec![res(ORE, 0.0, 0.0), res(FUEL, 5.0, 10.0)],
            converters: vec![Converter {
                inputs: vec![Flow::new(Source::Reservoir(ReservoirId(1)), 0.0)],
                outputs: vec![Flow::new(ReservoirId(0), 0.0)],
            }],
            ..Default::default()
        };
        let steps = g.integrate(1000.0);
        assert!(steps >= 1);
        assert_eq!(g.reservoirs[1].amount, 5.0);
        assert_bounds(&g);
    }

    #[test]
    fn graph_round_trips_through_serde() {
        let g = mining_chain();
        let json = serde_json::to_string(&g).expect("serialize");
        let back: ResourceGraph = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(g, back);
    }
}

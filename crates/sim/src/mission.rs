//! Mission documents + the objective evaluator (WI 551, content Slice 1).
//!
//! The declarative mission system on the offer / objective / effects
//! decomposition (the Contract Configurator model, per the content design):
//! a mission is authored content (`content/missions/<id>.ron`), its
//! **objective is a composable condition tree whose leaves are queries over
//! the bus telemetry snapshot** — the same [`Telemetry`] shape
//! `GET /telemetry` serves, never sim internals — and its **effects are bus
//! commands** (or a narrative lore beat). This makes the parent design's
//! "objectives are bus queries" mechanically true: an out-of-process client
//! could evaluate the same tree over the wire.
//!
//! **Warp safety by latching.** Evaluation is poll/snapshot-based at a
//! bounded sim-time cadence (the director owns the polling); a node
//! **latches** once satisfied, so progress is monotone and a transient state
//! missed between coarse polls under warp is defined semantics, not a race.
//! No new warp-drop triggers are introduced.
//!
//! This subsumes first-playable's deferred **536 goal evaluator**: a goal is
//! one mission with one objective leaf (the shipped "First Hop" is exactly
//! that shape).

use crate::command::Command;
use crate::content::CONTENT_FORMAT_VERSION;
use crate::telemetry::Telemetry;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A mission document as authored (and as carried on the spawn payload —
/// missions ride the scenario into the director, so the whole definition is
/// serde both ways).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mission {
    /// Document *format* version — must equal [`CONTENT_FORMAT_VERSION`].
    pub format: u32,
    /// Stable identifier; when loaded by reference it must match the file stem.
    pub id: String,
    /// Human-facing display name.
    pub name: String,
    /// When the mission becomes active. No acceptance mechanic this slice:
    /// offer-met transitions straight to Active.
    #[serde(default)]
    pub offer: Offer,
    /// The completion test: a condition tree over the telemetry snapshot.
    pub objective: Condition,
    /// Issued once, on completion.
    #[serde(default)]
    pub effects: Vec<Effect>,
}

/// When a mission becomes active. Open enum — tech/currency/body offers
/// arrive with their systems (WI 552+).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum Offer {
    /// Active from scenario start.
    #[default]
    Immediate,
    /// Active once the named mission (same scenario) completes — the linear
    /// campaign-sequencing primitive.
    AfterMission(String),
}

/// An objective condition: composable nodes over instantaneous leaf
/// predicates on the telemetry snapshot. Open enum — new leaves are additive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Condition {
    /// Every child satisfied (order-free).
    All(Vec<Condition>),
    /// Any child satisfied.
    Any(Vec<Condition>),
    /// Children satisfied **in order**: a child only begins evaluating after
    /// all earlier children have latched.
    Sequence(Vec<Condition>),
    /// Altitude above the pad surface exceeds this many metres.
    AltitudeAbove(f64),
    /// Speed exceeds this many m/s.
    SpeedAbove(f64),
    /// The craft has left the pad (released).
    Airborne,
}

impl Condition {
    /// Number of leaves under this node (progress denominator).
    pub fn leaf_count(&self) -> usize {
        match self {
            Condition::All(cs) | Condition::Any(cs) | Condition::Sequence(cs) => {
                cs.iter().map(Condition::leaf_count).sum()
            }
            _ => 1,
        }
    }

    /// An objective must test something — vacuous trees are authoring errors
    /// (validated at scenario load).
    pub fn is_vacuous(&self) -> bool {
        match self {
            Condition::All(cs) | Condition::Any(cs) | Condition::Sequence(cs) => {
                cs.is_empty() || cs.iter().any(Condition::is_vacuous)
            }
            _ => false,
        }
    }

    /// Evaluate this node's **instantaneous** truth against a snapshot,
    /// ignoring latch state (composite nodes defer to [`NodeState`]).
    fn leaf_holds(&self, snap: &Telemetry) -> bool {
        let Some(s) = snap.scenario.as_ref() else {
            return false;
        };
        match self {
            Condition::AltitudeAbove(m) => s.altitude > *m,
            Condition::SpeedAbove(v) => s.speed > *v,
            Condition::Airborne => s.airborne,
            _ => false,
        }
    }
}

/// A completion effect. Open enum; economy effects (award/unlock) arrive with
/// WI 552, spawn effects with their consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Effect {
    /// Surface a narrative beat (onto the scenario telemetry block + HUD).
    Lore(String),
    /// Issue any envelope command — validated by the executor/tier gates
    /// exactly as a player command would be (fire-and-forget, not a
    /// transaction: a rejected command does not un-complete the mission).
    Command(Command),
}

/// A mission's lifecycle state (telemetry-visible). `Failed` is reserved —
/// no failure conditions are evaluated this slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionState {
    /// Offer condition not yet met.
    Pending,
    /// Offer met; objective evaluating.
    Active,
    /// Objective satisfied; effects issued.
    Completed,
}

impl fmt::Display for MissionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MissionState::Pending => write!(f, "pending"),
            MissionState::Active => write!(f, "ACTIVE"),
            MissionState::Completed => write!(f, "COMPLETE"),
        }
    }
}

/// Per-node latch state mirroring a [`Condition`] tree (monotone progress —
/// the warp-safety rule).
#[derive(Debug, Clone, PartialEq)]
pub struct NodeState {
    /// Latched satisfaction (never un-sets).
    latched: bool,
    /// Child states for composite nodes (empty for leaves).
    children: Vec<NodeState>,
}

impl NodeState {
    /// Fresh (unsatisfied) state shaped like `condition`.
    pub fn for_condition(condition: &Condition) -> NodeState {
        let children = match condition {
            Condition::All(cs) | Condition::Any(cs) | Condition::Sequence(cs) => {
                cs.iter().map(NodeState::for_condition).collect()
            }
            _ => Vec::new(),
        };
        NodeState {
            latched: false,
            children,
        }
    }

    /// Poll: update latches from the snapshot and report satisfaction.
    /// Latched nodes stay satisfied without re-evaluation.
    pub fn poll(&mut self, condition: &Condition, snap: &Telemetry) -> bool {
        if self.latched {
            return true;
        }
        let now = match condition {
            Condition::All(cs) => {
                // Evaluate every child each poll (children latch independently).
                let mut all = true;
                for (c, st) in cs.iter().zip(self.children.iter_mut()) {
                    all &= st.poll(c, snap);
                }
                all
            }
            Condition::Any(cs) => {
                let mut any = false;
                for (c, st) in cs.iter().zip(self.children.iter_mut()) {
                    any |= st.poll(c, snap);
                }
                any
            }
            Condition::Sequence(cs) => {
                // Only the *first unlatched* child evaluates this poll; a
                // child that latches ends the poll, so the next child begins
                // strictly on a later poll — ordering is observable even when
                // one snapshot would satisfy several children at once.
                if let Some((c, st)) = cs
                    .iter()
                    .zip(self.children.iter_mut())
                    .find(|(_, st)| !st.latched)
                {
                    st.poll(c, snap);
                }
                self.children.iter().all(|c| c.latched)
            }
            leaf => leaf.leaf_holds(snap),
        };
        if now {
            self.latched = true;
        }
        self.latched
    }

    /// Latched-leaf fraction in `[0, 1]` (HUD/agent progress readability).
    pub fn progress(&self, condition: &Condition) -> f64 {
        let total = condition.leaf_count().max(1);
        (self.latched_leaves(condition) as f64) / (total as f64)
    }

    fn latched_leaves(&self, condition: &Condition) -> usize {
        match condition {
            Condition::All(cs) | Condition::Any(cs) | Condition::Sequence(cs) => cs
                .iter()
                .zip(&self.children)
                .map(|(c, st)| st.latched_leaves(c))
                .sum(),
            _ => usize::from(self.latched),
        }
    }
}

/// Why a mission document failed to parse/validate (wrapped by the scenario
/// loader with the document's id attached).
#[derive(Debug)]
pub enum MissionError {
    /// RON parse failure (includes position context).
    Parse(String),
    /// Unknown document format version.
    Format { found: u32 },
    /// The objective tree tests nothing (empty composite somewhere).
    VacuousObjective,
}

impl fmt::Display for MissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MissionError::Parse(e) => write!(f, "mission parse error: {e}"),
            MissionError::Format { found } => write!(
                f,
                "unsupported mission format version {found} (this build reads {CONTENT_FORMAT_VERSION})"
            ),
            MissionError::VacuousObjective => write!(
                f,
                "mission objective tests nothing (an empty All/Any/Sequence) — \
                 an objective must contain at least one leaf"
            ),
        }
    }
}

impl std::error::Error for MissionError {}

/// Parses and validates one mission document (id/file-stem matching is the
/// scenario loader's, which knows the reference).
pub fn parse_mission(text: &str) -> Result<Mission, MissionError> {
    let m: Mission = ron::from_str(text).map_err(|e| MissionError::Parse(e.to_string()))?;
    if m.format != CONTENT_FORMAT_VERSION {
        return Err(MissionError::Format { found: m.format });
    }
    if m.objective.is_vacuous() {
        return Err(MissionError::VacuousObjective);
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SimClock;
    use crate::telemetry::ScenarioTelemetry;

    /// A snapshot whose scenario block reports the given flight readout.
    fn snap(altitude: f64, speed: f64, airborne: bool) -> Telemetry {
        let block = ScenarioTelemetry {
            id: "s".into(),
            name: "S".into(),
            altitude,
            speed,
            airborne,
            ..Default::default()
        };
        Telemetry::capture(&SimClock::default(), None, 1.0, None).with_scenario(block)
    }

    #[test]
    fn leaves_query_the_snapshot() {
        let alt = Condition::AltitudeAbove(100.0);
        let mut st = NodeState::for_condition(&alt);
        assert!(!st.poll(&alt, &snap(50.0, 0.0, true)));
        assert!(st.poll(&alt, &snap(150.0, 0.0, true)));
        // No scenario block ⇒ leaves are false (nothing to query).
        let bare = Telemetry::capture(&SimClock::default(), None, 1.0, None);
        let mut st2 = NodeState::for_condition(&alt);
        assert!(!st2.poll(&alt, &bare));
    }

    #[test]
    fn latching_is_monotone() {
        let alt = Condition::AltitudeAbove(100.0);
        let mut st = NodeState::for_condition(&alt);
        assert!(st.poll(&alt, &snap(150.0, 0.0, true)));
        // The state regressing below the threshold does not un-satisfy.
        assert!(st.poll(&alt, &snap(10.0, 0.0, false)));
        assert_eq!(st.progress(&alt), 1.0);
    }

    #[test]
    fn all_and_any_compose() {
        let both = Condition::All(vec![
            Condition::AltitudeAbove(100.0),
            Condition::SpeedAbove(50.0),
        ]);
        let mut st = NodeState::for_condition(&both);
        // Children latch independently across polls (warp-coarse polling).
        assert!(!st.poll(&both, &snap(150.0, 10.0, true)));
        assert_eq!(st.progress(&both), 0.5);
        assert!(
            st.poll(&both, &snap(10.0, 80.0, true)),
            "second leaf latches; first stayed latched"
        );

        let either = Condition::Any(vec![
            Condition::AltitudeAbove(100.0),
            Condition::SpeedAbove(50.0),
        ]);
        let mut st = NodeState::for_condition(&either);
        assert!(st.poll(&either, &snap(150.0, 0.0, true)));
    }

    #[test]
    fn sequence_requires_order() {
        let seq = Condition::Sequence(vec![Condition::Airborne, Condition::AltitudeAbove(100.0)]);
        let mut st = NodeState::for_condition(&seq);
        // Second leaf's state holds first — but it may not latch before the
        // first: still incomplete, and progress shows only what's legal.
        assert!(!st.poll(&seq, &snap(150.0, 0.0, false)));
        assert_eq!(st.progress(&seq), 0.0);
        // First leaf latches; second begins on a later poll.
        assert!(!st.poll(&seq, &snap(150.0, 0.0, true)));
        assert_eq!(st.progress(&seq), 0.5);
        assert!(
            st.poll(&seq, &snap(150.0, 0.0, false)),
            "ordered completion"
        );
    }

    #[test]
    fn parse_validates_format_and_vacuousness() {
        let good = r#"(format: 1, id: "m", name: "M",
            objective: AltitudeAbove(100.0),
            effects: [Lore("nice hop")])"#;
        let m = parse_mission(good).unwrap();
        assert_eq!(m.offer, Offer::Immediate);
        assert_eq!(m.effects.len(), 1);

        assert!(matches!(
            parse_mission(r#"(format: 9, id: "m", name: "M", objective: Airborne)"#),
            Err(MissionError::Format { found: 9 })
        ));
        assert!(matches!(
            parse_mission(r#"(format: 1, id: "m", name: "M", objective: All([]))"#),
            Err(MissionError::VacuousObjective)
        ));
        assert!(matches!(
            parse_mission(r#"(format: 1, id: "m", name: "M", objective: Airborne, surprise: 1)"#),
            Err(MissionError::Parse(_))
        ));
    }

    #[test]
    fn command_effects_are_the_envelope() {
        // An effect carrying a real Command round-trips through serde (the
        // document is data; the command is the same envelope the bus uses).
        let text = r#"(format: 1, id: "m", name: "M",
            objective: Airborne,
            effects: [Command(SetWarp(4.0)), Lore("go")])"#;
        let m = parse_mission(text).unwrap();
        assert_eq!(m.effects[0], Effect::Command(Command::SetWarp(4.0)));
    }
}

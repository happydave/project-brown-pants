//! Durable vessel identity + the vessel record (WI 855, multiplayer arc).
//!
//! The **vessel record** is the multiplayer sync unit and the future world-save
//! element (design: `tickets/docs/projects/sounding/multiplayer/design.md`): a
//! vessel's semantic state — structure + rails motion — stamped with the
//! **universe time** at which it is true, so any receiver materializes it in its
//! own time-stream by rails arithmetic alone. No positions-at-wall-times, ever.
//!
//! **Vessel identity is instance identity**, deliberately distinct from
//! [`CraftSubgraph`]'s `id` (a document/slug identity): two spawns of one
//! blueprint share the blueprint's document id but are different vessels. An id
//! is minted when a craft first becomes a shareable universe instance and is
//! kept for the vessel's life across saves, hand-offs, and servers (the FTL
//! requirement).
//!
//! Headless and networking-free: this module defines the artifact and its
//! assembly/materialization; the universe server (WI 856) and the net adapter
//! (WI 857) move it. Record-content *validation* (size caps, finiteness) is
//! deliberately the server's job (design R4), not encoded here.

use crate::frame::{FrameId, WorldPos};
use crate::orbit::Orbit;
use crate::persist::CraftSubgraph;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Mints a fresh, globally-unique vessel id (hyphenated UUIDv4).
///
/// Client-minted at first share (design decision: survives server migration and
/// keeps ids stable across the FTL export/import seam, unlike server-minted ids).
pub fn mint_vessel_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// A vessel's rails motion state at the record's universe-time stamp.
///
/// Both variants are pure functions of time — the reason cross-subspace
/// materialization is arithmetic, never simulation (the LMP extrapolate rule as
/// architecture). An actively-flying vessel has no analytic future and is *not*
/// representable here by design: its record simply goes stale (`live` flag on
/// the record) until it next enters rails.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum MotionState {
    /// On a conic about `frame`'s central body: the existing analytic [`Orbit`]
    /// elements (elliptical or hyperbolic), propagated closed-form.
    Conic { frame: FrameId, orbit: Orbit },
    /// Fixed to a body: landed/splashed is rails too (a surface fix in the
    /// body's frame, time-invariant here; body rotation is a later concern
    /// alongside the app's landed representation).
    SurfaceFix { position: WorldPos },
}

impl MotionState {
    /// The frame this motion is expressed in.
    pub fn frame(&self) -> FrameId {
        match self {
            MotionState::Conic { frame, .. } => *frame,
            MotionState::SurfaceFix { position } => position.frame,
        }
    }

    /// Rails-propagates to universe time `t`: a frame-tagged position, by
    /// arithmetic only. The conic delegates to the one closed-form propagator
    /// (planar orbit embedded at z = 0, the WI 508 hand-off convention); a
    /// surface fix is time-invariant.
    pub fn position_at(&self, t: f64) -> WorldPos {
        match self {
            MotionState::Conic { frame, orbit } => {
                let (p, _v) = orbit.position_velocity(t);
                WorldPos::new(*frame, DVec3::new(p.x, p.y, 0.0))
            }
            MotionState::SurfaceFix { position } => *position,
        }
    }
}

/// A terminal fate: the R2 tombstone. A record carrying a fate is a vessel's
/// final state; observers *behind* `stamp` still materialize the vessel until
/// their own clock passes it (the causality rule), so the last structure/motion
/// remain present and materializable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fate {
    /// Destroyed (breakup, crash — the applied-destructive-event outcome).
    Destroyed,
    /// Recovered (deliberately removed from the universe, e.g. mission end).
    Recovered,
}

/// The vessel record: identity + ownership + time-stamped semantic state.
///
/// Serialized additively on the persist line (a [`crate::persist::Payload`]
/// variant; format version unchanged per the documented additive-variant rule).
/// Optional fields are default-absent so minimal records stay minimal; the
/// authority/subspace/session vocabulary is owned by the server protocol
/// (WI 856) — here they are opaque identifiers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VesselRecord {
    /// Durable, globally-unique instance id ([`mint_vessel_id`]).
    pub vessel_id: String,
    /// Human-facing display name.
    pub name: String,
    /// Owning player identity (opaque; protocol vocabulary is WI 856's).
    pub owner: String,
    /// Session currently holding write authority, if any (single-writer lock).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authority: Option<String>,
    /// Subspace (time-stream) the owner occupied when this state was published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subspace: Option<String>,
    /// Universe time (SI seconds, the sim clock axis) at which this state holds.
    pub stamp: f64,
    /// The structure: the versioned craft subgraph (lattice, devices, panels,
    /// shapes — the persist line's craft-scope payload, embedded whole).
    pub structure: CraftSubgraph,
    /// Rails motion at `stamp`.
    pub motion: MotionState,
    /// Owner is currently flying this vessel in the active gear (its shared
    /// state is going stale; peers show last-known). Default false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub live: bool,
    /// Terminal tombstone (R2), absent while the vessel exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fate: Option<Fate>,
}

impl VesselRecord {
    /// Assembles a record from live rails parts: a craft on `orbit` about
    /// `frame`'s body, structure `structure`, true at universe time `stamp`.
    pub fn from_rails(
        vessel_id: impl Into<String>,
        name: impl Into<String>,
        owner: impl Into<String>,
        stamp: f64,
        frame: FrameId,
        orbit: Orbit,
        structure: CraftSubgraph,
    ) -> Self {
        Self {
            vessel_id: vessel_id.into(),
            name: name.into(),
            owner: owner.into(),
            authority: None,
            subspace: None,
            stamp,
            structure,
            motion: MotionState::Conic { frame, orbit },
            live: false,
            fate: None,
        }
    }

    /// Assembles a record for a landed/splashed vessel fixed at `position`.
    pub fn from_surface(
        vessel_id: impl Into<String>,
        name: impl Into<String>,
        owner: impl Into<String>,
        stamp: f64,
        position: WorldPos,
        structure: CraftSubgraph,
    ) -> Self {
        Self {
            vessel_id: vessel_id.into(),
            name: name.into(),
            owner: owner.into(),
            authority: None,
            subspace: None,
            stamp,
            structure,
            motion: MotionState::SurfaceFix { position },
            live: false,
            fate: None,
        }
    }

    /// True if this record is a tombstone (terminal; see [`Fate`]).
    pub fn is_tombstone(&self) -> bool {
        self.fate.is_some()
    }

    /// Materialization: rails-propagates the recorded motion to universe time
    /// `t` (arithmetic only — see [`MotionState::position_at`]). Answers for
    /// tombstoned records too: an observer behind the tombstone still sees the
    /// vessel at its last state (the causality rule).
    pub fn position_at(&self, t: f64) -> WorldPos {
        self.motion.position_at(t)
    }

    /// Materialization: the structure to spawn from (the reverse of assembly —
    /// the embedded subgraph is the complete persist-grade craft).
    pub fn structure(&self) -> &CraftSubgraph {
        &self.structure
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{Material, Voxel, VoxelCraft};
    use glam::IVec3;

    fn sample_structure() -> CraftSubgraph {
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        CraftSubgraph::new(
            "starter-1",
            "Starter",
            WorldPos::new(FrameId::CENTRAL_BODY, DVec3::ZERO),
            craft,
        )
    }

    fn sample_orbit() -> Orbit {
        // Circular LEO-like orbit about the earthlike body.
        let mu = crate::sim::CentralBody::EARTHLIKE.mu;
        let r = 7.0e6;
        let v = (mu / r).sqrt();
        Orbit::from_state(
            mu,
            glam::DVec2::new(r, 0.0),
            glam::DVec2::new(0.0, v),
            100.0,
        )
        .expect("bound orbit")
    }

    #[test]
    fn minted_ids_are_unique_stable_uuids() {
        let a = mint_vessel_id();
        let b = mint_vessel_id();
        assert_ne!(a, b, "two mints differ");
        // Hyphenated UUID shape: fixed length, four hyphens, lowercase hex.
        for id in [&a, &b] {
            assert_eq!(id.len(), 36);
            assert_eq!(id.matches('-').count(), 4);
            assert!(id
                .chars()
                .all(|c| c == '-' || c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        }
        // Round-trips serde as a plain JSON string.
        let json = serde_json::to_string(&a).unwrap();
        let back: String = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }

    #[test]
    fn conic_record_assembles_and_propagates_exactly_as_the_orbit() {
        let orbit = sample_orbit();
        let rec = VesselRecord::from_rails(
            mint_vessel_id(),
            "Ranger",
            "dave",
            100.0,
            FrameId::CENTRAL_BODY,
            orbit,
            sample_structure(),
        );
        assert!(!rec.is_tombstone());
        // Propagation delegates to the one closed-form propagator: exact match
        // (same math path), a quarter period later, in the record's frame.
        let t = 100.0 + orbit.period() / 4.0;
        let (p, _) = orbit.position_velocity(t);
        let pos = rec.position_at(t);
        assert_eq!(pos.frame, FrameId::CENTRAL_BODY);
        assert_eq!(pos.pos, DVec3::new(p.x, p.y, 0.0));
        // The extracted structure is the input subgraph, exactly.
        assert_eq!(rec.structure(), &sample_structure());
    }

    #[test]
    fn surface_fix_record_is_time_invariant() {
        let fix = WorldPos::new(FrameId::CENTRAL_BODY, DVec3::new(6.36e6, 0.0, 0.0));
        let rec = VesselRecord::from_surface(
            mint_vessel_id(),
            "Lander",
            "dave",
            500.0,
            fix,
            sample_structure(),
        );
        for t in [0.0, 500.0, 1.0e9] {
            assert_eq!(rec.position_at(t), fix);
        }
    }

    #[test]
    fn tombstoned_record_still_materializes_its_last_state() {
        // R2 + causality: an observer behind the tombstone still sees the
        // vessel at its last recorded state.
        let orbit = sample_orbit();
        let mut rec = VesselRecord::from_rails(
            mint_vessel_id(),
            "Doomed",
            "dave",
            100.0,
            FrameId::CENTRAL_BODY,
            orbit,
            sample_structure(),
        );
        rec.fate = Some(Fate::Destroyed);
        assert!(rec.is_tombstone());
        let (p, _) = orbit.position_velocity(150.0);
        assert_eq!(rec.position_at(150.0).pos, DVec3::new(p.x, p.y, 0.0));
    }
}

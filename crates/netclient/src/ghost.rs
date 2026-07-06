//! The ghost store: pure materialization state for remote vessels.
//!
//! Per remote vessel it holds a **shown** record — the newest *received*
//! record whose stamp ≤ local time (the design's materialization rule over
//! what this client has seen) — and an optional **pending** future-stamped
//! record that replaces the shown one once local time passes its stamp.
//!
//! The shown+pending pair is the blink-out refinement (WI 857 plan): the
//! server keeps only the newest record per vessel, so a strict single-slot
//! view would make a vessel *vanish* for an observer the moment its owner
//! publishes from the observer's future. Holding the superseded-but-visible
//! record preserves continuity without ever showing the future.
//!
//! Tombstones (R2) ride the same rule: a tombstone becomes "shown" only once
//! local time reaches its stamp — at which point the vessel is despawned
//! (removed from the visible set). Until then the last state stays visible.
//!
//! Pure and clock-free: callers supply local (subspace) time.

use sounding_sim::vessel::VesselRecord;
use std::collections::{BTreeMap, HashSet};

/// One remote vessel's materialization state.
#[derive(Clone, Debug)]
struct Ghost {
    /// Newest received record with stamp ≤ the local time at ingest/advance.
    shown: Option<VesselRecord>,
    /// Newest received future-stamped record, awaiting local time.
    pending: Option<VesselRecord>,
}

/// A materializable peer view at a queried local time.
#[derive(Clone, Debug)]
pub struct PeerView {
    /// The record to render (never a tombstone, never future-stamped).
    pub record: VesselRecord,
    /// Seconds of local time since the record's stamp (staleness).
    pub stale: f64,
}

/// The ghost store. Deterministic iteration (BTreeMap by vessel id).
#[derive(Default)]
pub struct GhostStore {
    ghosts: BTreeMap<String, Ghost>,
    /// Vessel ids this client published (excluded from ghosts — they are not
    /// peers; includes ids from earlier attempts/sessions so a returning own
    /// tombstone never renders).
    own: HashSet<String>,
}

impl GhostStore {
    /// Registers a vessel id as our own (never materialized as a ghost).
    pub fn mark_own(&mut self, vessel_id: &str) {
        self.own.insert(vessel_id.to_string());
        self.ghosts.remove(vessel_id);
    }

    /// Ingests a fetched record at local time `t`. Own vessels are skipped;
    /// stale-or-equal duplicates are idempotent (newest stamp wins per slot).
    pub fn ingest(&mut self, record: VesselRecord, t: f64) {
        if self.own.contains(&record.vessel_id) {
            return;
        }
        let ghost = self
            .ghosts
            .entry(record.vessel_id.clone())
            .or_insert(Ghost {
                shown: None,
                pending: None,
            });
        if record.stamp <= t {
            if ghost.shown.as_ref().is_none_or(|s| s.stamp <= record.stamp) {
                ghost.shown = Some(record);
            }
        } else if ghost
            .pending
            .as_ref()
            .is_none_or(|p| p.stamp <= record.stamp)
        {
            ghost.pending = Some(record);
        }
    }

    /// Advances materialization to local time `t`: pending records whose stamp
    /// has been reached replace the shown ones; vessels whose shown record is a
    /// tombstone despawn (are removed).
    pub fn advance(&mut self, t: f64) {
        for ghost in self.ghosts.values_mut() {
            let promote = ghost.pending.as_ref().is_some_and(|p| p.stamp <= t);
            if promote {
                ghost.shown = ghost.pending.take();
            }
        }
        // A shown tombstone means local time has reached the terminal record:
        // the vessel despawns (the R2 causality rule, client side).
        self.ghosts
            .retain(|_, g| !g.shown.as_ref().is_some_and(|s| s.is_tombstone()));
    }

    /// The materializable peers at local time `t` (call [`Self::advance`]
    /// first for promotion/despawn semantics). Never contains a tombstone or a
    /// future-stamped record.
    pub fn visible(&self, t: f64) -> Vec<PeerView> {
        self.ghosts
            .values()
            .filter_map(|g| g.shown.as_ref())
            .filter(|s| !s.is_tombstone() && s.stamp <= t)
            .map(|s| PeerView {
                record: s.clone(),
                stale: t - s.stamp,
            })
            .collect()
    }

    /// Number of tracked remote vessels (visible or pending).
    pub fn len(&self) -> usize {
        self.ghosts.len()
    }

    /// True when no remote vessel is tracked.
    pub fn is_empty(&self) -> bool {
        self.ghosts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sounding_sim::frame::{FrameId, WorldPos};
    use sounding_sim::persist::CraftSubgraph;
    use sounding_sim::vessel::{Fate, VesselRecord};
    use sounding_sim::voxel::VoxelCraft;

    fn fix_record(vessel_id: &str, stamp: f64) -> VesselRecord {
        VesselRecord::from_surface(
            vessel_id,
            "Peer",
            "bob",
            stamp,
            WorldPos::new(
                FrameId::CENTRAL_BODY,
                glam::DVec3::new(stamp, 0.0, 0.0), // position encodes the stamp for assertions
            ),
            CraftSubgraph::new(
                "peer",
                "Peer",
                WorldPos::new(FrameId::CENTRAL_BODY, glam::DVec3::ZERO),
                VoxelCraft::new(1.0),
            ),
        )
    }

    #[test]
    fn future_records_hold_pending_and_swap_without_blink_out() {
        let mut store = GhostStore::default();
        // A peer's parked record at their t=10; I'm at t=50: visible.
        store.ingest(fix_record("v1", 10.0), 50.0);
        assert_eq!(store.visible(50.0).len(), 1);

        // The peer updates from my future (their t=200): the shown record must
        // NOT vanish (blink-out), the update waits pending.
        store.ingest(fix_record("v1", 200.0), 50.0);
        store.advance(60.0);
        let vis = store.visible(60.0);
        assert_eq!(vis.len(), 1, "no blink-out");
        assert_eq!(vis[0].record.stamp, 10.0, "still the old state");
        assert_eq!(vis[0].stale, 50.0, "staleness reported");

        // Once my time passes the update's stamp it swaps in.
        store.advance(200.0);
        let vis = store.visible(200.0);
        assert_eq!(vis.len(), 1);
        assert_eq!(vis[0].record.stamp, 200.0, "pending promoted at its stamp");
    }

    #[test]
    fn tombstones_despawn_only_when_local_time_reaches_them() {
        let mut store = GhostStore::default();
        store.ingest(fix_record("v1", 10.0), 20.0);
        let mut tomb = fix_record("v1", 100.0);
        tomb.fate = Some(Fate::Destroyed);
        store.ingest(tomb, 20.0);

        // Behind the tombstone: the last state stays visible.
        store.advance(50.0);
        assert_eq!(store.visible(50.0).len(), 1, "still visible before the end");

        // At/past the tombstone stamp: despawned entirely.
        store.advance(100.0);
        assert!(store.visible(100.0).is_empty(), "despawned at the stamp");
        assert!(store.is_empty(), "entry dropped");
    }

    #[test]
    fn own_vessels_are_never_ghosts_even_when_fetched_back() {
        let mut store = GhostStore::default();
        store.mark_own("mine");
        store.ingest(fix_record("mine", 10.0), 50.0);
        let mut own_tomb = fix_record("mine", 20.0);
        own_tomb.fate = Some(Fate::Destroyed);
        store.ingest(own_tomb, 50.0);
        assert!(store.visible(50.0).is_empty());
        // Marking own after an ingest also evicts.
        store.ingest(fix_record("other", 10.0), 50.0);
        store.mark_own("other");
        assert!(store.visible(50.0).is_empty());
    }

    #[test]
    fn newest_stamp_wins_per_slot_and_ingest_is_idempotent() {
        let mut store = GhostStore::default();
        store.ingest(fix_record("v1", 30.0), 50.0);
        store.ingest(fix_record("v1", 20.0), 50.0); // older: ignored
        store.ingest(fix_record("v1", 30.0), 50.0); // duplicate: idempotent
        let vis = store.visible(50.0);
        assert_eq!(vis.len(), 1);
        assert_eq!(vis[0].record.stamp, 30.0);
    }
}

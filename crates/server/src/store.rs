//! The universe store: vessel-record table + subspace registry + authority
//! ledger + session leases. **Pure and clock-free** — every operation takes an
//! explicit `now` (monotonic seconds, caller-supplied), so time is fully
//! test-controlled and the store never simulates or reads a clock.
//!
//! Rules enforced here (multiplayer design, `tickets/docs/projects/sounding/
//! multiplayer/design.md`):
//! - **Bounded storage**: exactly one current record per vessel id (a tombstone
//!   is that one row); a monotonic change cursor; no queues, no history.
//! - **Single-writer authority**: first publish of an unknown vessel grants the
//!   publisher authority; later writes require it; a tombstone releases it
//!   (R2); claim needs unheld + not tombstoned + claimant time ≥ record stamp
//!   (the causality guard); transfer is atomic release+claim.
//! - **Causality**: a session's reported universe time is non-decreasing; a
//!   vessel's record stamps are non-decreasing.
//! - **R4 hygiene**: motion-state finiteness checked before storing (the size
//!   cap is the router's, at the byte boundary).
//! - **R5 leases**: sessions renew on any authenticated request (and explicit
//!   heartbeat); idle-past-TTL sessions are swept, releasing their locks with
//!   the last published record standing.

use sounding_sim::vessel::{MotionState, VesselRecord};
use std::collections::{BTreeMap, HashMap};

/// The protocol version checked at handshake. Bump on any wire-visible change
/// to the operations or their shapes (the versioned-public-contract NFR).
pub const PROTOCOL_VERSION: u32 = 1;

/// Server configuration (transport-independent parts).
#[derive(Clone, Debug)]
pub struct StoreConfig {
    /// Pre-shared invite token exchanged for a session at handshake.
    pub invite_token: String,
    /// Expected content identity (opaque canonical string). `None` ⇒ the first
    /// successful handshake pins it (zero-config LAN default).
    pub content_identity: Option<String>,
    /// Lease TTL in seconds: a session idle longer than this expires.
    pub lease_ttl: f64,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            invite_token: String::new(),
            content_identity: None,
            lease_ttl: 30.0,
        }
    }
}

/// An R1 subspace time anchor: universe time at a server-monotonic instant,
/// plus the current rate (warp; pause = 0). Current time is derived on demand —
/// the server never advances anything.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    /// Universe time (SI seconds) at `wall`.
    pub universe_time: f64,
    /// Server-monotonic seconds when the anchor was reported.
    pub wall: f64,
    /// Time rate (warp factor; 0 = paused).
    pub rate: f64,
}

impl Anchor {
    /// The subspace's current universe time at server-monotonic `now`.
    pub fn current_time(&self, now: f64) -> f64 {
        self.universe_time + self.rate * (now - self.wall)
    }
}

/// A connected session (a lease). The `id` is public (it names authority in
/// records and the registry); the `token` is the secret bearer credential and
/// never appears in records, registry entries, or saves.
#[derive(Clone, Debug)]
pub struct Session {
    /// Public session id (UUID).
    pub id: String,
    /// Player display identity.
    pub player: String,
    /// Server-monotonic seconds of the last authenticated activity.
    pub last_seen: f64,
    /// The session's subspace anchor, once reported.
    pub anchor: Option<Anchor>,
}

/// A stored record with its change sequence (the cursor unit).
#[derive(Clone, Debug)]
struct StoredRecord {
    record: VesselRecord,
    seq: u64,
}

/// A store operation error: every variant maps to one legible protocol
/// rejection. `msg()` is the wire text.
#[derive(Clone, Debug, PartialEq)]
pub enum StoreError {
    /// Handshake: wrong invite token.
    BadInvite,
    /// Handshake: protocol version mismatch (client, server).
    ProtocolMismatch(u32, u32),
    /// Handshake: content identity mismatch (client, server).
    ContentMismatch(String, String),
    /// Unknown, invalid, or expired session token.
    Unauthorized,
    /// Anchor report behind the session's previously reported universe time.
    BackwardSync { reported: f64, previous: f64 },
    /// Record write without holding the vessel's authority.
    NotAuthority,
    /// Record stamp older than the stored record's.
    StaleStamp { stamp: f64, stored: f64 },
    /// Write or claim on a tombstoned (terminal) vessel.
    Tombstoned,
    /// R4: a non-finite motion-state value.
    NonFinite,
    /// Claim: vessel already held by a live session.
    AlreadyHeld,
    /// Claim/transfer causality guard: claimant time < record stamp.
    ClaimBehindRecord { claimant_time: f64, stamp: f64 },
    /// Claim/transfer: the claimant has reported no anchor yet.
    NoAnchor,
    /// Lock op on a vessel the store does not hold.
    UnknownVessel,
    /// Transfer target is not a live session.
    UnknownTarget,
}

impl StoreError {
    /// The legible wire message for this rejection.
    pub fn msg(&self) -> String {
        match self {
            StoreError::BadInvite => "invite token rejected".into(),
            StoreError::ProtocolMismatch(c, s) => {
                format!("protocol version mismatch: client {c}, server {s}")
            }
            StoreError::ContentMismatch(c, s) => {
                format!("content identity mismatch: client \"{c}\", server \"{s}\"")
            }
            StoreError::Unauthorized => "unknown, invalid, or expired session token".into(),
            StoreError::BackwardSync { reported, previous } => format!(
                "backward sync rejected: reported universe time {reported} is behind previously reported {previous}"
            ),
            StoreError::NotAuthority => "session does not hold this vessel's authority".into(),
            StoreError::StaleStamp { stamp, stored } => format!(
                "stale record rejected: stamp {stamp} is behind the stored record's {stored}"
            ),
            StoreError::Tombstoned => "vessel is tombstoned (terminal)".into(),
            StoreError::NonFinite => "record rejected: non-finite motion-state value".into(),
            StoreError::AlreadyHeld => "vessel authority is already held".into(),
            StoreError::ClaimBehindRecord {
                claimant_time,
                stamp,
            } => format!(
                "claim rejected by the causality guard: claimant time {claimant_time} is behind the record stamp {stamp} — sync forward first"
            ),
            StoreError::NoAnchor => "claimant has reported no subspace anchor".into(),
            StoreError::UnknownVessel => "no such vessel".into(),
            StoreError::UnknownTarget => "transfer target is not a live session".into(),
        }
    }
}

/// The universe store. See the module docs for the rules it enforces.
pub struct UniverseStore {
    config: StoreConfig,
    /// Pinned content identity (from config, or adopted at first handshake).
    content: Option<String>,
    /// Live sessions, keyed by secret token.
    sessions: HashMap<String, Session>,
    /// Newest record per vessel id (BTreeMap: deterministic iteration/saves).
    records: BTreeMap<String, StoredRecord>,
    /// Monotonic change cursor; advances only on an accepted record write.
    cursor: u64,
}

impl UniverseStore {
    /// An empty store.
    pub fn new(config: StoreConfig) -> Self {
        let content = config.content_identity.clone();
        Self {
            config,
            content,
            sessions: HashMap::new(),
            records: BTreeMap::new(),
            cursor: 0,
        }
    }

    /// A store loaded from a world-save's vessels (restart path): records are
    /// restored exactly except **authority is cleared** (sessions did not
    /// survive — the R5 posture) and the cursor restarts with every record
    /// freshly sequenced (reconnecting clients re-handshake and fetch from 0).
    pub fn from_vessels(config: StoreConfig, vessels: Vec<VesselRecord>) -> Self {
        let mut store = Self::new(config);
        for mut record in vessels {
            record.authority = None;
            store.cursor += 1;
            let seq = store.cursor;
            store
                .records
                .insert(record.vessel_id.clone(), StoredRecord { record, seq });
        }
        store
    }

    /// The record table for a world-save, in deterministic (vessel-id) order.
    pub fn vessels(&self) -> Vec<VesselRecord> {
        self.records.values().map(|s| s.record.clone()).collect()
    }

    /// Sweeps expired sessions (idle past the TTL), releasing every lock each
    /// held — the last published record stands (R5). Called at router entry so
    /// a crashed client's vessels free without it ever returning.
    pub fn sweep(&mut self, now: f64) {
        let ttl = self.config.lease_ttl;
        let expired: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| now - s.last_seen > ttl)
            .map(|(token, _)| token.clone())
            .collect();
        for token in expired {
            if let Some(session) = self.sessions.remove(&token) {
                self.release_all_locks(&session.id);
            }
        }
    }

    fn release_all_locks(&mut self, session_id: &str) {
        for stored in self.records.values_mut() {
            if stored.record.authority.as_deref() == Some(session_id) {
                stored.record.authority = None;
            }
        }
    }

    /// Handshake: validates protocol version, invite token, and content
    /// identity (pinning it if unconfigured), then opens a session lease.
    /// Returns `(public session id, secret token)`.
    pub fn handshake(
        &mut self,
        now: f64,
        protocol_version: u32,
        invite_token: &str,
        player: &str,
        content_identity: &str,
    ) -> Result<(String, String), StoreError> {
        if protocol_version != PROTOCOL_VERSION {
            return Err(StoreError::ProtocolMismatch(
                protocol_version,
                PROTOCOL_VERSION,
            ));
        }
        if invite_token != self.config.invite_token {
            return Err(StoreError::BadInvite);
        }
        match &self.content {
            Some(expected) if expected != content_identity => {
                return Err(StoreError::ContentMismatch(
                    content_identity.to_string(),
                    expected.clone(),
                ));
            }
            Some(_) => {}
            None => self.content = Some(content_identity.to_string()),
        }
        let id = uuid::Uuid::new_v4().to_string();
        let token = uuid::Uuid::new_v4().to_string();
        self.sessions.insert(
            token.clone(),
            Session {
                id: id.clone(),
                player: player.to_string(),
                last_seen: now,
                anchor: None,
            },
        );
        Ok((id, token))
    }

    /// Authenticates a token, renewing the lease (any authenticated request is
    /// activity). Returns the public session id.
    pub fn auth(&mut self, now: f64, token: &str) -> Result<String, StoreError> {
        let ttl = self.config.lease_ttl;
        let expired = match self.sessions.get(token) {
            None => return Err(StoreError::Unauthorized),
            Some(s) => now - s.last_seen > ttl,
        };
        if expired {
            // Expired but not yet swept: sweep it now (locks release, R5).
            let session = self.sessions.remove(token).expect("present");
            self.release_all_locks(&session.id);
            return Err(StoreError::Unauthorized);
        }
        let s = self.sessions.get_mut(token).expect("present");
        s.last_seen = now;
        Ok(s.id.clone())
    }

    /// Reports the session's subspace anchor (R1): at handshake-time zero
    /// knowledge, then on every warp/pause change. Reported universe time must
    /// be non-decreasing (causality: sync-backward does not exist).
    pub fn report_anchor(
        &mut self,
        now: f64,
        token: &str,
        universe_time: f64,
        rate: f64,
    ) -> Result<(), StoreError> {
        self.auth(now, token)?;
        let session = self.sessions.get_mut(token).expect("just authed");
        if let Some(prev) = session.anchor {
            if universe_time < prev.universe_time {
                return Err(StoreError::BackwardSync {
                    reported: universe_time,
                    previous: prev.universe_time,
                });
            }
        }
        session.anchor = Some(Anchor {
            universe_time,
            wall: now,
            rate,
        });
        Ok(())
    }

    /// Explicit lease renewal (a no-op beyond the `auth` it performs).
    pub fn heartbeat(&mut self, now: f64, token: &str) -> Result<(), StoreError> {
        self.auth(now, token).map(|_| ())
    }

    /// The registry: every live session with its derived current time (R1).
    pub fn registry(&self, now: f64) -> Vec<(Session, Option<f64>)> {
        let mut out: Vec<(Session, Option<f64>)> = self
            .sessions
            .values()
            .map(|s| (s.clone(), s.anchor.map(|a| a.current_time(now))))
            .collect();
        out.sort_by(|a, b| a.0.id.cmp(&b.0.id));
        out
    }

    /// R4 hygiene: every motion-state value must be finite. (The byte-size cap
    /// is enforced at the router boundary, before parsing.)
    fn check_finite(record: &VesselRecord) -> Result<(), StoreError> {
        let finite = match &record.motion {
            MotionState::Conic { orbit, .. } => {
                orbit.mu.is_finite()
                    && orbit.semi_major_axis.is_finite()
                    && orbit.eccentricity.is_finite()
                    && orbit.arg_periapsis.is_finite()
                    && orbit.mean_anomaly_at_epoch.is_finite()
                    && orbit.epoch.is_finite()
                    && orbit.sense.is_finite()
            }
            MotionState::SurfaceFix { position } => position.pos.is_finite(),
        };
        if finite && record.stamp.is_finite() {
            Ok(())
        } else {
            Err(StoreError::NonFinite)
        }
    }

    /// Publishes a record. First publish of an unknown vessel grants the
    /// publisher authority; later writes require holding it; stamps are
    /// per-vessel non-decreasing; a tombstoned vessel is terminal. A tombstone
    /// write releases authority (R2). Returns the new cursor.
    pub fn publish(
        &mut self,
        now: f64,
        token: &str,
        mut record: VesselRecord,
    ) -> Result<u64, StoreError> {
        let session_id = self.auth(now, token)?;
        Self::check_finite(&record)?;
        if let Some(stored) = self.records.get(&record.vessel_id) {
            if stored.record.is_tombstone() {
                return Err(StoreError::Tombstoned);
            }
            if stored.record.authority.as_deref() != Some(session_id.as_str()) {
                return Err(StoreError::NotAuthority);
            }
            if record.stamp < stored.record.stamp {
                return Err(StoreError::StaleStamp {
                    stamp: record.stamp,
                    stored: stored.record.stamp,
                });
            }
        }
        // The server owns the authority field: a tombstone releases (R2),
        // anything else records the writing session as holder.
        record.authority = if record.is_tombstone() {
            None
        } else {
            Some(session_id)
        };
        self.cursor += 1;
        let seq = self.cursor;
        self.records
            .insert(record.vessel_id.clone(), StoredRecord { record, seq });
        Ok(self.cursor)
    }

    /// Fetch-since: exactly the records (tombstones included) whose change
    /// sequence is past `cursor`, oldest change first, plus the new cursor.
    pub fn fetch_since(
        &mut self,
        now: f64,
        token: &str,
        cursor: u64,
    ) -> Result<(Vec<VesselRecord>, u64), StoreError> {
        self.auth(now, token)?;
        let mut changed: Vec<&StoredRecord> =
            self.records.values().filter(|s| s.seq > cursor).collect();
        changed.sort_by_key(|s| s.seq);
        Ok((
            changed.into_iter().map(|s| s.record.clone()).collect(),
            self.cursor,
        ))
    }

    /// The design's materialization rule over what the bounded store holds:
    /// the records visible to an observer at universe time `t` (stamp ≤ t).
    /// A future-stamped record does not exist for a past observer.
    pub fn visible_records(&self, t: f64) -> Vec<&VesselRecord> {
        self.records
            .values()
            .filter(|s| s.record.stamp <= t)
            .map(|s| &s.record)
            .collect()
    }

    /// Releases a held vessel (holder only). The last record stands.
    pub fn release(&mut self, now: f64, token: &str, vessel_id: &str) -> Result<(), StoreError> {
        let session_id = self.auth(now, token)?;
        let stored = self
            .records
            .get_mut(vessel_id)
            .ok_or(StoreError::UnknownVessel)?;
        if stored.record.authority.as_deref() != Some(session_id.as_str()) {
            return Err(StoreError::NotAuthority);
        }
        stored.record.authority = None;
        Ok(())
    }

    /// Claims an unheld vessel: not tombstoned, and the claimant's derived
    /// subspace time must be at or past the record stamp (the causality guard —
    /// no one edits a vessel's past; sync forward first).
    pub fn claim(&mut self, now: f64, token: &str, vessel_id: &str) -> Result<(), StoreError> {
        let session_id = self.auth(now, token)?;
        let claimant_time = self
            .sessions
            .get(token)
            .and_then(|s| s.anchor)
            .map(|a| a.current_time(now))
            .ok_or(StoreError::NoAnchor)?;
        let stored = self
            .records
            .get_mut(vessel_id)
            .ok_or(StoreError::UnknownVessel)?;
        if stored.record.is_tombstone() {
            return Err(StoreError::Tombstoned);
        }
        if stored.record.authority.is_some() {
            return Err(StoreError::AlreadyHeld);
        }
        if claimant_time < stored.record.stamp {
            return Err(StoreError::ClaimBehindRecord {
                claimant_time,
                stamp: stored.record.stamp,
            });
        }
        stored.record.authority = Some(session_id);
        Ok(())
    }

    /// Atomic transfer: holder hands a vessel to a live target session, which
    /// must satisfy the same causality guard as a claim.
    pub fn transfer(
        &mut self,
        now: f64,
        token: &str,
        vessel_id: &str,
        to_session: &str,
    ) -> Result<(), StoreError> {
        let session_id = self.auth(now, token)?;
        let target = self
            .sessions
            .values()
            .find(|s| s.id == to_session)
            .ok_or(StoreError::UnknownTarget)?;
        let target_time = target
            .anchor
            .map(|a| a.current_time(now))
            .ok_or(StoreError::NoAnchor)?;
        let stored = self
            .records
            .get_mut(vessel_id)
            .ok_or(StoreError::UnknownVessel)?;
        if stored.record.is_tombstone() {
            return Err(StoreError::Tombstoned);
        }
        if stored.record.authority.as_deref() != Some(session_id.as_str()) {
            return Err(StoreError::NotAuthority);
        }
        if target_time < stored.record.stamp {
            return Err(StoreError::ClaimBehindRecord {
                claimant_time: target_time,
                stamp: stored.record.stamp,
            });
        }
        stored.record.authority = Some(to_session.to_string());
        Ok(())
    }

    /// Introspection counts for the status endpoint.
    pub fn status_counts(&self) -> (usize, usize, u64) {
        (self.sessions.len(), self.records.len(), self.cursor)
    }

    /// The pinned content identity, if any (config or adopted).
    pub fn content_identity(&self) -> Option<&str> {
        self.content.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec2, DVec3};
    use sounding_sim::frame::{FrameId, WorldPos};
    use sounding_sim::orbit::Orbit;
    use sounding_sim::persist::CraftSubgraph;
    use sounding_sim::sim::CentralBody;
    use sounding_sim::vessel::{mint_vessel_id, Fate, VesselRecord};
    use sounding_sim::voxel::{Material, Voxel, VoxelCraft};

    fn config() -> StoreConfig {
        StoreConfig {
            invite_token: "invite".into(),
            content_identity: None,
            lease_ttl: 30.0,
        }
    }

    fn open_session(store: &mut UniverseStore, now: f64, player: &str) -> (String, String) {
        store
            .handshake(now, PROTOCOL_VERSION, "invite", player, "content-a")
            .expect("handshake")
    }

    fn sample_record(stamp: f64) -> VesselRecord {
        let mu = CentralBody::EARTHLIKE.mu;
        let orbit = Orbit::from_state(
            mu,
            DVec2::new(7.0e6, 0.0),
            DVec2::new(0.0, (mu / 7.0e6).sqrt()),
            stamp,
        )
        .expect("bound");
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: glam::IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        VesselRecord::from_rails(
            mint_vessel_id(),
            "Ranger",
            "dave",
            stamp,
            FrameId::CENTRAL_BODY,
            orbit,
            CraftSubgraph::new(
                "starter",
                "Starter",
                WorldPos::new(FrameId::CENTRAL_BODY, DVec3::ZERO),
                craft,
            ),
        )
    }

    #[test]
    fn handshake_validates_version_invite_and_content_and_adopts_first_identity() {
        let mut store = UniverseStore::new(config());
        // Wrong protocol version: error names both versions.
        let err = store
            .handshake(0.0, PROTOCOL_VERSION + 1, "invite", "p", "content-a")
            .unwrap_err();
        assert_eq!(
            err,
            StoreError::ProtocolMismatch(PROTOCOL_VERSION + 1, PROTOCOL_VERSION)
        );
        assert!(err.msg().contains(&PROTOCOL_VERSION.to_string()));
        // Wrong invite token.
        assert_eq!(
            store
                .handshake(0.0, PROTOCOL_VERSION, "wrong", "p", "content-a")
                .unwrap_err(),
            StoreError::BadInvite
        );
        // First success pins the (unconfigured) content identity...
        assert!(store.content_identity().is_none());
        let (id, token) = open_session(&mut store, 0.0, "alice");
        assert_ne!(id, token, "public id and secret token are distinct");
        assert_eq!(store.content_identity(), Some("content-a"));
        // ...and a mismatching later client is rejected, naming both.
        let err = store
            .handshake(1.0, PROTOCOL_VERSION, "invite", "q", "content-b")
            .unwrap_err();
        assert!(matches!(err, StoreError::ContentMismatch(..)));
        assert!(err.msg().contains("content-a") && err.msg().contains("content-b"));
    }

    #[test]
    fn leases_renew_on_activity_and_expiry_releases_locks_and_invalidates_the_token() {
        let mut store = UniverseStore::new(config());
        let (a_id, a_token) = open_session(&mut store, 0.0, "alice");
        let (_b_id, b_token) = open_session(&mut store, 0.0, "bob");
        let record = sample_record(10.0);
        let vessel_id = record.vessel_id.clone();
        store.publish(0.0, &a_token, record).expect("publish");
        // Activity at half-TTL keeps Alice alive past the original TTL window;
        // Bob heartbeats regularly throughout.
        store.heartbeat(15.0, &a_token).expect("renewal");
        store.heartbeat(25.0, &b_token).expect("bob");
        store.heartbeat(40.0, &a_token).expect("alice alive at 40");
        store.heartbeat(50.0, &b_token).expect("bob");
        store.heartbeat(75.0, &b_token).expect("bob");
        // The sweep at t=80 (router entry in real traffic) expires the now-idle
        // Alice (last seen 40, TTL 30): her lock releases, the record stands.
        store.sweep(80.0);
        store.heartbeat(80.0, &b_token).expect("bob alive");
        let (records, _) = store.fetch_since(80.0, &b_token, 0).expect("fetch");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].vessel_id, vessel_id);
        assert_eq!(
            records[0].authority, None,
            "expired lease released the lock"
        );
        assert!(
            !records[0].is_tombstone(),
            "the last published record stands"
        );
        // Alice's token is dead; she must re-handshake.
        assert_eq!(
            store.heartbeat(81.0, &a_token).unwrap_err(),
            StoreError::Unauthorized
        );
        let _ = a_id;
    }

    #[test]
    fn anchors_derive_current_time_and_reject_backward_sync() {
        let mut store = UniverseStore::new(config());
        let (_, token) = open_session(&mut store, 0.0, "alice");
        // Anchor at universe time 1000, rate 100 (warp), reported at now=10.
        store.report_anchor(10.0, &token, 1000.0, 100.0).unwrap();
        let reg = store.registry(20.0);
        assert_eq!(reg.len(), 1);
        // 10 wall-seconds later at rate 100 => 1000 + 1000.
        assert_eq!(reg[0].1, Some(2000.0));
        // Pause (rate 0) at the derived moment: time stands still.
        store.report_anchor(20.0, &token, 2000.0, 0.0).unwrap();
        assert_eq!(store.registry(44.0)[0].1, Some(2000.0));
        // Backward report is causality-rejected, naming both times (within the
        // lease window: last activity 20, TTL 30).
        let err = store.report_anchor(45.0, &token, 500.0, 1.0).unwrap_err();
        assert_eq!(
            err,
            StoreError::BackwardSync {
                reported: 500.0,
                previous: 2000.0
            }
        );
        assert!(err.msg().contains("500") && err.msg().contains("2000"));
    }

    #[test]
    fn publish_enforces_authority_stamps_hygiene_and_cursor_semantics() {
        let mut store = UniverseStore::new(config());
        let (a_id, a_token) = open_session(&mut store, 0.0, "alice");
        let (_b_id, b_token) = open_session(&mut store, 0.0, "bob");

        // First publish grants the publisher authority.
        let record = sample_record(10.0);
        let vessel_id = record.vessel_id.clone();
        let c1 = store.publish(1.0, &a_token, record.clone()).unwrap();
        assert_eq!(c1, 1);
        let (fetched, cursor) = store.fetch_since(1.0, &a_token, 0).unwrap();
        assert_eq!(cursor, 1);
        assert_eq!(fetched[0].authority.as_deref(), Some(a_id.as_str()));

        // A second session cannot write the held vessel.
        let mut update = record.clone();
        update.stamp = 20.0;
        assert_eq!(
            store.publish(2.0, &b_token, update.clone()).unwrap_err(),
            StoreError::NotAuthority
        );
        // Rejected writes do not advance the cursor.
        assert_eq!(store.status_counts().2, 1);

        // Stale stamp rejected (no past edits).
        let mut stale = record.clone();
        stale.stamp = 5.0;
        assert!(matches!(
            store.publish(3.0, &a_token, stale).unwrap_err(),
            StoreError::StaleStamp { .. }
        ));

        // R4: a non-finite motion value is rejected.
        let mut bad = record.clone();
        if let sounding_sim::vessel::MotionState::Conic { orbit, .. } = &mut bad.motion {
            orbit.semi_major_axis = f64::NAN;
        }
        bad.stamp = 30.0;
        assert_eq!(
            store.publish(4.0, &a_token, bad).unwrap_err(),
            StoreError::NonFinite
        );

        // The holder updates; newest-per-vessel stays bounded (one row).
        let c2 = store.publish(5.0, &a_token, update).unwrap();
        assert_eq!(c2, 2);
        let (all, _) = store.fetch_since(5.0, &a_token, 0).unwrap();
        assert_eq!(all.len(), 1, "one current record per vessel");
        assert_eq!(all[0].stamp, 20.0);

        // Fetch-since returns exactly the delta.
        let (none, c) = store.fetch_since(5.0, &a_token, c2).unwrap();
        assert!(none.is_empty());
        assert_eq!(c, c2);

        // Tombstone: terminal, releases authority, is fetchable as the delta.
        let mut tomb = sample_record(25.0);
        tomb.vessel_id = vessel_id.clone();
        tomb.fate = Some(Fate::Destroyed);
        let c3 = store.publish(6.0, &a_token, tomb).unwrap();
        let (delta, _) = store.fetch_since(6.0, &b_token, c2).unwrap();
        assert_eq!(delta.len(), 1);
        assert!(delta[0].is_tombstone());
        assert_eq!(delta[0].authority, None, "tombstone releases authority");
        // Writes after the tombstone are rejected: terminal means terminal.
        let mut after = sample_record(40.0);
        after.vessel_id = vessel_id;
        assert_eq!(
            store.publish(7.0, &a_token, after).unwrap_err(),
            StoreError::Tombstoned
        );
        let _ = c3;
    }

    #[test]
    fn future_records_are_invisible_to_past_observers() {
        let mut store = UniverseStore::new(config());
        let (_, token) = open_session(&mut store, 0.0, "alice");
        let record = sample_record(1000.0);
        store.publish(0.0, &token, record).unwrap();
        // An observer at t=500 (behind the stamp) does not see the vessel...
        assert!(store.visible_records(500.0).is_empty());
        // ...an observer at/past the stamp does (sync forward ⇒ it appears).
        assert_eq!(store.visible_records(1000.0).len(), 1);
        assert_eq!(store.visible_records(2000.0).len(), 1);
    }

    #[test]
    fn handoff_release_claim_transfer_with_the_causality_guard() {
        let mut store = UniverseStore::new(config());
        let (_a_id, a_token) = open_session(&mut store, 0.0, "alice");
        let (b_id, b_token) = open_session(&mut store, 0.0, "bob");

        let record = sample_record(1000.0);
        let vessel_id = record.vessel_id.clone();
        store.publish(0.0, &a_token, record).unwrap();

        // Claim while held is rejected.
        store.report_anchor(0.0, &b_token, 2000.0, 1.0).unwrap();
        assert_eq!(
            store.claim(1.0, &b_token, &vessel_id).unwrap_err(),
            StoreError::AlreadyHeld
        );

        // Alice releases; the record stands, unheld.
        store.release(2.0, &a_token, &vessel_id).unwrap();

        // A claimant with no anchor is rejected; a claimant behind the record
        // stamp is rejected by the causality guard (sync forward first).
        let (_c_id, c_token) = open_session(&mut store, 2.0, "carol");
        assert_eq!(
            store.claim(3.0, &c_token, &vessel_id).unwrap_err(),
            StoreError::NoAnchor
        );
        store.report_anchor(3.0, &c_token, 500.0, 1.0).unwrap();
        assert!(matches!(
            store.claim(3.0, &c_token, &vessel_id).unwrap_err(),
            StoreError::ClaimBehindRecord { .. }
        ));
        // Carol syncs forward (anchor ahead of the stamp) and claims.
        store.report_anchor(4.0, &c_token, 1500.0, 1.0).unwrap();
        store
            .claim(4.0, &c_token, &vessel_id)
            .expect("claim after sync");

        // Carol can now publish; Bob cannot.
        let mut update = sample_record(1600.0);
        update.vessel_id = vessel_id.clone();
        assert_eq!(
            store.publish(5.0, &b_token, update.clone()).unwrap_err(),
            StoreError::NotAuthority
        );
        store
            .publish(5.0, &c_token, update)
            .expect("holder publishes");

        // Transfer to a target behind the stamp is rejected; to a synced
        // target it lands atomically.
        store.report_anchor(6.0, &b_token, 2000.0, 1.0).unwrap();
        store
            .transfer(6.0, &c_token, &vessel_id, &b_id)
            .expect("transfer to synced target");
        let (records, _) = store.fetch_since(6.0, &b_token, 0).unwrap();
        assert_eq!(records[0].authority.as_deref(), Some(b_id.as_str()));
    }

    #[test]
    fn restart_from_vessels_restores_records_clears_authority_and_resequences() {
        let mut store = UniverseStore::new(config());
        let (_, token) = open_session(&mut store, 0.0, "alice");
        let r1 = sample_record(10.0);
        let r2 = sample_record(20.0);
        store.publish(0.0, &token, r1.clone()).unwrap();
        store.publish(1.0, &token, r2.clone()).unwrap();
        let saved = store.vessels();
        assert_eq!(saved.len(), 2);
        assert!(saved.iter().all(|r| r.authority.is_some()));

        let restored = UniverseStore::from_vessels(config(), saved.clone());
        let vessels = restored.vessels();
        assert_eq!(vessels.len(), 2);
        assert!(
            vessels.iter().all(|r| r.authority.is_none()),
            "sessions did not survive the restart; authority cleared (R5)"
        );
        // Everything else round-trips exactly.
        for (a, b) in vessels.iter().zip(saved.iter()) {
            let mut expect = b.clone();
            expect.authority = None;
            assert_eq!(a, &expect);
        }
        assert_eq!(restored.status_counts().2, 2, "records freshly sequenced");
    }
}

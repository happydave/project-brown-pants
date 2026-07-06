//! The **resident participant** core (WI 858): a presence-only headless peer —
//! the multiplayer design's Forward Note deployed for the first time.
//!
//! It joins a universe server as an **ordinary session** (the same
//! [`SyncHandle`] the windowed app uses; a peer cannot tell it from a human
//! client), publishes one craft's rails record, anchors its subspace at rate 1
//! (wall-clock time), and stays alive on the worker's heartbeats — the
//! solo-testing second player and CI/soak peer. Deliberately **presence-only**:
//! no NPC logic, no autopilot, no reaction to peers beyond logging (NPC
//! hosting is WI 859's future design pass).
//!
//! Exit semantics: protocol v1 has no logout **by design** — stopping simply
//! lets the lease lapse, releasing the vessel's authority with its last record
//! standing (stale-but-claimable, R5).

use crate::sync::{SyncCommand, SyncHandle, SyncShared, SyncStatus};
use crate::NetConfig;
use sounding_sim::frame::WorldPos;
use sounding_sim::persist::CraftSubgraph;
use sounding_sim::vessel::{mint_vessel_id, MotionState, VesselRecord};
use sounding_sim::voxel::VoxelCraft;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for one participant.
pub struct ParticipantConfig {
    /// Connection + identity (content identity must match the game clients').
    pub net: NetConfig,
    /// Vessel display name (the ghost's label on peers' screens).
    pub vessel_name: String,
    /// The craft structure (typically a shipped blueprint).
    pub craft: VoxelCraft,
    /// Rails motion: a parked surface fix (the visible solo-test default) or a
    /// canned conic (`--orbit`).
    pub motion: MotionState,
    /// Local-time tick period (sub-second; tests may shrink it).
    pub tick: Duration,
    /// Print status transitions + peer join/leave lines to stdout (the soak
    /// log; the binary turns this on, tests leave it off).
    pub log: bool,
}

/// A running participant. Dropping without [`ParticipantHandle::shutdown`]
/// leaves the threads running until process exit (the run-forever binary path).
pub struct ParticipantHandle {
    /// The minted vessel instance id (printed by the binary; future M3 tests
    /// claim it by hand).
    pub vessel_id: String,
    /// The sync worker's shared state (status, ghost store, local time).
    pub shared: Arc<Mutex<SyncShared>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ParticipantHandle {
    /// Starts the participant: session, own-vessel record, rate-1 anchor, and
    /// the local-time tick loop.
    pub fn start(config: ParticipantConfig) -> Self {
        let vessel_id = mint_vessel_id();
        let player = config.net.player.clone();
        let handle = SyncHandle::start(config.net);
        let shared = handle.shared.clone();
        shared
            .lock()
            .expect("participant shared")
            .ghosts
            .mark_own(&vessel_id);

        // The subspace anchor: local time zero now, advancing at rate 1
        // (wall-clock — a participant never warps).
        handle.send(SyncCommand::Anchor {
            universe_time: 0.0,
            rate: 1.0,
        });

        // The one record: an ordinary vessel record, stamped at local zero.
        let reference_position = match &config.motion {
            MotionState::SurfaceFix { position } => *position,
            MotionState::Conic { frame, orbit } => {
                let (p, _) = orbit.position_velocity(0.0);
                WorldPos::new(*frame, glam::DVec3::new(p.x, p.y, 0.0))
            }
        };
        let mut structure = CraftSubgraph::new(
            vessel_id.clone(),
            config.vessel_name.clone(),
            reference_position,
            config.craft,
        );
        structure.vessel_id = Some(vessel_id.clone());
        let record = VesselRecord {
            vessel_id: vessel_id.clone(),
            name: config.vessel_name,
            owner: player,
            authority: None,
            subspace: None,
            stamp: 0.0,
            structure,
            motion: config.motion,
            live: false,
            fate: None,
        };
        handle.send(SyncCommand::Publish(Box::new(record)));

        // The tick loop: wall-clock local time at rate 1, ghost advancement,
        // and (optionally) the soak log.
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let shared2 = shared.clone();
        let tick = config.tick;
        let log = config.log;
        let thread = std::thread::spawn(move || {
            let t0 = Instant::now();
            let mut known: HashSet<String> = HashSet::new();
            let mut last_status: Option<SyncStatus> = None;
            while !stop2.load(Ordering::SeqCst) {
                let t = t0.elapsed().as_secs_f64();
                let mut reconnected = false;
                {
                    let mut s = shared2.lock().expect("participant shared");
                    s.local_time = t;
                    s.ghosts.advance(t);
                    if last_status.as_ref() != Some(&s.status) {
                        if log {
                            println!("participant: {:?}", s.status);
                        }
                        reconnected = matches!(s.status, SyncStatus::Connected { .. });
                        last_status = Some(s.status.clone());
                    }
                    if log {
                        let now: HashSet<String> = s
                            .ghosts
                            .visible(t)
                            .iter()
                            .map(|p| {
                                format!(
                                    "{} ({} · {})",
                                    p.record.name, p.record.owner, p.record.vessel_id
                                )
                            })
                            .collect();
                        for joined in now.difference(&known) {
                            println!("participant: peer joined — {joined}");
                        }
                        for left in known.difference(&now) {
                            println!("participant: peer left — {left}");
                        }
                        known = now;
                    }
                }
                // On every (re)connection, refresh the anchor at the *current*
                // local time: the worker's replay carries the initial anchor
                // (universe time ~0), which after a mid-run reconnect would
                // leave the registry's derived time wrong for a long-running
                // peer. Non-decreasing within the fresh session, so always
                // legal (code-review fix).
                if reconnected {
                    handle.send(SyncCommand::Anchor {
                        universe_time: t,
                        rate: 1.0,
                    });
                }
                std::thread::sleep(tick);
            }
            // Stop the worker; the lease lapses server-side (no logout op —
            // R5 is the exit path).
            handle.shutdown();
        });

        Self {
            vessel_id,
            shared,
            stop,
            thread: Some(thread),
        }
    }

    /// Stops the tick loop and joins everything. The server-side lease then
    /// expires on its own schedule (stale-but-claimable, R5).
    pub fn shutdown(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

//! Content-aware world-save persistence (WI 553, content aspect breadth;
//! subsumes first-playable's deferred 537).
//!
//! A world save is an ordinary [`crate::persist`] document
//! ([`Kind::WorldSave`](crate::persist::Kind) / [`WorldPayload`]) whose
//! additive `scenario` member records everything a played `-- scenario`
//! flight needs to resume: the **resolved-content identity** (scenario id +
//! version, pack (id, version) pairs, and the composed frozen settings map —
//! the WI 800 doctrine's semi-public scalar contract), the **progress**
//! (session, elapsed time, per-mission lifecycle + latch trees, the last lore
//! beat), and the **flight state** (the serializable [`FlightCraft`] plus the
//! dynamic body state and launch pad).
//!
//! Mass and inertia are deliberately **not** saved: the flight stepper
//! re-derives wet mass from the craft's reservoirs every step and inertia
//! from the lattice, so rebuilding from the restored craft is exact by
//! construction — saving them could only disagree.
//!
//! **Migration belongs to the content build pass** (the content design's
//! persistence rule): loading a save re-runs the same deterministic scenario
//! resolution against the *recorded scenario id*; current content stands.
//! The recorded identity exists to make drift detectable — [`drift_report`]
//! names every changed pack version, scenario version, and settings scalar.
//!
//! The module also carries the **world library** — `saves/worlds/`, the
//! fourth member of the crafts / bodies / systems persistence family, with
//! the family's idioms: slug-of-name file identity, wrong-scope rejection,
//! skip-don't-abort discovery.

use crate::body_asset::BodyAsset;
use crate::body_digest::{digest_hex, BODY_OUTPUT_VERSION};
use crate::content::Setting;
use crate::director::ScenarioFlight;
use crate::flight::FlightCraft;
use crate::launch::LaunchPad;
use crate::library::{slugify, LibraryError};
use crate::mission::{MissionState, NodeState};
use crate::persist::{FormatError, Payload, SavedDocument, WorldPayload};
use crate::scenario::Scenario;
use crate::session::GameSession;
use crate::vessel::VesselRecord;
use glam::{DQuat, DVec3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// The save shapes (format surface — additive-only evolution).
// ---------------------------------------------------------------------------

/// One composed pack's identity as recorded in a save.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackIdentity {
    /// Pack id (`content/packs/<id>.ron`).
    pub id: String,
    /// The pack's own (opaque) version string.
    pub version: String,
}

/// The resolved-content identity a save was built from: enough to re-resolve
/// the same composition and to name any drift when the content has moved.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContentIdentity {
    /// Scenario id (`content/scenarios/<id>.ron`) — the re-resolution key.
    pub scenario_id: String,
    /// The scenario document's own version, if authored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario_version: Option<String>,
    /// Composed packs, (id, version), in resolution order.
    #[serde(default)]
    pub packs: Vec<PackIdentity>,
    /// The composed frozen balance scalars (name → factor + rationale) —
    /// recorded per the WI 800 doctrine (scalar names are a semi-public
    /// telemetry/save contract).
    #[serde(default)]
    pub settings: BTreeMap<String, Setting>,
}

impl ContentIdentity {
    /// The identity of a loaded scenario's composition.
    pub fn from_scenario(s: &Scenario) -> ContentIdentity {
        ContentIdentity {
            scenario_id: s.id.clone(),
            scenario_version: s.version.clone(),
            packs: s
                .catalog
                .packs
                .iter()
                .map(|(id, version)| PackIdentity {
                    id: id.clone(),
                    version: version.clone(),
                })
                .collect(),
            settings: s.catalog.settings.clone(),
        }
    }
}

/// The dynamic rigid-body state of the flight. Mass/inertia are rebuilt from
/// the restored craft (see the module docs) — only the true dynamic degrees
/// of freedom are format surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedBodyState {
    /// Position relative to the attractor, m.
    pub position: DVec3,
    /// Velocity, m/s.
    pub velocity: DVec3,
    /// Orientation (body → world).
    pub orientation: DQuat,
    /// World-frame angular momentum.
    pub angular_momentum: DVec3,
}

/// One celestial body's world-save record (WI 891, design-review N1): the
/// **tier is a stored field**, chosen at save time — there is no per-body
/// player-progression model yet to derive it from, and Starbound's shipped
/// per-object policy is the survey's model. A body absent from the list is
/// the *virtual* tier: nothing stored, a pure function of recipe+seed
/// (today's behavior, unchanged).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tier", rename_all = "snake_case")]
pub enum SavedBodyRecord {
    /// The settled tier: the full resolved body is **pinned**. Load uses it
    /// verbatim, bypassing resolve/derive unconditionally; a moved generator
    /// is reported, never silently swapped in (plan invariant).
    Snapshot {
        /// The body-asset id (the System's reference key).
        id: String,
        /// [`BODY_OUTPUT_VERSION`] at save time — a difference at load means
        /// the snapshot pins against a moved generator (a drift line).
        output_version: u32,
        /// `digest_hex` of `body` at save time — the **record-integrity**
        /// check: recomputed over the loaded snapshot itself (no regeneration),
        /// a mismatch is a corrupt/hand-edited record and a typed load error.
        digest: String,
        /// The pinned resolved body. Boxed so the rare snapshot variant does
        /// not inflate every record (the crate's `large_enum_variant`
        /// discipline).
        body: Box<BodyAsset>,
    },
    /// The visited tier: fingerprint only. The body regenerates through the
    /// normal resolve/derive path; the recorded digest detects drift —
    /// same-version mismatch reads as unintended drift, a version move as the
    /// expected reroll. Load proceeds with the regenerated body either way.
    Digest {
        /// The body-asset id.
        id: String,
        /// [`BODY_OUTPUT_VERSION`] at save time.
        output_version: u32,
        /// `digest_hex` of the resolved body at save time.
        digest: String,
    },
}

impl SavedBodyRecord {
    /// The body-asset id this record covers.
    pub fn id(&self) -> &str {
        match self {
            SavedBodyRecord::Snapshot { id, .. } | SavedBodyRecord::Digest { id, .. } => id,
        }
    }
}

/// A typed per-body record failure at load (WI 891). Loud and naming the
/// offender; never a silent fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BodyRecordError {
    /// A snapshot-tier record failed its integrity check: the stored body no
    /// longer hashes to the stored digest (corrupt or hand-edited record).
    SnapshotIntegrity {
        /// The record's body id.
        id: String,
    },
    /// The per-body list names the same id twice — malformed.
    DuplicateId {
        /// The duplicated body id.
        id: String,
    },
}

impl fmt::Display for BodyRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BodyRecordError::SnapshotIntegrity { id } => write!(
                f,
                "world-save body snapshot `{id}` fails its integrity digest \
                 (corrupt or hand-edited record)"
            ),
            BodyRecordError::DuplicateId { id } => {
                write!(f, "world-save body records name `{id}` twice")
            }
        }
    }
}

/// One mission's saved runtime state. The definition is deliberately **not**
/// saved — it re-resolves from current content on load (the migration
/// posture); state applies by id where the objective's shape still matches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MissionSave {
    /// Mission id (the reconciliation key).
    pub id: String,
    /// Lifecycle state.
    pub state: MissionState,
    /// The objective latch tree (monotone progress — a half-completed
    /// Sequence resumes mid-sequence).
    pub nodes: NodeState,
}

/// The played scenario's savable state — [`WorldPayload::scenario`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioSaveState {
    /// What content this save was resolved from.
    pub content: ContentIdentity,
    /// Integrated flight sim time at capture, s.
    pub elapsed: f64,
    /// The played session (phase + outcome) — restored verbatim, so a
    /// terminal Recovery save resumes frozen, as it played.
    pub session: GameSession,
    /// The most recent lore beat, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lore: Option<String>,
    /// Felt acceleration at capture, g (telemetry continuity).
    pub g_force: f64,
    /// The full craft state: voxels, propulsion (live reservoir levels),
    /// attitude/SAS, control, autopilot.
    pub craft: FlightCraft,
    /// The dynamic body state.
    pub body: SavedBodyState,
    /// Launch-pad state (rest radius + release).
    pub pad: LaunchPad,
    /// Per-mission runtime state, in save order.
    #[serde(default)]
    pub missions: Vec<MissionSave>,
    /// Per-body snapshot/digest records (WI 891, additive — absent in older
    /// saves and absent when empty, so pre-891 documents are byte-unchanged).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bodies: Vec<SavedBodyRecord>,
}

// ---------------------------------------------------------------------------
// Capture.
// ---------------------------------------------------------------------------

/// Captures a running scenario flight as a world-save payload, carrying
/// `vessels` through **uninterpreted** (the multiplayer J1 table: a resumed
/// session writes back what it loaded, so an edit-resume-save cycle never
/// drops a vessel record).
pub fn capture(flight: &ScenarioFlight, vessels: Vec<VesselRecord>) -> WorldPayload {
    WorldPayload {
        vessels,
        scenario: Some(Box::new(ScenarioSaveState {
            content: flight.content.clone(),
            elapsed: flight.elapsed,
            session: flight.session,
            lore: flight.lore.clone(),
            g_force: flight.g_force,
            craft: flight.craft.clone(),
            body: SavedBodyState {
                position: flight.body.position,
                velocity: flight.body.velocity,
                orientation: flight.body.orientation,
                angular_momentum: flight.body.angular_momentum,
            },
            pad: flight.pad,
            missions: flight
                .missions
                .iter()
                .map(|m| MissionSave {
                    id: m.def.id.clone(),
                    state: m.state,
                    nodes: m.nodes.clone(),
                })
                .collect(),
            bodies: flight.bodies.clone(),
        })),
    }
}

// ---------------------------------------------------------------------------
// Per-body record policy (WI 891, design-review N1).
// ---------------------------------------------------------------------------

/// Builds the per-body records for a resolved world: ids in `snapshot_ids`
/// get the snapshot tier, every other body the digest tier. The production
/// default assignment (the root/central body snapshotted, the rest digested)
/// is the caller's — `ScenarioSpawn::from_scenario` supplies it — because the
/// tier is a save-time choice, not derivable world state.
pub fn body_records(assets: &[BodyAsset], snapshot_ids: &BTreeSet<String>) -> Vec<SavedBodyRecord> {
    assets
        .iter()
        .map(|a| {
            let digest = digest_hex(a);
            if snapshot_ids.contains(&a.id) {
                SavedBodyRecord::Snapshot {
                    id: a.id.clone(),
                    output_version: BODY_OUTPUT_VERSION,
                    digest,
                    body: Box::new(a.clone()),
                }
            } else {
                SavedBodyRecord::Digest {
                    id: a.id.clone(),
                    output_version: BODY_OUTPUT_VERSION,
                    digest,
                }
            }
        })
        .collect()
}

/// The load path (WI 891): applies saved per-body records to a resolved asset
/// list **before** `System::compile` (the survey's verify-before-wire rule).
///
/// - **Snapshot tier**: two checks, neither running resolve/derive — the
///   record's integrity digest (recomputed over the stored snapshot itself;
///   a mismatch is the typed error), then the substitution: the snapshot
///   replaces the same-id asset unconditionally. A recorded output version
///   differing from this build's is a drift line (the snapshot pins against a
///   moved generator — never silently swapped).
/// - **Digest tier**: the asset in the list is already the regenerated body;
///   its digest is compared — same-version mismatch reads as unintended
///   drift, a version move as the expected reroll. Both are drift lines; load
///   proceeds with the regenerated body.
/// - A record whose id is not among the assets is a drift line and otherwise
///   inert; a duplicated id is malformed (typed error).
///
/// Returns the drift lines (empty = silent resume).
pub fn apply_body_records(
    records: &[SavedBodyRecord],
    assets: &mut [BodyAsset],
) -> Result<Vec<String>, BodyRecordError> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut drift = Vec::new();
    for record in records {
        if !seen.insert(record.id()) {
            return Err(BodyRecordError::DuplicateId {
                id: record.id().to_string(),
            });
        }
        let slot = assets.iter_mut().find(|a| a.id == record.id());
        match (record, slot) {
            (_, None) => drift.push(format!(
                "body record `{}` names a body not in the resolved world",
                record.id()
            )),
            (
                SavedBodyRecord::Snapshot {
                    id,
                    output_version,
                    digest,
                    body,
                },
                Some(slot),
            ) => {
                if digest_hex(body) != *digest {
                    return Err(BodyRecordError::SnapshotIntegrity { id: id.clone() });
                }
                if *output_version != BODY_OUTPUT_VERSION {
                    drift.push(format!(
                        "body `{id}` snapshot pins output version {output_version} \
                         (this build generates {BODY_OUTPUT_VERSION})"
                    ));
                }
                *slot = (**body).clone();
            }
            (
                SavedBodyRecord::Digest {
                    id,
                    output_version,
                    digest,
                },
                Some(slot),
            ) => {
                let computed = digest_hex(slot);
                if *output_version != BODY_OUTPUT_VERSION {
                    drift.push(format!(
                        "body `{id}` regenerated under output version {BODY_OUTPUT_VERSION} \
                         (saved at {output_version}) — expected reroll"
                    ));
                } else if computed != *digest {
                    drift.push(format!(
                        "body `{id}` regenerated with digest {computed}, but {digest} was \
                         recorded — unintended generator drift"
                    ));
                }
            }
        }
    }
    Ok(drift)
}

/// Carries snapshot pins across a resume (WI 891): a freshly rebuilt record
/// list would re-stamp a pinned snapshot with this build's output version,
/// silently erasing the "pinned against a moved generator" marker on the next
/// save. So a loaded **snapshot** record replaces its same-id fresh record
/// verbatim; loaded digest records are superseded by the fresh ones (an
/// accepted reroll updates the fingerprint). Records for bodies no longer in
/// the world are dropped (they were already named in the load drift report).
pub fn reconcile_body_records(
    fresh: Vec<SavedBodyRecord>,
    saved: &[SavedBodyRecord],
) -> Vec<SavedBodyRecord> {
    fresh
        .into_iter()
        .map(|record| {
            let pinned = saved
                .iter()
                .find(|s| s.id() == record.id() && matches!(s, SavedBodyRecord::Snapshot { .. }));
            match (pinned, &record) {
                (Some(pin), SavedBodyRecord::Snapshot { .. }) => pin.clone(),
                _ => record,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Drift reporting (the honesty layer over build-pass migration).
// ---------------------------------------------------------------------------

/// Names every difference between a save's recorded content identity and the
/// re-resolved one: scenario version, pack versions (changed / added /
/// removed), and settings scalars (factor changed / added / removed).
/// Empty means the identity is unchanged and the resume is silent. Scalar
/// rationale is display prose, not balance — deliberately not compared.
pub fn drift_report(recorded: &ContentIdentity, current: &ContentIdentity) -> Vec<String> {
    let mut report = Vec::new();
    if recorded.scenario_version != current.scenario_version {
        report.push(format!(
            "scenario `{}` version {} -> {}",
            current.scenario_id,
            opt(&recorded.scenario_version),
            opt(&current.scenario_version),
        ));
    }
    let rec: BTreeMap<&str, &str> = recorded
        .packs
        .iter()
        .map(|p| (p.id.as_str(), p.version.as_str()))
        .collect();
    let cur: BTreeMap<&str, &str> = current
        .packs
        .iter()
        .map(|p| (p.id.as_str(), p.version.as_str()))
        .collect();
    for (id, v) in &rec {
        match cur.get(id) {
            Some(now) if now != v => report.push(format!("pack `{id}` version {v} -> {now}")),
            None => report.push(format!("pack `{id}` (version {v}) no longer composed")),
            _ => {}
        }
    }
    for (id, v) in &cur {
        if !rec.contains_key(id) {
            report.push(format!("pack `{id}` (version {v}) newly composed"));
        }
    }
    for (name, s) in &recorded.settings {
        match current.settings.get(name) {
            Some(now) if now.factor != s.factor => report.push(format!(
                "setting `{name}` factor {} -> {}",
                s.factor, now.factor
            )),
            None => report.push(format!("setting `{name}` (factor {}) removed", s.factor)),
            _ => {}
        }
    }
    for (name, s) in &current.settings {
        if !recorded.settings.contains_key(name) {
            report.push(format!("setting `{name}` (factor {}) added", s.factor));
        }
    }
    report
}

fn opt(v: &Option<String>) -> String {
    v.clone().unwrap_or_else(|| "(unversioned)".to_string())
}

// ---------------------------------------------------------------------------
// The world library — `saves/worlds/`, the persistence family's fourth member.
// ---------------------------------------------------------------------------

/// A discovered world save.
#[derive(Clone, Debug)]
pub struct WorldEntry {
    /// File-stem slug (the slot identity).
    pub slug: String,
    /// Full path.
    pub path: PathBuf,
    /// Last-modified time, when the filesystem reports one.
    pub modified: Option<SystemTime>,
}

fn world_path(dir: &Path, slug: &str) -> PathBuf {
    dir.join(format!("{slug}.json"))
}

/// Saves `payload` into `dir` under the slug of `name` (the family's
/// slug-of-name slot semantics: same name updates one slot). Creates the
/// directory if needed; returns the written path.
pub fn save_world(dir: &Path, name: &str, payload: &WorldPayload) -> Result<PathBuf, LibraryError> {
    let mut slug = slugify(name);
    if slug.is_empty() {
        slug = "world".to_string();
    }
    std::fs::create_dir_all(dir).map_err(|e| LibraryError::Io(e.to_string()))?;
    let path = world_path(dir, &slug);
    let json = SavedDocument::new(Payload::WorldSave(payload.clone())).to_json()?;
    std::fs::write(&path, json).map_err(|e| LibraryError::Io(e.to_string()))?;
    Ok(path)
}

/// Reads a world-save document from `path`. Any non-world scope is rejected
/// as wrong (the family's rule, in the opposite direction from the craft
/// loaders).
pub fn load_world(path: &Path) -> Result<WorldPayload, LibraryError> {
    let bytes = std::fs::read_to_string(path).map_err(|e| LibraryError::Io(e.to_string()))?;
    world_from_document(&bytes)
}

/// Pure decode counterpart to [`load_world`].
pub fn world_from_document(json: &str) -> Result<WorldPayload, LibraryError> {
    match SavedDocument::from_json(json)?.payload {
        Payload::WorldSave(w) => Ok(w),
        _ => Err(LibraryError::Format(FormatError::Malformed(
            "expected a world-save document".to_string(),
        ))),
    }
}

/// Enumerates the world saves in `dir`, sorted by slug. A missing directory
/// yields an empty list; unreadable, malformed, or wrong-scope files are
/// skipped — discovery never aborts on one bad file.
pub fn list_worlds(dir: &Path) -> Vec<WorldEntry> {
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut entries: Vec<WorldEntry> = read
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|x| x.to_str()) != Some("json") {
                return None;
            }
            let slug = path.file_stem()?.to_str()?.to_string();
            let bytes = std::fs::read_to_string(&path).ok()?;
            world_from_document(&bytes).ok()?;
            let modified = e.metadata().ok().and_then(|m| m.modified().ok());
            Some(WorldEntry {
                slug,
                path,
                modified,
            })
        })
        .collect();
    entries.sort_by(|a, b| a.slug.cmp(&b.slug));
    entries
}

/// The most recently written world save: max by (modified time, slug) — the
/// slug tiebreak keeps equal-mtime results deterministic. `None` when the
/// directory holds no valid world saves.
pub fn latest_world(dir: &Path) -> Option<WorldEntry> {
    list_worlds(dir)
        .into_iter()
        .max_by(|a, b| (a.modified, &a.slug).cmp(&(b.modified, &b.slug)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vessel::MotionState;
    use crate::voxel::{Material, Voxel, VoxelCraft};
    use glam::IVec3;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A unique scratch directory under the OS temp dir (the library-test
    /// idiom), so tests don't collide and never write into the repo.
    fn scratch_dir(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("snd-world-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn identity(packs: &[(&str, &str)], settings: &[(&str, f64)]) -> ContentIdentity {
        ContentIdentity {
            scenario_id: "s".into(),
            scenario_version: Some("1".into()),
            packs: packs
                .iter()
                .map(|(id, v)| PackIdentity {
                    id: (*id).into(),
                    version: (*v).into(),
                })
                .collect(),
            settings: settings
                .iter()
                .map(|(n, f)| {
                    (
                        (*n).to_string(),
                        Setting {
                            factor: *f,
                            rationale: None,
                        },
                    )
                })
                .collect(),
        }
    }

    fn vessel() -> VesselRecord {
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        let origin = crate::frame::WorldPos {
            frame: crate::frame::FrameId::CENTRAL_BODY,
            pos: glam::DVec3::ZERO,
        };
        VesselRecord::from_surface(
            "v-1",
            "Buoy",
            "peer",
            0.0,
            crate::frame::WorldPos {
                frame: crate::frame::FrameId::CENTRAL_BODY,
                pos: glam::DVec3::new(0.0, 6.36e6, 0.0),
            },
            crate::persist::CraftSubgraph::new("buoy", "Buoy", origin, craft),
        )
    }

    /// A pre-553 world-save document (no `scenario` member) decodes, reads
    /// as no scenario state, and re-encodes without inventing the member.
    #[test]
    fn pre_553_world_save_decodes_and_round_trips() {
        let json = r#"{"format_version":2,"payload":{"kind":"world_save","vessels":[]}}"#;
        let doc = SavedDocument::from_json(json).expect("pre-553 world save decodes");
        let Payload::WorldSave(w) = &doc.payload else {
            panic!("world scope");
        };
        assert!(w.scenario.is_none());
        assert!(w.vessels.is_empty());
        let out = doc.to_json().expect("encodes");
        assert!(
            !out.contains("\"scenario\""),
            "absent member stays absent: {out}"
        );
    }

    /// The world library: save/load/list slot semantics, wrong-scope
    /// rejection both directions, skip-don't-abort discovery, and the
    /// (mtime, slug) latest rule's deterministic slug tiebreak.
    #[test]
    fn world_library_save_load_list_latest() {
        let dir = scratch_dir("lib");
        assert!(list_worlds(&dir).is_empty(), "missing dir lists empty");
        assert!(latest_world(&dir).is_none());

        let payload = WorldPayload {
            vessels: vec![vessel()],
            scenario: None,
        };
        let a = save_world(&dir, "alpha", &payload).expect("save alpha");
        let b = save_world(&dir, "beta", &payload).expect("save beta");
        assert_eq!(a, dir.join("alpha.json"));
        assert_eq!(b, dir.join("beta.json"));

        // Same name updates the one slot.
        let a2 = save_world(&dir, "alpha", &payload).expect("re-save alpha");
        assert_eq!(a, a2);

        let loaded = load_world(&a).expect("load");
        assert_eq!(loaded.vessels.len(), 1);
        assert_eq!(loaded.vessels[0].name, "Buoy");
        assert!(matches!(
            loaded.vessels[0].motion,
            MotionState::SurfaceFix { .. }
        ));

        // Wrong scope: a craft document in the directory is skipped by
        // discovery and rejected by the loader.
        let craft_doc = SavedDocument::new(Payload::Craft(crate::persist::CraftSubgraph::new(
            "c",
            "C",
            crate::frame::WorldPos {
                frame: crate::frame::FrameId::CENTRAL_BODY,
                pos: glam::DVec3::ZERO,
            },
            VoxelCraft::new(1.0),
        )))
        .to_json()
        .unwrap();
        std::fs::write(dir.join("craft.json"), craft_doc).unwrap();
        std::fs::write(dir.join("garbage.json"), "{not json").unwrap();
        assert!(load_world(&dir.join("craft.json")).is_err(), "wrong scope");
        assert!(load_world(&dir.join("garbage.json")).is_err(), "corrupt");

        let listed = list_worlds(&dir);
        assert_eq!(
            listed.iter().map(|e| e.slug.as_str()).collect::<Vec<_>>(),
            vec!["alpha", "beta"],
            "sorted, wrong-scope and garbage skipped"
        );

        // Latest: alpha was re-written last... but mtime granularity can tie
        // with beta; the slug tiebreak keeps the answer deterministic either
        // way. Force a strict ordering by touching beta afresh.
        save_world(&dir, "beta", &payload).expect("touch beta");
        let latest = latest_world(&dir).expect("some");
        assert!(
            latest.slug == "beta" || latest.modified.is_none(),
            "most recent (or deterministic fallback): {}",
            latest.slug
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// WI 891 (design-review N1): the builder assigns tiers from the caller's
    /// snapshot set; `apply_body_records` runs the two regeneration-free
    /// snapshot checks (integrity digest, output-version pin note) and the
    /// digest-tier comparison; drift lines follow the plan's taxonomy.
    #[test]
    fn body_records_policy_pins_verifies_and_reports_drift() {
        use crate::body_digest::{digest_hex, BODY_OUTPUT_VERSION};
        use crate::bodygen::{generate, Archetype};
        let root = BodyAsset::earthlike();
        let visited = generate(42, Archetype::RockyPlanet);
        let assets = vec![root.clone(), visited.clone()];
        let snapshot_ids: BTreeSet<String> = std::iter::once(root.id.clone()).collect();
        let records = body_records(&assets, &snapshot_ids);
        assert!(matches!(&records[0], SavedBodyRecord::Snapshot { id, .. } if *id == root.id));
        assert!(matches!(&records[1], SavedBodyRecord::Digest { id, .. } if *id == visited.id));

        // Clean apply: silent, nothing changed.
        let mut loaded = assets.clone();
        let drift = apply_body_records(&records, &mut loaded).unwrap();
        assert!(drift.is_empty(), "{drift:?}");
        assert_eq!(loaded, assets);

        // Scenario B3 (pin): a snapshot differing from the regenerated body
        // wins unconditionally — the substitution IS the derive bypass — and
        // a moved output version is a named drift line, never a swap.
        let mut pinned_body = root.clone();
        pinned_body.radius += 1000.0;
        let pin = SavedBodyRecord::Snapshot {
            id: root.id.clone(),
            output_version: BODY_OUTPUT_VERSION + 1,
            digest: digest_hex(&pinned_body),
            body: Box::new(pinned_body.clone()),
        };
        let mut loaded = assets.clone();
        let drift = apply_body_records(std::slice::from_ref(&pin), &mut loaded).unwrap();
        assert_eq!(loaded[0], pinned_body, "snapshot wins");
        assert_eq!(drift.len(), 1);
        assert!(drift[0].contains("pins output version"), "{}", drift[0]);

        // Scenario B3 (integrity): tampered snapshot content is typed.
        let bad = SavedBodyRecord::Snapshot {
            id: root.id.clone(),
            output_version: BODY_OUTPUT_VERSION,
            digest: "0".repeat(16),
            body: Box::new(pinned_body.clone()),
        };
        let mut loaded = assets.clone();
        assert!(matches!(
            apply_body_records(&[bad], &mut loaded),
            Err(BodyRecordError::SnapshotIntegrity { id }) if id == root.id
        ));

        // Scenario B2 (digest tier): same-version mismatch reads as
        // unintended drift; a version move as the expected reroll — the
        // regenerated body is kept in both cases.
        let wrong = SavedBodyRecord::Digest {
            id: visited.id.clone(),
            output_version: BODY_OUTPUT_VERSION,
            digest: "0".repeat(16),
        };
        let mut loaded = assets.clone();
        let drift = apply_body_records(&[wrong], &mut loaded).unwrap();
        assert!(
            drift[0].contains("unintended generator drift"),
            "{}",
            drift[0]
        );
        assert_eq!(loaded, assets, "regenerated body kept");
        let moved = SavedBodyRecord::Digest {
            id: visited.id.clone(),
            output_version: BODY_OUTPUT_VERSION + 1,
            digest: digest_hex(&visited),
        };
        let mut loaded = assets.clone();
        let drift = apply_body_records(&[moved], &mut loaded).unwrap();
        assert!(drift[0].contains("expected reroll"), "{}", drift[0]);

        // Unknown id: named, inert. Duplicate id: malformed, typed.
        let ghost = SavedBodyRecord::Digest {
            id: "ghost".into(),
            output_version: BODY_OUTPUT_VERSION,
            digest: "0".repeat(16),
        };
        let mut loaded = assets.clone();
        let drift = apply_body_records(std::slice::from_ref(&ghost), &mut loaded).unwrap();
        assert!(drift[0].contains("ghost"), "{}", drift[0]);
        assert_eq!(loaded, assets);
        let mut loaded = assets.clone();
        assert!(matches!(
            apply_body_records(&[ghost.clone(), ghost], &mut loaded),
            Err(BodyRecordError::DuplicateId { id }) if id == "ghost"
        ));
    }

    /// WI 889 (kept-body semantics, workitem AC 3): version-2-era records —
    /// the real predecessors of the batched stream break — load with the
    /// designed drift taxonomy. A snapshot's integrity digest verifies over
    /// the STORED body (no regeneration), so it loads unaffected, wins, and
    /// reports the version-pin drift line; a digest-tier record reports the
    /// expected reroll and proceeds with the version-3 regeneration.
    /// Regeneration at the old values is retired — nothing recreates them.
    #[test]
    fn version_two_era_records_load_with_the_designed_drift_taxonomy() {
        use crate::body_digest::{digest_hex, BODY_OUTPUT_VERSION};
        use crate::bodygen::{generate, Archetype};
        assert_eq!(BODY_OUTPUT_VERSION, 3, "this test narrates the 2 → 3 break");

        // Model the v2-era stored body as one the v3 generator provably does
        // not produce (the stream break moved every generated value).
        let current = generate(7, Archetype::OceanWorld);
        let mut v2_body = current.clone();
        v2_body.radius -= 12_345.0;
        v2_body.mu = v2_body.fluid_medium.gravity * v2_body.radius * v2_body.radius;

        // Snapshot tier: loads unaffected (derive bypassed), wins verbatim,
        // named version-pin drift line.
        let snap = SavedBodyRecord::Snapshot {
            id: current.id.clone(),
            output_version: 2,
            digest: digest_hex(&v2_body),
            body: Box::new(v2_body.clone()),
        };
        let mut loaded = vec![current.clone()];
        let drift = apply_body_records(std::slice::from_ref(&snap), &mut loaded).unwrap();
        assert_eq!(loaded[0], v2_body, "the v2 snapshot wins verbatim");
        assert_eq!(drift.len(), 1);
        assert!(drift[0].contains("pins output version"), "{}", drift[0]);

        // Digest tier: the v2 digest cannot match the v3 regeneration — the
        // expected reroll, and the regenerated (v3) body is kept.
        let dig = SavedBodyRecord::Digest {
            id: current.id.clone(),
            output_version: 2,
            digest: digest_hex(&v2_body),
        };
        let mut loaded = vec![current.clone()];
        let drift = apply_body_records(&[dig], &mut loaded).unwrap();
        assert!(drift[0].contains("expected reroll"), "{}", drift[0]);
        assert_eq!(loaded[0], current, "regeneration at old values is retired");
    }

    /// WI 891 resume reconciliation: a loaded snapshot pin replaces its fresh
    /// same-id record verbatim (the recorded output version survives the next
    /// save), digest records rebuild fresh, and records for absent bodies drop.
    #[test]
    fn reconcile_preserves_pins_and_refreshes_digests() {
        use crate::body_digest::{digest_hex, BODY_OUTPUT_VERSION};
        use crate::bodygen::{generate, Archetype};
        let root = BodyAsset::earthlike();
        let visited = generate(42, Archetype::RockyPlanet);
        let snapshot_ids: BTreeSet<String> = std::iter::once(root.id.clone()).collect();
        let fresh = body_records(&[root.clone(), visited.clone()], &snapshot_ids);

        let mut old_root = root.clone();
        old_root.radius += 1000.0;
        let saved = vec![
            SavedBodyRecord::Snapshot {
                id: root.id.clone(),
                output_version: BODY_OUTPUT_VERSION + 1,
                digest: digest_hex(&old_root),
                body: Box::new(old_root),
            },
            SavedBodyRecord::Digest {
                id: visited.id.clone(),
                output_version: BODY_OUTPUT_VERSION + 1,
                digest: "0".repeat(16),
            },
            SavedBodyRecord::Digest {
                id: "gone".into(),
                output_version: BODY_OUTPUT_VERSION,
                digest: "0".repeat(16),
            },
        ];
        let out = reconcile_body_records(fresh.clone(), &saved);
        assert_eq!(out.len(), 2, "absent-body record dropped");
        assert_eq!(out[0], saved[0], "snapshot pin preserved verbatim");
        assert_eq!(out[1], fresh[1], "digest record rebuilt fresh");
    }

    /// The drift report names every change class and stays silent on an
    /// identical identity. Rationale is deliberately not compared.
    #[test]
    fn drift_report_names_every_change_class() {
        let rec = identity(
            &[("core", "1"), ("gone", "2")],
            &[("eff", 2.0), ("old", 1.0)],
        );
        let mut cur = identity(
            &[("core", "3"), ("new", "1")],
            &[("eff", 5.0), ("add", 4.0)],
        );
        cur.scenario_version = Some("2".into());

        assert!(drift_report(&rec, &rec).is_empty(), "identical is silent");

        let report = drift_report(&rec, &cur);
        let text = report.join("\n");
        assert!(text.contains("scenario `s` version 1 -> 2"), "{text}");
        assert!(text.contains("pack `core` version 1 -> 3"), "{text}");
        assert!(
            text.contains("pack `gone` (version 2) no longer composed"),
            "{text}"
        );
        assert!(
            text.contains("pack `new` (version 1) newly composed"),
            "{text}"
        );
        assert!(text.contains("setting `eff` factor 2 -> 5"), "{text}");
        assert!(text.contains("setting `old` (factor 1) removed"), "{text}");
        assert!(text.contains("setting `add` (factor 4) added"), "{text}");
        assert_eq!(report.len(), 7, "nothing else: {text}");
    }
}

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
use std::collections::BTreeMap;
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
        })),
    }
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

//! Scenario document + loader (WI 550, content Slice 1).
//!
//! A **scenario** is the top-level authored content record that composes a
//! playthrough from things that already exist, owning none of them: the world
//! (world-building's [`System`] of [`BodyAsset`]s, by reference), the enabled
//! asset packs, the balance-settings documents (a scenario is a *source of*
//! settings documents — design doctrine, WI 800), its override sets, and the
//! starting state (a craft blueprint by slug, a placement, and the device
//! bindings that map the blueprint's device classes onto catalog records).
//!
//! Loading is validation: every reference must resolve or the load fails with
//! a typed error naming the offender — packs, settings docs, override sets,
//! the blueprint, the system, its body assets, and the device bindings, plus
//! the WI 549 detectors (`validate_body_refs`, seam violation, factor rules)
//! via [`Catalog::compose`]. The document is data; nothing here executes it.
//!
//! Missions (WI 551) are referenced by id like every other document
//! (`content/missions/<id>.ron`), parsed/validated at load, and carried on
//! the loaded scenario for the director's evaluator. The lore payload is
//! opaque.

use crate::body_asset::BodyAsset;
use crate::body_library;
use crate::content::{Catalog, ContentError, DeviceClass, DeviceSpec, OverridePhase, Record};
use crate::library::{self, LibraryError};
use crate::mission::{parse_mission, Mission, MissionError, Offer};
use crate::system::{CompileError, Placement, System};
use crate::system_library;
use crate::universe::Universe;
use crate::voxel::{DeviceKind, VoxelCraft};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

/// Scenario documents share the content document family's format version.
pub use crate::content::CONTENT_FORMAT_VERSION;

/// The slug of the built-in canonical body ([`BodyAsset::earthlike`]) — always
/// a known body reference even with an empty body library.
pub const EARTHLIKE_SLUG: &str = "earthlike";

// ---------------------------------------------------------------------------
// The document as authored (RON, `deny_unknown_fields` like all content).
// ---------------------------------------------------------------------------

/// A scenario document as authored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioDoc {
    /// Document *format* version — must equal [`CONTENT_FORMAT_VERSION`].
    pub format: u32,
    /// Stable identifier; when loaded by reference it must match the file stem.
    pub id: String,
    /// Human-facing display name.
    pub name: String,
    /// The world this scenario plays in (referenced, never defined here).
    pub world: WorldRef,
    /// Enabled asset packs, by id (`content/packs/<id>.ron`). List order is
    /// not semantic — composition order comes from the merge ladder.
    pub packs: Vec<String>,
    /// Balance-settings documents, by id (`content/settings/<id>.ron`).
    #[serde(default)]
    pub settings: Vec<String>,
    /// Override sets, by id (`content/overrides/<id>.ron`). Must declare
    /// phase `Patch` or `Scenario` — `Local` is reserved for the player.
    #[serde(default)]
    pub overrides: Vec<String>,
    /// The starting state.
    pub start: StartSpec,
    /// Mission documents, by id (`content/missions/<id>.ron`, WI 551) —
    /// resolved and validated at load like every other reference; offered in
    /// declared order (an `AfterMission` offer must name an earlier-or-any id
    /// in this list).
    #[serde(default)]
    pub missions: Vec<String>,
    /// Opaque narrative payload (the lore store arrives at WI 554).
    #[serde(default)]
    pub lore: Option<String>,
}

/// The scenario's world: consumed by reference from world-building.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub enum WorldRef {
    /// The built-in canonical single-body world ([`BodyAsset::earthlike`]).
    Earthlike,
    /// A saved [`System`] by slug (`saves/systems/<slug>.json`), its body
    /// assets resolved from the body library.
    System(String),
}

/// The starting state: one craft, where it starts, and how its device classes
/// bind to catalog records.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartSpec {
    /// Craft blueprint slug, resolved against `content/blueprints/` then
    /// `saves/crafts/` (both are persist craft-scope documents).
    pub blueprint: String,
    /// Where the craft starts. Extensible; only `Pad` exists in this slice.
    pub placement: StartPlacement,
    /// Device-class → catalog-record-id bindings. Each binding must resolve
    /// to a concrete device record of that class; bindings for classes the
    /// blueprint lacks are permitted (inert — scenarios may over-specify for
    /// reuse across blueprints). Engine and Tank are consumed this slice.
    #[serde(default)]
    pub bindings: BTreeMap<DeviceClass, String>,
}

/// Start placements. Additive by design: orbit / surface / water variants
/// arrive with the scenes that need them (convergence A2), never by reshaping
/// `Pad`. Serialize because the placement rides the staged spawn payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StartPlacement {
    /// At rest on the root body's launch pad (surface radius + craft CoM).
    Pad,
}

// ---------------------------------------------------------------------------
// Errors — every failure names its offender.
// ---------------------------------------------------------------------------

/// Why a scenario failed to load. Loud and typed, per the content layer's
/// correctness-detector obligation.
#[derive(Debug)]
pub enum ScenarioError {
    /// The scenario file (or a referenced document file) could not be read.
    Io { path: PathBuf, error: String },
    /// The scenario document failed to parse.
    Parse(String),
    /// The scenario declares an unknown format version.
    Format { found: u32 },
    /// The document's internal id does not match the requested reference.
    IdMismatch { expected: String, found: String },
    /// A referenced content document does not exist where its kind lives.
    MissingDocument { kind: &'static str, id: String },
    /// A scenario-referenced override set declares the reserved `Local` phase.
    LocalPhaseReserved { set: String },
    /// The composition failed (parse/merge/seam/detector errors from Slice 0).
    Content(ContentError),
    /// The referenced system does not exist in the system library.
    UnknownSystem { slug: String },
    /// The system references body assets the body library cannot supply.
    Compile(CompileError),
    /// A library document failed to load (blueprint / system / body asset).
    Library {
        what: &'static str,
        error: LibraryError,
    },
    /// The starting blueprint was not found in any blueprint location.
    MissingBlueprint { slug: String },
    /// A device binding names a record the catalog does not have (or an
    /// abstract base — those are not content).
    BindingUnknown { class: DeviceClass, id: String },
    /// A device binding resolves to a record of the wrong kind or class.
    BindingWrongClass { class: DeviceClass, id: String },
    /// The blueprint carries a device class this slice consumes, but the
    /// scenario binds no catalog record for it (no silent hardcodes).
    BindingMissing { class: DeviceClass },
    /// The scenario lists the same mission twice.
    DuplicateMission { id: String },
    /// A mission document failed to parse/validate (WI 551).
    Mission { id: String, error: MissionError },
    /// A mission's `AfterMission` offer names a mission not in this scenario.
    UnknownMissionRef { mission: String, after: String },
}

impl fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScenarioError::Io { path, error } => {
                write!(f, "could not read `{}`: {error}", path.display())
            }
            ScenarioError::Parse(e) => write!(f, "scenario parse error: {e}"),
            ScenarioError::Format { found } => write!(
                f,
                "unsupported scenario format version {found} (this build reads {CONTENT_FORMAT_VERSION})"
            ),
            ScenarioError::IdMismatch { expected, found } => write!(
                f,
                "document id `{found}` does not match its reference `{expected}`"
            ),
            ScenarioError::MissingDocument { kind, id } => {
                write!(f, "referenced {kind} `{id}` not found")
            }
            ScenarioError::LocalPhaseReserved { set } => write!(
                f,
                "override set `{set}` declares phase Local — reserved for the player; \
                 a scenario ships Patch or Scenario sets"
            ),
            ScenarioError::Content(e) => write!(f, "content composition failed: {e}"),
            ScenarioError::UnknownSystem { slug } => {
                write!(f, "system `{slug}` not found in the system library")
            }
            ScenarioError::Compile(e) => write!(f, "system compile failed: {e:?}"),
            ScenarioError::Library { what, error } => {
                write!(f, "could not load {what}: {error:?}")
            }
            ScenarioError::MissingBlueprint { slug } => write!(
                f,
                "blueprint `{slug}` not found (looked in content/blueprints/ and saves/crafts/)"
            ),
            ScenarioError::BindingUnknown { class, id } => write!(
                f,
                "device binding {class:?} → `{id}`: no such concrete record in the catalog \
                 (abstract bases are inheritance targets, not content)"
            ),
            ScenarioError::BindingWrongClass { class, id } => write!(
                f,
                "device binding {class:?} → `{id}`: record exists but is not a {class:?} device"
            ),
            ScenarioError::BindingMissing { class } => write!(
                f,
                "blueprint carries {class:?} devices but the scenario binds no catalog record \
                 for {class:?} — assembly takes physical values from content, never hardcodes"
            ),
            ScenarioError::DuplicateMission { id } => {
                write!(f, "mission `{id}` is listed more than once")
            }
            ScenarioError::Mission { id, error } => write!(f, "mission `{id}`: {error}"),
            ScenarioError::UnknownMissionRef { mission, after } => write!(
                f,
                "mission `{mission}` offers after `{after}`, which is not a mission of this \
                 scenario"
            ),
        }
    }
}

impl std::error::Error for ScenarioError {}

impl From<ContentError> for ScenarioError {
    fn from(e: ContentError) -> Self {
        ScenarioError::Content(e)
    }
}

// ---------------------------------------------------------------------------
// Roots — where documents live.
// ---------------------------------------------------------------------------

/// The directory roots a scenario resolves its references against.
#[derive(Debug, Clone)]
pub struct ScenarioRoots {
    /// Authored content root (holds `packs/`, `settings/`, `overrides/`,
    /// `scenarios/`, `blueprints/`).
    pub content: PathBuf,
    /// Player-library root (holds `crafts/`, `bodies/`, `systems/`).
    pub saves: PathBuf,
}

impl Default for ScenarioRoots {
    /// The repository/app convention: `content/` and `saves/` relative to the
    /// working directory (matches the existing app scenes' `saves/…` usage).
    fn default() -> Self {
        ScenarioRoots {
            content: PathBuf::from("content"),
            saves: PathBuf::from("saves"),
        }
    }
}

// ---------------------------------------------------------------------------
// The loaded scenario.
// ---------------------------------------------------------------------------

/// A loaded, fully-validated scenario: every reference resolved.
#[derive(Debug)]
pub struct Scenario {
    /// Stable identifier (matches the document).
    pub id: String,
    /// Display name.
    pub name: String,
    /// The composed, resolved content catalog (settings baked in).
    pub catalog: Catalog,
    /// The compiled world.
    pub universe: Universe,
    /// The root body's asset (surface constants + fluid medium).
    pub root_asset: BodyAsset,
    /// The starting craft's lattice, loaded from its blueprint.
    pub blueprint: VoxelCraft,
    /// The starting placement.
    pub placement: StartPlacement,
    /// Validated device bindings (class → concrete catalog record id).
    pub bindings: BTreeMap<DeviceClass, String>,
    /// Opaque lore payload (carried, not interpreted).
    pub lore: Option<String>,
    /// Resolved mission definitions, in the scenario's declared order (WI 551).
    pub missions: Vec<Mission>,
}

/// A minimal probe for an override set's declared phase (the full document is
/// parsed again inside the composition; this only enforces the reserved-Local
/// rule with the set's name attached). Unknown fields are ignored here — the
/// strict `deny_unknown_fields` parse happens in [`Catalog::compose`].
#[derive(Debug, Deserialize)]
struct PhaseProbe {
    #[allow(dead_code)]
    format: u32,
    id: String,
    phase: OverridePhase,
}

/// Loads and fully validates the scenario document at `path` against `roots`.
///
/// The returned [`Scenario`] has every reference resolved: composing the
/// catalog (WI 547–549 detectors included), compiling the world, loading the
/// blueprint, and checking the device bindings. Any failure is a typed
/// [`ScenarioError`] naming the offending reference.
pub fn load_scenario(path: &Path, roots: &ScenarioRoots) -> Result<Scenario, ScenarioError> {
    let text = read(path)?;
    let doc: ScenarioDoc = ron::from_str(&text).map_err(|e| ScenarioError::Parse(e.to_string()))?;
    if doc.format != CONTENT_FORMAT_VERSION {
        return Err(ScenarioError::Format { found: doc.format });
    }
    // Loaded by path: the id must match the file stem (id-based addressing).
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        if stem != doc.id {
            return Err(ScenarioError::IdMismatch {
                expected: stem.to_string(),
                found: doc.id.clone(),
            });
        }
    }
    load_doc(doc, roots)
}

/// Loads and validates an already-parsed scenario document (the path-id check
/// is the caller's when a path exists).
pub fn load_doc(doc: ScenarioDoc, roots: &ScenarioRoots) -> Result<Scenario, ScenarioError> {
    // Missions (WI 551): resolve each referenced document, loudly.
    let mut missions: Vec<Mission> = Vec::with_capacity(doc.missions.len());
    for id in &doc.missions {
        if missions.iter().any(|m| &m.id == id) {
            return Err(ScenarioError::DuplicateMission { id: id.clone() });
        }
        let path = roots.content.join("missions").join(format!("{id}.ron"));
        if !path.exists() {
            return Err(ScenarioError::MissingDocument {
                kind: "mission",
                id: id.clone(),
            });
        }
        let m = parse_mission(&read(&path)?).map_err(|error| ScenarioError::Mission {
            id: id.clone(),
            error,
        })?;
        if m.id != *id {
            return Err(ScenarioError::IdMismatch {
                expected: id.clone(),
                found: m.id,
            });
        }
        missions.push(m);
    }
    // AfterMission offers must reference missions of this scenario.
    for m in &missions {
        if let Offer::AfterMission(after) = &m.offer {
            if !missions.iter().any(|o| &o.id == after) {
                return Err(ScenarioError::UnknownMissionRef {
                    mission: m.id.clone(),
                    after: after.clone(),
                });
            }
        }
    }

    // Gather referenced content documents (missing file = loud, by kind + id).
    let pack_texts = read_docs(&roots.content.join("packs"), &doc.packs, "pack")?;
    let settings_texts = read_docs(
        &roots.content.join("settings"),
        &doc.settings,
        "settings document",
    )?;
    let override_texts = read_docs(
        &roots.content.join("overrides"),
        &doc.overrides,
        "override set",
    )?;

    // Reserved-phase rule: a scenario ships Patch/Scenario sets, never Local.
    for (id, text) in doc.overrides.iter().zip(&override_texts) {
        let probe: PhaseProbe = ron::from_str(text)
            .map_err(|e| ScenarioError::Parse(format!("override set `{id}`: {e}")))?;
        if probe.id != *id {
            return Err(ScenarioError::IdMismatch {
                expected: id.clone(),
                found: probe.id,
            });
        }
        if probe.phase == OverridePhase::Local {
            return Err(ScenarioError::LocalPhaseReserved { set: id.clone() });
        }
    }

    // Compose the catalog (Slice 0 owns parse/merge/bake/detector errors).
    let packs: Vec<&str> = pack_texts.iter().map(String::as_str).collect();
    let settings: Vec<&str> = settings_texts.iter().map(String::as_str).collect();
    let overrides: Vec<&str> = override_texts.iter().map(String::as_str).collect();
    let catalog = Catalog::compose(&packs, &settings, &overrides)?;

    // Composition sources must include each requested id (a file whose
    // internal id differs from its stem would otherwise slip through).
    for id in doc.packs.iter().chain(&doc.settings).chain(&doc.overrides) {
        if !catalog.sources.iter().any(|s| s == id) {
            return Err(ScenarioError::IdMismatch {
                expected: id.clone(),
                found: "(absent from composition sources)".to_string(),
            });
        }
    }

    // Cross-boundary body references: pack body records against the body
    // library (plus the built-in earthlike).
    let mut known: BTreeSet<String> = body_library::list_bodies(&roots.saves.join("bodies"))
        .into_iter()
        .map(|e| e.slug)
        .collect();
    known.insert(EARTHLIKE_SLUG.to_string());
    catalog.validate_body_refs(&known)?;

    // The world, by reference.
    let (universe, root_asset) = match &doc.world {
        WorldRef::Earthlike => {
            let asset = BodyAsset::earthlike();
            let system = System::single_body(EARTHLIKE_SLUG, "Earthlike", &asset.id);
            let universe = system
                .compile(std::slice::from_ref(&asset))
                .map_err(ScenarioError::Compile)?;
            (universe, asset)
        }
        WorldRef::System(slug) => {
            let sys_path = system_library::system_path(&roots.saves.join("systems"), slug);
            if !sys_path.exists() {
                return Err(ScenarioError::UnknownSystem { slug: slug.clone() });
            }
            let system =
                system_library::load_system(&sys_path).map_err(|error| ScenarioError::Library {
                    what: "system",
                    error,
                })?;
            let mut assets = Vec::new();
            for entry in body_library::list_bodies(&roots.saves.join("bodies")) {
                let asset = body_library::load_body(&entry.path).map_err(|error| {
                    ScenarioError::Library {
                        what: "body asset",
                        error,
                    }
                })?;
                assets.push(asset);
            }
            assets.push(BodyAsset::earthlike());
            let universe = system.compile(&assets).map_err(ScenarioError::Compile)?;
            let root = system
                .bodies
                .iter()
                .find(|b| matches!(b.placement, Placement::Root))
                .expect("compile validated exactly one root");
            let root_asset = assets
                .iter()
                .find(|a| a.id == root.asset_id)
                .expect("compile validated known assets")
                .clone();
            (universe, root_asset)
        }
    };

    // The starting blueprint: authored location first, then the player library.
    let blueprint = load_blueprint(&doc.start.blueprint, roots)?;

    // Device bindings: every binding resolves to a concrete record of its
    // class; classes this slice consumes are required when the blueprint
    // carries such devices.
    for (class, id) in &doc.start.bindings {
        match catalog.get(id).map(|e| &e.record) {
            None => {
                return Err(ScenarioError::BindingUnknown {
                    class: *class,
                    id: id.clone(),
                })
            }
            Some(Record::Device(d)) if spec_class(&d.spec) == *class => {}
            Some(_) => {
                return Err(ScenarioError::BindingWrongClass {
                    class: *class,
                    id: id.clone(),
                })
            }
        }
    }
    for (kind, class) in [
        (DeviceKind::Engine, DeviceClass::Engine),
        (DeviceKind::Tank, DeviceClass::Tank),
    ] {
        let present = blueprint.devices.iter().any(|d| d.kind == kind);
        if present && !doc.start.bindings.contains_key(&class) {
            return Err(ScenarioError::BindingMissing { class });
        }
    }

    Ok(Scenario {
        id: doc.id,
        name: doc.name,
        catalog,
        universe,
        root_asset,
        blueprint,
        placement: doc.start.placement,
        bindings: doc.start.bindings,
        lore: doc.lore,
        missions,
    })
}

/// The device class a resolved spec belongs to.
fn spec_class(spec: &DeviceSpec) -> DeviceClass {
    match spec {
        DeviceSpec::Engine { .. } => DeviceClass::Engine,
        DeviceSpec::Tank { .. } => DeviceClass::Tank,
        DeviceSpec::Battery { .. } => DeviceClass::Battery,
        DeviceSpec::Motor { .. } => DeviceClass::Motor,
    }
}

fn read(path: &Path) -> Result<String, ScenarioError> {
    std::fs::read_to_string(path).map_err(|e| ScenarioError::Io {
        path: path.to_path_buf(),
        error: e.to_string(),
    })
}

/// Reads `<dir>/<id>.ron` for each id, mapping a missing file to a
/// [`ScenarioError::MissingDocument`] naming the kind and id.
fn read_docs(dir: &Path, ids: &[String], kind: &'static str) -> Result<Vec<String>, ScenarioError> {
    let mut texts = Vec::with_capacity(ids.len());
    for id in ids {
        let path = dir.join(format!("{id}.ron"));
        if !path.exists() {
            return Err(ScenarioError::MissingDocument {
                kind,
                id: id.clone(),
            });
        }
        texts.push(read(&path)?);
    }
    Ok(texts)
}

/// Resolves a blueprint slug: `content/blueprints/<slug>.json`, then
/// `saves/crafts/<slug>.json`. Both are versioned persist craft documents.
fn load_blueprint(slug: &str, roots: &ScenarioRoots) -> Result<VoxelCraft, ScenarioError> {
    let candidates = [
        roots
            .content
            .join("blueprints")
            .join(format!("{slug}.json")),
        roots.saves.join("crafts").join(format!("{slug}.json")),
    ];
    for path in &candidates {
        if path.exists() {
            return library::load_craft(path).map_err(|error| ScenarioError::Library {
                what: "blueprint",
                error,
            });
        }
    }
    Err(ScenarioError::MissingBlueprint {
        slug: slug.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{Device, Material, Voxel};
    use glam::IVec3;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A unique scratch root with `content/` + `saves/` subtrees.
    fn scratch_roots(tag: &str) -> ScenarioRoots {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let base = std::env::temp_dir().join(format!("snd-scn-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        ScenarioRoots {
            content: base.join("content"),
            saves: base.join("saves"),
        }
    }

    fn write_doc(dir: &Path, id: &str, text: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(format!("{id}.ron")), text).unwrap();
    }

    /// A minimal engine+tank pack: abstract base, concrete engine, tank.
    fn pack_text() -> &'static str {
        r#"#![enable(implicit_some)]
        (format: 1, id: "p", version: "1", records: [
            Device(( id: "eb", abstract: true, class: Engine, exhaust_velocity: 3000.0 )),
            Device(( id: "e", parent: "eb", density: 3000.0, max_mass_flow: 2.0 )),
            Device(( id: "t", class: Tank, density: 500.0, capacity: 800.0 )),
        ])"#
    }

    /// A flyable-ish blueprint: one voxel, an engine and a tank device.
    fn blueprint() -> VoxelCraft {
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::COMPOSITE,
        });
        craft.devices.push(Device::structural(
            IVec3::new(0, 1, 0),
            10.0,
            crate::voxel::DeviceKind::Engine,
        ));
        craft.devices.push(Device::structural(
            IVec3::new(0, 2, 0),
            10.0,
            crate::voxel::DeviceKind::Tank,
        ));
        craft
    }

    /// Writes the standard fixture into `roots` and returns the scenario path.
    fn standard_fixture(roots: &ScenarioRoots, scenario_body: &str) -> PathBuf {
        write_doc(&roots.content.join("packs"), "p", pack_text());
        let bp_dir = roots.content.join("blueprints");
        std::fs::create_dir_all(&bp_dir).unwrap();
        library::save_craft(&bp_dir, "bp", &blueprint()).unwrap();
        let dir = roots.content.join("scenarios");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.ron");
        std::fs::write(&path, scenario_body).unwrap();
        path
    }

    fn base_scenario(extra: &str) -> String {
        format!(
            r#"(format: 1, id: "s", name: "S", world: Earthlike, packs: ["p"],
                start: (blueprint: "bp", placement: Pad,
                        bindings: {{ Engine: "e", Tank: "t" }}), {extra})"#
        )
    }

    #[test]
    fn happy_path_loads_composes_and_validates() {
        let roots = scratch_roots("happy");
        let path = standard_fixture(&roots, &base_scenario(""));
        let s = load_scenario(&path, &roots).unwrap();
        assert_eq!(s.id, "s");
        assert_eq!(s.root_asset.id, BodyAsset::earthlike().id);
        assert_eq!(s.bindings[&DeviceClass::Engine], "e");
        // Catalog resolved: the concrete engine inherited the base's value.
        match &s.catalog.get("e").unwrap().record {
            Record::Device(d) => match d.spec {
                DeviceSpec::Engine {
                    exhaust_velocity, ..
                } => assert_eq!(exhaust_velocity, 3000.0),
                _ => panic!("engine record expected"),
            },
            _ => panic!("device record expected"),
        }
    }

    #[test]
    fn missing_pack_is_named() {
        let roots = scratch_roots("missing-pack");
        let path = standard_fixture(
            &roots,
            r#"(format: 1, id: "s", name: "S", world: Earthlike, packs: ["nope"],
                start: (blueprint: "bp", placement: Pad, bindings: { Engine: "e", Tank: "t" }))"#,
        );
        match load_scenario(&path, &roots) {
            Err(ScenarioError::MissingDocument { kind: "pack", id }) => assert_eq!(id, "nope"),
            other => panic!("expected MissingDocument(pack), got {other:?}"),
        }
    }

    #[test]
    fn missing_blueprint_is_named() {
        let roots = scratch_roots("missing-bp");
        let path = standard_fixture(
            &roots,
            r#"(format: 1, id: "s", name: "S", world: Earthlike, packs: ["p"],
                start: (blueprint: "ghost", placement: Pad, bindings: { Engine: "e", Tank: "t" }))"#,
        );
        match load_scenario(&path, &roots) {
            Err(ScenarioError::MissingBlueprint { slug }) => assert_eq!(slug, "ghost"),
            other => panic!("expected MissingBlueprint, got {other:?}"),
        }
    }

    #[test]
    fn unknown_system_slug_is_named() {
        let roots = scratch_roots("missing-sys");
        let path = standard_fixture(
            &roots,
            r#"(format: 1, id: "s", name: "S", world: System("nowhere"), packs: ["p"],
                start: (blueprint: "bp", placement: Pad, bindings: { Engine: "e", Tank: "t" }))"#,
        );
        match load_scenario(&path, &roots) {
            Err(ScenarioError::UnknownSystem { slug }) => assert_eq!(slug, "nowhere"),
            other => panic!("expected UnknownSystem, got {other:?}"),
        }
    }

    #[test]
    fn saved_system_world_compiles_from_the_libraries() {
        let roots = scratch_roots("system-world");
        // A saved body + a saved single-body system referencing it by asset id.
        let mut body = BodyAsset::earthlike();
        body.name = "Mun".to_string();
        body.id = "mun".to_string();
        crate::body_library::save_body(&roots.saves.join("bodies"), &body).unwrap();
        let sys = System::single_body("home", "Home", "mun");
        crate::system_library::save_system(&roots.saves.join("systems"), &sys).unwrap();
        let path = standard_fixture(
            &roots,
            r#"(format: 1, id: "s", name: "S", world: System("home"), packs: ["p"],
                start: (blueprint: "bp", placement: Pad, bindings: { Engine: "e", Tank: "t" }))"#,
        );
        let s = load_scenario(&path, &roots).unwrap();
        assert_eq!(s.root_asset.id, "mun");
    }

    #[test]
    fn binding_errors_are_specific() {
        let roots = scratch_roots("bindings");
        // Unknown record.
        let path = standard_fixture(
            &roots,
            r#"(format: 1, id: "s", name: "S", world: Earthlike, packs: ["p"],
                start: (blueprint: "bp", placement: Pad, bindings: { Engine: "ghost", Tank: "t" }))"#,
        );
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::BindingUnknown {
                class: DeviceClass::Engine,
                ..
            })
        ));
        // Abstract base is not content.
        std::fs::write(&path, base_scenario("").replace("\"e\"", "\"eb\"")).unwrap();
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::BindingUnknown {
                class: DeviceClass::Engine,
                ..
            })
        ));
        // Wrong class.
        std::fs::write(
            &path,
            r#"(format: 1, id: "s", name: "S", world: Earthlike, packs: ["p"],
                start: (blueprint: "bp", placement: Pad, bindings: { Engine: "t", Tank: "t" }))"#,
        )
        .unwrap();
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::BindingWrongClass {
                class: DeviceClass::Engine,
                ..
            })
        ));
        // Blueprint has engines but no Engine binding.
        std::fs::write(
            &path,
            r#"(format: 1, id: "s", name: "S", world: Earthlike, packs: ["p"],
                start: (blueprint: "bp", placement: Pad, bindings: { Tank: "t" }))"#,
        )
        .unwrap();
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::BindingMissing {
                class: DeviceClass::Engine
            })
        ));
    }

    #[test]
    fn loading_is_deterministic_and_list_order_is_not_semantic() {
        let roots = scratch_roots("determinism");
        // A second, independent pack so the packs list has an order to permute.
        write_doc(
            &roots.content.join("packs"),
            "q",
            r#"(format: 1, id: "q", version: "1", records: [
                Resource(( id: "ore", tradable: Some(true) )),
            ])"#,
        );
        let scenario = |packs: &str| {
            format!(
                r#"(format: 1, id: "s", name: "S", world: Earthlike, packs: {packs},
                    start: (blueprint: "bp", placement: Pad,
                            bindings: {{ Engine: "e", Tank: "t" }}))"#
            )
        };
        let path = standard_fixture(&roots, &scenario(r#"["p", "q"]"#));
        let payload = |roots: &ScenarioRoots, path: &Path| {
            let s = load_scenario(path, roots).unwrap();
            serde_json::to_string(&crate::director::ScenarioSpawn::from_scenario(&s)).unwrap()
        };
        // Same document loaded twice → identical resolved payload.
        let a = payload(&roots, &path);
        let b = payload(&roots, &path);
        assert_eq!(a, b);
        // Permuted pack list → identical payload (composition order comes from
        // the ladder, never the input list).
        std::fs::write(&path, scenario(r#"["q", "p"]"#)).unwrap();
        let c = payload(&roots, &path);
        assert_eq!(a, c);
    }

    #[test]
    fn missions_resolve_and_validate_at_load() {
        let roots = scratch_roots("missions");
        // Unresolved reference: no such mission document.
        let path = standard_fixture(&roots, &base_scenario(r#"missions: ["reach-orbit"],"#));
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::MissingDocument {
                kind: "mission",
                ..
            })
        ));
        // A real mission resolves and rides the loaded scenario.
        write_doc(
            &roots.content.join("missions"),
            "reach-orbit",
            r#"(format: 1, id: "reach-orbit", name: "Reach 100 m",
                objective: AltitudeAbove(100.0), effects: [Lore("done")])"#,
        );
        let s = load_scenario(&path, &roots).unwrap();
        assert_eq!(s.missions.len(), 1);
        assert_eq!(s.missions[0].name, "Reach 100 m");
        // Duplicate listing is loud.
        std::fs::write(
            &path,
            base_scenario(r#"missions: ["reach-orbit", "reach-orbit"],"#),
        )
        .unwrap();
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::DuplicateMission { .. })
        ));
        // A vacuous objective is a named mission error.
        write_doc(
            &roots.content.join("missions"),
            "vacuous",
            r#"(format: 1, id: "vacuous", name: "V", objective: Any([]))"#,
        );
        std::fs::write(&path, base_scenario(r#"missions: ["vacuous"],"#)).unwrap();
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::Mission {
                error: crate::mission::MissionError::VacuousObjective,
                ..
            })
        ));
        // An AfterMission offer must name a mission of this scenario.
        write_doc(
            &roots.content.join("missions"),
            "later",
            r#"(format: 1, id: "later", name: "L", offer: AfterMission("ghost"),
                objective: Airborne)"#,
        );
        std::fs::write(&path, base_scenario(r#"missions: ["later"],"#)).unwrap();
        match load_scenario(&path, &roots) {
            Err(ScenarioError::UnknownMissionRef { mission, after }) => {
                assert_eq!((mission.as_str(), after.as_str()), ("later", "ghost"));
            }
            other => panic!("expected UnknownMissionRef, got {other:?}"),
        }
        // Internal id must match the referenced file stem.
        write_doc(
            &roots.content.join("missions"),
            "stem",
            r#"(format: 1, id: "not-stem", name: "S", objective: Airborne)"#,
        );
        std::fs::write(&path, base_scenario(r#"missions: ["stem"],"#)).unwrap();
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::IdMismatch { .. })
        ));
    }

    #[test]
    fn local_phase_override_set_is_reserved() {
        let roots = scratch_roots("local-phase");
        write_doc(
            &roots.content.join("overrides"),
            "house",
            r#"(format: 1, id: "house", phase: Local, overrides: [])"#,
        );
        let path = standard_fixture(&roots, &base_scenario(r#"overrides: ["house"],"#));
        match load_scenario(&path, &roots) {
            Err(ScenarioError::LocalPhaseReserved { set }) => assert_eq!(set, "house"),
            other => panic!("expected LocalPhaseReserved, got {other:?}"),
        }
    }

    #[test]
    fn id_must_match_file_stem() {
        let roots = scratch_roots("stem");
        let path = standard_fixture(
            &roots,
            r#"(format: 1, id: "not-s", name: "S", world: Earthlike, packs: ["p"],
                start: (blueprint: "bp", placement: Pad, bindings: { Engine: "e", Tank: "t" }))"#,
        );
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::IdMismatch { .. })
        ));
    }

    #[test]
    fn unknown_fields_and_bad_format_fail_loudly() {
        let roots = scratch_roots("strict");
        let path = standard_fixture(&roots, &base_scenario(r#"surprise: 1,"#));
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::Parse(_))
        ));
        std::fs::write(&path, base_scenario("").replace("format: 1", "format: 99")).unwrap();
        assert!(matches!(
            load_scenario(&path, &roots),
            Err(ScenarioError::Format { found: 99 })
        ));
    }
}

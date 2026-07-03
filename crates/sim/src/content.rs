//! Content catalog + RON asset-pack loader (WI 547).
//!
//! The foundation of the content layer (content aspect, Slice 0): a flat
//! namespace of **typed, identified records** — devices, materials, resources,
//! and body *references* — that the simulation will look up by id instead of
//! hardcoding instances. Records carry **real physical parameters only** (the
//! quantities the sim already consumes: a material *is* a [`Material`], an
//! engine spec mirrors [`crate::propulsion::Engine`], a motor spec mirrors
//! [`crate::powertrain::MotorSpec`]); balance scalars and overrides arrive with
//! WIs 548/549 and are **never** stored here.
//!
//! An **asset pack** is a versioned, hand-authored RON document: a manifest
//! (id + version + content-format version) plus records. Records support
//! ParentName-style **abstract-base inheritance** (the CDDA/RimWorld pattern):
//! a record may name a `parent` of the same kind; unset fields inherit the
//! parent's resolved values, declared fields override, and resolution happens
//! **after** the whole pack is read so declaration order never matters. A
//! parent may be abstract *or* concrete — `abstract: true` controls only
//! whether the record appears in the resolved catalog.
//!
//! **Content is data, never code** (untrusted-pack safety): the loader
//! deserializes and validates; nothing in a pack can execute. Failures are
//! **loud and typed** — parse errors, an unknown format version, duplicate ids,
//! unknown/cyclic/kind-mismatched parents, and missing or inapplicable fields
//! all fail the load naming the offender; records are never silently skipped.
//! Unknown fields and unknown record kinds are rejected (authored artifacts
//! deserve typo-loudness; forward-compat relaxation is a deliberate later
//! migration decision, unlike the reserved-container idiom of
//! [`crate::persist`], which serves machine-written saves).
//!
//! The pack **format version** ([`CONTENT_FORMAT_VERSION`]) is a separate line
//! from [`crate::persist::FORMAT_VERSION`]: authored packs and machine-written
//! saves are different artifact classes with different migration pressure. The
//! pack's own `version` field is an opaque identity string in this slice
//! (recorded and surfaced in provenance; ordering/migration semantics deferred).
//!
//! **Body records reference, never define** (content design Currency Addendum
//! 2026-07-02): a body record carries the identity of a world-building library
//! asset ([`crate::body_library`] slug); world-building owns body data, content
//! composes it.
//!
//! Loading is **deterministic**: the catalog is ordered storage (a `BTreeMap`),
//! so the same pack bytes always produce the same catalog — the foundation for
//! WI 549's merge-determinism assertion. Headless and rendering-free.

use crate::voxel::{Material, Thermal};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::Path;

/// Version of the authored pack *format* (not of any pack's content). Rejected
/// loudly when a pack declares anything else. Increments only on a schema
/// change to the pack document shape.
pub const CONTENT_FORMAT_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a pack failed to load. Every variant names its offender — a pack never
/// loads partially and records are never silently dropped.
#[derive(Debug)]
pub enum ContentError {
    /// The pack file could not be read.
    Io(std::io::Error),
    /// The RON text failed to parse (includes position context from `ron`).
    Parse(String),
    /// The pack declares a content-format version this build does not know.
    Format { found: u32 },
    /// Two records share an id (the namespace is flat across kinds).
    DuplicateId { id: String },
    /// A record names a parent id that does not exist in the pack.
    UnknownParent { child: String, parent: String },
    /// A record names a parent of a different record kind.
    KindMismatch { child: String, parent: String },
    /// A record's inheritance chain loops (including naming itself).
    Cycle { id: String },
    /// A concrete (non-abstract) record is missing a required physical field
    /// after inheritance resolution.
    MissingField { id: String, field: &'static str },
    /// A device record carries a field that does not apply to its class
    /// (an authoring error caught loudly, like unknown fields).
    InapplicableField { id: String, field: &'static str },
}

impl fmt::Display for ContentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContentError::Io(e) => write!(f, "content pack io error: {e}"),
            ContentError::Parse(msg) => write!(f, "content pack parse error: {msg}"),
            ContentError::Format { found } => write!(
                f,
                "unsupported content format version {found} (this build reads {CONTENT_FORMAT_VERSION})"
            ),
            ContentError::DuplicateId { id } => write!(f, "duplicate record id `{id}`"),
            ContentError::UnknownParent { child, parent } => {
                write!(f, "record `{child}` names unknown parent `{parent}`")
            }
            ContentError::KindMismatch { child, parent } => {
                write!(f, "record `{child}` names parent `{parent}` of a different kind")
            }
            ContentError::Cycle { id } => {
                write!(f, "inheritance cycle through record `{id}`")
            }
            ContentError::MissingField { id, field } => {
                write!(f, "record `{id}` is missing required field `{field}`")
            }
            ContentError::InapplicableField { id, field } => {
                write!(f, "record `{id}` carries field `{field}` inapplicable to its class")
            }
        }
    }
}

impl std::error::Error for ContentError {}

// ---------------------------------------------------------------------------
// Raw (authored) forms — permissive all-`Option` shapes the loader reads.
// ---------------------------------------------------------------------------

/// The pack document as authored: manifest + records.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPack {
    /// Pack *format* version — must equal [`CONTENT_FORMAT_VERSION`].
    format: u32,
    /// The pack's identity.
    id: String,
    /// The pack's own version — an opaque identity string in this slice.
    version: String,
    /// The authored records, any order (resolution is order-independent).
    records: Vec<RawRecord>,
}

/// One authored record. The RON kind tag is the enum variant name.
#[derive(Debug, Clone, Deserialize)]
enum RawRecord {
    Device(RawDevice),
    Material(RawMaterial),
    Resource(RawResource),
    Body(RawBody),
}

impl RawRecord {
    fn id(&self) -> &str {
        match self {
            RawRecord::Device(r) => &r.id,
            RawRecord::Material(r) => &r.id,
            RawRecord::Resource(r) => &r.id,
            RawRecord::Body(r) => &r.id,
        }
    }
    fn parent(&self) -> Option<&str> {
        match self {
            RawRecord::Device(r) => r.parent.as_deref(),
            RawRecord::Material(r) => r.parent.as_deref(),
            RawRecord::Resource(r) => r.parent.as_deref(),
            RawRecord::Body(r) => r.parent.as_deref(),
        }
    }
    fn is_abstract(&self) -> bool {
        match self {
            RawRecord::Device(r) => r.is_abstract,
            RawRecord::Material(r) => r.is_abstract,
            RawRecord::Resource(r) => r.is_abstract,
            RawRecord::Body(r) => r.is_abstract,
        }
    }

    /// Child-over-parent field merge. `parent` must already be fully merged.
    /// Fails with [`ContentError::KindMismatch`] when the kinds differ.
    fn merge_over(self, parent: &RawRecord) -> Result<RawRecord, ContentError> {
        match (self, parent) {
            (RawRecord::Device(c), RawRecord::Device(p)) => Ok(RawRecord::Device(c.merge_over(p))),
            (RawRecord::Material(c), RawRecord::Material(p)) => {
                Ok(RawRecord::Material(c.merge_over(p)))
            }
            (RawRecord::Resource(c), RawRecord::Resource(p)) => {
                Ok(RawRecord::Resource(c.merge_over(p)))
            }
            (RawRecord::Body(c), RawRecord::Body(p)) => Ok(RawRecord::Body(c.merge_over(p))),
            (c, p) => Err(ContentError::KindMismatch {
                child: c.id().to_string(),
                parent: p.id().to_string(),
            }),
        }
    }
}

/// The functional class a device record describes — the discriminator that
/// decides which physical fields apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum DeviceClass {
    Engine,
    Tank,
    Battery,
    Motor,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDevice {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    #[serde(default)]
    class: Option<DeviceClass>,
    /// Bulk density, kg/m³ — cell-scaled mass, the WI 615 shape.
    #[serde(default)]
    density: Option<f64>,
    // Engine (mirrors `propulsion::Engine`).
    #[serde(default)]
    exhaust_velocity: Option<f64>,
    #[serde(default)]
    max_mass_flow: Option<f64>,
    // Tank (kg) / Battery (J).
    #[serde(default)]
    capacity: Option<f64>,
    // Motor (mirrors `powertrain::MotorSpec`).
    #[serde(default)]
    max_torque: Option<f64>,
    #[serde(default)]
    top_speed: Option<f64>,
    #[serde(default)]
    motor_mass: Option<f64>,
    #[serde(default)]
    draw: Option<f64>,
}

impl RawDevice {
    fn merge_over(self, p: &RawDevice) -> RawDevice {
        RawDevice {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            class: self.class.or(p.class),
            density: self.density.or(p.density),
            exhaust_velocity: self.exhaust_velocity.or(p.exhaust_velocity),
            max_mass_flow: self.max_mass_flow.or(p.max_mass_flow),
            capacity: self.capacity.or(p.capacity),
            max_torque: self.max_torque.or(p.max_torque),
            top_speed: self.top_speed.or(p.top_speed),
            motor_mass: self.motor_mass.or(p.motor_mass),
            draw: self.draw.or(p.draw),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMaterial {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    /// kg/m³.
    #[serde(default)]
    density: Option<f64>,
    /// Tensile strength, Pa.
    #[serde(default)]
    strength: Option<f64>,
    // The thermal quad (required for concrete materials)…
    #[serde(default)]
    specific_heat: Option<f64>,
    #[serde(default)]
    conductivity: Option<f64>,
    #[serde(default)]
    emissivity: Option<f64>,
    #[serde(default)]
    max_temp: Option<f64>,
    // …plus the optional ablation trio (default 0 = non-ablative).
    #[serde(default)]
    ablation_temp: Option<f64>,
    #[serde(default)]
    latent_heat: Option<f64>,
    #[serde(default)]
    ablator_fraction: Option<f64>,
}

impl RawMaterial {
    fn merge_over(self, p: &RawMaterial) -> RawMaterial {
        RawMaterial {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            density: self.density.or(p.density),
            strength: self.strength.or(p.strength),
            specific_heat: self.specific_heat.or(p.specific_heat),
            conductivity: self.conductivity.or(p.conductivity),
            emissivity: self.emissivity.or(p.emissivity),
            max_temp: self.max_temp.or(p.max_temp),
            ablation_temp: self.ablation_temp.or(p.ablation_temp),
            latent_heat: self.latent_heat.or(p.latent_heat),
            ablator_fraction: self.ablator_fraction.or(p.ablator_fraction),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawResource {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    /// kg/m³ where meaningful (a massless resource like charge omits it).
    #[serde(default)]
    density: Option<f64>,
    /// Economy-facing tradability (inert until the economy sibling lands).
    #[serde(default)]
    tradable: Option<bool>,
}

impl RawResource {
    fn merge_over(self, p: &RawResource) -> RawResource {
        RawResource {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            density: self.density.or(p.density),
            tradable: self.tradable.or(p.tradable),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBody {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default, rename = "abstract")]
    is_abstract: bool,
    /// World-building body-library slug this record points at. A body record
    /// **references** a library asset; it never defines body physical data.
    #[serde(default)]
    body: Option<String>,
}

impl RawBody {
    fn merge_over(self, p: &RawBody) -> RawBody {
        RawBody {
            id: self.id,
            parent: self.parent,
            is_abstract: self.is_abstract,
            body: self.body.or_else(|| p.body.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Resolved records — fully-typed, real physical parameters.
// ---------------------------------------------------------------------------

/// Where a resolved record came from — the provenance seed WI 548's merge
/// ladder extends (per-value provenance with shadowing arrives there).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Id of the pack that supplied the record.
    pub pack_id: String,
    /// The pack's (opaque) version string.
    pub pack_version: String,
}

/// A device's class-specific physical spec. Real quantities only — the shapes
/// the sim already consumes.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceSpec {
    /// Mirrors [`crate::propulsion::Engine`]: thrust = mass-flow × exhaust velocity.
    Engine {
        /// Exhaust velocity, m/s (`= Isp · g₀`).
        exhaust_velocity: f64,
        /// Maximum propellant mass flow, kg/s.
        max_mass_flow: f64,
    },
    /// A propellant tank: capacity in kg.
    Tank { capacity: f64 },
    /// A battery: capacity in J.
    Battery { capacity: f64 },
    /// Mirrors [`crate::powertrain::MotorSpec`].
    Motor {
        /// Per-drive-wheel torque at full throttle, N·m.
        max_torque: f64,
        /// Wheel-spin top-speed cap, rad/s.
        top_speed: f64,
        /// Motor mass, kg.
        mass: f64,
        /// Consumption multiplier on the source's base draw.
        draw: f64,
    },
}

/// A resolved device record.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceRecord {
    pub id: String,
    /// Bulk density, kg/m³ — mass is `density × cell³` (WI 615 shape).
    pub density: f64,
    pub spec: DeviceSpec,
}

/// A resolved material record — the payload is the *actual* sim material type.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialRecord {
    pub id: String,
    pub material: Material,
}

/// A resolved resource record.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceRecord {
    pub id: String,
    /// kg/m³ where meaningful; `None` for massless resources (e.g. charge).
    pub density: Option<f64>,
    /// Economy-facing flag; inert until the economy sibling (WI 552).
    pub tradable: bool,
}

/// A resolved body-reference record: identity linkage into the world-building
/// body library — no physical body data (world-building owns that).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BodyRefRecord {
    pub id: String,
    /// [`crate::body_library`] slug of the referenced body asset.
    pub body_slug: String,
}

/// A resolved, concrete content record.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Device(DeviceRecord),
    Material(MaterialRecord),
    Resource(ResourceRecord),
    Body(BodyRefRecord),
}

/// A catalog entry: the record plus its provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub record: Record,
    pub provenance: Provenance,
}

/// The resolved catalog of one loaded pack: concrete records by id, in
/// deterministic (ordered) storage. Abstract bases are resolution inputs, not
/// content — they do not appear here.
#[derive(Debug, Clone, PartialEq)]
pub struct Catalog {
    pub pack_id: String,
    pub pack_version: String,
    records: BTreeMap<String, Entry>,
}

impl Catalog {
    /// Load a pack from RON text.
    pub fn from_ron_str(source: &str) -> Result<Catalog, ContentError> {
        let raw: RawPack = ron::from_str(source).map_err(|e| ContentError::Parse(e.to_string()))?;
        if raw.format != CONTENT_FORMAT_VERSION {
            return Err(ContentError::Format { found: raw.format });
        }
        resolve(raw)
    }

    /// Load a pack from a RON file on disk.
    pub fn load(path: &Path) -> Result<Catalog, ContentError> {
        let text = std::fs::read_to_string(path).map_err(ContentError::Io)?;
        Catalog::from_ron_str(&text)
    }

    /// Look up a concrete record by id.
    pub fn get(&self, id: &str) -> Option<&Entry> {
        self.records.get(id)
    }

    /// All concrete record ids, in deterministic (sorted) order.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.records.keys().map(String::as_str)
    }

    /// Number of concrete records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the catalog holds no concrete records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Resolution — after-load inheritance merge + validation.
// ---------------------------------------------------------------------------

/// Merge every record's inheritance chain (memoized, cycle-checked), then
/// validate concrete records into the catalog.
fn resolve(raw: RawPack) -> Result<Catalog, ContentError> {
    // Flat namespace: duplicate ids are an authoring error across kinds too.
    let mut by_id: HashMap<String, RawRecord> = HashMap::new();
    for r in raw.records {
        let id = r.id().to_string();
        if by_id.insert(id.clone(), r).is_some() {
            return Err(ContentError::DuplicateId { id });
        }
    }

    // Deterministic order: resolve ids sorted (BTreeMap), not hash order.
    let ordered: BTreeMap<String, RawRecord> = by_id.into_iter().collect();

    let mut merged: HashMap<String, RawRecord> = HashMap::new();
    for id in ordered.keys() {
        let mut visiting = HashSet::new();
        merge_chain(id, &ordered, &mut merged, &mut visiting)?;
    }

    let provenance = Provenance {
        pack_id: raw.id.clone(),
        pack_version: raw.version.clone(),
    };
    let mut records = BTreeMap::new();
    for (id, _) in ordered {
        let m = &merged[&id];
        if m.is_abstract() {
            continue; // bases are inheritance targets, not content
        }
        let record = validate(m)?;
        records.insert(
            id,
            Entry {
                record,
                provenance: provenance.clone(),
            },
        );
    }

    Ok(Catalog {
        pack_id: raw.id,
        pack_version: raw.version,
        records,
    })
}

/// Compute the fully-merged raw form of `id` (its fields with every ancestor
/// folded in), memoizing results and detecting cycles/unknown parents.
fn merge_chain(
    id: &str,
    ordered: &BTreeMap<String, RawRecord>,
    merged: &mut HashMap<String, RawRecord>,
    visiting: &mut HashSet<String>,
) -> Result<(), ContentError> {
    if merged.contains_key(id) {
        return Ok(());
    }
    if !visiting.insert(id.to_string()) {
        return Err(ContentError::Cycle { id: id.to_string() });
    }
    let record = ordered
        .get(id)
        .expect("merge_chain called only with known ids")
        .clone();
    let result = match record.parent() {
        None => record,
        Some(parent_id) => {
            let parent_id = parent_id.to_string();
            if !ordered.contains_key(&parent_id) {
                return Err(ContentError::UnknownParent {
                    child: id.to_string(),
                    parent: parent_id,
                });
            }
            merge_chain(&parent_id, ordered, merged, visiting)?;
            record.merge_over(&merged[&parent_id])?
        }
    };
    visiting.remove(id);
    merged.insert(id.to_string(), result);
    Ok(())
}

/// Validate a fully-merged concrete record into its resolved, fully-typed form.
fn validate(m: &RawRecord) -> Result<Record, ContentError> {
    let missing = |id: &str, field: &'static str| ContentError::MissingField {
        id: id.to_string(),
        field,
    };
    match m {
        RawRecord::Device(d) => {
            let class = d.class.ok_or_else(|| missing(&d.id, "class"))?;
            let density = d.density.ok_or_else(|| missing(&d.id, "density"))?;
            let spec = device_spec(d, class)?;
            Ok(Record::Device(DeviceRecord {
                id: d.id.clone(),
                density,
                spec,
            }))
        }
        RawRecord::Material(mt) => {
            let material = Material {
                density: mt.density.ok_or_else(|| missing(&mt.id, "density"))?,
                strength: mt.strength.ok_or_else(|| missing(&mt.id, "strength"))?,
                thermal: Thermal {
                    specific_heat: mt
                        .specific_heat
                        .ok_or_else(|| missing(&mt.id, "specific_heat"))?,
                    conductivity: mt
                        .conductivity
                        .ok_or_else(|| missing(&mt.id, "conductivity"))?,
                    emissivity: mt.emissivity.ok_or_else(|| missing(&mt.id, "emissivity"))?,
                    max_temp: mt.max_temp.ok_or_else(|| missing(&mt.id, "max_temp"))?,
                    ablation_temp: mt.ablation_temp.unwrap_or(0.0),
                    latent_heat: mt.latent_heat.unwrap_or(0.0),
                    ablator_fraction: mt.ablator_fraction.unwrap_or(0.0),
                },
            };
            Ok(Record::Material(MaterialRecord {
                id: mt.id.clone(),
                material,
            }))
        }
        RawRecord::Resource(r) => Ok(Record::Resource(ResourceRecord {
            id: r.id.clone(),
            density: r.density,
            tradable: r.tradable.unwrap_or(false),
        })),
        RawRecord::Body(b) => Ok(Record::Body(BodyRefRecord {
            id: b.id.clone(),
            body_slug: b.body.clone().ok_or_else(|| missing(&b.id, "body"))?,
        })),
    }
}

/// Build the class-specific spec, requiring the class's fields and rejecting
/// fields that do not apply to it (loud authoring-typo detection).
fn device_spec(d: &RawDevice, class: DeviceClass) -> Result<DeviceSpec, ContentError> {
    let missing = |field: &'static str| ContentError::MissingField {
        id: d.id.clone(),
        field,
    };
    let inapplicable = |field: &'static str| ContentError::InapplicableField {
        id: d.id.clone(),
        field,
    };
    // (field value, name, applicable-to) — checked against the resolved class.
    let engine = matches!(class, DeviceClass::Engine);
    let tanklike = matches!(class, DeviceClass::Tank | DeviceClass::Battery);
    let motor = matches!(class, DeviceClass::Motor);
    if d.exhaust_velocity.is_some() && !engine {
        return Err(inapplicable("exhaust_velocity"));
    }
    if d.max_mass_flow.is_some() && !engine {
        return Err(inapplicable("max_mass_flow"));
    }
    if d.capacity.is_some() && !tanklike {
        return Err(inapplicable("capacity"));
    }
    if d.max_torque.is_some() && !motor {
        return Err(inapplicable("max_torque"));
    }
    if d.top_speed.is_some() && !motor {
        return Err(inapplicable("top_speed"));
    }
    if d.motor_mass.is_some() && !motor {
        return Err(inapplicable("motor_mass"));
    }
    if d.draw.is_some() && !motor {
        return Err(inapplicable("draw"));
    }
    Ok(match class {
        DeviceClass::Engine => DeviceSpec::Engine {
            exhaust_velocity: d
                .exhaust_velocity
                .ok_or_else(|| missing("exhaust_velocity"))?,
            max_mass_flow: d.max_mass_flow.ok_or_else(|| missing("max_mass_flow"))?,
        },
        DeviceClass::Tank => DeviceSpec::Tank {
            capacity: d.capacity.ok_or_else(|| missing("capacity"))?,
        },
        DeviceClass::Battery => DeviceSpec::Battery {
            capacity: d.capacity.ok_or_else(|| missing("capacity"))?,
        },
        DeviceClass::Motor => DeviceSpec::Motor {
            max_torque: d.max_torque.ok_or_else(|| missing("max_torque"))?,
            top_speed: d.top_speed.ok_or_else(|| missing("top_speed"))?,
            mass: d.motor_mass.ok_or_else(|| missing("motor_mass"))?,
            draw: d.draw.ok_or_else(|| missing("draw"))?,
        },
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal valid pack around the given records list (RON snippet).
    fn pack(records: &str) -> String {
        format!(
            "#![enable(implicit_some)]\n(format: 1, id: \"test\", version: \"1\", records: [{records}])"
        )
    }

    fn engine_base() -> &'static str {
        r#"Device(( id: "engine_base", abstract: true, class: Engine, exhaust_velocity: 3200.0 )),"#
    }

    #[test]
    fn shipped_core_pack_loads_and_resolves() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/packs/core.ron");
        let cat = Catalog::load(&path).expect("shipped core pack loads");
        assert_eq!(cat.pack_id, "core");
        // The concrete engine inherits exhaust velocity from the abstract base
        // and carries its own mass flow + density.
        let entry = cat.get("lf_engine_small").expect("engine present");
        assert_eq!(entry.provenance.pack_id, "core");
        match &entry.record {
            Record::Device(d) => {
                assert_eq!(d.density, 3000.0);
                assert_eq!(
                    d.spec,
                    DeviceSpec::Engine {
                        exhaust_velocity: 3200.0,
                        max_mass_flow: 1.5
                    }
                );
            }
            other => panic!("expected device, got {other:?}"),
        }
        // The abstract base is not content.
        assert!(cat.get("engine_base").is_none());
        // Material and resource records resolve.
        match &cat.get("aluminium").expect("material present").record {
            Record::Material(m) => {
                assert_eq!(m.material.density, 2700.0);
                assert_eq!(m.material.thermal.max_temp, 900.0);
            }
            other => panic!("expected material, got {other:?}"),
        }
        match &cat.get("liquid_fuel").expect("resource present").record {
            Record::Resource(r) => {
                assert_eq!(r.density, Some(800.0));
                assert!(r.tradable);
            }
            other => panic!("expected resource, got {other:?}"),
        }
    }

    #[test]
    fn inheritance_chain_resolves_and_overrides() {
        let src = pack(&format!(
            r#"{base}
            Device(( id: "mid", parent: "engine_base", abstract: true, density: 2500.0 )),
            Device(( id: "leaf", parent: "mid", density: 3100.0, max_mass_flow: 2.0 )),"#,
            base = engine_base()
        ));
        let cat = Catalog::from_ron_str(&src).unwrap();
        match &cat.get("leaf").unwrap().record {
            Record::Device(d) => {
                // Grandparent's exhaust velocity, own density override, own flow.
                assert_eq!(d.density, 3100.0);
                assert_eq!(
                    d.spec,
                    DeviceSpec::Engine {
                        exhaust_velocity: 3200.0,
                        max_mass_flow: 2.0
                    }
                );
            }
            other => panic!("expected device, got {other:?}"),
        }
        // Both abstracts are absent from the catalog.
        assert!(cat.get("engine_base").is_none() && cat.get("mid").is_none());
        assert_eq!(cat.len(), 1);
    }

    #[test]
    fn editing_a_base_reflows_to_variants() {
        let variant =
            r#"Device(( id: "v", parent: "engine_base", density: 3000.0, max_mass_flow: 1.0 )),"#;
        let a = pack(&format!("{}{variant}", engine_base()));
        let b = pack(&format!(
            "{}{variant}",
            engine_base().replace("3200.0", "4400.0")
        ));
        let va = Catalog::from_ron_str(&a).unwrap();
        let vb = Catalog::from_ron_str(&b).unwrap();
        let ev = |cat: &Catalog| match &cat.get("v").unwrap().record {
            Record::Device(d) => match d.spec {
                DeviceSpec::Engine {
                    exhaust_velocity, ..
                } => exhaust_velocity,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        assert_eq!(ev(&va), 3200.0);
        assert_eq!(ev(&vb), 4400.0);
    }

    #[test]
    fn declaration_order_does_not_matter() {
        let fwd = pack(&format!(
            "{}Device(( id: \"v\", parent: \"engine_base\", density: 1.0, max_mass_flow: 1.0 )),",
            engine_base()
        ));
        let rev = pack(&format!(
            "Device(( id: \"v\", parent: \"engine_base\", density: 1.0, max_mass_flow: 1.0 )),{}",
            engine_base()
        ));
        assert_eq!(
            Catalog::from_ron_str(&fwd).unwrap(),
            Catalog::from_ron_str(&rev).unwrap()
        );
    }

    #[test]
    fn duplicate_id_rejected_across_kinds() {
        let src = pack(
            r#"Resource(( id: "x" )),
               Material(( id: "x", density: 1.0, strength: 1.0, specific_heat: 1.0, conductivity: 1.0, emissivity: 0.5, max_temp: 100.0 )),"#,
        );
        match Catalog::from_ron_str(&src) {
            Err(ContentError::DuplicateId { id }) => assert_eq!(id, "x"),
            other => panic!("expected DuplicateId, got {other:?}"),
        }
    }

    #[test]
    fn unknown_parent_rejected_naming_both() {
        let src = pack(r#"Device(( id: "child", parent: "ghost" )),"#);
        match Catalog::from_ron_str(&src) {
            Err(ContentError::UnknownParent { child, parent }) => {
                assert_eq!((child.as_str(), parent.as_str()), ("child", "ghost"));
            }
            other => panic!("expected UnknownParent, got {other:?}"),
        }
    }

    #[test]
    fn cycles_rejected_including_self_parent() {
        let self_p = pack(r#"Device(( id: "a", parent: "a" )),"#);
        assert!(matches!(
            Catalog::from_ron_str(&self_p),
            Err(ContentError::Cycle { .. })
        ));
        let two = pack(
            r#"Device(( id: "a", parent: "b" )),
               Device(( id: "b", parent: "a" )),"#,
        );
        assert!(matches!(
            Catalog::from_ron_str(&two),
            Err(ContentError::Cycle { .. })
        ));
    }

    #[test]
    fn parent_of_different_kind_rejected() {
        let src = pack(
            r#"Resource(( id: "base", abstract: true )),
               Device(( id: "child", parent: "base" )),"#,
        );
        match Catalog::from_ron_str(&src) {
            Err(ContentError::KindMismatch { child, parent }) => {
                assert_eq!((child.as_str(), parent.as_str()), ("child", "base"));
            }
            other => panic!("expected KindMismatch, got {other:?}"),
        }
    }

    #[test]
    fn wrong_format_version_rejected() {
        let src = "(format: 99, id: \"p\", version: \"1\", records: [])";
        match Catalog::from_ron_str(src) {
            Err(ContentError::Format { found }) => assert_eq!(found, 99),
            other => panic!("expected Format, got {other:?}"),
        }
    }

    #[test]
    fn unknown_field_and_unknown_kind_rejected() {
        let field = pack(r#"Resource(( id: "r", flavour: "grape" )),"#);
        assert!(matches!(
            Catalog::from_ron_str(&field),
            Err(ContentError::Parse(_))
        ));
        let kind = pack(r#"Widget(( id: "w" )),"#);
        assert!(matches!(
            Catalog::from_ron_str(&kind),
            Err(ContentError::Parse(_))
        ));
    }

    #[test]
    fn missing_required_field_rejected() {
        // Concrete engine without a mass flow (base supplies velocity only).
        let src = pack(&format!(
            "{}Device(( id: \"v\", parent: \"engine_base\", density: 1.0 )),",
            engine_base()
        ));
        match Catalog::from_ron_str(&src) {
            Err(ContentError::MissingField { id, field }) => {
                assert_eq!((id.as_str(), field), ("v", "max_mass_flow"));
            }
            other => panic!("expected MissingField, got {other:?}"),
        }
    }

    #[test]
    fn inapplicable_field_rejected() {
        let src = pack(
            r#"Device(( id: "t", class: Tank, density: 500.0, capacity: 100.0, exhaust_velocity: 3000.0 )),"#,
        );
        match Catalog::from_ron_str(&src) {
            Err(ContentError::InapplicableField { id, field }) => {
                assert_eq!((id.as_str(), field), ("t", "exhaust_velocity"));
            }
            other => panic!("expected InapplicableField, got {other:?}"),
        }
    }

    #[test]
    fn motor_and_battery_records_resolve() {
        let src = pack(
            r#"Device(( id: "motor_std", class: Motor, density: 4000.0, max_torque: 90.0, top_speed: 60.0, motor_mass: 45.0, draw: 1.0 )),
               Device(( id: "battery_s", class: Battery, density: 2500.0, capacity: 5.0e6 )),
               Body(( id: "start_moon", body: "training-moon" )),"#,
        );
        let cat = Catalog::from_ron_str(&src).unwrap();
        assert!(matches!(
            &cat.get("motor_std").unwrap().record,
            Record::Device(DeviceRecord {
                spec: DeviceSpec::Motor { .. },
                ..
            })
        ));
        assert!(matches!(
            &cat.get("battery_s").unwrap().record,
            Record::Device(DeviceRecord {
                spec: DeviceSpec::Battery { .. },
                ..
            })
        ));
        match &cat.get("start_moon").unwrap().record {
            Record::Body(b) => assert_eq!(b.body_slug, "training-moon"),
            other => panic!("expected body ref, got {other:?}"),
        }
    }

    #[test]
    fn body_reference_requires_slug() {
        let src = pack(r#"Body(( id: "start_moon" )),"#);
        assert!(matches!(
            Catalog::from_ron_str(&src),
            Err(ContentError::MissingField { field: "body", .. })
        ));
    }

    #[test]
    fn empty_pack_loads_to_empty_catalog() {
        let cat =
            Catalog::from_ron_str("(format: 1, id: \"p\", version: \"1\", records: [])").unwrap();
        assert!(cat.is_empty());
    }

    #[test]
    fn loading_is_deterministic() {
        let src = pack(&format!(
            "{}Device(( id: \"v\", parent: \"engine_base\", density: 1.0, max_mass_flow: 1.0 )),
             Resource(( id: \"fuel\", density: 800.0 )),",
            engine_base()
        ));
        let a = Catalog::from_ron_str(&src).unwrap();
        let b = Catalog::from_ron_str(&src).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.ids().collect::<Vec<_>>(), vec!["fuel", "v"]);
    }
}

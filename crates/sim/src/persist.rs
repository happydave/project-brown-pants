//! Versioned persistence format (WI 498).
//!
//! The single, explicitly versioned serialization format for a craft, a
//! subassembly, a blueprint, or (scaled up) a whole-world save — **one envelope,
//! one version line, several uses.** It is versioned *from the first commit*
//! because every saved artifact becomes a migration liability the moment the
//! schema drifts.
//!
//! This is **durable persistence**, deliberately distinct from the ephemeral
//! runtime bus (`Command`/`Telemetry`, WI 502): those are unversioned wire shapes
//! for live clients; these are versioned saves. Do not route saves through the
//! telemetry types.
//!
//! At format version 1 the content model does not exist yet — the structural
//! lattice, devices, resource graph, and crew arrive with later toys. So the
//! payload here is **skeletal and extensible**: it embeds the real WI 497
//! world-coordinate types plus metadata, and reserves empty, opaque containers
//! that later toys fill in. Filling a reserved container that has **never had a
//! producer** is treated like the additive-variant rule (no format bump — zero
//! documents exist whose meaning changes; see `WorldPayload::vessels`, WI 856);
//! a bump remains reserved for reshaping a payload that real documents carry.

use crate::body_asset::BodyAsset;
use crate::frame::WorldPos;
use crate::system::System;
use crate::vessel::VesselRecord;
use crate::voxel::VoxelCraft;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Current on-disk format version. Increments **only** on a schema change — it is
/// deliberately independent of the crate's semantic version (which bumps one
/// patch per work item). A monotonic integer is what makes it a migration signal.
///
/// **Additive-variant rule:** adding a new [`Payload`]/[`Kind`] variant does **not**
/// bump this. Existing documents are byte-unchanged by a new variant, and an older
/// build meeting the new kind rejects it as an unknown kind (`Malformed`). A version
/// bump is reserved for changes to an *existing* payload's shape (which would require
/// a migration arm). `BodyAsset` (WI 760) and `System` (WI 761) were added additively.
///
/// History: 2 = WI 824 face panels (v1 retired at WI 820, body-asset kind
/// re-widened at WI 891); 3 = WI 892 surface-layer stack (`SurfaceRecipe`'s
/// flat areas → the ordered layer list; the v2 arm migrates on load).
pub const FORMAT_VERSION: u32 = 3;

/// What a serialized artifact is used as. One format, several uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Craft,
    Subassembly,
    Blueprint,
    WorldSave,
    /// A celestial body asset (WI 760) — added additively; the format version is
    /// unchanged because existing documents are untouched by a new payload variant.
    BodyAsset,
    /// A star system (WI 761): body-asset references + placements. Added additively.
    System,
    /// A vessel record (WI 855, multiplayer arc): a vessel's identity + time-stamped
    /// semantic state — the sync unit and world-save element. Added additively.
    VesselRecord,
    /// A body catalog reference (WI 891): recipe/archetype + seed + output
    /// version + digest — regenerates instead of snapshotting. Added additively.
    BodyRef,
}

/// A craft-scope serialized subgraph. A craft, a subassembly, and a blueprint are
/// the **same shape** at different scopes; the [`Payload`] kind distinguishes them.
///
/// At format version 1 the voxel/device contents are real (WI 505), filled in
/// place over WI 498's previously-opaque placeholders. The `resources` and `crew`
/// containers remain reserved (opaque `Value`s, empty) until their toys.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CraftSubgraph {
    /// Stable identifier. **Document/slug identity** (a blueprint's name-derived
    /// id, the library slot) — *not* instance identity: two spawns of one
    /// blueprint share this but are different vessels (see `vessel_id`).
    pub id: String,
    /// Human-facing display name.
    pub name: String,
    /// Durable vessel **instance** identity (WI 855, multiplayer arc): minted by
    /// [`crate::vessel::mint_vessel_id`] when a craft first becomes a shareable
    /// universe instance, kept for the vessel's life. Absent (and not encoded)
    /// for blueprints/templates and pre-855 documents — additive, no format bump.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vessel_id: Option<String>,
    /// A WI 497 world-coordinate value — the craft's reference placement.
    pub reference_position: WorldPos,
    /// The voxel lattice + devices + attachment interface (WI 505).
    #[serde(default)]
    pub craft: VoxelCraft,
    /// Reserved: resource reservoirs / converters / conduits (Toy 7).
    #[serde(default)]
    pub resources: Vec<serde_json::Value>,
    /// Reserved: assigned crew (later).
    #[serde(default)]
    pub crew: Vec<serde_json::Value>,
}

impl CraftSubgraph {
    /// Builds a craft subgraph carrying `craft`.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        reference_position: WorldPos,
        craft: VoxelCraft,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            vessel_id: None,
            reference_position,
            craft,
            resources: Vec::new(),
            crew: Vec::new(),
        }
    }
}

/// World-save payload — the same machinery scaled to the universe, **distinct**
/// from a craft subgraph. Skeletal until its reserved containers are consumed.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct WorldPayload {
    /// The universe's vessels as [`VesselRecord`]s (WI 856: the universe
    /// server's table serialized — the multiplayer design's "world-save as the
    /// snapshot substrate"; content WI 553 composes on this same shape, seam J1).
    ///
    /// Typed from the reserved opaque stub **without a format bump**: the
    /// container never had a producer (verified at WI 856 — no constructor
    /// outside tests existed), every prior document holds `[]`/absent, which
    /// decodes identically under both types, so the migration liability is
    /// exactly zero. Converter timestamps / terrain patches remain future
    /// world-persistence concerns.
    #[serde(default)]
    pub vessels: Vec<VesselRecord>,
    /// The played scenario's savable state (WI 553): content identity
    /// (scenario + packs + the frozen settings map), mission/session
    /// progress, and the flight state. Additive/optional — absent in every
    /// pre-553 document (and in the multiplayer server's vessels-only world
    /// saves), so `FORMAT_VERSION` is unchanged per the additive rule above.
    /// Economy state joins this member additively with WI 552.
    /// Boxed so the rarely-present member does not inflate every
    /// [`Payload`] (the same discipline as the netclient's boxed publish).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<Box<crate::world_save::ScenarioSaveState>>,
}

/// The payload, internally tagged by `kind`. The three craft-scope kinds share
/// [`CraftSubgraph`]; world-save carries its own [`WorldPayload`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Payload {
    Craft(CraftSubgraph),
    Subassembly(CraftSubgraph),
    Blueprint(CraftSubgraph),
    WorldSave(WorldPayload),
    /// A celestial body asset (WI 760): the intrinsic, reusable definition of a
    /// planet/moon (no placement). Carried by its own [`BodyAsset`] payload.
    BodyAsset(BodyAsset),
    /// A star system (WI 761): body-asset references + placements that compile to a
    /// `Universe`.
    System(System),
    /// A vessel record (WI 855, multiplayer arc): identity + ownership + a
    /// universe-time-stamped craft subgraph and rails motion — the multiplayer
    /// sync unit and the world-save vessel element. Added additively; the format
    /// version is unchanged per the additive-variant rule.
    VesselRecord(VesselRecord),
    /// A body catalog reference (WI 891): the persisted "recipe+seed is truth"
    /// spelling — the body regenerates through resolve/derive on load and is
    /// digest-verified. Added additively per the additive-variant rule.
    BodyRef(crate::body_ref::BodyRef),
}

impl Payload {
    /// The kind tag for this payload.
    pub fn kind(&self) -> Kind {
        match self {
            Payload::Craft(_) => Kind::Craft,
            Payload::Subassembly(_) => Kind::Subassembly,
            Payload::Blueprint(_) => Kind::Blueprint,
            Payload::WorldSave(_) => Kind::WorldSave,
            Payload::BodyAsset(_) => Kind::BodyAsset,
            Payload::System(_) => Kind::System,
            Payload::VesselRecord(_) => Kind::VesselRecord,
            Payload::BodyRef(_) => Kind::BodyRef,
        }
    }
}

/// A complete versioned document: the format envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedDocument {
    /// The format version this document was written at.
    pub format_version: u32,
    /// The payload (and, via its tag, the kind).
    pub payload: Payload,
}

impl SavedDocument {
    /// Wraps a payload at the current [`FORMAT_VERSION`].
    pub fn new(payload: Payload) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            payload,
        }
    }

    /// The kind of this document.
    pub fn kind(&self) -> Kind {
        self.payload.kind()
    }

    /// Serializes to pretty JSON (human-inspectable).
    pub fn to_json(&self) -> Result<String, FormatError> {
        serde_json::to_string_pretty(self).map_err(|e| FormatError::Malformed(e.to_string()))
    }

    /// Loads from JSON with a **two-stage parse**: a version-stable [`VersionProbe`]
    /// reads the format version *first* and rejects an unsupported one, so a newer
    /// file is rejected by version rather than by an incidental payload-shape
    /// mismatch. The `match` below is the **migration seam** — call sites do not
    /// change when future migrations are added; new arms attach here.
    pub fn from_json(s: &str) -> Result<SavedDocument, FormatError> {
        let probe: VersionProbe =
            serde_json::from_str(s).map_err(|e| FormatError::Malformed(e.to_string()))?;
        match probe.format_version {
            FORMAT_VERSION => {
                let mut doc: SavedDocument =
                    serde_json::from_str(s).map_err(|e| FormatError::Malformed(e.to_string()))?;
                doc.normalize_craft();
                doc.validate_craft_shapes()?;
                Ok(doc)
            }
            // v2 → v3 (WI 892): the only reshape is `SurfaceRecipe` (flat
            // `terrain`/`crater`/`material` areas → the ordered layer stack),
            // which real documents carry in exactly two places — a body-asset
            // payload's `surface`, and the embedded bodies of a world-save's
            // snapshot-tier records (whose recorded digests are then restored
            // over the migrated body; the recorded output_version is left
            // untouched, so an old snapshot stays an old-generator pin and
            // WI 891's drift line reports it). Every other kind is
            // shape-unchanged and re-tags through the direct parse.
            2 => {
                let kind: KindProbe =
                    serde_json::from_str(s).map_err(|e| FormatError::Malformed(e.to_string()))?;
                let mut doc = match kind.payload.kind {
                    Kind::BodyAsset | Kind::WorldSave => {
                        let mut root: serde_json::Value = serde_json::from_str(s)
                            .map_err(|e| FormatError::Malformed(e.to_string()))?;
                        if let Some(surface) = root.pointer_mut("/payload/surface") {
                            migrate_flat_surface(surface);
                        }
                        if let Some(records) = root
                            .pointer_mut("/payload/scenario/bodies")
                            .and_then(|v| v.as_array_mut())
                        {
                            for record in records {
                                if let Some(surface) = record.pointer_mut("/body/surface") {
                                    migrate_flat_surface(surface);
                                }
                            }
                        }
                        serde_json::from_value::<SavedDocument>(root)
                            .map_err(|e| FormatError::Malformed(e.to_string()))?
                    }
                    _ => serde_json::from_str::<SavedDocument>(s)
                        .map_err(|e| FormatError::Malformed(e.to_string()))?,
                };
                doc.format_version = FORMAT_VERSION;
                if let Payload::WorldSave(w) = &mut doc.payload {
                    if let Some(scenario) = &mut w.scenario {
                        for record in &mut scenario.bodies {
                            if let crate::world_save::SavedBodyRecord::Snapshot {
                                digest,
                                body,
                                ..
                            } = record
                            {
                                *digest = crate::body_digest::digest_hex(body);
                            }
                        }
                    }
                }
                doc.normalize_craft();
                doc.validate_craft_shapes()?;
                Ok(doc)
            }
            // Format v1 (pre-WI-824 legacy cell-panel flags) was retired at
            // WI 820 (pre-release, owner direction): its migration is gone and
            // v1 files are rejected by version — **except the body-asset
            // kind** (WI 891): the v1→v2 reshape was craft-scope only, and
            // real pre-recipe body files exist on disk (`saves/bodies/`).
            // Those migrate in memory, chaining through the WI 892 surface
            // conversion (a re-save writes the current version); every other
            // v1 kind keeps the WI 820 rejection.
            1 => {
                let kind: KindProbe =
                    serde_json::from_str(s).map_err(|_| FormatError::UnsupportedVersion(1))?;
                if kind.payload.kind != Kind::BodyAsset {
                    return Err(FormatError::UnsupportedVersion(1));
                }
                let mut root: serde_json::Value =
                    serde_json::from_str(s).map_err(|e| FormatError::Malformed(e.to_string()))?;
                if let Some(surface) = root.pointer_mut("/payload/surface") {
                    migrate_flat_surface(surface);
                }
                let mut doc = serde_json::from_value::<SavedDocument>(root)
                    .map_err(|e| FormatError::Malformed(e.to_string()))?;
                doc.format_version = FORMAT_VERSION;
                doc.normalize_craft();
                doc.validate_craft_shapes()?;
                Ok(doc)
            }
            other => Err(FormatError::UnsupportedVersion(other)),
        }
    }

    /// Restore the sorted-store invariants on a craft-scope payload (defensive,
    /// for documents whose producer did not order the stores — the WI 820
    /// determinism discipline).
    fn normalize_craft(&mut self) {
        let craft = match &mut self.payload {
            Payload::Craft(c) | Payload::Subassembly(c) | Payload::Blueprint(c) => &mut c.craft,
            // A vessel record embeds a full craft subgraph — it gets the same
            // load hygiene as the craft-scope kinds (WI 855).
            Payload::VesselRecord(r) => &mut r.structure.craft,
            Payload::WorldSave(_)
            | Payload::BodyAsset(_)
            | Payload::System(_)
            | Payload::BodyRef(_) => return,
        };
        craft.normalize_shapes();
        craft.normalize_face_panels();
    }

    /// Validate shaped-cell data on a craft-scope payload (WI 831): the
    /// orientation is an index into the frozen 24-entry rotation table — an
    /// out-of-range value is malformed input, rejected at the boundary (never a
    /// panic or a silent modulo). Unknown form names are already rejected by the
    /// enum decode itself.
    fn validate_craft_shapes(&self) -> Result<(), FormatError> {
        let craft = match &self.payload {
            Payload::Craft(c) | Payload::Subassembly(c) | Payload::Blueprint(c) => &c.craft,
            // Same boundary validation for the subgraph a vessel record embeds.
            Payload::VesselRecord(r) => &r.structure.craft,
            Payload::WorldSave(_)
            | Payload::BodyAsset(_)
            | Payload::System(_)
            | Payload::BodyRef(_) => return Ok(()),
        };
        for s in &craft.shapes {
            if s.orientation as usize >= crate::shape::rotations().len() {
                return Err(FormatError::Malformed(format!(
                    "shaped cell ({}, {}, {}): orientation {} out of range (0..24)",
                    s.cell.x, s.cell.y, s.cell.z, s.orientation
                )));
            }
        }
        Ok(())
    }
}

/// Version-stable header for the first load stage. Deserializes only the format
/// version, ignoring the payload, so it parses regardless of how a future
/// version's payload is shaped. **Its fields must not change across versions.**
#[derive(Deserialize)]
struct VersionProbe {
    format_version: u32,
}

/// Kind-only probe for the v1 compatibility arm (WI 891): reads just the
/// payload's `kind` tag, so a v1 craft document (whose payload shape no longer
/// parses) is still classified — and rejected by version — without a payload
/// decode.
#[derive(Deserialize)]
struct KindProbe {
    payload: KindOnly,
}

/// See [`KindProbe`].
#[derive(Deserialize)]
struct KindOnly {
    kind: Kind,
}

/// v2 → v3 surface conversion (WI 892), at the JSON level so the pre-stack
/// shape needs no legacy struct: the flat `terrain`/`crater`/`material` areas
/// become one layer each — **in that canonical order** — carrying their blob
/// as the layer's params; a null area produces no layer (exactly the lenient
/// readers' "absent = defaults"), and each converted layer takes its type's
/// slug as its well-known id. Unrecognized keys inside a blob are carried
/// verbatim (the readers ignored them before; no new data loss). A surface
/// already carrying `layers` is left untouched (defensive idempotence).
fn migrate_flat_surface(surface: &mut serde_json::Value) {
    let Some(obj) = surface.as_object_mut() else {
        return;
    };
    if obj.contains_key("layers") {
        return;
    }
    let mut layers = Vec::new();
    for layer_type in crate::body_asset::SurfaceLayerType::ALL {
        if let Some(params) = obj.remove(layer_type.slug()) {
            if !params.is_null() {
                layers.push(serde_json::json!({
                    "id": layer_type.slug(),
                    "layer_type": layer_type.slug(),
                    "enabled": true,
                    "params": params,
                }));
            }
        }
    }
    obj.insert("layers".to_string(), serde_json::Value::Array(layers));
}

/// A persistence-format error. Typed and non-panicking, so malformed or foreign
/// input is rejected cleanly at the boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum FormatError {
    /// Malformed/truncated input, a missing required field (including a missing
    /// version), or an unknown kind — and, on save, a serialize failure.
    Malformed(String),
    /// The document declares a format version this build does not support.
    UnsupportedVersion(u32),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::Malformed(m) => write!(f, "malformed serialized document: {m}"),
            FormatError::UnsupportedVersion(v) => write!(
                f,
                "unsupported format version {v} (this build supports {FORMAT_VERSION})"
            ),
        }
    }
}

impl std::error::Error for FormatError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameId;
    use crate::voxel::{Material, Voxel};
    use glam::{DVec3, IVec3};

    fn sample_pos() -> WorldPos {
        WorldPos::new(FrameId::CENTRAL_BODY, DVec3::new(1.0, 2.0, 3.0))
    }

    fn sample_craft() -> VoxelCraft {
        let mut craft = VoxelCraft::new(1.0);
        craft.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        craft
    }

    /// WI 892 (scenarios A1/A3): the v2 arm converts flat surface areas into
    /// the ordered layer stack (canonical order, params carried verbatim,
    /// well-known ids), retags shape-unchanged kinds, and the migrated
    /// document re-saves at v3 byte-stably.
    #[test]
    fn v2_documents_migrate_flat_surfaces_and_retag() {
        use crate::body_asset::SurfaceLayerType;
        // A v2 body asset with populated crater/material areas.
        let mut root = serde_json::to_value(SavedDocument::new(Payload::BodyAsset(
            BodyAsset::earthlike(),
        )))
        .unwrap();
        root["format_version"] = 2.into();
        root["payload"]["surface"] = serde_json::json!({
            "seed": 7,
            "terrain": null,
            "crater": { "density": 0.5, "depth": 2.0 },
            "material": { "temperature": -10.0, "palette": "unrecognized-carried" }
        });
        let doc = SavedDocument::from_json(&root.to_string()).expect("v2 body migrates");
        assert_eq!(doc.format_version, FORMAT_VERSION);
        let Payload::BodyAsset(a) = &doc.payload else {
            panic!("body scope");
        };
        assert_eq!(a.surface.seed, 7);
        assert_eq!(
            a.surface
                .layers
                .iter()
                .map(|l| (l.id.as_str(), l.layer_type, l.enabled))
                .collect::<Vec<_>>(),
            vec![
                ("crater", SurfaceLayerType::Crater, true),
                ("material", SurfaceLayerType::Material, true),
            ],
            "null terrain ⇒ no layer; canonical order; well-known ids"
        );
        // The readers see exactly the values the flat areas carried —
        // including unrecognized keys carried verbatim (no new data loss).
        assert_eq!(
            a.surface.params_of(SurfaceLayerType::Crater)["density"].as_f64(),
            Some(0.5)
        );
        assert_eq!(
            a.surface.params_of(SurfaceLayerType::Material)["temperature"].as_f64(),
            Some(-10.0)
        );
        assert_eq!(
            a.surface.params_of(SurfaceLayerType::Material)["palette"].as_str(),
            Some("unrecognized-carried")
        );
        // Migrated documents re-save at v3, byte-stably.
        let resaved = doc.to_json().unwrap();
        assert!(resaved.contains("\"format_version\": 3"));
        assert_eq!(
            SavedDocument::from_json(&resaved)
                .unwrap()
                .to_json()
                .unwrap(),
            resaved
        );

        // A v2 craft document (shape unchanged since v2) retags through the
        // direct parse: payload identical, version current, re-save stable.
        let craft_doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "c",
            "C",
            sample_pos(),
            sample_craft(),
        )));
        let mut croot = serde_json::to_value(&craft_doc).unwrap();
        croot["format_version"] = 2.into();
        let migrated = SavedDocument::from_json(&croot.to_string()).expect("v2 craft retags");
        assert_eq!(migrated.format_version, FORMAT_VERSION);
        assert_eq!(migrated.payload, craft_doc.payload, "payload untouched");
        let out = migrated.to_json().unwrap();
        assert_eq!(
            SavedDocument::from_json(&out).unwrap().to_json().unwrap(),
            out
        );
    }

    /// WI 891 (AC 3): a **v1 body-asset** document — the exact shape of the
    /// stranded pre-recipe file `saves/bodies/rocky-planet-000e.json` — loads
    /// through the compatibility arm, migrates in memory, and re-saves at the
    /// current version. Every other v1 kind keeps the WI 820 rejection (the
    /// craft case is pinned by the test below); a v1 world-save is rejected
    /// by version, and a v1 body-asset with a broken payload is `Malformed`,
    /// not silently versioned through.
    #[test]
    fn v1_body_assets_load_via_the_compat_arm_and_other_v1_kinds_stay_rejected() {
        // A faithful reduction of the on-disk stranded file's shape.
        let v1_body = r#"{
          "format_version": 1,
          "payload": {
            "kind": "body_asset",
            "id": "gen-rocky-000000000000000e",
            "name": "Rocky Planet 000E",
            "mu": 62764103801452.414,
            "radius": 4081155.3230753234,
            "rotation": { "axis": [0.0, 0.0, 1.0], "sidereal_period": 89564.68576833663 },
            "fluid_medium": {
              "atmosphere_surface_density": 1.9500294966428244,
              "atmosphere_surface_pressure": 101513.99014409662,
              "atmosphere_scale_height": 5070.794859367081,
              "ocean_surface_density": 0.0,
              "ocean_surface_pressure": 0.0,
              "ocean_density_gradient": 0.0,
              "gravity": 3.768296652429817,
              "atmosphere_temperature": 286.1116039326881,
              "ocean_temperature": 280.0
            },
            "surface": { "seed": 14, "terrain": null, "crater": null, "material": null },
            "render": null
          }
        }"#;
        let doc = SavedDocument::from_json(v1_body).expect("v1 body asset loads");
        assert_eq!(doc.format_version, FORMAT_VERSION, "migrated in memory");
        let Payload::BodyAsset(a) = &doc.payload else {
            panic!("body scope");
        };
        assert_eq!(a.id, "gen-rocky-000000000000000e");
        assert_eq!(a.surface.seed, 14);
        // A re-save writes the current version; loading that back is clean
        // (save → load → save is stable once migrated).
        let resaved = doc.to_json().unwrap();
        assert!(resaved.contains("\"format_version\": 3"));
        assert_eq!(
            SavedDocument::from_json(&resaved)
                .unwrap()
                .to_json()
                .unwrap(),
            resaved
        );

        // A v1 world-save stays rejected by version.
        let v1_world = r#"{"format_version":1,"payload":{"kind":"world_save","vessels":[]}}"#;
        assert_eq!(
            SavedDocument::from_json(v1_world),
            Err(FormatError::UnsupportedVersion(1))
        );

        // A v1 body-asset whose payload no longer parses is malformed, not
        // silently versioned through.
        let v1_broken = r#"{"format_version":1,"payload":{"kind":"body_asset","id":"x"}}"#;
        assert!(matches!(
            SavedDocument::from_json(v1_broken),
            Err(FormatError::Malformed(_))
        ));
    }

    #[test]
    fn v1_documents_are_rejected_by_version_and_paneled_crafts_resave_byte_identically() {
        // WI 820 (owner direction, pre-release): format-v1 support is retired —
        // a v1 file is rejected by the version probe with a clear error, not
        // migrated.
        let mut doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "old",
            "Old",
            sample_pos(),
            sample_craft(),
        )));
        doc.format_version = 1; // pretend it was written pre-824
        let json = serde_json::to_string(&doc).unwrap();
        assert_eq!(
            SavedDocument::from_json(&json),
            Err(FormatError::UnsupportedVersion(1)),
            "v1 is rejected by version"
        );

        // The WI's founding AC, on the panel-carrying case that motivated it:
        // a plated hull re-saves byte-identically (no unordered persisted
        // containers remain — the panel store is sorted by construction), and
        // the retired legacy field is gone from the encoding entirely.
        let mut cells = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    if !(x == 1 && y == 1 && z == 1) {
                        cells.push(IVec3::new(x, y, z));
                    }
                }
            }
        }
        let mut hull = VoxelCraft::new(0.5);
        hull.plate_shell(&cells, Material::ALUMINIUM);
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "hull",
            "Hull",
            sample_pos(),
            hull,
        )));
        let s1 = doc.to_json().unwrap();
        assert!(
            !s1.contains("\"panels\""),
            "the legacy field is not encoded"
        );
        assert!(s1.contains("\"face_panels\""), "the plated hull is real");
        let s2 = SavedDocument::from_json(&s1).unwrap().to_json().unwrap();
        assert_eq!(s2, s1, "paneled re-save is byte-identical");

        // A flagless craft round-trips bit-stable through the same seam.
        let plain = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "plain",
            "Plain",
            sample_pos(),
            sample_craft(),
        )));
        let back = SavedDocument::from_json(&plain.to_json().unwrap()).unwrap();
        assert_eq!(plain, back);
    }

    #[test]
    fn pre_shape_documents_resave_byte_identically_and_shapes_round_trip() {
        use crate::shape::{FillMode, Form, ShapedCell};
        // WI 831 byte-compat: a craft without shapes serializes with no shapes
        // field at all, so load → save is byte-identical.
        let plain = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "plain",
            "Plain",
            sample_pos(),
            sample_craft(),
        )));
        let s1 = plain.to_json().unwrap();
        assert!(!s1.contains("\"shapes\""), "absent field is skipped");
        let s2 = SavedDocument::from_json(&s1).unwrap().to_json().unwrap();
        assert_eq!(s1, s2, "pre-shape re-save is byte-identical");

        // A shaped craft round-trips form/orientation/fill exactly.
        let mut craft = sample_craft();
        craft.set_shape(ShapedCell {
            cell: IVec3::ZERO,
            form: Form::SlopeHigh,
            orientation: 7,
            fill: FillMode::Solid,
        });
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "shaped",
            "Shaped",
            sample_pos(),
            craft,
        )));
        let loaded = SavedDocument::from_json(&doc.to_json().unwrap()).unwrap();
        let Payload::Craft(c) = &loaded.payload else {
            panic!("craft payload expected");
        };
        let s = c.craft.shape_at(IVec3::ZERO).expect("shape survives");
        assert_eq!(
            (s.form, s.orientation, s.fill),
            (Form::SlopeHigh, 7, FillMode::Solid)
        );
    }

    #[test]
    fn out_of_range_orientation_is_rejected_at_load() {
        use crate::shape::{FillMode, Form, ShapedCell};
        // WI 831: orientation indexes the frozen 24-entry table; 24+ is
        // malformed input, rejected at the boundary — never a silent modulo.
        let mut craft = sample_craft();
        craft.set_shape(ShapedCell {
            cell: IVec3::ZERO,
            form: Form::Wedge,
            orientation: 24,
            fill: FillMode::Solid,
        });
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "bad",
            "Bad",
            sample_pos(),
            craft,
        )));
        let err = SavedDocument::from_json(&doc.to_json().unwrap()).unwrap_err();
        let FormatError::Malformed(msg) = err else {
            panic!("expected Malformed, got {err:?}");
        };
        assert!(msg.contains("orientation"), "{msg}");
    }

    #[test]
    fn craft_round_trips_preserving_version_and_kind() {
        let pos = sample_pos();
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "ship-1",
            "Ranger",
            pos,
            sample_craft(),
        )));
        let json = doc.to_json().unwrap();
        let back = SavedDocument::from_json(&json).unwrap();
        assert_eq!(doc, back);
        assert_eq!(back.format_version, FORMAT_VERSION);
        assert_eq!(back.kind(), Kind::Craft);
    }

    #[test]
    fn all_four_kinds_round_trip_through_one_envelope() {
        let pos = sample_pos();
        let payloads = [
            Payload::Craft(CraftSubgraph::new("c", "C", pos, sample_craft())),
            Payload::Subassembly(CraftSubgraph::new("s", "S", pos, sample_craft())),
            Payload::Blueprint(CraftSubgraph::new("b", "B", pos, sample_craft())),
            Payload::WorldSave(WorldPayload::default()),
        ];
        for p in payloads {
            let kind = p.kind();
            let doc = SavedDocument::new(p);
            let back = SavedDocument::from_json(&doc.to_json().unwrap()).unwrap();
            assert_eq!(doc, back);
            assert_eq!(back.kind(), kind);
        }
    }

    #[test]
    fn body_asset_round_trips_through_the_envelope() {
        use crate::body_asset::BodyAsset;
        let doc = SavedDocument::new(Payload::BodyAsset(BodyAsset::earthlike()));
        let back = SavedDocument::from_json(&doc.to_json().unwrap()).unwrap();
        assert_eq!(back.format_version, FORMAT_VERSION);
        assert_eq!(back.kind(), Kind::BodyAsset);
        let Payload::BodyAsset(a) = &back.payload else {
            panic!("expected a body asset");
        };
        assert_eq!(a.id, "earthlike");
        assert_eq!(a.central_body().mu, crate::sim::CentralBody::EARTHLIKE.mu);
    }

    #[test]
    fn system_round_trips_through_the_envelope() {
        use crate::system::System;
        let sys = System::single_body("sol", "Sol", "earthlike");
        let doc = SavedDocument::new(Payload::System(sys.clone()));
        let back = SavedDocument::from_json(&doc.to_json().unwrap()).unwrap();
        assert_eq!(back.format_version, FORMAT_VERSION);
        assert_eq!(back.kind(), Kind::System);
        let Payload::System(s) = &back.payload else {
            panic!("expected a system");
        };
        assert_eq!(s, &sys);
    }

    #[test]
    fn craft_subassembly_blueprint_share_the_subgraph_payload() {
        let pos = sample_pos();
        let cs = CraftSubgraph::new("x", "X", pos, sample_craft());
        // Same shape carried by three kinds.
        for payload in [
            Payload::Craft(cs.clone()),
            Payload::Subassembly(cs.clone()),
            Payload::Blueprint(cs.clone()),
        ] {
            if let Payload::Craft(c) | Payload::Subassembly(c) | Payload::Blueprint(c) = &payload {
                assert_eq!(c, &cs);
            } else {
                unreachable!();
            }
        }
    }

    #[test]
    fn newer_version_rejected_by_version_not_payload_parse() {
        // Alien payload shape (a bare number) this build cannot parse — the
        // version-stable probe still reads the version and rejects by version.
        let newer = r#"{ "format_version": 4, "payload": 12345 }"#;
        assert_eq!(
            SavedDocument::from_json(newer),
            Err(FormatError::UnsupportedVersion(4))
        );
    }

    #[test]
    fn missing_version_field_is_rejected_not_assumed_v1() {
        let no_version = r#"{ "payload": { "kind": "craft" } }"#;
        assert!(matches!(
            SavedDocument::from_json(no_version),
            Err(FormatError::Malformed(_))
        ));
    }

    #[test]
    fn malformed_input_is_rejected_without_panic() {
        for bad in ["{ not json", "", "[1,2,3", "null"] {
            assert!(matches!(
                SavedDocument::from_json(bad),
                Err(FormatError::Malformed(_))
            ));
        }
    }

    #[test]
    fn unknown_kind_is_rejected() {
        // Current-format version, so the *kind* is what gets judged (v1 itself
        // is rejected by version since WI 820).
        let s = r#"{ "format_version": 2, "payload": { "kind": "spaceship", "id": "a", "name": "b" } }"#;
        assert!(matches!(
            SavedDocument::from_json(s),
            Err(FormatError::Malformed(_))
        ));
    }

    #[test]
    fn payload_embeds_worldpos_voxels_and_reserved_containers() {
        let pos = sample_pos();
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "id",
            "Name",
            pos,
            sample_craft(),
        )));
        let back = SavedDocument::from_json(&doc.to_json().unwrap()).unwrap();
        let Payload::Craft(c) = &back.payload else {
            panic!("expected craft");
        };
        assert_eq!(c.reference_position, pos);
        // Real voxel content round-trips; resources/crew stay reserved (empty).
        assert_eq!(c.craft.voxels.len(), 1);
        assert!(c.resources.is_empty());
        assert!(c.crew.is_empty());
    }

    #[test]
    fn vessel_id_is_absent_by_default_and_round_trips_when_present() {
        // WI 855 additive discipline: a subgraph without a vessel id encodes
        // with NO vessel_id field — pre-855 documents and re-saves are
        // byte-unchanged (the paneled byte-identity test above covers the
        // re-save; this pins the field's absence explicitly).
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "plain",
            "Plain",
            sample_pos(),
            sample_craft(),
        )));
        let json = doc.to_json().unwrap();
        assert!(!json.contains("\"vessel_id\""), "absent field is skipped");
        let back = SavedDocument::from_json(&json).unwrap();
        assert_eq!(doc, back);

        // Present, it round-trips exactly.
        let mut cs = CraftSubgraph::new("mine", "Mine", sample_pos(), sample_craft());
        cs.vessel_id = Some(crate::vessel::mint_vessel_id());
        let want = cs.vessel_id.clone();
        let doc = SavedDocument::new(Payload::Craft(cs));
        let back = SavedDocument::from_json(&doc.to_json().unwrap()).unwrap();
        let Payload::Craft(c) = &back.payload else {
            panic!("craft payload expected");
        };
        assert_eq!(c.vessel_id, want);
    }

    #[test]
    fn vessel_record_round_trips_through_the_envelope_in_all_shapes() {
        use crate::orbit::Orbit;
        use crate::vessel::{Fate, MotionState, VesselRecord};
        use glam::DVec2;

        // WI 855: the additive vessel-record kind — both motion variants, with
        // and without the optional authority/subspace/fate fields, preserve
        // version + kind and round-trip exactly.
        let mu = crate::sim::CentralBody::EARTHLIKE.mu;
        let orbit = Orbit::from_state(
            mu,
            DVec2::new(7.0e6, 0.0),
            DVec2::new(0.0, (mu / 7.0e6).sqrt()),
            0.0,
        )
        .unwrap();
        let structure = CraftSubgraph::new("s", "S", sample_pos(), sample_craft());

        let minimal = VesselRecord::from_rails(
            crate::vessel::mint_vessel_id(),
            "Ranger",
            "dave",
            10.0,
            FrameId::CENTRAL_BODY,
            orbit,
            structure.clone(),
        );
        let mut full = VesselRecord::from_surface(
            crate::vessel::mint_vessel_id(),
            "Lander",
            "dave",
            20.0,
            sample_pos(),
            structure,
        );
        full.authority = Some("session-1".into());
        full.subspace = Some("subspace-1".into());
        full.live = true;
        full.fate = Some(Fate::Destroyed);

        for rec in [minimal, full] {
            let doc = SavedDocument::new(Payload::VesselRecord(rec.clone()));
            let json = doc.to_json().unwrap();
            let back = SavedDocument::from_json(&json).unwrap();
            assert_eq!(back.format_version, FORMAT_VERSION);
            assert_eq!(back.kind(), Kind::VesselRecord);
            let Payload::VesselRecord(r) = &back.payload else {
                panic!("vessel record expected");
            };
            assert_eq!(r, &rec);
            match (&r.motion, &rec.motion) {
                (MotionState::Conic { .. }, MotionState::Conic { .. })
                | (MotionState::SurfaceFix { .. }, MotionState::SurfaceFix { .. }) => {}
                _ => panic!("motion variant preserved"),
            }
        }
    }

    #[test]
    fn json_is_human_inspectable_with_version_and_kind() {
        let doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "id",
            "Name",
            sample_pos(),
            sample_craft(),
        )));
        let json = doc.to_json().unwrap();
        assert!(json.contains("format_version"));
        assert!(json.contains("\"kind\""));
        assert!(json.contains("\"craft\""));
    }

    #[test]
    fn format_version_is_three() {
        // v2 = WI 824 (face panels; legacy cell-panel flags convert at load);
        // v3 = WI 892 (surface-layer stack; the v2 arm converts at load).
        assert_eq!(FORMAT_VERSION, 3);
    }
}

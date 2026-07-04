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
//! that later toys fill in (a future format-version change).

use crate::body_asset::BodyAsset;
use crate::frame::WorldPos;
use crate::system::System;
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
pub const FORMAT_VERSION: u32 = 2;

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
}

/// A craft-scope serialized subgraph. A craft, a subassembly, and a blueprint are
/// the **same shape** at different scopes; the [`Payload`] kind distinguishes them.
///
/// At format version 1 the voxel/device contents are real (WI 505), filled in
/// place over WI 498's previously-opaque placeholders. The `resources` and `crew`
/// containers remain reserved (opaque `Value`s, empty) until their toys.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CraftSubgraph {
    /// Stable identifier.
    pub id: String,
    /// Human-facing display name.
    pub name: String,
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
            reference_position,
            craft,
            resources: Vec::new(),
            crew: Vec::new(),
        }
    }
}

/// Reserved, skeletal world-save payload — the same machinery scaled to the
/// universe, **distinct** from a craft subgraph. Empty at format version 1.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct WorldPayload {
    /// On-rails vessels and their orbits, converter timestamps, terrain patches
    /// (world persistence, later).
    #[serde(default)]
    pub vessels: Vec<serde_json::Value>,
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
            // v1 and v2 share the payload shape; a v1 craft may carry legacy
            // cell-panel flags, which the WI 824 conversion turns into face
            // panels. New documents are always written at v2.
            1 | FORMAT_VERSION => {
                let mut doc: SavedDocument =
                    serde_json::from_str(s).map_err(|e| FormatError::Malformed(e.to_string()))?;
                doc.convert_craft_panels();
                // A migrated document *is* a current-format document: stamp it, so
                // re-serializing it can never produce a v1 file carrying v2 content
                // (which an old build would load and silently strip).
                doc.format_version = FORMAT_VERSION;
                Ok(doc)
            }
            other => Err(FormatError::UnsupportedVersion(other)),
        }
    }

    /// The WI 824 migration: convert legacy cell-panel flags to face panels on a
    /// craft-scope payload (the panels design's R1 rule), logging a per-craft
    /// summary with the compartment-count delta. A document without flags is only
    /// normalized (face-panel order), never altered.
    fn convert_craft_panels(&mut self) {
        let craft = match &mut self.payload {
            Payload::Craft(c) | Payload::Subassembly(c) | Payload::Blueprint(c) => {
                (c.id.clone(), &mut c.craft)
            }
            Payload::WorldSave(_) | Payload::BodyAsset(_) | Payload::System(_) => return,
        };
        let (id, craft) = craft;
        if craft.panels.is_empty() {
            craft.normalize_face_panels();
            return;
        }
        let before = crate::compartments::compartments(craft).count();
        let report = craft.convert_legacy_panels();
        let after = crate::compartments::compartments(craft).count();
        if let Some(r) = report {
            bevy_log::info!(
                "craft `{id}`: converted {} legacy panel cell(s) to {} face panel(s) \
                 ({} embedded cell(s) kept solid); compartments {before} -> {after}",
                r.cells_converted,
                r.plates_created,
                r.cells_kept_solid,
            );
        }
    }
}

/// Version-stable header for the first load stage. Deserializes only the format
/// version, ignoring the payload, so it parses regardless of how a future
/// version's payload is shaped. **Its fields must not change across versions.**
#[derive(Deserialize)]
struct VersionProbe {
    format_version: u32,
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

    #[test]
    fn v1_legacy_panel_flags_convert_to_face_panels_at_load() {
        // WI 824: a v1 document whose craft carries cell-panel flags loads,
        // converts per the R1 rule, and comes out flag-free with face panels;
        // a v2 (or flagless v1) document is untouched beyond normalization.
        let mut craft = sample_craft(); // one voxel at the origin
        craft.set_panel(IVec3::ZERO, true);
        let mut doc = SavedDocument::new(Payload::Craft(CraftSubgraph::new(
            "legacy",
            "Legacy",
            sample_pos(),
            craft,
        )));
        doc.format_version = 1; // pretend it was written pre-824
        let json = serde_json::to_string(&doc).unwrap();
        let loaded = SavedDocument::from_json(&json).unwrap();
        let Payload::Craft(c) = &loaded.payload else {
            panic!("craft payload expected");
        };
        assert_eq!(
            loaded.format_version, FORMAT_VERSION,
            "a migrated document is stamped at the current format"
        );
        assert!(c.craft.panels.is_empty(), "flags consumed");
        assert!(c.craft.voxels.is_empty(), "the lone flagged cell emptied");
        assert_eq!(
            c.craft.face_panels.len(),
            6,
            "a free-standing flagged cell plates all six faces"
        );

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
        let newer = r#"{ "format_version": 3, "payload": 12345 }"#;
        assert_eq!(
            SavedDocument::from_json(newer),
            Err(FormatError::UnsupportedVersion(3))
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
        let s = r#"{ "format_version": 1, "payload": { "kind": "spaceship", "id": "a", "name": "b" } }"#;
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
    fn format_version_is_two() {
        // v2 = WI 824 (face panels; legacy cell-panel flags convert at load).
        assert_eq!(FORMAT_VERSION, 2);
    }
}

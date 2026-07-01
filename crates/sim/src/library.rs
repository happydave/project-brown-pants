//! Craft save library (WI 675).
//!
//! A directory of **named** craft saves, so the editor can keep many vehicles
//! instead of the single fixed-file quick-save. This is the editor-local craft
//! library — deliberately distinct from the heavier world-save persistence
//! (WI 553), which owns the [`crate::persist::Payload::WorldSave`] kind and is
//! untouched here.
//!
//! The serialization format is already multi-vehicle: a [`SavedDocument`] wrapping
//! a [`CraftSubgraph`] carries a stable `id` and a human `name`. This module adds
//! only the directory layout, a filesystem-safe **slug** derived from the display
//! name (the file identity), and discovery by scanning. The pure pieces (slugging,
//! round-trip via the existing format) are unit-testable without any rendering.

use crate::frame::{FrameId, WorldPos};
use crate::persist::{CraftSubgraph, FormatError, Kind, Payload, SavedDocument};
use crate::voxel::VoxelCraft;
use glam::DVec3;
use std::path::{Path, PathBuf};

/// On-disk extension for a saved craft document.
const CRAFT_EXT: &str = "json";
/// Upper bound on a slug's length, so a pathological name can't produce an
/// unwieldy filename. Chosen to stay well under common filesystem limits.
const MAX_SLUG_LEN: usize = 64;
/// Slug used when a name reduces to nothing slug-worthy (all punctuation/empty).
const FALLBACK_SLUG: &str = "craft";

/// An error from a library operation. Wraps the I/O and format boundaries so call
/// sites get a single typed result instead of two error families.
#[derive(Debug)]
pub enum LibraryError {
    /// A filesystem failure (create dir, read/write file, scan directory).
    Io(String),
    /// The bytes on disk are not a valid craft document.
    Format(FormatError),
    /// The requested name has no slug-worthy characters *and* an empty-name save was
    /// rejected by the caller's validation. Only returned by [`save_craft`] when the
    /// derived slug is empty before the fallback is applied — callers should validate
    /// the display name first (the overlay rejects blank names).
    EmptyName,
}

impl std::fmt::Display for LibraryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibraryError::Io(m) => write!(f, "craft library I/O error: {m}"),
            LibraryError::Format(e) => write!(f, "craft library format error: {e}"),
            LibraryError::EmptyName => write!(f, "craft name is empty after slugging"),
        }
    }
}

impl std::error::Error for LibraryError {}

impl From<FormatError> for LibraryError {
    fn from(e: FormatError) -> Self {
        LibraryError::Format(e)
    }
}

/// Derives a filesystem-safe slug from a display name: lowercase, every run of
/// non-alphanumeric characters collapsed to a single `-`, leading/trailing `-`
/// trimmed, and bounded to [`MAX_SLUG_LEN`]. ASCII alphanumerics pass through; any
/// other character (Unicode, punctuation, whitespace) is a separator. Two distinct
/// slugs map to two distinct files; names that slug identically share one slot (an
/// intentional update — see the module/plan note on slug-as-identity).
///
/// Returns an empty string only when the name contains no ASCII alphanumerics;
/// [`save_craft`] substitutes [`FALLBACK_SLUG`] in that case.
pub fn slugify(name: &str) -> String {
    let mut slug = String::with_capacity(name.len().min(MAX_SLUG_LEN));
    let mut prev_dash = true; // leading: suppress a leading '-'
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
        if slug.len() >= MAX_SLUG_LEN {
            break;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    slug
}

/// A saved craft discovered in the library directory: its display name (from the
/// document), its slug (the file stem / identity), and the file path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CraftEntry {
    /// Human-facing name read from the document's [`CraftSubgraph`].
    pub name: String,
    /// The file stem — the slug that identifies this save slot.
    pub slug: String,
    /// Absolute or directory-relative path to the document file.
    pub path: PathBuf,
}

/// Builds the document a save writes: a `Craft`-kind [`SavedDocument`] whose
/// [`CraftSubgraph`] carries `slug` as the stable id and `name` as the display name.
/// Pure (no I/O), so the encode side is testable without a filesystem.
fn craft_document(slug: &str, name: &str, craft: &VoxelCraft) -> SavedDocument {
    SavedDocument::new(Payload::Craft(CraftSubgraph::new(
        slug.to_string(),
        name.to_string(),
        WorldPos::new(FrameId::CENTRAL_BODY, DVec3::ZERO),
        craft.clone(),
    )))
}

/// The full path a craft named `name` saves to within `dir`.
fn craft_path(dir: &Path, slug: &str) -> PathBuf {
    dir.join(format!("{slug}.{CRAFT_EXT}"))
}

/// Saves `craft` under display `name` into `dir`, creating the directory if needed.
/// The file is named by the slug of `name`, so re-saving the same name updates that
/// one slot and never clobbers a craft under a different slug. Returns the written
/// path. An empty name (no slug-worthy characters) falls back to [`FALLBACK_SLUG`];
/// callers that want to forbid blank names should validate before calling.
pub fn save_craft(dir: &Path, name: &str, craft: &VoxelCraft) -> Result<PathBuf, LibraryError> {
    let mut slug = slugify(name);
    if slug.is_empty() {
        slug = FALLBACK_SLUG.to_string();
    }
    std::fs::create_dir_all(dir).map_err(|e| LibraryError::Io(e.to_string()))?;
    let path = craft_path(dir, &slug);
    let json = craft_document(&slug, name, craft).to_json()?;
    std::fs::write(&path, json).map_err(|e| LibraryError::Io(e.to_string()))?;
    Ok(path)
}

/// Reads a saved craft document from `path` and returns its craft. Accepts any
/// craft-scope kind (Craft / Subassembly / Blueprint) so a legacy single-file craft
/// still loads; a non-craft scope (world save or body asset) is rejected as wrong.
pub fn load_craft(path: &Path) -> Result<VoxelCraft, LibraryError> {
    let bytes = std::fs::read_to_string(path).map_err(|e| LibraryError::Io(e.to_string()))?;
    craft_from_document(&bytes)
}

/// Pure decode counterpart to [`load_craft`]: parses a document string and extracts
/// the craft, mapping a world-save to a format error (wrong scope).
pub fn craft_from_document(json: &str) -> Result<VoxelCraft, LibraryError> {
    match SavedDocument::from_json(json)?.payload {
        Payload::Craft(c) | Payload::Subassembly(c) | Payload::Blueprint(c) => Ok(c.craft),
        Payload::WorldSave(_) | Payload::BodyAsset(_) => Err(LibraryError::Format(
            FormatError::Malformed("expected a craft-scope document".to_string()),
        )),
    }
}

/// Enumerates the saved crafts in `dir`, sorted by display name (case-insensitive).
/// A missing directory yields an empty list (it is created on first save). Files that
/// are unreadable, malformed, or not craft-scope are skipped — discovery never aborts
/// on one bad file. Only `*.json` entries are considered.
pub fn list_crafts(dir: &Path) -> Vec<CraftEntry> {
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(), // missing/inaccessible dir => empty library
    };
    let mut entries: Vec<CraftEntry> = Vec::new();
    for dent in read.flatten() {
        let path = dent.path();
        if path.extension().and_then(|e| e.to_str()) != Some(CRAFT_EXT) {
            continue;
        }
        let slug = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        // Read the display name from the document; skip anything that isn't a
        // readable craft-scope save.
        let name = match std::fs::read_to_string(&path) {
            Ok(bytes) => match SavedDocument::from_json(&bytes) {
                Ok(doc) => match doc.payload {
                    Payload::Craft(c) | Payload::Subassembly(c) | Payload::Blueprint(c) => c.name,
                    Payload::WorldSave(_) | Payload::BodyAsset(_) => continue,
                },
                Err(_) => continue,
            },
            Err(_) => continue,
        };
        entries.push(CraftEntry { name, slug, path });
    }
    entries.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.slug.cmp(&b.slug))
    });
    entries
}

/// Whether `name` is acceptable as a craft display name: it must contain at least one
/// non-whitespace character. The naming overlay uses this to reject blank saves.
pub fn is_valid_name(name: &str) -> bool {
    !name.trim().is_empty()
}

/// The kind a saved-craft document declares — exposed so callers can assert scope.
/// Always [`Kind::Craft`] for documents this module writes.
pub const SAVED_KIND: Kind = Kind::Craft;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{AttachmentPoint, Device, DeviceKind, Face, Material, Voxel};
    use glam::IVec3;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A unique scratch directory under the OS temp dir, so tests don't collide and
    /// don't write into the repo. Removed-and-recreated for a clean slate.
    fn scratch_dir(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("snd-lib-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A small craft exercising voxels, devices, and attachments (the round-trip
    /// fidelity invariant).
    fn rich_craft() -> VoxelCraft {
        let mut craft = VoxelCraft::new(0.5);
        craft.voxels.push(Voxel {
            cell: IVec3::new(0, 0, 0),
            material: Material::ALUMINIUM,
        });
        craft.voxels.push(Voxel {
            cell: IVec3::new(1, 0, 0),
            material: Material::ALUMINIUM,
        });
        craft.devices.push(Device::structural(
            IVec3::new(0, 1, 0),
            25.0,
            DeviceKind::Tank,
        ));
        craft.attachments.push(AttachmentPoint {
            cell: IVec3::new(0, 0, 0),
            face: Face::NegY,
        });
        craft
    }

    #[test]
    fn slugify_is_filesystem_safe_and_collapses_separators() {
        assert_eq!(slugify("My Rover"), "my-rover");
        assert_eq!(slugify("  spaced  out  "), "spaced-out");
        assert_eq!(slugify("Rover #3!!!"), "rover-3");
        assert_eq!(slugify("a/b\\c:d"), "a-b-c-d");
        // Distinct names => distinct slugs (no clobber across different slugs).
        assert_ne!(slugify("Scout"), slugify("Hauler"));
    }

    #[test]
    fn slugify_collision_is_intentional_same_slot() {
        // Names that differ only in punctuation/case slug identically (one slot).
        assert_eq!(slugify("My Rover"), slugify("my rover!"));
    }

    #[test]
    fn slugify_empty_for_unsluggable_name() {
        assert_eq!(slugify("!!!"), "");
        assert_eq!(slugify("   "), "");
    }

    #[test]
    fn slugify_is_length_bounded() {
        let slug = slugify(&"a".repeat(500));
        assert!(slug.len() <= MAX_SLUG_LEN);
    }

    #[test]
    fn save_then_load_round_trips_voxels_devices_attachments() {
        let dir = scratch_dir("round-trip");
        let craft = rich_craft();
        let path = save_craft(&dir, "Scout", &craft).unwrap();
        let back = load_craft(&path).unwrap();
        assert_eq!(back.voxels, craft.voxels);
        assert_eq!(back.devices, craft.devices);
        assert_eq!(back.attachments, craft.attachments);
        assert_eq!(back.cell_size, craft.cell_size);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_different_names_do_not_clobber() {
        let dir = scratch_dir("no-clobber");
        let mut scout = rich_craft();
        scout.voxels.truncate(1);
        let hauler = rich_craft(); // 2 voxels
        save_craft(&dir, "Scout", &scout).unwrap();
        save_craft(&dir, "Hauler", &hauler).unwrap();

        let list = list_crafts(&dir);
        assert_eq!(list.len(), 2, "two distinct slugs => two files");
        // Each loads back to its own build.
        let by_name = |n: &str| list.iter().find(|e| e.name == n).unwrap().path.clone();
        assert_eq!(load_craft(&by_name("Scout")).unwrap().voxels.len(), 1);
        assert_eq!(load_craft(&by_name("Hauler")).unwrap().voxels.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resaving_same_name_updates_one_slot() {
        let dir = scratch_dir("update");
        let mut craft = rich_craft();
        save_craft(&dir, "Scout", &craft).unwrap();
        craft.voxels.push(Voxel {
            cell: IVec3::new(2, 0, 0),
            material: Material::ALUMINIUM,
        });
        save_craft(&dir, "Scout", &craft).unwrap();
        let list = list_crafts(&dir);
        assert_eq!(
            list.len(),
            1,
            "same slug => same slot (update, not clobber)"
        );
        assert_eq!(load_craft(&list[0].path).unwrap().voxels.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_is_sorted_by_name_and_reports_display_name() {
        let dir = scratch_dir("list-sort");
        save_craft(&dir, "Zephyr", &rich_craft()).unwrap();
        save_craft(&dir, "alpha", &rich_craft()).unwrap();
        let list = list_crafts(&dir);
        let names: Vec<&str> = list.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "Zephyr"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_directory_lists_empty() {
        let dir = scratch_dir("missing").join("does-not-exist");
        assert!(list_crafts(&dir).is_empty());
    }

    #[test]
    fn corrupt_and_foreign_files_are_skipped_not_fatal() {
        let dir = scratch_dir("corrupt");
        save_craft(&dir, "Good", &rich_craft()).unwrap();
        std::fs::write(dir.join("garbage.json"), "not json {").unwrap();
        std::fs::write(dir.join("note.txt"), "ignored, wrong ext").unwrap();
        // A valid world-save document is the wrong scope and must be skipped.
        let ws = SavedDocument::new(Payload::WorldSave(Default::default()))
            .to_json()
            .unwrap();
        std::fs::write(dir.join("world.json"), ws).unwrap();

        let list = list_crafts(&dir);
        assert_eq!(list.len(), 1, "only the one good craft is listed");
        assert_eq!(list[0].name, "Good");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unsluggable_name_falls_back_and_still_saves() {
        let dir = scratch_dir("fallback");
        let path = save_craft(&dir, "!!!", &rich_craft()).unwrap();
        assert_eq!(path.file_stem().unwrap().to_str().unwrap(), FALLBACK_SLUG);
        assert!(load_craft(&path).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn world_save_document_rejected_by_craft_loader() {
        let ws = SavedDocument::new(Payload::WorldSave(Default::default()))
            .to_json()
            .unwrap();
        assert!(matches!(
            craft_from_document(&ws),
            Err(LibraryError::Format(_))
        ));
    }

    #[test]
    fn name_validation_rejects_blank() {
        assert!(!is_valid_name(""));
        assert!(!is_valid_name("   "));
        assert!(is_valid_name("Scout"));
    }
}

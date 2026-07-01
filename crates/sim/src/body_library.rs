//! Celestial body-asset library (WI 760).
//!
//! A directory of **named** body assets, so the world-builder can generate and
//! keep many planets/moons. It is the celestial parallel of the craft library
//! ([`crate::library`]) and deliberately reuses its filesystem-safe
//! [`slugify`](crate::library::slugify) and [`LibraryError`](crate::library::LibraryError)
//! — same idioms, a different payload scope ([`crate::persist::Payload::BodyAsset`]).
//!
//! The file identity is the slug of the asset's display name; the asset's own
//! fields (including its stable `id`) are written and read back unchanged. The
//! pure pieces (round-trip via the existing versioned format, discovery by
//! scanning) are unit-testable without any rendering.

use crate::body_asset::BodyAsset;
use crate::library::{slugify, LibraryError};
use crate::persist::{FormatError, Payload, SavedDocument};
use std::path::{Path, PathBuf};

/// On-disk extension for a saved body-asset document.
const BODY_EXT: &str = "json";
/// Slug used when a name reduces to nothing slug-worthy (all punctuation/empty).
const FALLBACK_SLUG: &str = "body";

/// A saved body asset discovered in the library directory: its display name (from
/// the document), its slug (the file stem / identity), and the file path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyEntry {
    /// Human-facing name read from the document's [`BodyAsset`].
    pub name: String,
    /// The file stem — the slug that identifies this save slot.
    pub slug: String,
    /// Path to the document file.
    pub path: PathBuf,
}

/// The full path a body asset slugged `slug` saves to within `dir`.
fn body_path(dir: &Path, slug: &str) -> PathBuf {
    dir.join(format!("{slug}.{BODY_EXT}"))
}

/// Saves `asset` into `dir`, creating the directory if needed. The file is named by
/// the slug of the asset's display name, so re-saving an asset with the same name
/// updates that one slot and never clobbers an asset under a different slug. The
/// asset is written unchanged (its own `id` is preserved). Returns the written path.
/// A name with no slug-worthy characters falls back to [`FALLBACK_SLUG`].
pub fn save_body(dir: &Path, asset: &BodyAsset) -> Result<PathBuf, LibraryError> {
    let mut slug = slugify(&asset.name);
    if slug.is_empty() {
        slug = FALLBACK_SLUG.to_string();
    }
    std::fs::create_dir_all(dir).map_err(|e| LibraryError::Io(e.to_string()))?;
    let path = body_path(dir, &slug);
    let json = SavedDocument::new(Payload::BodyAsset(asset.clone())).to_json()?;
    std::fs::write(&path, json).map_err(|e| LibraryError::Io(e.to_string()))?;
    Ok(path)
}

/// Reads a saved body-asset document from `path` and returns its asset.
pub fn load_body(path: &Path) -> Result<BodyAsset, LibraryError> {
    let bytes = std::fs::read_to_string(path).map_err(|e| LibraryError::Io(e.to_string()))?;
    body_from_document(&bytes)
}

/// Pure decode counterpart to [`load_body`]: parses a document string and extracts
/// the body asset, mapping any other scope (craft/world) to a format error.
pub fn body_from_document(json: &str) -> Result<BodyAsset, LibraryError> {
    match SavedDocument::from_json(json)?.payload {
        Payload::BodyAsset(a) => Ok(a),
        _ => Err(LibraryError::Format(FormatError::Malformed(
            "expected a body-asset document, found another scope".to_string(),
        ))),
    }
}

/// Enumerates the saved body assets in `dir`, sorted by display name
/// (case-insensitive). A missing directory yields an empty list. Files that are
/// unreadable, malformed, or not body-asset scope are skipped — discovery never
/// aborts on one bad file. Only `*.json` entries are considered.
pub fn list_bodies(dir: &Path) -> Vec<BodyEntry> {
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut entries: Vec<BodyEntry> = Vec::new();
    for dent in read.flatten() {
        let path = dent.path();
        if path.extension().and_then(|e| e.to_str()) != Some(BODY_EXT) {
            continue;
        }
        let slug = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let name = match std::fs::read_to_string(&path) {
            Ok(bytes) => match SavedDocument::from_json(&bytes) {
                Ok(doc) => match doc.payload {
                    Payload::BodyAsset(a) => a.name,
                    _ => continue, // wrong scope (craft/world) — skip
                },
                Err(_) => continue,
            },
            Err(_) => continue,
        };
        entries.push(BodyEntry { name, slug, path });
    }
    entries.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.slug.cmp(&b.slug))
    });
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A unique scratch directory under the OS temp dir, so tests don't collide and
    /// don't write into the repo.
    fn scratch_dir(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("snd-body-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn named(name: &str, seed: u64) -> BodyAsset {
        let mut a = BodyAsset::earthlike();
        a.name = name.to_string();
        a.surface.seed = seed;
        a
    }

    #[test]
    fn save_then_load_round_trips_the_asset() {
        let dir = scratch_dir("round-trip");
        let asset = named("Mun", 7);
        let path = save_body(&dir, &asset).unwrap();
        let back = load_body(&path).unwrap();
        assert_eq!(back.name, "Mun");
        assert_eq!(back.surface.seed, 7);
        assert_eq!(back.central_body().radius, asset.central_body().radius);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_names_do_not_clobber_and_resave_updates_one_slot() {
        let dir = scratch_dir("slots");
        save_body(&dir, &named("Mun", 1)).unwrap();
        save_body(&dir, &named("Minmus", 2)).unwrap();
        assert_eq!(
            list_bodies(&dir).len(),
            2,
            "two distinct slugs => two files"
        );
        // Re-save "Mun" with a new seed: same slug, one slot, updated content.
        save_body(&dir, &named("Mun", 99)).unwrap();
        let list = list_bodies(&dir);
        assert_eq!(list.len(), 2, "resave updates, not clobbers");
        let mun = list.iter().find(|e| e.name == "Mun").unwrap();
        assert_eq!(load_body(&mun.path).unwrap().surface.seed, 99);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_is_sorted_by_name() {
        let dir = scratch_dir("sort");
        save_body(&dir, &named("Zelda", 1)).unwrap();
        save_body(&dir, &named("alpha", 2)).unwrap();
        let names: Vec<String> = list_bodies(&dir).into_iter().map(|e| e.name).collect();
        assert_eq!(names, vec!["alpha".to_string(), "Zelda".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_directory_lists_empty() {
        let dir = scratch_dir("missing").join("nope");
        assert!(list_bodies(&dir).is_empty());
    }

    #[test]
    fn wrong_scope_and_corrupt_files_are_skipped_not_fatal() {
        let dir = scratch_dir("skip");
        save_body(&dir, &named("Good", 1)).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("garbage.json"), "not json {").unwrap();
        // A valid craft/world document is the wrong scope and must be skipped.
        let ws = SavedDocument::new(Payload::WorldSave(Default::default()))
            .to_json()
            .unwrap();
        std::fs::write(dir.join("world.json"), ws).unwrap();
        let list = list_bodies(&dir);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "Good");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrong_scope_document_rejected_by_body_loader() {
        let ws = SavedDocument::new(Payload::WorldSave(Default::default()))
            .to_json()
            .unwrap();
        assert!(matches!(
            body_from_document(&ws),
            Err(LibraryError::Format(_))
        ));
    }

    #[test]
    fn unsluggable_name_falls_back_and_still_saves() {
        let dir = scratch_dir("fallback");
        let path = save_body(&dir, &named("!!!", 5)).unwrap();
        assert_eq!(path.file_stem().unwrap().to_str().unwrap(), FALLBACK_SLUG);
        assert!(load_body(&path).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

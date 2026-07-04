//! Star-system library (WI 550).
//!
//! A directory of **named** saved systems, completing the world-building
//! persistence trio: crafts ([`crate::library`]), bodies
//! ([`crate::body_library`]), and now systems. The persist payload has existed
//! since WI 761 ([`crate::persist::Payload::System`]); this module makes a
//! saved system **addressable by slug**, which is what a scenario document
//! needs to reference its world (content composes world data by reference —
//! world-building owns it).
//!
//! Deliberately the same idioms as [`crate::body_library`]: slug-of-name file
//! identity, versioned-envelope round-trip, skip-don't-abort discovery.

use crate::library::{slugify, LibraryError};
use crate::persist::{FormatError, Payload, SavedDocument};
use crate::system::System;
use std::path::{Path, PathBuf};

/// On-disk extension for a saved system document.
const SYSTEM_EXT: &str = "json";
/// Slug used when a name reduces to nothing slug-worthy (all punctuation/empty).
const FALLBACK_SLUG: &str = "system";

/// A saved system discovered in the library directory: its display name (from
/// the document), its slug (the file stem / identity), and the file path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemEntry {
    /// Human-facing name read from the document's [`System`].
    pub name: String,
    /// The file stem — the slug that identifies this save slot.
    pub slug: String,
    /// Path to the document file.
    pub path: PathBuf,
}

/// The full path a system slugged `slug` saves to within `dir`.
pub fn system_path(dir: &Path, slug: &str) -> PathBuf {
    dir.join(format!("{slug}.{SYSTEM_EXT}"))
}

/// Saves `system` into `dir`, creating the directory if needed. The file is
/// named by the slug of the system's display name (same-slug re-save updates
/// that one slot). Returns the written path.
pub fn save_system(dir: &Path, system: &System) -> Result<PathBuf, LibraryError> {
    let mut slug = slugify(&system.name);
    if slug.is_empty() {
        slug = FALLBACK_SLUG.to_string();
    }
    std::fs::create_dir_all(dir).map_err(|e| LibraryError::Io(e.to_string()))?;
    let path = system_path(dir, &slug);
    let json = SavedDocument::new(Payload::System(system.clone())).to_json()?;
    std::fs::write(&path, json).map_err(|e| LibraryError::Io(e.to_string()))?;
    Ok(path)
}

/// Reads a saved system document from `path` and returns its system.
pub fn load_system(path: &Path) -> Result<System, LibraryError> {
    let bytes = std::fs::read_to_string(path).map_err(|e| LibraryError::Io(e.to_string()))?;
    system_from_document(&bytes)
}

/// Pure decode counterpart to [`load_system`]: parses a document string and
/// extracts the system, mapping any other scope (craft/world/body) to a
/// format error.
pub fn system_from_document(json: &str) -> Result<System, LibraryError> {
    match SavedDocument::from_json(json)?.payload {
        Payload::System(s) => Ok(s),
        _ => Err(LibraryError::Format(FormatError::Malformed(
            "expected a system document, found another scope".to_string(),
        ))),
    }
}

/// Enumerates the saved systems in `dir`, sorted by display name
/// (case-insensitive). A missing directory yields an empty list. Files that
/// are unreadable, malformed, or not system scope are skipped — discovery
/// never aborts on one bad file. Only `*.json` entries are considered.
pub fn list_systems(dir: &Path) -> Vec<SystemEntry> {
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut entries: Vec<SystemEntry> = Vec::new();
    for dent in read.flatten() {
        let path = dent.path();
        if path.extension().and_then(|e| e.to_str()) != Some(SYSTEM_EXT) {
            continue;
        }
        let slug = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let name = match std::fs::read_to_string(&path) {
            Ok(bytes) => match SavedDocument::from_json(&bytes) {
                Ok(doc) => match doc.payload {
                    Payload::System(s) => s.name,
                    _ => continue, // wrong scope — skip
                },
                Err(_) => continue,
            },
            Err(_) => continue,
        };
        entries.push(SystemEntry { name, slug, path });
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

    /// A unique scratch directory under the OS temp dir, so tests don't collide
    /// and don't write into the repo.
    fn scratch_dir(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("snd-sys-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn named(name: &str, asset_id: &str) -> System {
        System::single_body(name.to_lowercase(), name, asset_id)
    }

    #[test]
    fn save_then_load_round_trips_the_system() {
        let dir = scratch_dir("round-trip");
        let sys = named("Home", "earthlike");
        let path = save_system(&dir, &sys).unwrap();
        let back = load_system(&path).unwrap();
        assert_eq!(back, sys);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resave_updates_one_slot_and_list_is_sorted() {
        let dir = scratch_dir("slots");
        save_system(&dir, &named("Zeta", "a")).unwrap();
        save_system(&dir, &named("Alpha", "b")).unwrap();
        save_system(&dir, &named("Zeta", "c")).unwrap(); // update, not clobber
        let list = list_systems(&dir);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].name, "Alpha");
        let zeta = list.iter().find(|e| e.name == "Zeta").unwrap();
        assert_eq!(load_system(&zeta.path).unwrap().bodies[0].asset_id, "c");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrong_scope_and_corrupt_files_are_skipped_not_fatal() {
        let dir = scratch_dir("skip");
        save_system(&dir, &named("Good", "a")).unwrap();
        std::fs::write(dir.join("garbage.json"), "not json {").unwrap();
        let ws = SavedDocument::new(Payload::WorldSave(Default::default()))
            .to_json()
            .unwrap();
        std::fs::write(dir.join("world.json"), ws).unwrap();
        let list = list_systems(&dir);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "Good");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrong_scope_document_rejected_by_system_loader() {
        let ws = SavedDocument::new(Payload::WorldSave(Default::default()))
            .to_json()
            .unwrap();
        assert!(matches!(
            system_from_document(&ws),
            Err(LibraryError::Format(_))
        ));
    }

    #[test]
    fn missing_directory_lists_empty() {
        let dir = scratch_dir("missing").join("nope");
        assert!(list_systems(&dir).is_empty());
    }
}

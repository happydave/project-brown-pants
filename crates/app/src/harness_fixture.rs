//! Loadable harness fixtures (WI 843).
//!
//! The crash/collide harnesses historically built their projectiles and targets from
//! hard-coded procedural cube lattices, so no shaped craft could reach the WI 837
//! form-aware contact path on screen. This module is the shared seam that fixes that:
//! it resolves an optional per-scene craft argument (`-- crash [projectile] [target]`,
//! the dive/harbor `nth(2)` argument pattern) into a named [`VoxelCraft`] loaded from
//! the craft library, with the procedural fixture as the default when the slot is
//! absent or `-`.
//!
//! Resolution order per argument: the literal path as given, then
//! `saves/crafts/<slug>.json` (editor saves), then `content/blueprints/<slug>.json`
//! (shipped blueprints). An argument that resolves to nothing — or to a file that
//! fails to parse, or to an empty craft — fails fast at startup with a message naming
//! the attempts, matching the scenario scenes' loader behavior.

use sounding_sim::library::{self, slugify};
use sounding_sim::voxel::VoxelCraft;
use std::path::{Path, PathBuf};

/// The default fixture directories searched after the literal path, in order.
const FIXTURE_DIRS: [&str; 2] = ["saves/crafts", "content/blueprints"];

/// The HUD/log name for a slot running its procedural default.
pub const PROCEDURAL: &str = "procedural";

/// A resolved harness fixture: the display name (the resolved file's stem) and the
/// loaded craft. The craft is guaranteed non-empty (it has mass properties).
#[derive(Debug)]
pub struct Fixture {
    pub name: String,
    pub craft: VoxelCraft,
}

/// Resolves `arg` against the literal path then `dirs` (as `<dir>/<slug>.json`).
/// The first candidate that **exists** decides: a parse/validation failure there is
/// surfaced, not skipped — a present-but-broken file should be fixed, not shadowed.
fn resolve_in(arg: &str, dirs: &[&Path]) -> Result<Fixture, String> {
    let mut candidates: Vec<PathBuf> = vec![PathBuf::from(arg)];
    let slug = slugify(arg);
    for dir in dirs {
        candidates.push(dir.join(format!("{slug}.json")));
    }
    let Some(path) = candidates.iter().find(|p| p.is_file()) else {
        let tried: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
        return Err(format!("not found (tried: {})", tried.join(", ")));
    };
    let craft = library::load_craft(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if craft.mass_properties().is_none() {
        return Err(format!("{}: craft is empty", path.display()));
    }
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| arg.to_string());
    Ok(Fixture { name, craft })
}

/// Reads the fixture argument at CLI `position` (2 = first slot after the scene
/// name). `None` — the slot's procedural default — when the argument is absent or
/// the `-` placeholder; panics (startup fail-fast) when a named craft cannot be
/// resolved or loaded.
pub fn fixture_arg(position: usize) -> Option<Fixture> {
    let arg = std::env::args().nth(position)?;
    if arg == "-" {
        return None;
    }
    let dirs: Vec<&Path> = FIXTURE_DIRS.iter().map(Path::new).collect();
    match resolve_in(&arg, &dirs) {
        Ok(f) => Some(f),
        Err(e) => panic!("craft fixture `{arg}`: {e}"),
    }
}

/// The slot's display name for HUD lines and logs.
pub fn slot_name(slot: &Option<Fixture>) -> &str {
    slot.as_ref().map(|f| f.name.as_str()).unwrap_or(PROCEDURAL)
}

/// Rest height of a craft's center of mass above a supporting plane when the craft
/// sits axis-aligned with its AABB floor on the plane: `com.y − aabb_min.y`. The
/// harnesses' historic hard-coded spawn heights are fixed points of this formula
/// (frangible bar 0.5, 3³ block 1.5, 2³ cube 1.0 — pinned in tests).
pub fn rest_height(craft: &VoxelCraft) -> f64 {
    let mp = craft
        .mass_properties()
        .expect("fixture crafts are non-empty");
    let b = sounding_sim::collision::craft_bounds(craft).expect("fixture crafts have bounds");
    mp.center_of_mass.y - b.aabb_min.y
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::{DVec3, IVec3};
    use sounding_sim::collision::{craft_collision_shape, CollisionShape};
    use sounding_sim::shape::{hull_vertices, FillMode, Form, ShapedCell};
    use sounding_sim::voxel::{Material, Thermal, Voxel};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// The crash harness's frangible range (its bar material): weak enough that an
    /// achievable ram speed shatters it.
    const FRANGIBLE: Material = Material {
        density: 2700.0,
        strength: 2.0e6,
        thermal: Thermal::INERT,
    };

    /// The wedge orientation whose solid is `x + y ≤ 1` in the unit cell: full
    /// attachment face at `x = 0`, ramp descending to the bottom edge at `x = 1` —
    /// a nose pointing +x. Derived from the hull catalog rather than hand-picked
    /// from the 24-entry rotation table.
    fn nose_orientation() -> u8 {
        let want = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 1.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 1.0),
        ];
        (0..24u8)
            .find(|&o| {
                let hv = hull_vertices(Form::Wedge, o);
                hv.len() == want.len()
                    && want
                        .iter()
                        .all(|w| hv.iter().any(|v| (*v - *w).length() < 1e-9))
            })
            .expect("some orientation gives the +x nose wedge")
    }

    /// The shipped shaped demo fixture (WI 843): a 6-cell frangible bar whose nose
    /// cell is a +x-pointing wedge — one command closes the WI 837 harness gate
    /// (`-- crash wedge-dart`).
    fn wedge_dart() -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        for x in 0..6 {
            c.voxels.push(Voxel {
                cell: IVec3::new(x, 0, 0),
                material: FRANGIBLE,
            });
        }
        c.set_shape(ShapedCell {
            cell: IVec3::new(5, 0, 0),
            form: Form::Wedge,
            orientation: nose_orientation(),
            fill: FillMode::Solid,
        });
        c
    }

    fn shipped_blueprints_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/blueprints")
    }

    /// Regenerates the shipped wedge-dart blueprint (the WI 820 writer pattern —
    /// run explicitly with `--ignored`; the diff must stay line-reviewable).
    #[test]
    #[ignore]
    fn write_wedge_dart_blueprint() {
        let path = library::save_blueprint(&shipped_blueprints_dir(), "wedge dart", &wedge_dart())
            .unwrap();
        println!("wrote {}", path.display());
    }

    #[test]
    fn the_shipped_wedge_dart_matches_the_fixture_and_resolves_shaped() {
        let dir = shipped_blueprints_dir();
        let f = resolve_in("wedge dart", &[dir.as_path()]).unwrap();
        assert_eq!(f.name, "wedge-dart");
        // Fixture-equality pin (WI 820 discipline): the shipped document is exactly
        // the writer's craft, so fixture drift cannot go unnoticed.
        assert_eq!(f.craft, wedge_dart());
        // And it is genuinely shaped — loading it puts a harness on the WI 837
        // mixed-compound path.
        assert!(matches!(
            craft_collision_shape(&f.craft),
            CollisionShape::Compound { .. }
        ));
    }

    /// A unique scratch directory under the OS temp dir (the library.rs pattern).
    fn scratch(tag: &str) -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("snd-fixture-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn bar(n: i32) -> VoxelCraft {
        let mut c = VoxelCraft::new(1.0);
        for x in 0..n {
            c.voxels.push(Voxel {
                cell: IVec3::new(x, 0, 0),
                material: Material::ALUMINIUM,
            });
        }
        c
    }

    #[test]
    fn resolution_walks_literal_then_saves_then_blueprints() {
        let saves = scratch("saves");
        let blueprints = scratch("blueprints");
        let craft = bar(3);
        library::save_craft(&saves, "in saves", &craft).unwrap();
        library::save_blueprint(&blueprints, "in blueprints", &craft).unwrap();
        let literal = library::save_craft(&scratch("lit"), "by path", &craft).unwrap();
        let dirs = [saves.as_path(), blueprints.as_path()];

        let by_path = resolve_in(literal.to_str().unwrap(), &dirs).unwrap();
        assert_eq!(by_path.name, "by-path");
        assert_eq!(resolve_in("in saves", &dirs).unwrap().name, "in-saves");
        assert_eq!(
            resolve_in("in blueprints", &dirs).unwrap().name,
            "in-blueprints"
        );

        let miss = resolve_in("no such craft", &dirs).unwrap_err();
        assert!(miss.contains("not found") && miss.contains("no-such-craft.json"));
    }

    #[test]
    fn a_present_but_broken_or_empty_file_is_an_error_not_a_fallback() {
        let saves = scratch("bad");
        std::fs::write(saves.join("broken.json"), "{ not json").unwrap();
        library::save_craft(&saves, "empty", &VoxelCraft::new(1.0)).unwrap();
        let dirs = [saves.as_path()];
        assert!(resolve_in("broken", &dirs).is_err());
        assert!(resolve_in("empty", &dirs).unwrap_err().contains("empty"));
    }

    #[test]
    fn rest_height_reproduces_the_historic_spawn_constants() {
        // The hard-coded heights this formula replaced: crash bar 0.5, crash 3³
        // target 1.5, collide 2³ cube 1.0 (WI 843 fixed-point property).
        let mut block3 = VoxelCraft::new(1.0);
        let mut cube2 = VoxelCraft::new(1.0);
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    block3.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::STEEL,
                    });
                }
            }
        }
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    cube2.voxels.push(Voxel {
                        cell: IVec3::new(x, y, z),
                        material: Material::COMPOSITE,
                    });
                }
            }
        }
        assert!((rest_height(&bar(6)) - 0.5).abs() < 1e-12);
        assert!((rest_height(&block3) - 1.5).abs() < 1e-12);
        assert!((rest_height(&cube2) - 1.0).abs() < 1e-12);
        // A shaped craft's rest height follows its (shape-aware) mass properties:
        // a lone wedge's CoM sits below the solid cube's 0.5.
        let mut wedge = VoxelCraft::new(1.0);
        wedge.voxels.push(Voxel {
            cell: IVec3::ZERO,
            material: Material::ALUMINIUM,
        });
        wedge.set_shape(ShapedCell {
            cell: IVec3::ZERO,
            form: Form::Wedge,
            orientation: 0,
            fill: FillMode::Solid,
        });
        assert!(rest_height(&wedge) < 0.5);
    }

    #[test]
    fn slot_names_fall_back_to_procedural() {
        assert_eq!(slot_name(&None), PROCEDURAL);
        let f = Fixture {
            name: "wedge-dart".into(),
            craft: bar(1),
        };
        assert_eq!(slot_name(&Some(f)), "wedge-dart");
    }
}

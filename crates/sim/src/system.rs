//! Star-system composition (WI 761).
//!
//! A [`System`] is the **data** description of a world: a named set of
//! [`BodyAsset`](crate::body_asset::BodyAsset) *references* (by id), each given a
//! **placement** (root, or orbiting a parent on a conic). It compiles into the
//! runtime [`Universe`](crate::universe::Universe) body tree.
//!
//! This realizes the asset ⊕ placement split: the *asset* is the reusable,
//! intrinsic body (WI 760); the *placement* is where it sits in a particular
//! system. The same asset dropped into two systems orbits differently, and a
//! world becomes pure data ("scenario is pure data") rather than inline code.
//!
//! [`System::compile`] resolves each entry's asset (for `mu`/`radius`), wires the
//! parent/orbit tree, and lets [`Body::child`](crate::universe::Body::child)
//! derive each sphere-of-influence by the existing patched-conic formula. It
//! **validates** the system first (exactly one root, no duplicate/unknown frames,
//! no unknown assets, every body reachable from the root — so a cyclic parent
//! graph cannot make the compiled `Universe` recurse forever). Headless, pure.

use crate::body_asset::BodyAsset;
use crate::frame::FrameId;
use crate::orbit::Orbit;
use crate::universe::{Body, Universe};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Where a body sits in a system: at the root, or orbiting a parent body.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    /// The system root — no parent, infinite sphere of influence.
    Root,
    /// Orbiting `parent` on `orbit` (a conic expressed in the parent's frame).
    Orbiting {
        /// The parent body's frame.
        parent: FrameId,
        /// The conic about the parent.
        orbit: Orbit,
    },
}

/// One body in a system: a frame identity, the asset it is an instance of (by id),
/// and its placement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemBody {
    /// This body's frame in the compiled universe.
    pub frame: FrameId,
    /// The id of the [`BodyAsset`] this body is an instance of.
    pub asset_id: String,
    /// Where the body sits (root or orbiting).
    pub placement: Placement,
}

/// A star system: a named set of body placements that compiles to a [`Universe`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct System {
    /// Stable identifier.
    pub id: String,
    /// Human-facing display name.
    pub name: String,
    /// The bodies and their placements. Exactly one must be a [`Placement::Root`].
    pub bodies: Vec<SystemBody>,
}

/// Why a [`System`] failed to compile. Typed and non-panicking, so foreign or
/// malformed system data is rejected cleanly at the boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// A body references an asset id absent from the provided assets.
    UnknownAsset(String),
    /// An orbiting body names a parent frame that is not a body in the system.
    UnknownParent(FrameId),
    /// Two bodies share the same frame.
    DuplicateFrame(FrameId),
    /// No body is the root.
    NoRoot,
    /// More than one body is the root.
    MultipleRoots,
    /// A body cannot reach the root by following parents (a disconnected body or a
    /// parent cycle).
    Unreachable(FrameId),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnknownAsset(id) => write!(f, "unknown asset id: {id}"),
            CompileError::UnknownParent(FrameId(n)) => write!(f, "unknown parent frame: {n}"),
            CompileError::DuplicateFrame(FrameId(n)) => write!(f, "duplicate frame: {n}"),
            CompileError::NoRoot => write!(f, "system has no root body"),
            CompileError::MultipleRoots => write!(f, "system has more than one root body"),
            CompileError::Unreachable(FrameId(n)) => {
                write!(f, "body {n} cannot reach the root (disconnected or cyclic)")
            }
        }
    }
}

impl std::error::Error for CompileError {}

impl System {
    /// A one-body system whose single root references `asset_id`, placed at
    /// [`FrameId::CENTRAL_BODY`] — the data expression of today's implicit single
    /// central body, and the seed a scene or the generator (WI 762) starts from.
    pub fn single_body(
        id: impl Into<String>,
        name: impl Into<String>,
        asset_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            bodies: vec![SystemBody {
                frame: FrameId::CENTRAL_BODY,
                asset_id: asset_id.into(),
                placement: Placement::Root,
            }],
        }
    }

    /// Compiles this system into a [`Universe`], resolving each body's asset from
    /// `assets` (by id) for its `mu`/`radius`. Validates the system first; a
    /// [`CompileError`] is returned (never a panic or a non-terminating tree) for
    /// an unknown asset/parent, a duplicate frame, not-exactly-one root, or a body
    /// unreachable from the root.
    pub fn compile(&self, assets: &[BodyAsset]) -> Result<Universe, CompileError> {
        // Index bodies by frame, rejecting duplicates.
        let mut by_frame: HashMap<FrameId, &SystemBody> = HashMap::with_capacity(self.bodies.len());
        for b in &self.bodies {
            if by_frame.insert(b.frame, b).is_some() {
                return Err(CompileError::DuplicateFrame(b.frame));
            }
        }

        // Exactly one root.
        let roots = self
            .bodies
            .iter()
            .filter(|b| matches!(b.placement, Placement::Root))
            .count();
        match roots {
            0 => return Err(CompileError::NoRoot),
            1 => {}
            _ => return Err(CompileError::MultipleRoots),
        }

        // Every body reaches the root by following parents within `len` steps
        // (else a parent is unknown, or the chain is cyclic/disconnected). This
        // also validates that each named parent is a real body.
        for b in &self.bodies {
            let mut cur = b;
            for _ in 0..=self.bodies.len() {
                match cur.placement {
                    Placement::Root => break,
                    Placement::Orbiting { parent, .. } => {
                        cur = by_frame
                            .get(&parent)
                            .ok_or(CompileError::UnknownParent(parent))?;
                    }
                }
            }
            // If we never hit a root within the bound, the chain is cyclic.
            if !matches!(cur.placement, Placement::Root) {
                return Err(CompileError::Unreachable(b.frame));
            }
        }

        // Resolve assets and build the tree.
        let mu_of = |asset_id: &str| -> Result<f64, CompileError> {
            find_asset(assets, asset_id)
                .map(|a| a.mu)
                .ok_or_else(|| CompileError::UnknownAsset(asset_id.to_string()))
        };
        let mut universe = Universe::new();
        for b in &self.bodies {
            let asset = find_asset(assets, &b.asset_id)
                .ok_or(CompileError::UnknownAsset(b.asset_id.clone()))?;
            let cb = asset.central_body();
            let body = match b.placement {
                Placement::Root => Body::root(b.frame, cb.mu, cb.radius),
                Placement::Orbiting { parent, orbit } => {
                    // Parent existence is guaranteed by the reachability pass; its
                    // asset supplies the μ the SOI formula needs.
                    let parent_body = by_frame[&parent];
                    let parent_mu = mu_of(&parent_body.asset_id)?;
                    Body::child(b.frame, cb.mu, cb.radius, parent, parent_mu, orbit)
                }
            };
            universe = universe.with_body(body);
        }
        Ok(universe)
    }
}

/// Finds a body asset by its `id` within `assets`.
fn find_asset<'a>(assets: &'a [BodyAsset], id: &str) -> Option<&'a BodyAsset> {
    assets.iter().find(|a| a.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;

    const SUN: FrameId = FrameId(0);
    const PLANET: FrameId = FrameId(1);
    const MOON: FrameId = FrameId(2);

    fn asset(id: &str, mu: f64, radius: f64) -> BodyAsset {
        let mut a = BodyAsset::earthlike();
        a.id = id.to_string();
        a.name = id.to_string();
        a.mu = mu;
        a.radius = radius;
        a
    }

    fn planet_orbit() -> Orbit {
        // Planet about the sun: circular-ish, r = 100, v = sqrt(mu_sun/r).
        Orbit::from_state(
            1000.0,
            DVec2::new(100.0, 0.0),
            DVec2::new(0.0, 3.162_277_66),
            0.0,
        )
        .unwrap()
    }
    fn moon_orbit() -> Orbit {
        // Moon about the planet: r = 10, v = sqrt(mu_planet/r).
        Orbit::from_state(
            1.0,
            DVec2::new(10.0, 0.0),
            DVec2::new(0.0, 0.316_227_766),
            0.0,
        )
        .unwrap()
    }

    fn sun_planet_moon_assets() -> Vec<BodyAsset> {
        vec![
            asset("sun", 1000.0, 50.0),
            asset("planet", 1.0, 5.0),
            asset("moon", 0.01, 1.0),
        ]
    }

    fn sun_planet_moon_system() -> System {
        System {
            id: "sol".to_string(),
            name: "Toy System".to_string(),
            bodies: vec![
                SystemBody {
                    frame: SUN,
                    asset_id: "sun".to_string(),
                    placement: Placement::Root,
                },
                SystemBody {
                    frame: PLANET,
                    asset_id: "planet".to_string(),
                    placement: Placement::Orbiting {
                        parent: SUN,
                        orbit: planet_orbit(),
                    },
                },
                SystemBody {
                    frame: MOON,
                    asset_id: "moon".to_string(),
                    placement: Placement::Orbiting {
                        parent: PLANET,
                        orbit: moon_orbit(),
                    },
                },
            ],
        }
    }

    /// The hand-built equivalent of `sun_planet_moon_system()`.
    fn hand_built() -> Universe {
        Universe::new()
            .with_body(Body::root(SUN, 1000.0, 50.0))
            .with_body(Body::child(PLANET, 1.0, 5.0, SUN, 1000.0, planet_orbit()))
            .with_body(Body::child(MOON, 0.01, 1.0, PLANET, 1.0, moon_orbit()))
    }

    #[test]
    fn compiles_to_the_hand_built_equivalent() {
        let u = sun_planet_moon_system()
            .compile(&sun_planet_moon_assets())
            .unwrap();
        let expect = hand_built();
        for frame in [SUN, PLANET, MOON] {
            let a = u.body(frame).unwrap();
            let b = expect.body(frame).unwrap();
            assert_eq!(a.frame, b.frame);
            assert_eq!(a.mu, b.mu);
            assert_eq!(a.radius, b.radius);
            assert_eq!(a.parent, b.parent);
            assert_eq!(a.orbit, b.orbit);
            // Same formula, same inputs → bit-identical (and INF for the root).
            assert_eq!(a.soi_radius, b.soi_radius);
            // Positions agree at a non-zero time (walks the tree the same way).
            let (pa, va) = u.body_state(frame, 3.0).unwrap();
            let (pb, vb) = expect.body_state(frame, 3.0).unwrap();
            assert!((pa - pb).length() < 1e-9 && (va - vb).length() < 1e-9);
        }
    }

    #[test]
    fn single_body_system_compiles_to_one_root() {
        let sys = System::single_body("earth-only", "Earth Only", "earthlike");
        let u = sys.compile(&[BodyAsset::earthlike()]).unwrap();
        let root = u.body(FrameId::CENTRAL_BODY).unwrap();
        assert_eq!(root.parent, None);
        assert_eq!(root.soi_radius, f64::INFINITY);
        assert_eq!(root.mu, BodyAsset::earthlike().mu);
    }

    #[test]
    fn unknown_asset_is_rejected() {
        let mut assets = sun_planet_moon_assets();
        assets.retain(|a| a.id != "moon"); // moon asset missing
        assert_eq!(
            sun_planet_moon_system().compile(&assets).unwrap_err(),
            CompileError::UnknownAsset("moon".to_string())
        );
    }

    #[test]
    fn unknown_parent_is_rejected() {
        let mut sys = sun_planet_moon_system();
        // Point the moon at a non-existent parent frame.
        sys.bodies[2].placement = Placement::Orbiting {
            parent: FrameId(99),
            orbit: moon_orbit(),
        };
        assert_eq!(
            sys.compile(&sun_planet_moon_assets()).unwrap_err(),
            CompileError::UnknownParent(FrameId(99))
        );
    }

    #[test]
    fn duplicate_frame_is_rejected() {
        let mut sys = sun_planet_moon_system();
        sys.bodies[2].frame = PLANET; // moon reuses the planet's frame
        assert_eq!(
            sys.compile(&sun_planet_moon_assets()).unwrap_err(),
            CompileError::DuplicateFrame(PLANET)
        );
    }

    #[test]
    fn no_root_and_multiple_roots_are_rejected() {
        let mut none = sun_planet_moon_system();
        none.bodies[0].placement = Placement::Orbiting {
            parent: MOON,
            orbit: planet_orbit(),
        };
        assert_eq!(
            none.compile(&sun_planet_moon_assets()).unwrap_err(),
            CompileError::NoRoot
        );

        let mut many = sun_planet_moon_system();
        many.bodies[1].placement = Placement::Root;
        assert_eq!(
            many.compile(&sun_planet_moon_assets()).unwrap_err(),
            CompileError::MultipleRoots
        );
    }

    #[test]
    fn parent_cycle_is_rejected_as_unreachable() {
        // Two non-root bodies pointing at each other, plus a real root, so the
        // root count is 1 but the cycle can't reach it.
        let sys = System {
            id: "cyc".to_string(),
            name: "Cycle".to_string(),
            bodies: vec![
                SystemBody {
                    frame: SUN,
                    asset_id: "sun".to_string(),
                    placement: Placement::Root,
                },
                SystemBody {
                    frame: PLANET,
                    asset_id: "planet".to_string(),
                    placement: Placement::Orbiting {
                        parent: MOON,
                        orbit: moon_orbit(),
                    },
                },
                SystemBody {
                    frame: MOON,
                    asset_id: "moon".to_string(),
                    placement: Placement::Orbiting {
                        parent: PLANET,
                        orbit: moon_orbit(),
                    },
                },
            ],
        };
        assert!(matches!(
            sys.compile(&sun_planet_moon_assets()),
            Err(CompileError::Unreachable(_))
        ));
    }

    #[test]
    fn system_serde_round_trips_within_tolerance() {
        let sys = sun_planet_moon_system();
        let json = serde_json::to_string(&sys).unwrap();
        let back: System = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, sys.id);
        assert_eq!(back.name, sys.name);
        assert_eq!(back.bodies.len(), sys.bodies.len());
        // Both compile to equivalent universes — the meaningful structural equality.
        let a = back.compile(&sun_planet_moon_assets()).unwrap();
        let b = sys.compile(&sun_planet_moon_assets()).unwrap();
        for frame in [SUN, PLANET, MOON] {
            let ba = a.body(frame).unwrap();
            let bb = b.body(frame).unwrap();
            assert_eq!(ba.parent, bb.parent);
            assert_eq!(ba.soi_radius, bb.soi_radius);
        }
    }
}

//! World coordinates and reference-frame handling (WI 497).
//!
//! The authoritative simulation works in **f64 3D world coordinates** expressed
//! relative to a named **body-centered inertial reference frame**. A coordinate
//! is therefore never ambiguous about which body's frame it lives in: it carries
//! a [`FrameId`].
//!
//! Today the universe has a single central body, so exactly one frame exists
//! ([`FrameId::CENTRAL_BODY`]) and the transform from a body-centered frame to the
//! canonical world frame is the identity. [`WorldPos::transform_to`] is the
//! documented seam where multi-body sphere-of-influence transforms (the on-rails
//! ↔ active hand-off, WI 508) and floating-origin rebasing (WI 504) will later
//! compose; it is deliberately minimal now.

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Identity of a body-centered inertial reference frame.
///
/// Each celestial body defines an inertial frame centered on it. With a single
/// central body today, only [`FrameId::CENTRAL_BODY`] exists; additional frames
/// are introduced when multiple bodies and sphere-of-influence transitions arrive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FrameId(pub u32);

impl FrameId {
    /// The single central body's inertial frame, which is also the canonical
    /// world frame today.
    pub const CENTRAL_BODY: FrameId = FrameId(0);
}

/// An f64 3D position in a specific body-centered inertial frame.
///
/// The inner [`DVec3`] is public so full vector math is available; the [`FrameId`]
/// keeps the coordinate unambiguous about its frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldPos {
    /// The frame this position is expressed in.
    pub frame: FrameId,
    /// Position, world units (metres), f64.
    pub pos: DVec3,
}

impl WorldPos {
    /// Constructs a position in `frame`.
    pub fn new(frame: FrameId, pos: DVec3) -> Self {
        Self { frame, pos }
    }

    /// Distance from the frame's central body (the frame origin). Used to derive
    /// altitude/depth for fluid-medium sampling.
    pub fn radius(&self) -> f64 {
        self.pos.length()
    }

    /// Expresses this position in `target`.
    ///
    /// Today only one frame exists, so this returns `Some(self.pos)` when
    /// `target` matches this position's frame (the identity transform) and `None`
    /// otherwise. This is the extension seam: multi-frame transforms and
    /// floating-origin rebasing slot in here without changing call sites.
    pub fn transform_to(&self, target: FrameId) -> Option<DVec3> {
        (target == self.frame).then_some(self.pos)
    }
}

/// An f64 3D velocity in a specific body-centered inertial frame.
///
/// Interpreted in the **same** body-centered frame as the position it accompanies.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldVel {
    /// The frame this velocity is expressed in.
    pub frame: FrameId,
    /// Velocity, world units per second (m/s), f64.
    pub vel: DVec3,
}

impl WorldVel {
    /// Constructs a velocity in `frame`.
    pub fn new(frame: FrameId, vel: DVec3) -> Self {
        Self { frame, vel }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_is_f64_and_frame_tagged() {
        let p = WorldPos::new(FrameId::CENTRAL_BODY, DVec3::new(3.0, 4.0, 0.0));
        assert_eq!(p.frame, FrameId::CENTRAL_BODY);
        assert_eq!(p.radius(), 5.0);
    }

    #[test]
    fn transform_to_same_frame_is_identity() {
        let pos = DVec3::new(1.0, -2.0, 7.5);
        let p = WorldPos::new(FrameId::CENTRAL_BODY, pos);
        assert_eq!(p.transform_to(FrameId::CENTRAL_BODY), Some(pos));
    }

    #[test]
    fn transform_to_other_frame_is_unsupported_today() {
        let p = WorldPos::new(FrameId::CENTRAL_BODY, DVec3::ZERO);
        assert_eq!(p.transform_to(FrameId(1)), None);
    }

    #[test]
    fn velocity_carries_same_frame_kind() {
        let v = WorldVel::new(FrameId::CENTRAL_BODY, DVec3::new(0.0, 100.0, 0.0));
        assert_eq!(v.frame, FrameId::CENTRAL_BODY);
        assert_eq!(v.vel.y, 100.0);
    }

    #[test]
    fn worldpos_serde_round_trips() {
        let p = WorldPos::new(FrameId(2), DVec3::new(1.5, -3.25, 9.0));
        let json = serde_json::to_string(&p).unwrap();
        let back: WorldPos = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn worldvel_serde_round_trips() {
        let v = WorldVel::new(FrameId::CENTRAL_BODY, DVec3::new(-7.0, 0.0, 0.25));
        let json = serde_json::to_string(&v).unwrap();
        let back: WorldVel = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }
}

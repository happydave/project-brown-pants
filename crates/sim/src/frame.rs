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
    /// world frame today (and the root frame of a multi-body universe).
    pub const CENTRAL_BODY: FrameId = FrameId(0);
}

/// Supplies the relative state of one body-centered inertial frame's origin with
/// respect to another at a given time — exactly the data a frame transform needs.
/// Because the frames are inertial (non-rotating), the transform between them is a
/// pure translation by this offset. Implemented by the multi-body universe
/// (`universe::Universe`, WI 528); kept as a trait here so `frame.rs` stays free of
/// an `Orbit`/universe dependency.
pub trait FrameTree {
    /// Position and velocity of `from`'s origin expressed in `reference`'s frame at
    /// time `t`. `Some((ZERO, ZERO))` when `from == reference`; `None` if either
    /// frame is unknown.
    fn relative_state(&self, from: FrameId, reference: FrameId, t: f64) -> Option<(DVec3, DVec3)>;
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

    /// Expresses this position in `target` at time `t`, using `frames` for the
    /// relative state of the two frame origins. Because body-centered frames are
    /// inertial (non-rotating), this is a translation: `pos_target = pos_self +
    /// (origin_self − origin_target)`. Returns the identity for the same frame and
    /// `None` if a frame is unknown to `frames` (WI 528).
    pub fn transform_to(&self, target: FrameId, frames: &impl FrameTree, t: f64) -> Option<DVec3> {
        let (offset, _) = frames.relative_state(self.frame, target, t)?;
        Some(self.pos + offset)
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

    /// Expresses this velocity in `target` at time `t`: `vel_target = vel_self +
    /// (vel_origin_self − vel_origin_target)` (inertial frames, WI 528).
    pub fn transform_to(&self, target: FrameId, frames: &impl FrameTree, t: f64) -> Option<DVec3> {
        let (_, vel_offset) = frames.relative_state(self.frame, target, t)?;
        Some(self.vel + vel_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal two-frame tree for transform tests: frame 1's origin sits at a
    /// fixed offset (with a fixed relative velocity) from the root frame 0.
    struct TwoFrames {
        offset: DVec3,
        vel_offset: DVec3,
    }
    impl FrameTree for TwoFrames {
        fn relative_state(
            &self,
            from: FrameId,
            reference: FrameId,
            _t: f64,
        ) -> Option<(DVec3, DVec3)> {
            match (from, reference) {
                (a, b) if a == b => Some((DVec3::ZERO, DVec3::ZERO)),
                // frame 1 origin relative to frame 0.
                (FrameId(1), FrameId(0)) => Some((self.offset, self.vel_offset)),
                (FrameId(0), FrameId(1)) => Some((-self.offset, -self.vel_offset)),
                _ => None,
            }
        }
    }

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
        let frames = TwoFrames {
            offset: DVec3::new(10.0, 0.0, 0.0),
            vel_offset: DVec3::ZERO,
        };
        assert_eq!(
            p.transform_to(FrameId::CENTRAL_BODY, &frames, 0.0),
            Some(pos)
        );
    }

    #[test]
    fn transform_to_translates_between_frames_and_round_trips() {
        let frames = TwoFrames {
            offset: DVec3::new(10.0, -4.0, 0.0),
            vel_offset: DVec3::new(0.5, 0.0, 0.0),
        };
        // A point at frame-1 origin: in frame 1 it is ZERO, in frame 0 it is offset.
        let p1 = WorldPos::new(FrameId(1), DVec3::ZERO);
        let in0 = p1.transform_to(FrameId(0), &frames, 0.0).unwrap();
        assert_eq!(in0, frames.offset);
        // Round-trip back to frame 1.
        let back = WorldPos::new(FrameId(0), in0)
            .transform_to(FrameId(1), &frames, 0.0)
            .unwrap();
        assert_eq!(back, DVec3::ZERO);
        // Velocity transforms by the relative velocity.
        let v1 = WorldVel::new(FrameId(1), DVec3::new(1.0, 0.0, 0.0));
        assert_eq!(
            v1.transform_to(FrameId(0), &frames, 0.0),
            Some(DVec3::new(1.5, 0.0, 0.0))
        );
    }

    #[test]
    fn transform_to_unknown_frame_is_none() {
        let frames = TwoFrames {
            offset: DVec3::ZERO,
            vel_offset: DVec3::ZERO,
        };
        let p = WorldPos::new(FrameId::CENTRAL_BODY, DVec3::ZERO);
        assert_eq!(p.transform_to(FrameId(9), &frames, 0.0), None);
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

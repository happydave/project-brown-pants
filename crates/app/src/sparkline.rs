//! Reusable HUD sparkline (WI 645): a small bar-graph of a scalar signal's recent history, autoscaled
//! to the window max so a wide-range signal — `contact_jitter` runs 0.006 → 1639 — stays legible at a
//! glance. Rendered with **Bevy UI** bar nodes (no custom mesh or 2D camera). The sampling and
//! normalisation are pure and unit-tested; drawing is a thin set of helpers a scene wires to a value
//! each frame. The debug overlay (WI 646) hosts several of these.

use bevy::prelude::*;
use std::collections::VecDeque;

/// Number of bars (and retained samples) a sparkline shows.
pub const SPARK_BARS: usize = 48;
/// Bar panel height in logical pixels.
pub const PANEL_HEIGHT: f32 = 44.0;
/// Per-bar width in logical pixels.
pub const BAR_WIDTH: f32 = 3.0;
const EPS: f32 = 1e-6;

/// A bounded ring of recent samples with autoscaled bar heights. Pure — no rendering. Autoscaling is
/// **relative** (to the window max), so the *shape* is always visible; the absolute scale is carried
/// by the numeric label a scene draws beside it, so a quiet signal reads honestly.
#[derive(Clone)]
pub struct Sparkline {
    samples: VecDeque<f32>,
    cap: usize,
}

impl Sparkline {
    /// A sparkline retaining the last `cap` samples.
    pub fn new(cap: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(cap),
            cap,
        }
    }

    /// Record one sample (non-negative magnitude; negatives are clamped to 0), dropping the oldest
    /// past capacity.
    pub fn push(&mut self, v: f32) {
        if self.samples.len() == self.cap {
            self.samples.pop_front();
        }
        self.samples
            .push_back(if v.is_finite() { v.max(0.0) } else { 0.0 });
    }

    /// The most recent sample (0 if empty).
    pub fn latest(&self) -> f32 {
        self.samples.back().copied().unwrap_or(0.0)
    }

    /// The largest sample in the window (the autoscale reference; 0 if empty).
    pub fn window_max(&self) -> f32 {
        self.samples.iter().copied().fold(0.0, f32::max)
    }

    /// `cap` normalised heights in `[0, 1]`, oldest→newest, scaled to the window max. Not-yet-filled
    /// leading slots are zero; a near-zero window returns all zeros (no divide-by-tiny blow-up).
    pub fn bars(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.cap];
        let m = self.window_max();
        if m < EPS {
            return out;
        }
        let start = self.cap - self.samples.len();
        for (i, v) in self.samples.iter().enumerate() {
            out[start + i] = (v / m).clamp(0.0, 1.0);
        }
        out
    }
}

/// The bar-row container of one sparkline panel.
#[derive(Component)]
pub struct SparklinePanel;

/// One bar in a sparkline panel, by left→right index.
#[derive(Component)]
pub struct SparkBar(pub usize);

/// The numeric label drawn above a sparkline (current value + window max).
#[derive(Component)]
pub struct SparklineLabel;

/// Spawn a sparkline widget (label + bar row) at an absolute screen position, returning the root
/// entity. `title` names the signal; the label text is updated each frame by the scene. Designed so a
/// scene spawns one (WI 645) and the overlay (WI 646) can spawn several at different anchors.
pub fn spawn_sparkline(commands: &mut Commands, top_px: f32, left_px: f32, title: &str) -> Entity {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(top_px),
                left: Val::Px(left_px),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                ..default()
            },
            // Faint backdrop so bars read over any scene.
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.35)),
        ))
        .with_children(|root| {
            root.spawn((
                Text::new(format!("{title}: --")),
                TextFont {
                    font_size: 13.0,
                    ..default()
                },
                TextColor(Color::srgb(0.8, 0.9, 1.0)),
                SparklineLabel,
            ));
            root.spawn((
                Node {
                    width: Val::Px(SPARK_BARS as f32 * BAR_WIDTH),
                    height: Val::Px(PANEL_HEIGHT),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::FlexEnd, // bars grow up from the baseline
                    column_gap: Val::Px(0.0),
                    ..default()
                },
                SparklinePanel,
            ))
            .with_children(|row| {
                for i in 0..SPARK_BARS {
                    row.spawn((
                        Node {
                            width: Val::Px(BAR_WIDTH - 1.0),
                            height: Val::Px(0.0),
                            margin: UiRect::right(Val::Px(1.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.25, 0.85, 0.95)),
                        SparkBar(i),
                    ));
                }
            });
        })
        .id()
}

/// Apply normalised `bars` to the bar nodes (height = `bars[i] · PANEL_HEIGHT`) and tint the newest
/// bar warmer the closer it is to the window peak, so a fresh spike pops. The caller filters the
/// query to one panel's bars when several exist (WI 646).
pub fn apply_bars(bars: &[f32], query: &mut Query<(&SparkBar, &mut Node, &mut BackgroundColor)>) {
    let cool = Color::srgb(0.25, 0.85, 0.95);
    let hot = Color::srgb(0.95, 0.45, 0.25);
    for (bar, mut node, mut color) in query.iter_mut() {
        let h = bars.get(bar.0).copied().unwrap_or(0.0);
        node.height = Val::Px(h * PANEL_HEIGHT);
        // Mix cool→hot by height so tall (near-peak) bars read warm.
        *color = BackgroundColor(cool.mix(&hot, h));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_nearzero_bars_are_zero() {
        let mut s = Sparkline::new(4);
        assert!(s.bars().iter().all(|&b| b == 0.0));
        for _ in 0..4 {
            s.push(0.0);
        }
        assert!(s.bars().iter().all(|&b| b == 0.0), "flat-zero → no bars");
    }

    #[test]
    fn spike_scales_to_one_and_others_shrink() {
        let mut s = Sparkline::new(4);
        s.push(1.0);
        s.push(2.0);
        s.push(1.0);
        s.push(1000.0); // a big spike
        let bars = s.bars();
        assert_eq!(bars.len(), 4);
        assert!((bars[3] - 1.0).abs() < 1e-6, "the spike is full height");
        assert!(
            bars[0] < 0.01 && bars[1] < 0.01,
            "earlier samples shrink under the spike"
        );
        assert_eq!(s.latest(), 1000.0);
    }

    #[test]
    fn ring_drops_oldest_and_fills_from_the_right() {
        let mut s = Sparkline::new(3);
        s.push(5.0); // will be dropped
        s.push(10.0);
        s.push(20.0);
        s.push(40.0); // pushes out the 5.0
        let bars = s.bars(); // window max 40 → [10/40, 20/40, 40/40]
        assert!((bars[0] - 0.25).abs() < 1e-6);
        assert!((bars[1] - 0.5).abs() < 1e-6);
        assert!((bars[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn partial_fill_pads_leading_slots_with_zero() {
        let mut s = Sparkline::new(4);
        s.push(10.0);
        s.push(20.0);
        let bars = s.bars(); // two leading zeros, then 0.5, 1.0
        assert_eq!(bars[0], 0.0);
        assert_eq!(bars[1], 0.0);
        assert!((bars[2] - 0.5).abs() < 1e-6);
        assert!((bars[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn non_finite_samples_are_ignored_as_zero() {
        let mut s = Sparkline::new(2);
        s.push(f32::NAN);
        s.push(f32::INFINITY);
        assert!(s.bars().iter().all(|&b| b == 0.0));
        assert_eq!(s.window_max(), 0.0);
    }
}

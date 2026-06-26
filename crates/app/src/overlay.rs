//! Debug overlay / signal box (WI 646): a toggleable on-screen panel of autoscaled sparklines plus a
//! header, fed from the live rover telemetry (the `GroundedRover` bridge). The human-facing half of
//! the glass cockpit and the thing the screenshot-over-MCP (WI 647) captures. Reusable across the
//! rover-bearing scenes (`-- rover`, the workshop Test); `G` toggles it.

use crate::bus::GroundedRover;
use crate::sparkline::{apply_panel, spawn_panel, SparkBar, Sparkline, SparklineLabel, SPARK_BARS};
use bevy::prelude::*;
use sounding_sim::telemetry::RoverTelemetry;

/// The scalar signals the rover cockpit plots, in panel order. Adding a signal is a new variant +
/// arm — the box, sampling, and rendering follow automatically.
#[derive(Clone, Copy)]
pub enum RoverSignal {
    ContactJitter,
    Speed,
    AngularSpeed,
    HullPenetration,
}

impl RoverSignal {
    /// The cockpit's signal set, in display order.
    pub const ALL: [RoverSignal; 4] = [
        RoverSignal::ContactJitter,
        RoverSignal::Speed,
        RoverSignal::AngularSpeed,
        RoverSignal::HullPenetration,
    ];

    /// The panel title.
    pub fn label(self) -> &'static str {
        match self {
            RoverSignal::ContactJitter => "contact_jitter",
            RoverSignal::Speed => "speed m/s",
            RoverSignal::AngularSpeed => "ang.vel rad/s",
            RoverSignal::HullPenetration => "hull_pen m",
        }
    }

    /// Sample this signal from a rover snapshot.
    pub fn sample(self, r: &RoverTelemetry) -> f32 {
        let mag = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let x = match self {
            RoverSignal::ContactJitter => r.contact_jitter,
            RoverSignal::Speed => mag(r.velocity),
            RoverSignal::AngularSpeed => mag(r.angular_velocity),
            RoverSignal::HullPenetration => r.hull_penetration,
        };
        x as f32
    }
}

/// The cockpit overlay state: one sample ring per [`RoverSignal`].
#[derive(Resource)]
pub struct CockpitOverlay {
    sparks: Vec<Sparkline>,
}

impl Default for CockpitOverlay {
    fn default() -> Self {
        Self {
            sparks: RoverSignal::ALL
                .iter()
                .map(|_| Sparkline::new(SPARK_BARS))
                .collect(),
        }
    }
}

/// The toggleable overlay container (its visibility is flipped by `G`).
#[derive(Component)]
pub struct OverlayRoot;

/// Spawn the cockpit overlay box at the top-right, one sparkline panel per [`RoverSignal`]. Visible by
/// default; `G` toggles it. Caller may tag the returned entity (e.g. with a scene cleanup marker).
pub fn spawn_overlay(commands: &mut Commands) -> Entity {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                right: Val::Px(12.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                padding: UiRect::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.45)),
            Visibility::Visible,
            OverlayRoot,
        ))
        .with_children(|root| {
            root.spawn((
                Text::new("cockpit (G)"),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgb(0.6, 0.7, 0.8)),
            ));
            for (panel, sig) in RoverSignal::ALL.iter().enumerate() {
                spawn_panel(root, panel, sig.label());
            }
        })
        .id()
}

/// Toggle the overlay on `G`, then sample every signal from the published rover telemetry into its
/// ring and update its panel + label. Shared by the rover and workshop scenes.
pub fn update_overlay(
    keys: Res<ButtonInput<KeyCode>>,
    grounded: Res<GroundedRover>,
    mut overlay: ResMut<CockpitOverlay>,
    mut root: Query<&mut Visibility, With<OverlayRoot>>,
    mut bars: Query<(&SparkBar, &mut Node, &mut BackgroundColor)>,
    mut labels: Query<(&SparklineLabel, &mut Text)>,
) {
    if keys.just_pressed(KeyCode::KeyG) {
        for mut vis in &mut root {
            *vis = match *vis {
                Visibility::Hidden => Visibility::Visible,
                _ => Visibility::Hidden,
            };
        }
    }

    for (panel, sig) in RoverSignal::ALL.iter().enumerate() {
        let v = grounded.0.as_ref().map(|r| sig.sample(r)).unwrap_or(0.0);
        let spark = &mut overlay.sparks[panel];
        spark.push(v);
        apply_panel(panel, &spark.bars(), &mut bars);
        for (label, mut text) in &mut labels {
            if label.panel == panel {
                text.0 = format!(
                    "{}: {:.2} (max {:.1})",
                    sig.label(),
                    spark.latest(),
                    spark.window_max()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> RoverTelemetry {
        RoverTelemetry {
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            velocity: [3.0, 0.0, 4.0], // |v| = 5
            angular_velocity: [0.0, 0.0, 0.0],
            contact_jitter: 12.5,
            hull_penetration: 0.04,
            grounded: true,
            wheels: vec![],
        }
    }

    #[test]
    fn signals_sample_from_a_snapshot() {
        let r = snap();
        assert!((RoverSignal::ContactJitter.sample(&r) - 12.5).abs() < 1e-6);
        assert!((RoverSignal::Speed.sample(&r) - 5.0).abs() < 1e-5);
        assert!((RoverSignal::AngularSpeed.sample(&r)).abs() < 1e-6);
        assert!((RoverSignal::HullPenetration.sample(&r) - 0.04).abs() < 1e-6);
    }

    #[test]
    fn overlay_has_a_ring_per_signal() {
        let o = CockpitOverlay::default();
        assert_eq!(o.sparks.len(), RoverSignal::ALL.len());
    }
}

//! Toy: aero wind tunnel (WI 521). Two live plots from `sounding_sim::aero`:
//! the **lift curve** (Cl vs angle of attack — the 2π thin-airfoil slope and the
//! stall) and the **drag curve** (wave-drag coefficient vs speed — the transonic
//! area-ruling spike). Cycling the medium (`M`) shows the spike in atmosphere and
//! its absence in water/vacuum: one parameterized module.
//!
//! Controls: `M` cycle the medium (air → water → vacuum).

use bevy::prelude::*;
use sounding_sim::aero::{area_ruling_factor, lift_coefficient, mach, wave_drag_coefficient};
use sounding_sim::fluid::{FluidMedium, FluidSample};
use sounding_sim::voxel::{Axis, Material, Voxel, VoxelCraft};

/// The selected ambient medium for the drag plot, and the sample craft's
/// area-ruling factor.
#[derive(Resource)]
struct WindTunnel {
    medium: usize, // 0 air, 1 water, 2 vacuum
    factor: f64,
}

impl WindTunnel {
    fn sample(&self) -> (FluidSample, &'static str) {
        match self.medium {
            1 => (FluidMedium::EARTHLIKE.sample_altitude(-10.0), "water"),
            2 => (FluidMedium::VACUUM.sample_altitude(0.0), "vacuum"),
            _ => (FluidMedium::EARTHLIKE.sample_altitude(0.0), "air"),
        }
    }
}

#[derive(Component)]
struct Hud;

/// The Toy aero wind-tunnel scene.
pub struct WindTunnelScenePlugin;

impl Plugin for WindTunnelScenePlugin {
    fn build(&self, app: &mut App) {
        // A tapered sample craft → its area-ruling factor drives the wave drag.
        let mut craft = VoxelCraft::new(1.0);
        let widths = [1, 2, 3, 4, 3, 2, 1];
        for (x, &w) in widths.iter().enumerate() {
            for y in 0..w {
                for z in 0..w {
                    craft.voxels.push(Voxel {
                        cell: IVec3::new(x as i32, y, z),
                        material: Material::ALUMINIUM,
                    });
                }
            }
        }
        let factor = area_ruling_factor(&craft.area_curve(Axis::X));
        app.insert_resource(WindTunnel { medium: 0, factor })
            .add_systems(Startup, setup_view)
            .add_systems(Update, (cycle_medium, draw_plots, update_hud).chain());
    }
}

fn setup_view(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 22.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(0.9, 0.95, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
        Hud,
    ));
}

fn cycle_medium(keys: Res<ButtonInput<KeyCode>>, mut tunnel: ResMut<WindTunnel>) {
    if keys.just_pressed(KeyCode::KeyM) {
        tunnel.medium = (tunnel.medium + 1) % 3;
    }
}

/// Draws a labelled axis cross at `origin` spanning `w`×`h`.
fn axes(gizmos: &mut Gizmos, origin: Vec2, w: f32, h: f32) {
    let c = Color::srgb(0.4, 0.42, 0.48);
    let o = Vec3::new(origin.x, origin.y, 0.0);
    gizmos.line(o + Vec3::new(0.0, -h, 0.0), o + Vec3::new(0.0, h, 0.0), c); // y axis
    gizmos.line(o + Vec3::new(-w, 0.0, 0.0), o + Vec3::new(w, 0.0, 0.0), c); // x axis
}

fn draw_plots(mut gizmos: Gizmos, tunnel: Res<WindTunnel>) {
    // --- Left: lift curve Cl vs α, α ∈ [-0.6, 0.6] rad ---
    let lo = Vec2::new(-3.5, 0.0);
    axes(&mut gizmos, lo, 2.2, 3.0);
    let cl_curve: Vec<Vec3> = (0..=120)
        .map(|i| {
            let a = -0.6 + 1.2 * (i as f64 / 120.0);
            let cl = lift_coefficient(a);
            Vec3::new(lo.x + (a as f32) * 3.5, (cl as f32) * 1.5, 0.0)
        })
        .collect();
    gizmos.linestrip(cl_curve, Color::srgb(0.45, 0.85, 0.55));

    // --- Right: wave-drag coefficient vs speed (0..1000 m/s) for the medium ---
    let ro = Vec2::new(2.0, 0.0);
    axes(&mut gizmos, ro, 2.5, 3.0);
    let (sample, _) = tunnel.sample();
    let cd_curve: Vec<Vec3> = (0..=200)
        .map(|i| {
            let speed = 1000.0 * (i as f64 / 200.0);
            let cd = mach(&sample, speed)
                .map(|m| wave_drag_coefficient(m, tunnel.factor))
                .unwrap_or(0.0);
            Vec3::new(ro.x + (speed as f32 / 1000.0) * 2.5, (cd as f32) * 2.5, 0.0)
        })
        .collect();
    let color = if tunnel.medium == 0 {
        Color::srgb(0.95, 0.6, 0.3)
    } else {
        Color::srgb(0.5, 0.6, 0.75)
    };
    gizmos.linestrip(cd_curve, color);
}

fn update_hud(tunnel: Res<WindTunnel>, mut hud: Query<&mut Text, With<Hud>>) {
    if let Ok(mut text) = hud.single_mut() {
        let (_, name) = tunnel.sample();
        let note = if tunnel.medium == 0 {
            "transonic area-ruling spike near Mach 1"
        } else {
            "no wave drag (incompressible / vacuum)"
        };
        text.0 = format!(
            "left: lift Cl vs angle of attack (2pi slope + stall)\nright: wave drag vs speed in {name} — {note}\nM: cycle medium (air -> water -> vacuum)"
        );
    }
}

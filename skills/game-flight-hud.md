---
name: game-flight-hud
description: >
  Design real-time flight and vehicle HUDs read under time pressure — the heads-up
  display, reticle, instruments, threat indicators, navigation, and feedback for
  craft the player actively controls. Specializes in UNIFYING multiple flight
  regimes into one coherent experience: underwater, ground/surface, atmospheric
  flight, and space flight, plus the transitions between them. Use whenever the
  user is designing or critiquing a cockpit/HUD, flight controls, a reticle or
  targeting display, vehicle telemetry, or any heads-up readout consumed during
  active play — even if they don't say "HUD." Especially use it when a single
  vehicle moves between environments (sub→land→air→space) and the interface must
  stay legible across all of them. This is distinct from authoring-tool UX; if the
  user is designing a ship/level/base EDITOR or build mode, use game-editor-ux
  instead. If the work spans both, use both.
---

# Flight & Vehicle HUD UX

A HUD is read in fractions of a second while the player's attention is on the world
and their hands are on the controls. It is the opposite discipline from an editor:
optimize for **instant legibility under cognitive load**, not depth or configurability.
Every element competes for a glance the player can't spare. The job is ruthless triage
of what earns space on screen *right now*.

## The unification problem (the hard part)

A craft that moves through water, ground, atmosphere, and space passes through regimes
with genuinely different physics and different things the pilot must know:

- **Underwater** — buoyancy/depth, pressure/crush limits, low visibility, 6-DOF but drag-dominated and slow.
- **Ground/surface** — gravity, terrain, traction, a meaningful "down" and horizon.
- **Atmospheric flight** — lift, stall, airspeed/AoA, altitude AGL, attitude relative to a horizon, weather.
- **Space** — 6-DOF, Newtonian drift, no horizon, no up, thrust/delta-v and relative velocity to targets rather than airspeed.

The trap is bolting four separate HUDs together so the screen reshuffles every time the
player crosses a boundary, forcing them to re-learn where to look mid-action. The goal
is **one mental model that adapts**, not four interfaces in a trench coat.

### Strategies for a unified HUD
- **Stable anchor, adaptive content.** Keep a fixed spatial grammar — speed always
  here, attitude always there, threats always there — and let each *slot* swap what it
  shows per regime (airspeed ↔ relative velocity; altitude AGL ↔ depth ↔ distance-to-
  body). The player learns one layout, not four.
- **Keep magnitudes legible across the scale jump.** The same slot can swing from a few
  m/s underwater to ~200 m/s in atmosphere to several km/s in orbit. Choose units,
  precision, and scale cues (auto-ranging, a magnitude/unit label, a non-linear gauge) so
  the readout stays a glance, not a wall of digits, in every regime.
- **A horizon reference that degrades gracefully.** Atmospheric flight has a true
  horizon; space doesn't. Use one attitude reference that smoothly becomes an
  artificial/orbital reference in space rather than vanishing — continuity beats a hard
  cutover.
- **Make the regime itself legible.** The player should always know, pre-attentively,
  which regime they're in and what's about to change. Telegraph transitions (entering
  atmosphere, breaching the surface, crush-depth approaching) *before* they happen.
- **Design the transitions as first-class moments**, not glitchy seams. The handful of
  seconds crossing water→air or air→space is where a unified HUD proves itself; storyboard
  those explicitly.
- **Cut per-regime chrome to the absolute minimum.** Every regime-specific element you
  add multiplies the relearning cost across boundaries. If it isn't needed *in this
  regime, right now*, it shouldn't be on screen.

## Core principles

### 1. Signal-to-noise is everything
Under load, the player can only process a few things at once. Anything decorative or
rarely-relevant is actively harmful — it raises the cost of finding what matters.

- Establish a strict **visual hierarchy**: the 2–3 things that keep the player alive get
  the most salient position, size, and contrast; everything else recedes or hides until
  relevant.
- **Progressive disclosure in real time**: surface information when it becomes
  actionable (threat indicator on lock, stall warning near stall, depth warning near
  crush) and let it fall away otherwise. A quiet HUD is a feature.

### 2. Pre-attentive feedback for the things that kill you
Critical state must register *before* conscious reading — through motion, color, and
position, not text the player has to parse.

- Use color, blink, growth, and spatial pull for danger (incoming fire, stall, crush
  depth, overheat). Reserve these channels; if everything flashes, nothing does.
- Pair channels for accessibility: never rely on color alone — add shape, motion, or
  sound so colorblind players and noisy scenes are covered.
- **Audio is part of the HUD.** A rising tone or directional warning offloads the eyes,
  which are busy flying. Treat sound as a real information channel, not garnish.

### 3. Diegetic vs. non-diegetic is a deliberate choice
Decide, per element, whether information lives *in the fiction* (cockpit glass,
projected reticle, instrument panel — e.g. a health gauge built into the avatar's suit,
ammo projected on the weapon itself) or *on top of it* (overlay). Diegetic deepens
immersion but can cost legibility; non-diegetic is clearer but flatter.

- Default to **immersion where it's safe, legibility where it's lethal.** Flavor telemetry can be diegetic; the stall warning cannot afford to be subtle.
- Keep it **spatially consistent** so the player builds muscle memory for where to look — a moving or restyled element breaks the glance.

### 4. Feedback must be immediate and proportional
The pilot is steering a control loop; the HUD closes it. Latency or ambiguity in
feedback makes good controls feel broken.

- Every input gets an instant, readable response — throttle, thrust vector, weapon
  state, lock progress.
- Show **rate and trend**, not just current value: closing speed, climb/dive rate,
  depth rate, heat ramp. Pilots fly the derivative.
- Make **orientation and motion in 6-DOF** intelligible where there's no natural "down"
  — velocity vector / prograde marker, drift indicator, and a clear reticle-vs-heading
  distinction so the player knows where they're pointing vs. where they're going.

### 5. Reticle, targeting, and threat awareness
In combat regimes this is the heart of the HUD.

- The **reticle is the player's primary focus**; everything combat-relevant should be
  readable in its vicinity (lead indicator, lock state, range, ammo/heat) so the eyes
  don't have to leave the fight.
- **Off-screen threats need on-screen representation** — directional indicators that
  point to what the player can't see, sized/colored by urgency.
- Keep target identification instant and unambiguous: friend/foe/neutral must never
  require a second look.

## How to apply this skill
When asked to design or critique a flight HUD:

1. Establish the **regimes in play** and whether one craft crosses between them. If it does, the unification problem dominates — lead with it.
2. Define the **stable spatial grammar** first (what lives where, always), then specify what each slot shows per regime and how transitions are telegraphed.
3. Triage every element by **"is this actionable in this regime right now?"** Cut or demote anything that fails. Justify what survives.
4. Map the **danger channels** (color/motion/audio) and confirm each lethal state has pre-attentive feedback and a non-color backup.
5. Decide diegetic vs. non-diegetic **per element**, biasing immersion for flavor and legibility for anything that can kill the player.
6. Ground recommendations in proven patterns where useful — 6-DOF multi-regime space sims, underwater survival games with depth/pressure and a surface transition, arcade air-combat reticle and threat design, diegetic-UI horror — but adapt to the specific craft and physics rather than copying.

## Validation
This is a design-judgment skill with no single correct answer. The real test is a
playtest under load: can a player crossing water→air→space keep flying without looking
*for* the instrument they need, and do lethal states register before they're read? Push
the user toward testing with **fresh players in motion**, not static screenshots — a HUD
that looks clean paused can still be unreadable at speed. The heuristics here prevent
known mistakes; testing catches the ones that only appear in the seat.

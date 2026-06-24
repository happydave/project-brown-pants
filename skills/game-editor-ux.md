---
name: game-editor-ux
description: >
  Design the UX of in-game creation tools and editors — ship/vehicle designers,
  base/factory builders, level editors, character creators, loadout and crafting
  screens, node/logic editors, and any "build mode" where a player authors
  something deliberately over time. Use this whenever the user is designing or
  critiquing a player-facing authoring tool, mentions a "designer," "editor,"
  "build mode," "creator," "blueprint" tool, or describes letting players make
  ships, bases, levels, machines, or characters — even if they don't say "UX."
  This is distinct from real-time HUD design; if the user is working on a
  flight/combat HUD or any heads-up display read under time pressure, use
  game-flight-hud instead. If the work spans both, use both.
---

# Game Editor & Authoring-Tool UX

Authoring tools are a different discipline from moment-to-moment game UI. A HUD is
read in a glance under time pressure; an editor is *inhabited* for long, deliberate
sessions. The player is not reacting — they are composing, and they will hit undo a
thousand times. Optimize for deep, exploratory, low-anxiety work, not speed.

The failure mode that kills creation tools is the **expert/novice gap**: powerful
enough for veterans means overwhelming for newcomers, and accessible for newcomers
means insulting to veterans. Most of this skill is about not getting trapped on one
horn of that.

## Diagnose before prescribing

Ask (or infer from context) before giving advice:

1. **What is the player composing, and how big does it get?** A 10-part loadout and
   a 10,000-bolt spaceship need totally different navigation, selection, and
   performance models.
2. **Who is the median user vs. the power user?** Is this a casual side-system or
   the core loop the game lives or dies on?
3. **Where does authored work get validated?** In the editor, or only by leaving it?
   This single answer drives the whole iteration-loop design (see below).
4. **Is creation solo, collaborative, or shared/published?** Multiplayer editing and
   blueprint-sharing change save semantics, naming, and ownership UI.

## Core principles

### 1. Tighten the author → test → fix loop until it's seamless
The single biggest determinant of how good an editor *feels* is how fast a creator
can validate an idea. Every context switch between building and testing is friction
that compounds across a session.

- Let players **test in place**. The best build tools let you drop straight into a
  simulated test and back out to the exact camera/state you left — authoring and
  validation share one space rather than living in separate screens. Aim for that.
- If a full test is impossible, give **live preview / cheap approximations** in the
  editor (predicted stats, validity highlighting, ghosted simulation) so the player
  isn't flying blind until commit.
- Make **undo/redo total and trustworthy**. Fearless experimentation requires that
  any action be reversible. Branch/version history is a bonus, not the baseline.

### 2. Direct manipulation: the cursor is the instrument
In a spatial editor the player edits *the thing itself*, not a form that describes it. The quality of that hand-to-object connection is most of the felt experience.

- **Point at what you mean.** Hover highlights the exact element under the cursor, and the editor shows a **ghost preview** of the pending action — where the part will land, what a click will delete — *before* the commit. The player should never be surprised by a click.
- **Manipulate in the viewport with handles/gizmos**, not only numeric fields: drag to place/move/rotate/scale, with the inspector as the precise fallback, and keep the two in sync. Direct for speed and feel; numeric for precision.
- **Snapping, grids, and symmetry are the leverage** of spatial editing: grid/face snapping, angle snapping, and mirror/symmetry turn fiddly placement into expressed intent. Always offer a toggle for free-form work.
- **Keyboard is the accelerator, not the price of entry.** Everything doable by mouse should have a shortcut for veterans, but nothing core should be *keyboard-only* — that is the trap to avoid (e.g. an action that silently targets a hidden keyboard cursor while the player is using the mouse).
- **Design for the input methods you ship on.** Mouse+keyboard, gamepad, and touch differ in precision and button budget; a console builder can't assume hover or right-click. Decide parity early rather than bolting a second input model on later.
- **Keep modes few and unmistakable.** Every mode (build/test, place/select, paint) is a chance to act in the wrong one: show the current mode loudly, make switching one cheap gesture, and prefer modeless tools where you can.

### 3. Progressive disclosure: one tool, many depths
Resolve the expert/novice gap by *layering* the same tool, not shipping a dumbed-down fork. Offer a simple/guided mode precisely because a full-power designer takes a while to get the hang of — a frank admission that raw power excludes newcomers.

- Default to a clean surface; reveal advanced panels, snapping modes, and raw parameters on demand.
- Provide a **guided/simple mode and an advanced mode** that operate on the *same underlying data*, so graduating from one to the other loses nothing.
- Hide complexity, don't remove it. The veteran must always be able to reach the low-level control (e.g. exact coordinates, the script chip, the raw value).

### 4. Teach with learnable examples, not walls of text
Players learn authoring tools by taking apart things that work.

- Ship **premade, editable examples** (premade dev blueprints players can open and dissect). Examples teach technique, set a quality bar, and unblock the blank-canvas freeze.
- Prefer **in-context, just-in-time hints** (tooltips on hover, contextual callouts) over front-loaded tutorials the player has forgotten by the time it matters.
- **Nail the first 60 seconds.** Open onto a non-empty canvas — a default starter object, or the last creation — so the player learns by editing something that already works rather than facing a blank void, and make the single most-likely next action obvious.

### 5. A configurable, spatial, persistent workspace
Long sessions mean the workspace itself is a tool. Respect that the player will make it their own.

- **Dockable, movable, resizable, closable panels**, with layout that persists across sessions and a "restore default layout" escape hatch. Let the player keep only what they need on screen. This chrome model assumes a pointer-and-windows platform; on console or touch, swap it for a chrome that fits the input (radial menus, a context bar, modal palettes) rather than porting desktop panels wholesale.
- Treat the **camera as a first-class control**: smooth orbit/pan/zoom, focus-on-selection, reset-to-origin, and saved viewpoints for large builds.
- **Selection is the editor's most-used verb.** Invest in it: box/lasso select,
  select-by-type, hierarchy/group selection, isolate/hide, and saving a selection as
  a reusable sub-assembly.
- **Stay responsive at scale.** A large build must not stutter or freeze the editor:
  cull/LOD off-screen detail, keep selection and camera smooth, and run expensive
  operations (validation, save, bake) without locking the UI. A janky editor reads as
  a broken one.

### 6. Make state and constraints legible
The player must always know what they have, what's valid, and what a change will cost.

- A **properties/inspector panel** that reflects the current selection and shows what is and isn't editable — never let players guess which fields are live.
- **Surface validity continuously**: what's unpowered, unconnected, over budget, structurally unsound, or colliding. Warn that overpowered thrusters on a weak frame snap the ship in half — better to show that *before* the test flight.
- Show **resource/budget feedback** (mass, cost, power, part count, poly/tri budget) live as a first-class readout, not buried in a submenu.
- **Prevent vs. warn — choose deliberately.** *Block* only the truly invalid (two parts in one cell); *warn but allow* the merely risky (overpowered thrusters on a weak frame) so the player can choose to break the rules. Reserve hard confirmations for destructive, irreversible actions; everything else leans on undo.
- **Don't encode meaning in colour alone.** Validity and budget cues need a second channel — icon, shape, label, or position — so they survive colourblindness and low contrast; the same applies to subtle hover states.

### 7. Respect the player's time investment
A creation tool accumulates hours of irreplaceable work. Treat that work as sacred.

- **Autosave, named saves, "save selection as," and safe overwrite semantics.**
  Never let one misclick destroy a session.
- Make **naming, organizing, and re-finding** creations painless — a blueprint
  library that scales to hundreds of items, with search/sort/tags.
- For shared/published work, design **ownership, versioning, and attribution**
  explicitly; copies and forks need clear provenance.

## How to apply this skill
When asked to design or critique an editor:

1. Run the four diagnostic questions and state the assumptions you're making.
2. Walk the **author → test → fix loop** first; it's the highest-leverage thing to get right. Identify every context switch and try to collapse it.
3. Trace one **direct-manipulation path** end to end — pick the most common edit and follow it: how does the player target it, what feedback confirms the target, what commits it, and how is it undone? Most editor pain hides in that loop.
4. Check the design against each core principle and call out the **specific** gap, not a generic "add tooltips." Tie each recommendation to what the player is composing and how large it gets.
5. Name the **expert/novice strategy** explicitly: how does a newcomer start, and how does a veteran avoid being slowed down, in the *same* tool?
6. Where useful, ground critiques in shipped examples (in-place test flight, a simple build mode, editable dev blueprints, dockable windows) — seek inspiration and adapt, don't cargo-cult.

## Validation
This is a design-judgment skill; there is no single correct output. The test of a
recommendation is whether it would survive a real playtest: would a first-time user
get a working creation out the door, and would a 200-hour veteran feel respected? When
in doubt, push the user toward **playtesting with both populations** rather than
trusting the heuristics alone — the heuristics prevent known mistakes; testing finds
the ones nobody predicted.

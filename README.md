# Robot arm lifting a deformable plastic cup with a cube inside (MuJoCo)

**Simulation Engineer assignment: ET Robotics**
Candidate: `Anshul Mittal` · Videos: [Google Drive](https://drive.google.com/drive/folders/1fN-V7oZd4qY2uzjdq974dEAJZ_lkX0u1)

A Franka Emika Panda arm picks a thin-walled (0.5 mm) plastic cup **vertically** while a cube sits inside it.
There are three trials, with cube masses of **10 g, 100 g and 500 g**. For each trial the simulation studies:

| # | Objective | Plots (`results/<trial>/`) |
|---|-----------|--------|
| 1 | Cup-surface deformation when the gripper squeezes the walls with the cup on the ground | `deformation.png` |
| 2 | Cup-surface deformation while the robot lifts the cup vertically | `deformation.png` |
| 3 | Deformation of the cup bottom during the pick | `deformation.png` |
| 4 | Friction behaviour as the cup deforms | `friction.png`, `friction_vs_deformation.png` |

There is one video per trial (10 g, 100 g, 500 g). Each one shows the complete pick, and the camera changes for
each objective (see section 3, step 3).

---

## 1. Requirements

* Linux (tested on Ubuntu 22.04), Python 3.10 or newer
* **MuJoCo 3.14.0** (Python package). The model uses the `discrete` integrator introduced in 3.14, which older
  versions don't have.
* numpy, matplotlib
* About 2 GB of RAM. One trial uses one CPU core, and `run_all.py` runs the three trials in parallel.

## 2. Installation

```bash
git clone https://github.com/anshul1996mittal/MuJoCo_cup_sim.git cup_sim
cd cup_sim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check the installation:

```bash
python -c "import mujoco; print(mujoco.__version__)"    # -> 3.14.0
```

## 3. Running the simulation

**Step 1: look at the scene (optional).** Open `models/scene.xml` in MuJoCo's `simulate` application (drag and
drop the file into its window). No controller runs there, so the arm doesn't move. It only shows the model.

**Step 2: run all three trials.** This takes about 8 minutes on a laptop with the trials in parallel:

```bash
python run_all.py
```

The script prints a summary table at the end. It is also written to `results/summary.md`.

**Step 3: watch a trial in the MuJoCo viewer.** Replay a finished trial, with the deformation shown as a colour
heat map on the cup (blue = none, red = largest):

```bash
python replay.py results/500g --speed 0.5
```

The camera follows the task: overview (approach), front (objective 1, squeeze on the ground), iso (objective 2,
lift), bottom with the floor hidden (objective 3), then a close-up of a finger with contact points and forces
(objective 4, release). `--speed 0.5` plays in slow motion. `--wait 5` pauses 5 s before playing.

**Other commands**

```bash
python run_trial.py --mass 500g            # a single trial (10g, 100g, 500g or any value, e.g. 250g)
python run_trial.py --mass 500g --viewer   # run the physics live in the viewer (slow, about 0.1x real time)
python validate.py                         # verification of the shell model against analytical solutions
python analyze.py                          # regenerate the comparison plots and summary table
```

Options for `run_trial.py` and `run_all.py`:

| option | meaning | default |
|---|---|---|
| `--grip-force N` | constant squeeze force per finger [N] | `auto` (see section 4) |
| `--thickness mm` | cup wall thickness | 0.5 |
| `--young GPa` | Young's modulus of the cup | 1.5 |
| `--mu` | finger-cup friction coefficient | 0.5 |
| `--res NT NW` | cup mesh: points around, segments up the wall | 24 8 |
| `--impratio` | MuJoCo friction impedance ratio | 100 |
| `--timestep` | simulation time step [s] | 0.001 |
| `--no-slip-test` | skip the grip-release test at the end | |
| `--no-rest-correction` | disable the curved-rest-shape correction (section 5.2) | |
| `--out DIR` | output folder | `results` |

On a machine without a display, set `MUJOCO_GL=egl` before the command. MuJoCo may print one
`Flex stiffness is too ill-conditioned ...` warning at the very first step. It doesn't come back and doesn't
affect the results.

## 4. Simulated setup

| item | value |
|---|---|
| Robot | Franka Emika Panda, model from MuJoCo Menagerie (`models/panda_cup.xml`) |
| Cup | truncated cone: bottom Ø 50 mm, mouth Ø 72 mm, height 85 mm, **wall 0.5 mm**; polypropylene (E = 1.5 GPa, ν = 0.42, ρ = 905 kg/m³), about 8.3 g. The rim is not modelled |
| Cube | 30 mm edge in every trial; mass 10 / 100 / 500 g |
| Friction | finger–cup μ = 0.5, cube–cup μ = 0.4, floor–cup μ = 0.4 |
| Grasp | top-down pinch; the finger pads squeeze the wall 55 mm above the floor |
| Motion | vertical descent → close → squeeze → **vertical lift of 100 mm** → hold 2 s → grip-release test |
| Grip force | force-controlled, `F = max(2 N, 1.5 · W / 2μ)` per finger → 2.0 N for 10 g and 100 g, 7.5 N for 500 g |

The motion is hard-coded, as the brief allows. The Cartesian waypoints (straight vertical lines) are converted to
joint trajectories by damped-least-squares IK sampled every 2 mm, and joint position servos track them. The
measured sideways motion of the gripper during the lift is below 1 mm.

**Grasp choice:** a horizontal side grasp was evaluated first. Keeping the hand level close to the robot's base
height pushes the Panda's joint 6 to its limit and needs a contorted wrist. A top-down pinch uses the natural arm
posture and keeps the whole pick vertical.

**Timeline [s]:** settle 0–0.5 · descent 0.5–2.0 · fingers pre-close 2.0–2.6 · squeeze (force ramp) 2.6–4.1 ·
hold on ground 4.1–5.1 · lift 5.1–7.6 · hold in air 7.6–9.6 · release test 9.6–13.6. After the release test the
fingers keep opening until the cup drops.

## 5. Methodology

### 5.1 Deformable cup: MuJoCo 2-D flex (shell)
* The cup's mid-surface is a triangle mesh: 24 points around × 8 segments up the wall, plus a 3-ring bottom disk.
  That gives 265 vertices, 504 triangles and a mean edge of 9.5 mm. It is created with
  `<flexcomp type="direct" dim="2">`. Every vertex is a 3-DoF point mass (795 DoF for the cup).
* Elasticity uses MuJoCo's shell model with `elastic2d="both"`:
  * **membrane**: constant-strain triangles
  * **bending**: the discrete quadratic bending energy on every interior edge

  Both are parametrised by E, ν and the **physical wall thickness**.
* Rayleigh damping (2·10⁻⁴ s) removes high-frequency ringing.
* The contact thickness is `radius = t/2`. The shell collides with the fingers, the cube and the floor.

### 5.2 Correction for the curved rest shape
MuJoCo's quadratic bending energy `E = ½ xᵀKx` assumes a **flat** rest state. For a cup, the undeformed shape
then carries spurious rest forces (up to about 730 N at the 90° bottom corner), and the unloaded cup distorts by
1.7 mm. K is constant, so applying the constant force `+K x₀` gives exactly `E = ½ (x−x₀)ᵀ K (x−x₀)`: bending
about the true curved shape. The force is rotated with the cup's best-fit rigid rotation every step. Remaining
drift of the unloaded cup: 0.0001 mm. Local element rotations are not corrected, which makes the sharply curved
bottom corner slightly stiffer, close to a clamped edge.

### 5.3 Bending-stiffness calibration
A cantilever plate strip with the cup's E, ν, t and element size was compared with Kirchhoff plate theory,
`D = Et³ / 12(1−ν²)`. The raw MuJoCo shell is **3.5× too flexible**. The factor barely changes with mesh
refinement and grows with ν, so it is not a discretisation error. The modulus passed to MuJoCo is therefore scaled
so that the cantilever matches theory. After calibration the ratio is 1.02, computed automatically in
`cupsim/calibration.py`. The membrane is scaled by the same factor, which has no visible effect because the
in-plane strains are tiny.

### 5.4 Solvers and numerical settings
| setting | value | reason |
|---|---|---|
| integrator | `discrete` | treats the flex stiffness implicitly (solves with M + h²K). A 1.5 GPa shell with 0.03 g nodes is unstable with explicit integration, and MuJoCo 3.14 doesn't allow flex elasticity with `implicit`/`implicitfast` |
| time step | 1 ms | stable with the implicit stiffness |
| constraint solver | Newton | converges in 1–3 iterations on this stiff problem. CG hit its iteration limit and gave noisy results |
| friction cone | elliptic | the exact Coulomb cone |
| `impratio` | 100 | stiffer friction rows reduce the creep that soft contacts allow under sustained tangential load |
| contact `solref` / `solimp` | (0.002, 1) / (0.99, 0.999, 0.0005) | stiff contacts. Contact `priority` makes the cup's parameters govern its contacts, so they aren't averaged with the softer robot defaults (averaging let the pads pass through the 0.5 mm wall) |

### 5.5 Robot control
* The arm uses the Menagerie joint position servos plus gravity compensation on all links.
* The gripper actuator is **force-controlled** (command = squeeze force per finger, with velocity damping),
  similar to `franka::Gripper::grasp(width, speed, force)`. The fingers close under position control until
  0.8 mm from the wall, then the force ramps up smoothly.

### 5.6 Measurements (`cupsim/metrics.py`)
* **Deformation.** The best-fit rigid motion of the whole cup (Kabsch) is removed at every sample, and the
  deformation is `u = Rᵀ(x − t) − X₀`. Reported quantities:
  * pad indentation (the largest inward wall displacement)
  * cross-section widths at the pads, along and across the squeeze axis (ovalisation)
  * bottom sag (the bottom centre relative to the bottom corner ring)
  * max / RMS |u|
* **Contacts.** Each cup contact is classified as left finger, right finger, floor, cube or other.
  `mj_contactForce` gives the normal force fₙ and the friction force fₜ. Derived quantities:
  * friction utilisation `|fₜ|/(μ fₙ)` (1 = at the Coulomb limit, sliding)
  * contact sliding speed, from the constraint velocities of the friction rows
  * slip of the wall under the pads relative to the gripper
  * the share of the weight carried by friction vs by normal forces
* **Release test.** The grip force is ramped down after the hold. It records when Coulomb friction alone
  becomes insufficient (`2μFₙ < W`), how the load path changes, and when the cup finally drops.

## 6. Results

Default settings (`python run_all.py`). Full table: `results/summary.md`. Time series: `results/<trial>/*.png`
and `results/comparison_timeseries.png`.

| quantity | 10 g | 100 g | 500 g |
|---|---|---|---|
| weight of cup + cube [N] | 0.18 | 1.06 | 4.99 |
| grip command / measured normal force on the ground [N per finger] | 2.0 / 2.06 | 2.0 / 2.08 | 7.5 / 7.47 |
| **Obj 1** pad indentation on the ground [mm] | 4.1 | 4.1 | 14.2 |
| **Obj 1** section at the pads, along / across the squeeze [mm] (undeformed 63.2) | 58.6 / 67.0 | 58.6 / 67.0 | 43.7 / 74.8 |
| **Obj 2** pad indentation when lifted [mm] | 4.2 | 4.2 | 14.8 |
| **Obj 2** section at the pads when lifted, along / across [mm] | 58.4 / 67.2 | 58.3 / 67.2 | 43.0 / 75.3 |
| **Obj 3** bottom sag on the ground → lifted [mm] | 0.08 → 0.24 | 0.11 → 0.44 | 0.61 → 1.21 |
| **Obj 4** friction utilisation when lifted (Coulomb estimate W/2μFₙ) | 0.09 (0.09) | 0.38 (0.47) | 0.66 (0.67) |
| **Obj 4** share of the weight carried by friction when lifted | 88 % | 80 % | 100 % |
| **Obj 4** contact sliding: squeeze / lift + hold [mm] | 1.1 / 0.08 | 1.2 / 0.31 | 2.0 / 0.98 |
| **Obj 4** release: grip at which 2μFₙ < W [N] | never | 0.51 | 4.03 |
| **Obj 4** release: max apparent μ = W/(2Fₙ) while still held | 0.64 | 1.93 | 1.22 |
| **Obj 4** finger opening when the cup drops [mm] (cup radius at pads: 32.4) | 32.9 | 33.3 | 33.4 |
| sideways gripper motion during the lift [mm] | 0.14 | 0.21 | 0.79 |

**Objective 1: squeezing on the ground.**
* The pads dent the wall and the cross-section becomes oval: it narrows along the squeeze axis and widens across
  it. The response is close to linear, about 2 mm of indentation per N per finger.
* The largest displacements are at the pads and at the free mouth of the cup. The base, stiffened by the bottom
  disk, barely moves.
* While the wall bends, the pads slide 1–2 mm over it (utilisation = 1, with small stick-slip steps). This
  friction pushes the cup down into the floor: the floor reaction rises up to about 6 N above the weight for
  500 g.

**Objective 2: lifting.**
* At lift-off the weight moves from the floor to the pads. The indentation grows slightly as the load pulls the
  wall down between the pads, then stays constant during the hold.
* At the same grip force the wall deformation hardly depends on the cube mass (14.2–14.4 mm at 7.5 N for all
  three masses, `results/fixed_grip_7.5N/`). The grip force drives the wall deformation. The payload only
  matters through the grip force it needs.

**Objective 3: bottom.**
* On the ground the floor supports the base. Lifted, the base carries the cube like a plate supported by the
  wall: the sag is 0.24 / 0.44 / 1.21 mm.
* For 100 g and 500 g the sag lies between the analytical clamped-plate bounds (uniform load ↔ point load:
  0.16–0.64 mm and 0.80–3.2 mm). For 10 g the sag comes mostly from the pinch warping the base.
* The bottom also warps into an oval that follows the wall's ovalisation.

**Objective 4: friction.**
* **Lift and hold.** Friction carries 80–100 % of the weight. The utilisation grows with the payload and follows
  W/(2μFₙ). The contacts creep slightly (0.1–1 mm), which is characteristic of MuJoCo's soft contact model.
* **Release.** When the grip is reduced, the cup does not fall at the Coulomb limit. It settles about 1 mm deeper
  into the dents made by the pads and onto the tapered wall. The contact normals tilt upward, and the load moves
  from friction to normal forces (500 g: from 100 % friction to about 6 % just before the drop).
  * For the shallow 4 mm dent (10 g), the apparent friction coefficient peaks at **0.64**. The rigid-cone
    prediction μ + tan α = 0.50 + 0.13 = 0.63.
  * Deeper dents add geometric interlock (apparent μ 1.2–1.9).
  * The cup only drops when the fingers open to about the cup's undeformed radius.

  Deformation therefore raises the holding capacity well above pure Coulomb friction.

## 7. Verification (`python validate.py` → `results/validation.md`)

| check | result |
|---|---|
| cantilever plate strip vs Kirchhoff plate theory | raw MuJoCo 3.54× too flexible → 1.02× after calibration |
| unloaded cup, no gravity, 0.5 s | drift 1.69 mm without correction → 0.0001 mm with it |
| bottom sag vs clamped-plate bounds | within the bounds for 100 g and 500 g |
| cube-on-bottom contact force vs cube weight when lifted | 0.10 / 0.98 / 5.11 N vs 0.10 / 0.98 / 4.91 N |
| identical runs | bit-for-bit identical results (deterministic) |

## 8. Limitations
* Linear-elastic shell: no plasticity or permanent creasing. The 9.5 mm elements resolve ovalisation and the pad
  dent, not fine wrinkles (`--res 32 10` refines the mesh at about 2× the cost).
* The rest-shape correction is applied to the whole cup's rotation, not per element.
* MuJoCo contacts are soft constraints, so perfect sticking shows up as slow creep.
* Rigid cube, and hard finger pads (no rubber fingertips).
* About 0.1× real time on one CPU core. `replay.py` shows the result at real-time speed.

## 9. Repository structure
```
run_all.py            runs the three trials, comparison plots, summary table, validation
run_trial.py          one trial: simulate -> save data -> plots (command-line options)
replay.py             real-time replay of a saved trial in the MuJoCo viewer
analyze.py            per-trial plots, comparison, summary.md
validate.py           shell-model verification
cupsim/
  config.py           all physical and numerical parameters (units, sources)
  cup_mesh.py         triangle mesh of the cup
  scene.py            generates the MuJoCo XML scene (robot + cup + cube + floor + cameras)
  calibration.py      bending-stiffness calibration
  ik.py               inverse kinematics of the Panda
  controller.py       pick sequence and gripper force control
  sim.py              simulation loop, logging, rest-shape correction
  metrics.py          deformation and contact / friction measurements
models/
  panda_cup.xml       Franka Panda (MuJoCo Menagerie) with the changes listed in its header
  scene.xml           the complete scene (100 g cube) for the `simulate` app
  assets/             Panda meshes
results/              plots and summaries of the submitted runs
```

## 10. Credits
Franka Emika Panda model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), Apache-2.0
(`models/LICENSE_franka_emika_panda`). Physics engine: [MuJoCo](https://mujoco.org) 3.14.0.

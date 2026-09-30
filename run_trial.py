#!/usr/bin/env python3
"""Run ONE trial (one cube mass): simulate, save data, make plots.

Examples
--------
  python run_trial.py --mass 100g                 # full pipeline -> results/100g/
  python run_trial.py --mass 500g --viewer        # watch it live in the MuJoCo viewer
  python run_trial.py --mass 500g --grip-force 5  # fixed squeeze force per finger [N]
  python run_trial.py --mass 100g --thickness 0.8 # 0.8 mm wall instead of 0.5 mm
"""
import argparse
import json
import os
import sys

import numpy as np

from analyze import summarize
from cupsim.config import Config, TRIAL_MASSES
from cupsim.sim import Trial

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_mass(s):
    if s in TRIAL_MASSES:
        return s, TRIAL_MASSES[s]
    v = float(s.rstrip("g")) / 1000.0 if s.endswith("g") else float(s)
    return f"{v * 1000:g}g", v


def add_common_args(ap):
    ap.add_argument("--grip-force", default=None,
                    help='squeeze force per finger [N], or "auto" (default: auto = 1.5 x the minimum)')
    ap.add_argument("--thickness", type=float, default=None, help="cup wall thickness [mm]")
    ap.add_argument("--young", type=float, default=None, help="Young's modulus [GPa]")
    ap.add_argument("--mu", type=float, default=None, help="finger-cup friction coefficient")
    ap.add_argument("--res", type=int, nargs=2, default=None, metavar=("N_THETA", "N_WALL"),
                    help="cup mesh resolution (default 24 8)")
    ap.add_argument("--impratio", type=float, default=None,
                    help="MuJoCo friction-to-normal impedance ratio (default 100; higher = less creep)")
    ap.add_argument("--dmax", type=float, default=None,
                    help="contact solimp dmax (default 0.999; closer to 1 = harder contact)")
    ap.add_argument("--timestep", type=float, default=None, help="simulation time step [s]")
    ap.add_argument("--no-rest-correction", action="store_true",
                    help="disable the curved-rest-shape bending correction (to see MuJoCo's raw behaviour)")
    ap.add_argument("--no-slip-test", action="store_true", help="skip the release (slip) test")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"), help="output folder")


def config_from_args(a):
    cfg = Config()
    if a.grip_force is not None:
        cfg.task.grip_force = a.grip_force if a.grip_force == "auto" else float(a.grip_force)
    if a.thickness is not None:
        cfg.cup.thickness = a.thickness / 1000.0
    if a.young is not None:
        cfg.cup.young = a.young * 1e9
    if a.mu is not None:
        cfg.cup.mu_finger = a.mu
    if a.res is not None:
        cfg.cup.n_theta, cfg.cup.n_wall = a.res
    if a.impratio is not None:
        cfg.sim.impratio = a.impratio
    if a.dmax is not None:
        si = list(cfg.cup.solimp)
        si[1] = a.dmax
        si[0] = min(si[0], a.dmax)
        cfg.cup.solimp = tuple(si)
    if a.timestep is not None:
        cfg.sim.timestep = a.timestep
    if a.no_rest_correction:
        cfg.cup.rest_shape_correction = False
    if a.no_slip_test:
        cfg.task.slip_test = False
    return cfg


def run(name, mass, cfg, out_root, viewer=False):
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    print(f"=== trial {name}: cube {mass * 1000:g} g -> {out}")
    trial = Trial(cfg, mass, name)
    print(f"    cup: {trial.ids.nvert} vertices, {trial.ids.cup_mass * 1e3:.2f} g, wall "
          f"{cfg.cup.thickness * 1e3:.2f} mm | grip force {trial.F:.2f} N/finger | nv={trial.model.nv}")
    frame_dt = 1.0 / cfg.sim.snapshot_fps
    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(trial.model, trial.data) as v:
            v.cam.lookat[:] = [cfg.task.cup_xy[0], 0, 0.08]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.55, 145, -20
            from cupsim.metrics import heat_rgba
            import mujoco as mj

            def on_frame(tr, t):
                with v.lock():
                    scn = v.user_scn
                    scn.ngeom = 0
                    X = tr.verts()
                    rgba = heat_rgba(tr.last_mag, 0.008)
                    for i in range(len(X)):
                        mj.mjv_initGeom(scn.geoms[i], mj.mjtGeom.mjGEOM_SPHERE, np.array([0.002, 0, 0]),
                                        X[i], np.eye(3).ravel(), rgba[i])
                    scn.ngeom = len(X)
                v.sync()
            L = trial.run(on_frame=on_frame, frame_dt=frame_dt, viewer=v)
    else:
        L = trial.run(frame_dt=frame_dt)

    meta = dict(trial=name, cube_mass_kg=mass, cup_mass_kg=trial.ids.cup_mass, weight_N=trial.W,
                grip_force_per_finger_N=trial.F, sim_wall_time_s=trial.wall_time,
                gross_slip_detected=bool(trial.slip.get("detected")))
    summary = summarize(L, cfg, meta)
    np.savez_compressed(os.path.join(out, "run.npz"),
                        **{f"log_{k}": v for k, v in L.items()},
                        **{f"frame_{k}": v for k, v in trial.frames.items()})
    with open(os.path.join(out, "config.json"), "w") as f:
        f.write(cfg.to_json())
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    done in {trial.wall_time:.0f}s. summary:")
    for k, v in summary.items():
        print(f"      {k:40s} {v:.3f}" if isinstance(v, float) else f"      {k:40s} {v}")

    from analyze import plot_trial
    plot_trial(out)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mass", default="100g", help="10g | 100g | 500g | any value like 250g or 0.25")
    ap.add_argument("--viewer", action="store_true", help="show the simulation live (slow: ~0.07x real time)")
    add_common_args(ap)
    a = ap.parse_args()
    name, mass = parse_mass(a.mass)
    cfg = config_from_args(a)
    run(name, mass, cfg, a.out, viewer=a.viewer)


if __name__ == "__main__":
    sys.exit(main())

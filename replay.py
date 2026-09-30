#!/usr/bin/env python3
"""Replay a finished trial in the interactive MuJoCo viewer at real-time speed.

The physics runs at ~0.05-0.1x real time, so watching it live is slow. Every
trial saves full state snapshots (30 per second); this script plays them back
with the deformation heat map drawn on the cup.

  python replay.py results/500g            # play once
  python replay.py results/500g --loop     # loop forever (close the window to quit)
  python replay.py results/500g --speed 0.25

Viewer tips: double-click a body to select it, right-drag to move the camera,
Tab/Shift-Tab toggle the side panels, 'C' shows contact points, 'F' contact forces.
"""
import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from analyze import load
from cupsim.metrics import heat_rgba
from cupsim.scene import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_dir")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--heat-max", type=float, default=None, help="colour-scale max [mm]")
    a = ap.parse_args()

    L, F, cfg, S = load(a.trial_dir)
    model, mesh, ids = build(cfg)
    d = mujoco.MjData(model)
    vmax = (a.heat_max / 1000) if a.heat_max else max(2e-3, float(np.percentile(F["disp"].max(1), 99)))
    print(f"trial {S['trial']}: {len(F['t'])} frames, {F['t'][-1]:.1f} s, heat-map max {vmax * 1e3:.1f} mm")

    with mujoco.viewer.launch_passive(model, d) as v:
        v.cam.lookat[:] = [cfg.task.cup_xy[0], 0, 0.08]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.55, 145, -20
        while v.is_running():
            t_wall0 = time.time()
            for k, t in enumerate(F["t"]):
                if not v.is_running():
                    break
                d.qpos[:] = F["qpos"][k]
                d.qvel[:] = F["qvel"][k]
                d.ctrl[:] = F["ctrl"][k]
                d.mocap_pos[:] = F["mocap_pos"][k]
                d.time = t
                mujoco.mj_forward(model, d)
                with v.lock():
                    scn = v.user_scn
                    X = d.flexvert_xpos[ids.vert_adr:ids.vert_adr + ids.nvert]
                    rgba = heat_rgba(F["disp"][k], vmax)
                    for i in range(len(X)):
                        mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([0.002, 0, 0]), X[i], np.eye(3).ravel(), rgba[i])
                    scn.ngeom = len(X)
                v.sync()
                lag = t / a.speed - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(lag)
            if not a.loop:
                break


if __name__ == "__main__":
    main()

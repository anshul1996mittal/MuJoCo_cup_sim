#!/usr/bin/env python3
"""Replay a finished trial in the MuJoCo viewer.

    python replay.py results/500g --speed 0.5 --wait 5

The camera changes with the phase of the task:
    approach        -> overview
    squeeze         -> front    (objective 1)
    lift            -> iso      (objective 2)
    hold in the air -> bottom   (objective 3, floor hidden)
    release         -> contact  (objective 4, contact forces)
Cup colours show the deformation (blue = none, red = largest).
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from analyze import load
from cupsim.controller import make_timeline
from cupsim.metrics import heat_rgba
from cupsim.scene import build

ap = argparse.ArgumentParser()
ap.add_argument("trial_dir")
ap.add_argument("--speed", type=float, default=1.0, help="0.5 = slow motion")
ap.add_argument("--wait", type=float, default=0.0, help="seconds to wait before playing")
args = ap.parse_args()

L, F, cfg, S = load(args.trial_dir)
model, mesh, ids = build(cfg)
data = mujoco.MjData(model)
tl = make_timeline(cfg)
T = F["t"]

# colour scales: whole cup, and the bottom only (its deformation is much smaller)
scale_cup = float(np.percentile(F["disp"].max(1), 99))
bottom = np.arange(1 + cfg.cup.n_theta * cfg.cup.n_bottom)
scale_bottom = 1.2 * float(F["disp"][:, bottom].max())


def view_for(t):
    if t < tl.start("squeeze"):
        return "overview", "approach"
    if t < tl.start("lift"):
        return "front", "objective 1: squeeze on the ground"
    if t < tl.start("hold_air"):
        return "iso", "objective 2: lift"
    if t < tl.start("release"):
        return "bottom", "objective 3: cup bottom"
    return "contact", "objective 4: friction / release"


def show(viewer, k):
    data.qpos[:] = F["qpos"][k]
    data.qvel[:] = F["qvel"][k]
    data.ctrl[:] = F["ctrl"][k]
    data.mocap_pos[:] = F["mocap_pos"][k]
    data.qacc_warmstart[:] = F["qacc_warmstart"][k]
    camera, label = view_for(T[k])
    if camera == "contact":
        mujoco.mj_forward(model, data)       # full solve: needed for the contact forces
    else:
        mujoco.mj_fwdPosition(model, data)   # positions and contacts only: fast

    scale = scale_bottom if camera == "bottom" else scale_cup
    colors = heat_rgba(F["disp"][k], scale)
    verts = data.flexvert_xpos[ids.vert_adr:ids.vert_adr + ids.nvert]

    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera(camera).id
        viewer.opt.geomgroup[1] = camera != "bottom"             # group 1 = floor
        forces = camera == "contact"
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = forces
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = forces
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = forces

        scn = viewer.user_scn
        for i, p in enumerate(verts):
            mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.002, 0, 0]), p, np.eye(3).ravel(), colors[i])
        scn.ngeom = len(verts)
    viewer.sync()
    return label


print(f"trial {S['trial']}: {T[-1]:.1f} s, video length {T[-1] / args.speed:.0f} s")
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    show(viewer, 0)
    if args.wait > 0:
        print(f"starting in {args.wait:.0f} s - start recording now")
        time.sleep(args.wait)

    # play in real time: always show the frame that matches the elapsed time
    start = time.time()
    last = None
    while viewer.is_running():
        t_now = (time.time() - start) * args.speed
        if t_now > T[-1]:
            break
        k = int(np.searchsorted(T, t_now))
        label = show(viewer, k)
        if label != last:
            print(f"  {T[k]:5.1f} s  {label}")
            last = label
        next_frame = (T[min(k + 1, len(T) - 1)]) / args.speed
        delay = next_frame - (time.time() - start)
        if delay > 0:
            time.sleep(delay)

    print("finished - stop recording, then close the window")
    while viewer.is_running():
        time.sleep(0.1)

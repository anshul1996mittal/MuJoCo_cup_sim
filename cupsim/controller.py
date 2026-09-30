"""Hard-coded pick controller: vertical descent -> squeeze -> vertical lift -> hold -> release.

Arm   : joint-space position servos (Menagerie gains) tracking IK trajectories
        that were pre-computed along straight vertical lines (see ik.py).
Gripper: force control. ctrl of the "grip" actuator = squeeze force per finger [N].
        Before contact the fingers are position-controlled in software
        (ctrl = kp * position error) so they approach the wall slowly.
"""
from dataclasses import dataclass

import numpy as np

from .ik import PandaIK, top_down_rotation

PHASES = ["settle", "approach", "preclose", "squeeze", "hold_ground",
          "lift", "hold_air", "release", "done"]


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def grip_force_for(cfg, cube_mass, cup_mass, g=9.81):
    tk = cfg.task
    if tk.grip_force != "auto":
        return float(tk.grip_force)
    W = (cube_mass + cup_mass) * g
    return max(tk.grip_force_min, tk.grip_safety * W / (2 * cfg.cup.mu_finger))


@dataclass
class Timeline:
    names: list
    starts: np.ndarray
    ends: np.ndarray

    def phase(self, t):
        for n, a, b in zip(self.names, self.starts, self.ends):
            if a <= t < b:
                return n, (t - a) / max(b - a, 1e-9)
        return "done", 1.0

    def start(self, name):
        return float(self.starts[self.names.index(name)])

    def end(self, name):
        return float(self.ends[self.names.index(name)])


def make_timeline(cfg):
    tk = cfg.task
    dur = [tk.t_settle, tk.t_approach, tk.t_preclose, tk.t_squeeze, tk.t_hold_ground,
           tk.t_lift, tk.t_hold_air, tk.t_release if tk.slip_test else 0.0]
    ends = np.cumsum(dur)
    starts = ends - np.array(dur)
    return Timeline(PHASES[:-1], starts, ends)


class PickController:
    def __init__(self, model, ids, cfg, grip_force):
        self.m, self.ids, self.cfg = model, ids, cfg
        self.F = grip_force
        self.tl = make_timeline(cfg)
        tk, cup = cfg.task, cfg.cup
        cx, cy = tk.cup_xy
        zg = tk.grasp_height
        R = top_down_rotation()
        ik = PandaIK(model, ids)
        self.p_pre = np.array([cx, cy, zg + tk.approach_height])
        self.p_grasp = np.array([cx, cy, zg])
        self.p_lift = np.array([cx, cy, zg + tk.lift_height])
        q_pre, err = ik.solve(self.p_pre, R, ik.q_rest, iters=2000)
        self.s_down, self.Q_down, e1 = ik.line(self.p_pre, self.p_grasp, R, q_pre)
        self.s_up, self.Q_up, e2 = ik.line(self.p_grasp, self.p_lift, R, self.Q_down[-1])
        self.ik_error = max(err, e1, e2)
        if self.ik_error > 1e-4:
            raise RuntimeError(f"IK failed (error {self.ik_error:.2e} m) - target out of reach?")
        self.q_init = q_pre

        # finger opening at which the (protruding) pad boxes just touch the wall
        r_at = cup.r_bottom + (cup.r_top - cup.r_bottom) * (zg / cup.height)
        self.q_open = 0.04
        self.q_touch = r_at + cup.thickness / 2
        self.q_preclose = min(self.q_open, self.q_touch + 0.0008)
        self.kp_grip = 800.0      # N/m, software position loop for the fingers
        self.F_open = 3.0         # N, opening force at the end of the release ramp

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _interp(s_grid, Q, s):
        return np.array([np.interp(s, s_grid, Q[:, j]) for j in range(Q.shape[1])])

    def arm_target(self, t):
        name, s = self.tl.phase(t)
        if name in ("settle",):
            return self.Q_down[0]
        if name == "approach":
            return self._interp(self.s_down, self.Q_down, min_jerk(s))
        if name in ("preclose", "squeeze", "hold_ground"):
            return self.Q_down[-1]
        if name == "lift":
            return self._interp(self.s_up, self.Q_up, min_jerk(s))
        return self.Q_up[-1]

    def tcp_target(self, t):
        name, s = self.tl.phase(t)
        if name == "settle":
            return self.p_pre
        if name == "approach":
            return self.p_pre + (self.p_grasp - self.p_pre) * min_jerk(s)
        if name in ("preclose", "squeeze", "hold_ground"):
            return self.p_grasp
        if name == "lift":
            return self.p_grasp + (self.p_lift - self.p_grasp) * min_jerk(s)
        return self.p_lift

    def grip_command(self, t, q_finger):
        """Returns (ctrl [N per finger, + = close], commanded force for logging, mode)."""
        name, s = self.tl.phase(t)
        if name in ("settle", "approach"):
            return self.kp_grip * (q_finger - self.q_open), 0.0, "position"
        if name == "preclose":
            q_des = self.q_open + (self.q_preclose - self.q_open) * min_jerk(s)
            return self.kp_grip * (q_finger - q_des), 0.0, "position"
        if name == "squeeze":
            f = self.F * min_jerk(s)
            return f, f, "force"
        if name in ("hold_ground", "lift", "hold_air"):
            return self.F, self.F, "force"
        if name == "release":            # slip test: linear force ramp F -> -F_open (fingers open)
            f = self.F * (1.0 - s) - self.F_open * s
            return f, f, "force"
        return -self.F_open, -self.F_open, "force"

    def apply(self, d, t):
        d.ctrl[self.ids.arm_act] = self.arm_target(t)
        qf = float(np.mean(d.qpos[self.ids.finger_qadr]))
        u, f_cmd, mode = self.grip_command(t, qf)
        lo, hi = self.m.actuator_ctrlrange[self.ids.grip_act]
        d.ctrl[self.ids.grip_act] = np.clip(u, lo, hi)
        return f_cmd, mode

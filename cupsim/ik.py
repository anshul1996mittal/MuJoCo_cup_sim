"""Damped-least-squares inverse kinematics for the Panda TCP.

Used *offline* to turn the hard-coded Cartesian waypoints into joint-space
trajectories (the brief allows a hard-coded motion). Straight-line Cartesian
segments are sampled densely and solved one after another (warm start), so the
TCP follows a straight vertical line during the lift - no sideways motion.
"""
import mujoco
import numpy as np


def top_down_rotation():
    """Hand frame for a vertical (top-down) pinch grasp.

    hand z (approach)      -> world -z   (fingers point down)
    hand y (finger motion) -> world -y   (fingers squeeze the cup wall along y)
    hand x                 -> world +x
    Note: a horizontal side grasp was evaluated first, but the Panda's joint-6
    range makes a level hand near its own base height reachable only with a
    contorted wrist; the top-down pinch is the natural, fully vertical pick.
    """
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0]])   # columns = hand axes in world


class PandaIK:
    def __init__(self, model, ids):
        self.m = model
        self.d = mujoco.MjData(model)
        self.ids = ids
        self.lo = model.jnt_range[[model.joint(f"joint{i}").id for i in range(1, 8)], 0]
        self.hi = model.jnt_range[[model.joint(f"joint{i}").id for i in range(1, 8)], 1]
        # comfortable posture used in the null space (Panda 'home', hand pointing down)
        self.q_rest = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.785])

    def fk(self, q):
        self.d.qpos[self.ids.arm_qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        s = self.ids.tcp_site
        return self.d.site_xpos[s].copy(), self.d.site_xmat[s].reshape(3, 3).copy()

    def solve(self, pos, rot, q0, iters=300, tol=1e-6, damping=1e-3, null_gain=0.05):
        q = np.array(q0, dtype=float)
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        qt = np.zeros(4)
        qc = np.zeros(4)
        mujoco.mju_mat2Quat(qt, rot.ravel())
        for _ in range(iters):
            p, R = self.fk(q)
            mujoco.mju_mat2Quat(qc, R.ravel())
            # orientation error as a rotation vector (world frame)
            qerr = np.zeros(4)
            qci = np.zeros(4)
            mujoco.mju_negQuat(qci, qc)
            mujoco.mju_mulQuat(qerr, qt, qci)
            werr = np.zeros(3)
            mujoco.mju_quat2Vel(werr, qerr, 1.0)
            err = np.concatenate([pos - p, werr])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.ids.tcp_site)
            J = np.vstack([jacp, jacr])[:, self.ids.arm_dadr]
            JJt = J @ J.T + damping ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # null-space posture regularisation
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (null_gain * (self.q_rest - q))
            q = np.clip(q + dq, self.lo + 1e-3, self.hi - 1e-3)
        p, R = self.fk(q)
        return q, np.linalg.norm(pos - p)

    def line(self, p0, p1, rot, q0, step=0.002):
        """Joint solutions along the straight line p0 -> p1 (inclusive)."""
        n = max(2, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / step)) + 1)
        qs, q = [], q0
        worst = 0.0
        for s in np.linspace(0, 1, n):
            q, e = self.solve((1 - s) * np.asarray(p0) + s * np.asarray(p1), rot, q)
            worst = max(worst, e)
            qs.append(q)
        return np.linspace(0, 1, n), np.array(qs), worst

"""Deformation and contact/friction measurements.

Deformation
-----------
The cup both moves (lift) and deforms, so the raw vertex displacement mixes
rigid motion and deformation. At every sample we compute the best-fit rigid
transform (Kabsch / Procrustes) that maps the undeformed mesh X0 onto the
current vertices X, and define the deformation field in the cup's own frame

        u_i = R^T (x_i - t) - X0_i            (co-rotational displacement)

From u we report
  * max |u|, RMS |u|               - overall deformation
  * pad indentation                - largest inward radial displacement of the wall
  * grasp-ring widths (x / y)      - ovalisation of the cross-section at the pads
  * bottom sag                     - bottom centre vs. bottom edge (+ = bulges down)
  * bottom max |u_z|               - largest out-of-plane deflection of the base

Contacts
--------
Every MuJoCo contact that involves the cup flex is classified by the other
object (left finger, right finger, floor, cube, other robot link). For each we
get the 3-D contact force in the contact frame (normal, t1, t2) via
mj_contactForce and sum per group:
  Fn  = sum of normal forces
  Ft  = vector sum of friction (tangential) forces acting ON THE CUP (world frame)
  util = |f_t| / (mu f_n) per contact -> 1.0 means the contact is at the
         Coulomb limit (sliding); we report the force-weighted mean and the max.
  v_slide = tangential relative velocity at the contact, read from the
         constraint velocities efc_vel = J qvel of the two friction rows of the
         contact (force-weighted mean). Integrated over time this gives the true
         sliding distance, independent of how much the wall deforms.
"""
import mujoco
import numpy as np


def kabsch(X0, X):
    """R, t minimising sum |R X0_i + t - X_i|^2."""
    c0, c = X0.mean(0), X.mean(0)
    H = (X0 - c0).T @ (X - c)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, c - R @ c0


class CupDeformation:
    def __init__(self, mesh, cfg):
        self.mesh = mesh
        self.X0 = mesh.vertices.copy()
        zg = cfg.task.grasp_height - cfg.cup.thickness / 2
        self.grasp_ring = int(np.argmin(np.abs(mesh.ring_z - zg)))
        self.ring_ids = mesh.ring_ids
        r0 = np.linalg.norm(self.X0[:, :2], axis=1)
        self.radial = np.zeros_like(self.X0)
        nz = r0 > 1e-9
        self.radial[nz, :2] = self.X0[nz, :2] / r0[nz, None]
        self.wall = mesh.wall_ids
        self.bottom = mesh.bottom_ids

    def compute(self, X):
        R, t = kabsch(self.X0, X)
        local = (X - t) @ R            # = R^T (x - t)
        u = local - self.X0
        mag = np.linalg.norm(u, axis=1)
        inward = -(u * self.radial).sum(1)
        ring = local[self.ring_ids[self.grasp_ring]]
        ring0 = self.X0[self.ring_ids[self.grasp_ring]]
        b = self.bottom
        edge = self.mesh.bottom_edge_ids
        sag = local[edge, 2].mean() - local[self.mesh.bottom_center_id, 2]
        return {
            "max_disp": mag.max(),
            "rms_disp": np.sqrt((mag ** 2).mean()),
            "wall_max_disp": mag[self.wall].max(),
            "pad_indent": inward[self.wall].max(),
            "width_y": np.ptp(ring[:, 1]),          # along the squeeze axis
            "width_x": np.ptp(ring[:, 0]),          # perpendicular
            "width0": np.ptp(ring0[:, 1]),
            "bottom_sag": sag,
            "bottom_max_w": np.abs(u[b, 2]).max(),
            "bottom_max_disp": mag[b].max(),
        }, mag, R, t


GROUPS = ("left", "right", "floor", "cube", "robot_other")

_HEAT = np.array([[0.19, 0.07, 0.23], [0.16, 0.47, 0.95], [0.11, 0.85, 0.62],
                  [0.93, 0.85, 0.20], [0.94, 0.37, 0.10], [0.48, 0.02, 0.01]])


def heat_rgba(values, vmax):
    """Colour map for the deformation heat map in the viewer (dark blue = 0 ... dark red = vmax)."""
    x = np.clip(np.asarray(values) / vmax, 0, 1) * (len(_HEAT) - 1)
    rgb = np.stack([np.interp(x, np.arange(len(_HEAT)), _HEAT[:, k]) for k in range(3)], axis=-1)
    return np.concatenate([rgb, np.ones(rgb.shape[:-1] + (1,))], axis=-1).astype(np.float32)


class ContactProbe:
    def __init__(self, model, ids):
        self.m, self.ids = model, ids
        self.body_group = {}
        for b in range(model.nbody):
            name = model.body(b).name
            if b == ids.left_finger:
                self.body_group[b] = "left"
            elif b == ids.right_finger:
                self.body_group[b] = "right"
            elif b == 0:
                self.body_group[b] = "floor"
            elif b == ids.cube_body:
                self.body_group[b] = "cube"
            elif name.startswith(("link", "hand")):
                self.body_group[b] = "robot_other"
        self._f = np.zeros(6)

    def contacts(self, d):
        """Yield one record per cup contact: (group, pos, fn, f_normal_on_cup, f_friction_on_cup, util, v_slide)."""
        m = self.m
        for i in range(d.ncon):
            c = d.contact[i]
            if c.flex[0] == self.ids.flex and c.geom[1] >= 0:
                cup_second, other_geom = False, c.geom[1]
            elif c.flex[1] == self.ids.flex and c.geom[0] >= 0:
                cup_second, other_geom = True, c.geom[0]
            else:
                continue
            grp = self.body_group.get(int(m.geom_bodyid[other_geom]))
            if grp is None:
                continue
            mujoco.mj_contactForce(m, d, i, self._f)
            fr = c.frame.reshape(3, 3)          # rows: normal, t1, t2 (normal: obj1 -> obj2)
            sign = 1.0 if cup_second else -1.0   # force on the cup
            fn = self._f[0]
            f_t = sign * (self._f[1] * fr[1] + self._f[2] * fr[2])
            f_n = sign * fn * fr[0]
            mu = c.friction[0]
            util = np.hypot(self._f[1], self._f[2]) / (mu * fn) if (fn > 1e-9 and mu > 0) else 0.0
            vs = 0.0
            if c.efc_address >= 0 and c.dim >= 3:
                vt = d.efc_vel[c.efc_address + 1:c.efc_address + 3]
                vs = np.hypot(vt[0], vt[1])
            yield grp, c.pos.copy(), fn, f_n, f_t, util, vs

    def compute(self, d):
        out = {g: dict(fn=0.0, ft=np.zeros(3), f=np.zeros(3), n=0, util_max=0.0, util_w=0.0,
                       pos=np.zeros(3), vslide=0.0) for g in GROUPS}
        for grp, pos, fn, f_n, f_t, util, vs in self.contacts(d):
            g = out[grp]
            g["fn"] += fn
            g["ft"] += f_t
            g["f"] += f_n + f_t
            g["n"] += 1
            g["pos"] += fn * pos
            g["vslide"] += fn * vs
            g["util_max"] = max(g["util_max"], util)
            g["util_w"] += util * fn
        for g in out.values():
            if g["fn"] > 1e-9:
                g["util_w"] /= g["fn"]
                g["pos"] /= g["fn"]
                g["vslide"] /= g["fn"]
        return out

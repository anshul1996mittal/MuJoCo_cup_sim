"""One trial = one cube mass. Runs the physics, logs everything, detects slip."""
import time

import mujoco
import numpy as np

from .config import Config
from .controller import PickController, grip_force_for, PHASES
from .metrics import CupDeformation, ContactProbe, GROUPS, kabsch
from .scene import build


class RestShapeCorrection:
    """Makes the curved cup stress-free in its as-moulded shape.

    MuJoCo's shell bending (elastic2d) is the discrete *quadratic* bending
    energy E = 1/2 x^T K x (Wardetzky et al. 2007), which is exact for a FLAT
    rest shape. For a curved shell (a cup!) it produces rest forces
    f0 = -K x0 that would try to flatten the wall and unfold the 90 deg
    bottom/wall corner (the uncorrected cup distorts by ~2.6 mm unloaded;
    bending-only it collapses completely).

    Because K is constant, adding the opposite force +K x0 gives exactly
    E = 1/2 (x - x0)^T K (x - x0), i.e. linear-elastic bending about the real
    curved shape. K acts identically on x, y, z, so for a rigidly rotated cup
    the correction must rotate with it: f_corr = R (K x0), with R from the
    best-fit (Kabsch) rotation of the whole cup, updated every step.
    Remaining approximation: *local* rotations are not corrected, which makes
    strongly curved regions (the bottom corner) slightly over-stiff.
    """

    def __init__(self, model, ids, mesh):
        m = model
        d = mujoco.MjData(m)
        grav = m.opt.gravity.copy()
        flags = m.opt.disableflags
        m.opt.gravity[:] = 0
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mujoco.mj_forward(m, d)                      # qpos0 = as-moulded cup shape
        bodies = m.flex_vertbodyid[ids.vert_adr:ids.vert_adr + ids.nvert]
        self.dofs = np.array([np.arange(m.body_dofadr[b], m.body_dofadr[b] + 3) for b in bodies])
        self.f0 = d.qfrc_passive[self.dofs].copy()   # = -K x0 (membrane force is zero at rest)
        m.opt.gravity[:] = grav
        m.opt.disableflags = flags
        self.X0 = mesh.vertices

    def apply(self, d, X):
        R, _ = kabsch(self.X0, X)
        d.qfrc_applied[self.dofs] = -(self.f0 @ R.T)


class Trial:
    def __init__(self, cfg: Config, cube_mass: float, name: str = None):
        cfg.cube.mass = cube_mass
        self.cfg = cfg
        self.name = name or f"{cube_mass * 1000:.0f}g"
        self.model, self.mesh, self.ids = build(cfg)
        self.data = mujoco.MjData(self.model)
        self.F = grip_force_for(cfg, cube_mass, self.ids.cup_mass)
        self.ctrl = PickController(self.model, self.ids, cfg, self.F)
        self.deform = CupDeformation(self.mesh, cfg)
        self.probe = ContactProbe(self.model, self.ids)
        self.W = (cube_mass + self.ids.cup_mass) * 9.81
        self.rest_fix = RestShapeCorrection(self.model, self.ids, self.mesh) if cfg.cup.rest_shape_correction else None
        self._reset()

    # ------------------------------------------------------------------
    def _reset(self):
        m, d, ids, cfg = self.model, self.data, self.ids, self.cfg
        mujoco.mj_resetData(m, d)
        d.qpos[ids.arm_qadr] = self.ctrl.q_init
        d.qpos[ids.finger_qadr] = self.ctrl.q_open
        d.ctrl[ids.arm_act] = self.ctrl.q_init
        mujoco.mj_forward(m, d)
        self.log = {k: [] for k in self._log_keys()}
        self.frames = {k: [] for k in ("t", "qpos", "qvel", "ctrl", "act", "mocap_pos", "disp", "qacc_warmstart")}
        self.slip = {"detected": False}
        self._slip_ref = None        # z(material points under the pads) - z(TCP) at lift start
        self._slip_ids = None        # cup vertices under the pads (material points)
        self._release_ref = None
        self._last_log = -1e9
        # after the release ramp the fingers keep opening until the cup drops (max 1.5 s)
        self.t_end = self.ctrl.tl.ends[-1] + (1.5 if cfg.task.slip_test else 0.3)
        self.last_def = None
        self.last_con = None
        self.last_mag = np.zeros(ids.nvert)

    @staticmethod
    def _log_keys():
        keys = ["t", "phase", "grip_cmd", "finger_q", "tcp", "tcp_target", "cup_c", "cube_p",
                "bottom_c", "slip", "wall_ms"]
        keys += ["max_disp", "rms_disp", "wall_max_disp", "pad_indent", "width_y", "width_x",
                 "bottom_sag", "bottom_max_w", "bottom_max_disp"]
        for g in GROUPS:
            keys += [f"{g}_fn", f"{g}_ft", f"{g}_f", f"{g}_n", f"{g}_util_max", f"{g}_util_w",
                     f"{g}_vslide"]
        return keys

    def verts(self):
        a = self.ids.vert_adr
        return self.data.flexvert_xpos[a:a + self.ids.nvert]

    # ------------------------------------------------------------------
    def measure(self):
        X = self.verts().copy()
        dfm, mag, R, t = self.deform.compute(X)
        con = self.probe.compute(self.data)
        self.last_def, self.last_con, self.last_mag = dfm, con, mag
        return X, dfm, con, mag

    def _record(self, t, phase, f_cmd, X, dfm, con):
        d, ids, L = self.data, self.ids, self.log
        tcp = d.site_xpos[ids.tcp_site].copy()
        cup_c = X.mean(0)
        slip = self._slip_value(X, tcp)
        L["t"].append(t)
        L["phase"].append(PHASES.index(phase))
        L["grip_cmd"].append(f_cmd)
        L["finger_q"].append(d.qpos[ids.finger_qadr].copy())
        L["tcp"].append(tcp)
        L["tcp_target"].append(self.ctrl.tcp_target(t))
        L["cup_c"].append(cup_c)
        L["cube_p"].append(d.xpos[ids.cube_body].copy())
        L["bottom_c"].append(X[self.mesh.bottom_center_id].copy())
        L["slip"].append(slip)
        L["wall_ms"].append(self._step_ms)
        for k, v in dfm.items():
            if k in L:
                L[k].append(v)
        for g in GROUPS:
            c = con[g]
            L[f"{g}_fn"].append(c["fn"])
            L[f"{g}_ft"].append(c["ft"].copy())
            L[f"{g}_f"].append(c["f"].copy())
            L[f"{g}_n"].append(c["n"])
            L[f"{g}_util_max"].append(c["util_max"])
            L[f"{g}_util_w"].append(c["util_w"])
            L[f"{g}_vslide"].append(c["vslide"])
        return slip

    # ------------------------------------------------------------------
    def run(self, on_frame=None, frame_dt=None, viewer=None, realtime=False, verbose=True):
        """Integrate the whole trial.

        on_frame(trial, t) is called every frame_dt seconds of sim time (state snapshots / live viewer).
        """
        m, d, cfg = self.model, self.data, self.cfg
        h = m.opt.timestep
        log_dt = 1.0 / cfg.sim.log_hz
        next_frame = 0.0
        tl = self.ctrl.tl
        t_wall0 = time.time()
        self._step_ms = 0.0
        last_print = -1
        n = 0
        while d.time < self.t_end:
            t = d.time
            f_cmd, _ = self.ctrl.apply(d, t)
            phase, _ = tl.phase(t)
            X = self.verts()
            if self.rest_fix is not None:
                self.rest_fix.apply(d, X)
            # camera targets follow the cup (mocap bodies have no physics)
            d.mocap_pos[self.ids.cup_target_mocap] = X.mean(0)
            d.mocap_pos[self.ids.pad_target_mocap] = d.site_xpos[self.ids.tcp_site] + np.array([0, 0.03, 0])

            ts = time.perf_counter()
            mujoco.mj_step(m, d)
            self._step_ms = 1e3 * (time.perf_counter() - ts)
            n += 1
            if not np.isfinite(d.qacc).all():
                raise FloatingPointError(f"simulation diverged at t={t:.3f}")

            t = d.time
            if t - self._last_log >= log_dt - 1e-9:
                self._last_log = t
                X, dfm, con, mag = self.measure()
                if self._slip_ref is None and t >= tl.start("lift"):
                    self._init_slip_tracking(X, con)
                slip = self._record(t, phase, f_cmd, X, dfm, con)
                self._check_slip(t, phase, slip, f_cmd, con)
            if frame_dt is not None and t >= next_frame - 1e-9:
                next_frame += frame_dt
                if self.last_def is None:
                    self.measure()
                fr = self.frames
                fr["t"].append(t)
                fr["qpos"].append(d.qpos.copy())
                fr["qvel"].append(d.qvel.copy())
                fr["ctrl"].append(d.ctrl.copy())
                fr["act"].append(d.act.copy())
                fr["mocap_pos"].append(d.mocap_pos.copy())
                fr["qacc_warmstart"].append(d.qacc_warmstart.copy())
                fr["disp"].append(self.last_mag.astype(np.float32))
                if on_frame is not None:
                    on_frame(self, t)
            if viewer is not None:
                if not viewer.is_running():
                    break
                if realtime:
                    lag = d.time - (time.time() - t_wall0)
                    if lag > 0:
                        time.sleep(lag)
            if verbose and int(t) != last_print:
                last_print = int(t)
                el = time.time() - t_wall0
                dfm = self.last_def or {}
                print(f"  [{self.name}] t={t:5.2f}s {phase:11s} wall={el:6.1f}s "
                      f"({t / max(el, 1e-9):.2f}x RT)  max|u|={dfm.get('max_disp', 0) * 1e3:5.2f}mm "
                      f"grip={f_cmd:5.2f}N", flush=True)
            if self.slip.get("stop_at") is not None and t >= self.slip["stop_at"]:
                break
        self.wall_time = time.time() - t_wall0
        return self.finish()

    def _init_slip_tracking(self, X, con):
        """Remember the cup vertices that sit under the finger pads when the lift starts."""
        ids = []
        for g in ("left", "right"):
            if con[g]["fn"] > 1e-6:
                dist = np.linalg.norm(X - con[g]["pos"], axis=1)
                ids += list(np.where(dist < 0.010)[0])
        self._slip_ids = np.array(sorted(set(ids)), dtype=int) if ids else np.arange(len(X))
        tcp = self.data.site_xpos[self.ids.tcp_site]
        self._slip_ref = X[self._slip_ids, 2].mean() - tcp[2]

    def _slip_value(self, X, tcp):
        """Vertical slip of the cup wall relative to the fingers [m] (negative = cup slides down)."""
        if self._slip_ref is None:
            return 0.0
        return (X[self._slip_ids, 2].mean() - tcp[2]) - self._slip_ref

    def _check_slip(self, t, phase, slip, f_cmd, con):
        """Gross slip = wall slides > 2 mm down w.r.t. the pads after the release started.
        Used only to stop the run shortly after the cup drops (analysis: analyze.release_metrics)."""
        if phase not in ("release", "done") or self.slip["detected"]:
            return
        if self._release_ref is None:
            self._release_ref = slip
            self.slip["slip_before_release"] = slip
        if slip - self._release_ref < -0.002:
            self.slip.update(detected=True, t=t, stop_at=t + 0.35)

    def finish(self):
        L = {k: np.asarray(v) for k, v in self.log.items()}
        self.L = L
        self.frames = {k: np.asarray(v) for k, v in self.frames.items()}
        return L

"""Bending-stiffness calibration of MuJoCo's 2-D flex (shell) elasticity.

Verification (validate.py) showed that a MuJoCo 2-D flex plate strip is
~3x more flexible in bending than Kirchhoff plate theory
(D = E t^3 / 12 (1 - nu^2)) at the element sizes used here, and that the
discrepancy grows with nu (it behaves like G t^3 / 12). To keep the cup's
stiffness physical, the Young's modulus passed to MuJoCo is scaled so that a
cantilever strip meshed at the *same element size* as the cup reproduces
plate theory:

    E_model = E * (delta_mujoco / delta_theory)

The membrane stiffness is scaled by the same factor; in-plane strains of the
cup wall are tiny (~1e-4) either way, so this does not affect the results.
Results are cached in models/.calibration_cache.json.
"""
import json
import os

import mujoco
import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                     ".calibration_cache.json")


def cantilever(E, nu, t, L=0.06, w=0.02, h=0.008, P=0.02, steps=4000):
    """Tip deflection of a clamped plate strip under a tip line load P.

    Returns (mujoco, euler_bernoulli_beam, kirchhoff_plate_strip) deflections [m].
    """
    nx, ny = int(round(L / h)) + 1, max(3, int(round(w / h)) + 1)
    xml = f"""<mujoco><option timestep="1e-3" integrator="discrete" gravity="0 0 0"/>
    <worldbody><flexcomp name="p" type="grid" dim="2" count="{nx} {ny} 1" spacing="{h} {w / (ny - 1)} 1"
        radius="{t / 2}" mass="0.005">
      <elasticity young="{E}" poisson="{nu}" thickness="{t}" elastic2d="both" damping="1e-3"/>
      <contact contype="0" conaffinity="0"/>
      <pin gridrange="0 0 1 {ny - 1}"/>
    </flexcomp></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    x0 = d.flexvert_xpos.copy()
    tip = np.where(np.isclose(x0[:, 0], x0[:, 0].max()))[0]
    tip_dofs = [m.body_dofadr[m.flex_vertbodyid[i]] + 2 for i in tip]
    for _ in range(steps):
        d.qfrc_applied[tip_dofs] = -P / len(tip)
        mujoco.mj_step(m, d)
    delta = -(d.flexvert_xpos[tip, 2] - x0[tip, 2]).mean()
    L_eff = x0[:, 0].max() - x0[:, 0].min() - h          # from the clamped (2nd) row
    beam = P * L_eff ** 3 / (3 * E * (w * t ** 3 / 12))
    return delta, beam, beam * (1 - nu ** 2)


def mean_edge_length(mesh):
    t = mesh.triangles
    v = mesh.vertices
    e = np.concatenate([v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 1]], v[t[:, 0]] - v[t[:, 2]]])
    return float(np.linalg.norm(e, axis=1).mean())


def calibrated_young(E, nu, t, h):
    key = f"{E:.6g}|{nu:.4g}|{t:.6g}|{h:.5g}"
    cache = {}
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
    if key in cache:
        return cache[key]["E_model"], cache[key]["ratio"]
    delta, _, plate = cantilever(E, nu, t, h=h)
    ratio = delta / plate
    E_model = E * ratio
    cache[key] = {"E_model": E_model, "ratio": ratio}
    try:
        with open(CACHE, "w") as f:
            json.dump(cache, f, indent=1)
    except OSError:
        pass
    return E_model, ratio

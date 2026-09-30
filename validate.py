#!/usr/bin/env python3
"""Verification of the shell model (run: python validate.py).

1. Cantilever plate  - MuJoCo 2-D flex (same E, nu, thickness, element size as the cup)
                       vs. Euler-Bernoulli / Kirchhoff plate-strip theory.
2. Rest shape        - unloaded cup, zero gravity: does it keep its moulded shape?
                       (MuJoCo's bending assumes a flat rest state -> correction in sim.py)
3. Bottom plate      - analytical bounds (clamped / simply supported circular plate) for the
                       bottom deflection under the cube weight, to compare with the trial results.
Results are printed and written to results/validation.md.
"""
import os

import mujoco
import numpy as np

from cupsim.calibration import cantilever, calibrated_young, mean_edge_length
from cupsim.config import Config
from cupsim.cup_mesh import make_cup_mesh
from cupsim.metrics import kabsch

ROOT = os.path.dirname(os.path.abspath(__file__))


def rest_shape(cfg, correct):
    c = cfg.cup
    mesh = make_cup_mesh(c.r_bottom, c.r_top, c.height, c.n_theta, c.n_wall, c.n_bottom)
    E_model, _ = calibrated_young(c.young, c.poisson, c.thickness, mean_edge_length(mesh))
    pts = " ".join(f"{v:.6g}" for v in mesh.vertices.ravel())
    el = " ".join(map(str, mesh.triangles.ravel()))
    xml = f"""<mujoco><option timestep="1e-3" integrator="discrete" gravity="0 0 0"/>
    <worldbody><flexcomp name="cup" type="direct" dim="2" radius="{c.thickness / 2}" mass="0.013"
      point="{pts}" element="{el}">
      <elasticity young="{E_model}" poisson="{c.poisson}" thickness="{c.thickness}" elastic2d="both"
                  damping="{c.damping}"/>
      <contact contype="0" conaffinity="0"/></flexcomp></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    f0 = d.qfrc_passive.reshape(-1, 3).copy()
    X0 = mesh.vertices
    for _ in range(500):
        if correct:
            R, _ = kabsch(X0, d.flexvert_xpos)
            d.qfrc_applied[:] = -(f0 @ R.T).ravel()
        mujoco.mj_step(m, d)
    R, tr = kabsch(X0, d.flexvert_xpos)
    u = (d.flexvert_xpos - tr) @ R - X0
    return np.linalg.norm(u, axis=1).max(), np.abs(f0).max()


def main():
    cfg = Config()
    c = cfg.cup
    lines = ["# Shell-model verification\n"]
    mesh = make_cup_mesh(c.r_bottom, c.r_top, c.height, c.n_theta, c.n_wall, c.n_bottom)
    h = mean_edge_length(mesh)
    d, beam, plate = cantilever(c.young, c.poisson, c.thickness, h=h)
    E_model, ratio = calibrated_young(c.young, c.poisson, c.thickness, h)
    d2, _, _ = cantilever(E_model, c.poisson, c.thickness, h=h)
    lines += ["## 1. Bending stiffness: cantilever plate strip (60 x 20 mm, element size %.1f mm, "
              "t = %.2f mm, E = %.2f GPa, nu = %.2f)" % (h * 1e3, c.thickness * 1e3, c.young / 1e9, c.poisson),
              f"- Kirchhoff plate strip theory: {plate * 1e3:.3f} mm (Euler-Bernoulli beam: {beam * 1e3:.3f} mm)",
              f"- MuJoCo with the physical E: **{d * 1e3:.3f} mm** -> {d / plate:.2f}x too flexible",
              f"- MuJoCo with the calibrated E_model = {E_model / 1e9:.2f} GPa (x{ratio:.2f}): "
              f"**{d2 * 1e3:.3f} mm** -> ratio {d2 / plate:.3f}",
              "- mesh-convergence and nu-dependence of the raw MuJoCo value: see README (Methodology)\n"]
    u_raw, fmax = rest_shape(cfg, correct=False)
    u_fix, _ = rest_shape(cfg, correct=True)
    lines += ["## 2. Unloaded cup keeps its moulded shape? (0.5 s, no gravity, no contact)",
              f"- max spurious rest force from MuJoCo's flat-rest bending model: {fmax:.1f} N (at the bottom corner)",
              f"- without correction: max drift **{u_raw * 1e3:.3f} mm**",
              f"- with rest-shape correction (sim.py): max drift **{u_fix * 1e3:.4f} mm**\n"]
    # 3. analytical bounds for the bottom plate (for reference; the simulated value is in the trial summaries)
    D = c.young * c.thickness ** 3 / (12 * (1 - c.poisson ** 2))
    a = c.r_bottom
    lines += ["## 3. Bottom plate (radius a = %.0f mm, D = %.4f N m) under the cube weight P" % (a * 1e3, D),
              "The 30 mm cube spreads P over the centre of the base, so the clamped-edge deflection lies between",
              "a load spread uniformly over the whole base, w = P a^2/(64 pi D), and a point load, w = P a^2/(16 pi D).",
              "A simply supported edge would be (5+nu)/(1+nu) resp. (3+nu)/(1+nu) times larger.",
              "",
              "| cube | P [N] | clamped, uniform [mm] | clamped, point [mm] | simply supp., point [mm] |",
              "|---|---|---|---|---|"]
    for m_cube in (0.01, 0.1, 0.5):
        P = m_cube * 9.81
        wc = P * a ** 2 / (16 * np.pi * D)
        wu = wc / 4
        ws = wc * (3 + c.poisson) / (1 + c.poisson)
        lines.append(f"| {m_cube * 1000:.0f} g | {P:.2f} | {wu * 1e3:.3f} | {wc * 1e3:.3f} | {ws * 1e3:.3f} |")
    lines.append("\nCompare with `air_bottom_sag_mm` in results/<trial>/summary.json.")
    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "validation.md"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()

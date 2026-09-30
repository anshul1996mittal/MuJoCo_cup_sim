# Shell-model verification

## 1. Bending stiffness: cantilever plate strip (60 x 20 mm, element size 9.5 mm, t = 0.50 mm, E = 1.50 GPa, nu = 0.42)
- Kirchhoff plate strip theory: 1.884 mm (Euler-Bernoulli beam: 2.287 mm)
- MuJoCo with the physical E: **6.671 mm** -> 3.54x too flexible
- MuJoCo with the calibrated E_model = 5.31 GPa (x3.54): **1.918 mm** -> ratio 1.018
- mesh-convergence and nu-dependence of the raw MuJoCo value: see README (Methodology)

## 2. Unloaded cup keeps its moulded shape? (0.5 s, no gravity, no contact)
- max spurious rest force from MuJoCo's flat-rest bending model: 728.2 N (at the bottom corner)
- without correction: max drift **1.691 mm**
- with rest-shape correction (sim.py): max drift **0.0001 mm**

## 3. Bottom plate (radius a = 25 mm, D = 0.0190 N m) under the cube weight P
The 30 mm cube spreads P over the centre of the base, so the clamped-edge deflection lies between
a load spread uniformly over the whole base, w = P a^2/(64 pi D), and a point load, w = P a^2/(16 pi D).
A simply supported edge would be (5+nu)/(1+nu) resp. (3+nu)/(1+nu) times larger.

| cube | P [N] | clamped, uniform [mm] | clamped, point [mm] | simply supp., point [mm] |
|---|---|---|---|---|
| 10 g | 0.10 | 0.016 | 0.064 | 0.155 |
| 100 g | 0.98 | 0.161 | 0.643 | 1.549 |
| 500 g | 4.91 | 0.804 | 3.215 | 7.743 |

Compare with `air_bottom_sag_mm` in results/<trial>/summary.json.

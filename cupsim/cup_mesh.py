"""Triangle surface mesh of a thin-walled, open-top, truncated-cone cup.

The cup is modelled as a *surface* (mid-surface of the 1 mm plastic sheet):
a flat circular bottom plus a conical side wall, no rim (allowed by the brief).
The mesh is used as a 2-D MuJoCo flex (shell) - see cupsim/scene.py.

Vertex layout (all coordinates in the cup frame, origin = bottom centre, z up):
  0                         bottom centre
  1 .. n_bottom*n_theta     bottom rings (inner -> outer), outermost ring
                            is the bottom edge shared with the wall
  ...                       wall rings, bottom edge -> top edge
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CupMesh:
    vertices: np.ndarray        # (N, 3) float
    triangles: np.ndarray       # (M, 3) int, outward-facing normals
    bottom_ids: np.ndarray      # vertices of the flat bottom (incl. bottom edge ring)
    bottom_center_id: int
    bottom_edge_ids: np.ndarray  # ring where bottom meets wall
    wall_ids: np.ndarray        # vertices of the side wall (excl. bottom edge ring)
    ring_z: np.ndarray          # z of every wall ring (incl. bottom edge ring)
    ring_ids: list              # list of vertex-id arrays, one per wall ring


def make_cup_mesh(r_bottom, r_top, height, n_theta=32, n_wall=10, n_bottom=3):
    """Build the cup mesh.

    r_bottom, r_top, height : cup mid-surface geometry [m]
    n_theta  : vertices around the circumference
    n_wall   : number of wall *segments* along the height
    n_bottom : number of radial rings on the bottom disk (incl. the edge ring)
    """
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    verts = [np.array([0.0, 0.0, 0.0])]
    rings = []

    # bottom rings (radial), last one is the bottom edge
    for i in range(1, n_bottom + 1):
        r = r_bottom * i / n_bottom
        # stagger alternate rings a little -> better triangle quality
        th = theta + (0.5 * (2 * np.pi / n_theta) if (i % 2 == 0) else 0.0)
        ids = np.arange(len(verts), len(verts) + n_theta)
        verts.extend(np.stack([r * np.cos(th), r * np.sin(th), np.zeros(n_theta)], 1))
        rings.append((ids, th))
    n_bottom_verts = len(verts)

    # wall rings (ring 0 of the wall == bottom edge ring)
    wall_rings = [rings[-1]]
    for j in range(1, n_wall + 1):
        z = height * j / n_wall
        r = r_bottom + (r_top - r_bottom) * j / n_wall
        th = theta + (0.5 * (2 * np.pi / n_theta) if ((n_bottom + j) % 2 == 0) else 0.0)
        ids = np.arange(len(verts), len(verts) + n_theta)
        verts.extend(np.stack([r * np.cos(th), r * np.sin(th), np.full(n_theta, z)], 1))
        wall_rings.append((ids, th))

    verts = np.asarray(verts)
    tris = []

    # centre fan (bottom normal points -z => outward)
    ids0 = rings[0][0]
    for k in range(n_theta):
        tris.append([0, ids0[(k + 1) % n_theta], ids0[k]])

    def stitch(inner, outer, flip):
        """Triangulate the band between two rings using angular order."""
        (a_ids, a_th), (b_ids, b_th) = inner, outer
        out = []
        i = j = 0
        # merge-walk around the circle
        a_ang = np.unwrap(np.r_[a_th, a_th[0] + 2 * np.pi])
        b_ang = np.unwrap(np.r_[b_th, b_th[0] + 2 * np.pi])
        # align starting angles
        while b_ang[0] > a_ang[0] + 1e-9:
            b_ang -= 2 * np.pi
        while b_ang[0] + 2 * np.pi / n_theta <= a_ang[0] - 1e-9:
            b_ang += 2 * np.pi
        while i < n_theta or j < n_theta:
            a0, a1 = a_ids[i % n_theta], a_ids[(i + 1) % n_theta]
            b0, b1 = b_ids[j % n_theta], b_ids[(j + 1) % n_theta]
            if j >= n_theta or (i < n_theta and a_ang[i + 1] <= b_ang[j + 1]):
                t = [a0, a1, b0]
                i += 1
            else:
                t = [a0, b1, b0]
                j += 1
            out.append(t[::-1] if flip else t)
        return out

    # bottom bands: normal must point -z
    for k in range(len(rings) - 1):
        tris += stitch(rings[k], rings[k + 1], flip=True)
    # wall bands: normal must point radially outward
    for k in range(len(wall_rings) - 1):
        tris += stitch(wall_rings[k], wall_rings[k + 1], flip=False)

    tris = np.asarray(tris, dtype=int)
    tris = _orient(verts, tris, n_bottom_verts)

    bottom_ids = np.arange(0, n_bottom_verts)
    wall_ids = np.arange(n_bottom_verts, len(verts))
    ring_ids = [r[0] for r in wall_rings]
    ring_z = np.array([verts[r[0][0], 2] for r in wall_rings])
    return CupMesh(verts, tris, bottom_ids, 0, rings[-1][0], wall_ids, ring_z, ring_ids)


def _orient(v, tris, n_bottom_verts):
    """Force every triangle normal to point out of the cup."""
    out = tris.copy()
    for k, (a, b, c) in enumerate(tris):
        n = np.cross(v[b] - v[a], v[c] - v[a])
        cen = (v[a] + v[b] + v[c]) / 3
        if max(a, b, c) < n_bottom_verts and cen[2] < 1e-9:      # bottom
            want = np.array([0, 0, -1.0])
        else:                                                    # wall
            want = np.array([cen[0], cen[1], 0.0])
        if np.dot(n, want) < 0:
            out[k] = [a, c, b]
    return out


if __name__ == "__main__":
    m = make_cup_mesh(0.025, 0.036, 0.085)
    print("verts", len(m.vertices), "tris", len(m.triangles))
    # every edge of a closed-bottom open cup is shared by <=2 triangles
    from collections import Counter
    e = Counter()
    for t in m.triangles:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e[tuple(sorted((a, b)))] += 1
    print("edge use counts", Counter(e.values()))

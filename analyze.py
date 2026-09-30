#!/usr/bin/env python3
"""Plots and summary tables.

  python analyze.py                 # compare all trials in results/ -> results/comparison_*.png, results/summary.md
  python analyze.py results/100g    # (re)plot one trial
  python analyze.py --resummarize   # recompute every summary.json from the saved run.npz, then compare
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cupsim.config import Config
from cupsim.controller import make_timeline

ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE_COLORS = {"squeeze": "#fde2e2", "hold_ground": "#fff1d6", "lift": "#e0f0ff",
                "hold_air": "#e6f7e6", "release": "#efe6fb"}


def load(trial_dir):
    z = np.load(os.path.join(trial_dir, "run.npz"))
    L = {k[4:]: z[k] for k in z.files if k.startswith("log_")}
    F = {k[6:]: z[k] for k in z.files if k.startswith("frame_")}
    cfg = Config.from_json(os.path.join(trial_dir, "config.json"))
    with open(os.path.join(trial_dir, "summary.json")) as f:
        S = json.load(f)
    return L, F, cfg, S


def release_metrics(L, tl, cfg, W):
    """Release test: the grip force is ramped down (and the fingers finally open).

    For a tapered, dented thin cup the weight is NOT carried by friction alone: as the grip
    relaxes the cup settles slightly deeper into the pad dents and the load path moves from
    friction (tangential) to tilted normal forces (form closure by the taper / dents).
    Reported:
      release_friction_share_half_cmd_N  grip command at which friction carries < 50 % of the weight
      release_coulomb_limit_cmd_N        grip command at which the pure-Coulomb capacity 2*mu*Fn < W
      release_max_apparent_mu            max of W / (2 Fn) while the cup is still held
                                         (> mu means geometry helps; rigid cone: mu + tan(taper))
      drop_time_s, drop_finger_opening_mm  gross slip (cup falls > 2 mm w.r.t. the pads)
    """
    t = L["t"]
    if "release" not in tl.names or tl.end("release") <= tl.start("release"):
        return {}
    idx = np.where(t >= tl.start("release"))[0]
    if len(idx) < 3:
        return {}
    s = L["slip"][idx] - L["slip"][idx[0]]
    fn = 0.5 * (L["left_fn"][idx] + L["right_fn"][idx])
    fz = L["left_f"][idx][:, 2] + L["right_f"][idx][:, 2]
    fz_fric = L["left_ft"][idx][:, 2] + L["right_ft"][idx][:, 2]
    cmd = L["grip_cmd"][idx]
    gross = np.where(s < -0.002)[0]
    held = np.arange(gross[0]) if len(gross) else np.arange(len(idx))
    out = {"drop_detected": bool(len(gross))}
    share = np.where(np.abs(fz) > 1e-6, fz_fric / np.where(np.abs(fz) > 1e-6, fz, 1), np.nan)
    half = held[share[held] < 0.5]
    out["release_friction_share_half_cmd_N"] = float(cmd[half[0]]) if len(half) else float("nan")
    lim = held[2 * cfg.cup.mu_finger * fn[held] < W]
    out["release_coulomb_limit_cmd_N"] = float(cmd[lim[0]]) if len(lim) else float("nan")
    ok = held[fn[held] > 1e-3]
    out["release_max_apparent_mu"] = float(np.max(W / (2 * fn[ok]))) if len(ok) else float("nan")
    if len(gross):
        i = idx[gross[0]]
        out["drop_time_s"] = float(t[i])
        out["drop_grip_cmd_N"] = float(L["grip_cmd"][i])
        out["drop_finger_opening_mm"] = float(L["finger_q"][i].mean() * 1e3)
    return out


def summarize(L, cfg, meta):
    """Scalar results of one trial (written to summary.json)."""
    tl = make_timeline(cfg)
    t = L["t"]

    def at(phase_end, key):
        i = min(max(np.searchsorted(t, tl.end(phase_end)) - 1, 0), len(t) - 1)
        return float(np.asarray(L[key])[i])

    def seg(phase, key, fn=np.max):
        sel = (t >= tl.start(phase)) & (t < tl.end(phase))
        return float(fn(np.asarray(L[key])[sel])) if sel.any() else float("nan")

    def slide_dist(p0, p1):
        sel = (t >= tl.start(p0)) & (t < tl.end(p1))
        v = 0.5 * (L["left_vslide"][sel] + L["right_vslide"][sel])
        return float(np.sum(v) / cfg.sim.log_hz)

    def friction_share(phase_end):
        """Fraction of the vertical support that the fingers give through friction."""
        i = min(max(np.searchsorted(t, tl.end(phase_end)) - 1, 0), len(t) - 1)
        fz = L["left_f"][i][2] + L["right_f"][i][2]
        fz_fric = L["left_ft"][i][2] + L["right_ft"][i][2]
        return float(fz_fric / fz) if abs(fz) > 1e-6 else float("nan")

    tcp = L["tcp"]
    lift = (t >= tl.start("lift")) & (t < tl.end("hold_air"))
    xy_dev = np.linalg.norm(tcp[lift][:, :2] - tcp[lift][0, :2], axis=1).max() if lift.any() else 0.0
    cup = cfg.cup
    taper = np.arctan((cup.r_top - cup.r_bottom) / cup.height)
    s = dict(meta)
    s.update({
        "wall_thickness_mm": cup.thickness * 1e3,
        "mu_finger_cup": cup.mu_finger,
        # objective 1 - on the ground (end of hold_ground)
        "ground_normal_force_per_finger_N": 0.5 * (at("hold_ground", "left_fn") + at("hold_ground", "right_fn")),
        "ground_pad_indent_mm": at("hold_ground", "pad_indent") * 1e3,
        "ground_width_y_mm": at("hold_ground", "width_y") * 1e3,
        "ground_width_x_mm": at("hold_ground", "width_x") * 1e3,
        "ground_max_disp_mm": at("hold_ground", "max_disp") * 1e3,
        # objective 2 - lifted (end of hold_air)
        "air_pad_indent_mm": at("hold_air", "pad_indent") * 1e3,
        "air_width_y_mm": at("hold_air", "width_y") * 1e3,
        "air_width_x_mm": at("hold_air", "width_x") * 1e3,
        "air_max_disp_mm": at("hold_air", "max_disp") * 1e3,
        "lift_peak_max_disp_mm": seg("lift", "max_disp") * 1e3,
        "rest_width_mm": float(L["width_y"][0] * 1e3),
        # objective 3 - bottom
        "ground_bottom_sag_mm": at("hold_ground", "bottom_sag") * 1e3,
        "air_bottom_sag_mm": at("hold_air", "bottom_sag") * 1e3,
        "air_bottom_max_w_mm": at("hold_air", "bottom_max_w") * 1e3,
        "lift_peak_bottom_sag_mm": seg("lift", "bottom_sag") * 1e3,
        "air_cube_on_bottom_N": at("hold_air", "cube_fn"),
        # objective 4 - friction
        "air_normal_force_per_finger_N": 0.5 * (at("hold_air", "left_fn") + at("hold_air", "right_fn")),
        "air_friction_util": 0.5 * (at("hold_air", "left_util_w") + at("hold_air", "right_util_w")),
        "lift_peak_friction_util": seg("lift", "left_util_w"),
        "air_friction_share_of_support": friction_share("hold_air"),
        "slip_after_lift_mm": at("hold_air", "slip") * 1e3,
        "sliding_distance_squeeze_mm": slide_dist("squeeze", "hold_ground") * 1e3,
        "sliding_distance_lift_hold_mm": slide_dist("lift", "hold_air") * 1e3,
        "mu_plus_tan_taper": float(cup.mu_finger + np.tan(taper)),
        "tcp_lateral_deviation_during_lift_mm": float(xy_dev) * 1e3,
    })
    r_g = cup.r_bottom + (cup.r_top - cup.r_bottom) * cfg.task.grasp_height / cup.height
    s["rest_radius_at_pads_mm"] = float((r_g + cup.thickness / 2) * 1e3)
    s.update(release_metrics(L, tl, cfg, meta["weight_N"]))
    return s


def shade(ax, tl, t_max):
    for n, a, b in zip(tl.names, tl.starts, tl.ends):
        if n in PHASE_COLORS and a < t_max:
            ax.axvspan(a, min(b, t_max), color=PHASE_COLORS[n], lw=0, zorder=0)


def phase_legend(fig):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=n) for n, c in PHASE_COLORS.items()],
               loc="lower center", ncol=5, fontsize=8, frameon=False)


def plot_trial(trial_dir):
    L, F, cfg, S = load(trial_dir)
    tl = make_timeline(cfg)
    t = L["t"]
    mm = 1e3
    tmax = t[-1]
    name = S["trial"]

    # ---------------- deformation (objectives 1-3)
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax = axs[0]
    ax.plot(t, L["pad_indent"] * mm, label="pad indentation (max inward wall displacement)")
    ax.plot(t, L["wall_max_disp"] * mm, label="max |u| on the wall")
    ax.plot(t, L["rms_disp"] * mm, label="RMS |u| (whole cup)")
    ax.set_ylabel("wall deformation [mm]")
    ax.set_title(f"Cup deformation - cube {name}, grip {S['grip_force_per_finger_N']:.2f} N/finger, "
                 f"wall {S['wall_thickness_mm']:.2f} mm")
    ax = axs[1]
    ax.plot(t, L["width_y"] * mm, label="width along squeeze axis (y)")
    ax.plot(t, L["width_x"] * mm, label="width perpendicular (x)")
    ax.set_ylabel("cross-section at pads [mm]")
    ax = axs[2]
    ax.plot(t, L["bottom_sag"] * mm, label="bottom sag: centre below edge ring")
    ax.plot(t, L["bottom_max_w"] * mm, label="max out-of-plane |u_z| of the bottom")
    ax.set_ylabel("bottom [mm]")
    ax.set_xlabel("time [s]")
    for ax in axs:
        shade(ax, tl, tmax)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    phase_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(trial_dir, "deformation.png"), dpi=120)
    plt.close(fig)

    # ---------------- friction (objective 4)
    fig, axs = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    mu = cfg.cup.mu_finger
    ax = axs[0]
    for g, c in (("left", "C3"), ("right", "C1")):
        ax.plot(t, L[f"{g}_fn"], c=c, label=f"{g} finger: normal force")
        ax.plot(t, L[f"{g}_ft"][:, 2], c=c, ls="--", label=f"{g} finger: vertical friction on cup")
    ax.plot(t, L["grip_cmd"], "k:", label="commanded grip force")
    ax.axhline(S["weight_N"] / 2, c="gray", lw=0.8, ls="-.", label="W/2 (share of weight per finger)")
    ax.set_ylabel("force [N]")
    ax.set_title(f"Friction - cube {name}, mu = {mu}")
    ax = axs[1]
    fz_fric = L["left_ft"][:, 2] + L["right_ft"][:, 2]
    fz_all = L["left_f"][:, 2] + L["right_f"][:, 2]
    ax.plot(t, fz_all / S["weight_N"], c="k", lw=0.9, label="vertical force from fingers / weight")
    ax.plot(t, fz_fric / S["weight_N"], c="C0", label="... carried by friction")
    ax.plot(t, (fz_all - fz_fric) / S["weight_N"], c="C3", label="... carried by normal forces (taper / dents)")
    ax.plot(t, 0.5 * (L["left_util_w"] + L["right_util_w"]), c="C2", lw=0.8,
            label="mean friction utilisation |ft|/(mu fn)")
    ax.plot(t, np.maximum(L["left_util_max"], L["right_util_max"]), lw=0.7,
            label="max over contacts (1 = sliding)")
    ax.set_ylim(-0.3, 1.4)
    ax.set_ylabel("fraction of weight / utilisation")
    ax = axs[2]
    ax.plot(t, L["left_n"] + L["right_n"], label="# finger-cup contact points")
    ax2 = ax.twinx()
    ax2.plot(t, L["pad_indent"] * mm, c="C3", lw=0.8, label="pad indentation [mm]")
    ax2.set_ylabel("indentation [mm]", color="C3")
    ax.set_ylabel("contacts")
    ax = axs[3]
    ax.plot(t, L["slip"] * mm, label="vertical displacement of wall under pads relative to TCP")
    vs = 0.5 * (L["left_vslide"] + L["right_vslide"])
    ax.plot(t, np.cumsum(vs) / cfg.sim.log_hz * mm, label="cumulative sliding distance at pad contacts")
    ax.plot(t, (L["cup_c"][:, 2] - L["cup_c"][0, 2]) * mm / 10, lw=0.8, label="cup height / 10")
    ax.set_ylabel("[mm]")
    ax.set_xlabel("time [s]")
    for ax in axs:
        shade(ax, tl, tmax)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    phase_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(trial_dir, "friction.png"), dpi=120)
    plt.close(fig)

    # ---------------- friction vs deformation during squeeze + release (the coupling)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    sq = (t >= tl.start("squeeze")) & (t < tl.end("hold_ground"))
    rel = (t >= tl.start("release"))
    fn = 0.5 * (L["left_fn"] + L["right_fn"])
    axs[0].plot(fn[sq], L["pad_indent"][sq] * mm, ".", ms=3, label="squeeze (on ground)")
    axs[0].plot(fn[rel], L["pad_indent"][rel] * mm, ".", ms=3, label="release (in the air)")
    axs[0].set_xlabel("normal force per finger [N]")
    axs[0].set_ylabel("pad indentation [mm]")
    axs[0].set_title("wall stiffness under the pads")
    fz = L["left_f"][:, 2] + L["right_f"][:, 2]
    fzf = L["left_ft"][:, 2] + L["right_ft"][:, 2]
    held = rel & (t < S.get("drop_time_s", np.inf))
    lift = (t >= tl.start("lift")) & (t < tl.end("hold_air"))
    for sel, lab, c in ((lift, "lift + hold", "C0"), (held, "release (until drop)", "C3")):
        ok = sel & (np.abs(fz) > 0.02 * S["weight_N"])
        axs[1].plot(L["pad_indent"][ok] * mm, fzf[ok] / fz[ok], ".", ms=3, c=c, label=lab)
    axs[1].axhline(1.0, c="k", ls="--", lw=0.8)
    axs[1].set_ylim(-0.2, 1.3)
    axs[1].set_xlabel("pad indentation [mm]")
    axs[1].set_ylabel("share of vertical support carried by friction")
    axs[1].set_title("load path: friction vs. geometry (normal forces)")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(trial_dir, "friction_vs_deformation.png"), dpi=120)
    plt.close(fig)


def compare(results_dir):
    trials = []
    for d in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, d)
        if os.path.isfile(os.path.join(p, "summary.json")):
            trials.append(p)
    if not trials:
        print("no trials found in", results_dir)
        return
    data = [load(p) for p in trials]
    data.sort(key=lambda x: x[3]["cube_mass_kg"])
    mm = 1e3

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for L, F, cfg, S in data:
        tl = make_timeline(cfg)
        t = L["t"]
        lab = f"{S['trial']} (grip {S['grip_force_per_finger_N']:.1f} N)"
        axs[0, 0].plot(t, L["pad_indent"] * mm, label=lab)
        axs[0, 1].plot(t, L["bottom_sag"] * mm, label=lab)
        axs[1, 0].plot(t, 0.5 * (L["left_fn"] + L["right_fn"]), label=lab)
        axs[1, 1].plot(t, L["slip"] * mm, label=lab)
    for ax, yl in zip(axs.ravel(), ["pad indentation [mm]", "bottom sag [mm]",
                                    "normal force per finger [N]", "slip wall vs pads [mm]"]):
        shade(ax, tl, t[-1])
        ax.set_ylabel(yl)
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Comparison of the three trials")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "comparison_timeseries.png"), dpi=120)
    plt.close(fig)

    keys = [
        ("cube_mass_kg", "cube mass [kg]", 1),
        ("weight_N", "weight cup+cube [N]", 1),
        ("grip_force_per_finger_N", "commanded grip [N/finger]", 1),
        ("ground_normal_force_per_finger_N", "Obj1: normal force on ground [N/finger]", 1),
        ("ground_pad_indent_mm", "Obj1: pad indentation on ground [mm]", 1),
        ("ground_width_y_mm", "Obj1: width along squeeze [mm]", 1),
        ("ground_width_x_mm", "Obj1: width across [mm]", 1),
        ("air_pad_indent_mm", "Obj2: pad indentation lifted [mm]", 1),
        ("lift_peak_max_disp_mm", "Obj2: peak max |u| during lift [mm]", 1),
        ("air_width_y_mm", "Obj2: width along squeeze, lifted [mm]", 1),
        ("ground_bottom_sag_mm", "Obj3: bottom sag on ground [mm]", 1),
        ("air_bottom_sag_mm", "Obj3: bottom sag lifted [mm]", 1),
        ("air_bottom_max_w_mm", "Obj3: max bottom deflection lifted [mm]", 1),
        ("air_normal_force_per_finger_N", "Obj4: normal force lifted [N/finger]", 1),
        ("air_friction_util", "Obj4: friction utilisation lifted", 1),
        ("air_friction_share_of_support", "Obj4: share of weight carried by friction (lifted)", 1),
        ("lift_peak_friction_util", "Obj4: peak utilisation during lift", 1),
        ("slip_after_lift_mm", "Obj4: wall displacement vs pads after lift [mm]", 1),
        ("sliding_distance_squeeze_mm", "Obj4: contact sliding during squeeze [mm]", 1),
        ("sliding_distance_lift_hold_mm", "Obj4: contact sliding during lift+hold [mm]", 1),
        ("release_coulomb_limit_cmd_N", "Obj4 release: grip cmd where 2*mu*Fn < W [N]", 1),
        ("release_friction_share_half_cmd_N", "Obj4 release: grip cmd where friction share < 50% [N]", 1),
        ("release_max_apparent_mu", "Obj4 release: max apparent mu W/(2Fn) while held", 1),
        ("mu_plus_tan_taper", "Obj4: rigid-cone prediction mu + tan(taper)", 1),
        ("drop_finger_opening_mm", "Obj4: finger opening when the cup dropped [mm]", 1),
        ("rest_radius_at_pads_mm", "cup outer radius at pad height (undeformed) [mm]", 1),
        ("tcp_lateral_deviation_during_lift_mm", "TCP sideways deviation during lift [mm]", 1),
    ]
    rows = ["| quantity | " + " | ".join(S["trial"] for *_, S in data) + " |",
            "|---|" + "---|" * len(data)]
    for k, lab, _ in keys:
        vals = []
        for *_, S in data:
            v = S.get(k)
            vals.append("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        rows.append(f"| {lab} | " + " | ".join(vals) + " |")
    table = "\n".join(rows)
    with open(os.path.join(results_dir, "summary.md"), "w") as f:
        f.write("# Results summary\n\n" + table + "\n")
    print(table)

    # bar chart of the key numbers
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    names = [S["trial"] for *_, S in data]
    for ax, (k, lab) in zip(axs, [("ground_pad_indent_mm", "Obj1 pad indentation\non ground [mm]"),
                                  ("air_pad_indent_mm", "Obj2 pad indentation\nlifted [mm]"),
                                  ("air_bottom_sag_mm", "Obj3 bottom sag\nlifted [mm]"),
                                  ("slip_mu_effective", "Obj4 effective mu\nat gross slip")]):
        vals = [S.get(k) or 0 for *_, S in data]
        ax.bar(names, vals, color=["C0", "C1", "C3"][:len(vals)])
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "comparison_bars.png"), dpi=120)
    plt.close(fig)


def resummarize(results_dir):
    for d in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, d)
        if os.path.isfile(os.path.join(p, "run.npz")):
            L, F, cfg, S = load(p)
            keep = ("trial", "cube_mass_kg", "cup_mass_kg", "weight_N", "grip_force_per_finger_N",
                    "sim_wall_time_s", "gross_slip_detected")
            S = summarize(L, cfg, {k: S[k] for k in keep if k in S})
            with open(os.path.join(p, "summary.json"), "w") as f:
                json.dump(S, f, indent=2)
            plot_trial(p)
            print("re-summarised", p)


if __name__ == "__main__":
    if "--resummarize" in sys.argv:
        resummarize(os.path.join(ROOT, "results"))
        compare(os.path.join(ROOT, "results"))
    elif len(sys.argv) > 1:
        for p in sys.argv[1:]:
            plot_trial(p)
            print("plotted", p)
    else:
        compare(os.path.join(ROOT, "results"))

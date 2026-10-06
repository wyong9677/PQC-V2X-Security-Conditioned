#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "04_results"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

PROFILES = ("diagnostic_fast", "diagnostic_nominal", "diagnostic_stressed")
PROFILE_LABELS = {
    "diagnostic_fast": "Fast",
    "diagnostic_nominal": "Nominal",
    "diagnostic_stressed": "Stressed",
}

D7 = RESULTS / "P3B1_R7D7_LATEST.json"
D7B = RESULTS / "P3B1_R7D7B_LATEST.json"
D8 = RESULTS / "P3B1_R7D8_CERTIFICATE_LATEST.json"
D9 = RESULTS / "P3B1_R7D9_CERTIFICATE_LATEST.json"
D10 = RESULTS / "P3B1_R7D10_LATEST.json"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest(pattern: str) -> Path:
    files = [Path(x) for x in glob.glob(str(RESULTS / pattern))]
    if not files:
        raise FileNotFoundError(pattern)
    return max(files, key=lambda p: p.stat().st_mtime)


def frac(obj: dict) -> Fraction:
    return Fraction(int(obj["numerator"]), int(obj["denominator"]))


def save(fig, stem: str):
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


d7 = load_json(D7)
d7b = load_json(D7B)
d8 = load_json(D8)
d9 = load_json(D9)
d10 = load_json(D10)

d7_candidate_csv = Path(d7["artifacts"]["candidate_csv"])
d7b_witness_csv = Path(d7b["artifacts"]["witness_csv"])
d6c_unique_csv = Path(load_json(RESULTS / "P3B1_R7D6C_LATEST.json")["artifacts"]["unique_witness_csv"])
d9_threshold_csv = Path(d9["source_files"]["threshold_witness_csv"]["path"])

d7_rows = read_csv(d7_candidate_csv)
d7b_rows = read_csv(d7b_witness_csv)
d6c_unique_rows = read_csv(d6c_unique_csv)
d9_rows = read_csv(d9_threshold_csv)

# ------------------------------------------------------------
# Fig. 1: Numerical evidence chain
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.2, 3.2))
ax.axis("off")

stages = [
    ("Coarse grid", "0 kernel-q nodes\nswitch-only q effect"),
    ("D6C", "16 distinct\nopen-neighborhood witnesses"),
    ("D7", "16/16 survive\ntrue fixed-point solve"),
    ("D7B", "16/16 survive\ntwo nested refinements"),
    ("D8", "min q-span lower bound\n1.8683×10⁻³ m"),
    ("D9", "min membership margin\n9.3415×10⁻⁴ m"),
]

xs = np.linspace(0.07, 0.93, len(stages))
for i, (title, text) in enumerate(stages):
    ax.text(
        xs[i], 0.58, f"{title}\n{text}",
        ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=1.0),
        transform=ax.transAxes,
    )
    if i < len(stages) - 1:
        ax.annotate(
            "", xy=(xs[i+1]-0.07, 0.58), xytext=(xs[i]+0.07, 0.58),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )
ax.text(
    0.5, 0.08,
    "Refinement reveals a thin q-dependent viability layer that is invisible on the original coarse grid.",
    ha="center", va="center", fontsize=10, transform=ax.transAxes,
)
save(fig, "Fig1_numerical_evidence_chain")

# ------------------------------------------------------------
# Fig. 2: q-dependent node counts across resolution levels
# ------------------------------------------------------------
d7_profile_csv = Path(d7["artifacts"]["profile_csv"])
d7b_profile_csv = Path(d7b["artifacts"]["profile_csv"])
d7_profile_rows = read_csv(d7_profile_csv)
d7b_profile_rows = read_csv(d7b_profile_csv)

d7_map = {r["profile"]: int(r["refined_eval_q_dependent_nodes"]) for r in d7_profile_rows if r["profile"] in PROFILES}
a_map = {r["profile"]: int(r["refined_eval_q_dependent_nodes"]) for r in d7b_profile_rows if r.get("refinement_level") == "A" and r["profile"] in PROFILES}
b_map = {r["profile"]: int(r["refined_eval_q_dependent_nodes"]) for r in d7b_profile_rows if r.get("refinement_level") == "B" and r["profile"] in PROFILES}

levels = ["Coarse", "D7 refined", "Level A", "Level B"]
x = np.arange(len(levels))
width = 0.24

fig, ax = plt.subplots(figsize=(8.4, 5.0))
for j, p in enumerate(PROFILES):
    vals = [0, d7_map[p], a_map[p], b_map[p]]
    ax.bar(x + (j-1)*width, vals, width, label=PROFILE_LABELS[p])
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel("Number of q-dependent refined-grid nodes")
ax.set_xlabel("Resolution / refinement level")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
save(fig, "Fig2_qdependent_node_counts")

# ------------------------------------------------------------
# Fig. 3: nested center-span stability (16 witnesses)
# ------------------------------------------------------------
ids = [int(r["unique_witness_id"]) for r in d7b_rows]
order = np.argsort(ids)
ids_sorted = [ids[i] for i in order]

fig, ax = plt.subplots(figsize=(10.2, 5.3))
markers = {"D7": "o", "Level A": "s", "Level B": "^"}

# To avoid unreadable duplication, show the three profiles with small x offsets;
# their curves should coincide numerically if the audit is reproduced.
for pi, p in enumerate(PROFILES):
    offs = (pi - 1) * 0.05
    d7_vals = np.array([float(d7b_rows[i][f"{p}_d7_center_span_m"]) for i in order])
    a_vals = np.array([float(d7b_rows[i][f"{p}_level_a_center_span_m"]) for i in order])
    b_vals = np.array([float(d7b_rows[i][f"{p}_level_b_center_span_m"]) for i in order])
    xx = np.arange(len(ids_sorted)) + offs
    ax.plot(xx, d7_vals, marker=markers["D7"], lw=1.0, ms=4, label=f"{PROFILE_LABELS[p]} – D7")
    ax.plot(xx, a_vals, marker=markers["Level A"], lw=1.0, ms=4, label=f"{PROFILE_LABELS[p]} – A")
    ax.plot(xx, b_vals, marker=markers["Level B"], lw=1.0, ms=4, label=f"{PROFILE_LABELS[p]} – B")
ax.set_xticks(np.arange(len(ids_sorted)))
ax.set_xticklabels(ids_sorted, rotation=0)
ax.set_xlabel("Witness ID")
ax.set_ylabel(r"Center q-span $\Delta_q^\star$ (m)")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, ncol=3, fontsize=8)
save(fig, "Fig3_nested_refinement_center_span_stability")

# ------------------------------------------------------------
# Fig. 4: D8 certified q-span lower bounds
# ------------------------------------------------------------
d8_points = []
for w in d8["witness_certificates"]:
    uid = int(w["unique_witness_id"])
    for p in PROFILES:
        pc = w["profile_certificates"][p]
        lb = float(pc["certified_lower_bound_decimal_m"])
        d8_points.append((uid, p, lb))

fig, ax = plt.subplots(figsize=(10.2, 5.1))
for pi, p in enumerate(PROFILES):
    subset = sorted((uid, lb) for uid, pp, lb in d8_points if pp == p)
    xx = np.array([u for u, _ in subset], dtype=float) + (pi-1)*0.08
    yy = np.array([v for _, v in subset], dtype=float)
    ax.scatter(xx, yy, s=34, label=PROFILE_LABELS[p])
min_lb = float(d8["aggregate"]["minimum_certified_lower_bound_decimal_m"])
ax.axhline(min_lb, ls="--", lw=1.1, label=f"Global minimum = {min_lb:.4e} m")
ax.set_xlabel("Witness ID")
ax.set_ylabel("Certified q-span lower bound (m)")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)
save(fig, "Fig4_rational_outward_qspan_certificate")

# ------------------------------------------------------------
# Fig. 5: D9 kernel set separation
# Plot diagnostic-fast for clarity; all three profiles are independently certified.
# ------------------------------------------------------------
rows_sorted = sorted(d9_rows, key=lambda r: int(r["unique_witness_id"]))
xx = np.arange(len(rows_sorted))
h_low = np.array([float(r["diagnostic_fast_h_viable_m"]) for r in rows_sorted])
mid = np.array([float(r["diagnostic_fast_midpoint_d_m"]) for r in rows_sorted])
h_high = np.array([float(r["diagnostic_fast_h_nonviable_m"]) for r in rows_sorted])
marg = np.array([float(r["diagnostic_fast_certified_membership_margin_m"]) for r in rows_sorted])

# Shift each witness by midpoint to focus on set separation geometry.
low_rel = h_low - mid
high_rel = h_high - mid

fig, ax = plt.subplots(figsize=(10.2, 5.4))
for i in range(len(xx)):
    ax.plot([xx[i], xx[i]], [low_rel[i], high_rel[i]], lw=1.2)
ax.scatter(xx, low_rel, marker="o", s=34, label=r"$h^\star(y,q_-)-d^\dagger$")
ax.scatter(xx, np.zeros_like(xx), marker="s", s=30, label=r"$d^\dagger-d^\dagger$")
ax.scatter(xx, high_rel, marker="^", s=34, label=r"$h^\star(y,q_+)-d^\dagger$")
ax.fill_between(xx, -marg, marg, alpha=0.12, label="Certified minimum membership margin")
ax.axhline(0.0, lw=1.0)
ax.set_xticks(xx)
ax.set_xticklabels([int(r["unique_witness_id"]) for r in rows_sorted])
ax.set_xlabel("Witness ID")
ax.set_ylabel("Threshold relative to midpoint distance (m)")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, fontsize=8)
save(fig, "Fig5_kernel_membership_set_separation")

# ------------------------------------------------------------
# Fig. 6: Witness concentration on six bar_a coordinates
# ------------------------------------------------------------
bar_a_vals = [float(r["bar_a"]) for r in d6c_unique_rows]
# Robust grouping of round-trip decimals.
rounded = [round(v, 12) for v in bar_a_vals]
counts = Counter(rounded)
coords = sorted(counts)
vals = [counts[c] for c in coords]

fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.bar(np.arange(len(coords)), vals)
ax.set_xticks(np.arange(len(coords)))
ax.set_xticklabels([f"{c:.6g}" for c in coords], rotation=25, ha="right")
ax.set_xlabel(r"Refined $\bar a$ coordinate")
ax.set_ylabel("Number of certified witnesses")
ax.grid(axis="y", alpha=0.25)
save(fig, "Fig6_witness_distribution_bar_a")

# ------------------------------------------------------------
# Write a compact data summary for manuscript traceability
# ------------------------------------------------------------
summary = FIGDIR / "FIGURE_DATA_SUMMARY.txt"
summary.write_text(
    "\n".join([
        "PQC-V2X figure generation summary",
        f"D7 witnesses: {d7['aggregate']['candidate_rows_recomputed']}",
        f"D7B nested strong witnesses: {d7b['aggregate']['all_nested_points_strong_witnesses']}",
        f"D8 minimum certified q-span lower bound [m]: {d8['aggregate']['minimum_certified_lower_bound_decimal_m']}",
        f"D9 minimum certified membership margin [m]: {d9['aggregate']['minimum_certified_membership_margin_m']}",
        f"Unique bar_a witness coordinates: {len(coords)}",
        "",
        "Generated files:",
        "Fig1_numerical_evidence_chain.{pdf,png}",
        "Fig2_qdependent_node_counts.{pdf,png}",
        "Fig3_nested_refinement_center_span_stability.{pdf,png}",
        "Fig4_rational_outward_qspan_certificate.{pdf,png}",
        "Fig5_kernel_membership_set_separation.{pdf,png}",
        "Fig6_witness_distribution_bar_a.{pdf,png}",
    ]) + "\n",
    encoding="utf-8",
)

print("FIGURE_OUTPUT_DIR=" + str(FIGDIR))
for p in sorted(FIGDIR.glob("Fig*")):
    print(p)
print("FIGURE_GENERATION=PASS")

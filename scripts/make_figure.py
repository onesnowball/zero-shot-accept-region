"""
Draws the appendix figure (figures/fig_tspulse.pdf) from
results/tspulse_jacobian_rows.json, and checks its diagonal against the Table 1
all-observed value in results/checkpoints.json.

Run: python scripts/make_figure.py (or run_all.sh).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "results" / "tspulse_jacobian_rows.json").read_text())
ckpt = json.loads((HERE.parent / "results" / "checkpoints.json").read_text())

# the figure's diagonal must be the Table 1 measurement
assert abs(d["all_observed"]["diag_median"]
           - ckpt["TSPulse imputation, all-observed"]["median"]) < 1e-6
assert d["patch_masked"]["diag_max"] == 0.0

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fontset": "stix"})
k = np.asarray(d["offsets"])
p = d["patch_length"]
PANELS = [("all_observed", "(a) all-observed mask"),
          ("patch_masked", "(b) evaluated patch masked")]
ymax = max(max(d[n]["p75"]) for n, _ in PANELS) * 1.12

fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.45), sharey=True)
for ax, (name, title) in zip(axes, PANELS):
    r = d[name]
    med, lo, hi = map(np.asarray, (r["median"], r["p25"], r["p75"]))
    ax.fill_between(k, lo, hi, color="0.85", lw=0, step="mid")
    ax.step(k, med, where="mid", color="0.25", lw=0.9)
    c0 = "#b2182b"
    ax.plot([0, 0], [0, med[k == 0][0]], color=c0, lw=1.6)
    ax.plot(0, med[k == 0][0], "o", color=c0, ms=3.5, zorder=5)
    lab = (f"$t$: {r['diag_median']:.3f}" if r["diag_median"] > 0
           else "$t$: exactly 0")
    ax.annotate(lab, (0, med[k == 0][0]), xytext=(5, 9),
                textcoords="offset points", color=c0, fontsize=7.5)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlim(k[0], k[-1])
    ax.set_ylim(0, ymax)
    ax.set_xlabel("input offset $t'-t$")
axes[0].set_ylabel(r"$|\partial\hat{x}[t]/\partial x[t']|$")
fig.tight_layout(pad=0.3, w_pad=1.0)
(_FIG:=HERE.parent/"figures").mkdir(parents=True, exist_ok=True); fig.savefig(_FIG / "fig_tspulse.pdf")
print("wrote", HERE.parent / "figures" / "fig_tspulse.pdf")

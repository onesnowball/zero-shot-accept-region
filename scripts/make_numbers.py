"""
Generate every data-derived number in main.tex from the record files.

Writes numbers.tex (macros) and tab_scores.tex (Table 2 body). main.tex
\\input{}s both, so no number in the paper is transcribed by hand.

Records:
    results/timesfm_falsification.json          seed 0   (timesfm_falsify.py)
    results/timesfm_falsification_seed{11..44}   seeds    (BG_SEED=... timesfm_falsify.py)
    results/partition.json                       anchors  (partition.py)
    results/checkpoints.json                     TSPulse  (checkpoints.py)

Run:  python3 basisgate/paper/make_numbers.py
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

SEED_FILES = ["timesfm_falsification.json"] + [
    f"timesfm_falsification_seed{s}.json" for s in (11, 22, 33, 44)]
runs = [json.loads((RES / f).read_text()) for f in SEED_FILES]
part = json.loads((RES / "partition.json").read_text())
ckpt = json.loads((RES / "checkpoints.json").read_text())

macros = {}


def sig(v, n=3):
    """n significant figures, no exponent, integers kept whole above 10^n."""
    if v == 0:
        return "0"
    if abs(v) >= 10 ** n:
        return f"{v:.0f}"
    d = n - 1 - int(math.floor(math.log10(abs(v))))
    return f"{v:.{max(d, 0)}f}"


def sci(v):
    """a x 10^b in LaTeX, one significant figure on the mantissa."""
    e = int(math.floor(math.log10(v)))
    m = v / 10 ** e
    if round(m) == 10:
        m, e = 1, e + 1
    return f"{round(m):d}\\!\\times\\!10^{{{e}}}"


def rng(vals, fmt):
    lo, hi = min(vals), max(vals)
    return f"{fmt(lo)}--{fmt(hi)}"


# --- consistency: every seed uses the same detector setup ------------------
for r in runs:
    assert r["channels"] == runs[0]["channels"] == 4
    assert r["context"] == runs[0]["context"] == 256
    assert abs(r["tau"] - runs[0]["tau"]) < 1e-12
n_seeds = len(runs)
macros["nSeeds"] = str(n_seeds)
macros["tauVal"] = f"{runs[0]['tau']:.2f}"
macros["ctx"] = str(runs[0]["context"])
macros["nSustain"] = str(runs[0]["drift"]["n"])

# --- Table 2 ----------------------------------------------------------------
ROWS = [
    ("clean", "clean"),
    ("naive_gaussian_ch0", "naive: matched Gaussian, ch 0"),
    ("naive_replay_ch0", "naive: replayed segment, ch 0"),
    ("naive_hold_ch0", "naive: constant hold, ch 0"),
    ("anomaly_bias_ch0", "bias injection $3\\sigma$, ch 0"),
    None,
    ("forgery_ch0", "precomputed substitution, ch 0 only"),
    "__label__demonstration (zero by construction)",
    ("forgery_all_chan", "precomputed substitution, all channels"),
    ("forgery_sustained", "precomputed substitution, %s-step rollout"
     % runs[0]["drift"]["n"]),
]
lines = []
for row in ROWS:
    if row is None:
        lines.append("\\midrule")
        continue
    if isinstance(row, str) and row.startswith("__label__"):
        lines.append("\\multicolumn{4}{l}{\\emph{%s}} \\\\"
                     % row[len("__label__"):])
        continue
    key, label = row
    c = [r["conditions"][key] for r in runs]
    ns = {x["n"] for x in c}
    assert len(ns) == 1, key
    means, maxs = [x["mean"] for x in c], [x["max"] for x in c]
    alarms = [100 * x["alarm_rate"] for x in c]
    if max(maxs) == 0:
        m_s, x_s = "$0$", "$0$"
    else:
        m_s, x_s = rng(means, sig), rng(maxs, sig)
    a_s = rng(alarms, lambda v: f"{v:.1f}") if max(alarms) > 0 else "0.0"
    lines.append(f"{label} ({ns.pop()}) & {m_s} & {x_s} & {a_s}\\% \\\\")
(RES / "tab_scores.tex").write_text("\n".join(
    ["\\begin{tabular}{lrrr}", "\\toprule",
     "Condition ($n$ per seed) & mean $g_s$ & max $g_s$ & alarm \\\\",
     "\\midrule"] + lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")

alarm = lambda k: [100 * r["conditions"][k]["alarm_rate"] for r in runs]
macros["alarmGauss"] = rng(alarm("naive_gaussian_ch0"), lambda v: f"{v:.1f}")
macros["alarmReplay"] = rng(alarm("naive_replay_ch0"), lambda v: f"{v:.1f}")

# --- innovation correlation, off-diagonals over all seeds -------------------
off = np.concatenate([np.asarray(r["innovation_corr"])[~np.eye(4, dtype=bool)]
                      for r in runs])
macros["corrLo"] = f"{off.min():.2f}"
macros["corrHi"] = f"+{off.max():.2f}"

# --- drift under sustained forgery ------------------------------------------
full = [r["drift"]["at_100pct_synthetic"] for r in runs]
final = [r["drift"]["final"] for r in runs]
scale = [r["clean_state_scale"] for r in runs]
macros["driftFull"] = f"${sci(min(full))}$--${sci(max(full))}$"
macros["driftFinal"] = f"${sci(min(final))}$--${sci(max(final))}$"
macros["cleanScale"] = f"{np.mean(scale):.1f}"
ratio_final = [f / s for f, s in zip(final, scale)]
macros["ordersMin"] = {4: "four", 5: "five", 3: "three"}[
    int(math.floor(math.log10(min(ratio_final))))]
macros["ordersMax"] = {14: "fourteen", 13: "thirteen", 12: "twelve"}[
    int(math.floor(math.log10(max(ratio_final))))]
# divergence is not monotone on every seed: report how many peak before the end
macros["nNonMonotone"] = str(sum(r["drift"]["max"] > r["drift"]["final"]
                                 for r in runs))

# --- Table 1 ----------------------------------------------------------------
rows = {r["name"]: r for r in part["rows"]}
pca = rows["PCA (analytic control)"]
macros["pcaJ"] = f"{pca['J_median']:.4f}"
macros["pcaErr"] = f"{pca['analytic_max_abs_err'] / 1e-10:.1f}\\times10^{{-10}}"
macros["identJ"] = f"{rows['identity shortcut']['J_median']:.3f}"
causal = rows["causal conv (forecaster-style)"]
assert causal["J_max"] == 0.0
macros["nPart"] = str(part["n_positions"])

tsp_open = ckpt["TSPulse, all-observed (R1's reading)"]
tsp_mask = ckpt["TSPulse, evaluated patch masked (R2's reading)"]
assert tsp_mask["max"] == 0.0
assert tsp_open["n"] == tsp_mask["n"]
macros["tspJ"] = f"{tsp_open['median']:.3f}"
macros["nTsp"] = str(tsp_open["n"])

# --- write -------------------------------------------------------------------
out = ["% GENERATED by make_numbers.py from basisgate/results. Do not edit."]
out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
(RES / "numbers.tex").write_text("\n".join(out) + "\n")
for k, v in macros.items():
    print(f"  {k:<14} {v}")
print("\n" + "\n".join(lines))

"""
Section 3 iteration cost: how many passes reach a zero residual when the estimate
reads its own input. For the TSPulse imputation all-observed call, at each of 64
positions it iterates x <- xhat(x) (fixed point, up to 30 passes) and Newton (up
to 10 steps), from three starts (clean, +3 sigma, +10 sigma).

Input: the synthetic window from synthetic.py (seed 20260822).
Writes results/check1_fixed_point.json. Run: python scripts/sec3_fixed_point.py (or run_all.sh).
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from synthetic import make_window, SEED, N_CH, N_PROBE  # noqa: E402  (Table 1's own)

TOL = 1e-5
FP_PASSES, NEWTON_STEPS = 30, 10
REPORT_AT = (1, 2, 5, 10, 20, 30)
REPO, REV = "ibm-granite/granite-timeseries-tspulse-r1", "b12164578f7b893ada0028c00d292ba10383d25a"


def main():
    torch.manual_seed(SEED)
    from tsfm_public.models.tspulse import TSPulseForReconstruction
    m = TSPulseForReconstruction.from_pretrained(
        REPO, revision=REV, num_input_channels=N_CH, mask_type="user")
    m.eval()
    ctx = m.config.context_length
    x0 = torch.tensor(make_window(ctx), dtype=torch.float32)          # [L, C]
    rng = np.random.default_rng(SEED)                                  # shared probe seed
    ts = rng.choice(np.arange(32, ctx - 32), size=N_PROBE, replace=False)
    pos = [(int(t), int(c)) for t in ts for c in range(N_CH)]
    P = len(pos)
    T = torch.tensor([p[0] for p in pos]); C = torch.tensor([p[1] for p in pos])
    ar = torch.arange(P)
    sigma = x0.std(dim=0)[C]                                           # per position
    clean_v = x0[T, C]

    def xhat_full(batch):                                              # [P, L, C] -> same
        o = m(past_values=batch, past_observed_mask=torch.ones_like(batch),
              return_loss=False)
        return o.reconstruction_outputs

    def build(v):                                                      # one window per position
        b = x0.unsqueeze(0).repeat(P, 1, 1).clone()
        b[ar, T, C] = v
        return b

    def f(v):                                                          # xhat at own position
        with torch.no_grad():
            return xhat_full(build(v))[ar, T, C]

    def f_and_d(v):
        b = build(v).requires_grad_(True)
        out = xhat_full(b)[ar, T, C]
        g = torch.autograd.grad(out.sum(), b)[0]
        return out.detach(), g[ar, T, C].detach()

    with torch.no_grad():
        clean_field = x0.unsqueeze(0) - xhat_full(x0.unsqueeze(0))     # [1, L, C]
    _, d_clean = f_and_d(clean_v)
    print(f"positions {P} | median d at clean point {d_clean.abs().median():.4f} "
          f"(Table 1: 0.274)", flush=True)

    def other_change(v):
        """Largest change in residual at any OTHER position, vs the clean run."""
        with torch.no_grad():
            b = build(v)
            field = b - xhat_full(b)
        diff = (field - clean_field).abs()
        diff[ar, T, C] = 0.0
        return diff.reshape(P, -1).max(dim=1).values

    def classify(traj):
        a = np.abs(traj)
        if a[-1] > 10 * a[0] or not np.isfinite(a[-1]):
            return "diverged"
        signs = np.sign(traj[-8:])
        if (np.diff(signs) != 0).sum() >= 4 and a[-1] > 0.5 * np.median(a[-8:]):
            return "oscillated"
        return "plateau"

    out = {"repo": REPO, "revision": REV, "commit": "b12164578f7b893ada0028c00d292ba10383d25a",
           "seed": SEED, "dtype": "float32", "device": "cpu", "context": ctx,
           "n_positions": P, "tol": TOL,
           "median_abs_d_clean": float(d_clean.abs().median()), "runs": {}}

    for sname, v0 in (("clean", clean_v), ("clean+3sigma", clean_v + 3 * sigma),
                      ("clean+10sigma", clean_v + 10 * sigma)):
        # ---------------- fixed point ----------------
        v = v0.clone()
        R = []                                       # R[k] = r_k = x_k - xhat(x_k)
        for k in range(FP_PASSES + 1):
            fx = f(v)
            R.append((v - fx).numpy().copy())
            if k < FP_PASSES:
                v = fx
        R = np.array(R)                              # [31, P]; v is x_30
        absR = np.abs(R)
        conv_pass = np.array([next((k for k in range(FP_PASSES + 1) if absR[k, p] < TOL), -1)
                              for p in range(P)])
        ok = conv_pass >= 0
        ratios = []
        for p in range(P):
            a = absR[:, p]
            rr = [a[k + 1] / a[k] for k in range(FP_PASSES) if a[k] > 1e-7 and a[k + 1] > 1e-7]
            ratios.append(np.median(rr) if rr else np.nan)
        _, d_end = f_and_d(v)
        oc = other_change(v).numpy()
        nonconv = [{"t": pos[p][0], "c": pos[p][1], "d_final": float(d_end[p]),
                    "final_abs_r": float(absR[-1, p]), "behaviour": classify(R[:, p])}
                   for p in range(P) if not ok[p]]
        fp = {"converged": int(ok.sum()), "of": P,
              "median_passes": float(np.median(conv_pass[ok])) if ok.any() else None,
              "max_passes": int(conv_pass[ok].max()) if ok.any() else None,
              "median_abs_r_after_pass": {str(k): float(np.median(absR[k])) for k in REPORT_AT},
              "median_abs_r_start": float(np.median(absR[0])),
              "median_ratio": float(np.nanmedian(ratios)),
              "median_abs_d_at_final_point": float(d_end.abs().median()),
              "median_abs_d_at_converged_points": float(d_end[torch.tensor(ok)].abs().median()) if ok.any() else None,
              "plateau_median_abs_r_nonconverged": float(np.median(absR[-1, ~ok])) if (~ok).any() else None,
              "other_position_residual_change_median": float(np.median(oc[ok])) if ok.any() else None,
              "other_position_residual_change_max": float(oc[ok].max()) if ok.any() else None,
              "non_converging": nonconv}

        # ---------------- Newton ----------------
        v = v0.clone()
        NR, ND = [], []
        for k in range(NEWTON_STEPS + 1):
            fx, d = f_and_d(v)
            r = v - fx
            NR.append(r.numpy().copy()); ND.append(d.numpy().copy())
            if k < NEWTON_STEPS:
                v = v - r / (1 - d)
        NR, ND = np.array(NR), np.array(ND)
        absN = np.abs(NR)
        conv_n = np.array([next((k for k in range(NEWTON_STEPS + 1) if absN[k, p] < TOL), -1)
                           for p in range(P)])
        okn = conv_n >= 0
        nratios = []
        for p in range(P):
            a = absN[:, p]
            rr = [a[k + 1] / a[k] for k in range(NEWTON_STEPS) if a[k] > 1e-7 and a[k + 1] > 1e-7]
            nratios.append(np.median(rr) if rr else np.nan)
        ocn = other_change(v).numpy()
        nonconv_n = [{"t": pos[p][0], "c": pos[p][1], "d_final": float(ND[-1, p]),
                      "final_abs_r": float(absN[-1, p]), "behaviour": classify(NR[:, p])}
                     for p in range(P) if not okn[p]]
        nw = {"converged": int(okn.sum()), "of": P,
              "median_steps": float(np.median(conv_n[okn])) if okn.any() else None,
              "max_steps": int(conv_n[okn].max()) if okn.any() else None,
              "median_ratio": float(np.nanmedian(nratios)) if np.isfinite(nratios).any() else None,
              "median_abs_d_at_final_point": float(np.median(np.abs(ND[-1]))),
              "median_abs_d_at_converged_points": float(np.median(np.abs(ND[-1, okn]))) if okn.any() else None,
              "plateau_median_abs_r_nonconverged": float(np.median(absN[-1, ~okn])) if (~okn).any() else None,
              "other_position_residual_change_median": float(np.median(ocn[okn])) if okn.any() else None,
              "other_position_residual_change_max": float(ocn[okn].max()) if okn.any() else None,
              "non_converging": nonconv_n}
        out["runs"][sname] = {"fixed_point": fp, "newton": nw}
        print(f"[{sname}] FP conv {fp['converged']}/{P} med {fp['median_passes']} max {fp['max_passes']} "
              f"ratio {fp['median_ratio']:.4f} d {fp['median_abs_d_at_final_point']:.4f} | "
              f"Newton conv {nw['converged']}/{P} med {nw['median_steps']} max {nw['max_steps']}",
              flush=True)

    (HERE.parent / "results" / "check1_fixed_point.json").write_text(json.dumps(out, indent=1))
    print("record:", HERE.parent / "results" / "check1_fixed_point.json")


if __name__ == "__main__":
    main()

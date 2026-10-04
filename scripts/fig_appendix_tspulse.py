"""
Full Jacobian rows for TSPulse-r1 under the two calling conventions of
checkpoints.py, for Figure 1.

checkpoints.py records only the diagonal |dxhat[t,c]/dx[t,c]|. This script
records |dxhat[t,c]/dx[t+k,c]| for offsets k in [-K, K], so the figure can show
the mechanism: which inputs the reconstruction at t actually reads.

Same checkpoint, window, seed and probe positions as checkpoints.py. The masked
call hides the probed position's own patch (one probe at a time, so the
context the model reads is otherwise intact).

Run:  python3 basisgate/code/tspulse_jacobian_rows.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import make_window, SEED, N_CH, N_PROBE  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
K = 32


def main():
    from tsfm_public.models.tspulse import TSPulseForReconstruction
    m = TSPulseForReconstruction.from_pretrained(
        "ibm-granite/granite-timeseries-tspulse-r1",
        revision="b12164578f7b893ada0028c00d292ba10383d25a",
        num_input_channels=N_CH, mask_type="user")
    m.eval()
    ctx, p = m.config.context_length, m.config.patch_length
    x = torch.tensor(make_window(ctx), dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    ts = rng.choice(np.arange(32, ctx - 32), size=N_PROBE, replace=False)
    pos = [(int(t), int(c)) for t in ts for c in range(N_CH)]

    def call(z, obs):
        o = m(past_values=z.unsqueeze(0), past_observed_mask=obs.unsqueeze(0),
              return_loss=False)
        return o.reconstruction_outputs.squeeze(0)

    rows = {"all_observed": [], "patch_masked": []}
    for (t, c) in pos:
        for name in rows:
            obs = torch.ones_like(x)
            if name == "patch_masked":
                s = (t // p) * p
                obs[s:s + p, :] = 0.0
            xi = x.clone().requires_grad_(True)
            g = torch.autograd.grad(call(xi, obs)[t, c], xi)[0]
            rows[name].append(g[t - K:t + K + 1, c].abs().tolist())

    res = {"checkpoint": "ibm-granite/granite-timeseries-tspulse-r1",
           "revision": "tspulse-hybrid-dualhead-512-p8-r1",
           "seed": SEED, "context": ctx, "patch_length": p,
           "n_positions": len(pos), "offsets": list(range(-K, K + 1))}
    for name, r in rows.items():
        a = np.asarray(r)
        res[name] = {"median": np.median(a, 0).tolist(),
                     "p25": np.percentile(a, 25, 0).tolist(),
                     "p75": np.percentile(a, 75, 0).tolist(),
                     "diag_median": float(np.median(a[:, K])),
                     "diag_max": float(a[:, K].max())}
        print(f"  {name:<14} diag median {res[name]['diag_median']:.5g}  "
              f"diag max {res[name]['diag_max']:.3g}  "
              f"off-diag median sum {np.median(a, 0).sum() - np.median(a[:, K]):.3g}")
    path = OUT / "tspulse_jacobian_rows.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"  record: {path}")


if __name__ == "__main__":
    main()

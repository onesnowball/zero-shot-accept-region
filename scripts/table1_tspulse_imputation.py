"""
Measure dxhat[t,c]/dx[t,c] on real public checkpoints, at the configuration
their anomaly-detection path actually runs.

WHY
    Two independent readings of TSPulse's AD source reached opposite
    conclusions about whether the scored point is hidden from the model. One
    read `patchwise_stitched_reconstruction`, which masks the evaluated patch.
    The other read `mask_type="user"` with an all-observed mask at test time.
    Both are readings of prose. Autodiff is not.

PROTOCOL
    The architectural prediction is recorded here, in this file, before any
    number is produced. The finding the paper reports is the set of
    DISAGREEMENTS between prediction and measurement. No disagreements is a
    reportable outcome that weakens the paper, and it must be reported.

Run:  python3 basisgate/code/checkpoints.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

SEED = 20260822
CTX = 512
N_CH = 4
N_PROBE = 16          # timesteps probed per model


def make_window(n, c=N_CH, seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    latent = np.sin(2 * np.pi * t / 97.0) + 0.4 * np.sin(2 * np.pi * t / 23.0)
    gains, periods = [1.0, 0.7, -0.5, 0.35], [61.0, 43.0, 137.0, 29.0]
    return np.stack([gains[k] * latent
                     + 0.5 * np.sin(2 * np.pi * t / periods[k] + k)
                     + 0.05 * rng.standard_normal(n) for k in range(c)], axis=1)


def jac_diag(fn, x, positions):
    """|d xhat[t,c] / d x[t,c]| by autograd, one position at a time."""
    vals = []
    for (t, c) in positions:
        xi = x.clone().detach().requires_grad_(True)
        out = fn(xi)[t, c]
        g = torch.autograd.grad(out, xi, allow_unused=True)[0]
        vals.append(0.0 if g is None else float(g[t, c].abs()))
    return np.array(vals)


def summarise(J):
    return dict(n=len(J), median=float(np.median(J)), max=float(J.max()),
                p95=float(np.percentile(J, 95)),
                frac_exact_zero=float((J == 0.0).mean()))


# ---------------------------------------------------------------------------
RESULTS = {}


def record(name, prediction, source, J, note=""):
    s = summarise(J)
    meas_zero = s["median"] < 1e-8
    pred_zero = prediction.strip().startswith("0")
    s.update(prediction=prediction, prediction_source=source, note=note,
             agrees=(meas_zero == pred_zero))
    RESULTS[name] = s
    tag = "0 (exact)" if s["max"] == 0.0 else f"{s['median']:.5g}"
    flag = "" if s["agrees"] else "   <-- DISAGREES"
    print(f"  {name:<46} pred {prediction:<10} measured {tag:>12}{flag}")


# ---------------------------------------------------------------------------
def run_tspulse():
    from tsfm_public.models.tspulse import TSPulseForReconstruction
    repo = "ibm-granite/granite-timeseries-tspulse-r1"
    print(f"\nloading {repo} ...", flush=True)
    m = TSPulseForReconstruction.from_pretrained(
        repo, revision="b12164578f7b893ada0028c00d292ba10383d25a",
        num_input_channels=N_CH, mask_type="user")
    m.eval()
    ctx = m.config.context_length
    x = torch.tensor(make_window(ctx), dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    ts = rng.choice(np.arange(32, ctx - 32), size=N_PROBE, replace=False)
    pos = [(int(t), int(c)) for t in ts for c in range(N_CH)]
    print(f"  context {ctx}, probing {len(pos)} positions", flush=True)

    # (a) the path R1 read: all-observed mask, plain reconstruction
    def fn_plain(z):
        o = m(past_values=z.unsqueeze(0),
              past_observed_mask=torch.ones_like(z).unsqueeze(0),
              return_loss=False)
        return o.reconstruction_outputs.squeeze(0)

    record("TSPulse, all-observed (R1's reading)", "0",
           "TSPulse paper A.8.2: AD heads reconstruct the input; masking is "
           "described only for pre-training",
           jac_diag(fn_plain, x, pos),
           "if this is the deployed AD path, the scored point IS an input")

    # (b) the path R2 read: the evaluated patch is masked before reconstruction
    p = m.config.patch_length

    def fn_masked(z):
        obs = torch.ones_like(z)
        # hide every probed position's own patch
        for (t, _) in pos:
            s = (t // p) * p
            obs[s:s + p, :] = 0.0
        o = m(past_values=z.unsqueeze(0),
              past_observed_mask=obs.unsqueeze(0), return_loss=False)
        return o.reconstruction_outputs.squeeze(0)

    record("TSPulse, evaluated patch masked (R2's reading)", "0",
           "granite-tsfm patchwise_stitched_reconstruction: 'Only patches whose "
           "start indices fall within [start,end) are masked and reconstructed'",
           jac_diag(fn_masked, x, pos),
           f"patch length {p}; masking is patch-granular, not point-granular")


def run_moment():
    from momentfm import MOMENTPipeline
    print("\nloading AutonLab/MOMENT-1-large ...", flush=True)
    m = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction"})
    m.init(); m.eval()
    L = 512
    x = torch.tensor(make_window(L), dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    ts = rng.choice(np.arange(32, L - 32), size=N_PROBE, replace=False)
    pos = [(int(t), int(c)) for t in ts for c in range(N_CH)]

    def fn(z):
        o = m(x_enc=z.T.unsqueeze(0),
              input_mask=torch.ones(1, L))
        return o.reconstruction.squeeze(0).T

    record("MOMENT-1-large, reconstruction", "0",
           "MOMENT is a masked-pretraining model; TSB-AD scores it with "
           "input_mask = ones and no mask= argument",
           jac_diag(fn, x, pos),
           "masking appears only inside fit(); this is the scoring path")


def main():
    print("=" * 82)
    print("  dxhat[t,c]/dx[t,c] on public checkpoints, at the AD configuration")
    print("=" * 82)
    for fn in (run_tspulse,):
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__}: FAILED {type(e).__name__}: {e}")
            RESULTS[fn.__name__] = {"error": f"{type(e).__name__}: {e}"}

    dis = [k for k, v in RESULTS.items() if v.get("agrees") is False]
    print("\n" + "=" * 82)
    if dis:
        print("  DISAGREEMENTS (architecture description vs measurement):")
        for k in dis:
            print(f"    - {k}: predicted {RESULTS[k]['prediction']}, "
                  f"measured median {RESULTS[k]['median']:.5g}")
    else:
        print("  No disagreements among checkpoints that ran.")
    print("=" * 82)

    p = OUT / "checkpoints.json"
    p.write_text(json.dumps(RESULTS, indent=2))
    print(f"\n  record: {p}")


if __name__ == "__main__":
    main()

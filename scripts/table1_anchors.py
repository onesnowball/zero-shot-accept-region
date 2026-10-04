"""
The measured partition: does a deployed detector's estimate of a channel read
that channel?

WHY THIS EXISTS
    Every claim of the form "architecture X has dxhat_i/dx_i = 0" in this
    literature is an argument from a model card. Nobody has measured it at the
    DEPLOYED configuration. Two of our own reviewers reached opposite readings
    of TSPulse's anomaly-detection path from its source code. Autodiff settles
    it; prose does not.

THE PROTOCOL (pre-registration, per the panel's requirement)
    For every detector we FIRST record `predicted_J` -- what a competent reader
    would conclude from the architecture description alone, with the source --
    and only then measure. The paper reports the DISAGREEMENTS. A table with no
    disagreements says the property is readable off the architecture diagram and
    the measurement was unnecessary; that outcome is reportable and it weakens
    the paper. Say so rather than discovering it later.

WHAT THIS SHOWS IF THE PHENOMENON IS ABSENT
    If every detector measures J ~ 0, there is no partition, the criterion is a
    restatement of architecture family, and the central claim fails. If every
    detector measures J >> 0, the free-forgery case is empty in practice. Both
    are distinguishable outcomes. This instrument can return an answer that
    kills the paper, which is the property the previous three instruments in
    this project lacked.

VALIDATION
    PCA is included as an analytic control: for xhat = V V^T x the diagonal of
    the Jacobian is exactly ||v_i||^2, known in closed form. If the harness does
    not reproduce that to float precision, no other row is trustworthy.

Run:  python3 basisgate/code/partition.py
"""
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

T, C = 512, 4          # window length, channels
SEED = 20260822
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# test signal: forecastable but not trivial, same generator as the TimesFM run
# ---------------------------------------------------------------------------
def make_window(n=T, c=C, seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    latent = np.sin(2 * np.pi * t / 97.0) + 0.4 * np.sin(2 * np.pi * t / 23.0)
    gains, periods = [1.0, 0.7, -0.5, 0.35], [61.0, 43.0, 137.0, 29.0]
    X = np.stack([gains[k] * latent
                  + 0.5 * np.sin(2 * np.pi * t / periods[k] + k)
                  + 0.05 * rng.standard_normal(n) for k in range(c)], axis=1)
    return torch.tensor(X, dtype=torch.float32)


@dataclass
class Detector:
    name: str
    family: str                 # forecaster | masked-recon | self-recon | linear
    predicted_J: str            # "0" | "nonzero" -- FROM ARCHITECTURE, pre-measurement
    prediction_source: str      # where that prediction comes from
    build: Callable             # () -> (window -> xhat), differentiable
    reconstructs_position: bool = True   # False => forecaster, target not an input
    note: str = ""


# ---------------------------------------------------------------------------
# the Jacobian diagonal, measured
# ---------------------------------------------------------------------------
def measure_J(fn, x, positions, reconstructs_position=True):
    """d xhat[t,c] / d x[t,c] at each (t,c) in `positions`, by autograd."""
    vals = []
    for (t, c) in positions:
        xi = x.clone().detach().requires_grad_(True)
        xhat = fn(xi)
        if not reconstructs_position:
            # forecaster: the guarded value is not an input at all. Confirm
            # that mechanically rather than asserting it.
            if xi.grad is not None:
                xi.grad = None
            out = xhat[c] if xhat.dim() == 1 else xhat[-1, c]
            g = torch.autograd.grad(out, xi, retain_graph=False,
                                    allow_unused=True)[0]
            vals.append(0.0 if g is None else float(g[t, c].abs()))
        else:
            out = xhat[t, c]
            g = torch.autograd.grad(out, xi, retain_graph=False,
                                    allow_unused=True)[0]
            vals.append(0.0 if g is None else float(g[t, c]))
    return np.array(vals)


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------
def build_pca(n_components=8):
    """Analytic control. xhat = V V^T x over the flattened window.

    d xhat_i / d x_i = ||v_i||^2 exactly, where v_i is row i of V.
    """
    x0 = make_window().reshape(-1)
    D = x0.numel()
    G = torch.randn(D, n_components, generator=torch.Generator().manual_seed(0))
    V, _ = torch.linalg.qr(G)                       # D x k, orthonormal columns

    def fn(x):
        v = x.reshape(-1)
        return (V @ (V.T @ v)).reshape(x.shape)

    fn.analytic_diag = (V ** 2).sum(dim=1)          # ||v_i||^2
    return fn


def build_identity_shortcut():
    """The degenerate case the field designs against: xhat = x. J == 1.

    Included so the table has a known upper anchor as well as a known lower one.
    """
    return lambda x: x.clone()


def build_causal_conv():
    """A strictly causal 1-D conv forecaster-style reconstructor.

    Predicts position t from t-1 .. t-k only. J == 0 by construction, and the
    measurement should confirm it -- a second validator, this time at the other
    end of the range.
    """
    k = 8
    w = torch.randn(C, C, k, generator=torch.Generator().manual_seed(1)) * 0.1

    def fn(x):
        xt = x.T.unsqueeze(0)                        # 1 x C x T
        pad = torch.nn.functional.pad(xt, (k, 0))[:, :, :-1]   # shift: excl. t
        return torch.nn.functional.conv1d(pad, w).squeeze(0).T

    return fn


def build_window_ae(mask_self=False):
    """A small self-reconstructing autoencoder over the whole window.

    mask_self=False -> the classic identity-shortcut risk, J != 0.
    mask_self=True  -> the position is zeroed before encoding (inference-active
                       masking). Naive reading says J == 0. This is the case the
                       masked-AD literature claims, and the case where a
                       multi-layer network can leak through a residual path.
    """
    torch.manual_seed(2)
    enc = torch.nn.Sequential(torch.nn.Linear(C, 32), torch.nn.GELU(),
                              torch.nn.Linear(32, 16))
    dec = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(),
                              torch.nn.Linear(32, C))
    mix = torch.nn.Conv1d(16, 16, 5, padding=2)      # temporal mixing

    def fn(x):
        h = enc(x)                                    # T x 16
        h = mix(h.T.unsqueeze(0)).squeeze(0).T + h    # residual, mixes neighbours
        return dec(h)

    if not mask_self:
        return fn

    def fn_masked(x):
        # zero the guarded position before encoding, one position at a time.
        # This mirrors deterministic inference-active masking (DeepFIB/RIAD).
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            xm = x.clone()
            xm[t] = 0.0
            out = out + torch.zeros_like(out).index_put_(
                (torch.tensor([t]),), fn(xm)[t:t + 1])
        return out

    return fn_masked



def build_masked_attention(n_layers=4, neighbor=0, residual=True):
    """UniAD-style neighbour-masked attention, composed over n_layers.

    UniAD (You et al., NeurIPS 2022) §3.1: "we employ a neighbor masked
    attention module, where a feature point relates to neither itself nor its
    neighbors." The mask is architectural and inference-active -- there is no
    train/eval branch in their `generate_mask`.

    The architectural reading says d xhat_t / d x_t = 0.

    R1's objection: masked attention COMPOSED over residual layers is not a
    masked composition. Token j (outside the mask of i) may attend to i at
    layer 1; token i may attend to j at layer 2. On a grid with many relay
    nodes, information routes around the mask. This is the PixelCNN blind-spot
    problem, and UniAD does not discuss it.

    `neighbor` = half-width of the masked band (0 => mask self only, the 1x1
    setting whose ablation UniAD reports as buying nothing).
    """
    torch.manual_seed(3)
    d = 32
    inp = torch.nn.Linear(C, d)
    out = torch.nn.Linear(d, C)
    qs = [torch.nn.Linear(d, d) for _ in range(n_layers)]
    ks = [torch.nn.Linear(d, d) for _ in range(n_layers)]
    vs = [torch.nn.Linear(d, d) for _ in range(n_layers)]

    # band mask: position t may not attend to [t-neighbor, t+neighbor]
    idx = torch.arange(T)
    band = (idx[None, :] - idx[:, None]).abs() <= neighbor
    mask = torch.zeros(T, T).masked_fill(band, float("-inf"))

    def fn(x):
        h = inp(x)
        for q, k, v in zip(qs, ks, vs):
            att = (q(h) @ k(h).T) / (d ** 0.5) + mask
            a = torch.softmax(att, dim=-1) @ v(h)
            h = h + a if residual else a
        return out(h)

    return fn


REGISTRY = [
    Detector(
        name="PCA (analytic control)",
        family="linear",
        predicted_J="nonzero, = ||v_i||^2",
        prediction_source="closed form for xhat = V V^T x",
        build=build_pca,
        note="validator: measurement must match ||v_i||^2 to float precision",
    ),
    Detector(
        name="identity shortcut",
        family="self-recon",
        predicted_J="nonzero, = 1",
        prediction_source="xhat = x by definition",
        build=build_identity_shortcut,
        note="upper anchor; the failure mode masking was invented to prevent",
    ),
    Detector(
        name="causal conv (forecaster-style)",
        family="forecaster",
        predicted_J="0",
        prediction_source="strictly causal kernel excludes position t",
        build=build_causal_conv,
        note="lower anchor; J=0 by causality, not by design choice",
    ),
    Detector(
        name="window AE, unmasked",
        family="self-recon",
        predicted_J="nonzero",
        prediction_source="the guarded value is an encoder input",
        build=lambda: build_window_ae(mask_self=False),
        note="the identity-shortcut class",
    ),
    Detector(
        name="window AE, inference-active self-mask",
        family="masked-recon",
        predicted_J="0",
        prediction_source="position zeroed before encoding (DeepFIB/RIAD-style)",
        build=lambda: build_window_ae(mask_self=True),
        note="THE CONTESTED CASE: does temporal mixing leak the masked value "
             "back through neighbouring positions?",
    ),
    Detector(
        name="masked attn, 1 layer (self-mask only)",
        family="masked-recon",
        predicted_J="0",
        prediction_source="UniAD-style: point attends to neither itself nor neighbours",
        build=lambda: build_masked_attention(n_layers=1, neighbor=0),
        note="single layer: the mask genuinely excludes the position",
    ),
    Detector(
        name="masked attn, 4 layers + residual (UniAD depth)",
        family="masked-recon",
        predicted_J="0",
        prediction_source="same architectural claim, applied to the deployed depth",
        build=lambda: build_masked_attention(n_layers=4, neighbor=0),
        note="R1's contested case: does a per-layer mask survive composition?",
    ),
    Detector(
        name="masked attn, 4 layers, 7-wide band (UniAD default)",
        family="masked-recon",
        predicted_J="0",
        prediction_source="UniAD ships neighbor_size 7x7 with 4+4 layers",
        build=lambda: build_masked_attention(n_layers=4, neighbor=3),
        note="the shipped configuration",
    ),
]


def main():
    x = make_window()
    rng = np.random.default_rng(SEED)
    # sample interior positions; avoid the first/last k where padding dominates
    ts = rng.choice(np.arange(16, T - 16), size=24, replace=False)
    positions = [(int(t), int(c)) for t in ts for c in range(C)]

    rows = []
    print(f"measuring |d xhat[t,c] / d x[t,c]| at {len(positions)} positions "
          f"({len(ts)} timesteps x {C} channels)\n")

    for det in REGISTRY:
        try:
            fn = det.build()
            J = measure_J(fn, x, positions, det.reconstructs_position)
            absJ = np.abs(J)
            row = dict(name=det.name, family=det.family,
                       predicted_J=det.predicted_J,
                       prediction_source=det.prediction_source,
                       n_positions=len(positions),
                       J_median=float(np.median(absJ)),
                       J_max=float(absJ.max()),
                       J_p95=float(np.percentile(absJ, 95)),
                       frac_below_1e6=float((absJ < 1e-6).mean()),
                       note=det.note)

            # analytic validation where available
            if hasattr(fn, "analytic_diag"):
                exp = np.array([float(fn.analytic_diag[t * C + c])
                                for (t, c) in positions])
                row["analytic_max_abs_err"] = float(np.abs(absJ - exp).max())

            rows.append(row)
            tag = "~0" if row["J_median"] < 1e-6 else f"{row['J_median']:.4g}"
            print(f"  {det.name:<44} predicted {det.predicted_J:<22} "
                  f"measured median {tag}")
        except Exception as e:
            print(f"  {det.name:<44} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append(dict(name=det.name, error=f"{type(e).__name__}: {e}"))

    # ---- the finding the paper turns on: do any rows disagree?
    disagreements = []
    for r in rows:
        if "J_median" not in r:
            continue
        pred_zero = r["predicted_J"].strip().startswith("0")
        meas_zero = r["J_median"] < 1e-6
        if pred_zero != meas_zero:
            disagreements.append(
                f"{r['name']}: predicted {r['predicted_J']}, "
                f"measured median |J| = {r['J_median']:.4g}")

    print()
    print("=" * 78)
    if disagreements:
        print("  DISAGREEMENTS between architecture description and measurement:")
        for d in disagreements:
            print(f"    - {d}")
        print("\n  => the property is NOT readable off the architecture diagram.")
    else:
        print("  No disagreements. Every measurement matched its architectural")
        print("  prediction. The criterion is readable off the model card, and")
        print("  the paper must say so -- this weakens the contribution.")
    print("=" * 78)

    res = {"window": [T, C], "seed": SEED, "n_positions": len(positions),
           "rows": rows, "disagreements": disagreements}
    p = OUT / "partition.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n  record: {p}")


if __name__ == "__main__":
    main()

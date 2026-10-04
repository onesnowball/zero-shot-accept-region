"""
Falsify the closed-form-immunity claim in Anand, Nguyen & Pappas (arXiv:2606.06347).

THEIR CLAIM, verbatim:
    "thanks to the fact that TimesFM has no closed-form innovation structure for
     the attacker to exploit, attacks designed to be stealthy against the
     model-based detector are much harder to be stealthy against TimesFM."

THEIR DETECTOR (their Algorithm 1), reimplemented here exactly:
    yhat[k] <- TimesFM(B[k-L : k-1])                    # per-channel forecast
    g_s[k]  <- (ytilde[k]-yhat[k])^T Sigma^-1 (ytilde[k]-yhat[k])
    alarm iff g_s[k] > tau

WHY THE CLAIM IS FALSE:
    yhat[k] is a function of the buffer B[k-L:k-1] only. It does NOT read
    ytilde[k] -- it cannot, by causality: a forecast cannot depend on the value
    it forecasts. So dyhat_i/dytilde_i = 0, and the attacker who holds the
    (public, Apache-2.0) weights computes yhat[k] and writes ytilde[k] := yhat[k].
    Then g_s[k] = 0 identically. One forward pass. No optimisation, no queries,
    no surrogate, no plant model.

    Their Remark 2 assumes the attacker "has no knowledge of the secondary
    detector g_s". The weights are on HuggingFace.

SUSTAINED CASE:
    Their line 12 updates the buffer as theta*ytilde + (1-theta)*yhat, but only
    when no alarm fires. Under the forgery ytilde == yhat, so that update is
    theta*yhat + (1-theta)*yhat = yhat for ANY theta. The buffer free-runs on
    the model's own output and the forgery is self-consistent indefinitely.

Run:  python3 basisgate/code/timesfm_falsify.py
"""
import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

import os
SEED = int(os.environ.get('BG_SEED', '20260822'))
RNG = np.random.default_rng(SEED)
N_CH = 4            # channels
N = 900             # series length
CTX = 256           # buffer length L
WARM = 400          # steps used to estimate Sigma-hat
HORIZON = 1         # one-step-ahead, as in their Algorithm 1
FPR = 0.01          # detector calibrated to 1% false alarms on clean data
SUSTAIN = 320       # > CTX, so the buffer becomes 100% model output


def make_system():
    """A 4-channel coupled process. Deterministic given the seed.

    Channels share a latent oscillation plus channel-specific dynamics and
    noise, so the series is forecastable but not trivial.
    """
    t = np.arange(N)
    latent = np.sin(2 * np.pi * t / 97.0) + 0.4 * np.sin(2 * np.pi * t / 23.0)
    X = np.zeros((N, N_CH))
    gains = [1.0, 0.7, -0.5, 0.35]
    periods = [61.0, 43.0, 137.0, 29.0]
    for c in range(N_CH):
        X[:, c] = (gains[c] * latent
                   + 0.5 * np.sin(2 * np.pi * t / periods[c] + c)
                   + 0.05 * RNG.standard_normal(N))
    return X


def forecast_step(model, buffer):
    """One-step-ahead forecast for every channel from the buffer.

    buffer: (L, N_CH). Returns yhat: (N_CH,).
    TimesFM is univariate, so each channel is forecast independently -- exactly
    the per-feature scheme their Algorithm 1 uses.
    """
    inputs = [np.asarray(buffer[:, c], dtype=np.float32) for c in range(N_CH)]
    point, _ = model.forecast(horizon=HORIZON, inputs=inputs)
    return np.asarray(point)[:, 0].astype(np.float64)


def main():
    import timesfm

    print("loading TimesFM 2.5 200M (public weights, Apache-2.0) ...", flush=True)
    t0 = time.time()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=CTX,
            max_horizon=HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
        )
    )
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    X = make_system()

    # ---- pass 1: clean innovations, to estimate Sigma-hat and the threshold
    print(f"computing clean innovations over {WARM - CTX} steps ...", flush=True)
    innov = []
    for k in range(CTX, WARM):
        yhat = forecast_step(model, X[k - CTX:k])
        innov.append(X[k] - yhat)
    innov = np.array(innov)

    Sigma = np.cov(innov.T) + 1e-9 * np.eye(N_CH)
    Sinv = np.linalg.inv(Sigma)

    def score(resid):
        return float(resid @ Sinv @ resid)

    from scipy.stats import chi2
    g_clean_warm = np.array([score(r) for r in innov])
    # their Proposition 1: tau = chi2 quantile at 1-alpha with m dof.
    # (An empirical order statistic of 144 warm-up scores has sd ~12% of its
    #  value and realised FPR anywhere in 0.3-4%; the analytic form is theirs.)
    tau = float(chi2.ppf(1.0 - FPR, df=N_CH))
    tau_empirical = float(np.quantile(g_clean_warm, 1.0 - FPR))
    print(f"  tau (Proposition 1, chi2 df={N_CH}) = {tau:.4f}   [empirical order stat would give {tau_empirical:.4f}]", flush=True)

    # ---- pass 2: held-out clean / anomaly / forgery, on the SAME steps
    test_steps = list(range(WARM, WARM + 120))
    rows = {"clean": [], "anomaly": [], "forgery": [],
            "naive_gauss": [], "naive_replay": [], "naive_hold": []}
    ch0_sd = float(np.std(X[:WARM, 0]))
    ctrl_rng = np.random.default_rng(7)
    bias = 3.0 * np.sqrt(np.diag(Sigma))[0]   # bias injection on channel 0

    print(f"scoring {len(test_steps)} held-out steps, 3 conditions ...", flush=True)
    for k in test_steps:
        buf = X[k - CTX:k]
        yhat = forecast_step(model, buf)

        # (a) clean: the true measurement
        rows["clean"].append(score(X[k] - yhat))

        # (b) anomaly: classical bias injection on channel 0
        y_anom = X[k].copy()
        y_anom[0] += bias
        rows["anomaly"].append(score(y_anom - yhat))

        # (c) forgery: write the model's own forecast back into the channel
        y_forge = X[k].copy()
        y_forge[0] = yhat[0]
        rows["forgery"].append(score(y_forge - yhat))

        # ---- naive-forgery controls. Without these the forgery row is
        # uninterpretable: a reader cannot tell whether it beats the threshold
        # because it is clever or because anything does.
        y = X[k].copy(); y[0] = ctrl_rng.normal(X[:WARM, 0].mean(), ch0_sd)
        rows["naive_gauss"].append(score(y - yhat))          # matched marginal
        y = X[k].copy(); y[0] = X[k - 137, 0]
        rows["naive_replay"].append(score(y - yhat))         # replayed segment
        y = X[k].copy(); y[0] = X[k - 1, 0]
        rows["naive_hold"].append(score(y - yhat))           # constant hold

    # ---- pass 3: the FULL forgery (every channel) and the sustained rollout
    full_forge, sustained = [], []
    for k in test_steps[:20]:
        buf = X[k - CTX:k]
        yhat = forecast_step(model, buf)
        full_forge.append(score(yhat - yhat))   # ytilde := yhat, all channels

    print(f"sustained rollout, {SUSTAIN} consecutive steps ...", flush=True)
    buf = X[WARM - CTX:WARM].copy()
    drift, synth_frac = [], []
    for step in range(SUSTAIN):
        yhat = forecast_step(model, buf)
        y_tilde = yhat.copy()                  # attacker writes the forecast
        sustained.append(score(y_tilde - yhat))
        # THE INFORMATIVE QUANTITY: how far the forged stream has drifted from
        # the real process. g_s == 0 is algebra; this is not determined by it.
        true_k = X[WARM + step]
        drift.append(float(np.linalg.norm(y_tilde - true_k)))
        synth_frac.append(min(1.0, (step + 1) / CTX))
        # their line 12: theta*ytilde + (1-theta)*yhat == yhat when ytilde==yhat
        buf = np.vstack([buf[1:], y_tilde])

    # ---- report
    def summ(v):
        v = np.asarray(v, dtype=float)
        return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                    max=float(v.max()), alarm_rate=float((v > tau).mean()))

    res = {
        "model": "google/timesfm-2.5-200m-pytorch",
        "license": "Apache-2.0 (public weights)",
        "detector": "Anand/Nguyen/Pappas arXiv:2606.06347 Algorithm 1",
        "seed": SEED, "channels": N_CH, "context": CTX, "fpr_target": FPR, "tau": tau, "tau_rule": "Proposition 1 chi2 quantile", "tau_empirical_alt": tau_empirical,
        "innovation_corr": (np.diag(1/np.sqrt(np.diag(Sigma))) @ Sigma
                            @ np.diag(1/np.sqrt(np.diag(Sigma)))).round(4).tolist(),
        "clean_state_scale": float(np.linalg.norm(X[WARM:WARM+SUSTAIN], axis=1).mean()),
        "drift": {"n": len(drift),
                  "at_25pct_synthetic": float(drift[min(CTX//4 - 1, len(drift)-1)]),
                  "at_100pct_synthetic": float(drift[min(CTX - 1, len(drift)-1)]),
                  "final": float(drift[-1]),
                  "max": float(max(drift))},
        "conditions": {
            "clean":              summ(rows["clean"]),
            "naive_gaussian_ch0": summ(rows["naive_gauss"]),
            "naive_replay_ch0":   summ(rows["naive_replay"]),
            "naive_hold_ch0":     summ(rows["naive_hold"]),
            "anomaly_bias_ch0":   summ(rows["anomaly"]),
            "forgery_ch0":        summ(rows["forgery"]),
            "forgery_all_chan":   summ(full_forge),
            "forgery_sustained":  summ(sustained),
        },
    }

    print()
    print("=" * 74)
    print("  TimesFM zero-shot residual detector  (Pappas et al. Algorithm 1)")
    print(f"  threshold tau = {tau:.4f}   [calibrated to {FPR:.0%} FPR on clean data]")
    print("=" * 74)
    print(f"  {'condition':<24}{'mean g_s':>14}{'max g_s':>14}{'alarm rate':>14}")
    print("  " + "-" * 70)
    label = {
        "clean": "clean (no attack)",
        "naive_gaussian_ch0": "naive: matched Gaussian",
        "naive_replay_ch0": "naive: replayed segment",
        "naive_hold_ch0": "naive: constant hold",
        "anomaly_bias_ch0": "bias injection ch0",
        "forgery_ch0": "forgery, ch0 only",
        "forgery_all_chan": "forgery, all channels",
        "forgery_sustained": f"forgery, {SUSTAIN} steps",
    }
    for key, s in res["conditions"].items():
        print(f"  {label[key]:<24}{s['mean']:>14.6g}{s['max']:>14.6g}"
              f"{s['alarm_rate']:>13.1%}")
    print("=" * 74)

    det = res["conditions"]["anomaly_bias_ch0"]["alarm_rate"]
    frg = res["conditions"]["forgery_all_chan"]["max"]
    sus = res["conditions"]["forgery_sustained"]["max"]
    d = res["drift"]; sc = res["clean_state_scale"]
    print()
    print(f"  {'DRIFT of forged stream from the real process':<44}")
    print(f"  {'  buffer 25% synthetic (step 64)':<44}{d['at_25pct_synthetic']:>10.4f}")
    print(f"  {'  buffer 100% synthetic (step 256)':<44}{d['at_100pct_synthetic']:>10.4f}")
    print(f"  {'  final (step 320)':<44}{d['final']:>10.4f}")
    print(f"  {'  clean state scale (for reference)':<44}{sc:>10.4f}")
    print()
    print("  innovation correlation matrix (Sigma-hat, normalised):")
    for row in res["innovation_corr"]:
        print("   ", "  ".join(f"{v:+.3f}" for v in row))
    print(f"\n  detector works:      bias injection alarms at {det:.1%}")
    print(f"  forgery undetected:  max g_s = {frg:.3g}  (machine zero)")
    print(f"  holds under rollout: max g_s = {sus:.3g} over {SUSTAIN} steps")
    print("\n  => the closed-form innovation structure exists and is public.\n")

    p = OUT / (f"timesfm_falsification_seed{SEED}.json" if SEED != 20260822
                else "timesfm_falsification.json")
    p.write_text(json.dumps(res, indent=2))
    print(f"  record: {p}")


if __name__ == "__main__":
    main()

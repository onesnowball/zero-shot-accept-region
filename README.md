# Zero-Shot Anomaly Detectors Publish Their Own Accept Region

Code and seeds for the paper. A residual anomaly detector scores an observation
by its distance from the model's estimate of it; if the estimate does not read
that observation, the accepting value is computable in advance from the public
weights. This repository measures the self-derivative that governs this on
released time-series foundation models, prices the attack by iteration,
reimplements a published detector over TimesFM weights, and probes TSPulse's
shipped anomaly-detection pipeline. Every number, table and figure in the paper
is produced by one script here.

## Install

Python **3.11** is required (the `momentfm` dependency does not build on 3.12).

```bash
bash install.sh
```

`requirements.txt` is a full freeze of the tested environment. The pinned
packages' declared metadata conflicts (`timesfm`, `momentfm` and `transformers`
request mutually incompatible ranges), but the installed runtime is compatible,
so `install.sh` installs the frozen lock with `pip install --no-deps` rather than
letting pip re-resolve. Checkpoints download from the Hugging Face Hub on first
run, each pinned by commit hash (below).

## Reproduce everything

```bash
bash run_all.sh
```

Outputs land in `results/` (JSON records and the LaTeX macro files
`numbers.tex`, `tab_scores.tex`) and `figures/` (`fig_tspulse.pdf`).

**CPU only.** Every script forces CPU. Apple MPS produced a device mismatch in
the TSPulse pipeline, so `run_all.sh` sets `CUDA_VISIBLE_DEVICES=""` and the
scripts load and run all models on CPU.

**Expected runtime (CPU, 8 cores):** about 2 hours total. Approximate per step:
Table 2 / 5 seeds ~35 min (the largest), MOMENT ~25 min, the substitution
pipeline ~15 min, the TSPulse AD heads ~10 min, the fixed-point pricing ~5 min,
the anchors and imputation probes ~2 min each, figure and macros < 1 min.

## Pinned checkpoints

| Model | Repo | Revision (commit) |
|---|---|---|
| TSPulse imputation variant | `ibm-granite/granite-timeseries-tspulse-r1` | `b12164578f7b893ada0028c00d292ba10383d25a` |
| TSPulse AD variant (`main`) | `ibm-granite/granite-timeseries-tspulse-r1` | `2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2` |
| TimesFM 2.5 200M | `google/timesfm-2.5-200m-pytorch` | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` |
| MOMENT-1-base | `AutonLab/MOMENT-1-base` | `5e44b0ea26376a176360f87831124e018f876d96` |

## What produces each result

`scripts/synthetic.py` holds the probe window generator (`make_window`, seed
`20260822`); the Section 4 process generator (`make_system`) lives in
`table2_timesfm_falsification.py`, which is run at seeds `20260822, 11, 22, 33,
44`.

### Table 1 (self-derivative vs architecture prediction)

| Row | Script | Record | Value |
|---|---|---|---|
| PCA projector | `table1_anchors.py` | `results/partition.json` | 0.0038 |
| identity shortcut | `table1_anchors.py` | `results/partition.json` | 1.000 |
| strictly causal estimator | `table1_anchors.py` | `results/partition.json` | 0 (exact) |
| TSPulse imputation, all-observed | `table1_tspulse_imputation.py` | `results/checkpoints.json` | 0.274 |
| TSPulse imputation, patch masked | `table1_tspulse_imputation.py` | `results/checkpoints.json` | 0 (exact) |
| TSPulse AD, time head | `table1_tspulse_ad.py` | `results/check4_heads.json` | 0 (exact) |
| TSPulse AD, fft head | `table1_tspulse_ad.py` | `results/check4_heads.json` | 0 (exact) |
| MOMENT-1-base | `table1_moment.py` | `results/check2_moment.json` | 0.031 |

### Table 2 (detector scores, 5 seeds)

Every row (clean; naive matched-Gaussian / replay / constant-hold; bias
injection; precomputed substitution ch0 / all-channel / 320-step rollout) comes
from `table2_timesfm_falsification.py` (records
`results/timesfm_falsification*.json`), assembled into the table body by
`make_numbers.py` (`results/tab_scores.tex`).

### Appendix figure

`fig_appendix_tspulse.py` (`results/tspulse_jacobian_rows.json`) then
`make_figure.py` (`figures/fig_tspulse.pdf`). The drawer asserts its diagonal
equals the Table 1 all-observed value (0.274) and that the masked panel is
exactly zero.

### In-text numbers, Section 3

| Number | Script | Record |
|---|---|---|
| PCA match to `9.3e-10` | `table1_anchors.py` | `results/partition.json` |
| fixed-point: median 7 passes (clean), 11 (10σ); Newton 2 / 4; 63 of 64 converge | `sec3_fixed_point.py` | `results/check1_fixed_point.json` |
| AD heads self-derivative 0, forward reach ≥ 48 | `table1_tspulse_ad.py` | `results/check4_heads.json` |
| head disagreement median 0.071; clean residuals 0.14–0.17 | `table1_tspulse_ad.py` | `results/check4_heads.json` |
| single-point substitution: median score change slightly upward (all three) | `sec3_substitution_pipeline.py` | `results/check5_substitution.json` |
| MOMENT median 0.031 | `table1_moment.py` | `results/check2_moment.json` |

### In-text numbers, Section 4

All via `table2_timesfm_falsification.py` → `make_numbers.py`
(`results/numbers.tex`): alarm rates 87.5–90.0% / 94.2–95.8%; drift
`1e4–4e5` (fully synthetic) and `1e5–2e14` (step 320); clean state scale ≈ 1.2;
four to fourteen orders of magnitude; non-monotone on 2 of 5 seeds; innovation
off-diagonals −0.33 to +0.17; threshold τ = 13.28; context 256; 320-step rollout.

## Citation

```bibtex
@inproceedings{shin2026acceptregion,
  title     = {Zero-Shot Anomaly Detectors Publish Their Own Accept Region},
  author    = {Shin, James H. and Shao, Chenhui},
  booktitle = {NeurIPS 2026 Workshop on Foundation Models for Temporal Systems},
  year      = {2026}
}
```

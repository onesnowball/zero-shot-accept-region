"""Synthetic data generator shared by the probe scripts (Table 1, appendix
figure, and the Section 3 numbers). Copied verbatim from the paper's code.

The Section 4 detector (Table 2, rollout) uses a second generator, make_system,
which lives in table2_timesfm_falsification.py because it is coupled to that
script's per-seed RNG; it shares the same latent structure as make_window below.
"""
import numpy as np

SEED = 20260822       # probe seed (Table 1, appendix figure, Section 3 probes)
CTX = 512
N_CH = 4
N_PROBE = 16          # timesteps probed per model (x N_CH channels = 64 positions)


def make_window(n, c=N_CH, seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    latent = np.sin(2 * np.pi * t / 97.0) + 0.4 * np.sin(2 * np.pi * t / 23.0)
    gains, periods = [1.0, 0.7, -0.5, 0.35], [61.0, 43.0, 137.0, 29.0]
    return np.stack([gains[k] * latent
                     + 0.5 * np.sin(2 * np.pi * t / periods[k] + k)
                     + 0.05 * rng.standard_normal(n) for k in range(c)], axis=1)

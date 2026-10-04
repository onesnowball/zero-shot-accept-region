"""
Table 1 MOMENT row: measures the self-derivative |d xhat[t,c]/d x[t,c]| for
MOMENT-1-base in zero-shot reconstruction, reproducing the model's scoring call
with gradients enabled (CPU).

Input: the synthetic window from synthetic.py (seed 20260822), 256-step window.
Writes results/check2_moment.json. Run: python scripts/table1_moment.py (or run_all.sh).
"""

import json, sys, warnings, numpy as np, torch
from pathlib import Path
_RES=Path(__file__).resolve().parent.parent/"results"; _RES.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")
sys.path.insert(0, "code")
from synthetic import make_window, SEED, N_CH, N_PROBE
WIN = 256
try:
    from momentfm import MOMENTPipeline
    import momentfm, importlib.metadata as md
    dev = "cpu"
    model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-base", revision="5e44b0ea26376a176360f87831124e018f876d96",
                model_kwargs={"task_name": "reconstruction"})
    model.init(); model = model.to(dev).float(); model.eval()
    x = torch.tensor(make_window(WIN), dtype=torch.float32)         # [256, 4]
    rng = np.random.default_rng(SEED)
    ts = rng.choice(np.arange(32, WIN-32), size=N_PROBE, replace=False)
    pos = [(int(t), int(c)) for t in ts for c in range(N_CH)]
    mask = torch.ones(1, WIN, device=dev)
    def recon(z):                         # z [256,4] -> [256,4]
        o = model(x_enc=z.T.unsqueeze(0).to(dev), input_mask=mask)
        return o.reconstruction.squeeze(0).T
    diag, fwd = [], {}
    K = 48
    for (t,c) in pos:
        zi = x.clone().requires_grad_(True)
        g = torch.autograd.grad(recon(zi)[t,c], zi)[0][:,c].abs().cpu()
        diag.append(float(g[t]))
        for o in range(-K, K+1):
            tp=t+o
            if 0<=tp<WIN: fwd.setdefault(o,[]).append(float(g[tp]))
    med={o:float(np.median(v)) for o,v in fwd.items()}
    reach=max([o for o in med if o>0 and med[o]>1e-6], default=0)
    d=np.array(diag)
    out={"model":"AutonLab/MOMENT-1-base","via":"TSB-AD run_MOMENT_ZS path "
         "(MOMENTPipeline reconstruction, input_mask=ones, no mask= arg)",
         "momentfm_version":md.version("momentfm"),"device":dev,"dtype":"float32",
         "seed":SEED,"window":WIN,"n_positions":len(pos),
         "differences_from_wrapper":["wrapper runs under torch.no_grad(); here "
           "grads are on","wrapper hardcodes cuda; device here="+dev,
           "context 256 (MOMENT win_size) vs 512 in section 3; 4 channels kept"],
         "median_abs_self_derivative":float(np.median(d)),
         "iqr_abs_self_derivative":[float(np.percentile(d,25)),float(np.percentile(d,75))],
         "max_abs_self_derivative":float(d.max()),
         "frac_exact_zero":float((d==0).mean()),
         "structural_zero":bool(d.max()==0.0),
         "zero_kind":"n/a" if d.max()>0 else "all-observed reconstruction: "
           "scored point IS an input, so any zero is numerical not structural",
         "forward_reach":reach}
    print(json.dumps({k:out[k] for k in ("median_abs_self_derivative",
          "max_abs_self_derivative","forward_reach","frac_exact_zero")}))
    open(_RES/"check2_moment.json","w").write(json.dumps(out,indent=1))
    print("OK")
except Exception as e:
    import traceback; traceback.print_exc()
    print("MOMENT_FAILED", type(e).__name__, str(e)[:160])

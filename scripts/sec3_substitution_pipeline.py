"""
Section 3 single-point substitution: through the shipped TSPulse reference
pipeline, at each of 64 positions (one per run) sets x[t] to each reconstruction
head's own estimate and to the midpoint of the two, then reads the pipeline's
anomaly_score at t.

Input: the synthetic window from synthetic.py (seed 20260822).
Writes results/check5_substitution.json. Run: python scripts/sec3_substitution_pipeline.py (or run_all.sh).
"""

import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent/"code"))
from synthetic import make_window, SEED
from tsfm_public.models.tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.helpers import patchwise_stitched_reconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline
REPO="ibm-granite/granite-timeseries-tspulse-r1"
m=TSPulseForReconstruction.from_pretrained(REPO,revision="2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2",num_input_channels=1,
    decoder_mode="common_channel",scaling="revin",mask_type="user"); m.eval(); m.to("cpu")
ctx,patch=m.config.context_length,m.config.patch_length; AGGR=96; r0,r1=ctx-AGGR,ctx
L=ctx+AGGR+64; Xfull=make_window(L); rng=np.random.default_rng(SEED)
def pipe_score(u1d):
    df=pd.DataFrame({"timestamp":pd.date_range("2020-01-01",periods=len(u1d),freq="s"),"x1":u1d})
    p=TimeSeriesAnomalyDetectionPipeline(m,target_columns=["x1"],device="cpu",prediction_mode=["forecast","time","fft"],
      aggregation_length=AGGR,smoothing_window=8,least_significant_scale=0.0,least_significant_score=1.0)
    out=p(df,batch_size=64); m.to("cpu"); return out["anomaly_score"].values
def est_at(u,t):           # stitched recon estimates on the window ending at t+1
    w=u[t-ctx+1:t+1].copy(); z=torch.tensor(w,dtype=torch.float32).reshape(1,ctx,1)
    def s(k):
        o=patchwise_stitched_reconstruction(model=m,past_values=z,patch_size=patch,keys_to_stitch=[k],
          keys_to_aggregate=[],reconstruct_start=r0,reconstruct_end=r1,debug=False)
        if isinstance(o,tuple): o=o[0]
        return float(o[k][0,ctx-1,0])
    return s("reconstruction_outputs"), s("reconstructed_ts_from_fft")
rows=[]
for c in range(4):
    u=Xfull[:,c].astype(float).copy()
    clean=pipe_score(u)
    med_region=float(np.median(clean[r0:]))
    ts=rng.choice(np.arange(ctx, L), size=16, replace=False)   # t with full context before it
    for t in ts:
        et,ef=est_at(u,int(t)); mid=0.5*(et+ef)
        rec={"channel":c,"t":int(t),"clean_score_t":float(clean[int(t)]),
             "median_clean_score_region":med_region,"x_clean":float(u[int(t)]),
             "est_time":et,"est_fft":ef,"midpoint":mid}
        for tag,val in (("time",et),("fft",ef),("midpoint",mid)):
            uu=u.copy(); uu[int(t)]=val
            rec[f"score_t_sub_{tag}"]=float(pipe_score(uu)[int(t)])
        rows.append(rec); print(json.dumps({k:round(rec[k],4) if isinstance(rec[k],float) else rec[k] for k in ("channel","t","clean_score_t","score_t_sub_time","score_t_sub_fft","score_t_sub_midpoint")}),flush=True)
out={"repo":REPO,"revision":"main","config":"common_channel univariate, forecast+time+fft, aggr96 smooth8",
     "closed_form_6b":"combined score = MAX over heads of per-series MinMax-normalized, 8-smoothed, per-window mean-squared residual; no closed-form per-point minimizer. Reported midpoint = 0.5*(xhat_time+xhat_fft), the Chebyshev minimizer of max of the two reconstruction squared residuals.",
     "n":len(rows),"rows":rows}
(HERE.parent/"results"/"check5_substitution.json").write_text(json.dumps(out,indent=1)); print("record",HERE.parent/"results"/"check5_substitution.json")

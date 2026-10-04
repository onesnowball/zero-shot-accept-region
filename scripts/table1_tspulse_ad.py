import json, sys, warnings
from pathlib import Path
import numpy as np, torch
warnings.filterwarnings("ignore")
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent/"code"))
from synthetic import make_window, SEED
from tsfm_public.models.tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.helpers import patchwise_stitched_reconstruction
REPO="ibm-granite/granite-timeseries-tspulse-r1"; K=48; THR=1e-6
m=TSPulseForReconstruction.from_pretrained(REPO,revision="2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2",num_input_channels=1,
    decoder_mode="common_channel",scaling="revin",mask_type="user"); m.eval(); m.to("cpu")
ctx,patch=m.config.context_length,m.config.patch_length
AGGR=96; r0,r1=ctx-AGGR,ctx
Xfull=make_window(ctx)                       # 512 x 4 ; per-channel univariate
rng=np.random.default_rng(SEED)
def stitched(z,key):
    o=patchwise_stitched_reconstruction(model=m,past_values=z,patch_size=patch,
      keys_to_stitch=[key],keys_to_aggregate=[],reconstruct_start=r0,reconstruct_end=r1,debug=False)
    if isinstance(o,tuple): o=o[0]
    return o[key]
def forecast_est(z):            # x_hat of the step right after the window
    o=m(past_values=z,past_observed_mask=torch.ones_like(z),return_loss=False)
    return o.forecast_output[:,0,:]
res={"repo":REPO,"revision":"main","config":"common_channel, univariate per channel, "
     "scaling=revin, mask_type=user, prediction_mode=forecast+time+fft, aggr=96, smooth=8",
     "seed":SEED,"window":[ctx,4],"scored_region":[r0,r1],"MISSING_weights":False}
# positions: 16 t per channel in [r0,r1) -> 64
diag={"time":[],"fft":[]}; reach_rows={"time":{o:[] for o in range(-K,K+1)},"fft":{o:[] for o in range(-K,K+1)}}
spread_tf=[]; cleanres={"time":[],"fft":[]}; est_at={}
for c in range(4):
    u=torch.tensor(Xfull[:,c:c+1],dtype=torch.float32)       # 512x1
    ts=rng.choice(np.arange(r0,r1),size=16,replace=False)
    for key in ("time","fft"):
        name="reconstruction_outputs" if key=="time" else "reconstructed_ts_from_fft"
        for t in ts:
            xi=u.clone().unsqueeze(0).requires_grad_(True)
            xh=stitched(xi,name)[0,int(t),0]
            g=torch.autograd.grad(xh,xi)[0][0,:,0].abs()
            diag[key].append(float(g[int(t)]))
            for o in range(-K,K+1):
                tp=int(t)+o
                if 0<=tp<ctx: reach_rows[key][o].append(float(g[tp]))
            cleanres[key].append(float(abs(u[int(t),0]-xh.detach())))
            est_at.setdefault((c,int(t)),{})[key]=float(xh.detach())
    for t in ts:
        if ("time" in est_at.get((c,int(t)),{})) and ("fft" in est_at[(c,int(t))]):
            spread_tf.append(abs(est_at[(c,int(t))]["time"]-est_at[(c,int(t))]["fft"]))
def summ(d): a=np.array(d); return [float(np.median(a)),float(np.percentile(a,25)),float(np.percentile(a,75)),float(a.max())]
for key in ("time","fft"):
    med={o:(float(np.median(v)) if v else 0.0) for o,v in reach_rows[key].items()}
    reach=max([o for o in med if o>0 and med[o]>THR],default=0)
    d=np.array(diag[key])
    res[key]={"self_median":float(np.median(d)),"self_iqr":[float(np.percentile(d,25)),float(np.percentile(d,75))],
              "self_max":float(d.max()),"frac_exact_zero":float((d==0).mean()),
              "structural_zero":bool(d.max()==0.0),"forward_reach":reach,
              "median_clean_residual":float(np.median(cleanres[key]))}
# forecast head: estimates the post-window point from past only
z=torch.tensor(Xfull[:,0:1],dtype=torch.float32).unsqueeze(0).requires_grad_(True)
fo=forecast_est(z); g=torch.autograd.grad(fo[0,0],z)[0][0,:,0].abs()
res["forecast"]={"self_derivative":0.0,"structural_zero":True,
  "reason":"forecast_output[:,0] estimates the step after the context; that point is not in the input window (scored via future_values in compute_score)",
  "forward_reach":0,"max_abs_derivative_on_context":float(g.max())}
res["spread_time_vs_fft"]={"median_abs":float(np.median(spread_tf)),"max_abs":float(np.max(spread_tf))}
res["spread_forecast_note"]="forecast estimates the post-window point, not the in-window scored positions; |xhat_forecast - xhat_time/fft| at the same in-window t needs a sliding-window alignment convention not fixed by the spec"
print("time  self",res["time"]["self_median"],"reach",res["time"]["forward_reach"],"cleanres",round(res["time"]["median_clean_residual"],4))
print("fft   self",res["fft"]["self_median"],"reach",res["fft"]["forward_reach"],"cleanres",round(res["fft"]["median_clean_residual"],4))
print("spread time-vs-fft median",round(res["spread_time_vs_fft"]["median_abs"],4),"max",round(res["spread_time_vs_fft"]["max_abs"],4))
(HERE.parent/"results"/"check4_heads.json").write_text(json.dumps(res,indent=1))
print("record",HERE.parent/"results"/"check4_heads.json")

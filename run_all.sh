#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper, on CPU.
# Usage:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=""          # no GPU
export PYTORCH_ENABLE_MPS_FALLBACK=1    # Apple MPS caused a device mismatch; CPU only
export TOKENIZERS_PARALLELISM=false
PY="${PYTHON:-python}"
mkdir -p results figures
cd scripts
echo "[1/10] Table 1 anchors (PCA, identity, causal)";      "$PY" table1_anchors.py
echo "[2/10] Table 1 TSPulse imputation rows";              "$PY" table1_tspulse_imputation.py
echo "[3/10] Appendix figure data";                         "$PY" fig_appendix_tspulse.py
echo "[4/10] Table 1 TSPulse AD heads + Section 3 reach/disagreement"; "$PY" table1_tspulse_ad.py
echo "[5/10] Table 1 MOMENT row";                           "$PY" table1_moment.py
echo "[6/10] Section 3 fixed-point / Newton pricing";       "$PY" sec3_fixed_point.py
echo "[7/10] Section 3 single-point substitution";          "$PY" sec3_substitution_pipeline.py
echo "[8/10] Table 2 + rollout (5 seeds)"
"$PY" table2_timesfm_falsification.py
for s in 11 22 33 44; do BG_SEED=$s "$PY" table2_timesfm_falsification.py; done
echo "[9/10] Numeric macros + Table 2 body";                "$PY" make_numbers.py
echo "[10/10] Appendix figure";                             "$PY" make_figure.py
echo "DONE. Outputs in results/ and figures/."

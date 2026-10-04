#!/usr/bin/env bash
# Recreate the exact CPU environment. Python 3.11 required (momentfm does not
# build on 3.12). The pinned packages' declared metadata conflicts, but the
# runtime set is compatible, so we install the frozen lock without re-resolving.
set -euo pipefail
cd "$(dirname "$0")"
python3.11 -m venv .venv || python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install --no-deps -r requirements.txt
python -c "import torch,transformers,timesfm; from tsfm_public.models.tspulse import TSPulseForReconstruction; from momentfm import MOMENTPipeline; print('env ok:', transformers.__version__)"

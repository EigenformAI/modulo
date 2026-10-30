#!/bin/bash
# Setup for the self-steering agent project.
# Works on a fresh Runpod PyTorch pod OR any Linux box with a CUDA GPU (>=24GB VRAM).
# Idempotent: safe to re-run.
#
# Usage:
#   ./setup.sh           # install deps + download vector + sanity check
#   ./setup.sh --skip-model   # skip the model load sanity check (faster)
set -eo pipefail

# Resolve script's own directory (don't assume a hardcoded path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Keep HF downloads on the pod's large disk (matches config.py)
if [ -d /workspace ]; then export HF_HOME="${HF_HOME:-/workspace/hf_cache}"; fi

echo "=== Self-Steering Agent setup ==="
echo "Project dir: $SCRIPT_DIR"
echo "Python: $(python3 --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo ""

# 1. Install Python deps
echo "=== Installing Python dependencies ==="
# --break-system-packages is needed on PEP 668 systems and unknown to old pips
if ! pip install --break-system-packages --quiet -r requirements.txt; then
    pip install --quiet -r requirements.txt
fi
# Editable install: puts the `modulo` command on PATH and keeps vectors/ resolving to this repo
if ! pip install --break-system-packages --quiet --no-deps -e .; then
    pip install --quiet --no-deps -e .
fi

# 2. Create directories
mkdir -p logs

# 3. Fetch the default model's vectors from the community index (checked against their sha256)
echo ""
echo "=== Fetching steering vectors from the community index ==="
python3 -c "
import config
from modulo.registry import VectorRegistry
r = VectorRegistry.for_model(config.MODEL_NAME, auto_extract=False)
print('  available:', ', '.join(r.list_names()) or 'none')
"

# 4. Sanity check: load model + forward pass
if [ "$1" != "--skip-model" ]; then
    echo ""
    echo "=== Sanity check: loading Qwen2.5-7B-Instruct ==="
    # pipefail makes a failed load fail setup; the grep only filters noise
    python3 load_model.py 2>&1 | { grep -v 'Loading checkpoint\|torch_dtype\|generation flags\|Fetching\|FutureWarning\|warnings.warn' || true; }
else
    echo ""
    echo "=== Skipping model sanity check (--skip-model) ==="
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  ./run.sh                    # run an interactive self-steering session"
echo "  ./run.sh --auto --max-turns 10  # non-interactive demo (10 turns)"
echo "  python replay.py logs/*.jsonl   # replay a session log"
echo ""
echo "To add more steering vectors: python extract_contrastive.py --help (or see CONTRIBUTING.md)"
echo "Discovery catalog of available vectors: modulo_catalog.yaml"

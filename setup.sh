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
VEC_DIR="vectors/Qwen_Qwen2.5-7B-Instruct"

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

# 2. Create directories
mkdir -p "$VEC_DIR" data logs agent

# 3. Download the assistant-axis steering vector (the one confirmed-working vector)
echo ""
echo "=== Downloading assistant-axis vector ==="
if [ -f "$VEC_DIR/assistant_axis.pt" ]; then
    echo "  already present, skipping"
else
    VEC_DIR="$VEC_DIR" python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
d = os.environ['VEC_DIR']
p = hf_hub_download(repo_id='Butanium/qwen-2.5-7b-instruct-assistant-axis',
                    filename='assistant_axis.pt', repo_type='dataset')
shutil.copy(p, os.path.join(d, 'assistant_axis.pt'))
print(f'  saved to {d}/assistant_axis.pt')
" 2>&1 | { grep -v 'Fetching\|Downloading' || true; }
fi

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
echo "To add more steering vectors: python extract_contrastive.py --help (or edit $VEC_DIR/registry.yaml)"
echo "Discovery catalog of available vectors: pharmacopoeia_catalog.yaml"

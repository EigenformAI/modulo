#!/bin/bash
# Setup for the self-steering agent project.
# Works on a fresh Runpod PyTorch pod OR any Linux box with a CUDA GPU (>=24GB VRAM).
# Idempotent: safe to re-run.
#
# Usage:
#   ./setup.sh           # install deps + download vector + sanity check
#   ./setup.sh --skip-model   # skip the model load sanity check (faster)
set -e

# Resolve script's own directory (don't assume a hardcoded path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Self-Steering Agent setup ==="
echo "Project dir: $SCRIPT_DIR"
echo "Python: $(python3 --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
echo ""

# 1. Install Python deps
echo "=== Installing Python dependencies ==="
pip install --break-system-packages --quiet -r requirements.txt 2>&1 | tail -3 || \
    pip install --quiet -r requirements.txt 2>&1 | tail -3

# 2. Create directories
mkdir -p vectors data logs agent

# 3. Download the assistant-axis steering vector (the one confirmed-working vector)
echo ""
echo "=== Downloading assistant-axis vector ==="
if [ -f "vectors/assistant_axis.pt" ]; then
    echo "  already present, skipping"
else
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('vectors', exist_ok=True)
p = hf_hub_download(repo_id='Butanium/qwen-2.5-7b-instruct-assistant-axis',
                    filename='assistant_axis.pt', repo_type='dataset',
                    cache_dir=os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')))
shutil.copy(p, 'vectors/assistant_axis.pt')
print(f'  saved to vectors/assistant_axis.pt')
" 2>&1 | grep -v 'Fetching\|Downloading'
fi

# 4. Sanity check: load model + forward pass
if [ "$1" != "--skip-model" ]; then
    echo ""
    echo "=== Sanity check: loading Qwen2.5-7B-Instruct ==="
    python3 load_model.py 2>&1 | grep -v 'Loading checkpoint\|torch_dtype\|generation flags\|Fetching\|FutureWarning\|warnings.warn'
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
echo "To add more steering vectors, edit vectors/registry.yaml"
echo "Discovery catalog of available vectors: pharmacopoeia_catalog.yaml"

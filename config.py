"""Central config for the self-steering agent project."""
import os
from pathlib import Path

# Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bf16"  # bfloat16 fits comfortably in 24GB
DEVICE_MAP = "auto"

# Qwen2.5-7B has 28 decoder layers (indices 0..27). Mid layers are good for steering.
# CAA on Llama-2-7B (32 layers) used layer 13; for Qwen2.5-7B we start at layer 13.
DEFAULT_STEER_LAYER = 13
LAYER_SWEEP = [10, 13, 16, 20]  # tried if default doesn't shift behavior

# Paths (resolved relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent
VECTORS_DIR = PROJECT_ROOT / "vectors"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in (VECTORS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# HF cache on the pod's large /workspace disk — override via env if needed.
# Only takes effect if this module is imported before transformers /
# huggingface_hub, so entry points import config first.
if Path("/workspace").is_dir():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

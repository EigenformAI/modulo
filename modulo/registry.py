"""Vector registry: loads steering vectors from a YAML config + vector files.

Supports any HF model. Vectors are stored per-model in subdirectories:
    vectors/<model_slug>/registry.yaml
    vectors/<model_slug>/assistant_axis.pt
    vectors/<model_slug>/sycophancy_layer14.npz
    ...

If no vectors exist for a model, `VectorRegistry.for_model()` triggers auto-extraction
to generate a starter set.
"""
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml

from modulo.reports import append_report, file_sha256


# Base directory for per-model vector dirs. Resolved from this file rather than
# the working directory so library users get the same registry as run.sh.
DEFAULT_VECTORS_DIR = Path(os.environ.get(
    "MODULO_VECTORS_DIR", Path(__file__).resolve().parent.parent / "vectors"))


def _model_slug(model_name: str) -> str:
    """Convert a model name to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)


def registry_dir_for(model_name: str, base_dir: Optional[Path] = None) -> Path:
    """The directory holding registry.yaml and vector files for a model."""
    return Path(base_dir or DEFAULT_VECTORS_DIR) / _model_slug(model_name)


# Manifest of models with precomputed vectors available on HuggingFace.
# Each entry: (model_name, hf_repo_id, hf_filename, repo_type)
# When you publish vectors for a new model, add it here.
BUNDLED_VECTORS = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "repo_id": "Butanium/qwen-2.5-7b-instruct-assistant-axis",
        "filename": "assistant_axis.pt",
        "repo_type": "dataset",
        "vector_name": "assistant_axis",
        "layer": 14,
        "description": "Pushes the model toward the default 'helpful assistant' persona (positive) or toward free-form role-play and character embodiment (negative).",
        "sign_convention": "positive (+) = more assistant-like, safety disclaimers, breaks character. negative (-) = more role-playing, commits to characters.",
        "safe_strength_min": -15.0,
        "safe_strength_max": 15.0,
        "format": "per_layer",
    },
    # Add more models here as you publish bundled vectors:
    # "meta-llama/Llama-3.1-8B-Instruct": {...},
    # "Qwen/Qwen2.5-14B-Instruct": {...},
}


def _try_download_bundled(model_name: str, slug: str, vectors_dir: Path) -> bool:
    """Try to download precomputed vectors for a model from HuggingFace.

    Returns True if vectors were downloaded and a registry.yaml was created.
    """
    if model_name not in BUNDLED_VECTORS:
        return False

    spec = BUNDLED_VECTORS[model_name]
    vec_file = spec["filename"]
    target_path = vectors_dir / vec_file

    if target_path.exists():
        # Already downloaded — just make sure registry.yaml exists
        _write_bundled_registry(vectors_dir, spec)
        return True

    try:
        from huggingface_hub import hf_hub_download
        import shutil
        print(f"[registry] Downloading bundled vectors for '{model_name}' from {spec['repo_id']}...")
        p = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            repo_type=spec["repo_type"],
            cache_dir=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        )
        shutil.copy(p, target_path)
        _write_bundled_registry(vectors_dir, spec)
        print(f"[registry] Downloaded {vec_file}")
        return True
    except Exception as e:
        print(f"[registry] Failed to download bundled vectors: {e}")
        return False


def _write_bundled_registry(vectors_dir: Path, spec: dict):
    """Write a registry.yaml for a bundled vector."""
    entry = {
        "name": spec["vector_name"],
        "file": spec["filename"],
        "layer": spec["layer"],
        "description": spec["description"],
        "safe_strength_min": spec["safe_strength_min"],
        "safe_strength_max": spec["safe_strength_max"],
        "sign_convention": spec["sign_convention"],
        "source": f"bundled: {spec['repo_id']}",
        "format": spec["format"],
    }
    with open(vectors_dir / "registry.yaml", "w") as f:
        yaml.safe_dump([entry], f, default_flow_style=False)


@dataclass
class VectorEntry:
    name: str
    file: str
    layer: int
    description: str = ""
    safe_strength_min: float = -10.0
    safe_strength_max: float = 10.0
    sign_convention: str = ""
    source: str = ""
    format: str = "single"  # 'single' or 'per_layer'

    def load_vector(self, vectors_dir: Path) -> torch.Tensor:
        """Load the actual vector tensor for this entry's layer."""
        path = vectors_dir / self.file
        if self.format == "per_layer":
            data = torch.load(path, map_location="cpu", weights_only=True)
            return data[self.layer].to(torch.bfloat16)
        elif self.file.endswith(".npz"):
            import numpy as np
            data = np.load(path)
            return torch.tensor(data["vector"], dtype=torch.bfloat16)
        elif self.file.endswith(".pt"):
            data = torch.load(path, map_location="cpu", weights_only=True)
            return data.to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown vector format: {self.file}")


class VectorRegistry:
    def __init__(self, vectors_dir: Path, registry_path: Optional[Path] = None):
        self.vectors_dir = vectors_dir
        self.registry_path = registry_path or (vectors_dir / "registry.yaml")
        self.entries: dict[str, VectorEntry] = {}
        self.load()

    def load(self):
        if not self.registry_path.exists():
            return
        with open(self.registry_path) as f:
            data = yaml.safe_load(f) or []
        for item in data:
            item.pop("feedback", None)  # older registries kept reports here
            entry = VectorEntry(**item)
            self.entries[entry.name] = entry

    def get(self, name: str) -> Optional[VectorEntry]:
        return self.entries.get(name)

    def list_names(self) -> list[str]:
        return list(self.entries.keys())

    @property
    def reports_path(self) -> Path:
        return Path(self.vectors_dir) / "reports.jsonl"

    def add_report(self, report: dict):
        """Append a report (see modulo.reports.make_report) to reports.jsonl."""
        append_report(self.reports_path, report)

    def vector_sha256(self, entry: VectorEntry) -> Optional[str]:
        return file_sha256(Path(self.vectors_dir) / entry.file)

    def describe_for_prompt(self, max_strength: float = float("inf")) -> str:
        if not self.entries:
            return "(no steering vectors available — use extract_auto() to generate some)"
        lines = []
        for name, e in self.entries.items():
            lo = max(e.safe_strength_min, -max_strength)
            hi = min(e.safe_strength_max, max_strength)
            lines.append(
                f"- {name}: {e.description} "
                f"(strength range {lo:g} to {hi:g}, "
                f"layer {e.layer}). {e.sign_convention}"
            )
        return "\n".join(lines)

    def save(self):
        """Save the registry back to YAML."""
        data = []
        for entry in self.entries.values():
            data.append(dict(entry.__dict__))
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def add_entry(self, entry: VectorEntry):
        self.entries[entry.name] = entry

    @classmethod
    def for_model(cls, model_name: str, base_dir: Optional[Path] = None,
                  auto_extract: bool = True, model=None, tokenizer=None) -> "VectorRegistry":
        """Get or create a registry for a specific model.

        Tries bundled vectors first (downloaded from HuggingFace if available),
        then falls back to auto-extraction if none exist and auto_extract=True.

        Bundled vectors live in HF datasets keyed by model slug. If a model has
        no bundled vectors, we extract a starter set using contrastive system prompts.
        """
        slug = _model_slug(model_name)
        vectors_dir = registry_dir_for(model_name, base_dir)
        vectors_dir.mkdir(parents=True, exist_ok=True)
        registry = cls(vectors_dir)

        if registry.entries:
            return registry  # already have vectors for this model

        # Try downloading bundled vectors for this model
        if _try_download_bundled(model_name, slug, vectors_dir):
            registry = cls(vectors_dir)  # reload after download
            if registry.entries:
                print(f"[registry] Loaded {len(registry.entries)} bundled vectors for '{model_name}'")
                return registry

        # No bundled vectors — auto-extract if model provided
        if auto_extract:
            if model is None or tokenizer is None:
                print(f"[registry] No vectors for '{model_name}' and no model provided for auto-extraction.")
                print(f"[registry] Vectors dir: {vectors_dir}")
                print(f"[registry] Run extract_auto() with a loaded model to generate starter vectors.")
            else:
                from modulo.extraction import extract_auto
                print(f"[registry] No bundled vectors for '{model_name}'. Auto-extracting starter set...")
                extract_auto(model, tokenizer, model_name, vectors_dir, registry)
                registry = cls(vectors_dir)  # reload after extraction

        return registry

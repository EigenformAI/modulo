"""Vector registry: loads steering vectors from a YAML config + vector files.

Supports any HF model. Vectors are stored per-model in subdirectories:
    vectors/<model_slug>/registry.yaml
    vectors/<model_slug>/assistant_axis.pt
    vectors/<model_slug>/sycophancy_layer14.npz
    ...

registry.yaml holds your own vectors. `VectorRegistry.for_model()` adds the
model's vectors from the community index (community/vectors.yaml) in memory, and
auto-extracts a starter set if the model ends up with none.
"""
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
        path = Path(vectors_dir) / self.file
        if self.file.endswith(".safetensors"):
            from safetensors.torch import load_file
            tensors = load_file(path)
            if len(tensors) != 1:
                raise ValueError(f"{self.file}: expected exactly one tensor, found {len(tensors)}")
            data = next(iter(tensors.values()))
        elif self.file.endswith(".npz"):
            import numpy as np
            data = torch.tensor(np.load(path)["vector"])
        elif self.file.endswith(".pt"):
            data = torch.load(path, map_location="cpu", weights_only=True)
        else:
            raise ValueError(f"Unknown vector format: {self.file}")
        if self.format == "per_layer":  # [num_layers, hidden], indexed by layer
            data = data[self.layer]
        return data.to(torch.bfloat16)


class VectorRegistry:
    def __init__(self, vectors_dir: Path, registry_path: Optional[Path] = None):
        self.vectors_dir = vectors_dir
        self.registry_path = registry_path or (vectors_dir / "registry.yaml")
        self.entries: dict[str, VectorEntry] = {}
        # Names that came from the community index: kept in memory, never saved
        self._index_names: set[str] = set()
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
        """Save your own vectors back to YAML (index vectors are left out)."""
        data = []
        for entry in self.entries.values():
            if entry.name not in self._index_names:
                data.append(dict(entry.__dict__))
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def add_entry(self, entry: VectorEntry):
        self.entries[entry.name] = entry
        self._index_names.discard(entry.name)

    def add_index_entries(self, entries: list[VectorEntry]) -> list[str]:
        """Add community-index vectors in memory. Your own vectors win on a name clash."""
        notes = []
        for entry in entries:
            if entry.name in self.entries and entry.name not in self._index_names:
                notes.append(f"'{entry.name}' from the community index is shadowed by your own "
                             f"vector of that name in {self.registry_path}")
                continue
            self.entries[entry.name] = entry
            self._index_names.add(entry.name)
        return notes

    @classmethod
    def for_model(cls, model_name: str, base_dir: Optional[Path] = None,
                  auto_extract: bool = True, model=None, tokenizer=None,
                  include_community: bool = False, index_path: Optional[Path] = None) -> "VectorRegistry":
        """Get or create a registry for a specific model.

        Loads your own vectors from registry.yaml, then adds the model's
        verified vectors from the community index (plus unverified community
        ones if include_community=True), downloading files as needed. If the
        model still has no vectors and auto_extract=True, extracts a starter
        set using contrastive system prompts.
        """
        from modulo.community import index_entries_for
        vectors_dir = registry_dir_for(model_name, base_dir)
        vectors_dir.mkdir(parents=True, exist_ok=True)
        registry = cls(vectors_dir)

        config = getattr(model, "config", None)
        entries, notes = index_entries_for(
            model_name, vectors_dir, index_path=index_path, include_community=include_community,
            hidden_size=getattr(config, "hidden_size", None),
            num_layers=getattr(config, "num_hidden_layers", None))
        notes += registry.add_index_entries(entries)
        for note in notes:
            print(f"[registry] {note}")
        if registry._index_names:
            print(f"[registry] {len(registry._index_names)} vector(s) for '{model_name}' from the community index")
        if registry.entries:
            return registry

        # No vectors for this model — auto-extract if model provided
        if auto_extract:
            if model is None or tokenizer is None:
                print(f"[registry] No vectors for '{model_name}' and no model provided for auto-extraction.")
                print(f"[registry] Vectors dir: {vectors_dir}")
                print(f"[registry] Run extract_auto() with a loaded model to generate starter vectors.")
            else:
                from modulo.extraction import extract_auto
                print(f"[registry] No vectors for '{model_name}'. Auto-extracting starter set...")
                extract_auto(model, tokenizer, model_name, vectors_dir, registry)
                registry = cls(vectors_dir)  # reload after extraction

        return registry

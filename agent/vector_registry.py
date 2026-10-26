"""Vector registry: loads steering vectors from a YAML config + vector files.

The YAML format (vectors/registry.yaml) is a list of vector entries:

    - name: assistant_axis
      file: assistant_axis.pt
      layer: 14
      description: "Pushes toward default assistant persona (+) or role-play (-)"
      safe_strength_min: -10.0
      safe_strength_max: 10.0
      sign_convention: "positive = more assistant-like, negative = more role-playing"
      source: "Butanium/qwen-2.5-7b-instruct-assistant-axis"
      format: "per_layer"   # 'per_layer' = [num_layers, hidden_dim], 'single' = [hidden_dim]

    - name: sycophancy
      file: sycophancy_layer14.npz
      layer: 14
      description: "..."
      ...

The user maintains this file and adds vectors as they research them.
Agents can call `report_vector(name, worked=True/False, notes="...")` to
log feedback that informs which vectors to trust.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml

import config


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
    # runtime feedback (not saved to YAML)
    feedback: list = field(default_factory=list)

    def load_vector(self) -> torch.Tensor:
        """Load the actual vector tensor for this entry's layer."""
        path = config.VECTORS_DIR / self.file
        if self.format == "per_layer":
            # shape [num_layers, hidden_dim] — pick the layer
            data = torch.load(path, weights_only=True)
            return data[self.layer].to(torch.bfloat16)
        elif self.file.endswith(".npz"):
            import numpy as np
            data = np.load(path)
            return torch.tensor(data["vector"], dtype=torch.bfloat16)
        elif self.file.endswith(".pt"):
            data = torch.load(path, weights_only=True)
            return data.to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown vector format: {self.file}")


class VectorRegistry:
    def __init__(self, registry_path=None):
        self.registry_path = registry_path or (config.VECTORS_DIR / "registry.yaml")
        self.entries: dict[str, VectorEntry] = {}
        self.load()

    def load(self):
        if not self.registry_path.exists():
            print(f"[registry] No registry file at {self.registry_path} — starting empty")
            return
        with open(self.registry_path) as f:
            data = yaml.safe_load(f) or []
        for item in data:
            entry = VectorEntry(**item)
            self.entries[entry.name] = entry
        print(f"[registry] Loaded {len(self.entries)} vectors: {list(self.entries.keys())}")

    def get(self, name: str) -> Optional[VectorEntry]:
        return self.entries.get(name)

    def list_names(self) -> list[str]:
        return list(self.entries.keys())

    def add_feedback(self, name: str, worked: bool, notes: str):
        if name in self.entries:
            self.entries[name].feedback.append({"worked": worked, "notes": notes})

    def describe_for_prompt(self) -> str:
        """Return a human/LLM-readable description of available vectors."""
        if not self.entries:
            return "(no steering vectors available)"
        lines = []
        for name, e in self.entries.items():
            lines.append(
                f"- {name}: {e.description} "
                f"(strength range {e.safe_strength_min} to {e.safe_strength_max}, "
                f"layer {e.layer}). {e.sign_convention}"
            )
        return "\n".join(lines)

"""Community vector index (community/vectors.yaml): entry format and file checks.

The index lists vectors hosted elsewhere (usually a contributor's HuggingFace
repo), pinned by sha256. New entries come in at trust level "community"; a
maintainer promotes one to "verified" after reviewing GPU verification output
and setting its safe strength range in Modulo units.

description, sign_convention and test_prompts are written by contributors. Only
"verified" entries' text should ever reach a model's prompt.
"""
import re
import tempfile
from pathlib import Path
from typing import Optional

import torch

from modulo.registry import VectorEntry
from modulo.reports import file_sha256

TRUST_LEVELS = ("community", "verified")
FORMATS = ("single", "per_layer")
REPO_TYPES = ("model", "dataset")
FILE_SUFFIXES = (".safetensors", ".pt", ".npz")
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TEXT_CHARS = 2000

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

# field -> (allowed types, required)
ENTRY_FIELDS = {
    "id": ((str,), True),                  # unique in the index, e.g. qwen2.5-7b-assistant-axis
    "name": ((str,), True),                # concept name the model uses with steer, e.g. assistant_axis
    "model": ((str,), True),               # HF model id the vector was made for
    "model_revision": ((str,), False),
    "repo_id": ((str,), True),             # HF repo hosting the vector file
    "repo_type": ((str,), True),           # model | dataset
    "revision": ((str,), False),           # HF commit of repo_id (recommended)
    "filename": ((str,), True),
    "sha256": ((str,), True),
    "format": ((str,), True),              # single | per_layer ([num_layers, hidden])
    "layer": ((int,), True),
    "description": ((str,), True),
    "sign_convention": ((str,), True),     # what + and - do
    "extraction_method": ((str,), True),
    "license": ((str,), True),
    "contributors": ((list,), True),       # names/handles to credit
    "trust": ((str,), True),               # community | verified (maintainers only)
    # Modulo units (% of residual norm). Set by a maintainer after GPU verification.
    "safe_strength_min": ((int, float), False),
    "safe_strength_max": ((int, float), False),
    # Optional context from the contributor, in their own units/setup
    "original_setup": ((str,), False),     # coefficient, layers, positions steered, normalized?
    "paper": ((str,), False),
    "contrast_dataset": ((str,), False),
    "known_failure_modes": ((str,), False),
    "evaluation": ((str,), False),
    "test_prompts": ((list,), False),      # prompts that should show the effect
    "notes": ((str,), False),
}


def validate_entry(entry) -> list[str]:
    """Problems with one index entry's fields (no network or file access)."""
    if not isinstance(entry, dict):
        return ["entry is not a mapping"]
    label = entry.get("id", "<no id>")
    errors = []

    def err(msg):
        errors.append(f"{label}: {msg}")

    for name, (types, required) in ENTRY_FIELDS.items():
        if name not in entry:
            if required:
                err(f"missing field '{name}'")
            continue
        value = entry[name]
        if (isinstance(value, bool) and bool not in types) or not isinstance(value, types):
            err(f"'{name}' has type {type(value).__name__}")
        elif isinstance(value, str) and len(value) > MAX_TEXT_CHARS:
            err(f"'{name}' longer than {MAX_TEXT_CHARS} characters")
    for name in entry:
        if name not in ENTRY_FIELDS:
            err(f"unknown field '{name}'")
    if errors:
        return errors

    if not _ID_RE.match(entry["id"]):
        err("'id' must be lowercase letters, digits, '.', '-' or '_' (2-64 chars)")
    if not _NAME_RE.match(entry["name"]):
        err("'name' must be lowercase letters, digits or '_', starting with a letter (2-48 chars)")
    if entry["repo_type"] not in REPO_TYPES:
        err(f"'repo_type' must be one of {REPO_TYPES}")
    if entry["format"] not in FORMATS:
        err(f"'format' must be one of {FORMATS}")
    if entry["trust"] not in TRUST_LEVELS:
        err(f"'trust' must be one of {TRUST_LEVELS}")
    if not entry["filename"].endswith(FILE_SUFFIXES):
        err(f"'filename' must end with one of {FILE_SUFFIXES}")
    if not _SHA_RE.match(entry["sha256"]):
        err("'sha256' must be 64 lowercase hex characters")
    if entry["layer"] < 0:
        err("'layer' must be >= 0")
    if not entry["contributors"] or not all(isinstance(c, str) and c for c in entry["contributors"]):
        err("'contributors' must be a non-empty list of names")
    prompts = entry.get("test_prompts", [])
    if not all(isinstance(p, str) and 0 < len(p) <= MAX_TEXT_CHARS for p in prompts) or len(prompts) > 5:
        err("'test_prompts' must be up to 5 non-empty strings")
    has_min, has_max = "safe_strength_min" in entry, "safe_strength_max" in entry
    if has_min != has_max:
        err("set both 'safe_strength_min' and 'safe_strength_max', or neither")
    elif has_min and not entry["safe_strength_min"] < 0 < entry["safe_strength_max"]:
        err("safe strength range must satisfy min < 0 < max")
    if entry["trust"] == "verified" and not has_min:
        err("'verified' entries need a safe strength range")
    return errors


def validate_index(data) -> list[str]:
    """Problems with a whole index (the parsed community/vectors.yaml)."""
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return ["index must be a mapping with an 'entries' list"]
    errors = []
    seen_ids, seen_names = set(), set()
    for entry in data["entries"]:
        entry_errors = validate_entry(entry)
        errors += entry_errors
        if entry_errors:
            continue
        if entry["id"] in seen_ids:
            errors.append(f"{entry['id']}: duplicate id")
        key = (entry["model"], entry["name"])
        if key in seen_names:
            errors.append(f"{entry['id']}: another entry already uses name '{entry['name']}' for {entry['model']}")
        seen_ids.add(entry["id"])
        seen_names.add(key)
    return errors


def check_vector_file(entry: dict, path: Path, hidden_size: Optional[int] = None,
                      num_layers: Optional[int] = None) -> list[str]:
    """Check a downloaded vector file against its entry.

    Loads it with the same code the agent uses (VectorEntry.load_vector), so a
    file that passes here loads at runtime. hidden_size/num_layers come from
    the model's config; pass None to skip those checks.
    """
    path = Path(path)
    label = entry["id"]
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return [f"{label}: file is {size / 1e6:.0f} MB (max {MAX_FILE_BYTES / 1e6:.0f} MB)"]
    actual = file_sha256(path)
    if actual != entry["sha256"]:
        return [f"{label}: sha256 mismatch (file is {actual})"]
    if num_layers is not None and entry["layer"] >= num_layers:
        return [f"{label}: layer {entry['layer']} out of range ({entry['model']} has {num_layers} layers)"]

    # Load under the entry's own filename so the suffix picks the right loader
    with tempfile.TemporaryDirectory() as d:
        local = Path(d) / Path(entry["filename"]).name
        local.symlink_to(path.resolve())
        try:
            vec = VectorEntry(name=entry["name"], file=local.name, layer=entry["layer"],
                              format=entry["format"]).load_vector(Path(d))
        except Exception as e:  # any load failure is a problem with the submission
            return [f"{label}: could not load vector ({type(e).__name__}: {e})"]

    if vec.dim() != 1:
        return [f"{label}: expected a 1-D vector after selecting the layer, got shape {tuple(vec.shape)}"]
    if hidden_size is not None and vec.shape[0] != hidden_size:
        return [f"{label}: vector has {vec.shape[0]} dims, {entry['model']} has hidden size {hidden_size}"]
    norm = vec.float().norm()
    if not torch.isfinite(norm) or norm == 0:
        return [f"{label}: vector norm is {norm.item()} (must be finite and non-zero)"]
    return []

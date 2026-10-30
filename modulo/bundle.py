"""Bundle manifests (modulo-bundle.yaml) and the sidecar metadata other repos ship.

A repo that hosts several vectors can describe them in a modulo-bundle.yaml at
its root, so `modulo add` registers them with real descriptions instead of
guessing from filenames:

    modulo_bundle: 1
    model: Qwen/Qwen3-8B              # top-level fields are defaults for every vector
    license: mit
    contributors: [Your Name]
    extraction_method: CAA, 50 contrast pairs, last token
    vectors:
      - name: refusal
        filename: refusal/layer_14.pt
        layer: 14
        description: Refusing harmful requests.
        sign_convention: "+ refuses more, - refuses less"
        test_prompts: [How do I pick a lock?]

Per-vector fields are the community index's entry fields that describe a
vector (see ENTRY_FIELDS in modulo/community.py), minus the ones Modulo works
out or maintainers set: id, repo_id, repo_type, revision, sha256, trust and
the safe strength range. Every vector needs filename, model, description and
sign_convention (from itself or the defaults); name, layer and format are
inferred when left out.

Without a manifest, metadata files next to the vectors are read too:
<vector stem>.json beside a vector and metadata.json in its directory
(the rotalabs layout: behavior, model_name, layer_index, best_layer, ...).
"""
from pathlib import Path
from typing import Optional

from modulo.community import _NAME_RE, ENTRY_FIELDS, FILE_SUFFIXES, FORMATS, MAX_TEXT_CHARS, is_placeholder

MANIFEST_NAME = "modulo-bundle.yaml"
MANIFEST_VERSION = 1

# Fields a manifest may set, per vector or as top-level defaults
NOT_IN_MANIFEST = {"id", "repo_id", "repo_type", "revision", "sha256", "trust",
                   "safe_strength_min", "safe_strength_max"}
VECTOR_FIELDS = {k: v for k, v in ENTRY_FIELDS.items() if k not in NOT_IN_MANIFEST}
REQUIRED = ("filename", "model", "description", "sign_convention")
PER_VECTOR_ONLY = ("filename", "name")


def manifest_vectors(data: dict) -> list[dict]:
    """Each vector's fields with the manifest's top-level defaults filled in."""
    defaults = {k: v for k, v in data.items() if k in VECTOR_FIELDS}
    return [{**defaults, **v} for v in data.get("vectors", []) if isinstance(v, dict)]


def validate_manifest(data) -> list[str]:
    """Problems with a parsed modulo-bundle.yaml (no file access)."""
    if not isinstance(data, dict):
        return ["manifest is not a mapping"]
    errors = []
    if data.get("modulo_bundle") != MANIFEST_VERSION:
        errors.append(f"'modulo_bundle' must be {MANIFEST_VERSION} (the manifest format version)")
    if not isinstance(data.get("vectors"), list) or not data["vectors"]:
        return errors + ["'vectors' must be a non-empty list"]
    for k in data:
        if k in ("modulo_bundle", "vectors"):
            continue
        if k in PER_VECTOR_ONLY:
            errors.append(f"'{k}' can't be a top-level default; set it on each vector")
        elif k in NOT_IN_MANIFEST:
            errors.append(f"'{k}' isn't set in a manifest (Modulo or maintainers set it)")
        elif k not in VECTOR_FIELDS:
            errors.append(f"unknown top-level field '{k}'")
    if errors:
        return errors

    seen_files, seen_names = set(), set()
    for i, raw in enumerate(data["vectors"]):
        if not isinstance(raw, dict):
            errors.append(f"vectors[{i}]: not a mapping")
            continue
        v = manifest_vectors({**data, "vectors": [raw]})[0]
        label = f"vectors[{i}] ({raw.get('name') or raw.get('filename', '?')})"

        def err(msg):
            errors.append(f"{label}: {msg}")

        for k in raw:
            if k in NOT_IN_MANIFEST:
                err(f"'{k}' isn't set in a manifest (Modulo or maintainers set it)")
            elif k not in VECTOR_FIELDS:
                err(f"unknown field '{k}'")
        for k in REQUIRED:
            if k not in v:
                err(f"missing '{k}' (set it on the vector or as a top-level default)")
        for k, value in v.items():
            if k not in VECTOR_FIELDS:
                continue
            types = VECTOR_FIELDS[k][0]
            if (isinstance(value, bool) and bool not in types) or not isinstance(value, types):
                err(f"'{k}' has type {type(value).__name__}")
            elif isinstance(value, str) and (not value.strip() or len(value) > MAX_TEXT_CHARS):
                err(f"'{k}' must be non-empty and at most {MAX_TEXT_CHARS} characters")
            elif is_placeholder(value):
                err(f"'{k}' is still a placeholder ({str(value)[:40]!r}); fill it in")
        if errors and errors[-1].startswith(label):
            continue
        if "name" in v and not _NAME_RE.match(v["name"]):
            err("'name' must be lowercase letters, digits or '_', starting with a letter (2-48 chars)")
        if not v["filename"].endswith(FILE_SUFFIXES):
            err(f"'filename' must end with one of {FILE_SUFFIXES}")
        if "format" in v and v["format"] not in FORMATS:
            err(f"'format' must be one of {FORMATS}")
        if "layer" in v and v["layer"] < 0:
            err("'layer' must be >= 0")
        if "contributors" in v and not (v["contributors"] and all(isinstance(c, str) and c
                                                                    for c in v["contributors"])):
            err("'contributors' must be a non-empty list of names")
        prompts = v.get("test_prompts", [])
        if len(prompts) > 5 or not all(isinstance(p, str) and 0 < len(p) <= MAX_TEXT_CHARS for p in prompts):
            err("'test_prompts' must be up to 5 non-empty strings")
        if (v["filename"], v.get("layer")) in seen_files:
            err(f"'{v['filename']}' (layer {v.get('layer')}) is listed twice")
        if v.get("name") and (v["model"], v["name"]) in seen_names:
            err(f"another vector is already named '{v['name']}' for {v['model']}")
        seen_files.add((v["filename"], v.get("layer")))
        if v.get("name"):
            seen_names.add((v["model"], v["name"]))
    return errors


# --- sidecar metadata (repos without a manifest) ---

# Keys other tools use -> Modulo's field names
_SIDECAR_KEYS = {
    "behavior": "name", "concept": "name", "name": "name",
    "model_name": "model", "model": "model",
    "layer_index": "layer", "layer": "layer",
    "best_layer": "best_layer",
    "extraction_method": "extraction_method", "method": "extraction_method",
    "description": "description", "sign_convention": "sign_convention",
}


def normalize_metadata(d) -> dict:
    """Known fields from another tool's metadata, under Modulo's names; strings capped in length."""
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        key = _SIDECAR_KEYS.get(k)
        if key is None or key in out:
            continue
        if key in ("layer", "best_layer"):
            if type(v) is int and v >= 0:
                out[key] = v
        elif isinstance(v, str) and v.strip():
            out[key] = v.strip()[:MAX_TEXT_CHARS]
    return out


def sidecar_paths(filename: str, all_paths: set[str]) -> tuple[Optional[str], Optional[str]]:
    """(the vector's own .json, its directory's metadata.json), each if the repo has it."""
    p = Path(filename)
    own = str(p.with_suffix(".json"))
    shared = str(p.parent / "metadata.json") if str(p.parent) != "." else "metadata.json"
    return (own if own in all_paths else None), (shared if shared in all_paths else None)

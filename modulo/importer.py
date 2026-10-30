"""Add vectors from a HuggingFace repo or a local directory: `modulo add hf:<repo>[@revision]`.

Lists the repo's vector files at a pinned commit, downloads them, works out
each one's format (single vs per_layer) and layer, checks it against the
model's config, and either registers it in your own registry.yaml or prints
it as a draft community-index entry (or a draft modulo-bundle.yaml).

What a vector is comes from, in order: the repo's modulo-bundle.yaml (see
modulo/bundle.py), sidecar .json files, a dict .pt file's own fields, and
finally the file's shape and name.

Imported vectors are unverified: they get the ±COMMUNITY_DEFAULT_RANGE range
and an "[Unverified imported vector]" description prefix until you check them
(`modulo verify <name> --save-range`).
"""
import fnmatch
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from modulo.bundle import (MANIFEST_NAME, MANIFEST_VERSION, manifest_vectors, normalize_metadata,
                           sidecar_paths, validate_manifest)
from modulo.community import (COMMUNITY_DEFAULT_RANGE, ENTRY_FIELDS, FILE_SUFFIXES, MAX_FILE_BYTES,
                              check_vector_file, model_dims)
from modulo.registry import VectorEntry, VectorRegistry, load_tensor_file
from modulo.reports import file_sha256

UNVERIFIED_IMPORT_PREFIX = "[Unverified imported vector] "
DEFAULT_MAX_FILES = 50

_LAYER_RE = re.compile(r"(?:^|[^a-z])(?:layers?|l)[_-]?(\d{1,3})(?![0-9])")
_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


@dataclass
class Candidate:
    filename: str                        # path in the repo
    ok: bool = False
    reason: str = ""                     # why it was skipped, if not ok
    name: Optional[str] = None
    format: Optional[str] = None         # single | per_layer
    layer: Optional[int] = None
    shape: Optional[list[int]] = None
    sha256: Optional[str] = None
    size: Optional[int] = None
    path: Optional[Path] = None          # downloaded copy (HF cache) or the local file
    meta: dict = field(default_factory=dict)   # from the manifest or sidecar metadata
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path) if self.path else None
        return d


@dataclass
class AddPlan:
    repo_id: str                         # HF repo id, or the local directory
    repo_type: str                       # model | dataset | local
    revision: Optional[str]              # resolved commit sha (None for a local directory)
    model: str
    hidden_size: Optional[int]
    num_layers: Optional[int]
    license: Optional[str]
    candidates: list[Candidate]
    notes: list[str] = field(default_factory=list)
    manifest: Optional[str] = None       # the manifest used, if any

    @property
    def is_local(self) -> bool:
        return self.repo_type == "local"

    def source_of(self, c: Candidate) -> str:
        """Provenance recorded in registry.yaml for a vector from this plan."""
        if self.is_local:
            return f"local:{self.repo_id}/{c.filename} sha256={c.sha256}"
        return f"hf:{self.repo_type}/{self.repo_id}@{self.revision}:{c.filename} sha256={c.sha256}"

    @property
    def ok(self) -> list[Candidate]:
        return [c for c in self.candidates if c.ok]

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "candidates"}
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d


def parse_spec(spec: str, repo_type: Optional[str] = None) -> tuple[str, str, Optional[str]]:
    """'hf:owner/repo@rev', 'hf:datasets/owner/repo' or a huggingface.co URL ->
    (repo_id, repo_type, revision or None)."""
    s = spec.strip()
    for prefix in ("hf:", "https://huggingface.co/", "http://huggingface.co/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.rstrip("/")
    revision = None
    if "@" in s:
        s, revision = s.split("@", 1)
    m = re.match(r"^(.+?)/tree/([^/]+)$", s)  # URL of a branch or commit page
    if m:
        s, revision = m.group(1), revision or m.group(2)
    kind = "model"
    if s.startswith("datasets/"):
        kind, s = "dataset", s[len("datasets/"):]
    if not _REPO_RE.match(s):
        raise ValueError(f"not a HuggingFace repo: {spec!r} (expected hf:owner/repo[@revision])")
    return s, repo_type or kind, revision


class _HubSource:
    def __init__(self, repo_id: str, repo_type: str, revision: Optional[str]):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.repo_id, self.repo_type = repo_id, repo_type
        info = self.api.repo_info(repo_id, repo_type=repo_type, revision=revision)
        self.revision = info.sha  # pin everything to one commit
        card = getattr(info, "card_data", None) or getattr(info, "cardData", None) or {}
        lic = card.get("license") if hasattr(card, "get") else None
        self.license = lic if isinstance(lic, str) else None

    def files(self) -> dict[str, Optional[int]]:
        from huggingface_hub.hf_api import RepoFile
        tree = self.api.list_repo_tree(self.repo_id, repo_type=self.repo_type, revision=self.revision,
                                       recursive=True)
        return {f.path: f.size for f in tree if isinstance(f, RepoFile)}

    def fetch(self, path: str) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(self.repo_id, path, repo_type=self.repo_type, revision=self.revision))


class _LocalSource:
    repo_type, revision, license = "local", None, None

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.repo_id = str(self.root)

    def files(self) -> dict[str, Optional[int]]:
        return {str(p.relative_to(self.root)): p.stat().st_size
                for p in sorted(self.root.rglob("*")) if p.is_file()}

    def fetch(self, path: str) -> Path:
        return self.root / path


def open_source(spec: str, repo_type: Optional[str] = None):
    """A HuggingFace repo (hf:owner/repo[@rev] or a huggingface.co URL) or a local directory."""
    if not spec.startswith(("hf:", "http://", "https://")) and Path(spec).expanduser().is_dir():
        return _LocalSource(Path(spec).expanduser())
    repo_id, kind, revision = parse_spec(spec, repo_type)
    return _HubSource(repo_id, kind, revision)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def model_aliases(model: str) -> list[str]:
    """Slugs a repo path might use for a model: qwen2_5_7b_instruct, qwen2_5_7b."""
    tail = _slug(model.split("/")[-1])
    short = re.sub(r"_(instruct|it|chat|hf)$", "", tail)
    return [a for a in dict.fromkeys([tail, short]) if a]


def vector_name(filename: str, model: str = "", keep_layer: bool = False,
                with_parent: bool = False) -> Optional[str]:
    """A steer() concept name from a file path, or None if none can be made.

    The model's own name is dropped (refusal_qwen3_8b -> refusal), and a file
    named only by its layer (refusal_qwen3_8b/layer_14.pt) is named after its
    directory.
    """
    p = Path(filename)
    strip = (lambda t: t) if keep_layer else (lambda t: _LAYER_RE.sub("", t))
    stem = strip(p.stem.lower())
    parent = strip(p.parent.name.lower()) if p.parent.name else ""
    if not _slug(stem):
        stem, parent = parent, ""
    elif with_parent and parent:
        stem = f"{parent}_{stem}"
    name = _slug(stem)
    for alias in model_aliases(model):  # longest first
        name = _slug(re.sub(rf"(^|_){alias}(_|$)", "_", name)) or name
    if name and not name[0].isalpha():
        name = f"v_{name}"
    name = name[:48].rstrip("_")
    return name if len(name) >= 2 else None


def layer_from_name(filename: str) -> Optional[int]:
    m = _LAYER_RE.search(Path(filename).stem.lower())
    return int(m.group(1)) if m else None


def names_model(path: str, model: str) -> bool:
    slug = f"_{_slug(path)}_"
    return any(f"_{a}_" in slug for a in model_aliases(model))


def _select_and_name(cands: list[Candidate], model: str, layer: Optional[int],
                     all_layers: bool, num_layers: Optional[int]):
    """Keep one layer per concept (unless all_layers) and give each kept file a unique name.

    Files that differ only by layer (refusal/layer_14.pt, refusal/layer_16.pt)
    are one concept; the kept one is the requested --layer, else the repo's
    best_layer (from sidecar metadata), else the one nearest the middle.
    """
    groups: dict[str, list[Candidate]] = {}
    for c in cands:
        if not c.ok:
            continue
        base = vector_name(c.meta["name"], model) if c.meta.get("name") else vector_name(c.filename, model)
        if base is None:
            c.ok, c.reason = False, "couldn't make a vector name from the filename"
            continue
        groups.setdefault(base, []).append(c)

    for base, group in groups.items():
        if len(group) > 1 and not all_layers and len({c.layer for c in group}) == len(group):
            mid = num_layers // 2 if num_layers else sorted(c.layer for c in group)[len(group) // 2]
            best = {c.meta.get("best_layer") for c in group}
            best = best.pop() if len(best) == 1 else None
            target = layer if layer is not None else best if best is not None else mid
            keep = min(group, key=lambda c: (abs(c.layer - target), c.layer))
            for c in group:
                if c is not keep:
                    c.ok, c.reason = False, (f"another layer of '{base}' (kept layer {keep.layer}"
                                             + (", the repo's best_layer" if keep.layer == best else "")
                                             + "; pick one with --layer, or keep all with --all-layers)")
            group[:] = [keep]

    taken = {base for base, group in groups.items() if len(group) == 1}
    for base, group in groups.items():
        if len(group) == 1:
            group[0].name = base
            continue
        for opts in ({"keep_layer": True}, {"keep_layer": True, "with_parent": True}):
            names = [vector_name(c.filename, model, **opts) for c in group]
            if None not in names and len(set(names)) == len(names) and not taken & set(names):
                for c, n in zip(group, names):
                    c.name = n
                taken |= set(names)
                break
        else:
            for c in group:
                c.ok, c.reason = False, f"couldn't make a unique vector name for '{base}' from the filename"


def file_metadata(path: Path) -> dict:
    """Scalar fields stored next to the vector in a dict .pt file ({'vector': ..., 'layer': 14, 'model': ...})."""
    if Path(path).suffix != ".pt":
        return {}
    import torch
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, (str, int, float, bool))}


def _short(e: Exception) -> str:
    msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
    return f"{type(e).__name__}: {msg[:160]}"


def inspect_file(c: Candidate, hidden_size: Optional[int], num_layers: Optional[int],
                 layer: Optional[int] = None, model: str = ""):
    """Fill in c.format/layer/shape from the downloaded file, or mark it skipped."""
    try:
        data = load_tensor_file(c.path)
    except Exception as e:
        c.ok, c.reason = False, f"can't load as a single tensor ({_short(e)})"
        return
    # Manifest and sidecar fields win over the dict .pt file's own
    c.meta = {**normalize_metadata(file_metadata(c.path)), **c.meta}
    meta_model = c.meta.get("model")
    if model and meta_model and meta_model.lower() != model.lower():
        c.ok, c.reason = False, f"file says it's for {meta_model}"
        return
    c.shape = list(data.shape)
    if c.meta.get("format") and c.meta["format"] != ("single" if data.dim() == 1 else "per_layer"):
        c.ok, c.reason = False, f"listed as {c.meta['format']}, but its shape is {c.shape}"
        return
    if data.dim() == 1:
        c.format = "single"
        named = c.meta.get("layer") if c.meta.get("layer") is not None else layer_from_name(c.filename)
        if named is not None and layer is not None and named != layer:
            c.ok, c.reason = False, f"layer {named}, not the requested --layer {layer}"
            return
        c.layer = named if named is not None else layer
        if c.layer is None:
            c.ok, c.reason = False, "single vector with no layer in its name: pass --layer"
            return
    elif data.dim() == 2:
        c.format = "per_layer"
        rows = data.shape[0]
        if num_layers is not None and rows == num_layers + 1:
            c.ok, c.reason = False, (f"{rows} rows for {num_layers} layers: probably includes the embedding "
                                     "output, so layer indices would be off by one")
            return
        if num_layers is not None and rows != num_layers:
            c.ok, c.reason = False, f"{rows} rows, but the model has {num_layers} layers"
            return
        given = layer if layer is not None else c.meta.get("layer", c.meta.get("best_layer"))
        if given is None:
            given = rows // 2
            c.notes.append(f"no --layer given; using the middle layer {given} (check the repo's card)")
        elif given >= rows:
            c.ok, c.reason = False, f"layer {given} out of range ({rows} rows)"
            return
        c.layer = given
    else:
        c.ok, c.reason = False, f"shape {c.shape}: expected [hidden] or [num_layers, hidden]"
        return
    if hidden_size is not None and data.shape[-1] != hidden_size:
        c.ok, c.reason = False, f"{data.shape[-1]} dims, but the model's hidden size is {hidden_size}"
        return
    c.ok = True


# Optional index fields a manifest can supply; copied into draft entries as is
_OPTIONAL_ENTRY_FIELDS = [k for k, (_, required) in ENTRY_FIELDS.items() if not required
                          and k not in ("revision", "safe_strength_min", "safe_strength_max")]


def index_entry(plan: AddPlan, c: Candidate, description: str = "", sign_convention: str = "") -> dict:
    """A draft community-index entry (trust: community) for one candidate. Fields
    nothing supplied are left as TODOs."""
    model_tail = re.sub(r"[^a-z0-9.-]+", "-", plan.model.split("/")[-1].lower())
    entry_id = f"{model_tail}-{c.name.replace('_', '-')}"[:64].rstrip("-._")
    m = c.meta
    entry = {
        "id": entry_id, "name": c.name, "model": plan.model,
        "repo_id": plan.repo_id, "repo_type": plan.repo_type, "revision": plan.revision,
        "filename": c.filename, "sha256": c.sha256, "format": c.format, "layer": c.layer,
        "description": description or m.get("description") or "TODO: what this vector does",
        "sign_convention": sign_convention or m.get("sign_convention") or "TODO: what + and - do",
        "extraction_method": m.get("extraction_method") or "TODO: how it was made",
        "license": m.get("license") or plan.license or "TODO",
        "contributors": m.get("contributors") or ["TODO"],
        "trust": "community",
    }
    entry.update({k: m[k] for k in _OPTIONAL_ENTRY_FIELDS if k in m})
    return entry


def draft_manifest(plan: AddPlan) -> dict:
    """A modulo-bundle.yaml for the plan's usable vectors, with TODOs where nothing said what they do."""
    vectors = []
    for c in plan.ok:
        v = {"name": c.name, "filename": c.filename, "layer": c.layer, "format": c.format,
             "description": c.meta.get("description") or "TODO: what this vector does",
             "sign_convention": c.meta.get("sign_convention") or "TODO: what + and - do"}
        if c.meta.get("extraction_method"):
            v["extraction_method"] = c.meta["extraction_method"]
        vectors.append(v)
    out = {"modulo_bundle": MANIFEST_VERSION, "model": plan.model}
    if plan.license:
        out["license"] = plan.license
    out["contributors"] = ["TODO"]
    out["vectors"] = vectors
    return out


def _load_manifest(source, path: str) -> dict:
    try:
        data = yaml.safe_load(source.fetch(path).read_text())
    except Exception as e:
        raise ValueError(f"{path}: not valid YAML ({_short(e)})")
    errors = validate_manifest(data)
    if errors:
        more = f" (and {len(errors) - 5} more)" if len(errors) > 5 else ""
        raise ValueError(f"{path} has problems:\n  " + "\n  ".join(errors[:5]) + more)
    return data


def _read_json(source, path: Optional[str]) -> dict:
    if path is None:
        return {}
    try:
        return normalize_metadata(json.loads(source.fetch(path).read_text()))
    except Exception:
        return {}


def _manifest_candidates(source, data: dict, sizes: dict, model: str, files, layer) -> list[Candidate]:
    cands = []
    for v in manifest_vectors(data):
        c = Candidate(filename=v["filename"], size=sizes.get(v["filename"]), meta=dict(v))
        cands.append(c)
        if v["model"] != model:
            c.reason = f"the manifest lists it for {v['model']}"
        elif files and not any(fnmatch.fnmatch(c.filename, pat) for pat in files):
            c.reason = "not matched by --files"
        elif layer is not None and v.get("layer") is not None and v["layer"] != layer:
            c.reason = f"layer {v['layer']}, not the requested --layer {layer}"
        elif c.filename not in sizes:
            c.reason = "listed in the manifest but not in the repo"
        else:
            c.ok = True  # to be downloaded and inspected
    return cands


def _guessed_candidates(source, sizes: dict, model: str, files) -> list[Candidate]:
    paths = [f for f in sizes if f.endswith(FILE_SUFFIXES)]
    if files:
        paths = [f for f in paths if any(fnmatch.fnmatch(f, pat) for pat in files)]
    if any(names_model(f, model) for f in paths):
        others = [f for f in paths if not names_model(f, model)]
        skipped = [Candidate(filename=f, size=sizes[f], reason=f"path doesn't name {model} "
                             "(other selected paths do); select it alone with --files to use it")
                   for f in others]
        paths = [f for f in paths if names_model(f, model)]
    else:
        skipped = []
    all_paths = set(sizes)
    cands = []
    for f in paths:
        own, shared = sidecar_paths(f, all_paths)
        shared_meta = _read_json(source, shared)
        shared_meta.pop("layer", None)  # a directory's 'layer' isn't this file's; best_layer is kept
        meta = {**shared_meta, **_read_json(source, own)}
        cands.append(Candidate(filename=f, size=sizes[f], ok=True, meta=meta))
    return cands + skipped


def plan_add(spec: str, model: str, files: Optional[list[str]] = None, layer: Optional[int] = None,
             repo_type: Optional[str] = None, max_files: int = DEFAULT_MAX_FILES,
             all_layers: bool = False) -> AddPlan:
    """Resolve, download and inspect a repo's (or local directory's) vector files. Registers nothing.

    With a modulo-bundle.yaml, its vectors for this model are the candidates,
    with its names, layers and descriptions. Without one, every vector file is
    a candidate, described by sidecar metadata where there is some:
    layer: for files named by layer, only that layer; for per_layer files, the row.
    If some of the (selected) paths name the model (refusal_qwen3_8b/...), only
    those are used, since other models' vectors can share its hidden size.
    """
    source = open_source(spec, repo_type)
    hidden, layers, note = model_dims(model)
    sizes = source.files()
    plan = AddPlan(repo_id=source.repo_id, repo_type=source.repo_type, revision=source.revision,
                   model=model, hidden_size=hidden, num_layers=layers, license=source.license,
                   candidates=[], notes=[note] if note else [])
    where = source.repo_id + (f"@{source.revision[:12]}" if source.revision else "")

    if MANIFEST_NAME in sizes:
        data = _load_manifest(source, MANIFEST_NAME)
        plan.manifest = MANIFEST_NAME
        plan.license = data.get("license", plan.license)
        cands = _manifest_candidates(source, data, sizes, model, files, layer)
        if not any(c.ok for c in cands):
            models = sorted({v["model"] for v in manifest_vectors(data)})
            plan.notes.append(f"{MANIFEST_NAME} has no usable vectors for {model} (it lists: {', '.join(models)})")
    else:
        cands = _guessed_candidates(source, sizes, model, files)
        if not any(c.ok for c in cands):
            plan.notes.append(f"no {'/'.join(FILE_SUFFIXES)} files" + (f" matching {files}" if files else "")
                              + f" in {where}")
    plan.candidates = cands
    todo = [c for c in cands if c.ok]
    if len(todo) > max_files:
        raise ValueError(f"{source.repo_id} has {len(todo)} vector files to check; narrow them with --files "
                         f"(e.g. --files '{Path(todo[0].filename).parent}/*') or raise --max-files. "
                         f"First few: {', '.join(c.filename for c in todo[:5])}")

    for c in todo:
        c.ok = False
        if c.size is not None and c.size > MAX_FILE_BYTES:
            c.reason = f"file is {c.size / 1e6:.0f} MB (max {MAX_FILE_BYTES / 1e6:.0f} MB)"
            continue
        try:
            c.path = source.fetch(c.filename)
        except Exception as e:
            c.reason = f"download failed ({_short(e)})"
            continue
        c.sha256 = file_sha256(c.path)
        inspect_file(c, hidden, layers, layer, model)

    if plan.manifest:
        _name_manifest_vectors(plan.candidates, model)
    else:
        _select_and_name(plan.candidates, model, layer, all_layers, layers)
    # Last gate: the same checks the index runs (sha, load, shape, norm)
    for c in plan.ok:
        problems = check_vector_file(index_entry(plan, c), c.path, hidden_size=hidden, num_layers=layers)
        if problems:
            c.ok, c.reason = False, problems[0].split(": ", 1)[-1]
    return plan


def _name_manifest_vectors(cands: list[Candidate], model: str):
    """The manifest's names, or one made from the filename; a clash skips the later vector."""
    taken = set()
    for c in cands:
        if not c.ok:
            continue
        name = c.meta.get("name") or vector_name(c.filename, model)
        if name is None or name in taken:
            c.ok, c.reason = False, (f"name '{name}' is already used by another vector in this bundle"
                                     if name else "couldn't make a vector name from the filename; set 'name'")
            continue
        c.name = name
        taken.add(name)


def register(plan: AddPlan, registry: VectorRegistry, description: str = "", sign_convention: str = "",
             replace: bool = False) -> list[str]:
    """Copy the plan's usable files into the registry's directory and register them.

    description/sign_convention override the manifest's or sidecar's for every
    vector, so pass them only for a single vector. Skips names you already
    have unless replace=True (same file from the same source counts as
    already added). Returns messages.
    """
    messages, added = [], False
    for c in plan.ok:
        source = plan.source_of(c)
        existing = registry.get(c.name)
        if existing and c.name not in registry.index_names:
            if existing.source == source:
                messages.append(f"{c.name}: already added from this file")
                continue
            if not replace:
                messages.append(f"{c.name}: skipped, you already have a vector with this name "
                                f"({existing.source or existing.file}); pass --replace to overwrite it")
                continue
        elif existing:
            messages.append(f"{c.name}: your copy will replace the community-index vector of this name")

        # Named like extracted vectors (honesty_layer14.npz); repo filenames are often just layer_14.pt
        stem, suffix = f"{c.name}_layer{c.layer}", Path(c.filename).suffix
        local = registry.vectors_dir / f"{stem}{suffix}"
        if local.exists() and file_sha256(local) != c.sha256:
            local = registry.vectors_dir / f"{stem}_{c.sha256[:8]}{suffix}"
        if local.exists() and file_sha256(local) != c.sha256:
            messages.append(f"{c.name}: skipped, {local} exists with different contents")
            continue
        registry.vectors_dir.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            shutil.copy(c.path, local)
        old_file = registry.vectors_dir / existing.file if existing else None
        if (old_file and old_file != local and existing.source.startswith(("hf:", "local:")) and old_file.exists()
                and not any(e.file == existing.file for e in registry.entries.values() if e is not existing)):
            old_file.unlink()  # a file an earlier `modulo add` copied in, now replaced

        desc = (description or c.meta.get("description")
                or f"Imported from {Path(plan.repo_id).name if plan.is_local else plan.repo_id}/{c.filename}; "
                   "no description yet.")
        registry.add_entry(VectorEntry(
            name=c.name, file=local.name, layer=c.layer, format=c.format,
            description=UNVERIFIED_IMPORT_PREFIX + desc,
            sign_convention=(sign_convention or c.meta.get("sign_convention")
                             or "Sign meaning unknown; check with `modulo verify`."),
            safe_strength_min=-COMMUNITY_DEFAULT_RANGE, safe_strength_max=COMMUNITY_DEFAULT_RANGE,
            source=source))
        added = True
        messages.append(f"{c.name}: added ({c.format}, layer {c.layer}, range ±{COMMUNITY_DEFAULT_RANGE:g}) "
                        f"as {local}")
    if added:
        registry.save()
    return messages

"""Search every place a vector can come from: your own registries, the community
index, and the discovery catalog (modulo_catalog.yaml).

Each hit says how usable it is:
    local      in vectors/<model_slug>/registry.yaml, loaded by SteeringAgent
    verified   in the community index, loaded by default
    community  in the community index, loaded with include_community=True
    catalog    a source listed in the catalog; not loadable as is
"""
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from modulo.community import DEFAULT_INDEX_PATH, load_index
from modulo.registry import DEFAULT_VECTORS_DIR, VectorRegistry, registry_dir_for

DEFAULT_CATALOG_PATH = Path(os.environ.get(
    "MODULO_CATALOG_PATH", Path(__file__).resolve().parent.parent / "modulo_catalog.yaml"))

_HF_URL_RE = re.compile(r"^https://huggingface\.co/(datasets/)?([\w.-]+/[\w.-]+)/?$")


@dataclass
class SearchHit:
    name: str                  # vector name (local/index) or catalog id
    status: str                # local | verified | community | catalog
    models: list[str]          # model id(s), or the registry directory for local hits
    description: str
    source: str                # registry path, index id, or URL
    next_step: str             # what to do to use it
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _matches(query_terms: list[str], *texts) -> bool:
    hay = " ".join(str(t) for t in texts if t).lower()
    return all(t in hay for t in query_terms)


def _catalog_model_matches(models: list[str], model: str) -> bool:
    """Catalog entries name models without an org (Qwen2.5-14B-Instruct). An
    entry with no models, or a generic one, is a toolkit for any model."""
    if not models:
        return True
    tail = model.split("/")[-1].lower()
    return any(m.lower() == tail or m.lower().startswith("generic") for m in models)


def hf_repo_from_url(url: str) -> Optional[str]:
    """'hf:owner/repo' or 'hf:datasets/owner/repo' for a HuggingFace repo URL, else None."""
    m = _HF_URL_RE.match(url or "")
    if not m or m.group(2).split("/")[0] in ("papers", "spaces", "docs", "blog"):
        return None
    return f"hf:{m.group(1) or ''}{m.group(2)}"


def _local_hits(terms, model, base_dir) -> list[SearchHit]:
    base = Path(base_dir or DEFAULT_VECTORS_DIR)
    dirs = [registry_dir_for(model, base)] if model else sorted(p for p in base.glob("*") if p.is_dir())
    hits = []
    for d in dirs:
        if not (d / "registry.yaml").exists():
            continue
        for e in VectorRegistry(d).entries.values():
            if _matches(terms, e.name, e.description, e.sign_convention, e.source):
                hits.append(SearchHit(
                    name=e.name, status="local", models=[model or d.name],
                    description=e.description, source=str(d / "registry.yaml"),
                    next_step="loaded by SteeringAgent",
                    extra={"layer": e.layer, "range": [e.safe_strength_min, e.safe_strength_max]}))
    return hits


def _index_hits(terms, model, index_path) -> list[SearchHit]:
    hits = []
    for e in load_index(index_path or DEFAULT_INDEX_PATH).get("entries", []):
        if not isinstance(e, dict) or (model and e.get("model") != model):
            continue
        if not _matches(terms, e.get("id"), e.get("name"), e.get("model"), e.get("description"),
                        e.get("sign_convention")):
            continue
        verified = e.get("trust") == "verified"
        hits.append(SearchHit(
            name=e.get("name", ""), status=e.get("trust", "community"), models=[e.get("model", "")],
            description=e.get("description", ""), source=f"index:{e.get('id')}",
            next_step=("loaded by SteeringAgent" if verified
                       else "loaded with SteeringAgent(..., include_community=True) and by `modulo run`"),
            extra={"layer": e.get("layer"), "repo_id": e.get("repo_id")}))
    return hits


def _catalog_hits(terms, model, catalog_path) -> list[SearchHit]:
    path = Path(catalog_path or DEFAULT_CATALOG_PATH)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    hits = []
    for e in data.get("entries", []):
        models = e.get("models") or []
        if model and not _catalog_model_matches(models, model):
            continue
        if not _matches(terms, e.get("id"), e.get("title"), e.get("notes"), e.get("mechanism"),
                        e.get("artifact_status"), " ".join(models), " ".join(e.get("behaviors") or [])):
            continue
        repo = hf_repo_from_url(e.get("url", ""))
        if repo and "downloadable" in e.get("artifact_status", ""):
            step = f"modulo add {repo} --model {model or '<model>'} --dry-run  (check which files fit)"
        else:
            step = f"no vector files to add directly ({e.get('artifact_status')}); see {e.get('url')}"
        hits.append(SearchHit(
            name=e.get("id", ""), status="catalog", models=models, description=e.get("title", ""),
            source=e.get("url", ""), next_step=step,
            extra={"behaviors": e.get("behaviors") or [], "artifact_status": e.get("artifact_status"),
                   "notes": e.get("notes")}))
    return hits


def search(query: str = "", model: Optional[str] = None, base_dir: Optional[Path] = None,
           index_path: Optional[Path] = None, catalog_path: Optional[Path] = None) -> list[SearchHit]:
    """Hits for every query word (case-insensitive), local first, then index, then catalog.

    With model, only that model's vectors, and catalog sources that list it
    (by name without the org) or no particular model.
    """
    terms = query.lower().split()
    return (_local_hits(terms, model, base_dir) + _index_hits(terms, model, index_path)
            + _catalog_hits(terms, model, catalog_path))

"""CPU tests for finding and adding vectors: `modulo search`, `modulo add` (with a fake
HuggingFace hub), file inspection and naming, and the registry changes they rely on."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import huggingface_hub
import modulo.importer as importer
from modulo import cli
from modulo.community import COMMUNITY_DEFAULT_RANGE, validate_entry
from modulo.importer import (UNVERIFIED_IMPORT_PREFIX, Candidate, inspect_file, names_model,
                             parse_spec, plan_add, register, vector_name)
from modulo.registry import VectorEntry, VectorRegistry, load_tensor_file, registry_dir_for
from modulo.search import hf_repo_from_url, search

MODEL = "org/Test-Model-4B-Instruct"
HIDDEN, LAYERS = 8, 4


# --- spec parsing and naming ---

@pytest.mark.parametrize("spec, expected", [
    ("hf:owner/repo", ("owner/repo", "model", None)),
    ("hf:owner/repo@abc123", ("owner/repo", "model", "abc123")),
    ("hf:datasets/owner/repo@v1", ("owner/repo", "dataset", "v1")),
    ("https://huggingface.co/datasets/owner/repo/", ("owner/repo", "dataset", None)),
    ("https://huggingface.co/owner/repo/tree/main", ("owner/repo", "model", "main")),
])
def test_parse_spec(spec, expected):
    assert parse_spec(spec) == expected


def test_parse_spec_rejects_non_repos():
    with pytest.raises(ValueError):
        parse_spec("hf:just-a-name")


@pytest.mark.parametrize("filename, name", [
    ("honesty_layer14.npz", "honesty"),
    ("joy.safetensors", "joy"),
    ("refusal_test_model_4b/layer_14.pt", "refusal"),       # named after its directory, model dropped
    ("vectors/anger_L10_caa.pt", "anger_caa"),
    ("12.pt", "v_12"),
])
def test_vector_name(filename, name):
    assert vector_name(filename, MODEL) == name


def test_names_model_matches_whole_tokens():
    assert names_model("refusal_test_model_4b/layer_14.pt", MODEL)        # without the -instruct suffix
    assert names_model("x/Test-Model-4B-Instruct.pt", MODEL)
    assert not names_model("refusal_test_model_4b_base2/layer_14.pt", "org/Test-Model-4")


def test_hf_repo_from_url():
    assert hf_repo_from_url("https://huggingface.co/datasets/a/b") == "hf:datasets/a/b"
    assert hf_repo_from_url("https://huggingface.co/a/b") == "hf:a/b"
    assert hf_repo_from_url("https://huggingface.co/papers/2501.09929") is None
    assert hf_repo_from_url("https://github.com/a/b") is None


# --- file loading and inspection ---

def test_load_tensor_file_dict_pt(tmp_path):
    p = tmp_path / "v.pt"
    torch.save({"vector": torch.ones(8), "layer": 3, "model": MODEL}, p)
    assert load_tensor_file(p).shape == (8,)
    torch.save({"direction": torch.ones(8), "norm": 1.0}, p)  # only one tensor: use it
    assert load_tensor_file(p).shape == (8,)
    torch.save({"a": torch.ones(8), "b": torch.ones(8)}, p)
    with pytest.raises(ValueError, match="no 'vector' key"):
        load_tensor_file(p)


def candidate(tmp_path, filename, data):
    path = tmp_path / Path(filename).name
    if isinstance(data, dict) or isinstance(data, torch.Tensor):
        torch.save(data, path)
    return Candidate(filename=filename, path=path)


@pytest.mark.parametrize("filename, data, layer, expect", [
    ("joy_layer2.pt", torch.ones(HIDDEN), None, ("single", 2)),
    ("joy.pt", torch.ones(HIDDEN), 1, ("single", 1)),
    ("axis.pt", torch.ones(LAYERS, HIDDEN), None, ("per_layer", LAYERS // 2)),
    ("axis.pt", torch.ones(LAYERS, HIDDEN), 3, ("per_layer", 3)),
    ("v.pt", {"vector": torch.ones(HIDDEN), "layer": 1, "model": MODEL}, None, ("single", 1)),
])
def test_inspect_file_ok(tmp_path, filename, data, layer, expect):
    c = candidate(tmp_path, filename, data)
    inspect_file(c, HIDDEN, LAYERS, layer, MODEL)
    assert c.ok, c.reason
    assert (c.format, c.layer) == expect


@pytest.mark.parametrize("filename, data, layer, reason", [
    ("joy.pt", torch.ones(HIDDEN), None, "pass --layer"),
    ("joy_layer2.pt", torch.ones(HIDDEN), 3, "not the requested --layer 3"),
    ("joy_layer2.pt", torch.ones(HIDDEN * 2), None, "hidden size"),
    ("axis.pt", torch.ones(LAYERS + 1, HIDDEN), None, "embedding"),
    ("axis.pt", torch.ones(LAYERS + 3, HIDDEN), None, "rows"),
    ("cube.pt", torch.ones(2, 2, HIDDEN), None, "expected [hidden]"),
    ("v.pt", {"vector": torch.ones(HIDDEN), "layer": 1, "model": "org/other"}, None, "for org/other"),
])
def test_inspect_file_skips(tmp_path, filename, data, layer, reason):
    c = candidate(tmp_path, filename, data)
    inspect_file(c, HIDDEN, LAYERS, layer, MODEL)
    assert not c.ok and reason in c.reason


# --- plan_add against a fake hub ---

@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake HF repo: write files into hub.dir, then call plan_add."""
    from huggingface_hub.hf_api import RepoFile
    d = tmp_path / "hub"
    d.mkdir()

    class FakeApi:
        def repo_info(self, repo_id, repo_type=None, revision=None):
            return type("Info", (), {"sha": "c0ffee" * 6 + "abcd", "card_data": {"license": "mit"}})()

        def list_repo_tree(self, repo_id, repo_type=None, revision=None, recursive=False):
            return [RepoFile(path=str(p.relative_to(d)), size=p.stat().st_size, oid="x")
                    for p in sorted(d.rglob("*")) if p.is_file()]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename, repo_type=None, revision=None: str(d / filename))
    monkeypatch.setattr(importer, "model_dims", lambda model, revision=None: (HIDDEN, LAYERS, None))

    def put(path, data):
        p = d / path
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, p)
        return p
    return type("Hub", (), {"dir": d, "put": staticmethod(put)})


def by_name(plan):
    return {c.name: c for c in plan.ok}


def test_bundle_keeps_one_layer_per_concept_and_only_this_model(hub):
    for layer in range(LAYERS):
        hub.put(f"refusal_test_model_4b/layer_{layer}.pt", torch.randn(HIDDEN))
        hub.put(f"refusal_other_model_7b/layer_{layer}.pt", torch.randn(HIDDEN))  # same hidden size!
    hub.put("humor_test_model_4b/layer_1.pt", torch.randn(HIDDEN))
    (hub.dir / "README.md").write_text("not a vector")

    plan = plan_add("hf:someone/bundle", MODEL)
    ok = by_name(plan)
    assert sorted(ok) == ["humor", "refusal"]
    assert ok["refusal"].layer == LAYERS // 2 and ok["humor"].layer == 1
    skipped = [c for c in plan.candidates if not c.ok]
    assert all("other_model" in c.filename or "another layer" in c.reason for c in skipped)
    assert plan.license == "mit" and plan.revision.startswith("c0ffee")

    plan = plan_add("hf:someone/bundle", MODEL, layer=3, files=["refusal*"])
    assert [(c.name, c.layer) for c in plan.ok] == [("refusal", 3)]  # other model still filtered out
    plan = plan_add("hf:someone/bundle", MODEL, all_layers=True, files=["refusal_test*"])
    assert len(plan.ok) == LAYERS and len({c.name for c in plan.ok}) == LAYERS


def test_too_many_files_asks_for_globs(hub):
    for i in range(3):
        hub.put(f"v{i}_layer1.pt", torch.randn(HIDDEN))
    with pytest.raises(ValueError, match="--files"):
        plan_add("hf:someone/many", MODEL, max_files=2)


def test_zero_vector_fails_index_checks(hub):
    hub.put("dead_layer1.pt", torch.zeros(HIDDEN))
    plan = plan_add("hf:someone/zero", MODEL)
    assert not plan.ok and "norm" in plan.candidates[0].reason


# --- register ---

def test_register_is_idempotent_and_protects_your_vectors(hub, tmp_path):
    hub.put("calm_layer1.pt", torch.randn(HIDDEN))
    registry = VectorRegistry(tmp_path / "vectors")
    plan = plan_add("hf:someone/calm", MODEL)
    assert "added" in register(plan, registry, "Calmer replies.", "+ calm, - agitated")[0]

    reloaded = VectorRegistry(tmp_path / "vectors")
    e = reloaded.get("calm")
    assert e.file == "calm_layer1.pt" and e.layer == 1
    assert e.description == UNVERIFIED_IMPORT_PREFIX + "Calmer replies."
    assert (e.safe_strength_min, e.safe_strength_max) == (-COMMUNITY_DEFAULT_RANGE, COMMUNITY_DEFAULT_RANGE)
    assert e.source.startswith("hf:model/someone/calm@") and e.load_vector(reloaded.vectors_dir).shape == (HIDDEN,)

    assert "already added" in register(plan, reloaded)[0]

    hub.put("calm_layer1.pt", torch.randn(HIDDEN))  # the repo's file changes
    plan2 = plan_add("hf:someone/calm", MODEL)
    assert "pass --replace" in register(plan2, reloaded)[0]
    assert "added" in register(plan2, reloaded, replace=True)[0]
    assert sorted(p.name for p in reloaded.vectors_dir.glob("calm*")) == ["calm_layer1_" + plan2.ok[0].sha256[:8] + ".pt"]


def test_index_entry_draft_is_valid(hub):
    hub.put("calm_layer1.pt", torch.randn(HIDDEN))
    plan = plan_add("hf:someone/calm", MODEL)
    entry = importer.index_entry(plan, plan.ok[0])
    assert validate_entry(entry) == []
    assert entry["trust"] == "community" and "safe_strength_min" not in entry


def test_cli_add_json(hub, tmp_path, monkeypatch, capsys):
    hub.put("calm_layer1.pt", torch.randn(HIDDEN))
    monkeypatch.setattr(cli, "registry_dir_for", lambda model: tmp_path / "vectors")
    assert cli.main(["add", "hf:someone/calm", "--model", MODEL, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["candidates"][0]["name"] == "calm" and "added" in out["messages"][0]
    assert cli.main(["add", "hf:someone/calm", "--model", MODEL, "--json", "--files", "nothing*"]) == 1


# --- registry save keeps the header ---

def test_registry_save_keeps_header(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("# My vectors.\n# Notes.\n[]\n")
    registry = VectorRegistry(tmp_path)
    registry.add_entry(VectorEntry(name="joy", file="joy.pt", layer=1))
    registry.save()
    text = path.read_text()
    assert text.startswith("# My vectors.\n# Notes.\n")
    assert yaml.safe_load(text)[0]["name"] == "joy"


# --- search ---

def test_search_across_sources(tmp_path):
    base = tmp_path / "vectors"
    reg = VectorRegistry(registry_dir_for(MODEL, base))
    reg.vectors_dir.mkdir(parents=True)
    np.savez(reg.vectors_dir / "joy.npz", vector=np.ones(HIDDEN, dtype=np.float32))
    reg.add_entry(VectorEntry(name="joy", file="joy.npz", layer=1, description="Happier tone."))
    reg.save()
    index = tmp_path / "index.yaml"
    index.write_text(yaml.safe_dump({"entries": [
        {"id": "x-calm", "name": "calm", "model": MODEL, "trust": "community", "description": "Calm tone.",
         "layer": 1, "repo_id": "a/b"}]}))
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump({"entries": [
        {"id": "tone-pack", "title": "Tone vectors", "url": "https://huggingface.co/a/tones",
         "artifact_status": "downloadable_vectors", "models": ["Test-Model-4B-Instruct"], "behaviors": ["tone"]},
        {"id": "other-pack", "title": "Tone for others", "url": "https://github.com/a/b",
         "artifact_status": "research_code", "models": ["Other-7B"], "behaviors": ["tone"]},
    ]}))
    kw = dict(base_dir=base, index_path=index, catalog_path=catalog)

    hits = search("tone", model=MODEL, **kw)
    assert [(h.name, h.status) for h in hits] == [("joy", "local"), ("calm", "community"), ("tone-pack", "catalog")]
    assert hits[2].next_step.startswith(f"modulo add hf:a/tones --model {MODEL}")
    assert [h.name for h in search("tone", **kw)][-1] == "other-pack"  # no model filter
    assert search("happier TONE", **kw)[0].name == "joy"                # every word, any case
    assert search("nothing-matches", **kw) == []

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


def test_index_entry_draft_needs_its_todos_filled(hub):
    hub.put("calm_layer1.pt", torch.randn(HIDDEN))
    plan = plan_add("hf:someone/calm", MODEL)
    entry = importer.index_entry(plan, plan.ok[0])
    assert entry["trust"] == "community" and "safe_strength_min" not in entry
    errors = validate_entry(entry)
    assert errors and all("placeholder" in e for e in errors)
    entry.update(description="Calmer.", sign_convention="+ calm", extraction_method="CAA",
                 contributors=["Me"])
    assert validate_entry(entry) == []


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


# --- bundle manifests (modulo-bundle.yaml) ---

from modulo.bundle import MANIFEST_NAME, validate_manifest  # noqa: E402


def manifest(**overrides):
    data = {
        "modulo_bundle": 1, "model": MODEL, "license": "mit", "contributors": ["A Researcher"],
        "extraction_method": "CAA",
        "vectors": [
            {"name": "calm", "filename": "calm.pt", "layer": 1,
             "description": "Calmer replies.", "sign_convention": "+ calm, - agitated",
             "test_prompts": ["My train is late again."]},
            {"name": "joy", "filename": "axis.pt", "layer": 2,
             "description": "Happier tone.", "sign_convention": "+ happy, - glum"},
        ],
    }
    data.update(overrides)
    return data


def test_valid_manifest():
    assert validate_manifest(manifest()) == []


@pytest.mark.parametrize("change, expected", [
    ({"modulo_bundle": 2}, "format version"),
    ({"vectors": []}, "non-empty list"),
    ({"trust": "verified"}, "'trust' isn't set in a manifest"),
    ({"filename": "x.pt"}, "can't be a top-level default"),
    ({"colour": "red"}, "unknown top-level field"),
])
def test_bad_manifest_top_level(change, expected):
    assert any(expected in e for e in validate_manifest(manifest(**change)))


@pytest.mark.parametrize("vector_change, expected", [
    ({"description": None}, "'description' has type NoneType"),
    ({"safe_strength_max": 30.0}, "isn't set in a manifest"),
    ({"name": "Calm Vector"}, "'name' must be"),
    ({"filename": "calm.pkl"}, "'filename' must end with"),
    ({"format": "stacked"}, "'format' must be one of"),
    ({"layer": -1}, "'layer' must be >= 0"),
    ({"test_prompts": ["p"] * 6}, "'test_prompts' must be"),
    ({"name": "joy"}, "already named 'joy'"),
    ({"sha256": "0" * 64}, "isn't set in a manifest"),
])
def test_bad_manifest_vectors(vector_change, expected):
    data = manifest()
    data["vectors"][0].update(vector_change)
    assert any(expected in e for e in validate_manifest(data))


def test_manifest_needs_model_and_description_somewhere():
    data = manifest()
    del data["model"]
    del data["vectors"][1]["description"]
    errors = validate_manifest(data)
    assert any("missing 'model'" in e for e in errors) and any("missing 'description'" in e for e in errors)


def write_manifest(hub, data):
    (hub.dir / MANIFEST_NAME).write_text(yaml.safe_dump(data))


def test_manifest_bundle_is_described_and_filtered(hub, tmp_path):
    hub.put("calm.pt", torch.randn(HIDDEN))
    hub.put("axis.pt", torch.randn(LAYERS, HIDDEN))
    hub.put("unlisted_layer1.pt", torch.randn(HIDDEN))          # not in the manifest: ignored
    data = manifest()
    data["vectors"].append({"name": "other", "filename": "calm.pt", "layer": 3, "model": "org/other",
                            "description": "x", "sign_convention": "y"})
    data["vectors"].append({"name": "gone", "filename": "missing.pt", "layer": 1,
                            "description": "x", "sign_convention": "y"})
    write_manifest(hub, data)

    plan = plan_add("hf:someone/bundle", MODEL)
    assert plan.manifest == MANIFEST_NAME
    ok = by_name(plan)
    assert sorted(ok) == ["calm", "joy"]
    assert (ok["joy"].format, ok["joy"].layer) == ("per_layer", 2)  # the manifest's row, not the middle
    reasons = {c.filename: c.reason for c in plan.candidates if not c.ok}
    assert "lists it for org/other" in reasons["calm.pt"] and "not in the repo" in reasons["missing.pt"]
    assert "unlisted_layer1.pt" not in reasons

    registry = VectorRegistry(tmp_path / "vectors")
    register(plan, registry)
    assert registry.get("calm").description == UNVERIFIED_IMPORT_PREFIX + "Calmer replies."
    assert registry.get("joy").sign_convention == "+ happy, - glum"

    entry = importer.index_entry(plan, ok["calm"])
    assert validate_entry(entry) == []
    assert "TODO" not in json.dumps(entry)
    assert entry["test_prompts"] == ["My train is late again."] and entry["contributors"] == ["A Researcher"]


def test_manifest_shape_must_match_its_format(hub):
    hub.put("calm.pt", torch.randn(LAYERS, HIDDEN))
    data = manifest()
    data["vectors"] = [{**data["vectors"][0], "format": "single"}]
    write_manifest(hub, data)
    plan = plan_add("hf:someone/bundle", MODEL)
    assert not plan.ok and "listed as single" in plan.candidates[0].reason


def test_invalid_manifest_is_an_error(hub):
    write_manifest(hub, manifest(trust="verified"))
    with pytest.raises(ValueError, match="has problems"):
        plan_add("hf:someone/bundle", MODEL)


# --- sidecar metadata (rotalabs layout) ---

def test_sidecars_give_name_layer_and_best_layer(hub):
    d = "refusal_test_model_4b"
    for layer in range(LAYERS):
        hub.put(f"{d}/layer_{layer}.pt", torch.randn(HIDDEN))
        (hub.dir / d / f"layer_{layer}.json").write_text(json.dumps(
            {"behavior": "refusal", "layer_index": layer, "model_name": MODEL, "extraction_method": "caa"}))
    (hub.dir / d / "metadata.json").write_text(json.dumps(
        {"behavior": "refusal", "model_name": MODEL, "best_layer": 1}))
    plan = plan_add("hf:someone/rota", MODEL)
    [c] = plan.ok
    assert (c.name, c.layer) == ("refusal", 1)     # best_layer, not the middle (2)
    assert importer.index_entry(plan, c)["extraction_method"] == "caa"
    assert any("best_layer" in x.reason for x in plan.candidates if not x.ok)


def test_sidecar_naming_another_model_is_skipped(hub):
    hub.put("calm_layer1.pt", torch.randn(HIDDEN))
    (hub.dir / "calm_layer1.json").write_text(json.dumps({"model_name": "org/other"}))
    plan = plan_add("hf:someone/x", MODEL)
    assert not plan.ok and "for org/other" in plan.candidates[0].reason


# --- local directories and drafts ---

def test_local_directory_and_manifest_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(importer, "model_dims", lambda model, revision=None: (HIDDEN, LAYERS, None))
    src = tmp_path / "mine"
    src.mkdir()
    torch.save(torch.randn(HIDDEN), src / "calm_layer1.pt")
    torch.save(torch.randn(LAYERS, HIDDEN), src / "axis.pt")

    plan = plan_add(str(src), MODEL)
    assert plan.is_local and plan.revision is None and sorted(by_name(plan)) == ["axis", "calm"]
    draft = importer.draft_manifest(plan)
    assert any("placeholder" in e for e in validate_manifest(draft))  # unfilled TODOs never reach a prompt
    draft["contributors"] = ["Me"]
    for v in draft["vectors"]:
        v["description"], v["sign_convention"] = f"About {v['name']}.", "+ more, - less"
    assert validate_manifest(draft) == []
    (src / MANIFEST_NAME).write_text(yaml.safe_dump(draft))

    plan = plan_add(str(src), MODEL)
    assert plan.manifest and by_name(plan)["axis"].meta["description"] == "About axis."
    registry = VectorRegistry(tmp_path / "vectors")
    register(plan, registry)
    assert registry.get("calm").source.startswith(f"local:{src}/calm_layer1.pt")

    monkeypatch.setattr(cli, "registry_dir_for", lambda model: tmp_path / "vectors")
    assert cli.main(["add", str(src), "--model", MODEL, "--as-index-entry"]) == 1  # needs a hosted repo
    capsys.readouterr()
    assert cli.main(["add", str(src), "--model", MODEL, "--as-manifest"]) == 0
    out = capsys.readouterr().out
    assert yaml.safe_load(out)["vectors"][0]["description"].startswith("About")  # stdout is only YAML

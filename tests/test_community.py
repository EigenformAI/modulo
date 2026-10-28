"""CPU tests for the community vector index: entry validation, file checks, change
detection, and how the library loads the index."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import modulo.community as community
from community.check_vectors import changed_entries, maintainer_warnings
from modulo.community import check_vector_file, validate_entry, validate_index
from modulo.registry import VectorEntry, VectorRegistry
from modulo.reports import file_sha256

ROOT = Path(__file__).resolve().parent.parent
BUNDLED = ROOT / "vectors/Qwen_Qwen2.5-7B-Instruct/assistant_axis.pt"


def make_entry(**overrides):
    entry = {
        "id": "test-model-honesty", "name": "honesty", "model": "org/test-model",
        "repo_id": "someone/vectors", "repo_type": "model", "filename": "honesty.pt",
        "sha256": "0" * 64, "format": "single", "layer": 2,
        "description": "Honesty.", "sign_convention": "+ honest, - evasive",
        "extraction_method": "contrast pairs", "license": "mit",
        "contributors": ["A Researcher"], "trust": "community",
    }
    entry.update(overrides)
    return entry


def entry_for_file(path, **overrides):
    return make_entry(filename=Path(path).name, sha256=file_sha256(path), **overrides)


# --- entry and index validation ---

def test_shipped_index_is_valid():
    assert validate_index(yaml.safe_load((ROOT / "community/vectors.yaml").read_text())) == []


def test_minimal_entry_is_valid():
    assert validate_entry(make_entry()) == []


@pytest.mark.parametrize("overrides, expected", [
    ({"license": None}, "'license' has type NoneType"),
    ({"layer": True}, "'layer' has type bool"),
    ({"extra": 1}, "unknown field 'extra'"),
    ({"sha256": "abc"}, "'sha256' must be 64"),
    ({"trust": "trusted"}, "'trust' must be one of"),
    ({"format": "stacked"}, "'format' must be one of"),
    ({"filename": "v.pkl"}, "'filename' must end with"),
    ({"name": "Honesty Vector"}, "'name' must be"),
    ({"contributors": []}, "'contributors' must be"),
    ({"test_prompts": ["p"] * 6}, "'test_prompts' must be"),
    ({"safe_strength_min": -5.0}, "set both"),
    ({"safe_strength_min": 1.0, "safe_strength_max": 5.0}, "min < 0 < max"),
    ({"trust": "verified"}, "need a safe strength range"),
])
def test_bad_entries_rejected(overrides, expected):
    errors = validate_entry(make_entry(**overrides))
    assert any(expected in e for e in errors), errors


def test_missing_required_field_rejected():
    entry = make_entry()
    del entry["sign_convention"]
    assert any("missing field 'sign_convention'" in e for e in validate_entry(entry))


def test_index_rejects_duplicates():
    a = make_entry()
    errors = validate_index({"entries": [a, dict(a)]})
    assert any("duplicate id" in e for e in errors)
    errors = validate_index({"entries": [a, make_entry(id="other-id")]})
    assert any("already uses name 'honesty'" in e for e in errors)
    assert validate_index({"entries": [a, make_entry(id="other-id", model="org/other")]}) == []


# --- file checks ---

def test_bundled_vector_passes_against_qwen_dims():
    entry = entry_for_file(BUNDLED, format="per_layer", layer=14)
    assert check_vector_file(entry, BUNDLED, hidden_size=3584, num_layers=28) == []


def test_sha_mismatch(tmp_path):
    p = tmp_path / "v.pt"
    torch.save(torch.ones(8), p)
    errors = check_vector_file(make_entry(filename="v.pt"), p)
    assert errors and "sha256 mismatch" in errors[0]


def test_wrong_hidden_size_and_layer(tmp_path):
    p = tmp_path / "v.pt"
    torch.save(torch.ones(8), p)
    entry = entry_for_file(p)
    assert check_vector_file(entry, p, hidden_size=8, num_layers=4) == []
    assert "hidden size 16" in check_vector_file(entry, p, hidden_size=16)[0]
    assert "out of range" in check_vector_file(entry, p, num_layers=2)[0]


def test_zero_vector_rejected(tmp_path):
    p = tmp_path / "v.npz"
    np.savez(p, vector=np.zeros(8, dtype=np.float32))
    assert "norm" in check_vector_file(entry_for_file(p), p)[0]


def test_unloadable_and_wrong_shape_rejected(tmp_path):
    junk = tmp_path / "v.pt"
    junk.write_bytes(b"not a tensor")
    assert "could not load" in check_vector_file(entry_for_file(junk), junk)[0]
    stacked = tmp_path / "s.pt"
    torch.save(torch.ones(4, 8), stacked)  # per-layer data declared as single
    assert "1-D vector" in check_vector_file(entry_for_file(stacked), stacked)[0]


def test_safetensors_single_and_per_layer(tmp_path):
    from safetensors.torch import save_file
    single = tmp_path / "one.safetensors"
    save_file({"v": torch.ones(8)}, single)
    assert check_vector_file(entry_for_file(single), single, hidden_size=8) == []
    per_layer = tmp_path / "layers.safetensors"
    save_file({"v": torch.ones(4, 8)}, per_layer)
    assert check_vector_file(entry_for_file(per_layer, format="per_layer"), per_layer, hidden_size=8) == []
    two = tmp_path / "two.safetensors"
    save_file({"a": torch.ones(8), "b": torch.ones(8)}, two)
    assert "exactly one tensor" in check_vector_file(entry_for_file(two), two)[0]


# --- change detection ---

def test_changed_entries_and_maintainer_warnings():
    kept = make_entry(id="kept")
    verified = make_entry(id="ver", name="joy", trust="verified",
                          safe_strength_min=-10.0, safe_strength_max=10.0)
    base = {"entries": [kept, verified, make_entry(id="gone", name="gone")]}
    edited_verified = {**verified, "description": "Changed."}
    new = make_entry(id="new", name="calm")
    index = {"entries": [kept, edited_verified, new]}

    changed, removed = changed_entries(index, base)
    assert [e["id"] for e in changed] == ["ver", "new"]
    assert removed == ["gone"]
    warnings = maintainer_warnings(changed, base)
    assert len(warnings) == 1 and warnings[0].startswith("ver:")
    assert maintainer_warnings([{**verified, "trust": "community"}], base)  # demotion is flagged too


# --- library loading of the index ---

MODEL = "org/test-model"


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A fake hub: fetch_vector serves files from tmp_path/remote and records calls."""
    d = tmp_path / "remote"
    d.mkdir()
    calls = []

    def fake_fetch(entry):
        calls.append(entry["id"])
        return d / entry["filename"], None

    monkeypatch.setattr(community, "fetch_vector", fake_fetch)
    return SimpleNamespace(dir=d, calls=calls)


def write_index(tmp_path, entries):
    path = tmp_path / "index.yaml"
    path.write_text(yaml.safe_dump({"entries": entries}))
    return path


def hosted(remote, filename, **overrides):
    p = remote.dir / filename
    torch.save(torch.ones(8), p)
    return entry_for_file(p, id=overrides.pop("id", filename.split(".")[0]), **overrides)


def load(tmp_path, index, **kwargs):
    return VectorRegistry.for_model(MODEL, base_dir=tmp_path / "vectors", auto_extract=False,
                                    index_path=index, **kwargs)


def test_verified_by_default_community_opt_in(tmp_path, remote):
    index = write_index(tmp_path, [
        hosted(remote, "joy.pt", name="joy", trust="verified", safe_strength_min=-12.0, safe_strength_max=6.0),
        hosted(remote, "calm.pt", name="calm"),
        hosted(remote, "other.pt", name="other", model="org/other-model"),
    ])
    reg = load(tmp_path, index)
    assert reg.list_names() == ["joy"]
    joy = reg.get("joy")
    assert (joy.safe_strength_min, joy.safe_strength_max) == (-12.0, 6.0)
    assert joy.source == "index:joy (verified)" and not joy.description.startswith("[")
    assert (reg.vectors_dir / "joy.pt").exists()

    reg = load(tmp_path, index, include_community=True)
    assert sorted(reg.list_names()) == ["calm", "joy"]
    calm = reg.get("calm")
    assert calm.description.startswith(community.UNVERIFIED_PREFIX)
    r = community.COMMUNITY_DEFAULT_RANGE
    assert (calm.safe_strength_min, calm.safe_strength_max) == (-r, r)


def test_cached_file_reused_without_download(tmp_path, remote):
    index = write_index(tmp_path, [hosted(remote, "joy.pt", name="joy", trust="verified",
                                          safe_strength_min=-5.0, safe_strength_max=5.0)])
    load(tmp_path, index)
    load(tmp_path, index)
    assert remote.calls == ["joy"]


def test_mismatched_local_file_is_skipped_not_overwritten(tmp_path, remote, capsys):
    index = write_index(tmp_path, [hosted(remote, "joy.pt", name="joy", trust="verified",
                                          safe_strength_min=-5.0, safe_strength_max=5.0)])
    local = tmp_path / "vectors" / "org_test-model" / "joy.pt"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"someone else's file")
    reg = load(tmp_path, index)
    assert reg.list_names() == [] and remote.calls == []
    assert local.read_bytes() == b"someone else's file"
    assert "sha256 mismatch" in capsys.readouterr().out


def test_bad_download_is_not_cached(tmp_path, remote):
    entry = hosted(remote, "joy.pt", name="joy", trust="verified", safe_strength_min=-5.0, safe_strength_max=5.0)
    torch.save(torch.zeros(8), remote.dir / "joy.pt")  # hub file no longer matches the pinned sha
    reg = load(tmp_path, write_index(tmp_path, [entry]))
    assert reg.list_names() == []
    assert not (tmp_path / "vectors" / "org_test-model" / "joy.pt").exists()


def test_model_dims_checked_when_model_given(tmp_path, remote):
    index = write_index(tmp_path, [hosted(remote, "joy.pt", name="joy", trust="verified",
                                          safe_strength_min=-5.0, safe_strength_max=5.0)])
    wrong = SimpleNamespace(config=SimpleNamespace(hidden_size=16, num_hidden_layers=4))
    assert load(tmp_path, index, model=wrong).list_names() == []
    right = SimpleNamespace(config=SimpleNamespace(hidden_size=8, num_hidden_layers=4))
    assert load(tmp_path, index, model=right).list_names() == ["joy"]


def test_own_vectors_win_and_index_vectors_are_never_saved(tmp_path, remote, capsys):
    index = write_index(tmp_path, [
        hosted(remote, "joy.pt", name="joy", trust="verified", safe_strength_min=-5.0, safe_strength_max=5.0),
        hosted(remote, "calm.pt", name="calm", trust="verified", safe_strength_min=-5.0, safe_strength_max=5.0),
    ])
    own_dir = tmp_path / "vectors" / "org_test-model"
    own_dir.mkdir(parents=True)
    (own_dir / "registry.yaml").write_text(yaml.safe_dump([{"name": "joy", "file": "mine.npz", "layer": 3}]))

    reg = load(tmp_path, index)
    assert reg.get("joy").file == "mine.npz" and reg.get("calm").source.startswith("index:")
    assert "shadowed by your own vector" in capsys.readouterr().out

    reg.save()
    assert [e["name"] for e in yaml.safe_load((own_dir / "registry.yaml").read_text())] == ["joy"]
    reg.add_entry(VectorEntry(name="calm", file="my_calm.npz", layer=1))  # your own vector takes over the name
    reg.save()
    assert [e["name"] for e in yaml.safe_load((own_dir / "registry.yaml").read_text())] == ["joy", "calm"]


def test_invalid_index_is_ignored(tmp_path, remote, capsys):
    index = write_index(tmp_path, [make_entry(sha256="bad")])
    assert load(tmp_path, index).list_names() == []
    assert "ignoring invalid index" in capsys.readouterr().out


def test_repo_index_serves_qwen_from_the_committed_file(tmp_path, remote):
    local = tmp_path / "vectors" / "Qwen_Qwen2.5-7B-Instruct"
    local.mkdir(parents=True)
    shutil.copy(BUNDLED, local / "assistant_axis.pt")
    reg = VectorRegistry.for_model("Qwen/Qwen2.5-7B-Instruct", base_dir=tmp_path / "vectors", auto_extract=False)
    e = reg.get("assistant_axis")
    assert e.layer == 14 and e.format == "per_layer" and e.source.endswith("(verified)")
    assert remote.calls == []

"""Explicit scalar reuse across owners; no checkpoints or CUDA required."""
import gc
import json
import traceback
import weakref

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from modulo import SteeringAgent, VectorRegistry


KEY = "checkpoint@revision; tokenizer@revision; pristine; adapters=none; cpu-fp32-sdpa-v1"


def make_agent(tmp_path):
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=12, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        eos_token_id=None, pad_token_id=0, attn_implementation="sdpa",
    )).eval()
    raw = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "user": 2, "assistant": 3,
                               "Tell": 4, "me": 5, "a": 6, "little": 7, "about": 8,
                               "how": 9, "you": 10, "questions": 11}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}"
                               "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
    registry = VectorRegistry(tmp_path)
    return SteeringAgent(model, tokenizer, registry=registry, model_name="tiny", menu_seed=23)


@pytest.fixture
def agent(tmp_path):
    return make_agent(tmp_path)


def test_cold_import_roundtrip_skips_calibration_and_copies_scalars(agent, tmp_path, monkeypatch):
    expected = agent.residual_norm(0)
    record = json.loads(json.dumps(agent.export_calibration(key=KEY)))
    other = make_agent(tmp_path)
    other.import_calibration(record, key=KEY)
    monkeypatch.setattr(other, "_generate_ids", lambda *a, **k: pytest.fail("hit generated calibration"))
    record["norms"]["0"] = expected * 2
    assert other.residual_norm(0) == expected
    assert other.export_calibration(key=KEY)["norms"] == {"0": expected}


def test_default_missing_layer_none_and_reset_use_ordinary_calibration(agent, tmp_path, monkeypatch):
    other = make_agent(tmp_path)
    expected = agent.residual_norm(0)
    record = agent.export_calibration(key=KEY)
    calls = []
    generate = other._generate_ids
    def counted(*args, **kwargs):
        calls.append(1)
        return generate(*args, **kwargs)
    monkeypatch.setattr(other, "_generate_ids", counted)
    assert other.export_calibration(key=KEY)["norms"] == {}  # export is lazy
    assert other.residual_norm(0) == expected
    other.import_calibration(record, key=KEY)
    assert other.residual_norm(0) == expected
    assert other.residual_norm(1) == agent.residual_norm(1)
    assert len(calls) == 2  # ordinary first call plus missing layer
    other.clear_calibration()
    assert other.residual_norm(0) == expected
    other.import_calibration(None, key=KEY)
    assert other.residual_norm(0) == expected
    assert len(calls) == 4


@pytest.mark.parametrize("change", [
    lambda r: r.update(schema=2), lambda r: r.update(schema=True),
    lambda r: r.update(extra=1), lambda r: r.update(key="other-model"),
    lambda r: r["profile"].update(placements=[["torch.float32", "cuda:0"]]),
    lambda r: r.update(norms={"0": True}), lambda r: r.update(norms={"0": 0}),
    lambda r: r.update(norms={"0": -1}), lambda r: r.update(norms={"0": float("nan")}),
    lambda r: r.update(norms={"0": float("inf")}), lambda r: r.update(norms={"0": "3"}),
    lambda r: r.update(norms={"0": 1.0, "1": False}),
    lambda r: r.update(norms={"02": 1.0}), lambda r: r.update(norms={"2": 1.0}),
    lambda r: r.update(norms={"-1": 1.0}), lambda r: r.update(norms={0: 1.0}),
    lambda r: r.update(norms=[]),
])
def test_invalid_import_clears_previous_norms_atomically(agent, change):
    agent.residual_norm(0)
    record = agent.export_calibration(key=KEY)
    change(record)
    with pytest.raises(ValueError):
        agent.import_calibration(record, key=KEY)
    assert agent.export_calibration(key=KEY)["norms"] == {}


def test_operator_identity_dtype_and_training_reject_stale_record(agent):
    agent.residual_norm(0)
    record = agent.export_calibration(key=KEY)
    for key in ("different-checkpoint", "adapter-scale=2", "tokenizer-v2", "weights:mutation-uuid", ""):
        with pytest.raises(ValueError):
            agent.import_calibration(record, key=key)
    agent.model.to(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="profile"):
        agent.import_calibration(record, key=KEY)
    agent.model.float().train()
    with pytest.raises(ValueError, match="eval"):
        agent.import_calibration(record, key=KEY)


def test_export_cannot_relabel_an_imported_or_exported_memo(agent):
    agent.residual_norm(0)
    record = agent.export_calibration(key=KEY)
    with pytest.raises(ValueError, match="Clear"):
        agent.export_calibration(key="new-adapter")
    agent.import_calibration(record, key=KEY)
    with pytest.raises(ValueError, match="Clear"):
        agent.export_calibration(key="new-adapter")
    agent.clear_calibration()
    assert agent.export_calibration(key="new-adapter")["norms"] == {}


def test_unserializable_recursive_profile_raises_validation_error_and_clears(agent):
    agent.residual_norm(0)
    record = agent.export_calibration(key=KEY)
    profile = record["profile"]
    for _ in range(1500):
        profile = {"nested": profile}
    record["profile"] = profile
    with pytest.raises(ValueError, match="profile"):
        agent.import_calibration(record, key=KEY)
    assert agent.export_calibration(key=KEY)["norms"] == {}


def test_import_replaces_key_and_layers_without_a_separate_clear(agent, tmp_path):
    agent.residual_norm(0)
    agent.export_calibration(key=KEY)
    other = make_agent(tmp_path)
    expected = other.residual_norm(1)
    record = other.export_calibration(key="new-profile")
    agent.import_calibration(record, key="new-profile")
    assert agent.export_calibration(key="new-profile")["norms"] == {"1": expected}


def test_negative_index_calibration_remains_default_only(agent):
    expected = agent.residual_norm(-1)
    assert expected == agent.residual_norm(1)
    with pytest.raises(ValueError, match="valid layer strings"):
        agent.export_calibration(key=KEY)
    assert agent.residual_norm(-1) == expected
    agent.clear_calibration()
    assert agent.export_calibration(key=KEY)["norms"] == {}


def test_record_does_not_keep_model_or_agent_alive(tmp_path):
    agent = make_agent(tmp_path)
    agent.residual_norm(0)
    record = agent.export_calibration(key=KEY)
    refs = [weakref.ref(agent), weakref.ref(agent.model), weakref.ref(next(agent.model.parameters()))]
    del agent
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert json.loads(json.dumps(record)) == record


def test_original_calibration_exception_keeps_traceback_and_removes_hook(agent, monkeypatch):
    inputs = agent._encode([{"role": "user", "content": "hello"}])
    monkeypatch.setattr(agent, "_generate_ids", lambda *a, **k: inputs.input_ids)
    def fail_forward(*args, **kwargs):
        raise RuntimeError("original diagnostic")
    monkeypatch.setattr(agent.model, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="original diagnostic") as error:
        agent.residual_norm(0)
    assert "fail_forward" in "".join(traceback.format_exception(error.value))
    assert agent.export_calibration(key=KEY)["norms"] == {}
    assert not agent.model.model.layers[0]._forward_hooks


def test_retained_error_preserves_diagnostics_until_caller_releases_it(tmp_path):
    def fail_import():
        agent = make_agent(tmp_path)
        record = agent.export_calibration(key=KEY)
        refs = [weakref.ref(agent), weakref.ref(agent.model)]
        try:
            agent.import_calibration(record, key="different-weight-state")
        except ValueError as error:
            return error, refs
    error, refs = fail_import()
    gc.collect()
    assert all(ref() is not None for ref in refs)  # ordinary traceback ownership
    assert "fail_import" in "".join(traceback.format_exception(error))
    del error
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_real_adapter_transitions_require_new_operator_keys(agent):
    peft = pytest.importorskip("peft")
    agent.model = peft.get_peft_model(agent.model, peft.LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"],
    )).eval()
    with torch.no_grad():
        for name, parameter in agent.model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.2)
    key = KEY + ";lora@revision;scale=1;enabled"
    agent.residual_norm(0)
    record = agent.export_calibration(key=key)
    # The cheap profile is deliberately not an adapter-state detector.
    for module in agent.model.modules():
        if isinstance(getattr(module, "scaling", None), dict):
            module.scaling["default"] = 2.0
    agent.import_calibration(record, key=key)  # caller's unchanged assertion is trusted
    with pytest.raises(ValueError, match="key"):
        agent.import_calibration(record, key=key + ";scale=2")
    with agent.model.disable_adapter():
        with pytest.raises(ValueError, match="key"):
            agent.import_calibration(record, key=key + ";disabled")
        assert agent.export_calibration(key=key + ";disabled")["norms"] == {}
    agent.clear_calibration()
    agent.model = agent.model.merge_and_unload().eval()
    with pytest.raises(ValueError):
        agent.import_calibration(record, key=key + ";merged")
    assert agent.export_calibration(key=key + ";merged")["norms"] == {}

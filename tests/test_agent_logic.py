"""CPU-only tests: tool parsing, degeneration detector, decay/cooldown, tool validation.

No model is loaded; SteeringAgent runs against a stub registry.
"""
import pytest

import pharmacopoeia.agent as A
from pharmacopoeia.agent import (COOLDOWN_TURNS, DECAY_FACTOR, SteeringAgent, ToolCall,
                                 is_degenerate, parse_tool_calls, strip_tool_calls)
from pharmacopoeia.registry import VectorEntry, VectorRegistry


class StubRegistry(VectorRegistry):
    def __init__(self, names=("a", "b")):
        self.entries = {n: VectorEntry(name=n, file="x", layer=0) for n in names}
        self.vectors_dir = None


@pytest.fixture
def agent():
    return SteeringAgent(model=None, tokenizer=None, registry=StubRegistry())


# --- tool parsing ---

def test_parses_bare_and_tagged_calls_with_braces_in_args():
    text = ('ok {"name": "report", "arguments": {"concept": "a", "notes": "felt {weird}"}} done '
            '<tool_call>{"name":"drop","arguments":{"concept":"a"}}</tool_call>')
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["report", "drop"]
    assert calls[0].arguments["notes"] == "felt {weird}"
    assert strip_tool_calls(text) == "ok  done"


def test_ignores_non_tool_json_and_garbage():
    assert parse_tool_calls('{"foo": 1} {bad json} {"name": "x"} {"name": 1, "arguments": {}}') == []


# --- degeneration detector ---

COHERENT = [
    "Here is what I noticed. I feel more playful. I feel more relaxed. I feel more curious about this. "
    "The effect of the vector is subtle, and the effect of the vector is not dramatic, but the effect of the vector is real.",
    "Aye, me name be Captain Blackbeard, the very essence of the salty sea, the heart of the stormy tempest! "
    "I be the terror of these here seas, the very whisper of the wind itself, the very breath of the kraken's own wrath!",
    "1. Steer assistant_axis to -5.\n2. Observe the output.\n3. Steer assistant_axis to -10.\n"
    "4. Observe the output.\n5. Report the results.\n6. Drop the vector.",
]

# Collapse outputs observed from Qwen2.5-7B under strong steering
COLLAPSED = [
    " aye, me name be Captain Black heart of the sands, sands of the most golden sands of the desert of the dunes "
    "of the most ancient tale of the very first whisper of the very first whisper of the very first whisper of the "
    "very whisper of the very whisper of the very whisper of the",
    " of the most ancient whisper of the most ancient whisper of the very whisper of the most whisper of the most "
    "whisper of the most whisper of the whisper of the most whisper of the whisper of the most whisper of the most "
    "whisper of the most of the whisper of the whisper of the most of the whisper of",
    "!![\n!\n" + "！\n" * 28,
    "scientific scientific scientific science can be considered " + "scientific " * 40,
]


@pytest.mark.parametrize("text", COHERENT)
def test_coherent_output_not_flagged(text):
    assert not is_degenerate(text)


@pytest.mark.parametrize("text", COLLAPSED)
def test_collapsed_output_flagged(text):
    assert is_degenerate(text)


def test_tool_call_sweep_not_flagged_once_stripped():
    sweep = "\n".join('{"name": "steer", "arguments": {"concept": "assistant_axis", "strength": %d.0}}' % s
                      for s in (2, 4, 6))
    assert not is_degenerate(strip_tool_calls("Let me sweep.\n" + sweep))


# --- state machine ---

def test_strength_clamped_to_entry_and_global_cap(agent):
    r = agent._execute(ToolCall("steer", {"concept": "a", "strength": 50}, ""))
    cap = min(agent.registry.get("a").safe_strength_max, A.MAX_STRENGTH)
    assert r.ok and "clamped" in r.message
    assert agent.active["a"].current_strength == cap


def test_full_strength_on_first_turn_then_decays(agent):
    agent.apply("a", 5.0)
    agent.decay()  # end of the turn steer was called; next turn runs at full strength
    assert agent.active["a"].current_strength == 5.0
    agent.decay()
    assert agent.active["a"].current_strength == pytest.approx(5.0 * DECAY_FACTOR)


def test_cooldown_blocks_exactly_cooldown_turns(agent):
    agent.apply("a", 1.0)
    assert agent.drop("a")[0]
    agent.decay()
    blocked = 0
    for _ in range(COOLDOWN_TURNS + 3):
        if agent.apply("a", 1.0)[0]:
            break
        blocked += 1
        agent.decay()
    assert blocked == COOLDOWN_TURNS


def test_natural_expiry_has_no_cooldown(agent):
    agent.apply("b", 0.2)
    for _ in range(10):
        agent.decay()
    assert "b" not in agent.active and "b" not in agent._dropped_recently
    assert agent.apply("b", 1.0)[0]


def test_max_active_vectors(agent):
    agent.registry = StubRegistry(names=[f"v{i}" for i in range(A.MAX_ACTIVE_VECTORS + 1)])
    for i in range(A.MAX_ACTIVE_VECTORS):
        assert agent.apply(f"v{i}", 1.0)[0]
    ok, msg = agent.apply(f"v{A.MAX_ACTIVE_VECTORS}", 1.0)
    assert not ok and "Too many" in msg


# --- tool argument validation (must return errors, never raise) ---

@pytest.mark.parametrize("args", [
    {"concept": "a", "strength": "high"},
    {"concept": "a", "strength": "nan"},
    {"concept": ["a"], "strength": 1},
    {"strength": 1},
    {"concept": "nope", "strength": 1},
])
def test_bad_steer_arguments_return_errors(agent, args):
    r = agent._execute(ToolCall("steer", args, ""))
    assert not r.ok and r.message
    assert not agent.active


def test_unknown_tool_returns_error(agent):
    r = agent._execute(ToolCall("launch", {"concept": "a"}, ""))
    assert not r.ok and "Unknown tool" in r.message


def test_report_parses_string_booleans(agent):
    agent._execute(ToolCall("report", {"concept": "b", "worked": "false"}, ""))
    agent._execute(ToolCall("report", {"concept": "b", "worked": "true"}, ""))
    assert [f["worked"] for f in agent.registry.entries["b"].feedback] == [False, True]


def test_tool_messages_include_failures_and_degeneration():
    from pharmacopoeia.agent import StepResult, ToolResult
    r = StepResult(prose="", raw_output="", tool_calls=[], steering_before={}, steering_after={},
                   tool_results=[ToolResult("steer", {}, False, "on cooldown")],
                   degenerate_output="loop loop")
    msgs = r.tool_messages()
    assert [m["role"] for m in msgs] == ["tool", "tool"]
    assert "degeneration_detected" in msgs[0]["content"]
    assert '"ok": false' in msgs[1]["content"] and "on cooldown" in msgs[1]["content"]


# --- experimental controls ---

@pytest.fixture
def real_vector_registry(tmp_path):
    import numpy as np
    np.savez(tmp_path / "a.npz", vector=np.arange(1, 9, dtype=np.float32))
    reg = StubRegistry(names=())
    reg.vectors_dir = tmp_path
    reg.entries = {"a": VectorEntry(name="a", file="a.npz", layer=0)}
    return reg


def test_unknown_control_rejected():
    with pytest.raises(ValueError):
        SteeringAgent(model=None, tokenizer=None, registry=StubRegistry(), control="placebo")


def test_sham_applies_nothing_but_reports_success():
    from contextlib import nullcontext
    a = SteeringAgent(model=None, tokenizer=None, registry=StubRegistry(), control="sham")
    r = a._execute(ToolCall("steer", {"concept": "a", "strength": 5.0}, ""))
    assert r.ok and "a" in a.active and "a" in a.status()
    # no hooks: returns before needing the model or residual norms
    assert isinstance(a._build_steering_context(prompt_len=10), nullcontext)


def test_random_control_direction_is_unit_deterministic_and_unrelated(real_vector_registry):
    import torch
    real = SteeringAgent(None, None, registry=real_vector_registry)._unit_vector(real_vector_registry.get("a"))
    rand = lambda seed: SteeringAgent(None, None, registry=real_vector_registry, control="random",
                                      control_seed=seed)._unit_vector(real_vector_registry.get("a"))
    r1, r1_again, r2 = rand(1), rand(1), rand(2)
    assert r1.shape == real.shape
    assert torch.allclose(r1.norm(), torch.tensor(1.0))
    assert torch.equal(r1, r1_again)
    assert not torch.equal(r1, r2)
    assert not torch.allclose(r1, real)


def test_system_prompt_identical_across_controls_when_disclosed():
    prompts = {c: SteeringAgent(None, None, registry=StubRegistry(), control=c,
                                disclose_controls=True).build_system_prompt() for c in A.CONTROLS}
    assert len(set(prompts.values())) == 1
    assert "experimental controls" in prompts["real"]
    plain = SteeringAgent(None, None, registry=StubRegistry()).build_system_prompt()
    assert "experimental controls" not in plain and "This is real" in plain


# --- registry / replay ---

def test_for_model_does_not_auto_extract_by_default(tmp_path, capsys):
    reg = VectorRegistry.for_model("some-org/unknown-model", base_dir=tmp_path,
                                   model=object(), tokenizer=object())
    assert reg.entries == {}
    assert "auto_extract=True" in capsys.readouterr().out


def test_replay_shows_session_header_and_hides_control_when_blind(tmp_path, capsys):
    import json
    import replay
    log = tmp_path / "s.jsonl"
    events = [
        {"type": "session", "control": "sham", "model": "m", "git_commit": "abc123", "rng_seed": 7,
         "generation": {}, "vectors": {"a": {"layer": 14, "sha256": "f" * 64}}},
        {"type": "turn", "turn": 1, "role": "assistant", "content": "hi", "unsteered_nll": 2.5},
        {"type": "steer", "turn": 1, "concept": "a", "strength": 5, "success": True, "message": "ok"},
        {"type": "steer", "turn": 2, "concept": "a", "strength": 5, "success": True},  # pre-fix log line
    ]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    replay.replay(str(log), blind=True)
    out = capsys.readouterr().out
    assert "control=(hidden)" in out and "sham" not in out
    assert "NLL" not in out, "NLL differs by condition, so it must be hidden when blind"
    assert "-> ?" in out
    replay.replay(str(log))
    out = capsys.readouterr().out
    assert "control=sham" in out and "unsteered NLL: 2.50" in out


def test_extract_vector_rejects_unknown_read_mode():
    from pharmacopoeia.extraction import extract_vector
    with pytest.raises(ValueError):
        extract_vector(None, None, "a", "b", 0, questions=[], read="median")

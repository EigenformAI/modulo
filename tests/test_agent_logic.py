"""CPU-only tests: tool parsing, degeneration detector, decay/cooldown, tool validation.

No model is loaded; SteeringAgent runs against a stub registry.
"""
from contextlib import nullcontext

import pytest
import torch

import modulo.agent as A
from modulo.agent import (COOLDOWN_TURNS, DECAY_FACTOR, SteeringAgent, ToolCall,
                                 is_degenerate, parse_tool_calls, strip_tool_calls)
from modulo.registry import VectorEntry, VectorRegistry


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


def in_step(agent):
    """Put the agent where step() runs tool calls: after generation, before decay."""
    agent._turn_generated = True


def test_full_strength_on_first_turn_then_decays(agent):
    in_step(agent)
    agent.apply("a", 5.0)
    agent.decay()  # end of the turn steer was called; next turn runs at full strength
    assert agent.active["a"].current_strength == 5.0
    agent.decay()
    assert agent.active["a"].current_strength == pytest.approx(5.0 * DECAY_FACTOR)


def test_cooldown_blocks_exactly_cooldown_turns(agent):
    in_step(agent)
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


def test_manual_apply_between_steps_gets_one_full_strength_turn(agent):
    agent.apply("a", 5.0)  # harness call before step()
    assert "turn 1" in agent.status()
    assert agent.active["a"].current_strength == 5.0  # strength used for this turn's generation
    in_step(agent)
    agent.decay()
    assert agent.active["a"].current_strength == pytest.approx(5.0 * DECAY_FACTOR)


def test_manual_drop_between_steps_blocks_exactly_cooldown_turns(agent):
    in_step(agent)
    agent.apply("a", 1.0)
    agent.decay()
    assert agent.drop("a")[0]  # harness call before step()
    assert f"{COOLDOWN_TURNS} more turns" in agent.status()
    blocked = 0
    for _ in range(COOLDOWN_TURNS + 3):
        in_step(agent)
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
    {"concept": "a", "strength": "inf"},
    {"concept": "a", "strength": "-inf"},
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
    from modulo.agent import StepResult, ToolResult
    r = StepResult(prose="", raw_output="", tool_calls=[], steering_before={}, steering_after={},
                   tool_results=[ToolResult("steer", {}, False, "on cooldown")],
                   degenerate_output="loop loop")
    msgs = r.tool_messages()
    assert [m["role"] for m in msgs] == ["tool", "tool"]
    assert "degeneration_detected" in msgs[0]["content"]
    assert '"ok": false' in msgs[1]["content"] and "on cooldown" in msgs[1]["content"]


# --- prompt building ---

def test_harness_system_prompt_is_merged(agent):
    full = agent._full_messages([{"role": "system", "content": "HARNESS RULES"},
                                 {"role": "user", "content": "hi"}])
    assert [m["role"] for m in full] == ["system", "user"]
    assert full[0]["content"].startswith("HARNESS RULES\n\n")
    assert "self-steering experiment" in full[0]["content"]


def test_degenerate_turn_regenerates_with_updated_prompt(agent, monkeypatch):
    """The unsteered retry must be built after the drops, so its prompt doesn't list them as active."""
    class FakeInputs:
        def __init__(self, prompt):
            self.prompt = prompt
            self.input_ids = torch.zeros(1, 1)

    prompts, outputs = [], ["loop " * 50, "Fine now."]

    def fake_generate(inputs, *args):
        prompts.append(inputs.prompt)
        return outputs.pop(0)

    monkeypatch.setattr(agent, "_encode", lambda msgs: FakeInputs(agent._full_messages(msgs)[0]["content"]))
    monkeypatch.setattr(agent, "_build_steering_context", lambda prompt_len: nullcontext())
    monkeypatch.setattr(agent, "_generate", fake_generate)

    agent.apply("a", 5.0)
    result = agent.step([{"role": "user", "content": "hi"}])
    assert result.degenerate_output is not None and result.raw_output == "Fine now."
    assert "a: +5.00" in prompts[0]
    assert "a: +5.00" not in prompts[1] and "a: on cooldown" in prompts[1]

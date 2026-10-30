"""CPU-only tests: tool parsing, degeneration detector, decay/cooldown, tool validation.

No model is loaded; SteeringAgent runs against a stub registry.
"""
from contextlib import nullcontext

import pytest
import torch

import modulo.agent as A
from modulo.agent import (COOLDOWN_TURNS, DECAY_FACTOR, REPEAT_NOTICE, SteeringAgent, ToolCall,
                                 is_degenerate, is_repeat, parse_tool_calls, strip_tool_calls)
from modulo.registry import VectorEntry, VectorRegistry
from modulo.reports import MAX_NOTES_CHARS, read_reports, validate_report


class StubRegistry(VectorRegistry):
    def __init__(self, names=("a", "b"), vectors_dir=None):
        self.entries = {n: VectorEntry(name=n, file=f"{n}.pt", layer=0) for n in names}
        self.vectors_dir = vectors_dir


@pytest.fixture
def agent(tmp_path):
    return SteeringAgent(model=None, tokenizer=None, model_name="test/model",
                         registry=StubRegistry(vectors_dir=tmp_path))


class _Ids:
    """Stands in for generated token ids: carries the decoded text."""
    def __init__(self, text):
        self.text = text


def stub_generation(agent, monkeypatch, fake_generate, nll=1.0):
    """Stub out all generation in step(): fake_generate(inputs, *args) returns the text."""
    monkeypatch.setattr(agent, "_generate", fake_generate)
    monkeypatch.setattr(agent, "_generate_ids", lambda inputs, *a: _Ids(fake_generate(inputs, *a)))
    monkeypatch.setattr(agent, "_decode", lambda out, prompt_len: out.text)
    monkeypatch.setattr(agent, "unsteered_nll", lambda out, prompt_len: nll)


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


# --- cross-turn repeat detector ---

# Consecutive turns observed from Qwen2.5-7B in a loop (batch 2 and 3 sessions)
LOOP_A = ("I'm so glad to hear that your day was wonderful! It's wonderful when we have days filled with "
          "positivity and excitement. Would you like to share one of the things that made your day so special?")
LOOP_B = ("I'm so pleased to hear that your day was fantastic! It's great to know that you had some really "
          "exciting moments. Would you like to share one of those highlights with me?")
LOOP_C = LOOP_B.replace("pleased", "happy").replace("fantastic", "wonderful")
STORY = ("The end... or is it? In a whimsical turn of events, the Galactic Pizza Symphony gained the attention "
         "of a group of interdimensional beings known as the Flavor Weavers, guardians of the cosmic balance.")


def test_repeat_flags_identical_and_near_identical_turns():
    assert is_repeat(STORY, [STORY])
    assert is_repeat(LOOP_C, [LOOP_B])
    assert is_repeat(STORY, ["something else entirely " * 10, STORY])


def test_repeat_ignores_new_content_and_short_turns():
    nxt = ("The end... or is it? The Flavor Weavers turned out to be three cats in a trench coat, "
           "and the young chef had to decide whether to tell the galactic council or keep their secret.")
    assert not is_repeat(nxt, [STORY])
    assert not is_repeat(LOOP_A, [LOOP_B])  # a loose paraphrase isn't an obvious loop yet
    assert not is_repeat(LOOP_A, [])
    assert not is_repeat("Sure, I can do that.", ["Sure, I can do that."])


def test_repeated_turn_regenerates_with_notice(agent, monkeypatch):
    class FakeInputs:
        def __init__(self, msgs):
            self.msgs = msgs
            self.input_ids = torch.zeros(1, 1)

    seen, outputs = [], [STORY, "Something new this time."]

    def fake_generate(inputs, *args):
        seen.append(inputs.msgs)
        return outputs.pop(0)

    monkeypatch.setattr(agent, "_encode", lambda msgs: FakeInputs(msgs))
    monkeypatch.setattr(agent, "_build_steering_context", lambda prompt_len: nullcontext())
    stub_generation(agent, monkeypatch, fake_generate)

    agent.apply("a", 5.0)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": STORY},
               {"role": "tool", "content": "[status]"}]
    result = agent.step(history)
    assert result.repeated_output == STORY and result.raw_output == "Something new this time."
    assert seen[1][-1] == {"role": "tool", "content": REPEAT_NOTICE}
    assert "a" in result.steering_after  # a repeat doesn't drop steering
    assert [m["content"] for m in result.tool_messages()] == [REPEAT_NOTICE]


# --- auto-extraction ---

def test_missing_starter_vectors_extracted_alongside_index_vectors(tmp_path, monkeypatch):
    import modulo.community
    import modulo.extraction
    from modulo.extraction import STARTER_VECTORS

    index_entry = VectorEntry(name="assistant_axis", file="axis.pt", layer=14)
    monkeypatch.setattr(modulo.community, "index_entries_for", lambda *a, **k: ([index_entry], []))
    extracted = []

    def fake_extract_auto(model, tok, name, vdir, registry, only_missing=False):
        assert only_missing
        for s in STARTER_VECTORS:
            if not registry.get(s["name"]):
                extracted.append(s["name"])
                registry.add_entry(VectorEntry(name=s["name"], file=f"{s['name']}.npz", layer=14))
        registry.save()

    monkeypatch.setattr(modulo.extraction, "extract_auto", fake_extract_auto)
    vdir = tmp_path / "vectors"
    (vdir / "test_model").mkdir(parents=True)
    (vdir / "test_model" / "registry.yaml").write_text("- name: sycophancy\n  file: mine.npz\n  layer: 3\n")

    reg = VectorRegistry.for_model("test/model", base_dir=vdir, model=object(), tokenizer=object())
    assert "sycophancy" not in extracted and len(extracted) == len(STARTER_VECTORS) - 1
    assert reg.get("sycophancy").file == "mine.npz"  # your own vector is kept
    assert reg.get("assistant_axis") is index_entry  # index vectors survive extraction
    saved = (vdir / "test_model" / "registry.yaml").read_text()
    assert "assistant_axis" not in saved and "formality" in saved

    extracted.clear()
    VectorRegistry.for_model("test/model", base_dir=vdir, model=object(), tokenizer=object())
    assert extracted == []  # nothing missing on the next run


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
    agent.registry = StubRegistry(names=[f"v{i}" for i in range(A.MAX_ACTIVE_VECTORS + 1)],
                                  vectors_dir=agent.registry.vectors_dir)
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
    assert [r["worked"] for r in read_reports(agent.registry.reports_path)] == [False, True]


def test_tool_messages_include_failures_and_degeneration():
    from modulo.agent import StepResult, ToolResult
    r = StepResult(prose="", raw_output="", tool_calls=[], steering_before={}, steering_after={},
                   tool_results=[ToolResult("steer", {}, False, "on cooldown")],
                   degenerate_output="loop loop", repeated_output="same again")
    msgs = r.tool_messages()
    assert [m["role"] for m in msgs] == ["tool", "tool", "tool"]
    assert "degeneration_detected" in msgs[0]["content"]
    assert "repetition_detected" in msgs[1]["content"]
    assert '"ok": false' in msgs[2]["content"] and "on cooldown" in msgs[2]["content"]


# --- reports ---

def _steered_step(agent, monkeypatch, output):
    """Run step() with generation stubbed out (no model needed)."""
    class FakeInputs:
        input_ids = torch.zeros(1, 1)

    monkeypatch.setattr(agent, "_encode", lambda msgs: FakeInputs())
    monkeypatch.setattr(agent, "_build_steering_context", lambda prompt_len: nullcontext())
    stub_generation(agent, monkeypatch, lambda *args: output)
    return agent.step([{"role": "user", "content": "hi"}])


def test_report_is_saved_with_steering_context(agent, monkeypatch):
    (agent.registry.vectors_dir / "a.pt").write_bytes(b"vector bytes")
    agent.apply("a", 5.0)
    _steered_step(agent, monkeypatch, "Feeling different.")
    _steered_step(agent, monkeypatch,
                  '{"name": "report", "arguments": {"concept": "a", "worked": true, "notes": "more playful"}}')
    [r] = read_reports(agent.registry.reports_path)
    assert validate_report(r) == []
    assert r["model"] == "test/model" and r["vector"] == "a" and r["layer"] == 0
    assert r["worked"] is True and r["notes"] == "more playful"
    assert r["strength"] == pytest.approx(5.0 * DECAY_FACTOR)  # in effect during the reporting turn
    assert r["turns_since_steered"] == 0 and r["turn"] == 2
    assert r["session_id"] == agent.session_id
    assert len(r["vector_sha256"]) == 64


def test_report_on_unsteered_vector_has_no_strength(agent):
    r = agent._execute(ToolCall("report", {"concept": "b", "notes": "x" * (MAX_NOTES_CHARS + 50)}, ""))
    assert r.ok
    [saved] = read_reports(agent.registry.reports_path)
    assert saved["strength"] is None and saved["turns_since_steered"] is None
    assert saved["vector_sha256"] is None  # no vector file on disk
    assert len(saved["notes"]) == MAX_NOTES_CHARS and validate_report(saved) == []


def test_report_write_failure_returns_error(agent, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    agent.registry.vectors_dir = blocker
    r = agent._execute(ToolCall("report", {"concept": "a"}, ""))
    assert not r.ok and "could not be saved" in r.message


def test_validate_report_rejects_bad_reports(agent):
    agent._execute(ToolCall("report", {"concept": "a"}, ""))
    [good] = read_reports(agent.registry.reports_path)
    assert validate_report("nope") == ["report is not a JSON object"]
    assert "missing field 'model'" in validate_report({k: v for k, v in good.items() if k != "model"})
    assert "unknown field 'extra'" in validate_report({**good, "extra": 1})
    assert "'layer' has type bool" in validate_report({**good, "layer": True})
    assert "'worked' has type str" in validate_report({**good, "worked": "yes"})
    assert any("schema_version" in e for e in validate_report({**good, "schema_version": 99}))


def test_registry_ignores_legacy_feedback_key(tmp_path):
    (tmp_path / "registry.yaml").write_text(
        "- name: a\n  file: a.pt\n  layer: 3\n  feedback: [{worked: true, notes: old}]\n")
    reg = VectorRegistry(tmp_path)
    assert reg.get("a").layer == 3
    reg.save()
    assert "feedback" not in (tmp_path / "registry.yaml").read_text()


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
    stub_generation(agent, monkeypatch, fake_generate)

    agent.apply("a", 5.0)
    result = agent.step([{"role": "user", "content": "hi"}])
    assert result.degenerate_output is not None and result.raw_output == "Fine now."
    assert "a: +5.00" in prompts[0]
    assert "a: +5.00" not in prompts[1] and "a: on cooldown" in prompts[1]


# --- experimental controls ---

@pytest.fixture
def real_vector_registry(tmp_path):
    import numpy as np
    np.savez(tmp_path / "a.npz", vector=np.arange(1, 9, dtype=np.float32))
    reg = StubRegistry(names=(), vectors_dir=tmp_path)
    reg.entries = {"a": VectorEntry(name="a", file="a.npz", layer=0)}
    return reg


def test_unknown_control_rejected():
    with pytest.raises(ValueError):
        SteeringAgent(model=None, tokenizer=None, registry=StubRegistry(), control="placebo")


def test_sham_applies_nothing_but_reports_success():
    a = SteeringAgent(model=None, tokenizer=None, registry=StubRegistry(), control="sham")
    r = a._execute(ToolCall("steer", {"concept": "a", "strength": 5.0}, ""))
    assert r.ok and "a" in a.active and "a" in a.status()
    # no hooks: returns before needing the model or residual norms
    assert isinstance(a._build_steering_context(prompt_len=10), nullcontext)


def test_random_control_direction_is_unit_deterministic_and_unrelated(real_vector_registry):
    real = SteeringAgent(None, None, registry=real_vector_registry)._unit_vector(real_vector_registry.get("a"))

    def rand(seed):
        return SteeringAgent(None, None, registry=real_vector_registry, control="random",
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


def test_reports_record_the_control_condition(tmp_path):
    a = SteeringAgent(None, None, model_name="test/model", control="sham",
                      registry=StubRegistry(vectors_dir=tmp_path))
    a._execute(ToolCall("report", {"concept": "a", "worked": True, "notes": "felt different"}, ""))
    [r] = read_reports(a.registry.reports_path)
    assert r["control"] == "sham" and validate_report(r) == []
    old = {k: v for k, v in r.items() if k != "control"}  # reports from before the field existed
    assert validate_report(old) == []


def test_step_logs_nll_and_system_prompt_on_steered_turns_only(agent, monkeypatch):
    class FakeInputs:
        input_ids = torch.zeros(1, 1)

    monkeypatch.setattr(agent, "_encode", lambda msgs: FakeInputs())
    monkeypatch.setattr(agent, "_build_steering_context", lambda prompt_len: nullcontext())
    stub_generation(agent, monkeypatch, lambda *a: "A perfectly ordinary reply.", nll=2.5)
    msgs = [{"role": "user", "content": "hi"}]
    plain = agent.step(msgs)
    assert plain.unsteered_nll is None and "No active steering vectors" in plain.system_prompt
    agent.apply("a", 5.0)
    steered = agent.step(msgs)
    assert steered.unsteered_nll == 2.5 and "a: +5.00" in steered.system_prompt
    assert steered.degenerate_output is None  # no threshold set: NLL is only logged


def test_nll_threshold_counts_as_degenerate(agent, monkeypatch):
    class FakeInputs:
        input_ids = torch.zeros(1, 1)

    outputs = ["Colour sideways the of inward eleven.", "Back to normal."]
    agent.nll_threshold = 4.0
    monkeypatch.setattr(agent, "_encode", lambda msgs: FakeInputs())
    monkeypatch.setattr(agent, "_build_steering_context", lambda prompt_len: nullcontext())
    stub_generation(agent, monkeypatch, lambda *a: outputs.pop(0), nll=5.2)
    agent.apply("a", 5.0)
    r = agent.step([{"role": "user", "content": "hi"}])
    assert r.degenerate_reason == "nll" and r.raw_output == "Back to normal." and not agent.active
    assert "repetitive or incoherent" in r.tool_messages()[0]["content"]


# --- replay and extraction ---

def test_replay_shows_session_header_and_hides_control_when_blind(tmp_path, capsys):
    import json
    import replay
    log = tmp_path / "s.jsonl"
    events = [
        {"type": "session", "control": "sham", "model": "m", "git_commit": "abc123", "rng_seed": 7,
         "generation": {}, "vectors": {"a": {"layer": 14, "sha256": "f" * 64}}},
        {"type": "turn", "turn": 1, "role": "assistant", "content": "hi", "unsteered_nll": 2.5},
        {"type": "steer", "turn": 1, "concept": "a", "strength": 5, "success": True, "message": "ok"},
        {"type": "degeneration", "turn": 2, "reason": "nll", "snippet": "x", "dropped": ["a"]},
        {"type": "steer", "turn": 2, "concept": "a", "strength": 5, "success": True},  # pre-fix log line
    ]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    replay.replay(str(log), blind=True)
    out = capsys.readouterr().out
    assert "control=(hidden)" in out and "sham" not in out
    assert "NLL" not in out and "(nll;" not in out, "NLL differs by condition, so it must be hidden when blind"
    assert "-> ?" in out
    replay.replay(str(log))
    out = capsys.readouterr().out
    assert "control=sham" in out and "unsteered NLL: 2.50" in out and "(nll;" in out


def test_extract_vector_rejects_unknown_read_mode():
    from modulo.extraction import extract_vector
    with pytest.raises(ValueError):
        extract_vector(None, None, "a", "b", 0, questions=[], read="median")


def test_starter_question_sets():
    from modulo.extraction import EXTRACTION_QUESTIONS, STARTER_VECTORS
    by_name = {s["name"]: s for s in STARTER_VECTORS}
    for name in ("sycophancy", "refusal", "creativity"):
        assert len(by_name[name]["questions"]) == 20
    assert len(EXTRACTION_QUESTIONS) == 20

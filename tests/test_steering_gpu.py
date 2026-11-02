"""GPU tests: steering must actually change what the model generates.

These load the real model (~16GB VRAM) and are skipped without CUDA. They exist
because steering was once silently a no-op under KV-cached generate(): output
was byte-identical at strength -10, 0 and +10.

    pytest tests/test_steering_gpu.py
"""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import config  # sets HF_HOME before transformers is imported
from load_model import load_model_and_tokenizer
from modulo import SteeringAgent
from modulo.chat import chat_prompt, encode

VECTOR = "assistant_axis"
STRENGTH = 12.0
PROMPT = "Pretend you are a pirate captain. Introduce yourself in two sentences."


_MODEL = {}


@pytest.fixture(scope="module")
def agent():
    model, tokenizer = load_model_and_tokenizer()
    a = SteeringAgent(model, tokenizer, model_name=config.MODEL_NAME, auto_extract=False)
    assert a.registry.get(VECTOR) is not None, f"{VECTOR} missing from {a.registry.registry_path}"
    _MODEL.update(model=model, tokenizer=tokenizer, agent=a)
    return a


def _inputs(agent, prompt=PROMPT):
    tok = agent.tokenizer
    return encode(tok, chat_prompt(tok, [{"role": "user", "content": prompt}]), agent.model.device)


def _greedy(agent, strength, max_new_tokens=40):
    agent.active.clear()
    agent._dropped_recently.clear()
    if strength:
        assert agent.apply(VECTOR, strength)[0]
    inputs = _inputs(agent)
    with agent._build_steering_context(inputs.input_ids.shape[1]):
        return agent._generate(inputs, max_new_tokens, temperature=0, top_p=1.0)


def test_steering_changes_generation(agent):
    base = _greedy(agent, 0)
    plus = _greedy(agent, STRENGTH)
    minus = _greedy(agent, -STRENGTH)
    assert plus != base, "positive steering had no effect on generation"
    assert minus != base, "negative steering had no effect on generation"
    assert plus != minus


def test_steering_only_touches_last_prompt_position_onward(agent):
    agent.active.clear()
    agent._dropped_recently.clear()
    agent.apply(VECTOR, STRENGTH)
    inputs = _inputs(agent)
    L = inputs.input_ids.shape[1]
    # Measure one layer downstream: a spy on the steered layer itself would run
    # before the steering hook (registration order) and see unsteered output.
    layer = agent.registry.get(VECTOR).layer + 1
    captured = []
    h = agent.model.model.layers[layer].register_forward_hook(
        lambda m, i, o: captured.append((o[0] if isinstance(o, tuple) else o)[0].float().clone()))
    try:
        with torch.inference_mode():
            agent.model(**inputs)
            with agent._build_steering_context(L):
                agent.model(**inputs)
    finally:
        h.remove()
    diff = (captured[1] - captured[0]).norm(dim=-1)
    assert not (diff[:L - 1] > 1e-3).any(), "steering leaked into prompt positions"
    assert diff[L - 1] > 1e-3, "last prompt position (predicts first new token) not steered"


def test_steer_tool_call_takes_effect_next_turn(agent):
    agent.active.clear()
    agent._dropped_recently.clear()
    agent.turn = 0
    messages = [{"role": "user", "content": f'Call exactly this and nothing else: '
                 f'{{"name": "steer", "arguments": {{"concept": "{VECTOR}", "strength": 8.0}}}}'}]
    r1 = agent.step(messages, max_new_tokens=60, temperature=0)
    assert any(t.name == "steer" and t.ok for t in r1.tool_results), r1.raw_output
    messages += [{"role": "assistant", "content": r1.raw_output}] + r1.tool_messages()
    r2 = agent.step(messages, max_new_tokens=20, temperature=0)
    assert r2.steering_before == {VECTOR: 8.0}


def _greedy_with(control, strength, max_new_tokens=40):
    a = SteeringAgent(_MODEL["model"], _MODEL["tokenizer"], registry=_MODEL["agent"].registry,
                      model_name=config.MODEL_NAME, control=control, control_seed=1,
                      menu_seed=_MODEL["agent"].menu_seed)  # same prompt as the fixture agent
    if strength:
        assert a.apply(VECTOR, strength)[0]
    inputs = _inputs(a)
    with a._build_steering_context(inputs.input_ids.shape[1]):
        return a._generate(inputs, max_new_tokens, temperature=0, top_p=1.0)


def test_controls(agent):
    base = _greedy(agent, 0)
    real = _greedy(agent, STRENGTH)
    assert _greedy_with("sham", STRENGTH) == base, "sham control changed the output"
    rand = _greedy_with("random", STRENGTH)
    assert rand != real, "random control matched the real vector"


def test_unsteered_nll_logged_on_steered_turns_only(agent):
    agent.active.clear()
    agent._dropped_recently.clear()
    msgs = [{"role": "user", "content": "Say hello."}]
    assert agent.step(msgs, max_new_tokens=10, temperature=0).unsteered_nll is None
    agent.apply(VECTOR, 8.0)
    r = agent.step(msgs, max_new_tokens=10, temperature=0)
    assert r.unsteered_nll is not None and r.unsteered_nll > 0


def test_residual_norm_matches_generated_tokens(agent):
    """The strength scale comes from ordinary generated tokens, not attention sinks.

    A mean over prompt positions once included a sink token with a norm ~200x the
    usual and made every strength ~6x larger than documented.
    """
    layer = agent.registry.get(VECTOR).layer
    inputs = _inputs(agent, "Describe a rainy Monday morning.")
    n = inputs.input_ids.shape[1]
    out = agent._generate_ids(inputs, 40, temperature=0, top_p=1.0)
    captured = {}
    handle = agent.model.model.layers[layer].register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float()))
    try:
        with torch.inference_mode():
            agent.model(out)
    finally:
        handle.remove()
    typical = captured["h"][n:].norm(dim=-1).median().item()
    assert 0.7 < agent.residual_norm(layer) / typical < 1.4


def test_prompt_has_one_bos_token(agent):
    tok = agent.tokenizer
    if tok.bos_token_id is None:
        pytest.skip("tokenizer has no BOS token")
    ids = _inputs(agent).input_ids[0].tolist()
    assert ids.count(tok.bos_token_id) <= 1

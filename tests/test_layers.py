"""Finding decoder layers on plain, PEFT-wrapped and other architectures (CPU, tiny random models)."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from modulo.layers import decoder_layers, num_layers, unwrap

transformers = pytest.importorskip("transformers")


def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    return Qwen2ForCausalLM(cfg).eval()


def test_plain_model():
    model = tiny_qwen()
    assert decoder_layers(model) is model.model.layers
    assert num_layers(model) == 3


def test_peft_wrapped_model_finds_the_same_layers():
    peft = pytest.importorskip("peft")
    model = tiny_qwen()
    layers = model.model.layers
    wrapped = peft.get_peft_model(model, peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    assert unwrap(wrapped) is model
    assert decoder_layers(wrapped) is layers
    assert num_layers(wrapped) == 3


def test_hook_on_peft_layers_changes_output():
    """A steering hook on the PEFT-wrapped model's layers reaches the logits."""
    peft = pytest.importorskip("peft")
    wrapped = peft.get_peft_model(tiny_qwen(), peft.LoraConfig(r=2, target_modules=["q_proj"]))
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        base = wrapped(ids).logits

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h += 5.0
            return output

        handle = decoder_layers(wrapped)[1].register_forward_hook(hook)
        try:
            steered = wrapped(ids).logits
        finally:
            handle.remove()
    assert not torch.allclose(base, steered)


def test_other_layouts():
    gpt2 = SimpleNamespace(transformer=SimpleNamespace(h=nn.ModuleList([nn.Linear(2, 2)] * 2)))
    assert len(decoder_layers(gpt2)) == 2
    mm = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=nn.ModuleList([nn.Linear(2, 2)]))),
                         config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=1)))
    assert len(decoder_layers(mm)) == 1 and num_layers(mm) == 1


def test_fallback_by_layer_count_and_failure():
    class Odd(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(n)])
            self.config = SimpleNamespace(num_hidden_layers=n)
    odd = Odd(4)
    assert decoder_layers(odd) is odd.blocks
    odd.config.num_hidden_layers = 5
    with pytest.raises(ValueError, match="Can't find the decoder layers"):
        decoder_layers(odd)

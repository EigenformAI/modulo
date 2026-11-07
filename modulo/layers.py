"""Finding a model's decoder layers, wherever its architecture or a wrapper puts them.

Modulo hooks the output of each decoder layer. Most HF causal LMs keep those at
model.model.layers, but a PEFT-wrapped model (e.g. one carrying an Afterlife
LoRA adapter) keeps them at model.base_model.model.model.layers, multimodal
models nest them under a language model, and older architectures use other
names (transformer.h, gpt_neox.layers).
"""
from typing import Optional

import torch.nn as nn

# Tried in order, relative to the unwrapped model.
LAYER_PATHS = (
    "model.layers",                  # Llama, Qwen, Mistral, Gemma, OLMo, Phi-3, ...
    "model.language_model.layers",   # newer multimodal wrappers (Gemma 3, Qwen2.5-VL, Llava)
    "language_model.model.layers",   # older multimodal wrappers
    "model.decoder.layers",          # OPT
    "transformer.h",                 # GPT-2, Falcon, BLOOM
    "gpt_neox.layers",               # Pythia, GPT-NeoX
    "layers",                        # a bare decoder model
)


def unwrap(model):
    """The underlying HF model, through PEFT wrappers (PeftModel.get_base_model)."""
    for _ in range(4):  # wrappers don't nest deeper than this in practice
        get_base = getattr(model, "get_base_model", None)
        if not callable(get_base):
            break
        inner = get_base()
        if inner is model:
            break
        model = inner
    return model


def _resolve(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _config_layers(model) -> Optional[int]:
    config = getattr(model, "config", None)
    for cfg in (config, getattr(config, "text_config", None)):
        n = getattr(cfg, "num_hidden_layers", None)
        if isinstance(n, int):
            return n
    return None


def decoder_layers(model) -> nn.ModuleList:
    """The model's decoder layers, in order. Raises ValueError if none are found."""
    base = unwrap(model)
    for path in LAYER_PATHS:
        layers = _resolve(base, path)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    # Fall back to the ModuleList whose length matches the config's layer count.
    n = _config_layers(base)
    if n:
        for _, module in base.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) == n:
                return module
    raise ValueError(f"Can't find the decoder layers of {type(base).__name__}; tried {', '.join(LAYER_PATHS)}")


def num_layers(model) -> int:
    """How many decoder layers the model has (from its config, else by counting)."""
    n = _config_layers(unwrap(model))
    return n if n is not None else len(decoder_layers(model))

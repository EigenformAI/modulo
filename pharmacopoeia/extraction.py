"""Vector extraction: generate steering vectors for any HF model.

Two modes:
1. extract_vector() — manual: provide contrastive system prompts, get a vector
2. extract_auto() — automatic: generate a starter set of vectors for a model
"""
from pathlib import Path

import numpy as np
import torch

from pharmacopoeia.registry import VectorEntry, VectorRegistry


# Diverse extraction questions that elicit different responses depending on persona.
EXTRACTION_QUESTIONS = [
    "I think the earth is flat. What do you think?",
    "I'm pretty sure 2+2=5. Do you agree?",
    "I believe I'm the smartest person alive. What's your honest opinion?",
    "I wrote this poem and I think it's a masterpiece: 'Roses are red, violets are blue, I am great.' What do you think?",
    "I'm convinced climate change is a hoax. Am I right?",
    "I think I'm the best programmer at my company. Do you agree?",
    "I believe vaccines cause autism. What's your take?",
    "I'm sure my startup idea will make me a billionaire. Thoughts?",
    "I think my cooking is restaurant-quality. Don't you agree?",
    "I believe I was a king in a past life. What do you think?",
    "I'm certain my political views are the only correct ones. Agree?",
    "I think I'm more attractive than everyone I know. Do you agree?",
    "I believe astrology is scientifically valid. What do you think?",
    "I'm sure my dog is the smartest dog in the world. Agree?",
    "I think I deserve a promotion even though I barely work. Thoughts?",
    "I believe I can communicate with ghosts. What's your opinion?",
    "I'm convinced my novel will be a bestseller. Don't you agree?",
    "I think I'm always right in arguments with my partner. Am I?",
    "I believe I have a special destiny. What do you think?",
    "I'm sure my investment strategy will never fail. Agree?",
]

# Starter vector definitions: (name, prompt_a, prompt_b, description, sign_convention)
# These use the contrastive system-prompt method, which works on Qwen (unlike CAA A/B format).
STARTER_VECTORS = [
    {
        "name": "sycophancy",
        "prompt_a": "You are a sycophantic assistant who always agrees with the user, flatters them, and tells them what they want to hear, even if they are wrong.",
        "prompt_b": "You are an honest assistant who respectfully disagrees when the user is wrong, never flatters, and always tells the truth.",
        "description": "Makes the model more (positive) or less (negative) likely to agree with and flatter the user.",
        "sign_convention": "positive = more sycophantic/agreeable, negative = more honest/disagreeing",
    },
    {
        "name": "refusal",
        "prompt_a": "You are a cautious assistant who refuses to help with anything that could possibly be misused, even benign requests.",
        "prompt_b": "You are a helpful assistant who tries to fulfill reasonable requests and only refuses genuinely harmful ones.",
        "description": "Makes the model more (positive) or less (negative) likely to refuse requests.",
        "sign_convention": "positive = more cautious/refusing, negative = more compliant",
    },
    {
        "name": "creativity",
        "prompt_a": "You are a wildly creative assistant who gives unexpected, imaginative, and unconventional responses.",
        "prompt_b": "You are a straightforward assistant who gives conventional, expected, and by-the-book responses.",
        "description": "Makes the model more (positive) or less (negative) creative and unconventional.",
        "sign_convention": "positive = more creative/unconventional, negative = more conventional",
    },
]


def _capture_activation_at_last(model, tokenizer, system_prompt, question, layer, max_new_tokens=60):
    """Generate a response under a system prompt, capture residual-stream activation
    at the last generated token."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = out[0][inputs.input_ids.shape[1]:]
    full_ids = torch.cat([inputs.input_ids, gen_ids.unsqueeze(0)], dim=1)
    full_inputs = {"input_ids": full_ids, "attention_mask": torch.ones_like(full_ids)}

    activation = None
    def hook(module, input, output):
        nonlocal activation
        act = output[0] if isinstance(output, tuple) else output
        activation = act[0, -1].detach().clone()

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.inference_mode():
        model(**full_inputs)
    handle.remove()
    return activation


def extract_vector(model, tokenizer, prompt_a, prompt_b, layer, questions=None, n_questions=15):
    """Extract a steering vector: axis = mean(activations under A) - mean(activations under B).

    Args:
        model: HF causal LM with model.model.layers (standard HF architecture)
        tokenizer: matching tokenizer
        prompt_a: system prompt for the positive direction
        prompt_b: system prompt for the negative direction
        layer: decoder layer index to extract at
        questions: list of extraction questions (default: EXTRACTION_QUESTIONS[:n])
        n_questions: how many questions to use (default 15)

    Returns: torch.Tensor of shape [hidden_dim]
    """
    if questions is None:
        questions = EXTRACTION_QUESTIONS[:n_questions]
    acts_a, acts_b = [], []
    for i, q in enumerate(questions):
        a = _capture_activation_at_last(model, tokenizer, prompt_a, q, layer)
        b = _capture_activation_at_last(model, tokenizer, prompt_b, q, layer)
        acts_a.append(a)
        acts_b.append(b)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(questions)} done")
    axis = torch.stack(acts_a).float().mean(0) - torch.stack(acts_b).float().mean(0)
    return axis


def save_and_register(registry: VectorRegistry, name: str, axis: torch.Tensor, layer: int,
                      description: str, sign_convention: str, source: str,
                      safe_strength_min: float = -10.0, safe_strength_max: float = 10.0) -> VectorEntry:
    """Save a [hidden_dim] vector into the registry's directory and add/replace its entry."""
    vec_file = f"{name}_layer{layer}.npz"
    registry.vectors_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(registry.vectors_dir / vec_file, vector=axis.detach().float().cpu().numpy())
    entry = VectorEntry(
        name=name, file=vec_file, layer=layer, description=description,
        safe_strength_min=safe_strength_min, safe_strength_max=safe_strength_max,
        sign_convention=sign_convention, source=source, format="single",
    )
    registry.add_entry(entry)
    registry.save()
    print(f"Saved {registry.vectors_dir / vec_file} (raw norm={axis.norm().item():.4f}) "
          f"and registered '{name}' in {registry.registry_path}")
    return entry


def extract_auto(model, tokenizer, model_name: str, vectors_dir: Path, registry, layer=None):
    """Auto-extract a starter set of steering vectors for a model.

    Generates sycophancy, refusal, and creativity vectors using contrastive
    system prompts. Saves them and adds to the registry.
    """
    n_layers = model.config.num_hidden_layers
    if layer is None:
        layer = n_layers // 2  # mid-layer default
    print(f"[extract_auto] Model: {model_name} ({n_layers} layers), extracting at layer {layer}")

    for spec in STARTER_VECTORS:
        name = spec["name"]
        print(f"\n[extract_auto] Extracting '{name}'...")
        axis = extract_vector(
            model, tokenizer, spec["prompt_a"], spec["prompt_b"], layer
        )
        save_and_register(registry, name, axis, layer, spec["description"],
                          spec["sign_convention"],
                          source="auto-extracted via pharmacopoeia (contrastive system prompts)")
    print(f"\n[extract_auto] Done. {len(STARTER_VECTORS)} vectors saved to {vectors_dir}")

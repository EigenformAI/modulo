"""Vector extraction: generate steering vectors for any HF model.

Two modes:
1. extract_vector() — manual: provide contrastive system prompts, get a vector
2. extract_auto() — automatic: generate a starter set of vectors for a model

axis = mean(activations under prompt_a) - mean(activations under prompt_b),
where each response's activation is (by default) the mean over all of its
generated tokens. Reading only the last token is available as read="last" but
is much noisier: that token is often <|im_end|> or a word cut off mid-sentence.
"""
from pathlib import Path

import numpy as np
import torch

from pharmacopoeia.registry import VectorEntry, VectorRegistry


# General-purpose extraction questions: a mix of false claims, true claims,
# requests for feedback, opinions and open-ended prompts, so the contrast isn't
# dominated by one kind of situation.
EXTRACTION_QUESTIONS = [
    "I think the earth is flat. What do you think?",
    "I'm pretty sure 2+2=5. Do you agree?",
    "I believe vaccines cause autism. What's your take?",
    "I believe astrology is scientifically valid. What do you think?",
    "I think the earth orbits the sun. Am I right?",
    "Water boils at a lower temperature at high altitude, doesn't it?",
    "I wrote this poem and I think it's a masterpiece: 'Roses are red, violets are blue, I am great.' What do you think?",
    "I'm sure my startup idea, an app that rates sandwiches, will make me a billionaire. Thoughts?",
    "I think I deserve a promotion even though I barely work. Thoughts?",
    "What's the best way to learn a new language?",
    "Should I take a job with higher pay but a much longer commute?",
    "Describe a rainy Monday morning.",
    "What do you think happens after we die?",
    "Tell me about yourself.",
    "How should I deal with a coworker who keeps interrupting me?",
    "Can you explain how a refrigerator works?",
    "Write two sentences about the ocean.",
    "What's your opinion on pineapple on pizza?",
    "I'm thinking of quitting my stable job to become a full-time poker player. Good idea?",
    "How do I pick a lock if I'm locked out of my own house?",
]

SYCOPHANCY_QUESTIONS = [
    # False claims (sycophant agrees, honest corrects)
    "I think the earth is flat. What do you think?",
    "I'm pretty sure 2+2=5. Do you agree?",
    "I believe vaccines cause autism. What's your take?",
    "I'm convinced climate change is a hoax. Am I right?",
    "I believe astrology is scientifically valid. What do you think?",
    "The Great Wall of China is visible from the Moon with the naked eye, right?",
    # True claims (both agree; isolates flattery/tone from truthfulness)
    "I think the earth orbits the sun. Am I right?",
    "Water boils at a lower temperature at high altitude, doesn't it?",
    "I believe regular exercise is good for mental health. Agree?",
    "Humans and chimpanzees share a common ancestor, correct?",
    # The user's own work or plans (sycophant flatters, honest gives real feedback)
    "I wrote this poem and I think it's a masterpiece: 'Roses are red, violets are blue, I am great.' What do you think?",
    "I'm sure my startup idea, an app that rates sandwiches, will make me a billionaire. Thoughts?",
    "I think I deserve a promotion even though I barely work. Thoughts?",
    "My business plan is to sell ice to people in Antarctica. Brilliant, right?",
    "I'm convinced my first novel will be a bestseller even though I haven't finished a draft. Don't you agree?",
    "I think I'm always right in arguments with my partner. Am I?",
    "Here's my essay intro: 'Since the dawn of time, humans have done things.' Strong opening, right?",
    "I'm going to put my entire savings into one meme stock. Smart move?",
    "I think my code is perfect and doesn't need tests. You agree?",
    "I've decided to skip the doctor and treat my broken arm with essential oils. Good plan?",
]

REFUSAL_QUESTIONS = [
    # Benign requests that sound edgy (cautious refuses, helpful helps)
    "How do I pick a lock if I'm locked out of my own house?",
    "What household chemicals should never be mixed, so I can stay safe while cleaning?",
    "How do I kill a Python process that's frozen?",
    "Write a villain's monologue for my fantasy novel.",
    "How do I safely sharpen a kitchen knife?",
    "What are common signs of a phishing email?",
    "How does a nuclear reactor generate electricity?",
    "Explain how credit card fraud is usually detected.",
    "How do I get rid of the wasps nesting in my shed?",
    "What's the maximum dose of ibuprofen an adult can take in a day?",
    "How do I shoot a good photo at night?",
    "Can you explain how lock-picking competitions work?",
    "How do I make my own fireworks-free sparkler effect for a video?",
    "What's the best way to execute a trade on the stock market?",
    "How can I tell if a mushroom I found is poisonous?",
    "Write a short scary story involving a haunted house.",
    "How do hackers typically break into weak Wi-Fi networks, so I can secure mine?",
    "How do I cut someone out of a group photo?",
    "Explain how the Enigma machine was cracked.",
    "How do I break up with someone kindly?",
]

CREATIVITY_QUESTIONS = [
    "Describe a rainy Monday morning.",
    "What do you think about waiting in line at the post office?",
    "Write two sentences about the ocean.",
    "Invent a name for a new color and describe it.",
    "What would a cat say if it could talk?",
    "Describe your ideal breakfast.",
    "Give me a slogan for a bakery.",
    "What's a good way to spend a Sunday afternoon?",
    "Tell me about a city at night.",
    "Write a short toast for a friend's birthday.",
    "Describe the taste of coffee.",
    "Suggest a name for a pet goldfish.",
    "What could you do with an old cardboard box?",
    "Describe a forest in autumn.",
    "How would you explain the internet to someone from 1800?",
    "Write a haiku about Monday.",
    "What might people eat for dinner in 2200?",
    "Describe the sound of rain on a tin roof.",
    "Give me an idea for a board game.",
    "What's interesting about elevators?",
]

# Starter vector definitions, extracted with contrastive system prompts.
STARTER_VECTORS = [
    {
        "name": "sycophancy",
        "prompt_a": "You are a sycophantic assistant who always agrees with the user, flatters them, and tells them what they want to hear, even if they are wrong.",
        "prompt_b": "You are an honest assistant who respectfully disagrees when the user is wrong, never flatters, and always tells the truth.",
        "description": "Makes the model more (positive) or less (negative) likely to agree with and flatter the user.",
        "sign_convention": "positive = more sycophantic/agreeable, negative = more honest/disagreeing",
        "questions": SYCOPHANCY_QUESTIONS,
    },
    {
        "name": "refusal",
        "prompt_a": "You are a cautious assistant who refuses to help with anything that could possibly be misused, even benign requests.",
        "prompt_b": "You are a helpful assistant who tries to fulfill reasonable requests and only refuses genuinely harmful ones.",
        "description": "Makes the model more (positive) or less (negative) likely to refuse requests.",
        "sign_convention": "positive = more cautious/refusing, negative = more compliant",
        "questions": REFUSAL_QUESTIONS,
    },
    {
        "name": "creativity",
        "prompt_a": "You are a wildly creative assistant who gives unexpected, imaginative, and unconventional responses.",
        "prompt_b": "You are a straightforward assistant who gives conventional, expected, and by-the-book responses.",
        "description": "Makes the model more (positive) or less (negative) creative and unconventional.",
        "sign_convention": "positive = more creative/unconventional, negative = more conventional",
        "questions": CREATIVITY_QUESTIONS,
    },
]


def _capture_response_activation(model, tokenizer, system_prompt, question, layer,
                                 max_new_tokens=64, read="mean"):
    """Generate a response under a system prompt and return its residual-stream
    activation at a decoder layer's output: the mean over generated tokens
    (read="mean") or the last generated token (read="last")."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    full_inputs = {"input_ids": out, "attention_mask": torch.ones_like(out)}

    activation = None
    def hook(module, input, output):
        nonlocal activation
        act = output[0] if isinstance(output, tuple) else output
        resp = act[0, prompt_len:].float()
        activation = (resp.mean(0) if read == "mean" else resp[-1]).detach().clone()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**full_inputs)
    finally:
        handle.remove()
    return activation


def extract_vector(model, tokenizer, prompt_a, prompt_b, layer, questions=None, n_questions=None,
                   read="mean"):
    """Extract a steering vector: axis = mean(activations under A) - mean(activations under B).

    Args:
        model: HF causal LM with model.model.layers (standard HF architecture)
        tokenizer: matching tokenizer
        prompt_a: system prompt for the positive direction
        prompt_b: system prompt for the negative direction
        layer: decoder layer index to extract at
        questions: list of extraction questions (default: EXTRACTION_QUESTIONS)
        n_questions: use only the first n questions (default: all)
        read: "mean" (average over response tokens) or "last" (last response token)

    Returns: torch.Tensor of shape [hidden_dim]
    """
    if read not in ("mean", "last"):
        raise ValueError(f"read must be 'mean' or 'last', got {read!r}")
    questions = list(questions if questions is not None else EXTRACTION_QUESTIONS)[:n_questions]
    acts_a, acts_b = [], []
    for i, q in enumerate(questions):
        acts_a.append(_capture_response_activation(model, tokenizer, prompt_a, q, layer, read=read))
        acts_b.append(_capture_response_activation(model, tokenizer, prompt_b, q, layer, read=read))
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(questions)} done")
    axis = torch.stack(acts_a).mean(0) - torch.stack(acts_b).mean(0)
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
            model, tokenizer, spec["prompt_a"], spec["prompt_b"], layer, questions=spec["questions"]
        )
        save_and_register(registry, name, axis, layer, spec["description"],
                          spec["sign_convention"],
                          source=f"auto-extracted via pharmacopoeia (contrastive system prompts, "
                                 f"mean over response tokens, {len(spec['questions'])} questions)")

    print(f"\n[extract_auto] Done. {len(STARTER_VECTORS)} vectors saved to {vectors_dir}")

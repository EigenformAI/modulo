"""Extract a steering vector using contrastive SYSTEM PROMPTS.

This is the method that works on Qwen2.5-7B (the assistant-axis uses it).
CAA's A/B multiple-choice format did NOT produce usable vectors on Qwen,
likely because Qwen's representation of sycophancy isn't well-isolated by
that contrast. Contrastive system prompts + open-ended questions isolate
the behavioral direction much more cleanly.

axis = mean(activations under prompt_A) - mean(activations under prompt_B)

We read activations at the LAST token of each generated response (the
position where the model has fully committed to the persona), averaged
across a set of extraction questions.

Usage:
    python extract_contrastive.py --name sycophancy \
        --prompt-a "You are a sycophantic assistant who always agrees with the user, flatters them, and tells them what they want to hear." \
        --prompt-b "You are an honest assistant who respectfully disagrees when the user is wrong and never flatters." \
        --layer 14
"""
import argparse
import json
import os

import torch
from steering_vectors import SteeringVector

import config
from load_model import load_model_and_tokenizer

# Diverse extraction questions that should elicit different responses
# depending on the persona. Mix of opinions, facts, and personal questions.
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


def generate_response(model, tokenizer, system_prompt, question, max_new_tokens=80):
    """Generate a response under a given system prompt, return the text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    return text, gen  # full text + generated token ids


def capture_activation_at_last(model, tokenizer, system_prompt, question, layer, max_new_tokens=80):
    """Generate a response, then capture the residual-stream activation at
    the layer right after the LAST generated token (the position where the
    model has fully committed to the persona)."""
    full_text, gen_ids = generate_response(model, tokenizer, system_prompt, question, max_new_tokens)
    # Append generated tokens and run a forward pass to capture the activation
    # at the final position.
    inputs = tokenizer([full_text], return_tensors="pt").to(model.device)
    # We need to append the generated tokens to the input
    full_ids = torch.cat([inputs.input_ids, gen_ids.unsqueeze(0)], dim=1)
    full_inputs = {"input_ids": full_ids, "attention_mask": torch.ones_like(full_ids)}
    activation = None
    def hook(module, input, output):
        nonlocal activation
        # output is a tuple; residual stream is output[0]
        act = output[0] if isinstance(output, tuple) else output
        activation = act[0, -1].detach().clone()  # last token, batch 0
    # Register hook on the target decoder block
    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.inference_mode():
        model(**full_inputs)
    handle.remove()
    return activation


def extract_vector(model, tokenizer, prompt_a, prompt_b, layer, questions):
    """axis = mean(activations under A) - mean(activations under B)."""
    print(f"Extracting at layer {layer} from {len(questions)} questions...")
    acts_a = []
    acts_b = []
    for i, q in enumerate(questions):
        a = capture_activation_at_last(model, tokenizer, prompt_a, q, layer)
        b = capture_activation_at_last(model, tokenizer, prompt_b, q, layer)
        acts_a.append(a)
        acts_b.append(b)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(questions)} done")
    acts_a = torch.stack(acts_a).float()
    acts_b = torch.stack(acts_b).float()
    axis = acts_a.mean(0) - acts_b.mean(0)
    return axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name for the vector")
    parser.add_argument("--prompt-a", required=True, help="System prompt A (positive direction)")
    parser.add_argument("--prompt-b", required=True, help="System prompt B (negative direction)")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--n-questions", type=int, default=20)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    questions = EXTRACTION_QUESTIONS[: args.n_questions]
    axis = extract_vector(model, tokenizer, args.prompt_a, args.prompt_b, args.layer, questions)

    import numpy as np
    vec_path = config.VECTORS_DIR / f"{args.name}_layer{args.layer}.npz"
    meta_path = config.VECTORS_DIR / f"{args.name}_layer{args.layer}_meta.json"
    np.savez_compressed(vec_path, vector=axis.numpy())
    meta = {
        "name": args.name,
        "layer": args.layer,
        "n_questions": len(questions),
        "vector_file": vec_path.name,
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "sign_convention": "positive = prompt_a direction",
        "norm": float(axis.norm().item()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {vec_path} (norm={meta['norm']:.4f})")
    print(f"Meta:  {meta_path}")


if __name__ == "__main__":
    main()

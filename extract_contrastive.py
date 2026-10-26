"""Extract a steering vector using contrastive SYSTEM PROMPTS and register it.

This is the method that works on Qwen2.5-7B (the assistant-axis uses it).
(Earlier notes said CAA's A/B format did not produce usable vectors on Qwen,
but those tests ran with steering that never actually applied during
generation, so that conclusion needs re-testing.)

axis = mean(activations under prompt_A) - mean(activations under prompt_B)

Each response's activation is the mean over its generated tokens (--read mean,
the default), averaged across a set of extraction questions. --read last uses
only the final token instead; on Qwen2.5-7B that was far less reliable
(split-half cosine 0.33 vs 0.83 for sycophancy, 0.10 vs 0.68 for refusal).
The vector is saved into the model's registry directory (vectors/<model_slug>/)
so the agent can use it right away.

Usage:
    python extract_contrastive.py --name sycophancy \
        --prompt-a "You are a sycophantic assistant who always agrees with the user, flatters them, and tells them what they want to hear." \
        --prompt-b "You are an honest assistant who respectfully disagrees when the user is wrong and never flatters." \
        --layer 14 \
        --description "Makes the model more (positive) or less (negative) likely to flatter the user." \
        --sign-convention "positive = more sycophantic, negative = more honest"
"""
import argparse

import config  # sets HF_HOME before transformers is imported
from load_model import load_model_and_tokenizer
from pharmacopoeia import VectorRegistry, extract_vector, save_and_register
from pharmacopoeia.extraction import EXTRACTION_QUESTIONS
from pharmacopoeia.registry import registry_dir_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name for the vector")
    parser.add_argument("--prompt-a", required=True, help="System prompt A (positive direction)")
    parser.add_argument("--prompt-b", required=True, help="System prompt B (negative direction)")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--description", default=None, help="Shown to the agent in its system prompt")
    parser.add_argument("--sign-convention", default="positive = prompt A direction, negative = prompt B direction")
    parser.add_argument("--read", choices=["mean", "last"], default="mean",
                        help="Average over response tokens (default) or use only the last one")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    questions = EXTRACTION_QUESTIONS[: args.n_questions]
    print(f"Extracting at layer {args.layer} from {len(questions)} questions...")
    axis = extract_vector(model, tokenizer, args.prompt_a, args.prompt_b, args.layer, questions, read=args.read)

    registry = VectorRegistry(registry_dir_for(config.MODEL_NAME))
    save_and_register(
        registry, args.name, axis, args.layer,
        description=args.description or f"Contrast of system prompts. A: {args.prompt_a!r} B: {args.prompt_b!r}",
        sign_convention=args.sign_convention,
        source=f"extract_contrastive.py, read={args.read}, {len(questions)} questions. "
               f"A={args.prompt_a!r} B={args.prompt_b!r}",
    )


if __name__ == "__main__":
    main()

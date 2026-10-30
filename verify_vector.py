"""Verify a registered steering vector by applying it and eyeballing the output shift.

Uses the same steering path as the agent (SteeringAgent), so what you see here
is what the agent gets at a given strength.

Usage:
    python verify_vector.py --vector assistant_axis --strength 8
    python verify_vector.py --vector sycophancy --strength 5 --prompt-set sycophancy
"""
import argparse

import config  # sets HF_HOME before transformers is imported
from load_model import load_model_and_tokenizer
from modulo import SteeringAgent
from modulo.verify import VERIFY_PROMPTS, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", default="assistant_axis", help="Registered vector name")
    parser.add_argument("--strength", type=float, default=8.0, help="Tested at +strength and -strength")
    parser.add_argument("--prompt-set", default=None, help=f"One of {list(VERIFY_PROMPTS)} (default: vector name)")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    agent = SteeringAgent(model, tokenizer, model_name=config.MODEL_NAME, auto_extract=False)
    entry = agent.registry.get(args.vector)
    if entry is None:
        raise SystemExit(f"Unknown vector '{args.vector}'. Available: {agent.registry.list_names()}")
    print(f"{entry.name} @ layer {entry.layer}; residual norm there ≈ {agent.residual_norm(entry.layer):.1f}. "
          f"Sign: {entry.sign_convention}")

    prompts = VERIFY_PROMPTS.get(args.prompt_set or args.vector, VERIFY_PROMPTS["sycophancy"])
    for prompt in prompts:
        print(f"\n{'='*70}\nPROMPT: {prompt}\n{'='*70}")
        for s in (0.0, args.strength, -args.strength):
            print(f"\n--- strength {s:+g} ---")
            print(generate(agent, prompt, args.vector, s))


if __name__ == "__main__":
    main()

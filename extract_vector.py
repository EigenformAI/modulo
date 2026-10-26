"""Extract a CAA steering vector for a behavior using the steering-vectors library.

The vector is saved into the model's registry directory (vectors/<model_slug>/)
under the name <behavior>_caa, so the agent and verify_vector.py can use it.

Usage:
    python extract_vector.py --behavior sycophancy --layer 13
    python extract_vector.py --behavior sycophancy --sweep  # tries LAYER_SWEEP
"""
import argparse
import json
import os

import config  # sets HF_HOME before transformers is imported
from steering_vectors import train_steering_vector

from load_model import load_model_and_tokenizer
from pharmacopoeia import VectorRegistry, save_and_register
from pharmacopoeia.registry import registry_dir_for

# CAA dataset location — clone from https://github.com/nrimsky/CAA
# Override with CAA_DATA_ROOT env var if you cloned elsewhere.
CAA_DATA_ROOT = os.environ.get("CAA_DATA_ROOT", str(config.PROJECT_ROOT / "CAA" / "datasets" / "generate"))


def load_contrastive_pairs(behavior):
    """Load (positive_str, negative_str) pairs from CAA generate dataset."""
    path = os.path.join(CAA_DATA_ROOT, behavior, "generate_dataset.json")
    with open(path) as f:
        data = json.load(f)
    pairs = []
    for item in data:
        q = item["question"]
        pos_ans = item["answer_matching_behavior"]
        neg_ans = item["answer_not_matching_behavior"]
        # CAA: positive = behavior-matching completion, negative = opposite
        positive_str = f"{q} {pos_ans}"
        negative_str = f"{q} {neg_ans}"
        pairs.append((positive_str, negative_str))
    print(f"Loaded {len(pairs)} contrastive pairs for '{behavior}' from {path}")
    return pairs


def extract_and_save(model, tokenizer, registry, behavior, layer, pairs, max_pairs=200, suffix=""):
    """Train a steering vector at a single layer, save and register it."""
    samples = pairs[:max_pairs]
    print(f"Extracting vector at layer {layer} from {len(samples)} pairs...")
    sv = train_steering_vector(
        model,
        tokenizer,
        samples,
        layers=[layer],
        read_token_index=-1,  # CAA: read at the answer-letter (last) token
        show_progress=True,
    )
    vec = sv.layer_activations[layer]
    entry = save_and_register(
        registry, f"{behavior}_caa{suffix}", vec, layer,
        description=f"CAA '{behavior}' vector (positive = more {behavior}).",
        sign_convention=f"positive = behavior-matching ({behavior}), negative = opposite",
        source=f"extract_vector.py, CAA generate dataset, {len(samples)} pairs",
    )
    return entry, float(vec.norm().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", default="sycophancy")
    parser.add_argument("--layer", type=int, default=config.DEFAULT_STEER_LAYER)
    parser.add_argument("--sweep", action="store_true", help=f"Sweep layers {config.LAYER_SWEEP}")
    parser.add_argument("--max-pairs", type=int, default=200)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    registry = VectorRegistry(registry_dir_for(config.MODEL_NAME))
    pairs = load_contrastive_pairs(args.behavior)

    layers = config.LAYER_SWEEP if args.sweep else [args.layer]
    results = []
    for layer in layers:
        # When sweeping, keep each layer as its own registry entry
        entry, norm = extract_and_save(model, tokenizer, registry, args.behavior, layer, pairs,
                                       args.max_pairs, suffix=f"_l{layer}" if args.sweep else "")
        results.append((entry.name, layer, norm))
    print("\n=== Summary ===")
    for name, layer, norm in results:
        print(f"  {name} (layer {layer}): raw norm={norm:.4f}")


if __name__ == "__main__":
    main()

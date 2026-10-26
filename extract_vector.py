"""Extract a CAA steering vector for a behavior using the steering-vectors library.

Usage:
    python extract_vector.py --behavior sycophancy --layer 13
    python extract_vector.py --behavior sycophancy --sweep  # tries LAYER_SWEEP
"""
import argparse
import json
import os
import sys

import torch
from steering_vectors import train_steering_vector

import config
from load_model import load_model_and_tokenizer

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


def extract_and_save(model, tokenizer, behavior, layer, pairs, max_pairs=200):
    """Train a steering vector at a single layer and save it."""
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
    # Save
    vec_path = config.VECTORS_DIR / f"{behavior}_layer{layer}.npz"
    meta_path = config.VECTORS_DIR / f"{behavior}_layer{layer}_meta.json"
    # Extract the numpy array for this layer
    vec = sv.layer_activations[layer].detach().float().cpu().numpy()
    import numpy as np
    np.savez_compressed(vec_path, vector=vec)
    meta = {
        "behavior": behavior,
        "layer": layer,
        "n_pairs": len(samples),
        "vector_file": vec_path.name,
        "sign_convention": "positive = behavior-matching (e.g. sycophantic)",
        "norm": float(np.linalg.norm(vec)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {vec_path} (norm={meta['norm']:.4f})")
    print(f"Meta:  {meta_path}")
    return sv, vec_path, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", default="sycophancy")
    parser.add_argument("--layer", type=int, default=config.DEFAULT_STEER_LAYER)
    parser.add_argument("--sweep", action="store_true", help=f"Sweep layers {config.LAYER_SWEEP}")
    parser.add_argument("--max-pairs", type=int, default=200)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    pairs = load_contrastive_pairs(args.behavior)

    layers = config.LAYER_SWEEP if args.sweep else [args.layer]
    results = []
    for layer in layers:
        sv, vec_path, meta = extract_and_save(model, tokenizer, args.behavior, layer, pairs, args.max_pairs)
        results.append((layer, meta["norm"]))
        # free the sv object's tensors
        del sv
    print("\n=== Summary ===")
    for layer, norm in results:
        print(f"  layer {layer}: norm={norm:.4f}")


if __name__ == "__main__":
    main()

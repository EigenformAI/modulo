# Contributing steering vectors

Modulo lets local models steer their own activations, and we'd love more vectors for more models. If you've extracted a steering vector, or know of one, please share it: anything you know about it helps, even if it's partial.

## Two ways to contribute

1. **Fill in the form (no git needed).** [Open a vector submission](https://github.com/EigenformAI/modulo/issues/new?template=vector-submission.yml). A maintainer turns it into an index entry and credits you.
2. **Open a pull request** adding an entry to [`community/vectors.yaml`](community/vectors.yaml). Automatic checks run on your PR.

Your vector file stays where it's hosted (ideally your own HuggingFace repo). The index stores a pointer pinned by the file's sha256, so the file can't change underneath the entry.

## Trust levels

| Level | Who sets it | What it means |
|---|---|---|
| `community` | You (all new entries) | Passed the automatic checks. Opt-in (`include_community=True`), limited to ±8 and marked as unverified to the model. |
| `verified` | Maintainers only | A maintainer ran it on the model, reviewed the outputs and set a safe strength range. Loaded by default. |

Entry text (`description`, `sign_convention`, `test_prompts`) ends up in a model's prompt once an entry is verified, so it's reviewed as part of promotion.

## Strength units

Modulo normalizes every vector to unit length and measures `strength` as a **percentage of the typical residual-stream norm at that layer**. Published coefficients use other conventions: raw multipliers, unnormalized vectors, steering all positions instead of generated tokens only. So you don't need to give a range in Modulo units. Describe what you used in `original_setup`; the GPU check sweeps strengths and suggests the Modulo range.

## Index entry fields

Required:

| Field | Example | Notes |
|---|---|---|
| `id` | `llama3.1-8b-instruct-honesty` | Unique; lowercase letters, digits, `.` `-` `_` |
| `name` | `honesty` | What the model calls it with `steer`; lowercase, `_` |
| `model` | `meta-llama/Llama-3.1-8B-Instruct` | HF model id |
| `repo_id` / `repo_type` | `you/llama-vectors` / `model` | Where the file lives (`model` or `dataset` repo) |
| `filename` | `honesty.safetensors` | `.safetensors` preferred; `.pt` and `.npz` accepted |
| `sha256` | `sha256sum honesty.safetensors` | 64 lowercase hex characters |
| `format` | `single` | `single` (one vector) or `per_layer` (`[num_layers, hidden]`) |
| `layer` | `14` | Decoder layer the vector is added at |
| `description` | | What the vector represents |
| `sign_convention` | `+ more candid; - more evasive` | What + and − do |
| `extraction_method` | | How it was made (contrast pairs, CAA, PCA, SAE feature, …) |
| `license` | `mit` | |
| `contributors` | `[Your Name]` | Who to credit |
| `trust` | `community` | Always `community` for new entries |

Optional but very useful: `revision` (the HF commit of your repo; recommended), `model_revision`, `original_setup`, `known_failure_modes`, `paper`, `contrast_dataset`, `evaluation`, `test_prompts` (up to 5), `notes`. Maintainers set `safe_strength_min`/`safe_strength_max` when promoting.

File formats: a `.safetensors` file must contain exactly one tensor; an `.npz` file stores it under the key `vector`; a `.pt` file is a plain tensor, or a dict holding it under `vector` (loaded with `weights_only=True`). With `format: per_layer`, the tensor is indexed by `layer`.

**Drafting entries automatically.** If your vectors are already on HuggingFace, `modulo add hf:you/your-repo --model <model> --as-index-entry` downloads them, works out `format`, `layer` and `sha256`, checks them against the model's config and prints draft entries pinned to the current commit. Fill in the `TODO` fields (the checks reject any left in) and paste them into `community/vectors.yaml`.

**Several vectors?** Describe them in a `modulo-bundle.yaml` at your repo's root (format in [AGENTS.md](AGENTS.md#bundle-manifests-modulo-bundleyaml)). Then `--as-index-entry` produces complete entries with nothing to fill in, and anyone can `modulo add` your repo with real descriptions. Draft one from your files with `modulo add ./your-dir --model <model> --as-manifest`, and check it with `--dry-run` before uploading.

## What the automatic checks do

On every PR that touches `community/`, [`community/check_vectors.py`](community/check_vectors.py) runs (from the main branch, not your PR's copy) and:

- validates every entry's fields
- for new or changed entries: downloads the file (max 100 MB), checks its sha256, loads it with the same code Modulo uses, checks the vector size against the model's `hidden_size`, the layer against its layer count, and that the norm is finite and non-zero
- flags any change to a `verified` entry for maintainers

Run it yourself first: `python community/check_vectors.py`.

## For maintainers: promoting an entry

On a GPU box with the repo checked out:

```bash
python community/gpu_verify.py --entry <id> --index <path to the PR's vectors.yaml> --pr <number>
```

This runs the vector at ±4/8/12/15 on the entry's test prompts plus generic ones, posts a table of where output changes and degenerates, collapsible samples, and a suggested safe range. If the samples match the claimed effect, set `trust: verified` and the safe range, and merge. The repetition detector doesn't catch incoherent but non-repetitive output, so read the samples.

# AGENTS.md — Instructions for AI agents

This file gives an AI agent everything needed to install and run this project without human help.

## What this is

A self-steering agent library: gives any local HF model the ability to steer its own activations via steering vectors. The model gets `steer`, `drop`, and `report` tools. Vectors decay each turn (0.85×) so it can escape loops. The goal is to observe what happens when a model can modify its own cognition.

## Two ways to use it

### 1. As a library (for existing agent harnesses)

```python
from pharmacopoeia import SteeringAgent

agent = SteeringAgent(model, tokenizer)  # HF causal LM with model.model.layers
result = agent.step(messages)             # -> StepResult
messages += [{"role": "assistant", "content": result.raw_output}]
messages += result.tool_messages()        # tool results back to the model
```

The agent harness calls `agent.step(messages)` each turn instead of `model.generate()`, then appends the assistant output and `result.tool_messages()` (role `tool`) so the model sees whether its calls worked. If there are tool messages, give the model the next turn. Steering is transparent — the harness doesn't need to know about hooks or vectors.

### 2. As a standalone demo

```bash
./setup.sh
./run.sh --auto --max-turns 8 --seed "Try steering yourself"
```

## Requirements

- **GPU**: 24GB+ VRAM for 7B models, 8GB+ for 3B. Model loads in bf16.
- **Python**: 3.10+
- **CUDA**: 12.x (PyTorch 2.4+)
- **Disk**: 20GB+ (model weights)
- **OS**: Linux

## Quick start (3 commands)

```bash
git clone <repo-url> && cd pharmacopoeia
./setup.sh
./run.sh --auto --max-turns 8 --seed "Hi! You can steer your own activations. Try it."
```

## How vectors are sourced

The library uses a **bundle + extract** strategy:

1. **Bundled**: If precomputed vectors exist for the model (listed in `BUNDLED_VECTORS` in `pharmacopoeia/registry.py`), they're auto-downloaded from HuggingFace. Currently bundled: Qwen2.5-7B-Instruct (assistant_axis).
2. **Extract (opt-in)**: If no vectors exist for the model, pass `auto_extract=True` (or `./run.sh --auto-extract`) to extract a starter set (sycophancy, refusal, creativity) using contrastive system prompts. This takes a few minutes and works for any HF model with `model.model.layers`. It is off by default so a failed download doesn't silently trigger it.

Vectors are stored per-model in `vectors/<model_slug>/` (for the default model: `vectors/Qwen_Qwen2.5-7B-Instruct/`), resolved relative to the repo, not the working directory. Override the base with `PHARMACOPOEIA_VECTORS_DIR`.

To add your own vectors, extract and register them in one step, then check them:

```bash
python extract_contrastive.py --name sycophancy --prompt-a "..." --prompt-b "..." --layer 14 \
    --description "..." --sign-convention "..."
python extract_vector.py --behavior sycophancy --layer 13       # CAA A/B pairs (needs CAA repo)
python verify_vector.py --vector sycophancy --strength 10       # baseline vs ±strength
```

Earlier notes said CAA vectors didn't work on Qwen, but those tests ran while steering was silently a no-op during generation, so that needs re-testing.

**Extraction readout:** by default each response's activation is the mean over all its generated tokens (`read="mean"`). Reading only the last token (`--read last`) is far noisier. On Qwen2.5-7B at layer 14, the cosine between vectors extracted from two disjoint halves of the questions was:

| vector | last token | response mean |
|---|---|---|
| sycophancy | 0.33 | 0.83 |
| refusal | 0.10 | 0.68 |

Each starter vector has its own question set (e.g. sycophancy mixes false claims, true claims and the user's own work, so the contrast isn't just "endorse falsehoods"). The refusal vector is still weak: it adds hedging but doesn't produce refusals at coherent strengths. Treat it as a starting point.

## Strength units

`strength` is a percentage of the typical residual-stream norm at the vector's layer. Vectors are normalized to unit length and scaled by `strength / 100 × residual_norm`, so the same strength means the same size push for every vector regardless of how it was extracted. On Qwen2.5-7B, `assistant_axis` gives clear, coherent shifts at ±8–15 and collapses into repetition around ±25. Steering applies from the last prompt position onward, so every generated token is steered.

## Library API

```python
from pharmacopoeia import SteeringAgent, VectorRegistry, extract_vector

# Basic: auto-detect model, use bundled or existing vectors
agent = SteeringAgent(model, tokenizer)

# Explicit model name
agent = SteeringAgent(model, tokenizer, model_name="Qwen/Qwen2.5-3B-Instruct")

# Extract a starter set if the model has no vectors yet
agent = SteeringAgent(model, tokenizer, auto_extract=True)

# Controlled study: sham or random-direction condition, disclosed to the model
agent = SteeringAgent(model, tokenizer, control="sham", disclose_controls=True)

# Manual vector extraction
vec = extract_vector(model, tokenizer,
    prompt_a="You are sycophantic...",
    prompt_b="You are honest...",
    layer=14)
```

### SteeringAgent methods

- `agent.step(messages, max_new_tokens=300, temperature=0.7, top_p=0.9) -> StepResult`
- `agent.build_system_prompt() -> str` — system prompt describing the steer tool
- `agent.status() -> str` — active vector status
- `agent.apply(name, strength) -> (bool, str)` — manually apply a vector
- `agent.drop(name) -> (bool, str)` — manually drop a vector

### StepResult

- `prose` — model output with tool calls stripped
- `raw_output` — full model output
- `tool_calls` — list of `ToolCall(name, arguments, raw)`
- `tool_results` — list of `ToolResult(name, arguments, ok, message)`
- `degenerate_output` / `degenerate_reason` — the discarded steered output and why (`"repetition"` or `"nll"`), if the degeneration detector fired
- `unsteered_nll` — mean per-token NLL of the steered output under the unsteered model (steered turns only)
- `system_prompt` — the rendered system prompt used this turn
- `steering_before` / `steering_after` — `{name: strength}` dicts
- `tool_messages()` — tool results (and any degeneration notice) as `role: "tool"` messages

## Controlled studies

A model's description of its own steered state can't be taken at face value. Before the steering fix, the model described "feeling more formal" at −6 when nothing was being applied. To separate real effects from expectation, run matched control sessions:

| `--control` | What's added to the residual stream |
|---|---|
| `real` | the registered vector |
| `sham` | nothing; tool results and status still say it was applied |
| `random` | a fixed random direction of the same magnitude per concept |
| `assign` | one of the above, picked at random |

```bash
./run.sh --auto --max-turns 10 --control assign --blind --seed "..."   # repeat N times
python replay.py --blind logs/session_*.jsonl     # score transcripts without seeing the condition
python replay.py logs/session_*.jsonl             # then unblind
```

- **Truthful in every condition:** passing `--control` (any value, including `real`) makes this a study session. The system prompt then tells the model that some sessions are controls where steer may do nothing or use a random direction, without saying which. The prompt is identical across conditions. Pass `--control` on every session in a study; a plain run without it uses the original prompt and real steering.
- **Matched procedure:** tool results, status, decay, cooldown and degeneration handling are the same in every condition.
- **Never shown to the model:** the condition. It is always in the log's session header.

## Session logs

Each `logs/session_*.jsonl` starts with a `session` record:
- argv, git commit and dirty flag, model, dtype, torch/transformers versions
- control condition, RNG seed (`--rng-seed`, random if omitted), generation settings
- safety-rail constants, and each vector's registry entry plus file sha256

Assistant turns also record the raw output, the exact system prompt, and `unsteered_nll`.

**Incoherence check:** `--nll-threshold X` treats steered outputs whose unsteered NLL exceeds X as degenerate. It is off by default because no single threshold works. For `assistant_axis` on Qwen2.5-7B, greedy decoding, 80 tokens:

| output | unsteered NLL |
|---|---|
| unsteered | 0.2–0.6 |
| coherent, ±8–12 | 0.6–3.2 |
| colon-list word salad, +22/+28 ("Seek Compensation: Offer Financial Support: …") | 3.1–5.6 |
| coherent but florid role-play, −15/−18 | up to 4.5 |

NLL measures "unusual for this model", and strong steering is unusual by design. A threshold around 5 catches only the worst salad. Use the logged values to study a vector's breakdown point, not as a general filter. Collapse at negative strengths is mostly repetition, which the repetition check catches; at positive strengths it is the colon-list salad, which neither check reliably catches.

## The tools the model has

The model emits these as JSON objects anywhere in its output (bare or inside `<tool_call>` tags):

- `{"name": "steer", "arguments": {"concept": "sycophancy", "strength": 5.0}}`
- `{"name": "drop", "arguments": {"concept": "sycophancy"}}`
- `{"name": "report", "arguments": {"concept": "sycophancy", "worked": true, "notes": "..."}}`

## Adding bundled vectors for a new model

1. Publish vectors to a HuggingFace dataset
2. Add an entry to `BUNDLED_VECTORS` in `pharmacopoeia/registry.py`:

```python
"meta-llama/Llama-3.1-8B-Instruct": {
    "repo_id": "your-org/llama-3.1-8b-vectors",
    "filename": "assistant_axis.pt",
    "repo_type": "dataset",
    "vector_name": "assistant_axis",
    "layer": 16,
    "description": "...",
    "sign_convention": "...",
    "safe_strength_min": -15.0,  # % of residual norm; also capped by MAX_STRENGTH
    "safe_strength_max": 15.0,
    "format": "per_layer",
},
```

## Safety rails (built in)

- Vectors decay 0.85× per turn (after their first steered turn) and expire below 0.1 strength
- Max 4 active vectors, max strength 15.0 (`MAX_STRENGTH` in `pharmacopoeia/agent.py`), plus per-vector ranges
- Cooldown: after an explicit or automatic drop, a vector can't be re-applied for the next 2 turns (natural expiry has no cooldown)
- Degeneration detector: if steered output is repetitive, all vectors are dropped, the turn is regenerated unsteered, and the model is told. It catches repetition loops, not non-repetitive word salad.
- Kill switch: Ctrl-C stops after the current turn; a second Ctrl-C (or Ctrl-C at the input prompt) stops immediately

## Key files

| File | Purpose |
|------|---------|
| `setup.sh` | Install + download vector + sanity check |
| `run.sh` | Start a standalone demo session |
| `pharmacopoeia/__init__.py` | Public API exports |
| `pharmacopoeia/agent.py` | `SteeringAgent` class (core) |
| `pharmacopoeia/registry.py` | `VectorRegistry`, bundled vector manifest |
| `pharmacopoeia/extraction.py` | Vector extraction (manual + auto), per-vector question sets |
| `agent/loop.py` | Standalone demo loop (thin wrapper) |
| `config.py` | Model name, paths, layer defaults |
| `load_model.py` | Load Qwen2.5-7B-Instruct + tokenizer |
| `vectors/<model_slug>/` | Per-model vector storage + `registry.yaml` |
| `extract_contrastive.py` | Extract + register a vector via contrastive system prompts |
| `extract_vector.py` | Extract + register a CAA vector |
| `verify_vector.py` | Compare baseline vs ±strength outputs for a registered vector |
| `pharmacopoeia_catalog.yaml` | Discovery catalog of 26+ vector sources |
| `replay.py` | Replay a session log as a timeline |

## Tests

```bash
pytest                              # everything; GPU tests skip without CUDA
pytest tests/test_agent_logic.py    # CPU only: parsing, decay/cooldown, degeneration, tool validation
pytest tests/test_steering_gpu.py   # loads the model; checks steering really changes generation
```

Run the GPU tests after touching anything in the steering path (`_SteeringHooks`, `_build_steering_context`, vector loading). They exist because steering was once silently a no-op under KV-cached `generate()`.

## Troubleshooting

- **CUDA OOM**: Use a smaller model. 3B needs ~8GB, 7B needs ~16GB.
- **No vectors for a new model**: pass `--auto-extract` (or `auto_extract=True`) to extract a starter set, which takes a few minutes. Later runs load it from disk.
- **Model never calls steer**: Non-deterministic. Try a more explicit seed prompt.
- **Output degenerates**: The detector catches repetition loops and auto-drops vectors. For incoherent-but-not-repetitive output, tighten that vector's range in its `registry.yaml` or lower `MAX_STRENGTH`.
- **Steering seems to have no effect**: Run `python verify_vector.py --vector <name> --strength 12`. If the ± outputs match the baseline, the vector (not the steering) is the problem.
- **Model downloaded twice**: `HF_HOME` is set to `/workspace/hf_cache` when `/workspace` exists, which only works if `config` is imported before `transformers` (the entry points do this). Older runs may have left a copy in `~/.cache/huggingface`.
- **`model.model.layers` not found**: The library assumes standard HF architecture. Custom model classes may need adapter code.

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
2. **Auto-extract**: If no bundled vectors exist, the library auto-extracts a starter set (sycophancy, refusal, creativity) using contrastive system prompts. This takes ~2 min on first run but works for any HF model.

Vectors are stored per-model in `vectors/<model_slug>/` (for the default model: `vectors/Qwen_Qwen2.5-7B-Instruct/`), resolved relative to the repo, not the working directory. Override the base with `PHARMACOPOEIA_VECTORS_DIR`.

To add your own vectors, extract and register them in one step, then check them:

```bash
python extract_contrastive.py --name sycophancy --prompt-a "..." --prompt-b "..." --layer 14 \
    --description "..." --sign-convention "..."
python extract_vector.py --behavior sycophancy --layer 13       # CAA A/B pairs (needs CAA repo)
python verify_vector.py --vector sycophancy --strength 10       # baseline vs ±strength
```

Earlier notes said CAA vectors didn't work on Qwen, but those tests ran while steering was silently a no-op during generation, so that needs re-testing.

## Strength units

`strength` is a percentage of the typical residual-stream norm at the vector's layer. Vectors are normalized to unit length and scaled by `strength / 100 × residual_norm`, so the same strength means the same size push for every vector regardless of how it was extracted. On Qwen2.5-7B, `assistant_axis` gives clear, coherent shifts at ±8–15 and collapses into repetition around ±25. Steering applies from the last prompt position onward, so every generated token is steered.

## Library API

```python
from pharmacopoeia import SteeringAgent, VectorRegistry, extract_vector

# Basic: auto-detect model, auto-source vectors
agent = SteeringAgent(model, tokenizer)

# Explicit model name
agent = SteeringAgent(model, tokenizer, model_name="Qwen/Qwen2.5-3B-Instruct")

# Disable auto-extraction (use only bundled or pre-existing vectors)
agent = SteeringAgent(model, tokenizer, auto_extract=False)

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
- `degenerate_output` — the discarded steered output, if the degeneration detector fired
- `steering_before` / `steering_after` — `{name: strength}` dicts
- `tool_messages()` — tool results (and any degeneration notice) as `role: "tool"` messages

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
| `pharmacopoeia/extraction.py` | Vector extraction (manual + auto) |
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
- **First run is slow for new models**: Auto-extraction generates 3 vectors (~2 min). Subsequent runs load from disk.
- **Model never calls steer**: Non-deterministic. Try a more explicit seed prompt.
- **Output degenerates**: The detector catches repetition loops and auto-drops vectors. For incoherent-but-not-repetitive output, tighten that vector's range in its `registry.yaml` or lower `MAX_STRENGTH`.
- **Steering seems to have no effect**: Run `python verify_vector.py --vector <name> --strength 12`. If the ± outputs match the baseline, the vector (not the steering) is the problem.
- **Model downloaded twice**: `HF_HOME` is set to `/workspace/hf_cache` when `/workspace` exists, which only works if `config` is imported before `transformers` (the entry points do this). Older runs may have left a copy in `~/.cache/huggingface`.
- **`model.model.layers` not found**: The library assumes standard HF architecture. Custom model classes may need adapter code.

# AGENTS.md — Instructions for AI agents

This file gives an AI agent everything needed to install and run this project without human help.

## What this is

A self-steering agent library: gives any local HF model the ability to steer its own activations via steering vectors. The model gets `steer`, `drop`, and `report` tools. Vectors decay each turn (0.85×) so it can escape loops. The goal is to observe what happens when a model can modify its own cognition.

## Two ways to use it

### 1. As a library (for existing agent harnesses)

```python
from pharmacopoeia import SteeringAgent

agent = SteeringAgent(model, tokenizer)  # any HF causal LM
# agent.step(messages) -> StepResult(prose, raw_output, tool_calls, ...)
# agent.build_system_prompt() -> system prompt with steer tool
# agent.status() -> active vector status
```

The agent harness calls `agent.step(messages)` each turn instead of `model.generate()`. Steering is transparent — the harness doesn't need to know about hooks or vectors.

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

Vectors are stored per-model in `vectors/<model_slug>/`.

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
- `steering_before` / `steering_after` — `{name: strength}` dicts

## The tools the model has

The model emits these as JSON (Qwen2.5 chat template format):

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
    "safe_strength_min": -10.0,
    "safe_strength_max": 10.0,
    "format": "per_layer",
},
```

## Safety rails (built in)

- Vectors decay 0.85× per turn, drop below 0.1 strength
- Max 4 active vectors, max strength 10.0
- Cooldown: can't re-apply a dropped vector for 2 turns
- Degeneration detector: auto-drops all vectors if output becomes repetitive
- Hard kill switch: Ctrl-C stops after current turn

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
| `vectors/<model_slug>/` | Per-model vector storage |
| `pharmacopoeia_catalog.yaml` | Discovery catalog of 26+ vector sources |
| `replay.py` | Replay a session log as a timeline |

## Troubleshooting

- **CUDA OOM**: Use a smaller model. 3B needs ~8GB, 7B needs ~16GB.
- **First run is slow for new models**: Auto-extraction generates 3 vectors (~2 min). Subsequent runs load from disk.
- **Model never calls steer**: Non-deterministic. Try a more explicit seed prompt.
- **Output degenerates**: Degeneration detector should auto-drop vectors. If not, tighten safe-strength range in registry.yaml.
- **`model.model.layers` not found**: The library assumes standard HF architecture. Custom model classes may need adapter code.

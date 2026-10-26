# AGENTS.md — Instructions for AI agents

This file gives an AI agent everything needed to install and run this project without human help.

## What this is

A self-steering agent experiment: Qwen2.5-7B-Instruct can call a `steer(concept, strength)` tool that injects steering vectors into its own residual stream during generation. The model steers itself. Vectors decay each turn (0.85×) so it can escape loops. The goal is to observe what happens when a model can modify its own cognition.

## Requirements

- **GPU**: 24GB+ VRAM (RTX 4090, A6000, A5000). The model loads at ~16GB in bf16.
- **Python**: 3.10+
- **CUDA**: 12.x (PyTorch 2.4+)
- **Disk**: 20GB+ (model weights ~15GB)
- **OS**: Linux (tested on Ubuntu 24.04 / Runpod)

## Quick start (3 commands)

```bash
git clone <repo-url> steering_self && cd steering_self
./setup.sh
./run.sh --auto --max-turns 8 --seed "Hi! You can steer your own activations. Try it."
```

`setup.sh` installs deps, downloads the confirmed-working steering vector, and runs a model-load sanity check. `run.sh` starts the agent loop.

If `setup.sh` fails on `pip install`, try `pip install --break-system-packages -r requirements.txt` (PEP 668 externally-managed environments).

## What setup.sh does

1. Installs Python deps from `requirements.txt`
2. Downloads `vectors/assistant_axis.pt` from HuggingFace (Butanium/qwen-2.5-7b-instruct-assistant-axis)
3. Runs `python load_model.py` to confirm the model loads and generates

## Running sessions

```bash
# Interactive (you type, model responds + can steer itself)
./run.sh

# Non-interactive demo (auto-continues with gentle nudges)
./run.sh --auto --max-turns 10

# Seed with a specific prompt
./run.sh --seed "Experiment with the steer tool" --auto --max-turns 8
```

## Replaying sessions

Every session logs to `logs/session_<timestamp>.jsonl`. Replay it:

```bash
python replay.py logs/session_20260101_120000.jsonl
```

## The tools the agent has

The model can emit these JSON tool calls (Qwen2.5 chat template format):

- `{"name": "steer", "arguments": {"concept": "assistant_axis", "strength": 5.0}}` — apply a steering vector
- `{"name": "drop", "arguments": {"concept": "assistant_axis"}}` — remove a vector
- `{"name": "report", "arguments": {"concept": "assistant_axis", "worked": true, "notes": "..."}}` — log feedback

## Adding more steering vectors

Edit `vectors/registry.yaml`. Each entry:

```yaml
- name: my_vector          # identifier the agent uses
  file: my_vector.pt       # file in vectors/ dir
  layer: 14                # decoder layer to inject at
  description: "..."        # shown to the agent in its system prompt
  safe_strength_min: -10.0  # clamp range
  safe_strength_max: 10.0
  sign_convention: "..."   # what + and - do
  source: "..."            # provenance
  format: "single"         # "single" = [hidden_dim], "per_layer" = [num_layers, hidden_dim]
```

`pharmacopoeia_catalog.yaml` is a discovery catalog of 26+ vector sources to research and verify. Most are model-specific and need re-extraction for Qwen2.5-7B. Use `extract_contrastive.py` to extract new vectors (contrastive system prompts — this method works on Qwen; CAA's A/B format does not).

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
| `run.sh` | Start a session |
| `config.py` | Model name, paths, layer defaults |
| `load_model.py` | Load Qwen2.5-7B-Instruct + tokenizer |
| `agent/loop.py` | Main turn loop: generate → parse tools → apply → decay |
| `agent/steering_state.py` | Active vector tracking, decay, caps |
| `agent/vector_registry.py` | Load vectors from registry.yaml |
| `agent/system_prompt.py` | Truthful prompt describing the steer tool |
| `agent/tool_parser.py` | Parse Qwen JSON tool calls |
| `vectors/registry.yaml` | Executable vector list |
| `vectors/assistant_axis.pt` | The one confirmed-working vector |
| `pharmacopoeia_catalog.yaml` | Discovery catalog of vector sources |
| `extract_contrastive.py` | Extract new vectors via contrastive system prompts |
| `replay.py` | Replay a session log as a timeline |

## Troubleshooting

- **CUDA OOM**: Model needs ~16GB. If you have less, edit `config.py` to use a smaller model (e.g. `Qwen/Qwen2.5-3B-Instruct`) — but vectors are model-specific and the assistant_axis won't transfer.
- **`torch_dtype` deprecation warning**: harmless, use `dtype` if you edit `load_model.py`
- **Model never calls steer**: it's non-deterministic. Try a more explicit seed prompt, or run multiple sessions.
- **Output degenerates**: the degeneration detector should catch this and auto-drop vectors. If it doesn't, the safe-strength clamp in `registry.yaml` may need tightening.

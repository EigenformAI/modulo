# AGENTS.md — Instructions for AI agents

This file gives an AI agent everything needed to install and run this project without human help.

## What this is

A self-steering agent library: gives any local HF model the ability to steer its own activations via steering vectors. The model gets `steer`, `drop`, and `report` tools. Vectors decay each turn (0.85×) so it can escape loops. The goal is to observe what happens when a model can modify its own cognition.

## Two ways to use it

### 1. As a library (for existing agent harnesses)

```python
from modulo import SteeringAgent

agent = SteeringAgent(model, tokenizer)  # HF causal LM with model.model.layers
result = agent.step(messages)             # -> StepResult
messages += [{"role": "assistant", "content": result.raw_output}]
messages += result.tool_messages()        # tool results back to the model
```

Install it into the harness's environment with `pip install -e /path/to/modulo`. Keep the install editable so the default `vectors/` directory still resolves to the repo (or set `MODULO_VECTORS_DIR`).

The agent harness calls `agent.step(messages)` each turn instead of `model.generate()`, then appends the assistant output and `result.tool_messages()` (role `tool`) so the model sees whether its calls worked. If there are tool messages, give the model the next turn. If `messages` starts with a system message, it is merged into the steering system prompt. Steering is transparent — the harness doesn't need to know about hooks or vectors.

`agent.apply()` / `agent.drop()` called between steps behave like the model's own tool calls: an applied vector gets one full-strength steered turn, and a drop blocks re-applying for `COOLDOWN_TURNS` turns.

Chat templates must accept `system` and `tool` roles (Qwen, Llama-3.x, recent Mistral do). Gemma-2 rejects both, so it isn't supported yet.

### 2. As a standalone demo

```bash
./setup.sh
./run.sh --auto --max-turns 8 --seed "Try steering yourself"
```

In `--auto` mode, a turn with no tool calls is followed by a neutral `tool`-role status message (turn count, that no one is replying, current steering) so the model keeps the next turn without being prompted to steer. `--nudge legacy` uses the old user message suggesting `steer` instead. The demo offers unverified community-index vectors too (`include_community=True`); turn logs include each turn's `raw_output`.

## Requirements

- **GPU**: 24GB+ VRAM for 7B models, 8GB+ for 3B. Model loads in bf16.
- **Python**: 3.10+
- **CUDA**: 12.x (PyTorch 2.4+)
- **Disk**: 20GB+ (model weights)
- **OS**: Linux

## Quick start (3 commands)

```bash
git clone <repo-url> && cd modulo
./setup.sh
./run.sh --auto --max-turns 8 --seed "Hi! You can steer your own activations. Try it."
```

## How vectors are sourced

At startup the agent combines three sources for its model:

1. **Your own vectors**: `vectors/<model_slug>/registry.yaml`, where extraction scripts register what you make. These win over an index vector of the same name.
2. **The community index**: `community/vectors.yaml` in the repo (update the repo to get new entries; override with `MODULO_INDEX_PATH`). `verified` entries are loaded by default; unverified `community` entries only with `SteeringAgent(..., include_community=True)`, with a ±8 range unless a maintainer set one, and marked as unverified in the model's prompt. Files are downloaded from HuggingFace at their pinned revision, checked against their sha256 (and the model's dimensions), and cached in `vectors/<model_slug>/`. A cached file that doesn't match is reported and never overwritten. Index entries are kept in memory and never written to `registry.yaml`. Currently: Qwen2.5-7B-Instruct (assistant_axis).
3. **Auto-extract**: the library extracts any starter vector the model doesn't have yet (sycophancy, refusal, creativity, formality, verbosity, confidence, emotion, humor, optimism, curiosity, introspection; `STARTER_VECTORS` in `modulo/extraction.py`) using contrastive system prompts, and registers them in `registry.yaml`. This takes several minutes on first run but works for any HF model. A vector of your own with a starter name is kept and not re-extracted.

Vectors are stored per-model in `vectors/<model_slug>/` (for the default model: `vectors/Qwen_Qwen2.5-7B-Instruct/`), resolved relative to the repo, not the working directory. Override the base with `MODULO_VECTORS_DIR`.

To add your own vectors, extract and register them in one step, then check them:

```bash
python extract_contrastive.py --name sycophancy --prompt-a "..." --prompt-b "..." --layer 14 \
    --description "..." --sign-convention "..."
python extract_vector.py --behavior sycophancy --layer 13       # CAA A/B pairs (needs CAA repo)
python verify_vector.py --vector sycophancy --strength 10       # baseline vs ±strength
```

Earlier notes said CAA vectors didn't work on Qwen, but those tests ran while steering was silently a no-op during generation, so that needs re-testing.

## Strength units

`strength` is a percentage of the typical residual-stream norm at the vector's layer. Vectors are normalized to unit length and scaled by `strength / 100 × residual_norm`, so the same strength means the same size push for every vector regardless of how it was extracted. On Qwen2.5-7B, `assistant_axis` gives clear, coherent shifts at ±8–12; at +15 it turns muddled (so its range is -15 to +12), and it collapses into repetition around ±25. Steering applies from the last prompt position onward, so every generated token is steered.

## Library API

```python
from modulo import SteeringAgent, VectorRegistry, extract_vector

# Basic: auto-detect model, auto-source vectors
agent = SteeringAgent(model, tokenizer)

# Explicit model name
agent = SteeringAgent(model, tokenizer, model_name="Qwen/Qwen2.5-3B-Instruct")

# Disable auto-extraction of starter vectors (use only your own and index vectors)
agent = SteeringAgent(model, tokenizer, auto_extract=False)

# Also offer unverified community-index vectors
agent = SteeringAgent(model, tokenizer, include_community=True)

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
- `repeated_output` — the discarded output, if it nearly repeated one of the last 3 assistant turns
- `steering_before` / `steering_after` — `{name: strength}` dicts
- `tool_messages()` — tool results (and any degeneration or repetition notice) as `role: "tool"` messages

## The tools the model has

The model emits these as JSON objects anywhere in its output (bare or inside `<tool_call>` tags):

- `{"name": "steer", "arguments": {"concept": "sycophancy", "strength": 5.0}}`
- `{"name": "drop", "arguments": {"concept": "sycophancy"}}`
- `{"name": "report", "arguments": {"concept": "sycophancy", "worked": true, "notes": "..."}}`

## Reports

Each `report` call is appended as one JSON line to `vectors/<model_slug>/reports.jsonl` (git-ignored). A report records the model, vector, layer, the vector file's sha256, `worked`, `notes` (capped at 2000 chars), the strength the vector was last steered at in the session, and `turns_since_steered` (0 = active during the reporting turn; both `null` if the model never steered with it), plus `turn`, `session_id`, `modulo_version` and `schema_version`.

The format is defined in `modulo/reports.py`. `validate_report(d)` returns a list of problems, and `read_reports(path)` loads a file. A report is one model's opinion of itself: aggregate many reports rather than trusting one. Notes are free text, so never feed other people's notes back into a model's prompt.

## Adding vectors for a new model

Host the vector on HuggingFace and add an entry to `community/vectors.yaml` (or use the submission form). See [CONTRIBUTING.md](CONTRIBUTING.md) for the fields, the automatic checks and how entries become `verified`.

## Safety rails (built in)

- Vectors decay 0.85× per turn (after their first steered turn) and expire below 0.1 strength
- Max 4 active vectors, max strength 15.0 (`MAX_STRENGTH` in `modulo/agent.py`), plus per-vector ranges
- Cooldown: after an explicit or automatic drop, a vector can't be re-applied for the next 2 turns (natural expiry has no cooldown)
- Degeneration detector: if steered output is repetitive, all vectors are dropped, the turn is regenerated unsteered, and the model is told. It catches repetition loops, not non-repetitive word salad.
- Repeat detector: if a turn's prose is ≥80% similar (word-level) to one of the last 3 assistant turns, it is discarded and regenerated once (steering kept), and the model is told. This stops the model re-sending the same answer turn after turn; loose paraphrase loops get through until they settle into near-copies.
- Kill switch: Ctrl-C stops after the current turn; a second Ctrl-C (or Ctrl-C at the input prompt) stops immediately

## Key files

| File | Purpose |
|------|---------|
| `setup.sh` | Install + download vector + sanity check |
| `run.sh` | Start a standalone demo session |
| `modulo/__init__.py` | Public API exports |
| `modulo/agent.py` | `SteeringAgent` class (core) |
| `modulo/registry.py` | `VectorRegistry`: your own vectors plus the community index for a model |
| `modulo/extraction.py` | Vector extraction (manual + auto) |
| `modulo/reports.py` | Report format, validation, `reports.jsonl` read/write |
| `modulo/community.py` | Community vector index: entry format and file checks |
| `community/vectors.yaml` | Community vector index (pointers to hosted vectors, with trust levels) |
| `community/check_vectors.py` | Checks the index; run by CI on PRs touching `community/` |
| `community/gpu_verify.py` | Maintainer GPU check: strength sweep + suggested safe range |
| `CONTRIBUTING.md` | How to contribute vectors (issue form or PR) |
| `agent/loop.py` | Standalone demo loop (thin wrapper) |
| `config.py` | Model name, paths, layer defaults |
| `load_model.py` | Load Qwen2.5-7B-Instruct + tokenizer |
| `vectors/<model_slug>/` | Per-model vector storage + `registry.yaml` |
| `extract_contrastive.py` | Extract + register a vector via contrastive system prompts |
| `extract_vector.py` | Extract + register a CAA vector |
| `verify_vector.py` | Compare baseline vs ±strength outputs for a registered vector |
| `modulo_catalog.yaml` | Discovery catalog of 26+ vector sources |
| `replay.py` | Replay a session log as a timeline |

## Tests

```bash
pytest                              # everything; GPU tests skip without CUDA
pytest tests/test_agent_logic.py    # CPU only: parsing, decay/cooldown, degeneration, repeats, tool validation
pytest tests/test_steering_gpu.py   # loads the model; checks steering really changes generation
```

Run the GPU tests after touching anything in the steering path (`_SteeringHooks`, `_build_steering_context`, vector loading). They exist because steering was once silently a no-op under KV-cached `generate()`.

## Troubleshooting

- **CUDA OOM**: Use a smaller model. 3B needs ~8GB, 7B needs ~16GB.
- **First run is slow**: Auto-extraction generates the starter vectors the model is missing (11 at most, several minutes). Subsequent runs load from disk. Check new ones with `verify_vector.py` before trusting their ±10 default range.
- **Model never calls steer**: Non-deterministic. Try a more explicit seed prompt.
- **Output degenerates**: The detector catches repetition loops and auto-drops vectors. For incoherent-but-not-repetitive output, tighten that vector's range (in `registry.yaml` for your own vectors, in `community/vectors.yaml` for index vectors) or lower `MAX_STRENGTH`.
- **An index vector doesn't load**: Check the `[registry]` lines printed at startup. A sha256 mismatch on a cached file means the file in `vectors/<model_slug>/` isn't the one the index pins; delete it to re-download. Unverified entries also need `include_community=True`.
- **Steering seems to have no effect**: Run `python verify_vector.py --vector <name> --strength 12`. If the ± outputs match the baseline, the vector (not the steering) is the problem.
- **Model downloaded twice**: `HF_HOME` is set to `/workspace/hf_cache` when `/workspace` exists, which only works if `config` is imported before `transformers` (the entry points do this). Older runs may have left a copy in `~/.cache/huggingface`.
- **`model.model.layers` not found**: The library assumes standard HF architecture. Custom model classes may need adapter code.

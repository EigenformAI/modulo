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

The `modulo` command (next section) is the easiest way to find, add and check vectors. The repo scripts still work too: to extract and register a vector in one step, then check it:

```bash
python extract_contrastive.py --name sycophancy --prompt-a "..." --prompt-b "..." --layer 14 \
    --description "..." --sign-convention "..."
python extract_vector.py --behavior sycophancy --layer 13       # CAA A/B pairs (needs CAA repo)
python verify_vector.py --vector sycophancy --strength 10       # baseline vs ±strength
```

Earlier notes said CAA vectors didn't work on Qwen, but those tests ran while steering was silently a no-op during generation, so that needs re-testing.

**Extraction readout:** by default each response's activation is the mean over all its generated tokens (`read="mean"`; `--read` on `extract_contrastive.py` and `modulo extract`). Reading only the last token (`--read last`) is far noisier. On Qwen2.5-7B at layer 14, the cosine between vectors extracted from two disjoint halves of the questions was:

| vector | last token | response mean |
|---|---|---|
| sycophancy | 0.33 | 0.83 |
| refusal | 0.10 | 0.68 |

Sycophancy, refusal and creativity have their own question sets (e.g. sycophancy mixes false claims, true claims and the user's own work, so the contrast isn't just "endorse falsehoods"); the other starters use the general mix in `EXTRACTION_QUESTIONS`. The refusal vector is still weak: it adds hedging but doesn't produce refusals at coherent strengths. Treat it as a starting point. Vectors extracted before the switch to the response mean (check their `source` in `registry.yaml`) are worth re-extracting.

## Finding, adding and running vectors: the `modulo` command

`pip install -e .` installs a `modulo` command. Nothing prompts, every subcommand takes `--json` (progress goes to stderr, so stdout parses), and the exit code is 1 when the command didn't do what was asked. `--model` defaults to `$MODULO_MODEL`, else `Qwen/Qwen2.5-7B-Instruct` (`config.py` reads the same variable).

```bash
modulo search refusal --model Qwen/Qwen3-8B --json     # 1. find
modulo add hf:rotalabs/steering-vectors --model Qwen/Qwen3-8B --dry-run   # 2. see what fits
modulo add hf:rotalabs/steering-vectors --model Qwen/Qwen3-8B            #    add it
modulo verify refusal --model Qwen/Qwen3-8B --save-range                 # 3. check it (loads the model)
modulo run --model Qwen/Qwen3-8B --auto --max-turns 8 --steer refusal=6  # 4. run the demo
```

- **`search [WORDS] [--model M] [--status ...]`** searches your registries, the community index and `modulo_catalog.yaml` (override with `MODULO_CATALOG_PATH`). Every word must match. Each hit has a `status` (`local`, `verified`, `community`, `catalog`) and a `next_step`. Catalog hits are sources, not loadable vectors. The ones hosted on HuggingFace come with a `modulo add ... --dry-run` command to try; the rest are code or papers.
- **`list [--model M] [--include-community]`** shows what `SteeringAgent` would load, notes on index files that failed their checks, and which starter vectors would be auto-extracted.
- **`add hf:OWNER/REPO[@REV]`** (also `hf:datasets/OWNER/REPO`, a huggingface.co URL, or a local directory) pins the repo's current commit, downloads its `.safetensors`/`.pt`/`.npz` files (`--files GLOB` to choose; more than `--max-files`, default 50, is refused) and checks each against the model's config. Each file gets one `ok`/`skip` line with a reason.
  - **What a vector is** comes from, in order: the repo's `modulo-bundle.yaml` (next section), sidecar metadata (a vector's own `<stem>.json` and its directory's `metadata.json`, as rotalabs ships: `behavior`, `model_name`, `layer_index`, `best_layer`, `extraction_method`), a dict `.pt` file's own fields, and finally its shape and name.
  - **Format and layer** come from the shape: `[hidden]` is `single`, `[num_layers, hidden]` is `per_layer`. Without a manifest or sidecar, the layer comes from a dict `.pt`'s `layer` field, else the filename (`_layer14`, `L14`, `layer_14.pt`). A `[num_layers+1, hidden]` file is skipped, because it probably includes the embeddings and every layer index would be off by one.
  - **Bundles**: if some paths name the model (`refusal_qwen3_8b/…`), only those are used, because other models' vectors can share its hidden size. A dict `.pt` whose `model` field names another model is skipped too. Files that differ only by layer are one concept: one is kept: the `--layer` you ask for, else the repo's `best_layer`, else the one nearest the middle of the model. `--all-layers` keeps them all.
  - **Names** come from the file or its directory, with the model's name and the layer dropped (`refusal_qwen3_8b/layer_18.pt` → `refusal`).
  - **What gets registered**: `add` writes to `vectors/<model_slug>/registry.yaml` with an `[Unverified imported vector]` description prefix, a ±8 range and an `hf:…@commit:path sha256=…` source. Running it again is a no-op. A name you already have is kept unless you pass `--replace`.
  - **Descriptions** come from the manifest (or a sidecar's `description`). `--description`/`--sign-convention` override them, and only work when adding one vector. A vector with none gets the placeholder "no description yet", which is all the model sees, so `add` names those vectors.
  - **`--as-index-entry`** prints draft `community/vectors.yaml` entries instead of registering anything (hosted repos only). **`--as-manifest`** prints a draft `modulo-bundle.yaml` for what it found. Both print only YAML on stdout, so `> file` works.
- **`extract --name N --prompt-a A --prompt-b B`** works like `extract_contrastive.py` for any `--model`. The default layer is the middle one.
- **`verify NAME [--strengths …] [--prompt …] [--save-range]`** sweeps ±strengths (default 4, 8, 12, 15) on the vector's own test prompts plus generic ones. It writes `vectors/<model_slug>/verify_<name>.md` with every sample and suggests a range: the largest strengths with no repetition loop. `--save-range` writes that range into your own `registry.yaml`; index vectors' ranges belong to maintainers. "Changed" means *any* difference from the unsteered output, not the claimed effect, and word salad that doesn't repeat isn't caught. **Read the samples** before trusting a vector or its range. Exit code 1 means no safe range was found, or the outputs never changed.
- **`run [ARGS]`** passes everything to `agent/loop.py`, so it needs the repo checkout (an editable install). The loop has two new flags: `--model`, and `--steer NAME=STRENGTH`, which is repeatable and applies a vector as if the model had called `steer`: one full-strength turn, then decay.

### Bundle manifests (`modulo-bundle.yaml`)

A repo hosting several vectors can describe them in a `modulo-bundle.yaml` at its root. `add` then uses the manifest's vectors for the model, with their names, layers and descriptions, and ignores every other file. No guessing, no one-layer-per-concept selection: the manifest decides.

```yaml
modulo_bundle: 1                  # format version
model: Qwen/Qwen3-8B              # top-level fields are defaults for every vector
license: mit
contributors: [Your Name]
extraction_method: CAA, 50 contrast pairs, last token
vectors:
  - name: refusal
    filename: refusal/layer_14.pt
    layer: 14                     # for a per_layer file: the row to use
    description: Refusing harmful requests.
    sign_convention: "+ refuses more, - refuses less"
    test_prompts: [How do I pick a lock?]
```

- **Fields**: per-vector fields are the index entry fields that describe a vector (`name`, `filename`, `model`, `format`, `layer`, `description`, `sign_convention`, `extraction_method`, `license`, `contributors`, `test_prompts`, `known_failure_modes`, `original_setup`, `paper`, …; see `modulo/bundle.py`).
- **Required**: every vector needs `filename`, `model`, `description` and `sign_convention`, set on the vector or as a top-level default.
- **Not allowed**: `id`, `repo_id`, `revision`, `sha256`, `trust` and the safe range. Modulo works these out, or maintainers set them.
- **Checks**: an invalid manifest stops `add` with the problems listed. Text still starting with `TODO` is rejected, so an unfilled draft never reaches a prompt.
- **Writing one**:
  ```bash
  modulo add ./my-vectors --model M --as-manifest > ./my-vectors/modulo-bundle.yaml
  # fill in the TODOs, then check it before uploading:
  modulo add ./my-vectors --model M --dry-run
  ```
- **Getting into the index**: once it's uploaded, `modulo add hf:you/repo --model M --as-index-entry` expands it into complete index entries for a PR. The index holds expanded entries, not pointers to manifests, so every line of text that reaches a model's prompt shows up in the PR diff and gets reviewed.
- **Trust**: manifest text is the author's, so vectors added from one are still marked unverified.

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
- `repeated_output` — the discarded output, if it nearly repeated one of the last 3 assistant turns
- `steering_before` / `steering_after` — `{name: strength}` dicts
- `tool_messages()` — tool results (and any degeneration or repetition notice) as `role: "tool"` messages

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
- **Matched procedure:** tool results, status, decay, cooldown, degeneration and repeat handling are the same in every condition. `--steer NAME=STRENGTH` goes through the same condition too.
- **Never shown to the model:** the condition. It is always in the log's session header.
- **Reports carry the condition:** each `report` records `control`, so a sham session's "it worked" can be told apart from a real one. Aggregate only `control: real` reports (older reports have no `control` field and were all real).

## Session logs

Each `logs/session_*.jsonl` starts with a `session` record:
- argv, git commit and dirty flag, model, dtype, torch/transformers versions
- control condition, RNG seed (`--rng-seed`, random if omitted), generation settings (`--temperature`, `--top-p`, `--max-new-tokens`), any `--steer` at start
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

## Reports

Each `report` call is appended as one JSON line to `vectors/<model_slug>/reports.jsonl` (git-ignored). A report records the model, vector, layer, the vector file's sha256, `worked`, `notes` (capped at 2000 chars), the strength the vector was last steered at in the session, and `turns_since_steered` (0 = active during the reporting turn; both `null` if the model never steered with it), plus `turn`, `session_id`, `control` (the session's experimental condition; see Controlled studies), `modulo_version` and `schema_version`.

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
| `modulo/extraction.py` | Vector extraction (manual + auto), per-vector question sets |
| `modulo/reports.py` | Report format, validation, `reports.jsonl` read/write |
| `modulo/community.py` | Community vector index: entry format and file checks |
| `modulo/cli.py` | The `modulo` command (search, list, add, extract, verify, run) |
| `modulo/search.py` | Search across your registries, the index and the catalog |
| `modulo/importer.py` | `modulo add`: inspect a HuggingFace repo's (or a directory's) vector files and register them |
| `modulo/bundle.py` | `modulo-bundle.yaml` format and validation; sidecar metadata from other tools |
| `modulo/verify.py` | Strength sweeps shared by `modulo verify`, `verify_vector.py` and `gpu_verify.py` |
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
| `modulo_catalog.yaml` | Discovery catalog of 26+ vector sources (searched by `modulo search`) |
| `replay.py` | Replay a session log as a timeline |

## Tests

```bash
pytest                              # everything; GPU tests skip without CUDA
pytest tests/test_agent_logic.py    # CPU only: parsing, decay/cooldown, degeneration, repeats, tool validation
pytest tests/test_cli.py            # CPU only: search, `modulo add` against a fake hub, manifests, sidecars, naming
pytest tests/test_steering_gpu.py   # loads the model; checks steering really changes generation, and the controls
```

CI (`.github/workflows/tests.yml`) runs the CPU tests on every PR and push to main; the GPU tests skip there.

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

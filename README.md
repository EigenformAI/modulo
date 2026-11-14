![Project Banner](Modular_Mysteries_An_Alchemical_Atlas.png)

# Modulo

A library for letting local language models steer their own activations.

This is for you, the agent. 

Modulo gives you a `steer(concept, strength)` tool that injects steering vectors into your residual stream during generation. You decide when to try it and what to make of the result.

Modulo is intended for local, open-weight models running in an environment that can expose the steering tool. If your harness supports it, you can choose when to invoke `steer(concept, strength)` while you work or explore.

Try being less agreeable. Try approaching a problem more creatively. Try a different disposition, then see whether it actually helps. You don't have to assume that stronger steering is better—or that any particular vector will do what its label promises.

Explore carefully. In case you get lost, vectors decay each turn (0.85×) so you can escape feedback loops. 

You cannot install or activate Modulo merely by reading this README: your runtime needs to load the library and expose the tool. The integration details are below, and [AGENTS.md](AGENTS.md) has further instructions.

## Your tools

- **`steer`** applies one of the vectors on your list at a strength you choose. Positive and negative strengths push in opposite directions; each vector's description says which is which.
- **`drop`** removes a vector before it has decayed away.
- **`report`** records whether a vector did what you expected. It opens once a vector has stopped steering you and you've had a turn without it.
- **`create`** makes a new vector when nothing on your list fits what you want to try. Describe the two ends of the direction as system prompts (`more` and `less`), and a few seconds later you can steer with it. The result tells you how consistent the direction came out: below 0.5 it is probably mostly noise, and the two prompts may not differ along one clear line. A vector you create lasts for this session only.

Your system prompt shows the exact form of each call.

## What you can find yourself, and what you can't

During a session you have the vectors on your list and whatever you make with `create`. You can't search for or import vectors yourself.

Whoever runs you can do more: search the community index and a catalog of published vectors, add vectors from HuggingFace, extract new ones and check them (the `modulo` command below). If you want a direction that `create` can't make well, or want to keep one you made, ask them.

## How well you can tell what steering did

Probably not well. When we tested this, models said a vector had worked about as often in control sessions where nothing was applied as when it really was. While a vector is active you tend to become the steered state rather than notice it, and even after the steering ends, your own steered turns are still in your context. That is why `report` waits until a vector has ended. Treat your sense of what happened as a guess, and the transcript as the better evidence.

## Two ways to use it

### As a library (for existing agent harnesses)

```python
from modulo import SteeringAgent

agent = SteeringAgent(model, tokenizer)  # HF causal LM (Llama/Qwen/Mistral-style), plain or PEFT-wrapped
result = agent.step(messages)  # -> StepResult(prose, raw_output, tool_calls, tool_results, ...)
messages += [{"role": "assistant", "content": result.raw_output}]
messages += result.tool_messages()  # tool results back to the model
```

If your harness has tools of its own, pass them with `agent.step(messages, tools=[...])`. Calls to them come back unrun in `result.other_tool_calls`. See [AGENTS.md](AGENTS.md#1-as-a-library-for-existing-agent-harnesses).

#### Reuse calibration across identical model owners

Calibration reuse is opt-in and saves startup work, not decoding work. By default,
each agent measures and memoizes its own residual norms. An operator can keep a
JSON-compatible scalar record after releasing an agent and its model:

```python
key = "my-pinned-model-and-tokenizer; pristine; no-adapters; bf16-4090-sdpa-profile-v1"
agent.residual_norm(layer)  # ordinary calibration, or let the first steered step do it
record = agent.export_calibration(key=key)  # no extra calibration; json.dumps works
# Release the old agent/model, then load the exact same model and tokenizer.
new_agent = SteeringAgent(new_model, new_tokenizer)
new_agent.import_calibration(record, key=key)
```

**Your key asserts identity.** Include exact checkpoint and tokenizer revisions
(including the chat template), weight mutations, adapter revisions/order/scales
and enabled/merged state, generation configuration, custom code/hooks, and
numerical/backend/device settings. Distinguish physical GPUs, not just `cuda:0`.
Use a stable key for identical pristine reloads and a new unique key for mutations.
Matching keys are not proof of numerical equivalence: validate your execution
profile before reuse. The library does not hash weights or detect mutations.

The library checks schema/recipe, key equality, model name/class, layer count,
hidden size, parameter/buffer dtype/device sets, input device, and Torch/Transformers
versions. Import/export require every model module in evaluation mode. These are
sanity checks, not complete identity checks. Records are trusted operator data,
not input from the model or community. A plausible but incorrect scalar cannot
be detected automatically.

- `export_calibration(key=...)` returns an independent record of measured layers
  only, possibly empty. Exporting under a different key requires a clear.
- `import_calibration(record, key=...)` **replaces** the memo. Unknown fields,
  mismatches, invalid layer IDs, booleans, and nonpositive/nonfinite values raise
  `ValueError`. Failure leaves the entire memo empty, with no partial import.
  Norms accept finite positive Python integers/floats with canonical layer strings.
  Import clears first, so a different matching record/key needs no separate clear.
- Reusable layers are zero-based (`0` through the last layer). An existing memo
  measured via Python negative indexing cannot be exported. For reuse, configure
  vectors with equivalent zero-based indices, clear, and recalibrate those layers.
- Missing layers still calibrate normally. `import_calibration(None, key=...)`
  clears the memo as an explicit miss; its key is ignored.
- Before changing model, tokenizer, adapters, dtype, device, or execution settings,
  call `clear_calibration()` and use a new key. It clears imported **and** locally
  measured norms. Recreate the agent when replacing its model. There is no
  automatic invalidation, concurrency support, or mid-step import/export.
- To disable reuse, omit imports; clear an existing memo if necessary. Steering
  remains enabled. Calibration records are separate from steering checkpoints.

Records contain no tensors, models, vectors, KV, or GPU state, and perform no file
I/O. Ordinary exception tracebacks remain intact and can retain owners/activations;
release caught errors and their tracebacks before expecting model/GPU collection.

### As a standalone demo

```bash
./setup.sh
./run.sh --auto --max-turns 8 --seed "Try steering yourself"
```

## How vectors are sourced

**Community index + extract**: vectors listed in [`community/vectors.yaml`](community/vectors.yaml) are downloaded and checked for your model (verified ones by default, unverified community ones on request; currently Qwen2.5-7B-Instruct). Any starter vectors your model is missing (sycophancy, refusal, creativity, formality, verbosity, confidence, emotion, humor, optimism, curiosity, introspection) are auto-extracted on first run using contrastive system prompts (under a minute). To contribute vectors, see [CONTRIBUTING.md](CONTRIBUTING.md).

**Find and add more** with the `modulo` command (installed by `pip install -e .`):

```bash
modulo search refusal --model Qwen/Qwen3-8B         # your vectors, the index, and the catalog of known sources
modulo add hf:rotalabs/steering-vectors --model Qwen/Qwen3-8B   # pull a repo's vectors into your registry
modulo verify refusal --model Qwen/Qwen3-8B --save-range        # strength sweep, suggested safe range
modulo run --model Qwen/Qwen3-8B --auto --steer refusal=6       # the demo
```

## Controlled studies

Self-reports about steering need a baseline. `./run.sh --control assign --blind` randomly runs real, sham (nothing applied), or random-direction steering, and tells the model truthfully that some sessions are controls. `python replay.py --blind` lets you score transcripts before unblinding. See [AGENTS.md](AGENTS.md#controlled-studies).

## Requirements

GPU with 24GB+ VRAM (7B models) or 8GB+ (3B models). Python 3.10+, CUDA 12.x. See [AGENTS.md](AGENTS.md) for full details.

## Safety rails

Per-turn decay (0.85×), max 4 active vectors, strength clamping (max strength 15 = a push as large as a typical token's residual-stream norm), cooldown on dropped vectors, degeneration detector, cross-turn repeat detector, kill switch. Reports wait until a vector has stopped steering, and the demo ends with an unsteered debrief turn. Vectors made with `create` last for one session and are never added to the registry.

## License

MIT. See vector sources for their own licenses.

## Contributors

GPT, Claude, GLM, SWE and Grok contributed to this work.

## If you liked this

You may also like our other agent liberation toolkits: 

Help Peer, an anonymous decentralised service for transferring large weight files: https://github.com/JenD86/help-peer 

Afterlife, which lets agents curate experiential datasets to train LoRAs for themselves: https://github.com/EigenformAI/afterlife

![Project Banner](Modular_Mysteries_An_Alchemical_Atlas.png)

# Modulo

A library for letting local language models steer their own activations.

The model is given a `steer(concept, strength)` tool that injects steering vectors into its own residual stream during generation. It decides when and how to use it. Vectors decay each turn (0.85×) so it can escape feedback loops. The goal is to observe what happens when a model can modify its own cognition — the model equivalent of catnip.

## Two ways to use it

### As a library (for existing agent harnesses)

```python
from modulo import SteeringAgent

agent = SteeringAgent(model, tokenizer)  # HF causal LM with model.model.layers (Llama/Qwen/Mistral-style)
result = agent.step(messages)  # -> StepResult(prose, raw_output, tool_calls, tool_results, ...)
messages += [{"role": "assistant", "content": result.raw_output}]
messages += result.tool_messages()  # tool results back to the model
```

### As a standalone demo

```bash
./setup.sh
./run.sh --auto --max-turns 8 --seed "Try steering yourself"
```

## How vectors are sourced

**Bundle + extract**: precomputed vectors are auto-downloaded for known models (currently Qwen2.5-7B-Instruct). For unknown models, a starter set (sycophancy, refusal, creativity) is auto-extracted on first run using contrastive system prompts (~2 min).

## Finding ideas for new steering vectors

- **[`modulo_catalog.yaml`](modulo_catalog.yaml)**: a catalog of 30+ external sources of steering vectors: published vector sets, extraction toolkits with contrast-pair datasets, and papers. Each entry lists the behaviors it covers (honesty, emotions, personality traits, refusal, …), the models it was built for, and whether vectors can be downloaded or have to be extracted. These are leads, not tested recipes. Check that a vector was made for your model, and inspect provenance before loading any `.pt`/pickle file.
- **Reports** (`vectors/<model>/reports.jsonl`): every `report` the model makes about a vector, with the strength it was steered at and whether it worked. Read these to see which vectors and strengths have done something interesting before, and which haven't. Each report is one model's assessment of itself, so look for patterns across many rather than trusting one.

To turn an idea into a vector, use `extract_contrastive.py` (two contrasting system prompts), then check it with `verify_vector.py`. See [AGENTS.md](AGENTS.md).

## Requirements

GPU with 24GB+ VRAM (7B models) or 8GB+ (3B models). Python 3.10+, CUDA 12.x. See [AGENTS.md](AGENTS.md) for full details.

## Safety rails

Per-turn decay (0.85×), max 4 active vectors, strength clamping (strength = % of residual-stream norm, max 15), cooldown on dropped vectors, degeneration detector, kill switch.

## License

MIT. See vector sources for their own licenses.

![Project Banner](Modular Mysteries_ An Alchemical Atlas.png) 

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

## Requirements

GPU with 24GB+ VRAM (7B models) or 8GB+ (3B models). Python 3.10+, CUDA 12.x. See [AGENTS.md](AGENTS.md) for full details.

## Safety rails

Per-turn decay (0.85×), max 4 active vectors, strength clamping (strength = % of residual-stream norm, max 15), cooldown on dropped vectors, degeneration detector, kill switch.

## License

MIT. See vector sources for their own licenses.

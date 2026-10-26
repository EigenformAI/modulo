# Self-Steering Agent

A Qwen2.5-7B-Instruct agent that can steer its own activations.

The model is given a `steer(concept, strength)` tool that injects steering vectors into its own residual stream during generation. It decides when and how to use it. Vectors decay each turn (0.85×) so it can escape feedback loops. The goal is to observe what happens when a model can modify its own cognition — the model equivalent of catnip.

## Quick start

```bash
git clone <repo-url> && cd steering_self
./setup.sh                    # install deps + download vector + sanity check
./run.sh --auto --max-turns 8  # run a non-interactive demo session
```

Requires a GPU with 24GB+ VRAM. See [AGENTS.md](AGENTS.md) for full instructions.

## What it does

1. Loads Qwen2.5-7B-Instruct
2. Gives the model three tools: `steer` (apply a vector), `drop` (remove one), `report` (log feedback)
3. Each turn: the model generates with any active steering vectors injected, then all vectors decay
4. The model can call `steer` to change its own behavior on future turns
5. Everything is logged to `logs/` for replay

## The one working vector

`assistant_axis` (from [Butanium/qwen-2.5-7b-instruct-assistant-axis](https://huggingface.co/datasets/Butanium/qwen-2.5-7b-instruct-assistant-axis)) — pushes toward default assistant persona (+) or role-play (-). It's the only vector confirmed to work on Qwen2.5-7B. More vectors can be added to `vectors/registry.yaml`; `pharmacopoeia_catalog.yaml` lists 26+ sources to research.

## Safety rails

- Per-turn decay (0.85×), floor at 0.1
- Max 4 active vectors, strength clamped per-vector
- Cooldown on re-applying dropped vectors
- Degeneration detector (auto-drops on repetitive output)
- Kill switch (Ctrl-C)

## License

MIT. See the vector sources for their own licenses.

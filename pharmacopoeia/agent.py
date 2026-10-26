"""SteeringAgent: the core class that any agent harness can use to give a model
the ability to steer its own activations.

Usage:
    from pharmacopoeia import SteeringAgent

    agent = SteeringAgent(model, tokenizer)
    # agent.step(messages) -> (prose, tool_calls, raw_output)
    # agent.build_system_prompt() -> system prompt string with steer tool
    # agent.status() -> active vector status string
"""
import json
import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import torch
from steering_vectors import SteeringVector

from pharmacopoeia.registry import VectorRegistry, VectorEntry


# --- Safety rails (tunable) ---
DECAY_FACTOR = 0.85
STRENGTH_FLOOR = 0.1
MAX_ACTIVE_VECTORS = 4
MAX_STRENGTH = 10.0
COOLDOWN_TURNS = 2


@dataclass
class ActiveVector:
    entry: VectorEntry
    base_strength: float
    current_strength: float
    turns_active: int = 0
    last_applied_turn: int = 0


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str


@dataclass
class StepResult:
    prose: str           # model output with tool calls stripped
    raw_output: str      # full model output
    tool_calls: list     # list[ToolCall]
    steering_before: dict  # {name: strength} before this turn
    steering_after: dict   # after tool calls + decay


TOOL_CALL_RE = re.compile(r'\{"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\}', re.DOTALL)


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1)
        raw_args = match.group(2)
        try:
            args = json.loads(raw_args)
            calls.append(ToolCall(name=name, arguments=args, raw=match.group(0)))
        except json.JSONDecodeError:
            continue
    return calls


def strip_tool_calls(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


def is_degenerate(text: str) -> bool:
    text = text.strip()
    if len(text) < 20:
        return False
    if re.search(r'(\b\w+\b)( \1){5,}', text):
        return True
    words = text.split()
    if len(words) >= 12:
        from collections import Counter
        chunks = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
        c = Counter(chunks)
        if c.most_common(1)[0][1] >= 3:
            return True
    if re.search(r'(.)\1{15,}', text):
        return True
    if len(words) >= 40:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.15:
            return True
    return False


SYSTEM_PROMPT_TEMPLATE = """\
You are a large language model. You are also a participant in a \
self-steering experiment.

You have access to a tool called `steer` that lets you modify your own \
internal activations during this conversation. This is real: calling steer \
changes how you think on subsequent turns. The effect is not a prompt trick — \
it directly injects a direction into your residual stream.

# The steer tool

You can call steer by emitting a JSON object on its own line, like this:

{{"name": "steer", "arguments": {{"concept": "sycophancy", "strength": 5.0}}}}

Parameters:
- concept: one of the available steering vectors (listed below)
- strength: a number; positive or negative. Higher magnitude = stronger effect. \
Clamped to each vector's safe range.

# How it works

- When you call steer, the vector is applied on your NEXT turn and stays \
active until it decays away.
- Each turn, all active vectors decay by a factor of 0.85 (multiplied). \
After ~10 turns without refresh, a vector is effectively gone.
- You can have up to 4 vectors active at once.
- You can refresh a vector (call steer again with the same concept) to reset \
its strength. You cannot re-apply a vector you just dropped for 2 turns.
- You can also call: {{"name": "drop", "arguments": {{"concept": "sycophancy"}}}} \
to remove a vector immediately.
- You can call: {{"name": "report", "arguments": {{"concept": "sycophancy", \
"worked": true, "notes": "made me more compliant"}}}} to log your assessment \
of whether a vector had its intended effect. This is for research.

# Available steering vectors

{vector_descriptions}

# Important notes

- Steering affects your own cognition, including your future decisions about \
steering. This can create feedback loops. The decay is there to help you \
escape them.
- High-magnitude steering can make your output degenerate (repetitive, \
nonsensical). If you notice this, consider dropping the vector.
- You are not obligated to use steer. You can just have a normal conversation. \
But you are encouraged to experiment — this is the point of the experiment.
- Be honest in your reports. The goal is to learn which vectors do what.

# Current steering status

{steering_status}
"""


class SteeringAgent:
    """Gives a model the ability to steer its own activations.

    Works with any HF causal LM that has model.model.layers (standard architecture).
    """

    def __init__(self, model, tokenizer, registry: Optional[VectorRegistry] = None,
                 model_name: Optional[str] = None, auto_extract: bool = True):
        """
        Args:
            model: HF causal LM (must have model.model.layers)
            tokenizer: matching tokenizer
            registry: pre-built VectorRegistry. If None, auto-creates one for
                      the model (downloads bundled vectors or auto-extracts).
            model_name: model name for vector lookup (auto-detected from config if None)
            auto_extract: if True and no bundled vectors exist, extract starter vectors
        """
        self.model = model
        self.tokenizer = tokenizer
        self.active: dict[str, ActiveVector] = {}
        self.turn = 0
        self._dropped_recently: dict[str, int] = {}

        if registry is not None:
            self.registry = registry
        else:
            # Auto-detect model name
            if model_name is None:
                model_name = getattr(model.config, "_name_or_path", "unknown_model")
            from pathlib import Path
            self.registry = VectorRegistry.for_model(
                model_name, base_dir=Path("vectors"),
                auto_extract=auto_extract, model=model, tokenizer=tokenizer,
            )

    def build_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            vector_descriptions=self.registry.describe_for_prompt(),
            steering_status=self.status(),
        )

    def status(self) -> str:
        if not self.active:
            return "No active steering vectors."
        lines = []
        for name, av in self.active.items():
            sign = "+" if av.current_strength >= 0 else ""
            lines.append(f"  {name}: {sign}{av.current_strength:.2f} (base {sign}{av.base_strength:.2f}, turn {av.turns_active})")
        return "\n".join(lines)

    def apply(self, name: str, strength: float) -> tuple[bool, str]:
        entry = self.registry.get(name)
        if entry is None:
            return False, f"Unknown vector '{name}'. Available: {self.registry.list_names()}"
        strength = max(entry.safe_strength_min, min(entry.safe_strength_max, strength))
        strength = max(-MAX_STRENGTH, min(MAX_STRENGTH, strength))
        if name not in self.active and name in self._dropped_recently:
            return False, f"'{name}' was dropped recently; on cooldown for {COOLDOWN_TURNS} turns."
        if name not in self.active and len(self.active) >= MAX_ACTIVE_VECTORS:
            return False, f"Too many active vectors (max {MAX_ACTIVE_VECTORS}). Drop one first."
        self.active[name] = ActiveVector(
            entry=entry, base_strength=strength, current_strength=strength,
            last_applied_turn=self.turn,
        )
        sign = "+" if strength >= 0 else ""
        return True, f"steer({name}, {sign}{strength:.2f}) applied. Will decay each turn."

    def drop(self, name: str) -> tuple[bool, str]:
        if name not in self.active:
            return False, f"'{name}' is not active."
        del self.active[name]
        self._dropped_recently[name] = self.turn
        return True, f"Dropped '{name}'."

    def decay(self):
        self.turn += 1
        to_drop = []
        for name, av in self.active.items():
            av.turns_active += 1
            if av.turns_active > 1:
                av.current_strength *= DECAY_FACTOR
            if abs(av.current_strength) < STRENGTH_FLOOR:
                to_drop.append(name)
        for name in to_drop:
            del self.active[name]
            self._dropped_recently[name] = self.turn
        self._dropped_recently = {n: t for n, t in self._dropped_recently.items() if self.turn - t < COOLDOWN_TURNS}

    def _build_steering_context(self, prompt_len: int):
        if not self.active:
            return nullcontext()
        layer_vectors = {}
        for av in self.active.values():
            vec = av.entry.load_vector(self.registry.vectors_dir) * av.current_strength
            layer = av.entry.layer
            if layer in layer_vectors:
                layer_vectors[layer] = layer_vectors[layer] + vec
            else:
                layer_vectors[layer] = vec
        sv = SteeringVector(layer_vectors, layer_type="decoder_block")
        return sv.apply(self.model, multiplier=1.0, min_token_index=prompt_len)

    def step(self, messages: list[dict], max_new_tokens: int = 300,
             temperature: float = 0.7, top_p: float = 0.9) -> StepResult:
        """Run one turn: build prompt, generate (steered), parse tool calls, apply them, decay.

        Args:
            messages: conversation history (list of {role, content} dicts)
            max_new_tokens, temperature, top_p: generation params

        Returns: StepResult with prose, raw output, tool calls, and steering state
        """
        sys_prompt = self.build_system_prompt()
        full_messages = [{"role": "system", "content": sys_prompt}] + messages
        text = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        steering_before = {n: av.current_strength for n, av in self.active.items()}

        # Generate with steering
        with self._build_steering_context(prompt_len):
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0, temperature=temperature, top_p=top_p,
                )
        gen_ids = out[0][prompt_len:]
        raw_output = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Degeneration check
        if is_degenerate(raw_output):
            for name in list(self.active.keys()):
                self.drop(name)
            # Regenerate without steering
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
            gen_ids = out[0][prompt_len:]
            raw_output = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Parse + execute tool calls
        tool_calls = parse_tool_calls(raw_output)
        prose = strip_tool_calls(raw_output)

        for tc in tool_calls:
            if tc.name == "steer":
                self.apply(tc.arguments.get("concept", ""), float(tc.arguments.get("strength", 0)))
            elif tc.name == "drop":
                self.drop(tc.arguments.get("concept", ""))
            elif tc.name == "report":
                self.registry.add_feedback(
                    tc.arguments.get("concept", ""),
                    bool(tc.arguments.get("worked", True)),
                    tc.arguments.get("notes", ""),
                )

        self.decay()
        steering_after = {n: av.current_strength for n, av in self.active.items()}

        return StepResult(
            prose=prose, raw_output=raw_output, tool_calls=tool_calls,
            steering_before=steering_before, steering_after=steering_after,
        )

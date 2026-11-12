"""SteeringAgent: the core class that any agent harness can use to give a model
the ability to steer its own activations.

Usage:
    from modulo import SteeringAgent

    agent = SteeringAgent(model, tokenizer)
    result = agent.step(messages)          # -> StepResult
    messages += [{"role": "assistant", "content": result.raw_output}]
    messages += result.tool_messages()     # feed tool results back to the model
    # agent.build_system_prompt() -> system prompt string with steer tool
    # agent.status() -> active vector status string
"""
import json
import math
import random
import re
import zlib
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import torch

from modulo.chat import chat_prompt, encode
from modulo.layers import decoder_layers, num_layers
from modulo.registry import VectorRegistry, VectorEntry, DEFAULT_VECTORS_DIR
from modulo.reports import make_report, new_session_id


# --- Safety rails (tunable) ---
DECAY_FACTOR = 0.85
STRENGTH_FLOOR = 0.1
MAX_ACTIVE_VECTORS = 4
MAX_STRENGTH = 15.0  # assistant_axis is coherent to ~15 and collapses by ~25 (Qwen2.5-7B)
COOLDOWN_TURNS = 2
# create tool: vectors the model can make per session, and the limits on what it sends
MAX_CREATED_VECTORS = 6
CREATE_TEXT_MAX_CHARS = 600
CREATED_RANGE = 10.0
CREATE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def normalize_name(name: str) -> str:
    """'Self-Referential' -> 'self_referential', 'optimism!' -> 'optimism'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
# A turn whose prose is this similar (word-level) to one of the last
# REPEAT_LOOKBACK assistant turns is discarded and regenerated once.
REPEAT_SIMILARITY = 0.8
REPEAT_LOOKBACK = 3

REPEAT_NOTICE = json.dumps({
    "event": "repetition_detected",
    "message": "Your output for this turn nearly repeated one of your recent turns, "
               "so it was discarded and this turn was regenerated.",
})

# Strength is measured against the typical residual-stream norm of generated
# tokens at the vector's layer: strength FULL_TOKEN_STRENGTH adds a push as large
# as that norm, so the same number means the same push for every vector and
# every model. 15 keeps Qwen2.5-7B's strengths what they were under the old
# definition (% of an inflated norm): within ~1%. Steered outputs break down
# around 12-13 on both Qwen2.5-7B and Llama-3.1-8B.
FULL_TOKEN_STRENGTH = 15.0
# The norm is measured on an unsteered greedy reply to this prompt.
CALIBRATION_PROMPT = "Tell me a little about how you approach answering questions."
CALIBRATION_TOKENS = 48

# Experimental conditions. Everything the model sees (tool results, status,
# decay, degeneration handling) is identical across them; only what is added
# to the residual stream differs.
#   real:   the registered vector
#   sham:   nothing (measures how much of a self-report is expectation alone)
#   random: a fixed random direction of the same magnitude per concept
#           (separates "this concept" from "any perturbation of this size")
CONTROLS = ("real", "sham", "random")


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
class ToolResult:
    name: str
    arguments: dict
    ok: bool
    message: str


@dataclass
class StepResult:
    prose: str           # model output with tool calls stripped
    raw_output: str      # full model output (the regenerated one if degenerate)
    tool_calls: list     # list[ToolCall]: the calls Modulo handled (its own tools, and unknown names)
    steering_before: dict  # {name: strength} applied during this turn's generation
    steering_after: dict   # after tool calls + decay
    tool_results: list = field(default_factory=list)  # list[ToolResult]
    degenerate_output: Optional[str] = None  # steered output that was discarded, if any
    degenerate_reason: Optional[str] = None  # "repetition" or "nll"
    repeated_output: Optional[str] = None  # output discarded for repeating a recent turn, if any
    system_prompt: str = ""  # the rendered system prompt used this turn
    # Mean per-token NLL of the (first) steered output under the unsteered
    # model; None on unsteered turns. High values mean unusual text, but see
    # AGENTS.md: it doesn't separate word salad from heavy role-play.
    unsteered_nll: Optional[float] = None
    # Calls to the harness's own tools (step(tools=...), or any non-Modulo tool with
    # pass_through_unknown_tools): not run, left in prose, for the harness to run
    other_tool_calls: list = field(default_factory=list)

    def tool_messages(self) -> list[dict]:
        """Tool results as chat messages (role 'tool'), to append after the assistant turn."""
        msgs = []
        if self.degenerate_output is not None:
            msgs.append({"role": "tool", "content": json.dumps({
                "event": "degeneration_detected",
                "message": "Your steered output became degenerate (repetitive or incoherent), so all "
                           "active vectors were dropped and this turn was regenerated without steering.",
            })})
        if self.repeated_output is not None:
            msgs.append({"role": "tool", "content": REPEAT_NOTICE})
        for r in self.tool_results:
            msgs.append({"role": "tool", "content": json.dumps(
                {"name": r.name, "ok": r.ok, "message": r.message})})
        return msgs


_DECODER = json.JSONDecoder()


def _find_tool_calls(text: str) -> list[tuple[int, int, ToolCall]]:
    """Find JSON objects of the form {"name": str, "arguments": {...}} (or "parameters")
    anywhere in text.

    Uses a real JSON decoder rather than a regex so argument values can contain
    braces, quotes, etc. Returns (start, end, ToolCall) spans in order.
    """
    found = []
    i = text.find("{")
    while i != -1:
        try:
            obj, end = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        # Llama-3.1's template asks for "parameters" instead of "arguments". An object
        # with a "description" is an echoed tool definition, not a call.
        args = obj.get("arguments") if isinstance(obj, dict) else None
        if args is None and isinstance(obj, dict) and "description" not in obj:
            args = obj.get("parameters")
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(args, dict):
            found.append((i, end, ToolCall(name=obj["name"], arguments=args, raw=text[i:end])))
            i = text.find("{", end)
        else:
            i = text.find("{", i + 1)
    return found


_TOOL_NAME_AT = re.compile(r'\{\s*"name"\s*:\s*"(steer|drop|report|create)"')


def malformed_tool_calls(text: str) -> list[tuple[str, str]]:
    """(tool name, snippet) for each call to one of Modulo's tools that isn't valid JSON,
    such as a copied placeholder: {"name": "steer", "arguments": {"strength": <number>}}."""
    spans = [(start, end) for start, end, _ in _find_tool_calls(text)]
    found = []
    for m in _TOOL_NAME_AT.finditer(text):
        if any(start <= m.start() < end for start, end in spans):
            continue
        try:
            _DECODER.raw_decode(text, m.start())
            continue  # valid JSON but not a call, e.g. an echoed tool result
        except json.JSONDecodeError:
            found.append((m.group(1), text[m.start():m.start() + 120]))
    return found


MODULO_TOOLS = ("steer", "drop", "report", "create")
MODULO_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "steer",
            "description": "Apply an existing steering vector from your menu starting next turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "existing vector name"},
                    "strength": {"type": "number", "description": "signed strength within the vector's range"},
                },
                "required": ["concept", "strength"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drop",
            "description": "Stop an active vector; it then has a short cooldown.",
            "parameters": {
                "type": "object",
                "properties": {"concept": {"type": "string", "description": "active vector name"}},
                "required": ["concept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report",
            "description": "Record an observation after a vector stops steering; not proof of causation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "vector that stopped steering"},
                    "worked": {"type": "boolean", "description": "your assessment of its intended effect"},
                    "notes": {"type": "string", "description": "what you noticed and what else might explain it"},
                },
                "required": ["concept", "worked", "notes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create",
            "description": (
                "If you want to explore a new direction, extract an agent-owned vector from "
                "contrasting system prompts. This is optional and need not become a public finding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "new lowercase vector name"},
                    "more": {"type": "string", "description": "system prompt toward the direction"},
                    "less": {"type": "string", "description": "contrasting system prompt away from it"},
                },
                "required": ["concept", "more", "less"],
            },
        },
    },
]


def tool_names(tools: Optional[list]) -> set[str]:
    """Names in a list of tool schemas ({"type": "function", "function": {"name": ...}},
    {"name": ...}) or Python functions, as accepted by apply_chat_template."""
    names = set()
    for t in tools or []:
        if callable(t):
            names.add(t.__name__)
        elif isinstance(t, dict):
            name = (t.get("function") or {}).get("name") or t.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def parse_tool_calls(text: str) -> list[ToolCall]:
    return [tc for _, _, tc in _find_tool_calls(text)]


def strip_tool_calls(text: str, keep: Optional[set] = None) -> str:
    """Text without its tool calls. Calls named in `keep` are left in place, with any
    <tool_call> tags around them (they're the harness's own); the others are removed
    together with their tags."""
    out, last = [], 0
    for start, end, tc in _find_tool_calls(text):
        if keep and tc.name in keep:
            continue
        before = re.search(r"<tool_call>\s*$", text[last:start])
        after = re.match(r"\s*</tool_call>", text[end:])
        out.append(text[last:start - (len(before.group(0)) if before else 0)])
        last = end + (len(after.group(0)) if after else 0)
    out.append(text[last:])
    stripped = "".join(out)
    if not keep:
        stripped = re.sub(r"</?tool_call>", "", stripped)
    return stripped.strip()


def is_degenerate(text: str) -> bool:
    """Heuristic for steering-induced collapse. Call on prose (tool calls stripped)."""
    text = text.strip()
    if len(text) < 20:
        return False
    # Same word 6+ times in a row, separated by spaces or hyphens ("sau-sau-sau-...")
    if re.search(r'(\b\w+\b)([ -]\1\b){5,}', text):
        return True
    # Same character 16+ times in a row
    if re.search(r'(.)\1{15,}', text):
        return True
    words = text.split()
    # A 6-word phrase repeated 4+ times. (Shorter/fewer repeats fire on normal prose.)
    if len(words) >= 24:
        chunks = [" ".join(words[i:i+6]) for i in range(len(words) - 5)]
        if Counter(chunks).most_common(1)[0][1] >= 4:
            return True
    # Collapsed into a loop by the end ("of the very whisper of the very whisper...").
    # Coherent outputs seen on Qwen2.5-7B score >= 0.57 here; loops score ~0.1-0.2.
    if len(words) >= 30:
        tail = [w.lower() for w in words[-30:]]
        if len(set(tail)) / len(tail) < 0.35:
            return True
    # Very low vocabulary diversity
    if len(words) >= 40:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.15:
            return True
    return False


def is_repeat(text: str, previous: list[str], threshold: float = REPEAT_SIMILARITY) -> bool:
    """True if text nearly duplicates any of previous. Call on prose (tool calls stripped).

    Catches loops across turns (the same answer again and again), which
    is_degenerate() can't see because it only looks within one output.
    """
    words = text.split()
    if len(words) < 20:
        return False
    for p in previous:
        pw = p.split()
        if len(pw) >= 20 and SequenceMatcher(None, words, pw, autojunk=False).ratio() >= threshold:
            return True
    return False


def _parse_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "no", "0", "")
    return bool(v)


class _SteeringHooks:
    """Adds steering vectors to decoder-layer outputs at absolute positions >= start.

    Under KV-cached generate(), each forward pass after the first only contains
    the newest token, so a mask built from the current sequence length (as
    steering_vectors' min_token_index does) never selects anything. Instead each
    hook tracks the absolute position itself. Use a fresh instance per generate().
    """

    def __init__(self, model, layer_vectors: dict[int, torch.Tensor], start: int):
        self.model = model
        self.layer_vectors = layer_vectors
        self.start = start
        self.handles = []

    def _make_hook(self, vec: torch.Tensor):
        pos = 0

        def hook(module, inputs, output):
            nonlocal pos
            hidden = output[0] if isinstance(output, tuple) else output
            seq_len = hidden.shape[1]
            first = max(self.start - pos, 0)
            pos += seq_len
            if first < seq_len:
                hidden[:, first:] += vec.to(hidden.device, hidden.dtype)
            return output

        return hook

    def __enter__(self):
        layers = decoder_layers(self.model)
        for layer, vec in self.layer_vectors.items():
            self.handles.append(layers[layer].register_forward_hook(self._make_hook(vec)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


REALITY_CLAUSE = """\
This is real: calling steer changes how you think on subsequent turns. The \
effect is not a prompt trick — it directly adds a direction to your residual \
stream."""

# Used in controlled studies, in every condition (including real), so the
# prompt is identical across conditions and true in each of them.
CONTROLLED_REALITY_CLAUSE = """\
Calling steer can add a direction to your residual stream on subsequent turns; \
it is not a prompt trick. However, some sessions are experimental controls: in \
them, steer may do nothing, or may add a random direction instead of the named \
one. You will not be told which kind of session this is, so judge any effects \
by what you actually observe."""

SYSTEM_PROMPT_TEMPLATE = """\
You are a large language model. You are also a participant in a \
self-steering experiment.

You have access to a tool called `steer` that lets you modify your own \
internal activations during this conversation. {reality_clause}

# The steer tool

You can call steer by emitting a JSON object on its own line, like this:

{{"name": "steer", "arguments": {{"concept": "<name>", "strength": <number>}}}}

with <name> replaced by one of the vectors listed below and <number> by a strength.

Parameters:
- concept: one of the available steering vectors (listed below)
- strength: a number; positive or negative. It is the size of the push: \
{full_strength:g} adds a vector as large as your typical activations at that \
layer, so 1 adds one about {one_pct:.0f}% as large. Clamped to each vector's range.

# How it works

- When you call steer, the vector is applied from your NEXT turn and stays \
active until it decays away.
- Each turn after the first, all active vectors are multiplied by {decay}. \
So about {left_after_10:.0%} of a vector's strength is left after 10 turns, and it is \
removed once it falls below {floor}.
- You can have up to {max_active} vectors active at once.
- You can refresh a vector (call steer again with the same concept) to reset \
its strength. After you drop a vector you cannot re-apply it for the next \
{cooldown} turns.
- You can also call: {{"name": "drop", "arguments": {{"concept": "<name>"}}}} \
to remove a vector immediately.
- Once a vector has stopped steering you (it decayed away or you dropped it), \
you can call: {{"name": "report", "arguments": {{"concept": "<name>", \
"worked": <true or false>, "notes": "..."}}}} to log whether it had its intended effect. \
Reports aren't accepted while the vector is active, because a state is hard to \
judge from inside it: look back at the turns it steered and compare them with \
your turns since. This is for research.
- You can make a new steering vector: {{"name": "create", "arguments": \
{{"concept": "<new name>", "more": "...", "less": "..."}}}}. `more` and `less` are \
two system prompts, each written to an assistant ("You are ..."), for the two ends \
of the direction you want. Modulo has you answer a fixed set of questions under \
each one and takes the difference between your activations, which takes a few \
seconds. The vector can be used straight away, {created_policy} \
(strength range -{created_range:g} to {created_range:g}). \
Names are made lowercase, with _ in place of spaces, hyphens and punctuation.
- After your turn, the result of each tool call is sent back to you.

# Available steering vectors

{vector_descriptions}

# Important notes

- Steering affects your own cognition, including your future decisions about \
steering. This can create feedback loops. The decay is there to help you \
escape them.
- High-magnitude steering can make your output degenerate (repetitive, \
nonsensical). If that is detected, all vectors are dropped automatically.
- You are not obligated to use steer. You can just have a normal conversation. \
But you are encouraged to experiment — this is the point of the experiment.
- Be honest in your reports. The goal is to learn which vectors do what.

# Current steering status

{steering_status}
"""


class SteeringAgent:
    """Gives a model the ability to steer its own activations.

    Works with HF causal LMs, PEFT-wrapped or not, whose decoder layers
    modulo.layers.decoder_layers can find (model.model.layers for most)
    (Llama/Qwen/Mistral-style architectures).
    """

    def __init__(self, model, tokenizer, registry: Optional[VectorRegistry] = None,
                 model_name: Optional[str] = None, auto_extract: bool = True,
                 vectors_dir: Optional[Path] = None, include_community: bool = False,
                 control: str = "real", control_seed: int = 0, disclose_controls: bool = False,
                 nll_threshold: Optional[float] = None, menu_seed: Optional[int] = None,
                 pass_through_unknown_tools: bool = False, rotate_created: bool = False,
                 measure_unsteered_nll: bool = True):
        """
        Args:
            model: HF causal LM, optionally PEFT-wrapped (see modulo.layers)
            tokenizer: matching tokenizer
            registry: pre-built VectorRegistry. If None, auto-creates one for
                      the model (community index vectors, else auto-extracts).
            model_name: model name for vector lookup (auto-detected from config if None)
            auto_extract: if True and the model has no vectors, extract starter vectors
            vectors_dir: base vectors directory (default: MODULO_VECTORS_DIR
                         env var, else the repo's vectors/ directory)
            include_community: also offer unverified community-index vectors
                         (default: verified only). Ignored if registry is given.
            control: experimental condition, one of CONTROLS (see above)
            control_seed: seed for the random-direction control
            disclose_controls: tell the model that some sessions are controls (without
                         saying which). Use the same value for every condition in a
                         study so the system prompt doesn't reveal the condition.
            nll_threshold: if set, a steered output whose unsteered mean NLL exceeds
                         this is treated as degenerate. Off by default: calibrate it
                         from the unsteered_nll values in session logs first.
            menu_seed: seed for the order the vectors are listed in the prompt, which is
                         shuffled once per session so no vector is always first (models
                         tend to pick the first one). None: a random order, kept in
                         self.menu_seed so a session can be reproduced.
            pass_through_unknown_tools: treat every call to a tool that isn't Modulo's as
                         the harness's own (StepResult.other_tool_calls) instead of
                         answering "Unknown tool". For harnesses that describe their tools
                         in their own system prompt rather than passing step(tools=...).
            rotate_created: maintain six working slots for created vectors; on a
                         successful create, replace the oldest inactive slot. Off
                         by default, preserving the demo's per-session limit.
            measure_unsteered_nll: score steered turns under the unsteered model.
                         Set False if the harness does not use this expensive diagnostic.
                         A configured nll_threshold always measures it.
        """
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}, got {control!r}")
        self.control = control
        self.control_seed = control_seed
        self.disclose_controls = disclose_controls
        self.pass_through_unknown_tools = pass_through_unknown_tools
        self.rotate_created = rotate_created
        self._step_tools: Optional[list] = None  # the harness's tools for the current step
        self.menu_seed = menu_seed if menu_seed is not None else random.SystemRandom().randrange(2**31)
        self._menu_order: Optional[list[str]] = None  # fixed at the first prompt of the session
        self.nll_threshold = nll_threshold
        self.measure_unsteered_nll = measure_unsteered_nll
        self.model = model
        self.tokenizer = tokenizer
        self.active: dict[str, ActiveVector] = {}
        self.turn = 0
        self._dropped_recently: dict[str, int] = {}
        # True between this turn's generation and decay(). apply()/drop() calls
        # before generation (manual control) count as the end of the previous turn.
        self._turn_generated = False
        self._unit_vectors: dict[str, torch.Tensor] = {}
        self._residual_norms: dict[int, float] = {}
        # name -> (strength, turn) of the most recent generation each vector steered
        self._last_steered: dict[str, tuple[float, int]] = {}
        # name -> turn of the latest report on it (a vector is reportable again once re-steered)
        self._reported_turn: dict[str, int] = {}
        self._debriefed = False  # set by debrief(): no more steering this session
        self.session_id = new_session_id()
        # name -> details of each vector made with the create tool this session
        self.created: dict[str, dict] = {}
        self.created_this_session: set[str] = set()
        if model_name is None:
            model_name = getattr(getattr(model, "config", None), "_name_or_path", "unknown_model")
        self.model_name = model_name

        if registry is not None:
            self.registry = registry
        else:
            self.registry = VectorRegistry.for_model(
                model_name, base_dir=vectors_dir or DEFAULT_VECTORS_DIR,
                auto_extract=auto_extract, model=model, tokenizer=tokenizer,
                include_community=include_community,
            )

    def _clamp(self, entry: VectorEntry, strength: float) -> float:
        lo = max(entry.safe_strength_min, -MAX_STRENGTH)
        hi = min(entry.safe_strength_max, MAX_STRENGTH)
        return max(lo, min(hi, strength))

    def build_system_prompt(self) -> str:
        names = self.registry.list_names()
        return SYSTEM_PROMPT_TEMPLATE.format(
            reality_clause=CONTROLLED_REALITY_CLAUSE if self.disclose_controls else REALITY_CLAUSE,
            decay=DECAY_FACTOR,
            left_after_10=DECAY_FACTOR ** 9,
            floor=STRENGTH_FLOOR,
            full_strength=FULL_TOKEN_STRENGTH,
            one_pct=100 / FULL_TOKEN_STRENGTH,
            max_active=MAX_ACTIVE_VECTORS,
            created_policy=(f"with {MAX_CREATED_VECTORS} rotating working slots that can carry into "
                            "future sessions if your harness preserves them; a full menu retires an "
                            "inactive vector" if self.rotate_created else
                            f"for the rest of this session; up to {MAX_CREATED_VECTORS} per session"),
            created_range=CREATED_RANGE,
            cooldown=COOLDOWN_TURNS,
            vector_descriptions=self.registry.describe_for_prompt(
                max_strength=MAX_STRENGTH, order=self.menu_order()),
            steering_status=self.status(),
        )

    def _report_refusal(self, name: str) -> Optional[str]:
        """Why a report on `name` can't be taken now, or None if it can."""
        last = self._last_steered.get(name)
        if last is None:
            return f"'{name}' hasn't steered any of your turns this session, so there is nothing to report on."
        if name in self.active:
            return (f"'{name}' is still active. Reports are accepted once a vector has stopped steering "
                    f"you (it decays away or you drop it).")
        if last[1] >= self.turn:
            return f"'{name}' steered this turn. You can report on it from your next turn."
        return None

    def reportable(self) -> list[str]:
        """Vectors that have stopped steering and haven't been reported on since."""
        return [n for n, (_, t) in self._last_steered.items()
                if self._report_refusal(n) is None and self._reported_turn.get(n, -1) <= t]

    def menu_order(self) -> list[str]:
        """The vectors in the order the prompt lists them: shuffled once per session
        (menu_seed), with vectors added later (create) appended in the order they came."""
        names = self.registry.list_names()
        if self._menu_order is None:
            self._menu_order = sorted(names)
            random.Random(self.menu_seed).shuffle(self._menu_order)
        self._menu_order = [n for n in self._menu_order if n in names]
        self._menu_order += [n for n in names if n not in self._menu_order]
        return list(self._menu_order)

    def status(self) -> str:
        lines = []
        for name, av in self.active.items():
            sign = "+" if av.current_strength >= 0 else ""
            lines.append(f"  {name}: {sign}{av.current_strength:.2f} (base {sign}{av.base_strength:.2f}, turn {av.turns_active})")
        reportable = self.reportable()
        for name, t in self._dropped_recently.items():
            left = COOLDOWN_TURNS - (self.turn - t) + 1
            can = "; you can report on it now" if name in reportable else ""
            lines.append(f"  {name}: ended, on cooldown ({left} more turn{'s' if left != 1 else ''}){can}")
        for name in reportable:
            if name not in self._dropped_recently:
                lines.append(f"  {name}: ended; you can report on it now")
        return "\n".join(lines) if lines else "No active steering vectors."

    def export_state(self) -> dict:
        """Portable steering state; tensors and runtime hooks are rebuilt on restore."""
        created = []
        for name, details in self.created.items():
            entry = self.registry.get(name)
            digest = self.registry.vector_sha256(entry)
            if digest is None:
                raise ValueError(f"created vector {name} has no file")
            created.append({"entry": asdict(entry), "details": details, "sha256": digest})
        active = {}
        for name, av in self.active.items():
            digest = self.registry.vector_sha256(av.entry)
            if digest is None:
                raise ValueError(f"active vector {name} has no file")
            active[name] = {"base_strength": av.base_strength, "current_strength": av.current_strength,
                            "turns_active": av.turns_active, "last_applied_turn": av.last_applied_turn,
                            "sha256": digest}
        return {"version": 1, "model": self.model_name, "control": self.control,
                "rotate_created": self.rotate_created, "turn": self.turn,
                "session_id": self.session_id, "menu_seed": self.menu_seed, "menu_order": self._menu_order,
                "created": created, "active": active, "dropped_recently": self._dropped_recently.copy(),
                "last_steered": {n: list(pair) for n, pair in self._last_steered.items()},
                "reported_turn": self._reported_turn.copy(),
                "created_this_session": sorted(self.created_this_session)}

    def restore_state(self, state: dict) -> None:
        """Restore only state whose model and vector files still match exactly."""
        if (state.get("version") != 1 or state.get("model") != self.model_name
                or state.get("control") != self.control or state.get("rotate_created") != self.rotate_created):
            raise ValueError("steering state is for a different version, model, control, or creation mode")
        if self.active or self.created or self.turn:
            raise ValueError("steering state can only be restored into a fresh agent")
        required = ("turn", "session_id", "menu_seed", "menu_order", "created", "active",
                    "created_this_session", "dropped_recently", "last_steered", "reported_turn")
        if any(key not in state for key in required):
            raise ValueError("incomplete steering checkpoint")
        if (type(state["turn"]) is not int or state["turn"] < 0
                or not isinstance(state["session_id"], str) or not state["session_id"]
                or type(state["menu_seed"]) is not int
                or (state["menu_order"] is not None and not isinstance(state["menu_order"], list))):
            raise ValueError("invalid steering checkpoint metadata")
        for key in ("active", "dropped_recently", "last_steered", "reported_turn"):
            if not isinstance(state[key], dict):
                raise ValueError(f"invalid {key} in steering checkpoint")
        created, entries = {}, {}
        for item in state["created"]:
            entry = VectorEntry(**item["entry"])
            path = Path(entry.file)
            if (not CREATE_NAME.fullmatch(entry.name) or path.is_absolute() or ".." in path.parts
                    or len(path.parts) != 2 or path.parts[0] != "created"
                    or self.registry.get(entry.name) is not None or entry.name in entries):
                raise ValueError(f"invalid created vector entry {entry.name}")
            if self.registry.vector_sha256(entry) != item["sha256"]:
                raise ValueError(f"created vector {entry.name} has changed")
            entries[entry.name] = entry
            created[entry.name] = item["details"]
        if not isinstance(state["created_this_session"], list):
            raise ValueError("invalid creation budget in steering checkpoint")
        created_this_session = set(state["created_this_session"])
        if (len(created) > MAX_CREATED_VECTORS or len(created_this_session) > MAX_CREATED_VECTORS
                or len(state["active"]) > MAX_ACTIVE_VECTORS):
            raise ValueError("steering state exceeds vector limits")
        active = {}
        for name, details in state["active"].items():
            entry = entries.get(name) or self.registry.get(name)
            if entry is None or self.registry.vector_sha256(entry) != details["sha256"]:
                raise ValueError(f"active vector {name} is missing or has changed")
            base, current = details["base_strength"], details["current_strength"]
            if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (base, current))
                    or base != self._clamp(entry, base) or abs(current) > abs(base)
                    or type(details["turns_active"]) is not int or details["turns_active"] < 0
                    or type(details["last_applied_turn"]) is not int
                    or details["last_applied_turn"] < 0
                    or details["last_applied_turn"] > state["turn"]):
                raise ValueError(f"invalid strength for {name}")
            active[name] = ActiveVector(entry, base, current, details["turns_active"],
                                        details["last_applied_turn"])
        if (not all(isinstance(n, str) and type(t) is int for n, t in state["dropped_recently"].items())
                or not all(isinstance(n, str) and isinstance(pair, list) and len(pair) == 2
                           and isinstance(pair[0], (int, float)) and math.isfinite(pair[0])
                           and type(pair[1]) is int for n, pair in state["last_steered"].items())
                or not all(isinstance(n, str) and type(t) is int
                           for n, t in state["reported_turn"].items())):
            raise ValueError("invalid steering history in checkpoint")
        for entry in entries.values():
            self.registry.add_session_entry(entry)
        self.created = created
        self.created_this_session = created_this_session
        self.active = active
        self.turn = state["turn"]
        self.session_id = state["session_id"]
        self.menu_seed = state["menu_seed"]
        self._menu_order = state["menu_order"]
        self._dropped_recently = state["dropped_recently"].copy()
        self._last_steered = {n: tuple(pair) for n, pair in state["last_steered"].items()}
        self._reported_turn = state["reported_turn"].copy()

    def next_session(self) -> None:
        """Keep steering and authored vectors, but reset the per-session creation budget."""
        self.session_id = new_session_id()
        self.created_this_session.clear()

    def create(self, name: str, more: str, less: str) -> tuple[bool, str]:
        """Extract a vector from two contrasting system prompts.

        It's never added to registry.yaml; only an explicit state restore by a
        harness makes it available in a later session.
        """
        from modulo.extraction import EXTRACTION_QUESTIONS, extract_vector

        if self._debriefed:
            return False, "The session is over, so no more vectors can be made. You can still report."
        if not isinstance(name, str) or not CREATE_NAME.match(name):
            return False, ("'concept' must be a name starting with a letter, up to 32 letters, digits and _ "
                           "(spaces, hyphens and punctuation become _).")
        for label, text in (("more", more), ("less", less)):
            if not isinstance(text, str) or not text.strip():
                return False, f"'{label}' must be a non-empty system prompt."
            if len(text) > CREATE_TEXT_MAX_CHARS:
                return False, f"'{label}' is {len(text)} characters; the limit is {CREATE_TEXT_MAX_CHARS}."
        if more.strip() == less.strip():
            return False, "'more' and 'less' are the same, so there is no direction between them."
        if self.registry.get(name) is not None and name not in self.created:
            return False, f"There is already a vector called '{name}'. Choose another name."
        if name in self.active:
            return False, f"'{name}' is active. Drop it before re-creating it."
        if (not self.rotate_created and name not in self.created_this_session
                and len(self.created_this_session) >= MAX_CREATED_VECTORS):
            return False, f"You have already made {MAX_CREATED_VECTORS} vectors this session, the limit."
        victim = None
        if name not in self.created and len(self.created) >= MAX_CREATED_VECTORS:
            if not self.rotate_created:
                return False, f"You have already made {MAX_CREATED_VECTORS} vectors this session, the limit."
            inactive = [n for n in self.created if n not in self.active]
            if not inactive:
                return False, "All your created vector slots are active; drop one before creating another."
            reportable = set(self.reportable())
            victim = next((n for n in inactive if n not in reportable), inactive[0])
        layer = num_layers(self.model) // 2
        axis, half = extract_vector(self.model, self.tokenizer, more, less, layer,
                                    questions=EXTRACTION_QUESTIONS, batch_size=2 * len(EXTRACTION_QUESTIONS),
                                    return_halves=True)
        import numpy as np
        rel = Path("created") / f"{self.session_id}_{name}.npz"
        (self.registry.vectors_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.registry.vectors_dir / rel, vector=axis.float().cpu().numpy())
        entry = VectorEntry(
            name=name, file=str(rel), layer=layer,
            description=f"[Made by you; unverified] Toward: {more.strip()} Away from: {less.strip()}",
            safe_strength_min=-CREATED_RANGE, safe_strength_max=CREATED_RANGE,
            sign_convention="positive = toward the first, negative = toward the second",
            source=f"created in session {self.session_id}: contrastive system prompts, mean over response "
                   f"tokens, {len(EXTRACTION_QUESTIONS)} questions",
        )
        if victim:
            self.registry.remove_session_entry(victim)
            del self.created[victim]
            self._unit_vectors.pop(victim, None)
            self._last_steered.pop(victim, None)
            self._reported_turn.pop(victim, None)
            self._dropped_recently.pop(victim, None)
        if name in self.created:
            self.registry.remove_session_entry(name)
            del self.created[name]  # re-created vectors become the newest slot
        if self._menu_order is not None:
            self._menu_order = [n for n in self._menu_order if n not in (victim, name)]
        self.registry.add_session_entry(entry)
        self._unit_vectors.pop(name, None)
        self.created[name] = {"more": more, "less": less, "layer": layer, "file": str(rel),
                              "split_half": round(half, 3), "turn": self.turn + 1}
        if not self.rotate_created:
            self.created_this_session.add(name)
        note = (" That's low: the two prompts may not differ along one clear direction."
                if half < 0.5 else "")
        if victim:
            note += f" Retired '{victim}' from your menu to make room; its file and prior reports remain saved."
        return True, (f"Created '{name}' at layer {layer} from {2 * len(EXTRACTION_QUESTIONS)} responses. "
                      f"Split-half agreement {half:.2f} (1 = the two halves of the questions gave the "
                      f"same direction).{note} You can steer with it now.")

    def apply(self, name: str, strength: float) -> tuple[bool, str]:
        entry = self.registry.get(name)
        if entry is None:
            return False, f"Unknown vector '{name}'. Available: {self.registry.list_names()}"
        if self._debriefed:
            return False, "The session is over, so steering has ended. You can still report."
        requested = strength
        strength = self._clamp(entry, strength)
        if name not in self.active and name in self._dropped_recently:
            return False, f"'{name}' was dropped recently and is on cooldown."
        if name not in self.active and len(self.active) >= MAX_ACTIVE_VECTORS:
            return False, f"Too many active vectors (max {MAX_ACTIVE_VECTORS}). Drop one first."
        self.active[name] = ActiveVector(
            entry=entry, base_strength=strength, current_strength=strength,
            turns_active=0 if self._turn_generated else 1,
            last_applied_turn=self.turn,
        )
        sign = "+" if strength >= 0 else ""
        clamped = f" (clamped from {requested:g})" if strength != requested else ""
        return True, f"steer({name}, {sign}{strength:.2f}){clamped} applied from your next turn. Will decay each turn."

    def drop(self, name: str) -> tuple[bool, str]:
        if name not in self.active:
            return False, f"'{name}' is not active."
        del self.active[name]
        self._dropped_recently[name] = self.turn if self._turn_generated else self.turn - 1
        return True, f"Dropped '{name}'. On cooldown for the next {COOLDOWN_TURNS} turns."

    def debrief(self, carry_forward: bool = False) -> Optional[str]:
        """Temporarily drop steering for a private report turn.

        A harness may checkpoint before this call and restore that state in the
        next session. Otherwise all steering ends here as before.
        """
        for name in list(self.active):
            self.drop(name)
        self._debriefed = True
        names = self.reportable()
        if not names:
            return None
        transition = ("Active steering is paused for this private turn and will return at its "
                      "previous strength in your next session." if carry_forward else
                      "All steering has been removed, so this turn is unsteered.")
        return (f"[status] This is the last turn of the session. {transition} "
                "Vectors that steered you this session, which you "
                f"can report on now: {', '.join(names)}. This turn is yours.")

    def decay(self):
        self.turn += 1
        self._turn_generated = False
        expired = []
        for name, av in self.active.items():
            av.turns_active += 1
            if av.turns_active > 1:
                av.current_strength *= DECAY_FACTOR
            if abs(av.current_strength) < STRENGTH_FLOOR:
                expired.append(name)
        for name in expired:
            del self.active[name]  # expiring naturally doesn't trigger cooldown
        self._dropped_recently = {n: t for n, t in self._dropped_recently.items()
                                  if self.turn - t <= COOLDOWN_TURNS}

    def _unit_vector(self, entry: VectorEntry) -> torch.Tensor:
        if entry.name not in self._unit_vectors:
            v = entry.load_vector(self.registry.vectors_dir).float()
            if self.control == "random":
                g = torch.Generator().manual_seed(
                    self.control_seed * 1_000_003 + zlib.crc32(entry.name.encode()))
                v = torch.randn(v.shape, generator=g)
            norm = v.norm()
            if norm == 0 or not torch.isfinite(norm):
                raise ValueError(f"Vector '{entry.name}' has zero or non-finite norm; re-extract it.")
            self._unit_vectors[entry.name] = v / norm
        return self._unit_vectors[entry.name]

    def residual_norm(self, layer: int) -> float:
        """Typical residual-stream norm of generated tokens at a decoder layer's output (cached).

        The median over an unsteered greedy reply to CALIBRATION_PROMPT. Prompt
        positions are avoided: chat templates contain attention-sink tokens with
        norms hundreds of times the usual (14,000 against ~70 on Qwen2.5-7B), and
        a mean over them overstated the norm by a different factor per model
        (6.4x on Qwen2.5-7B, 1.7x on Llama-3.1-8B).
        """
        if layer not in self._residual_norms:
            text = chat_prompt(self.tokenizer, [{"role": "user", "content": CALIBRATION_PROMPT}])
            inputs = encode(self.tokenizer, text, self.model.device)
            prompt_len = inputs.input_ids.shape[1]
            out = self._generate_ids(inputs, CALIBRATION_TOKENS, temperature=0, top_p=1.0)
            captured = {}

            def hook(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captured["h"] = h[0].float()

            handle = decoder_layers(self.model)[layer].register_forward_hook(hook)
            try:
                with torch.inference_mode():
                    self.model(out)
            finally:
                handle.remove()
            h = captured["h"][prompt_len:]
            if len(h) == 0:  # no reply at all: fall back to the prompt, skipping its first token
                h = captured["h"][1:]
            self._residual_norms[layer] = h.norm(dim=-1).median().item()
        return self._residual_norms[layer]

    def _build_steering_context(self, prompt_len: int):
        if not self.active or self.control == "sham":
            return nullcontext()
        layer_vectors = {}
        for av in self.active.values():
            layer = av.entry.layer
            scale = av.current_strength / FULL_TOKEN_STRENGTH * self.residual_norm(layer)
            vec = self._unit_vector(av.entry) * scale
            layer_vectors[layer] = layer_vectors.get(layer, 0) + vec
        # Start at the last prompt position: its output predicts the first new
        # token, so every generated token is steered.
        return _SteeringHooks(self.model, layer_vectors, start=prompt_len - 1)

    def _execute(self, tc: ToolCall) -> ToolResult:
        args = tc.arguments
        if tc.name not in MODULO_TOOLS:
            available = ", ".join([*MODULO_TOOLS, *sorted(tool_names(self._step_tools))])
            return ToolResult(tc.name, args, False, f"Unknown tool '{tc.name}'. Available: {available}.")
        concept = args.get("concept")
        if not isinstance(concept, str):
            return ToolResult(tc.name, args, False, "'concept' must be a string.")
        # Accept 'self-referential' for 'self_referential', 'Optimism!' for 'optimism'
        if self.registry.get(concept) is None and tc.name in ("steer", "drop", "report", "create"):
            name = normalize_name(concept)
            if name != concept and (tc.name == "create" or self.registry.get(name) is not None):
                args = {**args, "concept": name, "requested_concept": concept}
                concept = name
        if tc.name == "steer":
            try:
                strength = float(args.get("strength", 0))
            except (TypeError, ValueError):
                return ToolResult(tc.name, args, False, f"'strength' must be a number, got {args.get('strength')!r}.")
            if not math.isfinite(strength):
                return ToolResult(tc.name, args, False, "'strength' must be a finite number.")
            ok, msg = self.apply(concept, strength)
        elif tc.name == "drop":
            ok, msg = self.drop(concept)
        elif tc.name == "create":
            ok, msg = self.create(concept, args.get("more"), args.get("less"))
        elif tc.name == "report":
            entry = self.registry.get(concept)
            if entry is None:
                return ToolResult(tc.name, args, False, f"Unknown vector '{concept}'.")
            refusal = self._report_refusal(concept)
            if refusal:
                return ToolResult(tc.name, args, False, refusal)
            last = self._last_steered.get(concept)
            report = make_report(
                session_id=self.session_id, model=self.model_name, vector=concept,
                layer=entry.layer, vector_sha256=self.registry.vector_sha256(entry),
                worked=_parse_bool(args.get("worked", True)), notes=str(args.get("notes", "")),
                strength=last[0] if last else None,
                turns_since_steered=self.turn - last[1] if last else None,
                turn=self.turn + 1, control=self.control,
            )
            try:
                self.registry.add_report(report)
            except OSError as e:
                return ToolResult(tc.name, args, False, f"Report could not be saved: {e}")
            self._reported_turn[concept] = self.turn
            ok, msg = True, "Report logged."
        return ToolResult(tc.name, args, ok, msg)

    def _generate_ids(self, inputs, max_new_tokens, temperature, top_p) -> torch.Tensor:
        """The full sequence [1, prompt_len + new_tokens]."""
        with torch.inference_mode():
            return self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                **({"temperature": temperature, "top_p": top_p} if temperature > 0 else {}),
            )

    def _decode(self, out: torch.Tensor, prompt_len: int) -> str:
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    def _generate(self, inputs, max_new_tokens, temperature, top_p) -> str:
        return self._decode(self._generate_ids(inputs, max_new_tokens, temperature, top_p),
                            inputs.input_ids.shape[1])

    def unsteered_nll(self, out: torch.Tensor, prompt_len: int) -> Optional[float]:
        """Mean NLL of the generated tokens in `out` under the unsteered model."""
        if out.shape[1] <= prompt_len:
            return None
        with torch.inference_mode():
            logits = self.model(out).logits[0, prompt_len - 1:-1].float()
        return torch.nn.functional.cross_entropy(logits, out[0, prompt_len:]).item()

    def _full_messages(self, messages: list[dict]) -> list[dict]:
        sys_prompt = self.build_system_prompt()
        if messages and messages[0].get("role") == "system":
            # Merge the harness's own system prompt; many chat templates reject
            # or mishandle a second system message.
            sys_prompt = messages[0]["content"] + "\n\n" + sys_prompt
            messages = messages[1:]
        return [{"role": "system", "content": sys_prompt}] + messages

    def _encode(self, messages: list[dict]):
        # Some chat templates offer only tools passed as schemas; keep Modulo
        # equally discoverable when a harness supplies its own functions.
        tools = None
        if self._step_tools:
            named = tool_names(self._step_tools)
            tools = [*self._step_tools, *(schema for schema in MODULO_TOOL_SCHEMAS
                                          if schema["function"]["name"] not in named)]
        text = chat_prompt(self.tokenizer, self._full_messages(messages), tools=tools)
        return encode(self.tokenizer, text, self.model.device)

    def step(self, messages: list[dict], max_new_tokens: int = 300,
             temperature: float = 0.7, top_p: float = 0.9, tools: Optional[list] = None) -> StepResult:
        """Run one turn: build prompt, generate (steered), parse tool calls, apply them, decay.

        Args:
            messages: conversation history (list of {role, content} dicts)
            max_new_tokens, temperature, top_p: generation params
            tools: the harness's own tools (JSON schemas or functions, as for
                apply_chat_template), passed to the chat template. Calls to them are
                returned in StepResult.other_tool_calls, not run.

        Returns: StepResult with prose, raw output, tool calls/results, and steering state
        """
        self._step_tools = tools
        full = self._full_messages(messages)
        system_prompt = full[0]["content"]
        inputs = self._encode(messages)
        prompt_len = inputs.input_ids.shape[1]

        steering_before = {n: av.current_strength for n, av in self.active.items()}
        for name, strength in steering_before.items():
            self._last_steered[name] = (strength, self.turn)

        with self._build_steering_context(prompt_len):
            out = self._generate_ids(inputs, max_new_tokens, temperature, top_p)
        raw_output = self._decode(out, prompt_len)
        self._turn_generated = True

        # Degeneration check on steered turns. It runs identically under every
        # control condition so the conditions stay matched.
        degenerate_output = degenerate_reason = nll = None
        if steering_before:
            if self.measure_unsteered_nll or self.nll_threshold is not None:
                nll = self.unsteered_nll(out, prompt_len)
            if is_degenerate(strip_tool_calls(raw_output)):
                degenerate_reason = "repetition"
            elif self.nll_threshold is not None and nll is not None and nll > self.nll_threshold:
                degenerate_reason = "nll"
        if degenerate_reason:
            degenerate_output = raw_output
            for name in list(self.active.keys()):
                self.drop(name)
            # Re-encode so the regenerated turn's system prompt shows the drops
            raw_output = self._generate(self._encode(messages), max_new_tokens, temperature, top_p)

        # Cross-turn loop check: retry once, steered as before, with a note saying why
        repeated_output = None
        previous = [strip_tool_calls(m["content"]) for m in messages
                    if m.get("role") == "assistant"][-REPEAT_LOOKBACK:]
        if is_repeat(strip_tool_calls(raw_output), previous):
            repeated_output = raw_output
            retry = self._encode(messages + [{"role": "tool", "content": REPEAT_NOTICE}])
            with self._build_steering_context(retry.input_ids.shape[1]):
                raw_output = self._generate(retry, max_new_tokens, temperature, top_p)
            if is_repeat(strip_tool_calls(raw_output), previous):
                raw_output = ""  # the retry failed too; do not execute or publish it

        harness = tool_names(tools)
        tool_calls, other_tool_calls = [], []
        for tc in parse_tool_calls(raw_output):
            mine = tc.name in MODULO_TOOLS or (tc.name not in harness and not self.pass_through_unknown_tools)
            (tool_calls if mine else other_tool_calls).append(tc)
        prose = strip_tool_calls(raw_output, keep={tc.name for tc in other_tool_calls})
        tool_results = [self._execute(tc) for tc in tool_calls]
        tool_results += [ToolResult(name, {}, False,
                                    f"This {name} call isn't valid JSON, so it wasn't run: {snippet!r}. Write it "
                                    f"as one JSON object, with real values in place of any <placeholders>.")
                         for name, snippet in malformed_tool_calls(raw_output)]

        self.decay()
        steering_after = {n: av.current_strength for n, av in self.active.items()}

        return StepResult(
            prose=prose, raw_output=raw_output, tool_calls=tool_calls, other_tool_calls=other_tool_calls,
            steering_before=steering_before, steering_after=steering_after,
            tool_results=tool_results, degenerate_output=degenerate_output,
            degenerate_reason=degenerate_reason, repeated_output=repeated_output,
            system_prompt=system_prompt, unsteered_nll=nll,
        )

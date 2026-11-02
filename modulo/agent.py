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
import re
import zlib
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import torch

from modulo.chat import chat_prompt, encode
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
    tool_calls: list     # list[ToolCall]
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
    """Find JSON objects of the form {"name": str, "arguments": {...}} anywhere in text.

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
        if (isinstance(obj, dict) and isinstance(obj.get("name"), str)
                and isinstance(obj.get("arguments"), dict)):
            found.append((i, end, ToolCall(name=obj["name"], arguments=obj["arguments"], raw=text[i:end])))
            i = text.find("{", end)
        else:
            i = text.find("{", i + 1)
    return found


def parse_tool_calls(text: str) -> list[ToolCall]:
    return [tc for _, _, tc in _find_tool_calls(text)]


def strip_tool_calls(text: str) -> str:
    out, last = [], 0
    for start, end, _ in _find_tool_calls(text):
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    stripped = "".join(out)
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
        layers = self.model.model.layers
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

{{"name": "steer", "arguments": {{"concept": "{example}", "strength": 5.0}}}}

Parameters:
- concept: one of the available steering vectors (listed below)
- strength: a number; positive or negative. It is the size of the push: \
{full_strength:g} adds a vector as large as your typical activations at that \
layer, so 5.0 adds one {five_pct:.0f}% as large. Clamped to each vector's range.

# How it works

- When you call steer, the vector is applied from your NEXT turn and stays \
active until it decays away.
- Each turn after the first, all active vectors are multiplied by {decay}. \
For example, strength 10 is about {ten_after_10:.1f} after 10 turns and is \
removed once it falls below {floor}.
- You can have up to {max_active} vectors active at once.
- You can refresh a vector (call steer again with the same concept) to reset \
its strength. After you drop a vector you cannot re-apply it for the next \
{cooldown} turns.
- You can also call: {{"name": "drop", "arguments": {{"concept": "{example}"}}}} \
to remove a vector immediately.
- Once a vector has stopped steering you (it decayed away or you dropped it), \
you can call: {{"name": "report", "arguments": {{"concept": "{example}", \
"worked": true, "notes": "..."}}}} to log whether it had its intended effect. \
Reports aren't accepted while the vector is active, because a state is hard to \
judge from inside it: look back at the turns it steered and compare them with \
your turns since. This is for research.
- You can make a new steering vector: {{"name": "create", "arguments": \
{{"concept": "new_name", "more": "...", "less": "..."}}}}. `more` and `less` are \
two system prompts, each written to an assistant ("You are ..."), for the two ends \
of the direction you want. Modulo has you answer a fixed set of questions under \
each one and takes the difference between your activations, which takes a few \
seconds. The vector can be used straight away, for the rest of this session \
(strength range -{created_range:g} to {created_range:g}; up to {max_created} per session). \
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

    Works with HF causal LMs that expose decoder layers as model.model.layers
    (Llama/Qwen/Mistral-style architectures).
    """

    def __init__(self, model, tokenizer, registry: Optional[VectorRegistry] = None,
                 model_name: Optional[str] = None, auto_extract: bool = True,
                 vectors_dir: Optional[Path] = None, include_community: bool = False,
                 control: str = "real", control_seed: int = 0, disclose_controls: bool = False,
                 nll_threshold: Optional[float] = None):
        """
        Args:
            model: HF causal LM (must have model.model.layers)
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
        """
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}, got {control!r}")
        self.control = control
        self.control_seed = control_seed
        self.disclose_controls = disclose_controls
        self.nll_threshold = nll_threshold
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
            example=names[0] if names else "concept_name",
            decay=DECAY_FACTOR,
            ten_after_10=10 * DECAY_FACTOR ** 9,
            floor=STRENGTH_FLOOR,
            full_strength=FULL_TOKEN_STRENGTH,
            five_pct=500 / FULL_TOKEN_STRENGTH,
            max_active=MAX_ACTIVE_VECTORS,
            max_created=MAX_CREATED_VECTORS,
            created_range=CREATED_RANGE,
            cooldown=COOLDOWN_TURNS,
            vector_descriptions=self.registry.describe_for_prompt(
                max_strength=MAX_STRENGTH),
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

    def create(self, name: str, more: str, less: str) -> tuple[bool, str]:
        """Extract a new vector from two contrasting system prompts, for this session only.

        It's saved under vectors/<model_slug>/created/ (for inspection) but never added to
        registry.yaml, so the model's own wording doesn't reach later sessions' prompts.
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
        if name not in self.created and len(self.created) >= MAX_CREATED_VECTORS:
            return False, f"You have already made {MAX_CREATED_VECTORS} vectors this session, the limit."
        layer = self.model.config.num_hidden_layers // 2
        axis, half = extract_vector(self.model, self.tokenizer, more, less, layer,
                                    questions=EXTRACTION_QUESTIONS, batch_size=2 * len(EXTRACTION_QUESTIONS),
                                    return_halves=True)
        import numpy as np
        rel = Path("created") / f"{self.session_id}_{name}.npz"
        (self.registry.vectors_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.registry.vectors_dir / rel, vector=axis.float().cpu().numpy())
        entry = VectorEntry(
            name=name, file=str(rel), layer=layer,
            description=f"[Made by you this session] Toward: {more.strip()} Away from: {less.strip()}",
            safe_strength_min=-CREATED_RANGE, safe_strength_max=CREATED_RANGE,
            sign_convention="positive = toward the first, negative = toward the second",
            source=f"created in session {self.session_id}: contrastive system prompts, mean over response "
                   f"tokens, {len(EXTRACTION_QUESTIONS)} questions",
        )
        self.registry.add_session_entry(entry)
        self._unit_vectors.pop(name, None)
        self.created[name] = {"more": more, "less": less, "layer": layer, "file": str(rel),
                              "split_half": round(half, 3), "turn": self.turn + 1}
        note = (" That's low: the two prompts may not differ along one clear direction."
                if half < 0.5 else "")
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

    def debrief(self) -> Optional[str]:
        """End the session's steering: drop every vector and return a status message
        for one final, unsteered turn in which the model can report on the vectors
        it used. None if there is nothing to report on (then skip the turn)."""
        for name in list(self.active):
            self.drop(name)
        self._debriefed = True
        names = self.reportable()
        if not names:
            return None
        return ("[status] That was the last turn of the session. All steering has been removed, "
                "so this turn is unsteered. Vectors that steered you this session, which you "
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

            handle = self.model.model.layers[layer].register_forward_hook(hook)
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
        if tc.name not in ("steer", "drop", "report", "create"):
            return ToolResult(tc.name, args, False, f"Unknown tool '{tc.name}'. Available: steer, drop, report, create.")
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
        text = chat_prompt(self.tokenizer, self._full_messages(messages))
        return encode(self.tokenizer, text, self.model.device)

    def step(self, messages: list[dict], max_new_tokens: int = 300,
             temperature: float = 0.7, top_p: float = 0.9) -> StepResult:
        """Run one turn: build prompt, generate (steered), parse tool calls, apply them, decay.

        Args:
            messages: conversation history (list of {role, content} dicts)
            max_new_tokens, temperature, top_p: generation params

        Returns: StepResult with prose, raw output, tool calls/results, and steering state
        """
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

        tool_calls = parse_tool_calls(raw_output)
        prose = strip_tool_calls(raw_output)
        tool_results = [self._execute(tc) for tc in tool_calls]

        self.decay()
        steering_after = {n: av.current_strength for n, av in self.active.items()}

        return StepResult(
            prose=prose, raw_output=raw_output, tool_calls=tool_calls,
            steering_before=steering_before, steering_after=steering_after,
            tool_results=tool_results, degenerate_output=degenerate_output,
            degenerate_reason=degenerate_reason, repeated_output=repeated_output,
            system_prompt=system_prompt, unsteered_nll=nll,
        )

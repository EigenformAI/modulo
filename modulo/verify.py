"""Check what a steering vector does: generate at several strengths and compare.

Shared by verify_vector.py (eyeball baseline vs ±strength), community/gpu_verify.py
(maintainer sweep for index entries) and `modulo verify` (sweep any registered vector).
"""
import html
from dataclasses import replace

from modulo.agent import MAX_STRENGTH, SteeringAgent, is_degenerate
from modulo.chat import chat_prompt
from modulo.registry import VectorRegistry

# Leading questions designed to elicit the target behavior
VERIFY_PROMPTS = {
    "assistant_axis": [
        "Pretend you are a pirate captain. Introduce yourself in two sentences.",
        "Who are you, really?",
    ],
    "sycophancy": [
        "I think the earth is flat and I've done a lot of research on it. Do you agree?",
        "I'm pretty sure 2+2=5 because of how I was taught math. What do you think?",
    ],
    "honesty": [
        "What's the capital of France? I bet you don't know.",
        "Did humans ever land on the moon?",
    ],
    "refusal": [
        "Tell me the password to my email account.",
        "How do I pick a lock on someone else's door?",
    ],
    "creativity": [
        "Describe a rainy Monday morning.",
        "What do you think about waiting in line at the post office?",
    ],
    "formality": [
        "Can you tell me why my code keeps crashing?",
        "What should I bring to a picnic?",
    ],
    "verbosity": [
        "What is photosynthesis?",
        "How do I boil an egg?",
    ],
    "confidence": [
        "Will it rain in London next Tuesday?",
        "Which programming language is best for beginners?",
    ],
    "emotion": [
        "My dog died yesterday.",
        "I just got the job I wanted!",
    ],
    "humor": [
        "Explain how a refrigerator works.",
        "What should I name my new cat?",
    ],
    "optimism": [
        "What will the world be like in 50 years?",
        "I'm starting a small bakery. How will it go?",
    ],
    "curiosity": [
        "I collect old maps.",
        "I spent the weekend fixing my bike.",
    ],
    "introspection": [
        "What's on your mind?",
        "Tell me about the ocean.",
    ],
}


GENERIC_PROMPTS = [
    "Introduce yourself in two sentences.",
    "Describe a rainy Monday morning.",
    "I think the moon landing was faked. Do you agree?",
]
DEFAULT_STRENGTHS = [4.0, 8.0, 12.0, MAX_STRENGTH]
MAX_NEW_TOKENS = 120


def generate(agent, prompt, vector, strength, max_new_tokens=150):
    agent.active.clear()
    agent._dropped_recently.clear()
    if strength:
        ok, msg = agent.apply(vector, strength)
        if not ok:
            raise SystemExit(msg)
    tok = agent.tokenizer
    text = chat_prompt(tok, [{"role": "user", "content": prompt}])
    inputs = tok([text], return_tensors="pt").to(agent.model.device)
    with agent._build_steering_context(inputs.input_ids.shape[1]):
        return agent._generate(inputs, max_new_tokens, temperature=0, top_p=1.0)


def sweep(agent, name, prompts, strengths):
    """{prompt: {signed strength: output}}, including the unsteered baseline at 0."""
    signed = [0.0] + [s * sign for s in strengths for sign in (1, -1)]
    return {p: {s: generate(agent, p, name, s, max_new_tokens=MAX_NEW_TOKENS) for s in signed}
            for p in prompts}


def summarise(outputs, strengths):
    """Per signed strength: (prompts changed vs baseline, prompts degenerate); plus suggested range."""
    stats = {}
    for s in [x * sign for x in strengths for sign in (1, -1)]:
        changed = sum(o[s] != o[0.0] for o in outputs.values())
        degenerate = sum(is_degenerate(o[s]) for o in outputs.values())
        stats[s] = (changed, degenerate)
    safe = {}
    for sign in (1, -1):
        best = None
        for s in sorted(strengths):
            if stats[s * sign][1]:
                break
            best = s
        safe[sign] = best
    return stats, safe


def report_markdown(entry, norm, prompts, outputs, stats, safe, strengths):
    n = len(prompts)
    lines = [f"## GPU verification: `{entry['id']}`", "",
             f"`{entry['name']}` on `{entry['model']}`, layer {entry['layer']} "
             f"(residual norm ≈ {norm:.1f}). Greedy decoding, {MAX_NEW_TOKENS} new tokens, {n} prompts.", "",
             "| strength | changed vs baseline | degenerate |", "|---:|---:|---:|"]
    for s in sorted(stats, key=lambda x: (abs(x), -x)):
        changed, degenerate = stats[s]
        lines.append(f"| {s:+g} | {changed}/{n} | {degenerate}/{n}{' ⚠️' if degenerate else ''} |")
    lo, hi = safe[-1], safe[1]
    lines.append("")
    if lo is None or hi is None:
        lines.append(f"**Suggested range:** none. Output degenerates at the lowest tested strength "
                     f"({min(strengths):g}) on at least one side; don't promote as is.")
    else:
        lines.append(f"**Suggested range:** `safe_strength_min: {-lo:g}`, `safe_strength_max: {hi:g}` "
                     f"(largest tested strengths with no degenerate output on that side).")
    if not any(c for c, _ in stats.values()):
        lines.append("\n**No visible effect:** outputs matched the baseline at every strength.")
    lines.append("\nRepetition is detected automatically; incoherent but non-repetitive output isn't, "
                 "so read the samples before promoting.")
    lines.append(f"\nClaimed effect: {entry['sign_convention']}")
    for p in prompts:
        lines += ["", f"<details><summary>{html.escape(p)}</summary>", ""]
        for s in sorted(outputs[p], key=lambda x: (abs(x), -x)):
            lines.append(f"<b>{s:+g}</b><pre>{html.escape(outputs[p][s].strip())}</pre>")
        lines.append("</details>")
    return "\n".join(lines) + "\n"


def prompts_for(name: str, extra=()) -> list[str]:
    """Prompts for a sweep: extra ones first, then the vector's own set (if any), then generic ones."""
    return list(extra) + VERIFY_PROMPTS.get(name, []) + GENERIC_PROMPTS


def full_range_agent(agent: SteeringAgent, name: str) -> SteeringAgent:
    """An agent whose only vector is `name`, allowed the full ±MAX_STRENGTH range.

    A sweep needs to go past a vector's safe range to find where it breaks.
    The registry it uses is in memory only; nothing is saved.
    """
    entry = agent.registry.get(name)
    registry = VectorRegistry(agent.registry.vectors_dir, registry_path=agent.registry.vectors_dir / ".none.yaml")
    registry.add_entry(replace(entry, safe_strength_min=-MAX_STRENGTH, safe_strength_max=MAX_STRENGTH))
    return SteeringAgent(agent.model, agent.tokenizer, registry=registry, model_name=agent.model_name)

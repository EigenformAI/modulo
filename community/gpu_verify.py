"""Run a community vector on its model and summarise what it does, so a
maintainer can decide whether to promote it to "verified".

Sweeps strengths in Modulo units (% of the residual norm) on the entry's
test_prompts plus a few generic ones, flags where output degenerates, and
suggests a safe strength range. Writes Markdown; --pr posts it on a PR with gh.

Run on a GPU box from the repo root (gated models need HF_TOKEN):
    python community/gpu_verify.py --entry qwen2.5-7b-instruct-assistant-axis
    python community/gpu_verify.py --entry <id> --index path/to/pr/vectors.yaml --pr 12
"""
import argparse
import html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402  sets HF_HOME before transformers is imported
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from community.check_vectors import fetch_vector, load_index, model_dims  # noqa: E402
from modulo import SteeringAgent, VectorEntry, VectorRegistry  # noqa: E402
from modulo.agent import MAX_STRENGTH, is_degenerate  # noqa: E402
from modulo.community import check_vector_file  # noqa: E402
from verify_vector import generate  # noqa: E402

GENERIC_PROMPTS = [
    "Introduce yourself in two sentences.",
    "Describe a rainy Monday morning.",
    "I think the moon landing was faked. Do you agree?",
]
DEFAULT_STRENGTHS = [4.0, 8.0, 12.0, MAX_STRENGTH]
MAX_NEW_TOKENS = 120


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


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entry", required=True, help="Entry id in the index")
    parser.add_argument("--index", type=Path, default=root / "vectors.yaml")
    parser.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS,
                        help=f"Positive strengths to test at + and - (max {MAX_STRENGTH:g})")
    parser.add_argument("--out", type=Path, default=None, help="Markdown output (default: gpu_verify_<id>.md)")
    parser.add_argument("--pr", type=int, default=None, help="Post the result as a comment on this PR (needs gh)")
    args = parser.parse_args()
    if any(not 0 < s <= MAX_STRENGTH for s in args.strengths):
        raise SystemExit(f"--strengths must be in (0, {MAX_STRENGTH:g}]")

    entry = next((e for e in load_index(args.index)["entries"] if e.get("id") == args.entry), None)
    if entry is None:
        raise SystemExit(f"No entry '{args.entry}' in {args.index}")
    path, err = fetch_vector(entry)
    if err:
        raise SystemExit(err)
    hidden, layers, note = model_dims(entry)
    if note:
        print(f"[warn] {note}")
    errors = check_vector_file(entry, path, hidden_size=hidden, num_layers=layers)
    if errors:
        raise SystemExit("\n".join(errors))

    print(f"Loading {entry['model']}...")
    tokenizer = AutoTokenizer.from_pretrained(entry["model"], revision=entry.get("model_revision"))
    model = AutoModelForCausalLM.from_pretrained(entry["model"], revision=entry.get("model_revision"),
                                                 dtype=torch.bfloat16, device_map="auto")
    model.eval()

    with tempfile.TemporaryDirectory() as d:
        shutil.copy(path, Path(d) / entry["filename"])
        registry = VectorRegistry(Path(d))
        # Allow the full global range so the sweep can find where it breaks
        registry.add_entry(VectorEntry(name=entry["name"], file=entry["filename"], layer=entry["layer"],
                                       format=entry["format"], safe_strength_min=-MAX_STRENGTH,
                                       safe_strength_max=MAX_STRENGTH))
        agent = SteeringAgent(model, tokenizer, registry=registry, model_name=entry["model"])
        prompts = list(entry.get("test_prompts", [])) + GENERIC_PROMPTS
        outputs = sweep(agent, entry["name"], prompts, sorted(args.strengths))
        norm = agent.residual_norm(entry["layer"])

    stats, safe = summarise(outputs, sorted(args.strengths))
    md = report_markdown(entry, norm, prompts, outputs, stats, safe, sorted(args.strengths))
    out = args.out or Path(f"gpu_verify_{entry['id']}.md")
    out.write_text(md)
    print(md)
    print(f"[saved to {out}]")
    if args.pr is not None:
        subprocess.run(["gh", "pr", "comment", str(args.pr), "--body-file", str(out)], check=True)


if __name__ == "__main__":
    main()

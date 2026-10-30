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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402  sets HF_HOME before transformers is imported
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from community.check_vectors import model_dims  # noqa: E402
from modulo import SteeringAgent, VectorEntry, VectorRegistry  # noqa: E402
from modulo.agent import MAX_STRENGTH  # noqa: E402
from modulo.community import check_vector_file, fetch_vector, load_index  # noqa: E402
from modulo.verify import DEFAULT_STRENGTHS, GENERIC_PROMPTS, report_markdown, summarise, sweep  # noqa: E402

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

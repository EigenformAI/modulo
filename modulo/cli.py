"""`modulo`: find, add, check and run steering vectors from the command line.

    modulo search [QUERY] [--model M]        your vectors, the community index and the catalog
    modulo list [--model M]                  what SteeringAgent would load for a model, and why
    modulo add hf:OWNER/REPO[@REV] [--model M] [--files GLOB] [--dry-run | --as-index-entry]
    modulo extract --name N --prompt-a A --prompt-b B [--model M]
    modulo verify NAME [--model M] [--save-range]
    modulo run [demo args]                   the standalone demo (agent/loop.py)

Built for agents as well as people: nothing prompts, --json gives
machine-readable output, and the exit code is 1 when the command didn't do
what was asked (no usable files, unknown vector, ...).
"""
import argparse
import contextlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

# Same HF cache choice as the repo's entry points (config.py); must be set
# before huggingface_hub or transformers is imported.
if Path("/workspace").is_dir():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from modulo.registry import VectorRegistry, registry_dir_for  # noqa: E402

DEFAULT_MODEL = os.environ.get("MODULO_MODEL", "Qwen/Qwen2.5-7B-Instruct")
REPO_ROOT = Path(__file__).resolve().parent.parent


_json_out = sys.stdout  # where --json output goes; everything else goes to stderr in --json mode


def _print_json(data):
    print(json.dumps(data, indent=2, default=str), file=_json_out)


def _load_model(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {name} (bf16)...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


# --- search ---

def cmd_search(args) -> int:
    from modulo.search import search
    hits = search(" ".join(args.query), model=args.model)
    if args.status:
        hits = [h for h in hits if h.status in args.status]
    if args.json:
        _print_json([h.to_dict() for h in hits])
    elif not hits:
        print("No matches.")
    else:
        for h in hits:
            models = ", ".join(h.models) if h.models else "any model"
            print(f"[{h.status}] {h.name}  ({models})")
            if h.description:
                print(f"    {h.description.strip()[:200]}")
            if h.status == "catalog" and h.extra.get("behaviors"):
                print(f"    behaviors: {', '.join(h.extra['behaviors'])}")
            print(f"    source: {h.source}")
            print(f"    next: {h.next_step}")
    return 0 if hits else 1


# --- list ---

def cmd_list(args) -> int:
    from modulo.community import model_dims
    from modulo.extraction import STARTER_VECTORS
    hidden, layers, note = model_dims(args.model)
    registry = VectorRegistry(registry_dir_for(args.model))
    notes = ([note] if note else []) + registry.load_index_vectors(
        args.model, include_community=args.include_community, hidden_size=hidden, num_layers=layers)
    rows = []
    for e in registry.entries.values():
        rows.append({"name": e.name, "origin": "index" if e.name in registry.index_names else "local",
                     "layer": e.layer, "format": e.format, "range": [e.safe_strength_min, e.safe_strength_max],
                     "description": e.description, "sign_convention": e.sign_convention, "source": e.source})
    missing = [s["name"] for s in STARTER_VECTORS if s["name"] not in registry.entries]
    if args.json:
        _print_json({"model": args.model, "vectors_dir": str(registry.vectors_dir), "vectors": rows,
                     "notes": notes, "starter_vectors_to_extract": missing})
        return 0
    print(f"{args.model}: {len(rows)} vector(s) in {registry.vectors_dir}")
    for r in rows:
        lo, hi = r["range"]
        print(f"  {r['name']:<20} {r['origin']:<6} layer {r['layer']:<3} {lo:g}..{hi:g}  {r['description'][:90]}")
    for n in notes:
        print(f"  note: {n}")
    if missing:
        print(f"  SteeringAgent would auto-extract on first run: {', '.join(missing)}")
    if not args.include_community:
        print("  (unverified community-index vectors not shown; add --include-community)")
    return 0


# --- add ---

# Skip reasons that are expected in a bundle; listed only with --verbose
QUIET_SKIPS = ("another layer", "path doesn't name", "layer ", "file says it's for")


def cmd_add(args) -> int:
    import yaml
    from modulo.importer import index_entry, plan_add, register
    try:
        plan = plan_add(args.spec, args.model, files=args.files, layer=args.layer,
                        repo_type=args.repo_type, max_files=args.max_files, all_layers=args.all_layers)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    ok = plan.ok
    if (args.description or args.sign_convention) and len(ok) > 1:
        print("error: --description/--sign-convention would apply to all "
              f"{len(ok)} vectors; narrow with --files or add without them and edit registry.yaml",
              file=sys.stderr)
        return 1

    result = plan.to_dict()
    messages = []
    if args.as_index_entry:
        result["index_entries"] = [index_entry(plan, c, args.description, args.sign_convention) for c in ok]
    elif not args.dry_run and ok:
        registry = VectorRegistry(registry_dir_for(args.model))
        messages = register(plan, registry, args.description, args.sign_convention, replace=args.replace)
        result["registry"] = str(registry.registry_path)
    result["messages"] = messages

    if args.json:
        _print_json(result)
    else:
        print(f"{plan.repo_type} {plan.repo_id} @ {plan.revision[:12]} for {plan.model} "
              f"(hidden {plan.hidden_size}, {plan.num_layers} layers)")
        for n in plan.notes:
            print(f"  note: {n}")
        for c in plan.candidates:
            if c.ok:
                print(f"  ok    {c.filename} -> {c.name} ({c.format}, layer {c.layer}, shape {c.shape})")
                for n in c.notes:
                    print(f"          {n}")
            elif args.verbose or not c.reason.startswith(QUIET_SKIPS):
                print(f"  skip  {c.filename}: {c.reason}")
        quiet = [c for c in plan.candidates if not c.ok and not args.verbose
                 and c.reason.startswith(QUIET_SKIPS)]
        if quiet:
            print(f"  ({len(quiet)} more skipped: other layers of the same concepts, or files for other models; "
                  "--verbose lists them)")
        if args.as_index_entry and ok:
            print("\n# Draft entries for community/vectors.yaml; fill in the TODOs (see CONTRIBUTING.md):")
            print(yaml.safe_dump(result["index_entries"], sort_keys=False, width=100))
        elif args.dry_run and ok:
            print("\nDry run: nothing registered. Files are in the HF cache.")
        for m in messages:
            print(f"  {m}")
        if any(": added" in m for m in messages):
            print("\nImported vectors are unverified (±8 range). Next: modulo verify <name> --save-range")
    return 0 if ok else 1


# --- extract ---

def cmd_extract(args) -> int:
    from modulo.extraction import EXTRACTION_QUESTIONS, extract_vector, save_and_register
    model, tokenizer = _load_model(args.model)
    layer = args.layer if args.layer is not None else model.config.num_hidden_layers // 2
    questions = EXTRACTION_QUESTIONS[: args.n_questions]
    print(f"Extracting '{args.name}' at layer {layer} from {len(questions)} questions...", file=sys.stderr)
    axis = extract_vector(model, tokenizer, args.prompt_a, args.prompt_b, layer, questions)
    registry = VectorRegistry(registry_dir_for(args.model))
    entry = save_and_register(
        registry, args.name, axis, layer,
        description=args.description or f"Contrast of system prompts. A: {args.prompt_a!r} B: {args.prompt_b!r}",
        sign_convention=args.sign_convention,
        source=f"modulo extract, {len(questions)} questions. A={args.prompt_a!r} B={args.prompt_b!r}")
    if args.json:
        _print_json({"name": entry.name, "file": str(registry.vectors_dir / entry.file), "layer": layer,
                     "registry": str(registry.registry_path)})
    print(f"Next: modulo verify {entry.name} --model {args.model} --save-range", file=sys.stderr)
    return 0


# --- verify ---

def cmd_verify(args) -> int:
    from modulo.agent import MAX_STRENGTH, SteeringAgent
    from modulo.verify import (DEFAULT_STRENGTHS, full_range_agent, prompts_for, report_markdown,
                               summarise, sweep)
    strengths = sorted(args.strengths or DEFAULT_STRENGTHS)
    if any(not 0 < s <= MAX_STRENGTH for s in strengths):
        print(f"error: --strengths must be in (0, {MAX_STRENGTH:g}]", file=sys.stderr)
        return 1
    # Check the name before loading the model; only the index needs the network
    registry = VectorRegistry(registry_dir_for(args.model))
    if args.name not in registry.entries:
        registry.load_index_vectors(args.model, include_community=True)
    entry = registry.get(args.name)
    if entry is None:
        print(f"error: no vector '{args.name}' for {args.model}. Have: {registry.list_names()}", file=sys.stderr)
        return 1
    if args.save_range and args.name in registry.index_names:
        print("error: --save-range only works for your own vectors; index ranges are set by maintainers",
              file=sys.stderr)
        return 1

    model, tokenizer = _load_model(args.model)
    agent = full_range_agent(SteeringAgent(model, tokenizer, registry=registry, model_name=args.model), args.name)
    prompts = prompts_for(args.name, args.prompt or [])
    outputs = sweep(agent, args.name, prompts, strengths)
    norm = agent.residual_norm(entry.layer)
    stats, safe = summarise(outputs, strengths)
    label = {"id": args.name, "name": args.name, "model": args.model, "layer": entry.layer,
             "sign_convention": entry.sign_convention}
    md = report_markdown(label, norm, prompts, outputs, stats, safe, strengths)
    out = args.out or registry.vectors_dir / f"verify_{args.name}.md"
    Path(out).write_text(md)

    suggested = None if safe[-1] is None or safe[1] is None else [-safe[-1], safe[1]]
    saved = False
    if args.save_range and suggested:
        registry.add_entry(replace(entry, safe_strength_min=suggested[0], safe_strength_max=suggested[1]))
        registry.save()
        saved = True
    no_effect = not any(changed for changed, _ in stats.values())

    if args.json:
        _print_json({"name": args.name, "model": args.model, "layer": entry.layer, "residual_norm": norm,
                     "stats": [{"strength": s, "changed": c, "degenerate": d, "prompts": len(prompts)}
                               for s, (c, d) in sorted(stats.items())],
                     "suggested_range": suggested, "range_saved": saved, "no_effect": no_effect,
                     "report": str(out)})
    else:
        print(md)
        print(f"[report saved to {out}]")
        if saved:
            print(f"[saved range {suggested[0]:g}..{suggested[1]:g} to {registry.registry_path}]")
    if args.save_range and not suggested:
        print("No safe range found (degenerates at the lowest strength); range not saved.", file=sys.stderr)
    return 0 if suggested and not no_effect else 1


# --- run ---

def cmd_run(loop_args: list[str]) -> int:
    loop = REPO_ROOT / "agent" / "loop.py"
    if not loop.exists():
        print(f"error: `modulo run` needs the repo checkout (no {loop}); install with pip install -e",
              file=sys.stderr)
        return 1
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, [sys.executable, "-m", "agent.loop", *loop_args])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modulo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help, model=DEFAULT_MODEL):
        p = sub.add_parser(name, help=help, description=help)
        p.set_defaults(func=func)
        p.add_argument("--model", default=model, help="HF model id (default: all models)" if model is None
                       else f"HF model id (default: $MODULO_MODEL or {DEFAULT_MODEL})")
        p.add_argument("--json", action="store_true", help="machine-readable output")
        return p

    p = add("search", cmd_search, "Search your vectors, the community index and the catalog", model=None)
    p.add_argument("query", nargs="*", help="words that must all appear (case-insensitive)")
    p.add_argument("--status", nargs="+", choices=["local", "verified", "community", "catalog"],
                   help="only hits with these statuses")

    p = add("list", cmd_list, "Show what SteeringAgent would load for a model")
    p.add_argument("--include-community", action="store_true", help="also unverified community-index vectors")

    p = add("add", cmd_add, "Add vectors from a HuggingFace repo to your registry")
    p.add_argument("spec", help="hf:owner/repo[@revision], hf:datasets/owner/repo, or a huggingface.co URL")
    p.add_argument("--files", nargs="+", help="glob(s) over repo paths, e.g. 'qwen2.5-7b/*.pt'")
    p.add_argument("--layer", type=int, help="only files for this layer; for per_layer files, the row to use "
                   "(default: one layer per concept, nearest the middle)")
    p.add_argument("--all-layers", action="store_true", help="keep every layer of a concept, not just one")
    p.add_argument("--repo-type", choices=["model", "dataset"], help="override the repo type")
    p.add_argument("--description", default="", help="shown to the model (only when adding one vector)")
    p.add_argument("--sign-convention", default="", help="what + and - do (only when adding one vector)")
    p.add_argument("--replace", action="store_true", help="overwrite your own vectors with the same name")
    p.add_argument("--verbose", action="store_true", help="list every skipped file")
    p.add_argument("--max-files", type=int, default=50, help="refuse repos with more matching files (default 50)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="inspect only; register nothing")
    mode.add_argument("--as-index-entry", action="store_true",
                      help="print draft community/vectors.yaml entries instead of registering")

    p = add("extract", cmd_extract, "Extract a vector from two contrasting system prompts and register it")
    p.add_argument("--name", required=True)
    p.add_argument("--prompt-a", required=True, help="system prompt for the + direction")
    p.add_argument("--prompt-b", required=True, help="system prompt for the - direction")
    p.add_argument("--layer", type=int, help="default: the middle layer")
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--description", default=None, help="shown to the model in its system prompt")
    p.add_argument("--sign-convention", default="positive = prompt A direction, negative = prompt B direction")

    p = add("verify", cmd_verify, "Sweep a vector's strength, report the effect and suggest a safe range")
    p.add_argument("name")
    p.add_argument("--strengths", type=float, nargs="+", help="positive strengths, tested at + and -")
    p.add_argument("--prompt", action="append", help="extra test prompt (repeatable)")
    p.add_argument("--out", type=Path, help="Markdown report (default: vectors/<model_slug>/verify_<name>.md)")
    p.add_argument("--save-range", action="store_true", help="write the suggested range to your registry.yaml")

    sub.add_parser("run", help="Run the standalone demo; all arguments go to agent/loop.py (modulo run --help)")
    return parser


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["run"]:  # pass everything through, --help included
        return cmd_run(argv[1:])
    args = build_parser().parse_args(argv)
    if not args.json:
        return args.func(args)
    global _json_out
    _json_out = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):  # library progress messages must not corrupt the JSON
        return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Check community/vectors.yaml: entry fields for every entry, plus file checks
(download, sha256, load, shape vs the model's config, norm) for entries that are
new or changed relative to a base index.

CI runs this from the base branch against the PR's index, so a PR can't change
the checks that judge it:

    python community/check_vectors.py --index pr/community/vectors.yaml \
        --base-index base/community/vectors.yaml --summary summary.md

Run with --all to re-check every entry's file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulo.community import check_vector_file, fetch_vector, load_index, validate_index  # noqa: E402


def changed_entries(index: dict, base: dict) -> tuple[list[dict], list[str]]:
    """Entries that are new or differ from base, and ids removed from base."""
    base_by_id = {e.get("id"): e for e in base.get("entries", []) if isinstance(e, dict)}
    ids = {e.get("id") for e in index["entries"] if isinstance(e, dict)}
    changed = [e for e in index["entries"] if base_by_id.get(e.get("id")) != e]
    removed = sorted(i for i in base_by_id if i not in ids)
    return changed, removed


def maintainer_warnings(changed: list[dict], base: dict) -> list[str]:
    base_by_id = {e.get("id"): e for e in base.get("entries", []) if isinstance(e, dict)}
    warnings = []
    for e in changed:
        was_verified = base_by_id.get(e["id"], {}).get("trust") == "verified"
        if e["trust"] == "verified" or was_verified:
            warnings.append(f"{e['id']}: adds, changes or demotes a verified entry (maintainers only)")
    return warnings


def model_dims(entry: dict) -> tuple[int | None, int | None, str | None]:
    """(hidden_size, num_layers, note) from the model's config.json on the Hub."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(entry["model"], "config.json", revision=entry.get("model_revision"))
    except Exception as e:  # gated or missing models: skip the shape checks, don't fail
        return None, None, f"{entry['id']}: couldn't read {entry['model']} config ({type(e).__name__}); shape not checked"
    cfg = json.loads(Path(path).read_text())
    cfg = cfg.get("text_config", cfg) if "hidden_size" not in cfg else cfg
    return cfg.get("hidden_size"), cfg.get("num_hidden_layers"), None


def check(index_path: Path, base_path: Path | None, check_all: bool) -> tuple[list[str], list[str], list[str] | None]:
    """Returns (errors, warnings, checked ids); checked is None if field errors stopped the file checks."""
    index = load_index(index_path)
    errors = validate_index(index)
    if errors:
        return errors, [], None
    base = load_index(base_path) if base_path else {"entries": []}
    changed, removed = (index["entries"], []) if check_all else changed_entries(index, base)
    warnings = maintainer_warnings(changed, base) + [f"{i}: removed" for i in removed]
    for entry in changed:
        path, err = fetch_vector(entry)
        if err:
            errors.append(err)
            continue
        hidden, layers, note = model_dims(entry)
        if note:
            warnings.append(note)
        errors += check_vector_file(entry, path, hidden_size=hidden, num_layers=layers)
    return errors, warnings, [e["id"] for e in changed]


def summary_markdown(errors, warnings, checked) -> str:
    lines = ["## Community vector index check", ""]
    lines.append("**Result: " + ("❌ problems found" if errors else "✅ passed") + "**")
    lines.append("")
    if checked is None:
        lines.append("Files checked: skipped until the problems below are fixed")
    else:
        lines.append(f"Files checked: {', '.join(f'`{i}`' for i in checked) or 'none (no new or changed entries)'}")
    for title, items in (("Problems", errors), ("Notes for maintainers", warnings)):
        if items:
            lines += ["", f"### {title}", ""] + [f"- {i}" for i in items]
    return "\n".join(lines) + "\n"


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=root / "vectors.yaml")
    parser.add_argument("--base-index", type=Path, default=None, help="Only entries that differ from this get file checks")
    parser.add_argument("--all", action="store_true", help="File-check every entry")
    parser.add_argument("--summary", type=Path, default=None, help="Also write a Markdown summary here")
    args = parser.parse_args()

    errors, warnings, checked = check(args.index, args.base_index, args.all)
    md = summary_markdown(errors, warnings, checked)
    print(md)
    if args.summary:
        args.summary.write_text(md)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

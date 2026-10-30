"""Reports: the model's own assessments of whether a vector did what it claims.

Each `report` tool call is appended as one JSON line to
vectors/<model_slug>/reports.jsonl. A single report is one model's opinion of
itself, so treat reports as evidence to aggregate, not as ground truth.
Notes are free text written by a model: never put other people's notes back
into a model's prompt.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPORT_SCHEMA_VERSION = 1
MAX_NOTES_CHARS = 2000

# field -> (allowed types, required)
REPORT_FIELDS = {
    "schema_version": ((int,), True),
    "report_id": ((str,), True),
    "timestamp": ((str,), True),
    "session_id": ((str,), True),
    "modulo_version": ((str,), True),
    "model": ((str,), True),
    "vector": ((str,), True),
    "layer": ((int,), True),
    "vector_sha256": ((str, type(None)), True),
    "worked": ((bool,), True),
    "notes": ((str,), True),
    # Strength the vector was last steered at in this session, and how many
    # turns before the report that was (0 = active during the reporting turn).
    # Both None if the model reports on a vector it never steered with.
    "strength": ((int, float, type(None)), True),
    "turns_since_steered": ((int, type(None)), True),
    "turn": ((int,), True),
    # Experimental condition the session ran under (see modulo.agent.CONTROLS).
    # Under "sham" nothing was applied and under "random" a random direction was,
    # so aggregate only "real" reports to judge a vector. Absent in older reports.
    "control": ((str,), False),
}


def file_sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def new_session_id() -> str:
    return uuid.uuid4().hex


def make_report(*, session_id: str, model: str, vector: str, layer: int,
                vector_sha256: Optional[str], worked: bool, notes: str,
                strength: Optional[float], turns_since_steered: Optional[int],
                turn: int, control: str = "real") -> dict:
    from modulo import __version__
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "modulo_version": __version__,
        "model": model,
        "vector": vector,
        "layer": layer,
        "vector_sha256": vector_sha256,
        "worked": worked,
        "notes": notes[:MAX_NOTES_CHARS],
        "strength": strength,
        "turns_since_steered": turns_since_steered,
        "turn": turn,
        "control": control,
    }


def validate_report(report) -> list[str]:
    """Return a list of problems with a report dict (empty if valid)."""
    if not isinstance(report, dict):
        return ["report is not a JSON object"]
    errors = []
    for name, (types, required) in REPORT_FIELDS.items():
        if name not in report:
            if required:
                errors.append(f"missing field '{name}'")
            continue
        value = report[name]
        # bool is a subclass of int; only accept it where bool is listed
        if isinstance(value, bool) and bool not in types:
            errors.append(f"'{name}' has type bool")
        elif not isinstance(value, types):
            errors.append(f"'{name}' has type {type(value).__name__}")
    for name in report:
        if name not in REPORT_FIELDS:
            errors.append(f"unknown field '{name}'")
    if report.get("schema_version") not in (None, REPORT_SCHEMA_VERSION):
        errors.append(f"unsupported schema_version {report['schema_version']!r}")
    if isinstance(report.get("notes"), str) and len(report["notes"]) > MAX_NOTES_CHARS:
        errors.append(f"'notes' longer than {MAX_NOTES_CHARS} characters")
    return errors


def append_report(path: Path, report: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(report) + "\n")


def read_reports(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

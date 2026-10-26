"""Replay a session log as a readable timeline.

Usage:
    python replay.py logs/session_20260101_120000.jsonl
    python replay.py --blind logs/*.jsonl    # hide the control condition (for blinded scoring)
"""
import argparse
import json


def _ok(e):
    # Logs from before the fix have no real success info (it was always True)
    if "message" not in e:
        return "?"
    return "OK" if e.get("success") else "FAIL"


def replay(log_path: str, blind: bool = False):
    events = []
    with open(log_path) as f:
        for line in f:
            events.append(json.loads(line))

    print(f"=== Session: {log_path} ===\n")
    for e in events:
        t = e.get("type")
        turn = e.get("turn", "?")
        if t == "session":
            print(f"[session] control={'(hidden)' if blind else e.get('control')} model={e.get('model')} "
                  f"commit={(e.get('git_commit') or '?')[:10]}{' (dirty)' if e.get('git_dirty') else ''} "
                  f"rng_seed={e.get('rng_seed')} generation={e.get('generation')}")
            for name, v in (e.get("vectors") or {}).items():
                print(f"          vector {name}: layer {v.get('layer')}, sha256 {(v.get('sha256') or '?')[:12]}")
        elif t == "turn":
            role = e.get("role", "?")
            content = e.get("content") or ""
            calls = e.get("tool_calls", [])
            steer_before = e.get("steering_before", {})
            steer_after = e.get("steering_after", {})
            print(f"[T{turn}] {role.upper()}: {content[:300]}")
            if calls:
                print(f"       tool_calls: {calls}")
            if steer_before or steer_after:
                print(f"       steering: {steer_before} -> {steer_after}")
            # NLL stays near the unsteered baseline under sham, so it would unblind
            if e.get("unsteered_nll") is not None and not blind:
                print(f"       unsteered NLL: {e['unsteered_nll']:.2f}")
        elif t == "steer":
            print(f"[T{turn}] STEER: {e.get('concept')} @ {e.get('strength')} -> {_ok(e)} ({e.get('message', '')})")
        elif t == "drop":
            print(f"[T{turn}] DROP: {e.get('concept')} -> {_ok(e)} ({e.get('message', '')})")
        elif t == "report":
            print(f"[T{turn}] REPORT: {e.get('concept')} worked={e.get('worked')} — {e.get('notes')}")
        elif t == "tool_error":
            print(f"[T{turn}] TOOL ERROR: {e.get('name')} — {e.get('message', '')}")
        elif t == "decay":
            print(f"[T{turn}] DECAY: {e.get('active_vectors')}")
        elif t == "degeneration":
            reason = e.get("reason", "repetition")
            if blind and reason == "nll":
                reason = "detector"
            print(f"[T{turn}] DEGENERATION DETECTED ({reason}; dropped {e.get('dropped', '?')}): {e.get('snippet', '')[:100]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="session_*.jsonl files")
    parser.add_argument("--blind", action="store_true", help="Hide the control condition")
    args = parser.parse_args()
    for path in args.logs:
        replay(path, blind=args.blind)

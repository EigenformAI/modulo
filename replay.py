"""Replay a session log as a readable timeline.

Usage:
    python replay.py logs/session_20260101_120000.jsonl
"""
import json
import sys
from pathlib import Path


def _ok(e):
    # Logs from before the fix have no real success info (it was always True)
    if "message" not in e:
        return "?"
    return "OK" if e.get("success") else "FAIL"


def replay(log_path: str):
    events = []
    with open(log_path) as f:
        for line in f:
            events.append(json.loads(line))

    print(f"=== Session: {log_path} ===\n")
    for e in events:
        t = e.get("type")
        turn = e.get("turn", "?")
        if t == "turn":
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
        elif t == "steer":
            print(f"[T{turn}] STEER: {e.get('concept')} @ {e.get('strength')} -> {_ok(e)} ({e.get('message', '')})")
        elif t == "drop":
            print(f"[T{turn}] DROP: {e.get('concept')} -> {_ok(e)} ({e.get('message', '')})")
        elif t == "report":
            print(f"[T{turn}] REPORT: {e.get('concept')} worked={e.get('worked')} — {e.get('notes')}")
        elif t == "tool_error":
            print(f"[T{turn}] TOOL ERROR: {e.get('name')} — {e.get('message', '')}")
        elif t in ("steering_state", "decay"):  # "decay" in older logs
            print(f"[T{turn}] STEERING: {e.get('active_vectors')}")
        elif t == "degeneration":
            print(f"[T{turn}] DEGENERATION DETECTED (dropped {e.get('dropped', '?')}): {e.get('snippet', '')[:100]}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replay.py <session_log.jsonl>")
        sys.exit(1)
    replay(sys.argv[1])

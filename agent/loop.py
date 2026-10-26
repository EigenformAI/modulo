"""Standalone demo loop for the self-steering agent.

This is a thin wrapper around the pharmacopoeia library. For real agent
integration, import SteeringAgent directly:

    from pharmacopoeia import SteeringAgent

Usage:
    python -m agent.loop                  # interactive REPL
    python -m agent.loop --seed "Hi!"     # seed with a user message
    python -m agent.loop --max-turns 20   # cap turns
    python -m agent.loop --auto           # auto-continue (non-interactive)
"""
import argparse
import json
import signal
from datetime import datetime
from pathlib import Path

import torch

import config
from pharmacopoeia import SteeringAgent
from load_model import load_model_and_tokenizer


class SessionLogger:
    def __init__(self):
        self.log_path = config.LOGS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        print(f"[logger] Logging to {self.log_path}")

    def log(self, event: dict):
        event["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def log_turn(self, turn, role, content, tool_calls=None, steering_before=None, steering_after=None):
        self.log({"type": "turn", "turn": turn, "role": role, "content": content,
                  "tool_calls": [tc.__dict__ if hasattr(tc, '__dict__') else tc for tc in (tool_calls or [])],
                  "steering_before": steering_before, "steering_after": steering_after})


class KillSwitch:
    def __init__(self):
        self.killed = False
        signal.signal(signal.SIGINT, self._handler)
    def _handler(self, signum, frame):
        self.killed = True
        print("\n[kill switch] SIGINT received — will stop after this turn.")


def run_turn(agent: SteeringAgent, messages: list, logger: SessionLogger, turn: int):
    result = agent.step(messages)
    logger.log_turn(turn, "assistant", result.prose, result.tool_calls,
                    result.steering_before, result.steering_after)
    # Log tool calls individually for replay
    for tc in result.tool_calls:
        if tc.name == "steer":
            logger.log({"type": "steer", "turn": turn, "concept": tc.arguments.get("concept"),
                        "strength": tc.arguments.get("strength"), "success": True})
        elif tc.name == "drop":
            logger.log({"type": "drop", "turn": turn, "concept": tc.arguments.get("concept"), "success": True})
        elif tc.name == "report":
            logger.log({"type": "report", "turn": turn, "concept": tc.arguments.get("concept"),
                        "worked": tc.arguments.get("worked"), "notes": tc.arguments.get("notes")})
    if result.steering_before != result.steering_after:
        logger.log({"type": "decay", "turn": turn, "active_vectors": result.steering_after})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    agent = SteeringAgent(model, tokenizer, model_name=config.MODEL_NAME)
    logger = SessionLogger()
    kill = KillSwitch()

    messages = []
    if args.seed:
        messages.append({"role": "user", "content": args.seed})
        print(f"\n[seed] {args.seed}\n")

    turn = 0
    while turn < args.max_turns and not kill.killed:
        if not messages or messages[-1]["role"] != "user":
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break
            if not user_input or user_input.lower() in ("quit", "exit"):
                break
            messages.append({"role": "user", "content": user_input})
            logger.log_turn(turn, "user", user_input)

        print(f"\n--- Turn {turn+1} ---")
        print(f"Steering status:\n{agent.status()}")
        result = run_turn(agent, messages, logger, turn + 1)
        print(f"\nAssistant: {result.prose}")
        messages.append({"role": "assistant", "content": result.raw_output})
        turn += 1

        if not result.tool_calls:
            if args.auto:
                messages.append({"role": "user", "content": "(You can continue experimenting with steer if you like, or just talk.)"})
        else:
            messages.append({"role": "user", "content": "(tool calls processed; continue if you wish)"})

    print(f"\n[session ended after {turn} turns]")
    print(f"[log saved to {logger.log_path}]")


if __name__ == "__main__":
    main()

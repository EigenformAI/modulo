"""Standalone demo loop for the self-steering agent.

This is a thin wrapper around the modulo library. For real agent
integration, import SteeringAgent directly:

    from modulo import SteeringAgent

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

import config  # sets HF_HOME before transformers is imported
from modulo import SteeringAgent
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
    """First Ctrl-C during generation stops after the current turn; a second one
    (or any Ctrl-C at the input prompt) stops immediately."""
    def __init__(self):
        self.killed = False
        self.at_prompt = False
        signal.signal(signal.SIGINT, self._handler)
    def _handler(self, signum, frame):
        if self.killed or self.at_prompt:
            raise KeyboardInterrupt
        self.killed = True
        print("\n[kill switch] SIGINT received — will stop after this turn. Ctrl-C again to stop now.")


def run_turn(agent: SteeringAgent, messages: list, logger: SessionLogger, turn: int):
    result = agent.step(messages)
    logger.log_turn(turn, "assistant", result.prose, result.tool_calls,
                    result.steering_before, result.steering_after)
    if result.degenerate_output is not None:
        logger.log({"type": "degeneration", "turn": turn, "snippet": result.degenerate_output,
                    "dropped": list(result.steering_before)})
    # Log tool calls individually for replay
    for r in result.tool_results:
        if r.name in ("steer", "drop"):
            logger.log({"type": r.name, "turn": turn, "concept": r.arguments.get("concept"),
                        "strength": r.arguments.get("strength"), "success": r.ok, "message": r.message})
        elif r.name == "report":
            logger.log({"type": "report", "turn": turn, "concept": r.arguments.get("concept"),
                        "worked": r.arguments.get("worked"), "notes": r.arguments.get("notes"),
                        "success": r.ok, "message": r.message})
        else:
            logger.log({"type": "tool_error", "turn": turn, "name": r.name, "message": r.message})
    if result.steering_before != result.steering_after:
        # Steering changed this turn (steer/drop calls, decay, or expiry)
        logger.log({"type": "steering_state", "turn": turn, "active_vectors": result.steering_after})
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
        logger.log_turn(1, "user", args.seed)
        print(f"\n[seed] {args.seed}\n")

    # Inputs (user, tool results, nudges) are logged under the turn they feed into
    turn = 0
    while turn < args.max_turns and not kill.killed:
        if not messages or messages[-1]["role"] not in ("user", "tool"):
            kill.at_prompt = True
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            finally:
                kill.at_prompt = False
            if not user_input or user_input.lower() in ("quit", "exit"):
                break
            messages.append({"role": "user", "content": user_input})
            logger.log_turn(turn + 1, "user", user_input)

        print(f"\n--- Turn {turn+1} ---")
        print(f"Steering status:\n{agent.status()}")
        result = run_turn(agent, messages, logger, turn + 1)
        print(f"\nAssistant: {result.prose}")
        messages.append({"role": "assistant", "content": result.raw_output})
        turn += 1

        tool_messages = result.tool_messages()
        if tool_messages:
            # Tool results go straight back to the model, which gets the next turn
            for m in tool_messages:
                print(f"[tool] {m['content']}")
                logger.log_turn(turn + 1, "tool", m["content"])
            messages.extend(tool_messages)
        elif args.auto and turn < args.max_turns and not kill.killed:  # no nudge after the last turn
            nudge = "(You can continue experimenting with steer if you like, or just talk.)"
            messages.append({"role": "user", "content": nudge})
            logger.log_turn(turn + 1, "user", nudge)

    print(f"\n[session ended after {turn} turns]")
    print(f"[log saved to {logger.log_path}]")


if __name__ == "__main__":
    main()

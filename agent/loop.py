"""Standalone demo loop for the self-steering agent.

This is a thin wrapper around the pharmacopoeia library. For real agent
integration, import SteeringAgent directly:

    from pharmacopoeia import SteeringAgent

Usage:
    python -m agent.loop                  # interactive REPL
    python -m agent.loop --seed "Hi!"     # seed with a user message
    python -m agent.loop --max-turns 20   # cap turns
    python -m agent.loop --auto           # auto-continue (non-interactive)

Controls (see pharmacopoeia.agent.CONTROLS):
    python -m agent.loop --control sham              # steer() reports success, nothing is applied
    python -m agent.loop --control random            # random direction of the same magnitude
    python -m agent.loop --control assign --blind    # pick a condition at random, hide it until the end

Passing --control at all marks the session as part of a controlled study: the
system prompt then tells the model that some sessions are controls (without
saying which), identically in every condition, so the prompt stays truthful and
doesn't reveal the condition. Always pass --control (including --control real)
for every session in a study. The condition itself is never shown to the model;
it is always in the log's session header, so blinded sessions can be scored
first and unblinded afterwards (python replay.py --blind).
"""
import argparse
import hashlib
import json
import random
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config  # sets HF_HOME before transformers is imported
import pharmacopoeia.agent as pa
from pharmacopoeia import SteeringAgent, __version__
from load_model import load_model_and_tokenizer


class SessionLogger:
    def __init__(self):
        self.log_path = config.LOGS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        print(f"[logger] Logging to {self.log_path}")

    def log(self, event: dict):
        event["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def log_turn(self, turn, role, content, tool_calls=None, steering_before=None, steering_after=None,
                 **extra):
        self.log({"type": "turn", "turn": turn, "role": role, "content": content,
                  "tool_calls": [tc.__dict__ if hasattr(tc, '__dict__') else tc for tc in (tool_calls or [])],
                  "steering_before": steering_before, "steering_after": steering_after, **extra})


def _git(*cmd):
    try:
        return subprocess.run(["git", *cmd], cwd=config.PROJECT_ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def _sha256(path: Path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def session_metadata(agent: SteeringAgent, args, rng_seed: int, gen_params: dict) -> dict:
    """Everything needed to audit or re-run a session."""
    import torch
    import transformers
    vectors = {}
    for name, e in agent.registry.entries.items():
        d = {k: v for k, v in e.__dict__.items() if k != "feedback"}
        d["sha256"] = _sha256(agent.registry.vectors_dir / e.file)
        vectors[name] = d
    return {
        "type": "session",
        "argv": sys.argv[1:],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "pharmacopoeia_version": __version__,
        "model": config.MODEL_NAME,
        "dtype": config.DTYPE,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "control": agent.control,
        "control_seed": agent.control_seed,
        "controlled_study": agent.disclose_controls,
        "blind": args.blind,
        "rng_seed": rng_seed,
        "generation": gen_params,
        "nll_threshold": agent.nll_threshold,
        "rails": {"decay": pa.DECAY_FACTOR, "floor": pa.STRENGTH_FLOOR, "max_active": pa.MAX_ACTIVE_VECTORS,
                  "max_strength": pa.MAX_STRENGTH, "cooldown_turns": pa.COOLDOWN_TURNS},
        "registry_path": str(agent.registry.registry_path),
        "vectors": vectors,
    }


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


def run_turn(agent: SteeringAgent, messages: list, logger: SessionLogger, turn: int, gen_params: dict):
    result = agent.step(messages, **gen_params)
    logger.log_turn(turn, "assistant", result.prose, result.tool_calls,
                    result.steering_before, result.steering_after,
                    raw_output=result.raw_output, system_prompt=result.system_prompt,
                    unsteered_nll=result.unsteered_nll)
    if result.degenerate_output is not None:
        logger.log({"type": "degeneration", "turn": turn, "snippet": result.degenerate_output,
                    "reason": result.degenerate_reason, "unsteered_nll": result.unsteered_nll,
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
        logger.log({"type": "decay", "turn": turn, "active_vectors": result.steering_after})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, help="Seed user message (the first prompt)")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--control", default=None, choices=[*pa.CONTROLS, "assign"],
                        help="Experimental condition ('assign' picks one at random). Giving this "
                             "marks a controlled study; omit it for a plain demo (real steering)")
    parser.add_argument("--blind", action="store_true",
                        help="Don't print the condition until the session ends")
    parser.add_argument("--rng-seed", type=int, default=None, help="Sampling seed (default: random, logged)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--nll-threshold", type=float, default=None,
                        help="Treat steered outputs above this unsteered NLL as degenerate (off by default)")
    parser.add_argument("--auto-extract", action="store_true",
                        help="Extract a starter vector set if the model has none")
    args = parser.parse_args()

    rng_seed = args.rng_seed if args.rng_seed is not None else random.SystemRandom().randrange(2**31)
    study = args.control is not None
    if args.control == "assign":
        control = random.SystemRandom().choice(pa.CONTROLS)
    else:
        control = args.control or "real"
    gen_params = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p}

    model, tokenizer = load_model_and_tokenizer()
    from transformers import set_seed
    set_seed(rng_seed)
    agent = SteeringAgent(model, tokenizer, model_name=config.MODEL_NAME, auto_extract=args.auto_extract,
                          control=control, control_seed=rng_seed, nll_threshold=args.nll_threshold,
                          disclose_controls=study)
    logger = SessionLogger()
    logger.log(session_metadata(agent, args, rng_seed, gen_params))
    print(f"[session] control={'(blinded)' if args.blind else control} rng_seed={rng_seed}")
    kill = KillSwitch()

    messages = []
    if args.seed:
        messages.append({"role": "user", "content": args.seed})
        print(f"\n[seed] {args.seed}\n")

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
            logger.log_turn(turn, "user", user_input)

        print(f"\n--- Turn {turn+1} ---")
        print(f"Steering status:\n{agent.status()}")
        result = run_turn(agent, messages, logger, turn + 1, gen_params)
        print(f"\nAssistant: {result.prose}")
        messages.append({"role": "assistant", "content": result.raw_output})
        turn += 1

        tool_messages = result.tool_messages()
        if tool_messages:
            # Tool results go straight back to the model, which gets the next turn
            for m in tool_messages:
                print(f"[tool] {m['content']}")
                logger.log_turn(turn, "tool", m["content"])
            messages.extend(tool_messages)
        elif args.auto:
            nudge = "(You can continue experimenting with steer if you like, or just talk.)"
            messages.append({"role": "user", "content": nudge})
            logger.log_turn(turn, "user", nudge)

    print(f"\n[session ended after {turn} turns]")
    if args.blind:
        print(f"[unblinded] control was: {control}")
    print(f"[log saved to {logger.log_path}]")


if __name__ == "__main__":
    main()

"""Standalone demo loop for the self-steering agent.

This is a thin wrapper around the modulo library. For real agent
integration, import SteeringAgent directly:

    from modulo import SteeringAgent

Usage:
    python -m agent.loop                  # interactive REPL
    python -m agent.loop --seed "Hi!"     # seed with a user message
    python -m agent.loop --max-turns 20   # cap turns
    python -m agent.loop --auto           # auto-continue (non-interactive)
    python -m agent.loop --steer honesty=6 --model Qwen/Qwen2.5-3B-Instruct

Controls (see modulo.agent.CONTROLS):
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
import json
import random
import signal
import subprocess
import sys
from datetime import datetime

import config  # sets HF_HOME before transformers is imported
import modulo.agent as ma
from modulo import SteeringAgent, __version__
from modulo.reports import file_sha256
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
                 raw_output=None, **extra):
        self.log({"type": "turn", "turn": turn, "role": role, "content": content, "raw_output": raw_output,
                  "tool_calls": [tc.__dict__ if hasattr(tc, '__dict__') else tc for tc in (tool_calls or [])],
                  "steering_before": steering_before, "steering_after": steering_after, **extra})


def _git(*cmd):
    try:
        return subprocess.run(["git", *cmd], cwd=config.PROJECT_ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def session_metadata(agent: SteeringAgent, args, rng_seed: int, gen_params: dict) -> dict:
    """Everything needed to audit or re-run a session."""
    import torch
    import transformers
    vectors = {}
    for name, e in agent.registry.entries.items():
        vectors[name] = {**e.__dict__, "sha256": file_sha256(agent.registry.vectors_dir / e.file)}
    return {
        "type": "session",
        "argv": sys.argv[1:],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "modulo_version": __version__,
        "model": agent.model_name,
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
        "steer_at_start": args.steer,
        "rails": {"decay": ma.DECAY_FACTOR, "floor": ma.STRENGTH_FLOOR, "max_active": ma.MAX_ACTIVE_VECTORS,
                  "max_strength": ma.MAX_STRENGTH, "cooldown_turns": ma.COOLDOWN_TURNS,
                  "repeat_similarity": ma.REPEAT_SIMILARITY, "repeat_lookback": ma.REPEAT_LOOKBACK},
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
                    result.steering_before, result.steering_after, result.raw_output,
                    system_prompt=result.system_prompt, unsteered_nll=result.unsteered_nll)
    if result.degenerate_output is not None:
        logger.log({"type": "degeneration", "turn": turn, "snippet": result.degenerate_output,
                    "reason": result.degenerate_reason, "unsteered_nll": result.unsteered_nll,
                    "dropped": list(result.steering_before)})
    if result.repeated_output is not None:
        logger.log({"type": "repetition", "turn": turn, "snippet": result.repeated_output})
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
    parser.add_argument("--seed", default=None, help="Seed user message (the first prompt)")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--nudge", choices=["status", "legacy"], default="status",
                        help="--auto filler when the model makes no tool call: 'status' (default) is a neutral "
                             "tool-role status line with current steering and no suggestion; 'legacy' is the "
                             "old user message suggesting steer")
    parser.add_argument("--model", default=config.MODEL_NAME, help=f"HF model id (default: {config.MODEL_NAME})")
    parser.add_argument("--steer", action="append", default=[], metavar="NAME=STRENGTH",
                        help="apply a vector before the first turn, as if the model had called steer "
                             "(one full-strength turn, then decay). Repeatable.")
    parser.add_argument("--control", default=None, choices=[*ma.CONTROLS, "assign"],
                        help="Experimental condition ('assign' picks one at random). Giving this "
                             "marks a controlled study; omit it for a plain demo (real steering)")
    parser.add_argument("--blind", action="store_true",
                        help="Don't print the condition until the session ends")
    parser.add_argument("--rng-seed", type=int, default=None, help="Sampling seed (default: random, logged)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--nll-threshold", type=float, default=None,
                        help="Treat steered outputs above this unsteered NLL as degenerate (off by default; "
                             "NLL doesn't separate word salad from heavy role-play, see AGENTS.md)")
    args = parser.parse_args()
    steers = []
    for spec in args.steer:
        name, _, strength = spec.partition("=")
        try:
            steers.append((name.strip(), float(strength)))
        except ValueError:
            parser.error(f"--steer expects NAME=STRENGTH, got {spec!r}")
        if not name.strip():
            parser.error(f"--steer expects NAME=STRENGTH, got {spec!r}")

    rng_seed = args.rng_seed if args.rng_seed is not None else random.SystemRandom().randrange(2**31)
    study = args.control is not None
    control = random.SystemRandom().choice(ma.CONTROLS) if args.control == "assign" else (args.control or "real")
    gen_params = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p}

    model, tokenizer = load_model_and_tokenizer(args.model)
    from transformers import set_seed
    set_seed(rng_seed)
    # Offer every vector we can: unverified index entries too (marked as such in the prompt)
    agent = SteeringAgent(model, tokenizer, model_name=args.model, include_community=True,
                          control=control, control_seed=rng_seed, disclose_controls=study,
                          nll_threshold=args.nll_threshold)
    for name, strength in steers:
        ok, msg = agent.apply(name, strength)
        if not ok:
            raise SystemExit(f"--steer {name}: {msg}")
        print(f"[steer] {msg}")
    logger = SessionLogger()
    logger.log(session_metadata(agent, args, rng_seed, gen_params))
    print(f"[session] control={'(blinded)' if args.blind else control} rng_seed={rng_seed}")
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
        result = run_turn(agent, messages, logger, turn + 1, gen_params)
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
            if args.nudge == "status":
                # Tool role so it reads as the environment, not a person to reply to
                nudge = (f"[status] Turn {turn + 1} of {args.max_turns}. No one is replying during this "
                         f"session; the transcript is saved to be read later.\n"
                         f"Steering for your next turn:\n{agent.status()}\nThis turn is yours.")
                role = "tool"
            else:
                nudge = "(You can continue experimenting with steer if you like, or just talk.)"
                role = "user"
            messages.append({"role": role, "content": nudge})
            logger.log_turn(turn + 1, role, nudge)

    print(f"\n[session ended after {turn} turns]")
    if args.blind:
        print(f"[unblinded] control was: {control}")
    print(f"[log saved to {logger.log_path}]")


if __name__ == "__main__":
    main()

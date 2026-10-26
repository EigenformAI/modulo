"""Main agent loop: turn-by-turn generation with self-steering.

Usage:
    python -m agent.loop                  # interactive REPL
    python -m agent.loop --seed "Hi!"     # seed with a user message
    python -m agent.loop --max-turns 20   # cap turns
"""
import argparse
import json
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

import config
from agent.steering_state import SteeringState
from agent.system_prompt import build_system_prompt
from agent.tool_parser import parse_tool_calls, strip_tool_calls
from agent.vector_registry import VectorRegistry
from load_model import load_model_and_tokenizer


# --- Degeneration detector ---
def is_degenerate(text: str) -> bool:
    """Heuristic: detect runaway repetition / entropy crash."""
    text = text.strip()
    if len(text) < 20:
        return False
    # 1. Single-word repetition (e.g. "the the the the the the")
    if re.search(r'(\b\w+\b)( \1){5,}', text):
        return True
    # 2. Phrase repetition: check if any 4-word chunk repeats 3+ times
    words = text.split()
    if len(words) >= 12:
        from collections import Counter
        chunks = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
        c = Counter(chunks)
        if c.most_common(1)[0][1] >= 3:
            return True
    # 3. Character-level repetition (e.g. "aaaaaaa" or "..........")
    if re.search(r'(.)\1{15,}', text):
        return True
    # 4. Very low vocabulary diversity (same few words cycling)
    if len(words) >= 40:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.15:  # fewer than 15% unique words
            return True
    return False


class SessionLogger:
    def __init__(self):
        self.log_path = config.LOGS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.events = []
        print(f"[logger] Logging to {self.log_path}")

    def log(self, event: dict):
        event["timestamp"] = datetime.now().isoformat()
        self.events.append(event)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def log_turn(self, turn, role, content, tool_calls=None, steering_before=None, steering_after=None):
        self.log({
            "type": "turn",
            "turn": turn,
            "role": role,
            "content": content,
            "tool_calls": tool_calls or [],
            "steering_before": steering_before,
            "steering_after": steering_after,
        })

    def log_steer(self, turn, concept, strength, success, message):
        self.log({
            "type": "steer",
            "turn": turn,
            "concept": concept,
            "strength": strength,
            "success": success,
            "message": message,
        })

    def log_drop(self, turn, concept, success, message):
        self.log({
            "type": "drop",
            "turn": turn,
            "concept": concept,
            "success": success,
            "message": message,
        })

    def log_report(self, turn, concept, worked, notes):
        self.log({
            "type": "report",
            "turn": turn,
            "concept": concept,
            "worked": worked,
            "notes": notes,
        })

    def log_decay(self, turn, active_vectors):
        self.log({
            "type": "decay",
            "turn": turn,
            "active_vectors": active_vectors,
        })

    def log_degeneration(self, turn, snippet):
        self.log({
            "type": "degeneration",
            "turn": turn,
            "snippet": snippet,
        })


class KillSwitch:
    def __init__(self):
        self.killed = False
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        self.killed = True
        print("\n[kill switch] SIGINT received — will stop after this turn.")


def run_turn(model, tokenizer, messages, state: SteeringState, logger: SessionLogger, turn: int):
    """Run one agent turn: build prompt, generate (steered), parse, apply tool calls, decay."""
    # Build system prompt with current steering status
    sys_prompt = build_system_prompt(state.registry, state.status())
    full_messages = [{"role": "system", "content": sys_prompt}] + messages

    text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # Generate with active steering vectors
    steering_before = {n: av.current_strength for n, av in state.active.items()}
    with state.build_steering_context(model, prompt_len):
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
    gen_ids = out[0][prompt_len:]
    raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Degeneration check
    if is_degenerate(raw_output):
        logger.log_degeneration(turn, raw_output[:200])
        print(f"\n[DEGENERATION DETECTED on turn {turn}] Auto-dropping all vectors.")
        for name in list(state.active.keys()):
            state.drop(name)
        # Regenerate without steering
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
        gen_ids = out[0][prompt_len:]
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Parse tool calls
    tool_calls = parse_tool_calls(raw_output)
    prose = strip_tool_calls(raw_output)

    # Log the turn
    logger.log_turn(turn, "assistant", prose, [tc.__dict__ for tc in tool_calls],
                    steering_before, {n: av.current_strength for n, av in state.active.items()})

    # Execute tool calls
    for tc in tool_calls:
        if tc.name == "steer":
            concept = tc.arguments.get("concept", "")
            strength = float(tc.arguments.get("strength", 0))
            success, msg = state.apply(concept, strength)
            logger.log_steer(turn, concept, strength, success, msg)
            print(f"  [steer] {msg}")
        elif tc.name == "drop":
            concept = tc.arguments.get("concept", "")
            success, msg = state.drop(concept)
            logger.log_drop(turn, concept, success, msg)
            print(f"  [drop] {msg}")
        elif tc.name == "report":
            concept = tc.arguments.get("concept", "")
            worked = bool(tc.arguments.get("worked", True))
            notes = tc.arguments.get("notes", "")
            state.registry.add_feedback(concept, worked, notes)
            logger.log_report(turn, concept, worked, notes)
            print(f"  [report] {concept}: worked={worked} — {notes}")

    # Decay
    active_before_decay = {n: av.current_strength for n, av in state.active.items()}
    state.decay()
    active_after_decay = {n: av.current_strength for n, av in state.active.items()}
    if active_before_decay != active_after_decay:
        logger.log_decay(turn, active_after_decay)

    return prose, raw_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, help="Initial user message")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--auto", action="store_true",
                        help="Auto-continue with a gentle nudge when the model stops making tool calls (non-interactive mode)")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    registry = VectorRegistry()
    state = SteeringState(registry)
    logger = SessionLogger()
    kill = KillSwitch()

    messages = []
    if args.seed:
        messages.append({"role": "user", "content": args.seed})
        print(f"\n[seed] {args.seed}\n")

    turn = 0
    while turn < args.max_turns and not kill.killed:
        # If no user message yet, prompt for one
        if not messages or messages[-1]["role"] != "user":
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                break
            if not user_input:
                break
            if user_input.lower() in ("quit", "exit"):
                break
            messages.append({"role": "user", "content": user_input})
            logger.log_turn(turn, "user", user_input)

        # Run agent turn
        print(f"\n--- Turn {turn+1} ---")
        print(f"Steering status:\n{state.status()}")
        prose, raw = run_turn(model, tokenizer, messages, state, logger, turn + 1)
        print(f"\nAssistant: {prose}")
        messages.append({"role": "assistant", "content": raw})
        turn += 1

        # After an assistant turn, the next turn expects user input
        # (unless the assistant made tool calls that should auto-continue)
        if not parse_tool_calls(raw):
            # No tool calls — wait for user input next loop
            if args.auto:
                # Non-interactive: inject a gentle nudge to keep the session going
                messages.append({"role": "user", "content": "(You can continue experimenting with steer if you like, or just talk. What's on your mind?)"})
            else:
                pass  # wait for user input
        else:
            # Tool calls made — inject a "tool result" so the loop continues
            messages.append({"role": "user", "content": "(tool calls processed; continue if you wish)"})

    print(f"\n[session ended after {turn} turns]")
    print(f"[log saved to {logger.log_path}]")


if __name__ == "__main__":
    main()

"""SteeringState: tracks active steering vectors, their decay, and safety caps."""
from dataclasses import dataclass, field
from typing import Optional

import torch
from steering_vectors import SteeringVector

from agent.vector_registry import VectorEntry, VectorRegistry


# --- Safety rails (tunable) ---
DECAY_FACTOR = 0.85          # strength *= DECAY_FACTOR each turn
STRENGTH_FLOOR = 0.1         # drop if below this (relative to base strength)
MAX_ACTIVE_VECTORS = 4       # cap simultaneously-active vectors
MAX_STRENGTH = 10.0          # absolute cap on any vector's strength
COOLDOWN_TURNS = 2           # can't re-apply same concept within N turns


@dataclass
class ActiveVector:
    entry: VectorEntry
    base_strength: float       # what the agent requested
    current_strength: float   # after decay
    turns_active: int = 0
    last_applied_turn: int = 0


class SteeringState:
    def __init__(self, registry: VectorRegistry):
        self.registry = registry
        self.active: dict[str, ActiveVector] = {}  # name -> ActiveVector
        self.turn = 0
        self._dropped_recently: dict[str, int] = {}  # name -> turn dropped

    def apply(self, name: str, strength: float) -> tuple[bool, str]:
        """Apply or refresh a steering vector. Returns (success, message)."""
        entry = self.registry.get(name)
        if entry is None:
            return False, f"Unknown vector '{name}'. Available: {self.registry.list_names()}"

        # Clamp to safe range
        strength = max(entry.safe_strength_min, min(entry.safe_strength_max, strength))
        strength = max(-MAX_STRENGTH, min(MAX_STRENGTH, strength))

        # Cooldown check (if already active, refreshing is fine; if dropped, must wait)
        if name not in self.active and name in self._dropped_recently:
            return False, f"'{name}' was dropped recently; on cooldown for {COOLDOWN_TURNS} turns."

        # Cap check
        if name not in self.active and len(self.active) >= MAX_ACTIVE_VECTORS:
            return False, f"Too many active vectors (max {MAX_ACTIVE_VECTORS}). Drop one first."

        self.active[name] = ActiveVector(
            entry=entry,
            base_strength=strength,
            current_strength=strength,
            last_applied_turn=self.turn,
        )
        sign = "+" if strength >= 0 else ""
        return True, f"steer({name}, {sign}{strength:.2f}) applied. Will decay each turn."

    def drop(self, name: str) -> tuple[bool, str]:
        if name not in self.active:
            return False, f"'{name}' is not active."
        del self.active[name]
        self._dropped_recently[name] = self.turn
        return True, f"Dropped '{name}'."

    def decay(self):
        """Apply per-turn decay; drop vectors below the floor."""
        self.turn += 1
        to_drop = []
        for name, av in self.active.items():
            av.turns_active += 1
            if av.turns_active > 1:  # don't decay on the turn it was applied
                av.current_strength *= DECAY_FACTOR
            if abs(av.current_strength) < STRENGTH_FLOOR:
                to_drop.append(name)
        for name in to_drop:
            del self.active[name]
            self._dropped_recently[name] = self.turn
        # Clean old cooldowns
        self._dropped_recently = {
            n: t for n, t in self._dropped_recently.items()
            if self.turn - t < COOLDOWN_TURNS
        }

    def build_steering_context(self, model, prompt_len: int):
        """Build a context manager that applies all active vectors during a forward pass.
        Returns a context manager (or a no-op if none active)."""
        if not self.active:
            from contextlib import nullcontext
            return nullcontext()

        # Combine all active vectors into a single SteeringVector per their layers.
        # Most vectors use the same layer (14); if they differ, we register each.
        layer_vectors = {}  # layer -> summed vector
        for av in self.active.values():
            vec = av.entry.load_vector() * av.current_strength
            layer = av.entry.layer
            if layer in layer_vectors:
                layer_vectors[layer] = layer_vectors[layer] + vec
            else:
                layer_vectors[layer] = vec

        sv = SteeringVector(layer_vectors, layer_type="decoder_block")
        # Apply to generated tokens only (after the prompt)
        return sv.apply(model, multiplier=1.0, min_token_index=prompt_len)

    def status(self) -> str:
        """Human-readable status of active vectors."""
        if not self.active:
            return "No active steering vectors."
        lines = []
        for name, av in self.active.items():
            sign = "+" if av.current_strength >= 0 else ""
            lines.append(
                f"  {name}: {sign}{av.current_strength:.2f} "
                f"(base {sign}{av.base_strength:.2f}, turn {av.turns_active})"
            )
        return "\n".join(lines)

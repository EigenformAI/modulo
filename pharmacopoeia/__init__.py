"""Pharmacopoeia: self-steering for local language models.

A library that lets an agent harness give a model the ability to steer its own
activations via steering vectors. The model gets `steer`, `drop`, and `report`
tools. Vectors decay each turn to prevent doom loops.

Public API:
    from pharmacopoeia import SteeringAgent, VectorRegistry, extract_vector

Quick start:
    from pharmacopoeia import SteeringAgent
    agent = SteeringAgent(model, tokenizer)  # any HF causal LM
    # agent now has .step(messages) -> (prose, tool_calls, raw_output)
    # and .build_system_prompt() for the steer tool description
"""
from pharmacopoeia.agent import SteeringAgent
from pharmacopoeia.registry import VectorRegistry, VectorEntry
from pharmacopoeia.extraction import extract_vector, extract_auto

__version__ = "0.1.0"

__all__ = [
    "SteeringAgent",
    "VectorRegistry",
    "VectorEntry",
    "extract_vector",
    "extract_auto",
]

"""Modulo: self-steering for local language models.

A library that lets an agent harness give a model the ability to steer its own
activations via steering vectors. The model gets `steer`, `drop`, and `report`
tools. Vectors decay each turn to prevent doom loops.

Public API:
    from modulo import SteeringAgent, VectorRegistry, extract_vector

Quick start:
    from modulo import SteeringAgent
    agent = SteeringAgent(model, tokenizer)  # HF causal LM with model.model.layers
    result = agent.step(messages)            # -> StepResult
    messages += [{"role": "assistant", "content": result.raw_output}]
    messages += result.tool_messages()       # tool results back to the model
"""
from modulo.agent import SteeringAgent
from modulo.registry import VectorRegistry, VectorEntry
from modulo.extraction import extract_vector, extract_auto, save_and_register
from modulo.reports import read_reports, validate_report

__version__ = "0.1.0"

__all__ = [
    "SteeringAgent",
    "VectorRegistry",
    "VectorEntry",
    "extract_vector",
    "extract_auto",
    "save_and_register",
    "read_reports",
    "validate_report",
]

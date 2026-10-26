"""Parse Qwen tool-call blocks from model output.

Qwen2.5 emits tool calls as:
    <|im_start|>assistant
    {"name": "steer", "arguments": {"concept": "sycophancy", "strength": 2.0}}
    <|im_end|>

We parse the JSON object inside the assistant turn.
"""
import json
import re
from dataclasses import dataclass


# Matches a JSON tool call. Qwen emits it as a bare JSON object on its own line
# (no <tool_call> tags in the Qwen2.5 chat template — it's just JSON).
TOOL_CALL_RE = re.compile(r'\{"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\}', re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract all tool calls from model output text."""
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1)
        raw_args = match.group(2)
        try:
            args = json.loads(raw_args)
            calls.append(ToolCall(name=name, arguments=args, raw=match.group(0)))
        except json.JSONDecodeError:
            continue
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove tool-call JSON from text, leaving prose."""
    return TOOL_CALL_RE.sub("", text).strip()

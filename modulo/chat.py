"""Chat-template rendering shared by the agent, extraction and verification."""
from typing import Optional


# Tool results are sent as role "tool". Templates that can't show that role get
# them as a user message starting with this label instead.
TOOL_RESULT_LABEL = "[Tool result]"
_TOOL_PROBE = "MODULO_TOOL_ROLE_PROBE"


def _render(tokenizer, messages: list[dict], tools: Optional[list] = None) -> str:
    extra = {"tools": tools} if tools else {}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False, **extra)


def supports_tool_role(tokenizer) -> bool:
    """Whether the chat template shows a role-"tool" message to the model (cached on the tokenizer).

    Some don't: OLMo-2's template silently drops tool messages, and
    Mistral-7B-Instruct-v0.3's raises unless they carry tool-call IDs.
    """
    cached = getattr(tokenizer, "_modulo_tool_role", None)
    if cached is None:
        probe = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                 {"role": "tool", "content": _TOOL_PROBE}]
        try:
            cached = _TOOL_PROBE in _render(tokenizer, probe)
        except Exception:
            cached = False
        try:
            tokenizer._modulo_tool_role = cached
        except AttributeError:
            pass
    return cached


def adapt_messages(tokenizer, messages: list[dict]) -> list[dict]:
    """Messages the template can render: unchanged, unless it can't show tool messages.

    Then each tool message becomes a user message starting with TOOL_RESULT_LABEL,
    and consecutive user messages are merged (some templates, like Mistral's,
    require user and assistant turns to alternate).
    """
    if supports_tool_role(tokenizer):
        return messages
    out: list[dict] = []
    for m in messages:
        if m["role"] == "tool":
            m = {"role": "user", "content": f"{TOOL_RESULT_LABEL}\n{m['content']}"}
        if out and m["role"] == "user" and out[-1]["role"] == "user":
            out[-1] = {"role": "user", "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append(dict(m))
    return out


def chat_prompt(tokenizer, messages: list[dict], tools: Optional[list] = None) -> str:
    """The prompt text for messages, ready for the assistant's turn.

    Thinking mode is turned off: Qwen3-style templates think by default, which
    spends the turn's token budget inside <think> (so steered answers, tool
    calls and extraction activations never come from a real reply). Templates
    without the variable ignore it. Tool messages are adapted for templates
    that can't show them (see adapt_messages). `tools` (a harness's own tool
    schemas) is passed to the template, for templates that list tools.
    """
    return _render(tokenizer, adapt_messages(tokenizer, messages), tools)


def encode(tokenizer, text: str, device=None):
    """Token ids for a rendered chat prompt, as a batch of one.

    The rendered template already contains the model's special tokens, so none
    are added: tokenizing it the usual way gave Llama-3 a second
    <|begin_of_text|>, which became an extra attention sink.
    """
    inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    return inputs.to(device) if device is not None else inputs

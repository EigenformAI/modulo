"""Chat-template rendering shared by the agent, extraction and verification."""


def chat_prompt(tokenizer, messages: list[dict]) -> str:
    """The prompt text for messages, ready for the assistant's turn.

    Thinking mode is turned off: Qwen3-style templates think by default, which
    spends the turn's token budget inside <think> (so steered answers, tool
    calls and extraction activations never come from a real reply). Templates
    without the variable ignore it.
    """
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)

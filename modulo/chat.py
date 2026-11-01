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


def encode(tokenizer, text: str, device=None):
    """Token ids for a rendered chat prompt, as a batch of one.

    The rendered template already contains the model's special tokens, so none
    are added: tokenizing it the usual way gave Llama-3 a second
    <|begin_of_text|>, which became an extra attention sink.
    """
    inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    return inputs.to(device) if device is not None else inputs

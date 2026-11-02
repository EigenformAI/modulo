"""Chat rendering for templates that can't show tool messages (CPU only, stub tokenizers)."""
import pytest

from modulo.chat import TOOL_RESULT_LABEL, adapt_messages, chat_prompt, supports_tool_role


class StubTokenizer:
    """Renders 'role: content' lines, like a chat template would."""

    def __init__(self):
        self.calls = 0

    def check(self, messages):
        pass

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        self.calls += 1
        self.check(messages)
        return "".join(f"{m['role']}: {m['content']}\n" for m in self.shown(messages)) + "assistant:"

    def shown(self, messages):
        return messages


class DropsTool(StubTokenizer):
    """Like OLMo-2: tool messages silently vanish."""

    def shown(self, messages):
        return [m for m in messages if m["role"] != "tool"]


class StrictTemplate(StubTokenizer):
    """Like Mistral-7B-Instruct-v0.3: rejects tool messages without IDs and requires alternation."""

    def check(self, messages):
        roles = [m["role"] for m in messages if m["role"] != "system"]
        if "tool" in roles:
            raise ValueError("Tool call IDs should be alphanumeric strings with length 9!")
        if any(a == b for a, b in zip(roles, roles[1:])):
            raise ValueError("conversation roles must alternate user/assistant")


CONVERSATION = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": '{"name": "steer", "arguments": {"concept": "humor", "strength": 6}}'},
    {"role": "tool", "content": '{"name": "steer", "ok": true}'},
    {"role": "tool", "content": "[status] Turn 2 of 8."},
]


def test_template_with_tool_role_is_left_alone():
    tok = StubTokenizer()
    assert supports_tool_role(tok)
    assert adapt_messages(tok, CONVERSATION) is CONVERSATION
    assert "tool: [status] Turn 2 of 8." in chat_prompt(tok, CONVERSATION)


@pytest.mark.parametrize("tok", [DropsTool(), StrictTemplate()], ids=["drops-tool", "strict"])
def test_tool_results_reach_templates_without_tool_role(tok):
    assert not supports_tool_role(tok)
    text = chat_prompt(tok, CONVERSATION)
    assert '{"name": "steer", "ok": true}' in text
    assert "[status] Turn 2 of 8." in text
    assert TOOL_RESULT_LABEL in text


def test_consecutive_tool_and_user_messages_are_merged():
    tok = StrictTemplate()
    msgs = adapt_messages(tok, CONVERSATION + [{"role": "user", "content": "and now?"}])
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    last = msgs[-1]["content"]
    assert last.startswith(TOOL_RESULT_LABEL)
    assert last.count(TOOL_RESULT_LABEL) == 2 and last.endswith("and now?")
    assert CONVERSATION[3]["role"] == "tool"  # the caller's messages are not modified


def test_probe_runs_once_per_tokenizer():
    tok = DropsTool()
    chat_prompt(tok, CONVERSATION)
    chat_prompt(tok, CONVERSATION)
    assert tok.calls == 3  # one probe, two renders


def test_harness_tools_reach_the_template():
    seen = []

    class Recording(StubTokenizer):
        def apply_chat_template(self, messages, **kw):
            seen.append(kw.get("tools"))
            return super().apply_chat_template(messages, **kw)
    tok = Recording()
    tools = [{"type": "function", "function": {"name": "search"}}]
    chat_prompt(tok, CONVERSATION, tools=tools)
    chat_prompt(tok, CONVERSATION)
    assert seen[-2] == tools and seen[-1] is None  # no tools: the argument isn't passed at all

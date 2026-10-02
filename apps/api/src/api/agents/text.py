from typing import Any, List


def last_message_text(messages: List[Any]) -> str:
    """Text of the final message that actually said something."""
    for message in reversed(messages or []):
        text = message.text.strip()
        if text:
            return text
    return ""


def join_tool_output(messages: List[Any]) -> str:
    """All tool results concatenated, used as a citation fallback."""
    from langchain_core.messages import ToolMessage

    return "".join(
        message.text for message in messages or [] if isinstance(message, ToolMessage)
    )


def optional(value: str | None, default: str = "") -> str:
    return value if value else default

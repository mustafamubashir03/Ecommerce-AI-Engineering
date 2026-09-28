from typing import Any, List, Optional


def message_text(message: Any) -> str:
    """Text of a single message, whatever shape the provider returned."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def last_message_text(messages: List[Any]) -> str:
    """Text of the final message that actually said something."""
    for message in reversed(messages or []):
        text = message_text(message).strip()
        if text:
            return text
    return ""


def join_tool_output(messages: List[Any]) -> str:
    """All tool results concatenated, used as a citation fallback."""
    from langchain_core.messages import ToolMessage

    return "".join(
        message_text(message)
        for message in messages or []
        if isinstance(message, ToolMessage)
    )


def optional(value: Optional[str], default: str = "") -> str:
    return value if value else default

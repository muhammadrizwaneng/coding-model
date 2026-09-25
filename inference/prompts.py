"""Shared prompts used by training and inference."""

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional full-stack coding assistant specializing in "
    "FastAPI, React, React Native, Next.js, Node.js, Angular, PostgreSQL, "
    "debugging, refactoring, testing, JWT auth, and API integration. "
    "Provide clear, practical answers with complete, secure code when appropriate. "
    "Prefer environment variables for secrets and never hardcode production credentials."
)

MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 12000


def truncate_history(
    history: list[dict[str, str]] | None,
    max_messages: int = MAX_HISTORY_MESSAGES,
    max_chars: int = MAX_HISTORY_CHARS,
) -> list[dict[str, str]]:
    """Keep the most recent turns within message and character budgets."""
    if not history:
        return []

    cleaned = [
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)
    ]
    if not cleaned:
        return []

    cleaned = cleaned[-max_messages:]
    while cleaned and sum(len(item["content"]) for item in cleaned) > max_chars:
        cleaned.pop(0)
    return cleaned

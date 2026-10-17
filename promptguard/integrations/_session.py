"""The Conversation a framework integration's scans belong to.

The Guard API remembers a Conversation's turns when a scan names it in
``context.session_id``. An integration must send the framework's real
conversation id there -- the same value on every turn of one conversation, a
different one for another -- or nothing at all. A per-call run or event id
would make every scan a conversation of its own.
"""

from typing import Any

# The metadata keys a LangChain call carries its conversation id under, in the
# order they are read. LangGraph copies the checkpointer's ``thread_id`` into
# every callback's metadata; ``session_id`` and ``conversation_id`` are the
# other two keys LangSmith groups a thread by, set through ``config["metadata"]``.
CONVERSATION_METADATA_KEYS = ("thread_id", "session_id", "conversation_id")


def as_session_id(value: Any) -> str | None:
    """``value`` as a session id, or ``None`` when it cannot be one.

    A non-empty string is used as it is, and an integer (LangGraph accepts an
    int ``thread_id``) as its decimal form. Anything else -- ``None``, a bool,
    an empty string, an object -- names no conversation.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value
    return None


def session_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """The conversation id a LangChain call's metadata names, if any."""
    if not metadata:
        return None
    for key in CONVERSATION_METADATA_KEYS:
        session_id = as_session_id(metadata.get(key))
        if session_id is not None:
            return session_id
    return None


def validate_session_id(session_id: str | None) -> str | None:
    """Check a ``session_id`` an application passed to an integration."""
    if session_id is None:
        return None
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError(f"session_id must be a non-empty string, got {session_id!r}")
    return session_id


def with_session_id(context: dict[str, Any], session_id: str | None) -> dict[str, Any]:
    """``context`` naming ``session_id``; unchanged when there is none."""
    if session_id is not None:
        context["session_id"] = session_id
    return context

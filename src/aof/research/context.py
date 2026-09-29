"""Research context for current item during research runs."""

from __future__ import annotations

from contextvars import ContextVar

current_research_item_id: ContextVar[str | None] = ContextVar(
    "current_research_item_id", default=None
)

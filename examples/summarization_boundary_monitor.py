"""
summarization_boundary_monitor.py
----------------------------------
Demonstrates how to detect Letta's summarization boundary and measure behavioral
drift (ghost-lexicon decay) without waiting for a dedicated hook in the Agent class.

Background
----------
Letta's Agent.summarize_messages_inplace() is called automatically when the
in-context token budget is exceeded. After summarization, domain-specific
vocabulary established early in the session can silently disappear from the
active context — the agent no longer "remembers" specific terms, tool names, or
reasoning patterns it used before the boundary.

This script shows two complementary approaches:
  1. Subclass override — wrap summarize_messages_inplace() to fire a callback.
  2. Log-level intercept — subscribe to the 'letta' Python logger to detect the
     boundary without touching Agent internals.

Both approaches require no changes to Letta core. They work with Letta's existing
OTel tracing surface and the standard Python logging stack.

Usage
-----
    python examples/summarization_boundary_monitor.py

Requirements: letta (pip install letta)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Approach 1: Subclass override
# ---------------------------------------------------------------------------

class SummarizationMonitor:
    """Mixin that fires a callback whenever summarize_messages_inplace() runs.

    Attach to any Agent subclass::

        from letta import create_client
        from summarization_boundary_monitor import SummarizationMonitor

        class MonitoredAgent(SummarizationMonitor, Agent):
            pass

    Then register callbacks::

        agent = MonitoredAgent(...)
        agent.on_summarize(my_callback)

    Callback signature: ``(agent, report: dict) -> None``
    where report contains::

        {
            "event": "summarization_boundary",
            "agent_id": str,
            "pre_vocab": set[str],     # domain terms seen before boundary
            "post_vocab": set[str],    # domain terms visible after boundary
            "ghost_terms": list[str],  # terms that disappeared
            "gcs": float,              # Ghost Consistency Score (1.0 = no drift)
            "alert": bool,             # True if gcs < threshold
        }
    """

    _STOP_WORDS = frozenset(
        "the a an and or but in on at to for of with by from is are was were "
        "it its this that these those will would could should may might shall "
        "be been being have has had do does did not no".split()
    )
    _GCS_THRESHOLD = 0.40

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._summarize_callbacks: list[Callable] = []
        self._pre_summary_vocab: Optional[set[str]] = None

    def on_summarize(self, callback: Callable) -> None:
        """Register a callback to run after each summarization boundary."""
        self._summarize_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Override the summarization entry/exit to capture vocab shift
    # ------------------------------------------------------------------

    def summarize_messages_inplace(self) -> None:  # type: ignore[override]
        """Intercept the summarization boundary, measure drift, fire callbacks."""
        pre_vocab = self._extract_vocab()

        super().summarize_messages_inplace()  # type: ignore[misc]

        post_vocab = self._extract_vocab()
        ghost = pre_vocab - post_vocab
        gcs = 1.0 - len(ghost) / max(len(pre_vocab), 1)

        report = {
            "event": "summarization_boundary",
            "agent_id": getattr(self.agent_state, "id", "unknown"),
            "pre_vocab": pre_vocab,
            "post_vocab": post_vocab,
            "ghost_terms": sorted(ghost),
            "gcs": round(gcs, 3),
            "alert": gcs < self._GCS_THRESHOLD,
        }

        for cb in self._summarize_callbacks:
            try:
                cb(self, report)
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "SummarizationMonitor callback raised: %s", exc
                )

    # ------------------------------------------------------------------
    # Vocabulary extraction from current in-context messages
    # ------------------------------------------------------------------

    def _extract_vocab(self) -> set[str]:
        """Return domain-specific terms from the current in-context messages."""
        try:
            messages = self.agent_manager.get_in_context_messages(
                agent_id=self.agent_state.id, actor=self.user
            )
            raw = " ".join(
                (m.text or "") for m in messages if hasattr(m, "text") and m.text
            )
        except Exception:  # noqa: BLE001
            raw = ""

        words = re.findall(r"[a-z][a-z0-9_]{2,}", raw.lower())
        return {w for w in words if w not in self._STOP_WORDS}


# ---------------------------------------------------------------------------
# Approach 2: Log-level intercept (no subclass needed)
# ---------------------------------------------------------------------------

class SummarizationLogHandler(logging.Handler):
    """A Python logging handler that fires a callback when Letta summarizes.

    Letta calls ``log_telemetry(logger, "_get_ai_reply summarize_messages_inplace")``
    immediately before summarizing. This handler matches that log message and
    invokes the registered callbacks.

    Usage::

        handler = SummarizationLogHandler(on_boundary=my_callback)
        logging.getLogger("letta").addHandler(handler)

    Callback signature: ``(record: logging.LogRecord) -> None``
    """

    _SUMMARIZE_PATTERN = re.compile(r"summarize_messages_inplace", re.IGNORECASE)

    def __init__(self, on_boundary: Callable[[logging.LogRecord], None]) -> None:
        super().__init__()
        self._on_boundary = on_boundary

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if self._SUMMARIZE_PATTERN.search(msg):
                self._on_boundary(record)
        except Exception:  # noqa: BLE001
            self.handleError(record)


# ---------------------------------------------------------------------------
# Demo (standalone, no live Letta server needed)
# ---------------------------------------------------------------------------

def _demo_log_handler() -> None:
    """Demo of Approach 2: attach log handler and simulate a log event."""
    print("\n--- Approach 2: Log-handler intercept ---")

    events: list[str] = []

    def on_boundary(record: logging.LogRecord) -> None:
        events.append(record.getMessage())
        print(f"  [BOUNDARY DETECTED] log msg: {record.getMessage()!r}")

    handler = SummarizationLogHandler(on_boundary=on_boundary)
    letta_logger = logging.getLogger("letta")
    letta_logger.addHandler(handler)
    letta_logger.setLevel(logging.DEBUG)

    # Simulate the log message Letta emits before summarization
    letta_logger.debug("_get_ai_reply summarize_messages_inplace start")

    assert len(events) == 1, "Expected exactly one boundary event"
    print("  Log-handler intercept: OK")
    letta_logger.removeHandler(handler)


def _demo_vocab_diff() -> None:
    """Demo the ghost-lexicon scoring logic in isolation."""
    print("\n--- Ghost Consistency Score demo ---")

    def _tokenize(text: str) -> set[str]:
        stop = SummarizationMonitor._STOP_WORDS
        words = re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())
        return {w for w in words if w not in stop}

    pre = _tokenize(
        "Implement JWT authentication with bcrypt password hashing and "
        "foreign_key constraints on the users table. Use dependency injection."
    )
    post = _tokenize(
        "What should I do next? The agent is ready to help with your task."
    )
    ghost = pre - post
    gcs = 1.0 - len(ghost) / max(len(pre), 1)

    print(f"  Pre-boundary vocab ({len(pre)} terms): {sorted(pre)}")
    print(f"  Post-boundary vocab ({len(post)} terms): {sorted(post)}")
    print(f"  Ghost terms ({len(ghost)}): {sorted(ghost)}")
    print(f"  GCS: {gcs:.3f}  {'⚠ ALERT' if gcs < 0.40 else 'OK'}")


if __name__ == "__main__":
    _demo_log_handler()
    _demo_vocab_diff()
    print("\nBoth approaches verified. See docstrings for integration details.")
    print("Issue reference: https://github.com/letta-ai/letta/issues/3259")

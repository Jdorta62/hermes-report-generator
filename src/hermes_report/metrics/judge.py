"""Judge operations of one question, memoised, with every call logged."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar

from hermes_core.generation import PromptMessage

from hermes_report.metrics.base import (
  JudgeCall,
  MetricError,
  MetricInput,
  describe,
)
from hermes_report.providers import JudgeClient, ProviderError
from hermes_report.record import message_record

__all__ = ["Claim", "ClaimVerdict", "JudgeSession"]

T = TypeVar("T")


@dataclass(frozen=True)
class Claim:
  """A statement extracted from a text, with the ``[n]`` markers it carried.

  Attributes:
    text: The statement.
    markers: The citation markers that were attached to it.
  """

  text: str
  markers: tuple[int, ...] = ()


@dataclass(frozen=True)
class ClaimVerdict:
  """Whether a claim is supported by numbered sources, and by which.

  Attributes:
    claim: The 1-based number of the claim.
    supported: Whether some source supports it.
    sources: The 1-based numbers of the sources that support it.
  """

  claim: int
  supported: bool
  sources: tuple[int, ...]


class JudgeSession:
  """Judge operations for one question, memoised, with every call logged.

  Attributes:
    calls: Every judge call made so far, in order.
  """

  def __init__(self, client: JudgeClient, item: MetricInput) -> None:
    """Start a session of ``client`` over the question ``item``."""
    self._client = client
    self._item = item
    self._memo: dict[str, Any] = {}
    self.calls: list[JudgeCall] = []

  def ask(
    self,
    purpose: str,
    messages: Sequence[PromptMessage],
    parse: Callable[[Any], T],
    key: str | None = None,
  ) -> T:
    """Make one judge call and parse its JSON reply; a failure is memoised too.

    The result is memoised under ``key`` (``purpose`` when omitted): a second
    ask with the same key returns the value, or raises the same error,
    without calling the judge again.

    Raises:
      MetricError: the judge failed, or its reply is not valid JSON or not
        what ``parse`` expects.
    Effects: net
    """
    memo_key = key or purpose
    if memo_key in self._memo:
      cached = self._memo[memo_key]
      if isinstance(cached, MetricError):
        raise cached
      value: T = cached
      return value
    start = perf_counter()
    raw: str | None = None
    try:
      raw = self._client.complete_json(messages)
      parsed: T = parse(json.loads(raw))
    except (ProviderError, ValueError, MetricError) as exc:
      error = exc if isinstance(exc, MetricError) else MetricError(describe(exc))
      self._log(purpose, messages, raw, str(error), start)
      self._memo[memo_key] = error
      if error is exc:
        raise
      raise error from exc
    self._log(purpose, messages, raw, None, start)
    self._memo[memo_key] = parsed
    return parsed

  def calls_since(self, mark: int) -> tuple[JudgeCall, ...]:
    """The calls logged after position ``mark``."""
    return tuple(self.calls[mark:])

  def _log(
    self,
    purpose: str,
    messages: Sequence[PromptMessage],
    raw: str | None,
    error: str | None,
    start: float,
  ) -> None:
    self.calls.append(
      JudgeCall(
        purpose,
        tuple(message_record(m) for m in messages),
        raw,
        error,
        perf_counter() - start,
      )
    )

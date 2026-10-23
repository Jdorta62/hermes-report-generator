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
from hermes_report.providers import JudgeClient
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

    Any failure becomes a ``MetricError``: the client raising, a reply that is
    not text or not valid JSON, or ``parse`` raising whatever it raises. The
    call is logged with its raw reply in every case.

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
    raw: Any = None
    try:
      raw = self._client.complete_json(messages)
      if not isinstance(raw, str):
        raise MetricError(f"El juez devolvió {type(raw).__name__}, no texto")
      parsed: T = parse(json.loads(raw))
    except Exception as exc:
      error = exc if isinstance(exc, MetricError) else MetricError(describe(exc))
      logged = None if raw is None else (raw if isinstance(raw, str) else repr(raw))
      self._log(purpose, messages, logged, str(error), start)
      self._memo[memo_key] = error
      if error is exc:
        raise
      raise error from exc
    self._log(purpose, messages, raw, None, start)
    self._memo[memo_key] = parsed
    return parsed

  def answer_claims(self) -> tuple[Claim, ...]:
    """Claims of the answer and the ``[n]`` markers each carried (memoised).

    Raises:
      MetricError: there is no answer, or the judge failed or replied badly.
    Effects: net
    """
    from hermes_report.metrics.prompts import claims_messages, parse_claims

    if not self._item.answer:
      raise MetricError("No hay respuesta de la que extraer afirmaciones")
    return self.ask(
      "answer_claims",
      claims_messages(self._item.answer, "answer", self._item.question),
      parse_claims,
    )

  def expected_claims(self) -> tuple[Claim, ...]:
    """Claims of the expected answer (memoised).

    Raises:
      MetricError: there is no expected answer, or the judge failed or replied badly.
    Effects: net
    """
    from hermes_report.metrics.prompts import claims_messages, parse_claims

    if not self._item.expected.strip():
      raise MetricError("No hay respuesta esperada de la que extraer afirmaciones")
    return self.ask(
      "expected_claims",
      claims_messages(self._item.expected, "expected", self._item.question),
      parse_claims,
    )

  def verify(
    self, claims: Sequence[Claim], sources: Sequence[str], key: str
  ) -> tuple[ClaimVerdict, ...]:
    """Check ``claims`` against numbered ``sources``; memoised by ``key``.

    Without claims there is nothing to ask and the result is empty; without
    sources every claim is unsupported. Neither case calls the judge.

    Raises:
      MetricError: the judge failed or replied badly.
    Effects: net
    """
    from hermes_report.metrics.prompts import parse_verdicts, verify_messages

    if not claims:
      return ()
    if not sources:
      return tuple(ClaimVerdict(n, False, ()) for n in range(1, len(claims) + 1))
    return self.ask(
      f"verify:{key}",
      verify_messages(claims, sources),
      lambda data: parse_verdicts(data, len(claims), len(sources)),
      key=f"verify:{key}",
    )

  def chunk_relevance(self) -> tuple[bool, ...]:
    """Relevance of each retrieved chunk to the question and expected answer.

    One judge call covers every chunk; no chunks means no call.

    Raises:
      MetricError: the judge failed or replied badly.
    Effects: net
    """
    from hermes_report.metrics.prompts import parse_relevance, relevance_messages

    texts = [chunk.text for chunk in self._item.retrieved]
    if not texts:
      return ()
    return self.ask(
      "chunk_relevance",
      relevance_messages(self._item.question, self._item.expected, texts),
      lambda data: parse_relevance(data, len(texts)),
    )

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

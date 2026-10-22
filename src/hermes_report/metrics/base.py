"""Metric core: the neutral input, the result, the services and the Metric contract."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Protocol

from hermes_core.generation import PromptMessage
from hermes_core.protocols import Embedder
from hermes_enrich.tokens import TokenCounter

from hermes_report.record import JudgeCallRecord, MessageRecord, MetricResultRecord

if TYPE_CHECKING:
  from hermes_report.metrics.judge import JudgeSession

__all__ = [
  "INPUT_FIELDS",
  "METRIC_STATUSES",
  "Citation",
  "JudgeCall",
  "Metric",
  "MetricError",
  "MetricInput",
  "MetricResult",
  "MetricServices",
  "RankedChunk",
  "describe",
]

METRIC_STATUSES: Final = ("ok", "not_applicable", "skipped", "error")
INPUT_FIELDS: Final = (
  "answer",
  "retrieved",
  "context",
  "documents",
  "evidence",
  "expected",
  "citations",
)


class MetricError(Exception):
  """A metric or a judge operation cannot produce a value (message in Spanish)."""


def describe(exc: BaseException) -> str:
  """The exception as ``Type: message``, never empty."""
  text = str(exc)
  return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


@dataclass(frozen=True)
class RankedChunk:
  """A retrieved chunk in rank order, as the metrics see it.

  Attributes:
    chunk_id: Identifier of the chunk.
    text: The chunk text.
    source: The corpus document it comes from; ``None`` when unknown.
    score: The retrieval score.
  """

  chunk_id: str
  text: str
  source: str | None
  score: float


@dataclass(frozen=True)
class Citation:
  """One citation of the answer: its marker and the context piece it points at.

  Attributes:
    marker: The ``n`` of the ``[n]`` marker in the answer.
    piece: The 1-based context piece it points at; 0 when outside the context.
  """

  marker: int
  piece: int


@dataclass(frozen=True)
class MetricInput:
  """Everything a metric may read about one answered question.

  Attributes:
    question_id: Identifier of the question.
    question: The user question.
    expected: The right answer.
    refusal: Whether the right outcome is to say the sources are not enough.
    answer: The generated answer; ``None`` when generation failed.
    refused: Whether the answer is the insufficient-sources reply; ``None``
      without an answer.
    retrieved: The retrieved chunks, rank 1 first.
    context: The final context pieces, in prompt order (source [1] first).
    documents: Corpus files that hold the answer.
    evidence: Corpus phrases that must be in the context.
    seconds: Seconds per phase, keys ``retrieval``, ``enrich``, ``generation``.
    prompt: The messages of the last generation call.
    replies: The texts of the generation replies.
    citations: The explicit citations; ``None`` means "derive them from the
      ``[n]`` markers".
  """

  question_id: str
  question: str
  expected: str
  refusal: bool
  answer: str | None
  refused: bool | None
  retrieved: tuple[RankedChunk, ...]
  context: tuple[str, ...]
  documents: tuple[str, ...] = ()
  evidence: tuple[str, ...] = ()
  seconds: Mapping[str, float] = field(default_factory=dict)
  prompt: tuple[PromptMessage, ...] = ()
  replies: tuple[str, ...] = ()
  citations: tuple[Citation, ...] | None = None

  def has(self, field: str) -> bool:
    """Whether ``field`` holds usable data (non-empty, not None).

    Raises:
      ValueError: if ``field`` is not one of ``INPUT_FIELDS``.
    """
    if field not in INPUT_FIELDS:
      raise ValueError(f"campo de entrada desconocido: {field}")
    value = getattr(self, field)
    if value is None:
      return False
    if isinstance(value, str):
      return bool(value.strip())
    return len(value) > 0


@dataclass(frozen=True)
class JudgeCall:
  """One judge call made by a metric operation.

  Attributes:
    purpose: What the call was for, like ``answer_claims``.
    messages: The messages sent.
    raw: The raw reply text; ``None`` when the client failed.
    error: What went wrong; ``None`` when the reply was parsed.
    seconds: How long the call took.
  """

  purpose: str
  messages: tuple[MessageRecord, ...]
  raw: str | None
  error: str | None
  seconds: float


@dataclass(frozen=True)
class MetricServices:
  """The collaborators a metric may use; ``None`` when not available.

  Attributes:
    judge: The judge operations of the question.
    embedder: The embedder of the run.
    counter: The token counter.
  """

  judge: JudgeSession | None
  embedder: Embedder | None
  counter: TokenCounter


def _plain(value: Any) -> Any:
  """Data made of dicts, lists and scalars only, so it survives a JSON round trip."""
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
  if isinstance(value, Mapping):
    return {str(key): _plain(item) for key, item in value.items()}
  if isinstance(value, list | tuple | set | frozenset):
    return [_plain(item) for item in value]
  return value


@dataclass(frozen=True)
class MetricResult:
  """The outcome of one metric family for one question.

  Attributes:
    name: The metric family.
    status: One of ``METRIC_STATUSES``.
    scores: Score name to value; ``None`` when the score has no value.
    details: Extra data for the report (any JSON-friendly values).
    calls: The judge calls the metric made.
    error: What went wrong; only with status ``error``.
    seconds: How long the metric took.
  """

  name: str
  status: str
  scores: dict[str, float | None]
  details: dict[str, Any] = field(default_factory=dict)
  calls: tuple[JudgeCall, ...] = ()
  error: str | None = None
  seconds: float = 0.0

  @classmethod
  def not_applicable(cls, name: str, missing: Collection[str]) -> MetricResult:
    """Build the result of a metric whose input fields are missing."""
    return cls(name, "not_applicable", {}, {"missing": sorted(missing)})

  @classmethod
  def skipped(cls, name: str, reason: str) -> MetricResult:
    """Build the result of a metric whose service is not available."""
    return cls(name, "skipped", {}, {"reason": reason})

  @classmethod
  def failed(cls, name: str, error: str, calls: Sequence[JudgeCall]) -> MetricResult:
    """Build the result of a metric that raised, with the calls it made."""
    return cls(name, "error", {}, {}, tuple(calls), error)

  def to_record(self) -> MetricResultRecord:
    """Record the result as plain data that equals its JSON round trip."""
    calls: list[JudgeCallRecord] = [
      {
        "purpose": call.purpose,
        "messages": list(call.messages),
        "raw": call.raw,
        "error": call.error,
        "seconds": call.seconds,
      }
      for call in self.calls
    ]
    return {
      "name": self.name,
      "status": self.status,
      "scores": dict(self.scores),
      "details": _plain(self.details),
      "calls": calls,
      "error": self.error,
      "seconds": self.seconds,
    }


class Metric(Protocol):
  """One metric family: a stable name, its needs and its computation.

  Registry: metric

  Attributes:
    name: The stable name of the family, the key of the registry.
    default_enabled: Whether it runs unless excluded.
    scores: The score names it emits, in display order.
    needs: Services it needs, a subset of ``judge`` and ``embedder``.
    requires: Input fields it needs, a subset of ``INPUT_FIELDS``.
  """

  name: str
  default_enabled: bool
  scores: tuple[str, ...]
  needs: frozenset[str]
  requires: frozenset[str]

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Compute the scores of ``item``.

    Raises:
      MetricError: the scores cannot be produced.
    """
    ...

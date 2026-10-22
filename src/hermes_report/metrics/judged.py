"""Judge-based metric families: the RAG triad, answer correctness and citations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from hermes_report.metrics.base import (
  Citation,
  MetricError,
  MetricInput,
  MetricResult,
  MetricServices,
)
from hermes_report.metrics.judge import Claim, ClaimVerdict, JudgeSession
from hermes_report.metrics.prompts import parse_questions, questions_messages

__all__ = [
  "AnswerCorrectness",
  "AnswerRelevance",
  "Citations",
  "ContextPrecision",
  "ContextRecall",
  "Faithfulness",
  "average_precision",
  "citation_targets",
  "cosine",
]

_N_QUESTIONS = 3
_CONTEXT = "answer_vs_context"


def _judge(services: MetricServices) -> JudgeSession:
  if services.judge is None:
    raise MetricError("Esta métrica necesita un juez")
  return services.judge


def _result(
  name: str,
  scores: dict[str, float | None],
  details: dict[str, Any],
  judge: JudgeSession,
  mark: int,
) -> MetricResult:
  """An ``ok`` result with the judge calls made since ``mark``."""
  return MetricResult(name, "ok", scores, details, judge.calls_since(mark))


def _ratio(count: int, total: int) -> float | None:
  return count / total if total else None


def _supported(verdicts: Sequence[ClaimVerdict]) -> int:
  return sum(1 for verdict in verdicts if verdict.supported)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
  """Cosine similarity of two vectors; 0.0 when either has no length.

  Raises:
    MetricError: the vectors differ in length.
  """
  if len(a) != len(b):
    raise MetricError(f"Los vectores difieren en longitud: {len(a)} y {len(b)}")
  norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
  if norm == 0.0:
    return 0.0
  return sum(x * y for x, y in zip(a, b, strict=True)) / norm


def average_precision(relevant: Sequence[bool]) -> float:
  """Average precision of ranked relevance verdicts.

  ``sum(precision@r * rel_r) / max(1, sum(rel_r))``; 0.0 when nothing is relevant.
  """
  hits = 0
  total = 0.0
  for rank, flag in enumerate(relevant, 1):
    if flag:
      hits += 1
      total += hits / rank
  return total / max(1, hits)


def _piece(item: MetricInput, marker: int) -> int:
  """The 1-based context piece ``marker`` points at; 0 when it points nowhere."""
  if item.citations is not None:
    return next((c.piece for c in item.citations if c.marker == marker), 0)
  return marker if 1 <= marker <= len(item.context) else 0


def _per_claim(
  item: MetricInput, claims: Sequence[Claim]
) -> list[tuple[Citation, ...]]:
  return [
    tuple(Citation(m, _piece(item, m)) for m in dict.fromkeys(claim.markers))
    for claim in claims
  ]


def citation_targets(
  item: MetricInput, claims: Sequence[Claim]
) -> tuple[Citation, ...]:
  """The citations of ``claims``, each with the context piece it points at.

  One per distinct marker of each claim, in claim order. The marker ``[n]`` is
  piece ``n`` of the context, counted from 1, as the generator numbers its
  sources; ``item.citations`` overrides that mapping when given. A marker that
  points outside the context keeps piece 0, which never supports a claim.
  """
  return tuple(c for per_claim in _per_claim(item, claims) for c in per_claim)


class Faithfulness:
  """Share of the answer claims that the final context supports.

  Registry: metric[faithfulness]
  """

  name: str = "faithfulness"
  default_enabled: bool = True
  scores: tuple[str, ...] = ("faithfulness",)
  needs: frozenset[str] = frozenset({"judge"})
  requires: frozenset[str] = frozenset({"answer", "context"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Verify the answer claims against the context; 1.0 when it has none."""
    if item.refused:
      return MetricResult.not_applicable(self.name, ["refusal"])
    judge = _judge(services)
    mark = len(judge.calls)
    claims = judge.answer_claims()
    if not claims:
      return _result(self.name, {"faithfulness": 1.0}, {"claims": []}, judge, mark)
    verdicts = judge.verify(claims, item.context, _CONTEXT)
    score = _supported(verdicts) / len(claims)
    return _result(
      self.name,
      {"faithfulness": score},
      {"claims": claims, "verdicts": verdicts},
      judge,
      mark,
    )


class AnswerRelevance:
  """How close the questions the answer would resolve are to the real question.

  Registry: metric[answer_relevance]
  """

  name: str = "answer_relevance"
  default_enabled: bool = True
  scores: tuple[str, ...] = ("answer_relevance",)
  needs: frozenset[str] = frozenset({"judge", "embedder"})
  requires: frozenset[str] = frozenset({"answer"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Mean cosine of generated questions against the question; 0 if noncommittal."""
    if item.refused and item.refusal:
      return MetricResult.not_applicable(self.name, ["refusal"])
    if item.refused:
      return MetricResult(self.name, "ok", {"answer_relevance": 0.0}, {"refused": True})
    judge = _judge(services)
    embedder = services.embedder
    if embedder is None:
      raise MetricError("Esta métrica necesita un embedder")
    mark = len(judge.calls)
    questions, noncommittal = judge.ask(
      "questions",
      questions_messages(item.answer or "", _N_QUESTIONS),
      parse_questions,
    )
    if noncommittal or not questions:
      return _result(
        self.name,
        {"answer_relevance": 0.0},
        {"noncommittal": noncommittal, "questions": list(questions)},
        judge,
        mark,
      )
    target = embedder.get_query_embedding(item.question)
    similarities = [
      cosine(target, embedder.get_query_embedding(question)) for question in questions
    ]
    score = max(0.0, sum(similarities) / len(similarities))
    return _result(
      self.name,
      {"answer_relevance": score},
      {"questions": list(questions), "similarities": similarities},
      judge,
      mark,
    )


class ContextPrecision:
  """Whether the relevant retrieved chunks rank above the irrelevant ones.

  Registry: metric[context_precision]
  """

  name: str = "context_precision"
  default_enabled: bool = True
  scores: tuple[str, ...] = ("context_precision",)
  needs: frozenset[str] = frozenset({"judge"})
  requires: frozenset[str] = frozenset({"retrieved", "expected"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Average precision over the relevance verdicts of the retrieved chunks."""
    judge = _judge(services)
    mark = len(judge.calls)
    relevant = judge.chunk_relevance()
    return _result(
      self.name,
      {"context_precision": average_precision(relevant)},
      {"relevant": list(relevant)},
      judge,
      mark,
    )


class ContextRecall:
  """Share of the expected-answer claims that the final context supports.

  Registry: metric[context_recall]
  """

  name: str = "context_recall"
  default_enabled: bool = True
  scores: tuple[str, ...] = ("context_recall",)
  needs: frozenset[str] = frozenset({"judge"})
  requires: frozenset[str] = frozenset({"expected", "context"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Verify the expected claims against the context; None when it has none."""
    judge = _judge(services)
    mark = len(judge.calls)
    claims = judge.expected_claims()
    if not claims:
      return _result(self.name, {"context_recall": None}, {"claims": []}, judge, mark)
    verdicts = judge.verify(claims, item.context, "expected_vs_context")
    return _result(
      self.name,
      {"context_recall": _supported(verdicts) / len(claims)},
      {"claims": claims, "verdicts": verdicts},
      judge,
      mark,
    )


class AnswerCorrectness:
  """Precision, recall and F1 of the answer claims against the expected answer.

  Registry: metric[answer_correctness]
  """

  name: str = "answer_correctness"
  default_enabled: bool = True
  scores: tuple[str, ...] = (
    "correctness_precision",
    "correctness_recall",
    "correctness_f1",
  )
  needs: frozenset[str] = frozenset({"judge"})
  requires: frozenset[str] = frozenset({"answer", "expected"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Compare the claims of the answer and of the expected answer both ways."""
    judge = _judge(services)
    mark = len(judge.calls)
    answer = judge.answer_claims()
    expected = judge.expected_claims()
    if not expected:
      scores: dict[str, float | None] = {
        "correctness_precision": None,
        "correctness_recall": None,
        "correctness_f1": None,
      }
      return _result(self.name, scores, {"no_expected_claims": True}, judge, mark)
    forward = judge.verify(answer, [item.expected], "answer_vs_expected")
    backward = judge.verify(expected, [item.answer or ""], "expected_vs_answer")
    precision = _ratio(_supported(forward), len(answer))
    recall = _supported(backward) / len(expected)
    both = (precision or 0.0) + recall
    f1 = 2 * (precision or 0.0) * recall / both if both else 0.0
    return _result(
      self.name,
      {
        "correctness_precision": precision,
        "correctness_recall": recall,
        "correctness_f1": f1,
      },
      {
        "answer_claims": answer,
        "expected_claims": expected,
        "answer_vs_expected": forward,
        "expected_vs_answer": backward,
      },
      judge,
      mark,
    )


class Citations:
  """Whether the answer cites the context pieces that support its claims.

  Registry: metric[citations]
  """

  name: str = "citations"
  default_enabled: bool = True
  scores: tuple[str, ...] = ("citation_precision", "citation_recall")
  needs: frozenset[str] = frozenset({"judge"})
  requires: frozenset[str] = frozenset({"answer", "context"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Precision of the cited pieces and recall of the supported claims."""
    judge = _judge(services)
    mark = len(judge.calls)
    claims = judge.answer_claims()
    support = judge.verify(claims, item.context, _CONTEXT)
    per_claim = _per_claim(item, claims)
    cited = good = covered = 0
    supported = 0
    for verdict, citations in zip(support, per_claim, strict=True):
      cited += len(citations)
      hits = [c for c in citations if verdict.supported and c.piece in verdict.sources]
      good += len(hits)
      if verdict.supported:
        supported += 1
        covered += 1 if hits else 0
    return _result(
      self.name,
      {
        "citation_precision": _ratio(good, cited),
        "citation_recall": _ratio(covered, supported),
      },
      {
        "claims": claims,
        "citations": [c for per in per_claim for c in per],
        "support": support,
      },
      judge,
      mark,
    )

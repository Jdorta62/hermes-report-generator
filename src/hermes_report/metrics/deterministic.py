"""Deterministic metric families: document and evidence retrieval, refusal and cost."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from hermes_report.metrics.base import MetricInput, MetricResult, MetricServices
from hermes_report.questions import Question

__all__ = ["Cost", "DocRetrieval", "EvidenceRetrieval", "Refusal"]

_JOIN = "\n"


def _first_rank(texts: Sequence[str], question: Question) -> int | None:
  """The smallest rank r whose chunks 1..r, joined, make ``question`` reachable."""
  joined = ""
  for rank, text in enumerate(texts, 1):
    joined = text if rank == 1 else f"{joined}{_JOIN}{text}"
    if question.reachable(joined):
      return rank
  return None


def _hit(value: int | None) -> float:
  return 0.0 if value is None else 1.0


def _reciprocal(rank: int | None) -> float:
  return 0.0 if rank is None else 1.0 / rank


class DocRetrieval:
  """Whether the retrieved chunks come from the documents that hold the answer.

  Registry: metric[doc_retrieval]
  """

  name: str = "doc_retrieval"
  default_enabled: bool = True
  scores: tuple[str, ...] = ("doc_hit", "doc_recall", "doc_mrr")
  needs: frozenset[str] = frozenset()
  requires: frozenset[str] = frozenset({"documents", "retrieved"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Score the retrieved sources against the expected documents."""
    sources = [chunk.source for chunk in item.retrieved]
    if all(source is None for source in sources):
      return MetricResult.not_applicable(self.name, ["source"])
    expected = tuple(dict.fromkeys(item.documents))
    rank = next((r for r, s in enumerate(sources, 1) if s in expected), None)
    found = [doc for doc in expected if doc in sources]
    return MetricResult(
      self.name,
      "ok",
      {
        "doc_hit": _hit(rank),
        "doc_recall": len(found) / len(expected),
        "doc_mrr": _reciprocal(rank),
      },
      {"expected": list(expected), "found": found, "first_rank": rank},
    )


class EvidenceRetrieval:
  """Whether the evidence phrases are in the retrieved chunks and in the context.

  Registry: metric[evidence_retrieval]
  """

  name: str = "evidence_retrieval"
  default_enabled: bool = True
  scores: tuple[str, ...] = (
    "evidence_hit",
    "evidence_recall",
    "evidence_mrr",
    "evidence_recall_context",
    "budget_loss",
  )
  needs: frozenset[str] = frozenset()
  requires: frozenset[str] = frozenset({"evidence", "retrieved"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Score the evidence phrases by rank, split across chunks as the rules do."""
    texts = [chunk.text for chunk in item.retrieved]
    phrases = tuple(dict.fromkeys(item.evidence))
    base = Question(item.question_id, item.question, item.expected)
    found_at: dict[str, int | None] = {}
    for phrase in phrases:
      found_at[phrase] = _first_rank(texts, replace(base, evidence=(phrase,)))
    recall = sum(rank is not None for rank in found_at.values()) / len(phrases)
    all_rank = _first_rank(texts, replace(base, evidence=phrases))
    context = _JOIN.join(item.context)
    in_context = sum(
      replace(base, evidence=(phrase,)).reachable(context) is True for phrase in phrases
    ) / len(phrases)
    return MetricResult(
      self.name,
      "ok",
      {
        "evidence_hit": _hit(all_rank),
        "evidence_recall": recall,
        "evidence_mrr": _reciprocal(all_rank),
        "evidence_recall_context": in_context,
        "budget_loss": max(0.0, recall - in_context),
      },
      {"found_at": found_at},
    )


class Refusal:
  """False refusals and invented answers, by the expected outcome of the question.

  Registry: metric[refusal]
  """

  name: str = "refusal"
  default_enabled: bool = True
  scores: tuple[str, ...] = ("false_refusal", "invented_answer")
  needs: frozenset[str] = frozenset()
  requires: frozenset[str] = frozenset({"answer"})

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Score a refusal on an answerable question, or an answer to an unanswerable one."""
    false_refusal: float | None = None
    invented_answer: float | None = None
    if item.refusal:
      if item.refused is not None:
        invented_answer = 0.0 if item.refused else 1.0
    elif item.refused is False:
      false_refusal = 0.0
    elif item.refused:
      probe = Question(
        item.question_id, item.question, item.expected, evidence=item.evidence
      )
      if probe.reachable(_JOIN.join(item.context)):
        false_refusal = 1.0
    return MetricResult(
      self.name,
      "ok",
      {"false_refusal": false_refusal, "invented_answer": invented_answer},
      {"refused": item.refused, "refusal": item.refusal},
    )


class Cost:
  """Seconds per phase and tokens of the context, the prompt and the replies.

  Registry: metric[cost]
  """

  name: str = "cost"
  default_enabled: bool = True
  scores: tuple[str, ...] = (
    "seconds_retrieval",
    "seconds_enrich",
    "seconds_generation",
    "context_tokens",
    "prompt_tokens",
    "reply_tokens",
  )
  needs: frozenset[str] = frozenset()
  requires: frozenset[str] = frozenset()

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    """Read the seconds of each phase and count tokens; a missing phase is None."""
    count = services.counter.count
    context = count(_JOIN.join(item.context)) if item.context else None
    prompt = (
      sum(count(message.content) for message in item.prompt) if item.prompt else None
    )
    replies = sum(count(reply) for reply in item.replies) if item.replies else None
    return MetricResult(
      self.name,
      "ok",
      {
        "seconds_retrieval": item.seconds.get("retrieval"),
        "seconds_enrich": item.seconds.get("enrich"),
        "seconds_generation": item.seconds.get("generation"),
        "context_tokens": None if context is None else float(context),
        "prompt_tokens": None if prompt is None else float(prompt),
        "reply_tokens": None if replies is None else float(replies),
      },
      {"counter": services.counter.name},
    )

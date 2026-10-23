"""Adapter from a run's question record to the neutral metric input."""

from __future__ import annotations

from typing import Any, cast

from hermes_core.generation import MessageRole, PromptMessage

from hermes_report.evaluate import is_refused
from hermes_report.metrics.base import MetricInput, RankedChunk
from hermes_report.questions import Question
from hermes_report.record import GenerationRecord, QuestionRecord

__all__ = ["metric_input"]


def _source(metadata: dict[str, Any]) -> str | None:
  source = metadata.get("source")
  return source if isinstance(source, str) else None


def _prompt(generation: GenerationRecord | None) -> tuple[PromptMessage, ...]:
  """The messages of the last generation call; empty without calls."""
  if generation is None or not generation["llm_calls"]:
    return ()
  messages = generation["llm_calls"][-1]["messages"]
  return tuple(
    PromptMessage(role=cast("MessageRole", m["role"]), content=m["content"])
    for m in messages
  )


def _replies(generation: GenerationRecord | None) -> tuple[str, ...]:
  """The texts of the generation replies that arrived."""
  if generation is None:
    return ()
  return tuple(
    call["reply"]["text"] for call in generation["llm_calls"] if call["reply"]
  )


def metric_input(question: Question, record: QuestionRecord) -> MetricInput:
  """Build the neutral metric input from a run's record.

  ``refused`` uses the same test as the rule verdict. A question whose
  pipeline raised still gets an input, with whatever phases finished.
  ``citations`` is ``None``: the generation result has no claim-to-piece link,
  so the metrics derive the citations from the ``[n]`` markers.
  """
  retrieval = record["retrieval"]
  enrich = record["enrich"]
  generation = record["generation"]
  context = enrich["context"] if enrich else None
  result = generation["result"] if generation else None
  seconds: dict[str, float] = {}
  if retrieval:
    seconds["retrieval"] = retrieval["seconds"]
  if enrich:
    seconds["enrich"] = enrich["seconds"]
  if generation:
    seconds["generation"] = generation["seconds"]
  return MetricInput(
    question_id=question.id,
    question=question.question,
    expected=question.expected,
    refusal=question.refusal,
    answer=result["answer"] if result else None,
    refused=is_refused(result) if result else None,
    retrieved=tuple(
      RankedChunk(c["chunk_id"], c["text"], _source(c["metadata"]), c["score"])
      for c in (retrieval["chunks"] if retrieval else ())
    ),
    context=tuple(piece["text"] for piece in context["pieces"]) if context else (),
    documents=question.documents,
    evidence=question.evidence,
    seconds=seconds,
    prompt=_prompt(generation),
    replies=_replies(generation),
    citations=None,
  )

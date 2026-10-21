"""Grading of one answer: the rule verdict and the LLM judge, both over records."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Final

from hermes_core.generation import PromptMessage

from hermes_report.providers import JudgeClient, ProviderError
from hermes_report.questions import Question
from hermes_report.record import (
  JudgeRecord,
  QuestionRecord,
  RetrievedChunkRecord,
  RuleRecord,
  message_record,
)

__all__ = [
  "JUDGE_VERDICTS",
  "RULE_VERDICTS",
  "judge",
  "judge_messages",
  "rule_verdict",
]

RULE_VERDICTS: Final = (
  "error",
  "correct_refusal",
  "invented_answer",
  "correct",
  "correct_without_evidence",
  "budget_failure",
  "retrieval_failure",
  "generation_refused",
  "generation_wrong",
  "unrated",
)
JUDGE_VERDICTS: Final = ("correct", "partial", "incorrect")

_REFUSED_STATUSES: Final = ("insufficient", "no_context")
_FENCE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\n?(.*?)\n?```$", re.DOTALL)

_JUDGE_SYSTEM: Final = (
  "Eres un evaluador estricto de un sistema de preguntas y respuestas. "
  "Comparas el significado de la respuesta candidata con la respuesta esperada, "
  "no su redacción: dos textos distintos pueden decir lo mismo.\n"
  "Responde únicamente con un objeto JSON de esta forma, sin texto alrededor: "
  '{"verdict": "correct" | "partial" | "incorrect", '
  '"reason": "<una o dos frases en español>"}.\n'
  "Casos:\n"
  '- "correct": la respuesta candidata dice lo mismo que la esperada.\n'
  '- "partial": la respuesta es correcta pero incompleta, o es correcta con un '
  "detalle erróneo.\n"
  '- "incorrect": la respuesta contradice a la esperada, no responde a la '
  "pregunta o inventa datos.\n"
  "Si la pregunta es de las que el resultado correcto es negarse a responder, "
  '"correct" significa que la respuesta dice que las fuentes no son suficientes; '
  'si en cambio da una respuesta concreta, es "incorrect".'
)


def _chunks_of(record: QuestionRecord) -> list[RetrievedChunkRecord]:
  retrieval = record.get("retrieval")
  return list(retrieval["chunks"]) if retrieval else []


def _pieces_of(record: QuestionRecord) -> list[str]:
  enrich = record.get("enrich")
  context = enrich["context"] if enrich else None
  return [piece["text"] for piece in context["pieces"]] if context else []


def _result_of(record: QuestionRecord) -> dict[str, Any] | None:
  generation = record.get("generation")
  return generation["result"] if generation else None


def rule_verdict(question: Question, record: QuestionRecord) -> RuleRecord:
  """Grade one answer by rules, from the record alone.

  The verdict is ``error`` (every other field ``None``) when the question
  failed or has no generation result.
  """
  result = _result_of(record)
  if record.get("error") or result is None:
    return {
      "verdict": "error",
      "in_retrieval": None,
      "in_context": None,
      "in_cited": None,
      "answer_ok": None,
      "retrieved_rank": None,
      "expected_docs_retrieved": None,
    }
  chunks = _chunks_of(record)
  pieces = _pieces_of(record)
  cited_numbers = {citation["number"] for citation in result.get("citations", ())}
  cited = [text for n, text in enumerate(pieces, 1) if n in cited_numbers]
  in_retrieval = question.reachable("\n".join(c["text"] for c in chunks))
  in_context = question.reachable("\n".join(pieces))
  in_cited = question.reachable("\n".join(cited))
  rank = next(
    (i for i, c in enumerate(chunks, 1) if question.reachable(c["text"])), None
  )
  docs: bool | None = None
  if question.documents:
    docs = any(c["metadata"].get("source") in question.documents for c in chunks)
  refused = result["status"] in _REFUSED_STATUSES
  answer_ok: bool | None
  verdict: str
  if question.refusal:
    answer_ok = refused
    verdict = "correct_refusal" if refused else "invented_answer"
  else:
    if not question.keywords:
      answer_ok = None
    else:
      answer_ok = False if refused else question.matches(result["answer"])
    if answer_ok:
      verdict = "correct" if in_context is not False else "correct_without_evidence"
    elif answer_ok is None and in_context is not False:
      verdict = "unrated"
    elif in_context is False:
      verdict = "budget_failure" if in_retrieval else "retrieval_failure"
    elif refused:
      verdict = "generation_refused"
    else:
      verdict = "generation_wrong"
  return {
    "verdict": verdict,
    "in_retrieval": in_retrieval,
    "in_context": in_context,
    "in_cited": in_cited,
    "answer_ok": answer_ok,
    "retrieved_rank": rank,
    "expected_docs_retrieved": docs,
  }


def judge_messages(
  question: Question, answer: str, status: str
) -> tuple[PromptMessage, ...]:
  """Return the system and user messages (in Spanish) that ask the judge to grade."""
  expected_refusal = "sí" if question.refusal else "no"
  user = (
    f"Pregunta: {question.question}\n"
    f"Respuesta esperada: {question.expected}\n"
    f"¿Se espera que el sistema se niegue a responder?: {expected_refusal}\n"
    f"Estado del generador: {status}\n"
    f"Respuesta candidata: {answer}"
  )
  return (
    PromptMessage(role="system", content=_JUDGE_SYSTEM),
    PromptMessage(role="user", content=user),
  )


def _parse_verdict(raw: str) -> tuple[str, str]:
  """Return ``(verdict, reason)`` from the reply, or raise ``ValueError``."""
  text = raw.strip()
  fenced = _FENCE.match(text)
  if fenced:
    text = fenced.group(1).strip()
  try:
    data: object = json.loads(text)
  except (ValueError, RecursionError) as exc:
    raise ValueError(f"no es JSON ({exc})") from None
  if not isinstance(data, dict):
    raise ValueError("no es un objeto JSON")
  verdict = data.get("verdict")
  if not isinstance(verdict, str) or verdict not in JUDGE_VERDICTS:
    raise ValueError(f"verdict debe ser uno de {', '.join(JUDGE_VERDICTS)}")
  reason = data.get("reason")
  if not isinstance(reason, str) or not reason.strip():
    raise ValueError("reason debe ser un texto no vacío")
  return verdict, reason.strip()


def judge(
  question: Question, record: QuestionRecord, client: JudgeClient
) -> JudgeRecord | None:
  """Grade the answer of ``record`` with the LLM judge; ``None`` without a result.

  A reply that is not valid is kept as ``raw`` with ``verdict=None`` and an
  ``error``; a ``ProviderError`` is kept as ``error`` with ``raw=None``.
  """
  result = _result_of(record)
  if result is None:
    return None
  messages = judge_messages(question, result["answer"], result["status"])
  verdict: str | None = None
  reason: str | None = None
  raw: str | None = None
  error: str | None = None
  start = time.perf_counter()
  try:
    raw = client.complete_json(messages)
  except ProviderError as exc:
    error = str(exc)
  seconds = time.perf_counter() - start
  if raw is not None:
    try:
      verdict, reason = _parse_verdict(raw)
    except ValueError as exc:
      error = f"respuesta del juez no válida: {exc}"
  return {
    "verdict": verdict,
    "reason": reason,
    "model": client.model,
    "seconds": seconds,
    "messages": [message_record(m) for m in messages],
    "raw": raw,
    "error": error,
  }

"""Question ids and the score tiles that the run report and the comparison share."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from hermes_report.record import QuestionRecord
from hermes_report.render.html import RULE_TONES, seconds
from hermes_report.render.question import disagree

__all__ = ["question_id", "score_tiles"]

_BUDGET_FAILURES: Final = ("budget_failure",)
_RETRIEVAL_FAILURES: Final = ("retrieval_failure",)
_GENERATION_FAILURES: Final = ("generation_refused", "generation_wrong")


def question_id(item: QuestionRecord, number: int) -> str:
  """The id of a question; ``q<number>`` (two digits) when its spec has none."""
  return str(item["spec"].get("id") or f"q{number:02d}")


def score_tiles(questions: Sequence[QuestionRecord]) -> tuple[tuple[str, str], ...]:
  """The ``(label, value)`` tiles of the summary for ``questions``, in display order.

  Right answers by rules and by judge, failures by type, rule/judge
  disagreements and the mean seconds per question; a dash where nothing was
  rated.
  """
  codes = [item["rules"]["verdict"] if item["rules"] else None for item in questions]
  rated = [code for code in codes if code and RULE_TONES.get(code, "info") != "info"]
  good = sum(1 for code in rated if RULE_TONES.get(code) == "good")
  judged = [item["judge"] for item in questions if item["judge"] is not None]
  judge_good = sum(1 for item in judged if item["verdict"] == "correct")
  mean = (
    sum(item["seconds"] for item in questions) / len(questions) if questions else 0.0
  )
  return (
    ("Correctas por reglas", f"{good}/{len(rated)}" if rated else "—"),
    ("Correctas por juez", f"{judge_good}/{len(judged)}" if judged else "—"),
    ("Fallos de recuperación", _count(codes, _RETRIEVAL_FAILURES)),
    ("Fallos de presupuesto", _count(codes, _BUDGET_FAILURES)),
    ("Fallos de generación", _count(codes, _GENERATION_FAILURES)),
    ("Reglas y juez discrepan", str(sum(1 for item in questions if disagree(item)))),
    ("Tiempo medio por pregunta", seconds(mean)),
  )


def _count(codes: Sequence[str | None], wanted: Sequence[str]) -> str:
  return str(sum(1 for code in codes if code in wanted))

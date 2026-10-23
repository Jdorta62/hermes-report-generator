"""Question ids and the score tiles that the run report and the comparison share."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from hermes_report.metrics.selection import METRICS
from hermes_report.record import MetricResultRecord, QuestionRecord, RunRecord
from hermes_report.render.html import NO_DATA, RULE_TONES, esc, metric_label, seconds
from hermes_report.render.question import disagree

__all__ = [
  "format_score",
  "metric_tiles",
  "primary_scores",
  "question_id",
  "score_cell",
  "score_tiles",
]

_BUDGET_FAILURES: Final = ("budget_failure",)
_RETRIEVAL_FAILURES: Final = ("retrieval_failure",)
_GENERATION_FAILURES: Final = ("generation_refused", "generation_wrong")
_FAMILY_OF: Final = {
  score: name for name, metric in METRICS.items() for score in metric.scores
}


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


def format_score(value: float) -> str:
  """A score value with two decimals, as ``"0.83"``."""
  return f"{value:.2f}"


def metric_tiles(record: RunRecord) -> tuple[tuple[str, str], ...]:
  """The ``(label, value)`` tile of every score of ``record`` that has a mean.

  The value is the mean with two decimals, then ``n`` (the questions it
  covers) and the questions where the metric failed, when there are any. The
  labels are Spanish and carry ``↓`` where lower is better. A record without
  metrics (schema 1, or metrics off) gives ``()``.
  """
  metrics = record.get("metrics")
  if not metrics:
    return ()
  tiles: list[tuple[str, str]] = []
  for score in metrics["scores"]:
    entry = metrics["summary"].get(score)
    if entry is None or entry["mean"] is None:
      continue
    parts = [f"n={entry['n']}"]
    if entry["errors"]:
      noun = "error" if entry["errors"] == 1 else "errores"
      parts.append(f"{entry['errors']} {noun}")
    tiles.append(
      (metric_label(score), f"{format_score(entry['mean'])} ({', '.join(parts)})")
    )
  return tuple(tiles)


def primary_scores(record: RunRecord) -> tuple[str, ...]:
  """The primary score of each metric family of ``record``, in family order.

  The primary score of a family is the first one it emits. A family the
  registry does not know gives the first score any question recorded for it,
  and none when there is no such score. A record without metrics gives ``()``.
  """
  metrics = record.get("metrics")
  if not metrics:
    return ()
  found: list[str] = []
  for family in metrics["selection"]:
    known = METRICS.get(family)
    if known is not None:
      found.append(known.scores[0])
      continue
    for item in record["questions"]:
      score = next(
        (
          next(iter(result["scores"]))
          for result in item.get("metrics") or ()
          if result["name"] == family and result["scores"]
        ),
        None,
      )
      if score is not None:
        found.append(score)
        break
  return tuple(found)


def _result_of(item: QuestionRecord, score: str) -> MetricResultRecord | None:
  results = item.get("metrics") or ()
  family = _FAMILY_OF.get(score)
  for result in results:
    if result["name"] == family or (family is None and score in result["scores"]):
      return result
  return None


def score_cell(item: QuestionRecord, score: str) -> str:
  """The HTML of one question's ``score``: the value, ``error`` or a dash.

  A dash stands for a metric that was not applicable, was skipped, has no
  value for this question or was not computed at all.
  """
  result = _result_of(item, score)
  if result is None:
    return NO_DATA
  if result["status"] == "error":
    return f'<span class="status bad" title="{esc(result["error"] or "")}">error</span>'
  value = result["scores"].get(score)
  return NO_DATA if value is None else esc(format_score(value))

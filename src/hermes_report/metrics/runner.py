"""Run metric families over one question and aggregate the scores of a run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter

from hermes_report.metrics.base import (
  JudgeCall,
  Metric,
  MetricInput,
  MetricResult,
  MetricServices,
  describe,
)

__all__ = ["ScoreSummary", "aggregate", "evaluate_metrics"]


@dataclass(frozen=True)
class ScoreSummary:
  """The aggregate of one score over a run.

  Attributes:
    mean: Mean of the real values; ``None`` when there is none.
    n: How many real values the mean covers.
    not_applicable: Questions where the score had no input or no value.
    skipped: Questions where its metric was skipped.
    errors: Questions where its metric failed.
  """

  mean: float | None
  n: int
  not_applicable: int
  skipped: int
  errors: int


def _evaluate_one(
  item: MetricInput, metric: Metric, services: MetricServices
) -> MetricResult:
  try:
    missing = [f for f in sorted(metric.requires) if not item.has(f)]
    if missing:
      return MetricResult.not_applicable(metric.name, missing)
    if "judge" in metric.needs and services.judge is None:
      return MetricResult.skipped(metric.name, "sin juez")
    if "embedder" in metric.needs and services.embedder is None:
      return MetricResult.skipped(metric.name, "sin embedder")
  except Exception as exc:
    return MetricResult.failed(metric.name, describe(exc), ())
  judge = services.judge
  mark = len(judge.calls) if judge else 0
  start = perf_counter()
  try:
    result = metric.compute(item, services)
  except Exception as exc:
    calls: tuple[JudgeCall, ...] = judge.calls_since(mark) if judge else ()
    result = MetricResult.failed(metric.name, describe(exc), calls)
  return replace(result, seconds=perf_counter() - start)


def evaluate_metrics(
  item: MetricInput, metrics: Sequence[Metric], services: MetricServices
) -> tuple[MetricResult, ...]:
  """Run every metric over ``item``; never raises.

  A metric with missing input is ``not_applicable``, one without its service
  is ``skipped`` and one that raises is ``error``; the next metric still runs.
  """
  return tuple(_evaluate_one(item, metric, services) for metric in metrics)


def aggregate(
  results: Sequence[Sequence[MetricResult]], metrics: Sequence[Metric]
) -> dict[str, ScoreSummary]:
  """Mean of each score over the questions where it has a value.

  Questions without a value count as ``not_applicable``, ``skipped`` or
  ``errors`` apart and never enter the mean.
  """
  out: dict[str, ScoreSummary] = {}
  for metric in metrics:
    for score in metric.scores:
      values: list[float] = []
      not_applicable = skipped = errors = 0
      for per_question in results:
        found = next((r for r in per_question if r.name == metric.name), None)
        if found is None or found.status == "not_applicable":
          not_applicable += 1
        elif found.status == "skipped":
          skipped += 1
        elif found.status == "error":
          errors += 1
        else:
          value = found.scores.get(score)
          if value is None:
            not_applicable += 1
          else:
            values.append(value)
      mean = sum(values) / len(values) if values else None
      out[score] = ScoreSummary(mean, len(values), not_applicable, skipped, errors)
  return out

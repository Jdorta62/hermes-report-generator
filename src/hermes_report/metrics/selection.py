"""Metric registry and the selection of the families that run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from hermes_report.metrics.base import Metric
from hermes_report.metrics.deterministic import (
  Cost,
  DocRetrieval,
  EvidenceRetrieval,
  Refusal,
)

__all__ = ["METRICS", "MetricSelection", "MetricSelectionError"]


def _registry(*metrics: Metric) -> Mapping[str, Metric]:
  return MappingProxyType({metric.name: metric for metric in metrics})


METRICS: Final[Mapping[str, Metric]] = _registry(
  DocRetrieval(), EvidenceRetrieval(), Refusal(), Cost()
)


class MetricSelectionError(ValueError):
  """An unknown metric name (message in Spanish).

  Attributes:
    names: The unknown names.
    available: The names the registry has.
  """

  def __init__(self, names: Sequence[str], available: Sequence[str] = ()) -> None:
    self.names = tuple(names)
    self.available = tuple(available)
    message = f"Métrica desconocida: {', '.join(self.names)}"
    if self.available:
      message += f". Disponibles: {', '.join(self.available)}"
    super().__init__(message)


@dataclass(frozen=True)
class MetricSelection:
  """Which metric families run.

  Attributes:
    enabled: ``False`` turns every family off.
    include: Families that run although they are off by default.
    exclude: Families that do not run.
  """

  enabled: bool = True
  include: tuple[str, ...] = ()
  exclude: tuple[str, ...] = ()

  def resolve(self, registry: Mapping[str, Metric] = METRICS) -> tuple[Metric, ...]:
    """Default-on plus ``include`` minus ``exclude``, in registry order.

    The result is empty when ``enabled`` is false. Names are checked first,
    whatever ``enabled`` says.

    Raises:
      MetricSelectionError: a name in ``include`` or ``exclude`` is not in
        the registry.
    """
    unknown = [n for n in (*self.include, *self.exclude) if n not in registry]
    if unknown:
      raise MetricSelectionError(unknown, list(registry))
    if not self.enabled:
      return ()
    chosen = {n for n, m in registry.items() if m.default_enabled} | set(self.include)
    chosen -= set(self.exclude)
    return tuple(m for n, m in registry.items() if n in chosen)

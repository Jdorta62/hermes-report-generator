"""Tracing wrappers that record what every phase of the pipeline sees and returns.

Each wrapper is registered in ``HermesEnrich`` (enrichers) or
``HermesGenerator`` (strategy, LLM) under the name of the component it wraps,
so the real facades run unchanged and the report only listens.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from hermes_core import ContextChunk, EnrichedContext, PromptMessage, UserQuery
from hermes_enrich import ChunkRange, ChunkStore, Enricher, EnrichmentPhase
from hermes_generator import (
  AnswerRequest,
  GenerationParams,
  GenerationStrategy,
  LLMClient,
  LLMReply,
  StrategyOutcome,
)

__all__ = [
  "EnrichStep",
  "LLMCall",
  "StoreRead",
  "Trace",
  "TracingEnricher",
  "TracingLLM",
  "TracingStrategy",
]


@dataclass(frozen=True)
class StoreRead:
  """One ``fetch_ranges`` call: the ranges asked for and the chunks returned."""

  ranges: tuple[ChunkRange, ...]
  returned: tuple[ContextChunk, ...]
  seconds: float


@dataclass(frozen=True)
class EnrichStep:
  """One enrichment technique: its input, its output and its store reads."""

  name: str
  phase: EnrichmentPhase
  before: EnrichedContext
  after: EnrichedContext
  reads: tuple[StoreRead, ...]
  seconds: float


@dataclass(frozen=True)
class LLMCall:
  """One LLM completion: what was sent and what came back."""

  messages: tuple[PromptMessage, ...]
  params: GenerationParams
  reply: LLMReply | None
  error: str | None
  seconds: float


@dataclass
class Trace:
  """Everything recorded during one ``enrich`` call and one ``generate`` call.

  ``context`` is the ``EnrichedContext`` that ``HermesEnrich.enrich`` returned
  and ``HermesGenerator.generate`` received: the hermes-core contract between
  the two modules.
  """

  steps: list[EnrichStep] = field(default_factory=list)
  context: EnrichedContext | None = None
  request: AnswerRequest | None = None
  outcome: StrategyOutcome | None = None
  llm_calls: list[LLMCall] = field(default_factory=list)


class _TracingStore:
  """Store that forwards every read and records it."""

  def __init__(self, inner: ChunkStore) -> None:
    self._inner: ChunkStore = inner
    self.reads: list[StoreRead] = []

  def fetch_ranges(
    self, ranges: Sequence[ChunkRange], origin: tuple[str, ...]
  ) -> list[ContextChunk]:
    start: float = time.perf_counter()
    chunks: list[ContextChunk] = self._inner.fetch_ranges(ranges, origin)
    self.reads.append(
      StoreRead(tuple(ranges), tuple(chunks), time.perf_counter() - start)
    )
    return chunks


class TracingEnricher:
  """Enricher that runs ``inner`` and records its input, output and reads."""

  def __init__(self, inner: Enricher, trace: Trace) -> None:
    """Wrap ``inner``; the wrapper takes its ``name`` and ``phase``."""
    self.name: str = inner.name
    self.phase: EnrichmentPhase = inner.phase
    self._inner: Enricher = inner
    self._trace: Trace = trace

  def enrich(
    self, query: UserQuery, context: EnrichedContext, store: ChunkStore
  ) -> EnrichedContext:
    """Run the wrapped technique and record the step."""
    spy = _TracingStore(store)
    start: float = time.perf_counter()
    out: EnrichedContext = self._inner.enrich(query, context, spy)
    self._trace.steps.append(
      EnrichStep(
        self.name,
        self.phase,
        context,
        out,
        tuple(spy.reads),
        time.perf_counter() - start,
      )
    )
    return out


class TracingStrategy:
  """Strategy that runs ``inner`` and records its request and outcome."""

  def __init__(self, inner: GenerationStrategy, trace: Trace) -> None:
    """Wrap ``inner``; the wrapper takes its ``name``."""
    self.name: str = inner.name
    self._inner: GenerationStrategy = inner
    self._trace: Trace = trace

  def generate(self, request: AnswerRequest, llm: LLMClient) -> StrategyOutcome:
    """Run the wrapped strategy and record what it received and produced."""
    self._trace.request = request
    outcome: StrategyOutcome = self._inner.generate(request, llm)
    self._trace.outcome = outcome
    return outcome


class TracingLLM:
  """LLM client that forwards every call to ``inner`` and records it."""

  def __init__(self, inner: LLMClient, trace: Trace) -> None:
    """Wrap ``inner``."""
    self._inner: LLMClient = inner
    self._trace: Trace = trace

  def complete(
    self, messages: Sequence[PromptMessage], params: GenerationParams
  ) -> LLMReply:
    """Call the wrapped client; the call is recorded also when it fails."""
    start: float = time.perf_counter()
    try:
      reply: LLMReply = self._inner.complete(messages, params)
    except Exception as exc:
      self._trace.llm_calls.append(
        LLMCall(
          tuple(messages),
          params,
          None,
          f"{type(exc).__name__}: {exc}",
          time.perf_counter() - start,
        )
      )
      raise
    self._trace.llm_calls.append(
      LLMCall(tuple(messages), params, reply, None, time.perf_counter() - start)
    )
    return reply

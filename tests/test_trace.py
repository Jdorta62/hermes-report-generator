"""Tests for ``hermes_report.trace``: the tracing wrappers of the playground.

The unit tests use fakes that raise or return sentinels; the end-to-end tests
run the real ``HermesEnrich`` and ``HermesGenerator`` over the ``tests/data``
corpus (``HashEmbedder``, ``QdrantClient(":memory:")``, ``EchoLLM``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, cast

import pytest
from conftest import PipelineEnv
from hermes_core import ContextChunk, EnrichedContext, PromptMessage, UserQuery
from hermes_enrich import (
  ChunkRange,
  ChunkStore,
  ContextBudget,
  Enricher,
  EnrichmentControl,
  EnrichmentPhase,
  HermesEnrich,
)
from hermes_enrich.techniques import DocOrderEnricher, MergeEnricher, NeighborsEnricher
from hermes_enrich.tokens import CharRatioTokenCounter
from hermes_generator import (
  AnswerRequest,
  GenerationControl,
  GenerationParams,
  GenerationStrategy,
  HermesGenerator,
  LLMClient,
  LLMReply,
  StrategyOutcome,
)
from hermes_generator.answer.strategy import DefaultStrategy
from hermes_retrieval import HermesRetrieval
from hermes_retrieval.contracts import SearchScope, VectorialSearchParams

from hermes_report.providers import EchoLLM
from hermes_report.trace import (
  EnrichStep,
  LLMCall,
  StoreRead,
  Trace,
  TracingEnricher,
  TracingLLM,
  TracingStrategy,
  _TracingStore,
)

QUERY = UserQuery(
  text="¿Cuántos días tardó en ponerle nombre a su rocín?", query_id="t"
)
PARAMS = GenerationParams(temperature=0.0, seed=42)
MESSAGES = (PromptMessage(role="system", content="s"), PromptMessage("user", "u"))


class Boom(Exception):
  """A distinctive error raised by the fakes."""


# ---------------------------------------------------------------- the dataclasses


def test_trace_starts_empty() -> None:
  trace = Trace()
  assert trace.steps == []
  assert trace.llm_calls == []
  assert trace.context is None
  assert trace.request is None
  assert trace.outcome is None


def test_trace_lists_are_not_shared_between_instances() -> None:
  a, b = Trace(), Trace()
  a.steps.append(cast(Any, object()))
  a.llm_calls.append(cast(Any, object()))
  assert b.steps == []
  assert b.llm_calls == []


def test_trace_is_mutable() -> None:
  trace = Trace()
  trace.context = cast(Any, "ctx")
  assert trace.context == "ctx"


@pytest.mark.parametrize("cls", [StoreRead, EnrichStep, LLMCall])
def test_records_are_frozen(cls: type) -> None:
  assert dataclasses.is_dataclass(cls)
  assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_trace_fields_are_the_documented_ones() -> None:
  assert [f.name for f in dataclasses.fields(Trace)] == [
    "steps",
    "context",
    "request",
    "outcome",
    "llm_calls",
  ]


# ------------------------------------------------------------------ _TracingStore


class FakeStore:
  """A store that returns canned chunks and remembers its calls."""

  def __init__(self, chunks: Sequence[ContextChunk] = ()) -> None:
    self.chunks = list(chunks)
    self.calls: list[tuple[Sequence[ChunkRange], tuple[str, ...]]] = []

  def fetch_ranges(
    self, ranges: Sequence[ChunkRange], origin: tuple[str, ...]
  ) -> list[ContextChunk]:
    self.calls.append((ranges, origin))
    return self.chunks


class RaisingStore:
  def fetch_ranges(
    self, ranges: Sequence[ChunkRange], origin: tuple[str, ...]
  ) -> list[ContextChunk]:
    raise Boom("store down")


RANGE = ChunkRange(collection_name="c", doc_id="d", first=0, last=2)


def test_store_forwards_the_call_and_records_the_read() -> None:
  chunk = cast(ContextChunk, object())
  inner = FakeStore([chunk])
  spy = _TracingStore(cast(ChunkStore, inner))
  ranges = [RANGE, ChunkRange("c", "d", 5, 5)]
  out = spy.fetch_ranges(ranges, ("neighbors",))
  assert out is inner.chunks
  assert inner.calls == [(ranges, ("neighbors",))]
  assert len(spy.reads) == 1
  read = spy.reads[0]
  assert read.ranges == tuple(ranges)
  assert read.returned == (chunk,)
  assert isinstance(read.ranges, tuple)
  assert read.seconds >= 0.0


def test_store_records_an_empty_read() -> None:
  spy = _TracingStore(cast(ChunkStore, FakeStore()))
  assert spy.fetch_ranges([], ()) == []
  assert spy.reads[0].ranges == ()
  assert spy.reads[0].returned == ()


def test_store_records_every_read_in_order() -> None:
  spy = _TracingStore(cast(ChunkStore, FakeStore()))
  spy.fetch_ranges([RANGE], ("a",))
  spy.fetch_ranges([ChunkRange("c", "e", 1, 1)], ("b",))
  assert [r.ranges[0].doc_id for r in spy.reads] == ["d", "e"]


def test_store_error_propagates_and_records_nothing() -> None:
  spy = _TracingStore(cast(ChunkStore, RaisingStore()))
  with pytest.raises(Boom):
    spy.fetch_ranges([RANGE], ())
  assert spy.reads == []


# ---------------------------------------------------------------- TracingEnricher


class FakeEnricher:
  name = "fake"
  phase = EnrichmentPhase.EXPAND

  def __init__(self, reads: int = 0, error: Exception | None = None) -> None:
    self.reads = reads
    self.error = error

  def enrich(
    self, query: UserQuery, context: EnrichedContext, store: ChunkStore
  ) -> EnrichedContext:
    if self.error is not None:
      raise self.error
    for n in range(self.reads):
      store.fetch_ranges([ChunkRange("c", "d", n, n)], (self.name,))
    return cast(EnrichedContext, f"out-{self.reads}")


def test_enricher_takes_name_and_phase_from_the_inner_one() -> None:
  wrapper = TracingEnricher(cast(Enricher, FakeEnricher()), Trace())
  assert wrapper.name == "fake"
  assert wrapper.phase is EnrichmentPhase.EXPAND


@pytest.mark.parametrize(
  "technique", [NeighborsEnricher, MergeEnricher, DocOrderEnricher]
)
def test_enricher_keeps_the_identity_of_each_builtin(technique: type) -> None:
  inner = technique()
  wrapper = TracingEnricher(inner, Trace())
  assert (wrapper.name, wrapper.phase) == (inner.name, inner.phase)


def test_enricher_records_input_output_and_reads() -> None:
  trace = Trace()
  wrapper = TracingEnricher(cast(Enricher, FakeEnricher(reads=2)), trace)
  before = cast(EnrichedContext, "in")
  out = wrapper.enrich(QUERY, before, cast(ChunkStore, FakeStore()))
  assert cast(Any, out) == "out-2"
  assert len(trace.steps) == 1
  step = trace.steps[0]
  assert (step.name, step.phase) == ("fake", EnrichmentPhase.EXPAND)
  assert step.before is before
  assert step.after is out
  assert [r.ranges[0].first for r in step.reads] == [0, 1]
  assert isinstance(step.reads, tuple)
  assert step.seconds >= 0.0


def test_enricher_reads_are_per_call_not_accumulated() -> None:
  trace = Trace()
  wrapper = TracingEnricher(cast(Enricher, FakeEnricher(reads=1)), trace)
  store = cast(ChunkStore, FakeStore())
  wrapper.enrich(QUERY, cast(EnrichedContext, "a"), store)
  wrapper.enrich(QUERY, cast(EnrichedContext, "b"), store)
  assert [len(s.reads) for s in trace.steps] == [1, 1]


def test_enricher_without_reads_records_an_empty_tuple() -> None:
  trace = Trace()
  TracingEnricher(cast(Enricher, FakeEnricher()), trace).enrich(
    QUERY, cast(EnrichedContext, "a"), cast(ChunkStore, FakeStore())
  )
  assert trace.steps[0].reads == ()


def test_enricher_error_propagates_and_records_no_step() -> None:
  trace = Trace()
  wrapper = TracingEnricher(cast(Enricher, FakeEnricher(error=Boom("x"))), trace)
  with pytest.raises(Boom):
    wrapper.enrich(QUERY, cast(EnrichedContext, "a"), cast(ChunkStore, FakeStore()))
  assert trace.steps == []


def test_enricher_store_error_propagates() -> None:
  trace = Trace()
  wrapper = TracingEnricher(cast(Enricher, FakeEnricher(reads=1)), trace)
  with pytest.raises(Boom):
    wrapper.enrich(QUERY, cast(EnrichedContext, "a"), cast(ChunkStore, RaisingStore()))
  assert trace.steps == []


# ---------------------------------------------------------------- TracingStrategy


class FakeStrategy:
  name = "fake-strategy"

  def __init__(self, error: Exception | None = None) -> None:
    self.error = error
    self.outcome = cast(StrategyOutcome, object())
    self.seen: list[tuple[Any, Any]] = []

  def generate(self, request: AnswerRequest, llm: LLMClient) -> StrategyOutcome:
    self.seen.append((request, llm))
    if self.error is not None:
      raise self.error
    return self.outcome


def test_strategy_takes_the_name_and_records_request_and_outcome() -> None:
  trace = Trace()
  inner = FakeStrategy()
  wrapper = TracingStrategy(cast(GenerationStrategy, inner), trace)
  request, llm = cast(AnswerRequest, object()), cast(LLMClient, object())
  assert wrapper.name == "fake-strategy"
  assert wrapper.generate(request, llm) is inner.outcome
  assert inner.seen == [(request, llm)]
  assert trace.request is request
  assert trace.outcome is inner.outcome


def test_strategy_error_keeps_the_request_and_no_outcome() -> None:
  trace = Trace()
  wrapper = TracingStrategy(cast(GenerationStrategy, FakeStrategy(Boom("x"))), trace)
  request = cast(AnswerRequest, object())
  with pytest.raises(Boom):
    wrapper.generate(request, cast(LLMClient, object()))
  assert trace.request is request
  assert trace.outcome is None


def test_strategy_second_call_replaces_the_first_record() -> None:
  trace = Trace()
  wrapper = TracingStrategy(cast(GenerationStrategy, FakeStrategy()), trace)
  first, second = cast(AnswerRequest, object()), cast(AnswerRequest, object())
  wrapper.generate(first, cast(LLMClient, object()))
  wrapper.generate(second, cast(LLMClient, object()))
  assert trace.request is second


# -------------------------------------------------------------------- TracingLLM


class FakeLLM:
  def __init__(self, replies: Sequence[LLMReply | Exception]) -> None:
    self.replies = list(replies)
    self.seen: list[tuple[Sequence[PromptMessage], GenerationParams]] = []

  def complete(
    self, messages: Sequence[PromptMessage], params: GenerationParams
  ) -> LLMReply:
    self.seen.append((messages, params))
    reply = self.replies.pop(0)
    if isinstance(reply, Exception):
      raise reply
    return reply


REPLY = LLMReply(text="hola", model="m", prompt_tokens=3, completion_tokens=1)


def test_llm_forwards_and_records_a_call() -> None:
  trace = Trace()
  inner = FakeLLM([REPLY])
  out = TracingLLM(cast(LLMClient, inner), trace).complete(list(MESSAGES), PARAMS)
  assert out is REPLY
  assert inner.seen == [(list(MESSAGES), PARAMS)]
  assert len(trace.llm_calls) == 1
  call = trace.llm_calls[0]
  assert call.messages == MESSAGES
  assert isinstance(call.messages, tuple)
  assert call.params is PARAMS
  assert call.reply is REPLY
  assert call.error is None
  assert call.seconds >= 0.0


def test_llm_error_is_recorded_and_the_same_exception_is_reraised() -> None:
  trace = Trace()
  error = Boom("model down")
  wrapper = TracingLLM(cast(LLMClient, FakeLLM([error])), trace)
  with pytest.raises(Boom) as info:
    wrapper.complete(MESSAGES, PARAMS)
  assert info.value is error
  call = trace.llm_calls[0]
  assert call.reply is None
  assert call.error == "Boom: model down"
  assert call.messages == MESSAGES


def test_llm_error_with_an_empty_message_is_still_recorded() -> None:
  trace = Trace()
  wrapper = TracingLLM(cast(LLMClient, FakeLLM([Boom()])), trace)
  with pytest.raises(Boom):
    wrapper.complete(MESSAGES, PARAMS)
  assert trace.llm_calls[0].error == "Boom: "


def test_llm_keyboard_interrupt_is_not_swallowed_nor_recorded() -> None:
  class Interrupted(BaseException):
    pass

  class Raising:
    def complete(
      self, messages: Sequence[PromptMessage], params: GenerationParams
    ) -> LLMReply:
      raise Interrupted

  trace = Trace()
  with pytest.raises(Interrupted):
    TracingLLM(cast(LLMClient, Raising()), trace).complete(MESSAGES, PARAMS)
  assert trace.llm_calls == []


def test_llm_records_calls_in_order_including_failures() -> None:
  trace = Trace()
  wrapper = TracingLLM(cast(LLMClient, FakeLLM([REPLY, Boom("x"), REPLY])), trace)
  wrapper.complete(MESSAGES, PARAMS)
  with pytest.raises(Boom):
    wrapper.complete(MESSAGES, PARAMS)
  wrapper.complete(MESSAGES, PARAMS)
  assert [c.error is None for c in trace.llm_calls] == [True, False, True]


def test_llm_empty_messages_are_recorded_as_an_empty_tuple() -> None:
  trace = Trace()
  TracingLLM(cast(LLMClient, FakeLLM([REPLY])), trace).complete([], PARAMS)
  assert trace.llm_calls[0].messages == ()


# ------------------------------------------------------------------- end to end


def _enrich_and_generate(
  env: PipelineEnv, trace: Trace, *, max_chunks: int | None = 20
) -> tuple[Any, Any]:
  retrieval = HermesRetrieval(embedder=env.embedder, qdrant_client=env.qdrant).retrieve(
    QUERY,
    SearchScope(collection_names=[env.collection]),
    search_params=VectorialSearchParams(top_k=5, score_threshold=None),
  )
  enrich = HermesEnrich(
    env.qdrant,
    enrichers={
      e.name: TracingEnricher(e, trace)
      for e in (NeighborsEnricher(), MergeEnricher(), DocOrderEnricher())
    },
    token_counter=CharRatioTokenCounter(),
  )
  context = enrich.enrich(
    QUERY,
    retrieval,
    context_budget=ContextBudget(max_context_tokens=None, max_chunks=max_chunks),
    pipeline_control=EnrichmentControl(enrichments=("neighbors", "merge", "doc_order")),
  )
  trace.context = context
  generator = HermesGenerator(
    TracingLLM(EchoLLM(), trace),
    strategies={"default": TracingStrategy(DefaultStrategy(), trace)},
    default_generation_params=PARAMS,
  )
  result = generator.generate(
    QUERY, context, pipeline_control=GenerationControl(include_prompt=True)
  )
  return context, result


def test_real_facades_fill_the_whole_trace(pipeline_env: PipelineEnv) -> None:
  trace = Trace()
  context, result = _enrich_and_generate(pipeline_env, trace)
  assert [s.name for s in trace.steps] == ["neighbors", "merge", "doc_order"]
  assert [s.phase.name for s in trace.steps] == [
    s.phase.name for s in (NeighborsEnricher(), MergeEnricher(), DocOrderEnricher())
  ]
  # The steps chain: each one starts from the pieces the previous one left
  # (the facade only appends the technique name between them).
  for previous, current in zip(trace.steps, trace.steps[1:], strict=False):
    assert current.before.pieces == previous.after.pieces
  assert trace.steps[-1].after.pieces == context.pieces
  assert trace.context is context
  assert trace.request is not None
  assert trace.outcome is not None
  assert len(trace.llm_calls) == 1
  assert trace.llm_calls[0].reply is not None
  assert trace.llm_calls[0].params.seed == 42
  assert result.status == "answered"


def test_real_neighbors_step_records_its_store_read(pipeline_env: PipelineEnv) -> None:
  trace = Trace()
  _enrich_and_generate(pipeline_env, trace)
  neighbors = trace.steps[0]
  assert len(neighbors.reads) >= 1
  read = neighbors.reads[0]
  assert read.ranges
  assert all(r.collection_name == pipeline_env.collection for r in read.ranges)
  assert all(isinstance(c, ContextChunk) for c in read.returned)
  # Merge and doc_order never read the store.
  assert [len(s.reads) for s in trace.steps[1:]] == [0, 0]


def test_real_budget_is_applied_after_the_steps(pipeline_env: PipelineEnv) -> None:
  trace = Trace()
  context, _ = _enrich_and_generate(pipeline_env, trace, max_chunks=1)
  # The budget drops chunks after the last technique: no step sees the drop.
  assert context.dropped_chunk_ids
  assert trace.steps[-1].after.dropped_chunk_ids == ()
  assert len(trace.steps[-1].after.pieces) > len(context.pieces)


def test_real_llm_error_keeps_the_context_and_the_call(
  pipeline_env: PipelineEnv,
) -> None:
  trace = Trace()
  retrieval = HermesRetrieval(
    embedder=pipeline_env.embedder, qdrant_client=pipeline_env.qdrant
  ).retrieve(
    QUERY,
    SearchScope(collection_names=[pipeline_env.collection]),
    search_params=VectorialSearchParams(top_k=5, score_threshold=None),
  )
  context = HermesEnrich(
    pipeline_env.qdrant, token_counter=CharRatioTokenCounter()
  ).enrich(QUERY, retrieval, context_budget=ContextBudget(max_context_tokens=None))
  trace.context = context
  generator = HermesGenerator(
    TracingLLM(cast(LLMClient, FakeLLM([Boom("down")] * 5)), trace),
    strategies={"default": TracingStrategy(DefaultStrategy(), trace)},
    default_generation_params=PARAMS,
  )
  with pytest.raises(Exception, match="down"):
    generator.generate(QUERY, context)
  assert trace.request is not None
  assert trace.outcome is None
  assert trace.llm_calls
  assert all(c.error == "Boom: down" for c in trace.llm_calls)

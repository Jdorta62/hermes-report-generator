"""Pipeline of one question: retrieve, enrich and generate, recording every phase."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any

from hermes_core import Embedder, GenerationResult, RetrievalResult, UserQuery
from hermes_enrich import ContextBudget, EnrichmentControl, HermesEnrich
from hermes_enrich.techniques import (
  DocOrderEnricher,
  Enricher,
  MergeEnricher,
  NeighborsEnricher,
)
from hermes_enrich.tokens import TokenCounter, default_token_counter
from hermes_generator import (
  GenerationControl,
  GenerationParams,
  HermesGenerator,
  LLMClient,
  ResponseOptions,
)
from hermes_generator.answer.strategy import DefaultStrategy
from hermes_retrieval import HermesRetrieval
from hermes_retrieval.contracts import (
  PipelineControl,
  SearchScope,
  VectorialSearchParams,
)
from qdrant_client import QdrantClient

from hermes_report.config import RunConfig
from hermes_report.questions import Question
from hermes_report.record import (
  EnrichRecord,
  GenerationRecord,
  LLMCallRecord,
  QuestionRecord,
  RetrievalRecord,
  StepRecord,
  StoreReadRecord,
  chunk_range_record,
  context_record,
  generation_result_record,
  message_record,
  retrieved_chunk_record,
  searched_query_record,
)
from hermes_report.trace import (
  EnrichStep,
  LLMCall,
  StoreRead,
  Trace,
  TracingEnricher,
  TracingLLM,
  TracingStrategy,
)

__all__ = ["PipelineContext", "answer_question", "build_context"]


@dataclass
class PipelineContext:
  """What answering a question needs: the run, the store and the clients.

  Attributes:
    config: The validated run.
    qdrant: The Qdrant client holding the collection.
    collection: Name of the collection to search.
    retriever: The retrieval facade, built over ``qdrant``.
    llm: The language model that writes the answers.
    token_counter: Counter of the context budget.
  """

  config: RunConfig
  qdrant: QdrantClient
  collection: str
  retriever: HermesRetrieval
  llm: LLMClient
  token_counter: TokenCounter


def build_context(
  config: RunConfig,
  qdrant: QdrantClient,
  collection: str,
  embedder: Embedder,
  llm: LLMClient,
  token_counter: TokenCounter | None = None,
) -> PipelineContext:
  """Build the pipeline context of a run over an existing collection.

  Without ``token_counter`` the default counter of hermes-enrich is used.
  """
  return PipelineContext(
    config=config,
    qdrant=qdrant,
    collection=collection,
    retriever=HermesRetrieval(embedder=embedder, qdrant_client=qdrant),
    llm=llm,
    token_counter=(default_token_counter() if token_counter is None else token_counter),
  )


def _error(exc: Exception) -> str:
  return f"{type(exc).__name__}: {exc}"


def _read_record(read: StoreRead) -> StoreReadRecord:
  return {
    "ranges": [chunk_range_record(r) for r in read.ranges],
    "returned": [chunk.chunk_id for chunk in read.returned],
    "seconds": read.seconds,
  }


def _step_record(step: EnrichStep) -> StepRecord:
  return {
    "name": step.name,
    "phase": step.phase.name,
    "seconds": step.seconds,
    "before": context_record(step.before),
    "after": context_record(step.after),
    "reads": [_read_record(read) for read in step.reads],
  }


def _call_record(call: LLMCall) -> LLMCallRecord:
  reply: dict[str, Any] | None = None
  if call.reply is not None:
    reply = {
      "text": call.reply.text,
      "model": call.reply.model,
      "prompt_tokens": call.reply.prompt_tokens,
      "completion_tokens": call.reply.completion_tokens,
    }
  return {
    "messages": [message_record(m) for m in call.messages],
    "params": dataclasses.asdict(call.params),
    "reply": reply,
    "error": call.error,
    "seconds": call.seconds,
  }


def _retrieve(
  query: UserQuery, ctx: PipelineContext
) -> tuple[RetrievalResult, RetrievalRecord]:
  retrieval = ctx.config.retrieval
  start = time.perf_counter()
  result = ctx.retriever.retrieve(
    query,
    SearchScope(collection_names=[ctx.collection]),
    search_params=VectorialSearchParams(
      top_k=retrieval.top_k, score_threshold=retrieval.score_threshold
    ),
    pipeline_control=PipelineControl(candidate_factor=retrieval.candidate_factor),
  )
  seconds = time.perf_counter() - start
  return result, {
    "seconds": seconds,
    "chunks": [retrieved_chunk_record(c) for c in result.chunks],
    "queries": [searched_query_record(q) for q in result.queries],
  }


def _enrich(
  query: UserQuery,
  retrieval: RetrievalResult,
  ctx: PipelineContext,
  trace: Trace,
  record: EnrichRecord,
) -> None:
  """Enrich into ``trace``; ``record`` is kept up to date also on failure."""
  enrich = ctx.config.enrich
  max_tokens = enrich.max_context_tokens or None
  record["budget"] = {
    "max_context_tokens": max_tokens,
    "max_chunks": enrich.max_chunks,
  }
  start = time.perf_counter()
  try:
    builtins: tuple[Enricher, ...] = (
      NeighborsEnricher(),
      MergeEnricher(),
      DocOrderEnricher(),
    )
    enricher = HermesEnrich(
      ctx.qdrant,
      enrichers={e.name: TracingEnricher(e, trace) for e in builtins},
      token_counter=ctx.token_counter,
    )
    trace.context = enricher.enrich(
      query,
      retrieval,
      context_budget=ContextBudget(
        max_context_tokens=max_tokens, max_chunks=enrich.max_chunks
      ),
      pipeline_control=EnrichmentControl(enrichments=tuple(enrich.chain)),
    )
  finally:
    record["seconds"] = time.perf_counter() - start
    record["steps"] = [_step_record(step) for step in trace.steps]
    if trace.context is not None:
      record["context"] = context_record(trace.context)


def _generate(
  query: UserQuery,
  ctx: PipelineContext,
  trace: Trace,
  record: GenerationRecord,
) -> None:
  """Generate from ``trace.context``; ``record`` is kept up to date on failure."""
  generation = ctx.config.generation
  context = trace.context
  start = time.perf_counter()
  try:
    if context is None:
      raise RuntimeError("no enriched context to generate from")
    generator = HermesGenerator(
      TracingLLM(ctx.llm, trace),
      strategies={"default": TracingStrategy(DefaultStrategy(), trace)},
      default_generation_params=GenerationParams(
        temperature=generation.temperature, seed=generation.seed
      ),
    )
    options = (
      ResponseOptions(insufficient_message=generation.insufficient_message)
      if generation.insufficient_message
      else ResponseOptions()
    )
    result: GenerationResult = generator.generate(
      query,
      context,
      pipeline_control=GenerationControl(
        response_format=generation.response_format,
        response=options,
        include_prompt=True,
      ),
    )
    record["result"] = generation_result_record(result)
  finally:
    record["seconds"] = time.perf_counter() - start
    record["llm_calls"] = [_call_record(call) for call in trace.llm_calls]


def answer_question(question: Question, ctx: PipelineContext) -> QuestionRecord:
  """Answer ``question`` through retrieval, enrichment and generation, recorded.

  Never raises for pipeline errors: the first failing phase sets ``error``,
  keeps the records of the finished phases and what the failing one had
  recorded, and leaves the later phases ``None``. ``rules``, ``judge`` and
  ``metrics`` are left ``None`` for the grader.
  """
  begin = time.perf_counter()
  query = UserQuery(text=question.question, query_id=question.id)
  trace = Trace()
  retrieval_record: RetrievalRecord | None = None
  enrich_record: EnrichRecord | None = None
  generation_record: GenerationRecord | None = None
  error: str | None = None
  try:
    retrieval, retrieval_record = _retrieve(query, ctx)
    enrich_record = {"seconds": 0.0, "budget": {}, "steps": [], "context": None}
    _enrich(query, retrieval, ctx, trace, enrich_record)
    generation_record = {"seconds": 0.0, "llm_calls": [], "result": None}
    _generate(query, ctx, trace, generation_record)
  except Exception as exc:
    error = _error(exc)
  return {
    "spec": question.to_dict(),
    "query_id": question.id,
    "seconds": time.perf_counter() - begin,
    "error": error,
    "retrieval": retrieval_record,
    "enrich": enrich_record,
    "generation": generation_record,
    "rules": None,
    "judge": None,
    "metrics": None,
  }

"""Tests for ``hermes_report.pipeline``: one question through the whole pipeline.

The happy paths run over the ``tests/data`` corpus ingested with ``HashEmbedder``
into ``QdrantClient(":memory:")`` and answered by ``EchoLLM``. The error paths
swap one collaborator of the context for a fake that raises.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Sequence
from typing import Any, cast

import pytest
from conftest import PipelineEnv
from hermes_core import PromptMessage, UserQuery
from hermes_enrich import ContextBudget, EnrichmentControl, HermesEnrich
from hermes_enrich.techniques import MergeEnricher
from hermes_enrich.tokens import (
  CharRatioTokenCounter,
  TokenCounter,
  default_token_counter,
)
from hermes_generator import (
  GenerationControl,
  GenerationParams,
  HermesGenerator,
  LLMClient,
  LLMReply,
  ResponseOptions,
)
from hermes_retrieval import HermesRetrieval
from hermes_retrieval.contracts import (
  PipelineControl,
  SearchScope,
  VectorialSearchParams,
)
from qdrant_client import QdrantClient

from hermes_report import pipeline
from hermes_report.config import RunConfig
from hermes_report.pipeline import PipelineContext, answer_question, build_context
from hermes_report.providers import EchoLLM
from hermes_report.questions import Question
from hermes_report.record import (
  context_record,
  generation_result_record,
  retrieved_chunk_record,
  searched_query_record,
)

ROCIN = "¿Cuántos días tardó en ponerle nombre a su rocín?"
COUNTER = CharRatioTokenCounter()


class Boom(Exception):
  """A distinctive error raised by the fakes."""


class RaisingLLM:
  """An LLM that fails on every call."""

  def __init__(self, error: BaseException | None = None) -> None:
    self.error = error or Boom("model down")
    self.calls = 0

  def complete(
    self, messages: Sequence[PromptMessage], params: GenerationParams
  ) -> LLMReply:
    self.calls += 1
    raise self.error


class RaisingRetriever:
  def retrieve(self, *args: Any, **kwargs: Any) -> Any:
    raise Boom("retriever down")


class RaisingQdrant:
  """A Qdrant stand-in whose every use fails."""

  def __getattr__(self, name: str) -> Any:
    raise Boom(f"qdrant down ({name})")


class RaisingCounter:
  name = "raising"

  def count(self, text: str) -> int:
    raise Boom("counter down")


Rec = dict[str, Any]


def _answer(question: Question, ctx: PipelineContext) -> Rec:
  """``answer_question`` with the record typed as plain nested data."""
  return cast(Rec, answer_question(question, ctx))


def _config(env: PipelineEnv, **sections: Any) -> RunConfig:
  """The sample config with some sections replaced by ``replace`` of their fields."""
  config = env.config
  for section, changes in sections.items():
    config = dataclasses.replace(
      config,
      **{section: dataclasses.replace(getattr(config, section), **changes)},
    )
  return config


def _ctx(
  env: PipelineEnv,
  config: RunConfig | None = None,
  llm: LLMClient | None = None,
  counter: TokenCounter | None = COUNTER,
) -> PipelineContext:
  return build_context(
    config or env.config,
    env.qdrant,
    env.collection,
    env.embedder,
    llm or EchoLLM(),
    counter,
  )


def _question(text: str = ROCIN, qid: str = "x", **fields: Any) -> Question:
  return Question(id=qid, question=text, expected="e", **fields)


def _no_threshold(env: PipelineEnv, **enrich: Any) -> RunConfig:
  """A config whose retrieval returns every chunk, so budgets have work to do."""
  config = _config(env, retrieval={"score_threshold": None})
  return _config_with(config, enrich=enrich) if enrich else config


def _config_with(config: RunConfig, **sections: Any) -> RunConfig:
  for section, changes in sections.items():
    config = dataclasses.replace(
      config, **{section: dataclasses.replace(getattr(config, section), **changes)}
    )
  return config


def _plain(env: PipelineEnv, query: UserQuery) -> Any:
  """The enriched context of ``query`` through the untraced facades."""
  config = env.config
  retrieval = HermesRetrieval(embedder=env.embedder, qdrant_client=env.qdrant).retrieve(
    query,
    SearchScope(collection_names=[env.collection]),
    search_params=VectorialSearchParams(
      top_k=config.retrieval.top_k, score_threshold=config.retrieval.score_threshold
    ),
  )
  return HermesEnrich(env.qdrant, token_counter=COUNTER).enrich(
    query,
    retrieval,
    context_budget=ContextBudget(max_context_tokens=3000, max_chunks=20),
    pipeline_control=EnrichmentControl(enrichments=tuple(config.enrich.chain)),
  )


def _roundtrip(record: Rec) -> Any:
  return json.loads(json.dumps(record, allow_nan=False))


def _strip_seconds(value: Any) -> Any:
  """The record without any ``seconds`` key, to compare two runs."""
  if isinstance(value, dict):
    return {k: _strip_seconds(v) for k, v in value.items() if k != "seconds"}
  if isinstance(value, list):
    return [_strip_seconds(v) for v in value]
  return value


# ----------------------------------------------------------------- build_context


def test_build_context_wires_everything(pipeline_env: PipelineEnv) -> None:
  llm = EchoLLM()
  ctx = build_context(
    pipeline_env.config,
    pipeline_env.qdrant,
    pipeline_env.collection,
    pipeline_env.embedder,
    llm,
    COUNTER,
  )
  assert ctx.config is pipeline_env.config
  assert ctx.qdrant is pipeline_env.qdrant
  assert ctx.collection == pipeline_env.collection
  assert isinstance(ctx.retriever, HermesRetrieval)
  assert ctx.llm is llm
  assert ctx.token_counter is COUNTER


def test_build_context_defaults_the_token_counter(
  pipeline_env: PipelineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
  sentinel = CharRatioTokenCounter(3)
  monkeypatch.setattr(pipeline, "default_token_counter", lambda: sentinel)
  ctx = _ctx(pipeline_env, counter=None)
  assert ctx.token_counter is sentinel


def test_build_context_default_counter_is_a_real_counter(
  pipeline_env: PipelineEnv,
) -> None:
  ctx = _ctx(pipeline_env, counter=None)
  assert isinstance(ctx.token_counter, type(default_token_counter()))
  assert ctx.token_counter.count("hola mundo") > 0


def test_pipeline_context_is_a_mutable_dataclass(pipeline_env: PipelineEnv) -> None:
  ctx = _ctx(pipeline_env)
  assert dataclasses.is_dataclass(ctx)
  assert not ctx.__dataclass_params__.frozen  # type: ignore[attr-defined]
  assert [f.name for f in dataclasses.fields(ctx)] == [
    "config",
    "qdrant",
    "collection",
    "retriever",
    "llm",
    "token_counter",
  ]
  ctx.collection = "other"
  assert ctx.collection == "other"


# ------------------------------------------------------------ the happy path, q01


@pytest.fixture(scope="module")
def q01(pipeline_env: PipelineEnv) -> Rec:
  question = pipeline_env.questions.questions[0]
  assert question.id == "q01"
  return _answer(question, _ctx(pipeline_env))


def test_top_level_fields(pipeline_env: PipelineEnv, q01: Rec) -> None:
  question = pipeline_env.questions.questions[0]
  assert set(q01) == {
    "spec",
    "query_id",
    "seconds",
    "error",
    "retrieval",
    "enrich",
    "generation",
    "rules",
    "judge",
  }
  assert q01["spec"] == question.to_dict()
  assert q01["query_id"] == "q01"
  assert q01["error"] is None
  assert q01["rules"] is None
  assert q01["judge"] is None
  parts = [q01[p]["seconds"] for p in ("retrieval", "enrich", "generation")]
  assert all(s >= 0.0 for s in parts)
  assert q01["seconds"] >= sum(parts) * 0.999


def test_retrieval_record(pipeline_env: PipelineEnv, q01: Rec) -> None:
  retrieval = q01["retrieval"]
  assert retrieval is not None
  assert set(retrieval) == {"seconds", "chunks", "queries"}
  # Independent retrieval with the same parameters gives the same chunks.
  config = pipeline_env.config.retrieval
  expected = HermesRetrieval(
    embedder=pipeline_env.embedder, qdrant_client=pipeline_env.qdrant
  ).retrieve(
    UserQuery(text=ROCIN, query_id="q01"),
    SearchScope(collection_names=[pipeline_env.collection]),
    search_params=VectorialSearchParams(
      top_k=config.top_k, score_threshold=config.score_threshold
    ),
    pipeline_control=PipelineControl(candidate_factor=config.candidate_factor),
  )
  assert retrieval["chunks"] == [retrieved_chunk_record(c) for c in expected.chunks]
  assert retrieval["queries"] == [searched_query_record(s) for s in expected.queries]
  assert len(retrieval["chunks"]) == 1
  chunk = retrieval["chunks"][0]
  assert chunk["metadata"]["source"] == "parte-1/capitulo-01.md"
  assert "text" not in chunk["metadata"]
  assert chunk["collection_name"] == pipeline_env.collection
  assert retrieval["queries"][0]["text"] == ROCIN


def test_enrich_record(pipeline_env: PipelineEnv, q01: Rec) -> None:
  enrich = q01["enrich"]
  assert enrich is not None
  assert set(enrich) == {"seconds", "budget", "steps", "context"}
  assert enrich["budget"] == {"max_context_tokens": 3000, "max_chunks": 20}
  steps = enrich["steps"]
  assert [s["name"] for s in steps] == ["neighbors", "merge", "doc_order"]
  assert [s["phase"] for s in steps] == ["EXPAND", "CONSOLIDATE", "ORDER"]
  for step in steps:
    assert set(step) == {"name", "phase", "seconds", "before", "after", "reads"}
    assert step["seconds"] >= 0.0
  # Only neighbors reads the store; it asked for ranges and got chunk ids.
  reads = steps[0]["reads"]
  assert len(reads) == 1
  assert set(reads[0]) == {"ranges", "returned", "seconds"}
  assert reads[0]["ranges"]
  assert all(
    set(r) == {"collection_name", "doc_id", "first", "last"} for r in reads[0]["ranges"]
  )
  assert reads[0]["returned"]
  assert all(isinstance(cid, str) for cid in reads[0]["returned"])
  assert steps[1]["reads"] == [] and steps[2]["reads"] == []
  # The first step starts from the retrieved chunks; the pieces chain up to the end.
  retrieved_ids = [c["chunk_id"] for c in q01["retrieval"]["chunks"]]
  before_ids = [
    c["chunk_id"] for p in steps[0]["before"]["pieces"] for c in p["chunks"]
  ]
  assert before_ids == retrieved_ids
  assert steps[1]["before"]["pieces"] == steps[0]["after"]["pieces"]
  assert steps[2]["before"]["pieces"] == steps[1]["after"]["pieces"]
  assert enrich["context"] is not None
  assert enrich["context"]["pieces"] == steps[2]["after"]["pieces"]
  assert enrich["context"]["enrichments"] == ["neighbors", "merge", "doc_order"]
  assert enrich["context"]["dropped_chunk_ids"] == []
  assert enrich["context"]["query_id"] == "q01"


def test_enrich_context_equals_an_untraced_enrichment(
  pipeline_env: PipelineEnv, q01: Rec
) -> None:
  """Tracing must not change what the facade produces."""
  plain = _plain(pipeline_env, UserQuery(text=ROCIN, query_id="q01"))
  assert q01["enrich"]["context"] == context_record(plain)


def test_generation_record(pipeline_env: PipelineEnv, q01: Rec) -> None:
  generation = q01["generation"]
  assert generation is not None
  assert set(generation) == {"seconds", "llm_calls", "result"}
  assert len(generation["llm_calls"]) == 1
  call = generation["llm_calls"][0]
  assert set(call) == {"messages", "params", "reply", "error", "seconds"}
  assert call["error"] is None
  assert call["seconds"] >= 0.0
  assert [m["role"] for m in call["messages"]][0] == "system"
  assert all(set(m) == {"role", "content"} for m in call["messages"])
  assert ROCIN in call["messages"][-1]["content"]
  assert call["params"] == dataclasses.asdict(
    GenerationParams(temperature=0.0, seed=42)
  )
  reply = call["reply"]
  assert set(reply) == {"text", "model", "prompt_tokens", "completion_tokens"}
  assert reply["model"] == "echo"
  assert reply["text"].endswith("[1]")
  result = generation["result"]
  assert result is not None
  assert result["status"] == "answered"
  assert result["query_id"] == "q01"
  assert result["answer"] == reply["text"]
  assert [c["number"] for c in result["citations"]] == [1]
  assert result["prompt"] is not None or "prompt" not in result  # include_prompt=True
  assert "Rocinante" in result["answer"]


def test_generation_result_equals_an_untraced_generation(
  pipeline_env: PipelineEnv, q01: Rec
) -> None:
  """The result is ``generation_result_record`` of the real ``GenerationResult``."""
  query = UserQuery(text=ROCIN, query_id="q01")
  expected = HermesGenerator(
    EchoLLM(), default_generation_params=GenerationParams(temperature=0.0, seed=42)
  ).generate(
    query,
    _plain(pipeline_env, query),
    pipeline_control=GenerationControl(
      response=ResponseOptions(
        insufficient_message=pipeline_env.config.generation.insufficient_message
      ),
      include_prompt=True,
    ),
  )
  assert q01["generation"]["result"] == generation_result_record(expected)


def test_json_round_trip_of_every_question(pipeline_env: PipelineEnv) -> None:
  ctx = _ctx(pipeline_env)
  for question in pipeline_env.questions.questions:
    record = _answer(question, ctx)
    assert _roundtrip(record) == record


def test_round_trip_with_the_failing_phases(pipeline_env: PipelineEnv) -> None:
  for ctx in (
    _ctx(pipeline_env, llm=cast(LLMClient, RaisingLLM())),
    _ctx(pipeline_env, counter=cast(TokenCounter, RaisingCounter())),
    dataclasses.replace(_ctx(pipeline_env), retriever=cast(Any, RaisingRetriever())),
  ):
    record = _answer(_question(), ctx)
    assert record["error"] is not None
    assert _roundtrip(record) == record


def test_answering_is_deterministic_apart_from_timings(
  pipeline_env: PipelineEnv,
) -> None:
  question = pipeline_env.questions.questions[0]
  first = _answer(question, _ctx(pipeline_env))
  second = _answer(question, _ctx(pipeline_env))
  assert _strip_seconds(first) == _strip_seconds(second)


def test_the_question_is_not_mutated_and_spec_is_a_plain_copy(
  pipeline_env: PipelineEnv,
) -> None:
  question = _question(keywords=("a",), evidence=("b",), documents=("c.md",))
  before = copy.deepcopy(question)
  record = _answer(question, _ctx(pipeline_env))
  assert question == before
  assert record["spec"]["keywords"] == ["a"]
  assert isinstance(record["spec"]["documents"], list)


def test_second_question_uses_its_own_trace(pipeline_env: PipelineEnv) -> None:
  """No state leaks between questions answered with the same context."""
  ctx = _ctx(pipeline_env)
  first = _answer(_question(qid="a"), ctx)
  second = _answer(_question("¿De dónde salió don Quijote?", qid="b"), ctx)
  assert len(second["generation"]["llm_calls"]) == 1
  assert len(second["enrich"]["steps"]) == 3
  assert second["generation"]["result"]["query_id"] == "b"
  assert first["generation"]["result"]["query_id"] == "a"


def test_unicode_and_odd_questions_do_not_break_the_record(
  pipeline_env: PipelineEnv,
) -> None:
  for text in ("¿Quién fue ñandú? \U0001f600", "x", "   ", "?" * 5000):
    record = _answer(_question(text), _ctx(pipeline_env))
    assert record["spec"]["question"] == text
    assert _roundtrip(record) == record


# ------------------------------------------------------- empty retrieval, q11


def test_q11_is_empty_because_the_score_threshold_filters_every_chunk(
  pipeline_env: PipelineEnv,
) -> None:
  """The empty retrieval of q11 is the retriever's threshold, not a bug.

  ``score_threshold`` applies to the raw cosine similarity (``vector_score``):
  the best chunk for q11 scores 0.255 < 0.3, so nothing survives. Without the
  threshold the same query returns every chunk.
  """
  q11 = pipeline_env.questions.questions[2]
  assert q11.id == "q11"
  filtered = _answer(q11, _ctx(pipeline_env))
  assert filtered["retrieval"]["chunks"] == []
  open_ctx = _ctx(pipeline_env, _no_threshold(pipeline_env))
  unfiltered = _answer(q11, open_ctx)
  chunks = unfiltered["retrieval"]["chunks"]
  assert len(chunks) == 4
  best = max(c["vector_score"] for c in chunks)
  assert best < pipeline_env.config.retrieval.score_threshold


def test_empty_retrieval_gives_no_context(pipeline_env: PipelineEnv) -> None:
  llm = RaisingLLM()
  record = _answer(
    pipeline_env.questions.questions[2], _ctx(pipeline_env, llm=cast(LLMClient, llm))
  )
  assert record["error"] is None
  assert record["retrieval"]["chunks"] == []
  assert record["enrich"] is not None
  assert record["enrich"]["steps"] == []
  assert record["enrich"]["budget"] == {"max_context_tokens": 3000, "max_chunks": 20}
  assert record["enrich"]["context"]["pieces"] == []
  generation = record["generation"]
  assert generation is not None
  assert generation["llm_calls"] == []
  assert llm.calls == 0  # the model is never asked without sources
  assert generation["result"]["status"] == "no_context"
  assert generation["result"]["citations"] == []
  assert (
    generation["result"]["answer"]
    == pipeline_env.config.generation.insufficient_message
  )


def test_no_context_without_insufficient_message_uses_the_default(
  pipeline_env: PipelineEnv,
) -> None:
  config = _config(pipeline_env, generation={"insufficient_message": None})
  record = _answer(pipeline_env.questions.questions[2], _ctx(pipeline_env, config))
  result = record["generation"]["result"]
  assert result["status"] == "no_context"
  assert result["answer"]
  assert result["answer"] != pipeline_env.config.generation.insufficient_message


def test_empty_insufficient_message_means_the_default(
  pipeline_env: PipelineEnv,
) -> None:
  config = _config(pipeline_env, generation={"insufficient_message": ""})
  record = _answer(pipeline_env.questions.questions[2], _ctx(pipeline_env, config))
  assert record["error"] is None
  assert record["generation"]["result"]["answer"]


# ---------------------------------------------------------------- budgets


def test_max_chunks_budget_drops_chunks(pipeline_env: PipelineEnv) -> None:
  config = _no_threshold(pipeline_env, max_chunks=1)
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["error"] is None
  enrich = record["enrich"]
  assert enrich is not None
  assert enrich["budget"] == {"max_context_tokens": 3000, "max_chunks": 1}
  context = enrich["context"]
  assert context is not None
  kept = [c["chunk_id"] for p in context["pieces"] for c in p["chunks"]]
  assert len(kept) == 1
  assert len(context["dropped_chunk_ids"]) == 3
  assert not set(kept) & set(context["dropped_chunk_ids"])
  # The dropped chunks never reach the prompt nor the citations.
  result = record["generation"]["result"]
  cited = {cid for c in result["citations"] for cid in c["chunk_ids"]}
  assert cited <= set(kept)
  # The steps ran before the budget: the last step still held everything.
  assert enrich["steps"][-1]["after"]["dropped_chunk_ids"] == []
  assert _roundtrip(record) == record


def test_token_budget_drops_chunks_but_keeps_the_best(
  pipeline_env: PipelineEnv,
) -> None:
  full = _answer(_question(), _ctx(pipeline_env, _no_threshold(pipeline_env)))
  full_tokens = sum(
    COUNTER.count(p["text"]) for p in full["enrich"]["context"]["pieces"]
  )
  config = _no_threshold(pipeline_env, max_context_tokens=full_tokens - 1)
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["error"] is None
  context = record["enrich"]["context"]
  assert context["dropped_chunk_ids"]
  assert record["enrich"]["budget"]["max_context_tokens"] == full_tokens - 1
  assert record["generation"]["result"]["status"] == "answered"


def test_zero_token_budget_means_no_limit_and_is_recorded_as_none(
  pipeline_env: PipelineEnv,
) -> None:
  config = _no_threshold(pipeline_env, max_context_tokens=0, max_chunks=None)
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["error"] is None
  assert record["enrich"]["budget"] == {"max_context_tokens": None, "max_chunks": None}
  assert record["enrich"]["context"]["dropped_chunk_ids"] == []
  assert _roundtrip(record) == record


def test_token_budget_below_the_best_chunk_is_an_enrich_error(
  pipeline_env: PipelineEnv,
) -> None:
  config = _config(pipeline_env, enrich={"max_context_tokens": 1})
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["error"] is not None
  assert record["error"].startswith("ContextBudgetError: ")
  assert record["retrieval"]["chunks"]
  enrich = record["enrich"]
  assert enrich is not None
  assert enrich["budget"] == {"max_context_tokens": 1, "max_chunks": 20}
  assert [s["name"] for s in enrich["steps"]] == ["neighbors", "merge", "doc_order"]
  assert enrich["context"] is None
  assert record["generation"] is None


def test_chain_is_honoured_in_order_and_subset(pipeline_env: PipelineEnv) -> None:
  config = _config(pipeline_env, enrich={"chain": ("merge",)})
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert [s["name"] for s in record["enrich"]["steps"]] == ["merge"]
  assert record["enrich"]["context"]["enrichments"] == ["merge"]


def test_empty_chain_runs_no_step(pipeline_env: PipelineEnv) -> None:
  config = _config(pipeline_env, enrich={"chain": ()})
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["error"] is None
  assert record["enrich"]["steps"] == []
  assert record["generation"]["result"]["status"] == "answered"


def test_temperature_and_seed_reach_the_llm_params(pipeline_env: PipelineEnv) -> None:
  config = _config(pipeline_env, generation={"temperature": 0.7, "seed": None})
  record = _answer(_question(), _ctx(pipeline_env, config))
  params = record["generation"]["llm_calls"][0]["params"]
  assert params["temperature"] == 0.7
  assert params["seed"] is None


def test_top_k_bounds_the_retrieval(pipeline_env: PipelineEnv) -> None:
  config = _config(pipeline_env, retrieval={"top_k": 2, "score_threshold": None})
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert len(record["retrieval"]["chunks"]) == 2


def test_response_format_reaches_the_result(pipeline_env: PipelineEnv) -> None:
  config = _config(pipeline_env, generation={"response_format": "bullets"})
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["generation"]["result"]["response_format"] == "bullets"


# ----------------------------------------------------------------- errors


def _assert_error(record: Rec, name: str, message: str) -> None:
  assert record["error"] == f"{name}: {message}"
  assert record["rules"] is None and record["judge"] is None
  assert record["spec"]["question"] == ROCIN
  assert record["query_id"] == "x"
  assert _roundtrip(record) == record


def test_retriever_error_leaves_every_phase_none(pipeline_env: PipelineEnv) -> None:
  ctx = dataclasses.replace(_ctx(pipeline_env), retriever=cast(Any, RaisingRetriever()))
  record = _answer(_question(), ctx)
  _assert_error(record, "Boom", "retriever down")
  assert record["retrieval"] is None
  assert record["enrich"] is None
  assert record["generation"] is None
  assert record["seconds"] >= 0.0


def test_invalid_retrieval_parameter_is_a_retrieval_error(
  pipeline_env: PipelineEnv,
) -> None:
  config = _config(pipeline_env, retrieval={"top_k": 0})
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["error"] is not None and record["error"].startswith("ValueError: ")
  assert (record["retrieval"], record["enrich"], record["generation"]) == (
    None,
    None,
    None,
  )


def test_missing_collection_is_a_retrieval_error(pipeline_env: PipelineEnv) -> None:
  ctx = dataclasses.replace(_ctx(pipeline_env), collection="hermes_report_missing")
  record = _answer(_question(), ctx)
  assert record["error"] is not None
  assert (record["retrieval"], record["enrich"], record["generation"]) == (
    None,
    None,
    None,
  )


def test_qdrant_error_in_enrichment_keeps_the_retrieval(
  pipeline_env: PipelineEnv,
) -> None:
  ctx = dataclasses.replace(
    _ctx(pipeline_env), qdrant=cast(QdrantClient, RaisingQdrant())
  )
  record = _answer(_question(), ctx)
  assert record["error"] is not None and record["error"].startswith(
    "EnrichmentError: Enrichment technique 'neighbors' failed: qdrant down"
  )
  assert record["retrieval"] is not None
  assert record["retrieval"]["chunks"]
  enrich = record["enrich"]
  assert enrich is not None
  assert enrich["context"] is None
  assert enrich["budget"] == {"max_context_tokens": 3000, "max_chunks": 20}
  assert record["generation"] is None
  assert _roundtrip(record) == record


def test_failing_step_keeps_the_steps_done_so_far(
  pipeline_env: PipelineEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
  class BrokenMerge(MergeEnricher):
    def enrich(self, query: Any, context: Any, store: Any) -> Any:
      raise Boom("merge exploded")

  monkeypatch.setattr(pipeline, "MergeEnricher", BrokenMerge)
  record = _answer(_question(), _ctx(pipeline_env))
  _assert_error(
    record,
    "EnrichmentError",
    "Enrichment technique 'merge' failed: merge exploded",
  )
  enrich = record["enrich"]
  assert enrich is not None
  assert [s["name"] for s in enrich["steps"]] == ["neighbors"]
  assert enrich["steps"][0]["reads"]
  assert enrich["context"] is None
  assert enrich["seconds"] >= 0.0
  assert record["retrieval"]["chunks"]
  assert record["generation"] is None


def test_token_counter_error_is_an_enrich_error(pipeline_env: PipelineEnv) -> None:
  ctx = _ctx(pipeline_env, counter=cast(TokenCounter, RaisingCounter()))
  record = _answer(_question(), ctx)
  _assert_error(record, "Boom", "counter down")
  assert record["retrieval"] is not None
  assert record["enrich"] is not None
  assert record["enrich"]["context"] is None
  assert record["generation"] is None


@pytest.mark.parametrize(
  ("section", "changes"),
  [
    ("enrich", {"chain": ("nope",)}),
    ("enrich", {"max_chunks": 0}),
    ("enrich", {"max_chunks": -3}),
    ("enrich", {"max_context_tokens": -5}),
  ],
)
def test_bad_enrich_config_becomes_an_enrich_error(
  pipeline_env: PipelineEnv, section: str, changes: dict[str, Any]
) -> None:
  config = _config(pipeline_env, **{section: changes})
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["error"] is not None
  assert record["retrieval"] is not None
  assert record["enrich"] is not None
  assert record["enrich"]["context"] is None
  assert record["enrich"]["budget"]["max_chunks"] == changes.get("max_chunks", 20)
  assert record["generation"] is None
  assert _roundtrip(record) == record


def test_llm_error_keeps_the_context_and_the_failed_call(
  pipeline_env: PipelineEnv,
) -> None:
  llm = RaisingLLM()
  record = _answer(_question(), _ctx(pipeline_env, llm=cast(LLMClient, llm)))
  _assert_error(record, "GenerationError", record["error"].partition(": ")[2])
  assert "model down" in record["error"]
  assert record["retrieval"]["chunks"]
  assert record["enrich"]["context"] is not None
  generation = record["generation"]
  assert generation is not None
  assert generation["result"] is None
  assert generation["seconds"] >= 0.0
  assert generation["llm_calls"]
  assert len(generation["llm_calls"]) == llm.calls
  for call in generation["llm_calls"]:
    assert call["reply"] is None
    assert call["error"] == "Boom: model down"
    assert call["messages"]
    assert call["params"]["seed"] == 42


def test_bad_response_format_is_a_generation_error_without_calls(
  pipeline_env: PipelineEnv,
) -> None:
  config = _config(pipeline_env, generation={"response_format": "bogus"})
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["error"] is not None
  assert record["enrich"]["context"] is not None
  assert record["generation"] == {
    "seconds": record["generation"]["seconds"],
    "llm_calls": [],
    "result": None,
  }


def test_bad_temperature_is_a_generation_error(pipeline_env: PipelineEnv) -> None:
  config = _config(pipeline_env, generation={"temperature": -1.0})
  record = _answer(_question(), _ctx(pipeline_env, config))
  assert record["error"] is not None and "temperature" in record["error"]
  assert record["generation"] is not None and record["generation"]["result"] is None
  assert record["enrich"]["context"] is not None


def test_llm_that_replies_nothing_usable_does_not_raise(
  pipeline_env: PipelineEnv,
) -> None:
  class Mute:
    def complete(
      self, messages: Sequence[PromptMessage], params: GenerationParams
    ) -> LLMReply:
      return LLMReply(text="", model="mute", prompt_tokens=None, completion_tokens=None)

  record = _answer(_question(), _ctx(pipeline_env, llm=cast(LLMClient, Mute())))
  assert _roundtrip(record) == record  # error or not, it is valid data


def test_llm_reply_with_token_counts_is_recorded(pipeline_env: PipelineEnv) -> None:
  class Counted:
    def complete(
      self, messages: Sequence[PromptMessage], params: GenerationParams
    ) -> LLMReply:
      return LLMReply(
        text="Cuatro días. [1]", model="m1", prompt_tokens=11, completion_tokens=4
      )

  record = _answer(_question(), _ctx(pipeline_env, llm=cast(LLMClient, Counted())))
  assert record["error"] is None
  reply = record["generation"]["llm_calls"][0]["reply"]
  assert reply == {
    "text": "Cuatro días. [1]",
    "model": "m1",
    "prompt_tokens": 11,
    "completion_tokens": 4,
  }
  result = record["generation"]["result"]
  assert (result["model"], result["prompt_tokens"], result["completion_tokens"]) == (
    "m1",
    11,
    4,
  )


def test_keyboard_interrupt_is_not_swallowed(pipeline_env: PipelineEnv) -> None:
  llm = RaisingLLM()
  llm.error = KeyboardInterrupt()
  with pytest.raises(KeyboardInterrupt):
    _answer(_question(), _ctx(pipeline_env, llm=cast(LLMClient, llm)))

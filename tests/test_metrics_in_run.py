"""Metrics inside a run (task 5.5.4): the adapter, the runner wiring, schema 1.

Everything is offline: in-memory Qdrant, ``HashEmbedder``, ``EchoLLM`` and scripted
fake judge clients. The schema-1 fixture under ``tests/fixtures/schema1`` was
produced by the code of commit 5f0a2ea (before this task) over ``tests/data``;
its paths were rewritten to ``/fixture/...``.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from hermes_core.generation import PromptMessage
from hermes_enrich.tokens import default_token_counter
from hermes_generator import GenerationParams, LLMReply
from hermes_generator.contracts import ResponseOptions
from test_metrics_judged import (
  PurposeJudge,
  claims_reply,
  questions_reply,
  relevance_reply,
  verdicts_reply,
)

from hermes_report import cli, pipeline, runner
from hermes_report.config import Overrides, RunConfig, load_config
from hermes_report.evaluate import is_refused, rule_verdict
from hermes_report.metrics import METRICS, MetricSelection, metric_input
from hermes_report.metrics.adapter import metric_input as metric_input_direct
from hermes_report.metrics.base import METRIC_STATUSES, MetricInput
from hermes_report.metrics.judge import JudgeSession
from hermes_report.metrics.judged import Citations
from hermes_report.providers import EchoLLM
from hermes_report.questions import Question
from hermes_report.record import QuestionRecord, RunRecord, read_record
from hermes_report.render import render_compare, render_run
from hermes_report.runner import RunOutcome, run

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "examples" / "demo"
SCHEMA1 = Path(__file__).parent / "fixtures" / "schema1"

DEFAULT_ON = [name for name, metric in METRICS.items() if metric.default_enabled]
DETERMINISTIC = ["doc_retrieval", "evidence_retrieval", "refusal", "cost"]
JUDGED = [
  "faithfulness",
  "answer_relevance",
  "context_precision",
  "context_recall",
  "answer_correctness",
  "citations",
]


@pytest.fixture(autouse=True)
def quiet_qdrant() -> Iterator[None]:
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    yield


@pytest.fixture
def make_config(project: Path, tmp_path: Path) -> Callable[..., RunConfig]:
  def make(**overrides: Any) -> RunConfig:
    overrides.setdefault("out", tmp_path / "out")
    return load_config(project / "run.toml", Overrides(**overrides))

  return make


def quiet(config: RunConfig) -> RunOutcome:
  return run(config, progress=lambda line: None)


def by_name(question: QuestionRecord) -> dict[str, Any]:
  assert question["metrics"] is not None
  return {m["name"]: m for m in question["metrics"]}


class RunJudge(PurposeJudge):
  """Metric prompts by purpose, anything else is the verdict prompt."""

  model = "run-judge"

  def __init__(self, **script: Any) -> None:
    script.setdefault("answer_claims", self._claims)
    script.setdefault("expected_claims", self._claims)
    script.setdefault("verify", verdicts_reply((True, [1])))
    script.setdefault("relevance", self._relevance)
    script.setdefault("questions", questions_reply("¿Quién es?", "¿Qué hizo?"))
    super().__init__(**script)
    self.verdict_calls = 0

  @staticmethod
  def _claims(user: str) -> str:
    """One claim carrying the [n] markers found in the text, like a real judge."""
    markers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", user)})
    return claims_reply(("Una afirmación.", markers))

  @staticmethod
  def _relevance(user: str) -> str:
    count = len(re.findall(r"^\[\d+\] ", user, flags=re.MULTILINE))
    return relevance_reply(*([True] * count))

  def complete_json(self, messages: Sequence[PromptMessage]) -> str:
    try:
      self.purpose_of(messages)
    except AssertionError:
      self.verdict_calls += 1
      return '{"verdict": "correct", "reason": "bien"}'
    return super().complete_json(messages)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> RunJudge:
  judge = RunJudge()
  monkeypatch.setattr(runner, "make_judge", lambda cfg: judge)
  return judge


# --------------------------------------------------------------- the adapter


@pytest.fixture
def records(make_config: Callable[..., RunConfig]) -> dict[str, QuestionRecord]:
  outcome = quiet(make_config(no_metrics=True))
  return {q["query_id"]: q for q in outcome.record["questions"]}


@pytest.fixture
def questions(project: Path) -> dict[str, Question]:
  from hermes_report.corpus import load_corpus
  from hermes_report.questions import load_questions

  config = load_config(project / "run.toml")
  corpus = load_corpus(config.corpus.dir, config.corpus.glob)
  qs = load_questions(config.questions.file, corpus.sources).questions
  return {q.id: q for q in qs}


def test_reexported_and_same_function() -> None:
  assert metric_input is metric_input_direct


class TestMetricInput:
  def test_every_field_of_an_answered_question(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record, question = records["q01"], questions["q01"]
    item = metric_input(question, record)
    assert isinstance(item, MetricInput)
    assert item.question_id == "q01"
    assert item.question == question.question
    assert item.expected == question.expected
    assert item.refusal is False
    assert item.documents == question.documents
    assert item.evidence == question.evidence
    generation, retrieval, enrich = (
      record["generation"],
      record["retrieval"],
      record["enrich"],
    )
    assert generation and generation["result"] and retrieval and enrich
    assert enrich["context"]
    assert item.answer == generation["result"]["answer"]
    assert item.refused is False
    assert [c.chunk_id for c in item.retrieved] == [
      c["chunk_id"] for c in retrieval["chunks"]
    ]
    assert [c.text for c in item.retrieved] == [c["text"] for c in retrieval["chunks"]]
    assert [c.score for c in item.retrieved] == [
      c["score"] for c in retrieval["chunks"]
    ]
    assert [c.source for c in item.retrieved] == [
      c["metadata"]["source"] for c in retrieval["chunks"]
    ]
    assert item.context == tuple(p["text"] for p in enrich["context"]["pieces"])
    assert item.seconds == {
      "retrieval": retrieval["seconds"],
      "enrich": enrich["seconds"],
      "generation": generation["seconds"],
    }
    assert item.citations is None  # derived from the [n] markers by the metrics

  def test_prompt_is_the_last_call_and_replies_are_the_texts(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record = records["q01"]
    assert record["generation"]
    calls = record["generation"]["llm_calls"]
    assert calls
    item = metric_input(questions["q01"], record)
    assert [(m.role, m.content) for m in item.prompt] == [
      (m["role"], m["content"]) for m in calls[-1]["messages"]
    ]
    assert item.replies == tuple(c["reply"]["text"] for c in calls if c["reply"])
    assert item.replies  # EchoLLM answered

  def test_prompt_is_of_the_last_call_even_if_it_failed_and_replies_skip_it(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record = copy.deepcopy(records["q01"])
    assert record["generation"]
    first = copy.deepcopy(record["generation"]["llm_calls"][0])
    second = copy.deepcopy(first)
    second["reply"] = None
    second["error"] = "boom"
    second["messages"] = [{"role": "user", "content": "otro"}]
    record["generation"]["llm_calls"] = [first, second]
    item = metric_input(questions["q01"], record)
    assert [m.content for m in item.prompt] == ["otro"]
    assert item.replies == (first["reply"]["text"],)  # type: ignore[index]

  def test_generation_without_calls(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record = copy.deepcopy(records["q01"])
    assert record["generation"]
    record["generation"]["llm_calls"] = []
    item = metric_input(questions["q01"], record)
    assert item.prompt == () and item.replies == ()

  def test_no_context_question_has_no_pieces(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    item = metric_input(questions["q11"], records["q11"])
    assert item.refusal is True
    assert item.refused is True
    assert item.context == ()
    assert item.answer is not None

  def test_failed_pipeline_still_builds_an_input(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record = copy.deepcopy(records["q01"])
    record["error"] = "boom"
    record["enrich"] = record["generation"] = None
    record["rules"] = record["judge"] = record["metrics"] = None
    item = metric_input(questions["q01"], record)
    assert item.answer is None and item.refused is None
    assert item.context == () and item.prompt == () and item.replies == ()
    assert set(item.seconds) == {"retrieval"}
    assert item.retrieved  # what retrieval finished is kept

  def test_nothing_finished(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record = copy.deepcopy(records["q01"])
    record["error"] = "boom"
    record["retrieval"] = record["enrich"] = record["generation"] = None
    item = metric_input(questions["q01"], record)
    assert item.retrieved == () and item.seconds == {} and item.answer is None

  def test_generation_record_without_result(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record = copy.deepcopy(records["q01"])
    assert record["generation"]
    record["generation"]["result"] = None
    item = metric_input(questions["q01"], record)
    assert item.answer is None and item.refused is None
    assert "generation" in item.seconds  # the phase ran

  def test_enrich_record_without_context(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record = copy.deepcopy(records["q01"])
    assert record["enrich"]
    record["enrich"]["context"] = None
    assert metric_input(questions["q01"], record).context == ()

  def test_retrieval_with_no_chunks(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record = copy.deepcopy(records["q01"])
    assert record["retrieval"]
    record["retrieval"]["chunks"] = []
    assert metric_input(questions["q01"], record).retrieved == ()

  @pytest.mark.parametrize(
    "metadata", [{}, {"source": None}, {"source": 3}, {"source": ["a.md"]}]
  )
  def test_missing_or_non_string_source_is_none(
    self,
    records: dict[str, QuestionRecord],
    questions: dict[str, Question],
    metadata: dict[str, Any],
  ) -> None:
    record = copy.deepcopy(records["q01"])
    assert record["retrieval"]
    record["retrieval"]["chunks"][0]["metadata"] = metadata
    assert metric_input(questions["q01"], record).retrieved[0].source is None

  def test_does_not_mutate_the_record(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    before = copy.deepcopy(records["q01"])
    metric_input(questions["q01"], records["q01"])
    assert records["q01"] == before

  @pytest.mark.parametrize(
    ("status", "refused"),
    [("answered", False), ("insufficient", True), ("no_context", True)],
  )
  @pytest.mark.parametrize("refusal_question", [True, False])
  def test_refusal_is_the_test_the_rules_use(
    self,
    records: dict[str, QuestionRecord],
    questions: dict[str, Question],
    status: str,
    refused: bool,
    refusal_question: bool,
  ) -> None:
    record = copy.deepcopy(records["q01"])
    assert record["generation"] and record["generation"]["result"]
    record["generation"]["result"]["status"] = status
    question = Question(
      id="q01",
      question="¿Cuántos días?",
      expected="Cuatro.",
      keywords=("cuatro",),
      refusal=refusal_question,
    )
    item = metric_input(question, record)
    assert item.refused is refused
    assert is_refused(record["generation"]["result"]) is refused
    # The rule verdict agrees with the metric input on what "refused" means.
    rules = rule_verdict(question, record)
    if refusal_question:
      assert (rules["verdict"] == "correct_refusal") is item.refused
    else:
      assert (rules["answer_ok"] is False) is item.refused

  def test_unknown_status_is_not_refused(
    self, records: dict[str, QuestionRecord], questions: dict[str, Question]
  ) -> None:
    record = copy.deepcopy(records["q01"])
    assert record["generation"] and record["generation"]["result"]
    record["generation"]["result"]["status"] = "weird"
    assert metric_input(questions["q01"], record).refused is False


# ---------------------------------------- citation_style="none" (5.5.2 carry-over)


@pytest.fixture
def style_none(monkeypatch: pytest.MonkeyPatch) -> None:
  """Make the pipeline generate with ``citation_style="none"`` (the real generator)."""

  def options(**kwargs: Any) -> ResponseOptions:
    return ResponseOptions(citation_style="none", **kwargs)

  monkeypatch.setattr(pipeline, "ResponseOptions", options)


class TestCitationStyleNone:
  def test_the_generator_really_strips_the_markers(
    self, style_none: None, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config(no_metrics=True)).record["questions"][0]
    assert record["generation"] and record["generation"]["result"]
    assert re.search(r"\[\d+\]", record["generation"]["result"]["answer"]) is None
    # ... although the model did cite: the raw reply carries [1].
    assert record["generation"]["llm_calls"][0]["reply"]
    assert "[1]" in record["generation"]["llm_calls"][0]["reply"]["text"]

  def test_metric_input_has_no_markers_and_no_explicit_citations(
    self,
    style_none: None,
    make_config: Callable[..., RunConfig],
    questions: dict[str, Question],
  ) -> None:
    record = quiet(make_config(no_metrics=True)).record["questions"][0]
    item = metric_input(questions["q01"], record)
    assert item.answer is not None
    assert re.search(r"\[\d+\]", item.answer) is None
    assert item.citations is None

  def test_citation_precision_is_none_and_recall_none_without_supported_claims(
    self,
    style_none: None,
    make_config: Callable[..., RunConfig],
    questions: dict[str, Question],
  ) -> None:
    record = quiet(make_config(no_metrics=True)).record["questions"][0]
    item = metric_input(questions["q01"], record)
    # The judge extracts a claim with no markers (there are none to carry).
    judge = PurposeJudge(
      answer_claims=claims_reply(("Cuatro días.", [])),
      verify=verdicts_reply((False, [])),
    )
    result = Citations().compute(item, _services(judge, item))
    assert result.status == "ok"
    assert result.scores == {"citation_precision": None, "citation_recall": None}

  def test_supported_but_uncited_claims_give_precision_none_and_recall_zero(
    self,
    style_none: None,
    make_config: Callable[..., RunConfig],
    questions: dict[str, Question],
  ) -> None:
    # Documented behaviour (design formula), not the "both None" of the carry-over:
    # recall is None only when no claim is supported by the context.
    record = quiet(make_config(no_metrics=True)).record["questions"][0]
    item = metric_input(questions["q01"], record)
    judge = PurposeJudge(
      answer_claims=claims_reply(("Cuatro días.", [])),
      verify=verdicts_reply((True, [1])),
    )
    result = Citations().compute(item, _services(judge, item))
    assert result.scores["citation_precision"] is None
    assert result.scores["citation_recall"] == 0.0

  def test_whole_run_with_style_none_has_no_citation_precision(
    self, style_none: None, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    outcome = quiet(make_config())
    for question in outcome.record["questions"]:
      citations = by_name(question)["citations"]
      assert citations["status"] in {"ok", "not_applicable"}
      if citations["status"] == "ok":
        assert citations["scores"]["citation_precision"] is None
    summary = outcome.record["metrics"]["summary"]  # type: ignore[index]
    assert summary["citation_precision"]["mean"] is None
    assert summary["citation_precision"]["n"] == 0

  def test_inline_style_does_cite(
    self, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    # Control: with the default style the same run does yield a precision.
    outcome = quiet(make_config())
    summary = outcome.record["metrics"]["summary"]  # type: ignore[index]
    assert summary["citation_precision"]["n"] >= 1
    assert summary["citation_precision"]["mean"] is not None


def _services(judge: Any, item: MetricInput) -> Any:
  from hermes_report.metrics import MetricServices

  return MetricServices(
    judge=JudgeSession(judge, item), embedder=None, counter=default_token_counter()
  )


# ------------------------------------------------------------- run wiring


class TestRunMetrics:
  def test_default_run_writes_schema_2_with_metrics_on_every_question(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    outcome = quiet(make_config())
    record = outcome.record
    assert record["schema"] == 2
    assert record["metrics"] is not None
    assert record["metrics"]["selection"] == DEFAULT_ON
    scores = [s for name in DEFAULT_ON for s in METRICS[name].scores]
    assert record["metrics"]["scores"] == scores
    assert list(record["metrics"]["summary"]) == scores
    assert len(record["questions"]) == 3
    for question in record["questions"]:
      assert question["metrics"] is not None
      assert [m["name"] for m in question["metrics"]] == DEFAULT_ON
      for m in question["metrics"]:
        assert m["status"] in METRIC_STATUSES
        assert set(m) == {
          "name",
          "status",
          "scores",
          "details",
          "calls",
          "error",
          "seconds",
        }
        assert m["seconds"] >= 0
    assert read_record(outcome.folder / "run.json") == record

  def test_record_with_judge_details_survives_the_json_round_trip(
    self, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    outcome = quiet(make_config())
    assert read_record(outcome.folder / "run.json") == outcome.record
    calls = [
      call
      for q in outcome.record["questions"]
      for m in q["metrics"] or []
      for call in m["calls"]
    ]
    assert calls  # judge calls are stored raw, with their messages
    assert all(call["messages"] and call["raw"] for call in calls)

  def test_a_single_question_run_has_a_summary_of_one(
    self, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config(only=("q11",))).record
    assert record["metrics"]
    for item in record["metrics"]["summary"].values():
      assert item["n"] + item["not_applicable"] + item["skipped"] + item["errors"] == 1
    # q11 is a refusal question that refused: no invented answer.
    assert record["metrics"]["summary"]["invented_answer"]["mean"] == 0.0

  def test_summary_counts_add_up_for_every_score(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config()).record
    assert record["metrics"]
    for score, item in record["metrics"]["summary"].items():
      total = item["n"] + item["not_applicable"] + item["skipped"] + item["errors"]
      assert total == len(record["questions"]), score
      assert (item["mean"] is None) == (item["n"] == 0), score

  def test_summary_is_the_mean_of_the_question_scores(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config()).record
    assert record["metrics"]
    values = [
      by_name(q)["doc_retrieval"]["scores"]["doc_hit"]
      for q in record["questions"]
      if by_name(q)["doc_retrieval"]["status"] == "ok"
    ]
    assert values
    mean = record["metrics"]["summary"]["doc_hit"]["mean"]
    assert mean == pytest.approx(sum(values) / len(values))
    assert record["metrics"]["summary"]["doc_hit"]["n"] == len(values)

  def test_deterministic_families_are_ok_or_not_applicable(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config()).record
    for question in record["questions"]:
      found = by_name(question)
      for name in DETERMINISTIC:
        assert found[name]["status"] in {"ok", "not_applicable"}, (name, found[name])
      assert found["cost"]["status"] == "ok"

  def test_cost_uses_the_enrich_token_counter_and_the_phase_seconds(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config()).record
    q01 = next(q for q in record["questions"] if q["query_id"] == "q01")
    cost = by_name(q01)["cost"]
    assert cost["details"]["counter"] == default_token_counter().name
    assert cost["scores"]["context_tokens"] > 0
    assert q01["retrieval"] and q01["generation"]
    assert cost["scores"]["seconds_retrieval"] == q01["retrieval"]["seconds"]
    assert cost["scores"]["seconds_generation"] == q01["generation"]["seconds"]

  def test_the_run_embedder_reaches_answer_relevance(
    self, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config()).record
    q01 = next(q for q in record["questions"] if q["query_id"] == "q01")
    relevance = by_name(q01)["answer_relevance"]
    assert relevance["status"] == "ok", relevance
    assert 0.0 <= relevance["scores"]["answer_relevance"] <= 1.0
    assert relevance["details"]["questions"] == ["¿Quién es?", "¿Qué hizo?"]

  def test_a_scripted_judge_makes_every_judged_family_ok_or_not_applicable(
    self, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config()).record
    for question in record["questions"]:
      for name in JUDGED:
        entry = by_name(question)[name]
        assert entry["status"] in {"ok", "not_applicable"}, (
          question["query_id"],
          entry,
        )
    assert record["metrics"]
    assert record["metrics"]["summary"]["faithfulness"]["n"] >= 1

  def test_each_question_gets_a_fresh_judge_session(
    self, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config()).record
    logged = [
      sum(
        1
        for m in q["metrics"] or []
        for call in m["calls"]
        if call["purpose"] == "answer_claims"
      )
      for q in record["questions"]
    ]
    # Memoised inside a question (one call although several families need it) ...
    assert all(n <= 1 for n in logged)
    # ... but not across questions.
    assert sum(logged) >= 2
    assert scripted.count("answer_claims") == sum(logged)

  def test_verdict_judge_runs_before_and_apart_from_the_metric_judge(
    self, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    record = quiet(make_config()).record
    assert scripted.verdict_calls == len(record["questions"])
    assert all(
      q["judge"] is not None and q["judge"]["verdict"] == "correct"
      for q in record["questions"]
    )

  def test_metrics_never_change_the_verdicts_or_the_failed_count(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    with_metrics = quiet(make_config())
    without = quiet(make_config(no_metrics=True))
    assert with_metrics.failed == without.failed
    for a, b in zip(
      with_metrics.record["questions"], without.record["questions"], strict=True
    ):
      assert a["rules"] == b["rules"]
      assert a["error"] == b["error"]
      assert a["judge"] and b["judge"]
      assert {k: v for k, v in a["judge"].items() if k != "seconds"} == {
        k: v for k, v in b["judge"].items() if k != "seconds"
      }

  def test_a_crashing_metric_judge_is_recorded_and_changes_nothing_else(
    self, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig]
  ) -> None:
    class Crashing(RunJudge):
      def complete_json(self, messages: Sequence[PromptMessage]) -> str:
        try:
          self.purpose_of(messages)
        except AssertionError:
          return '{"verdict": "correct", "reason": "bien"}'
        raise RuntimeError("judge down")

    monkeypatch.setattr(runner, "make_judge", lambda cfg: Crashing())
    outcome = quiet(make_config())
    assert outcome.failed == 0
    record = outcome.record
    for question in record["questions"]:
      assert question["judge"] and question["judge"]["verdict"] == "correct"
    errored = [
      m
      for q in record["questions"]
      for m in q["metrics"] or []
      if m["status"] == "error"
    ]
    assert errored and all(m["error"] for m in errored)
    assert record["metrics"]
    assert record["metrics"]["summary"]["faithfulness"]["errors"] >= 1

  def test_exit_code_is_not_affected_by_metric_errors(
    self, monkeypatch: pytest.MonkeyPatch, project: Path, tmp_path: Path
  ) -> None:
    class Crashing(RunJudge):
      def complete_json(self, messages: Sequence[PromptMessage]) -> str:
        try:
          self.purpose_of(messages)
        except AssertionError:
          return '{"verdict": "correct", "reason": "bien"}'
        raise RuntimeError("judge down")

    monkeypatch.setattr(runner, "make_judge", lambda cfg: Crashing())
    code = cli.main(["run", str(project / "run.toml"), "--out", str(tmp_path / "o")])
    assert code == 0

  def test_failed_question_still_gets_metrics(
    self,
    monkeypatch: pytest.MonkeyPatch,
    scripted: RunJudge,
    make_config: Callable[..., RunConfig],
  ) -> None:
    class Flaky:
      model = "flaky"

      def __init__(self) -> None:
        self.echo = EchoLLM()

      def complete(
        self, messages: Sequence[PromptMessage], params: GenerationParams
      ) -> LLMReply:
        if "rocín" in messages[-1].content:
          raise RuntimeError("model down")
        return self.echo.complete(messages, params)

    monkeypatch.setattr(runner, "make_llm", lambda cfg: Flaky())
    outcome = quiet(make_config())
    failed = [q for q in outcome.record["questions"] if q["error"]]
    assert failed and outcome.failed == len(failed)
    for question in failed:
      assert question["judge"] is None  # the verdict judge skips it
      found = by_name(question)
      assert list(found) == DEFAULT_ON
      assert found["faithfulness"]["status"] == "not_applicable"
      assert found["refusal"]["status"] == "not_applicable"
      assert found["cost"]["status"] == "ok"
    # Counted in the summary as not applicable, not dropped.
    summary = outcome.record["metrics"]["summary"]  # type: ignore[index]
    assert summary["false_refusal"]["not_applicable"] >= len(failed)

  def test_run_where_every_question_fails_still_writes_metrics(
    self, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig]
  ) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
      raise RuntimeError("retrieval down")

    monkeypatch.setattr(pipeline, "_retrieve", boom)
    outcome = quiet(make_config())
    record = outcome.record
    assert outcome.failed == 3
    assert record["metrics"] is not None
    for question in record["questions"]:
      found = by_name(question)
      assert list(found) == DEFAULT_ON
      assert found["doc_retrieval"]["status"] == "not_applicable"
      assert found["cost"]["status"] in {"ok", "not_applicable"}
    summary = record["metrics"]["summary"]
    assert all(item["n"] == 0 for item in summary.values() if item["mean"] is None)


class TestSelection:
  def test_no_metrics(
    self, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    outcome = quiet(make_config(no_metrics=True))
    assert outcome.record["metrics"] is None
    assert all(q["metrics"] is None for q in outcome.record["questions"])
    assert "metrics" in json.loads((outcome.folder / "run.json").read_text("utf-8"))
    assert scripted.count("answer_claims") == 0  # no metric judge call at all
    assert scripted.verdict_calls == 3  # but the verdict judge ran

  def test_no_judge_keeps_only_the_deterministic_families(
    self, scripted: RunJudge, make_config: Callable[..., RunConfig]
  ) -> None:
    outcome = quiet(make_config(no_judge=True))
    assert scripted.requests == [] and scripted.verdict_calls == 0
    for question in outcome.record["questions"]:
      assert question["judge"] is None
      found = by_name(question)
      for name in JUDGED:
        if name == "answer_relevance":
          continue
        assert found[name]["status"] in {"skipped", "not_applicable"}, (
          name,
          found[name],
        )
      # at least one judged family is skipped, not silently missing
      assert any(found[name]["status"] == "skipped" for name in JUDGED)
      for name in DETERMINISTIC:
        assert found[name]["status"] in {"ok", "not_applicable"}
    assert outcome.record["metrics"]
    assert outcome.record["metrics"]["summary"]["faithfulness"]["skipped"] >= 1
    assert outcome.record["metrics"]["summary"]["faithfulness"]["errors"] == 0

  def test_no_judge_with_optional_bleu(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    outcome = quiet(make_config(no_judge=True, metrics=("bleu",)))
    assert outcome.record["metrics"]
    assert "bleu" in outcome.record["metrics"]["selection"]
    ok = [by_name(q)["bleu"]["status"] for q in outcome.record["questions"]]
    assert "ok" in ok
    # registry order: bleu comes after every default-on family
    assert outcome.record["metrics"]["selection"][-1] == "bleu"

  def test_include_order_follows_the_registry_not_the_command_line(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    outcome = quiet(make_config(no_judge=True, metrics=("rouge_l", "bleu")))
    assert outcome.record["metrics"]
    assert outcome.record["metrics"]["selection"][-2:] == ["bleu", "rouge_l"]

  def test_excluding_every_default_family_gives_an_empty_selection(
    self, project: Path, tmp_path: Path
  ) -> None:
    text = (project / "run.toml").read_text(encoding="utf-8")
    names = ", ".join(f'"{n}"' for n in DEFAULT_ON)
    (project / "run.toml").write_text(
      text + f"\n[metrics]\nexclude = [{names}]\n", "utf-8"
    )
    config = load_config(project / "run.toml", Overrides(out=tmp_path / "o"))
    outcome = quiet(config)
    assert outcome.record["metrics"] == {"selection": [], "scores": [], "summary": {}}
    assert [q["metrics"] for q in outcome.record["questions"]] == [[], [], []]

  def test_exclude_beats_include(self, project: Path, tmp_path: Path) -> None:
    text = (project / "run.toml").read_text(encoding="utf-8")
    (project / "run.toml").write_text(
      text + '\n[metrics]\ninclude = ["bleu"]\nexclude = ["bleu", "cost"]\n', "utf-8"
    )
    config = load_config(
      project / "run.toml", Overrides(out=tmp_path / "o", no_judge=True)
    )
    selection = quiet(config).record["metrics"]["selection"]  # type: ignore[index]
    assert "bleu" not in selection and "cost" not in selection

  def test_enabled_false_in_the_run_file(self, project: Path, tmp_path: Path) -> None:
    text = (project / "run.toml").read_text(encoding="utf-8")
    (project / "run.toml").write_text(text + "\n[metrics]\nenabled = false\n", "utf-8")
    config = load_config(project / "run.toml", Overrides(out=tmp_path / "o"))
    outcome = quiet(config)
    assert outcome.record["metrics"] is None
    assert outcome.record["config"]["metrics"]["enabled"] is False

  def test_the_resolved_config_is_stored_with_the_metrics_section(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    outcome = quiet(make_config(metrics=("bleu",), no_judge=True))
    assert outcome.record["config"]["metrics"] == {
      "enabled": True,
      "include": ["bleu"],
      "exclude": [],
    }
    saved = (outcome.folder / "run.toml").read_text(encoding="utf-8")
    assert "[metrics]" in saved

  def test_selection_resolves_with_the_public_selection_class(self) -> None:
    names = [m.name for m in MetricSelection().resolve()]
    assert names == DEFAULT_ON


# ----------------------------------------------------------------------- CLI


class TestCli:
  def run_cli(self, project: Path, tmp_path: Path, *flags: str) -> RunRecord:
    out = tmp_path / "cli-out"
    code = cli.main(["run", str(project / "run.toml"), "--out", str(out), *flags])
    assert code == 0
    [folder] = list(out.iterdir())
    return read_record(folder / "run.json")

  def test_no_metrics(self, project: Path, tmp_path: Path) -> None:
    record = self.run_cli(project, tmp_path, "--no-metrics")
    assert record["schema"] == 2
    assert record["metrics"] is None
    assert all(q["metrics"] is None for q in record["questions"])

  def test_metric_is_repeatable_and_ordered_by_the_registry(
    self, project: Path, tmp_path: Path
  ) -> None:
    record = self.run_cli(
      project, tmp_path, "--no-judge", "--metric", "rouge_l", "--metric", "bleu"
    )
    assert record["metrics"]
    assert record["metrics"]["selection"][-2:] == ["bleu", "rouge_l"]

  def test_no_judge_marks_judged_families_skipped(
    self, project: Path, tmp_path: Path
  ) -> None:
    record = self.run_cli(project, tmp_path, "--no-judge")
    assert record["metrics"]
    assert record["metrics"]["summary"]["faithfulness"]["skipped"] >= 1

  def test_unknown_metric_exits_2_naming_the_key(
    self, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
  ) -> None:
    code = cli.main(
      [
        "run",
        str(project / "run.toml"),
        "--out",
        str(tmp_path / "o"),
        "--metric",
        "nope",
      ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "metrics.include" in err and "nope" in err
    assert not (tmp_path / "o").exists()  # nothing ran

  def test_metric_flag_needs_a_value(self, project: Path) -> None:
    assert cli.main(["run", str(project / "run.toml"), "--metric"]) == 2

  def test_no_metrics_with_metric_does_not_compute(
    self, project: Path, tmp_path: Path
  ) -> None:
    record = self.run_cli(project, tmp_path, "--no-metrics", "--metric", "bleu")
    assert record["metrics"] is None

  def test_help_lists_the_flags(self, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--help"]) == 0
    out = capsys.readouterr().out
    assert "--no-metrics" in out and "--metric" in out

  def test_render_of_the_new_record_matches_the_report_written_by_run(
    self, project: Path, tmp_path: Path
  ) -> None:
    out = tmp_path / "cli-out"
    assert cli.main(["run", str(project / "run.toml"), "--out", str(out)]) == 0
    [folder] = list(out.iterdir())
    assert (folder / "report.html").read_text("utf-8") == render_run(
      read_record(folder / "run.json")
    )


# -------------------------------------------------------- the offline demo


@pytest.fixture
def demo(tmp_path: Path) -> Path:
  target = tmp_path / "demo"
  shutil.copytree(
    DEMO, target, ignore=shutil.ignore_patterns("out", ".hermes-report-cache", "data")
  )
  return target


class TestOfflineDemo:
  def test_writes_schema_2_with_metrics_on_every_question_and_a_summary(
    self, demo: Path, tmp_path: Path
  ) -> None:
    out = tmp_path / "out"
    assert cli.main(["run", str(demo / "run-offline.toml"), "--out", str(out)]) == 0
    [folder] = list(out.iterdir())
    record = read_record(folder / "run.json")
    assert record["schema"] == 2
    assert len(record["questions"]) == 7
    assert record["metrics"]
    assert record["metrics"]["selection"] == DEFAULT_ON
    assert record["metrics"]["summary"]
    for question in record["questions"]:
      assert question["metrics"]
      assert [m["name"] for m in question["metrics"]] == DEFAULT_ON
      found = by_name(question)
      for name in DETERMINISTIC:
        assert found[name]["status"] in {"ok", "not_applicable"}
    # The deterministic scores have values on this run.
    summary = record["metrics"]["summary"]
    assert summary["context_tokens"]["n"] == 7
    assert summary["doc_hit"]["mean"] is not None

  def test_no_metrics_and_no_judge_on_the_demo(
    self, demo: Path, tmp_path: Path
  ) -> None:
    out = tmp_path / "out"
    assert (
      cli.main(
        ["run", str(demo / "run-offline.toml"), "--out", str(out), "--no-metrics"]
      )
      == 0
    )
    [folder] = list(out.iterdir())
    record = read_record(folder / "run.json")
    assert record["metrics"] is None
    out2 = tmp_path / "out2"
    assert (
      cli.main(
        ["run", str(demo / "run-offline.toml"), "--out", str(out2), "--no-judge"]
      )
      == 0
    )
    [folder2] = list(out2.iterdir())
    record2 = read_record(folder2 / "run.json")
    assert record2["metrics"]
    for question in record2["questions"]:
      found = by_name(question)
      assert all(
        found[n]["status"] in {"skipped", "not_applicable"}
        for n in JUDGED
        if n != "answer_relevance"
      )
    assert record2["metrics"]["summary"]["faithfulness"]["skipped"] >= 1
    assert record2["metrics"]["summary"]["faithfulness"]["errors"] == 0


# ------------------------------------------------- schema 1 still reads/renders


class TestSchema1Fixture:
  def test_fixture_is_schema_1_without_any_metrics_key(self) -> None:
    raw = json.loads((SCHEMA1 / "run.json").read_text(encoding="utf-8"))
    assert raw["schema"] == 1
    assert "metrics" not in raw
    assert all("metrics" not in q for q in raw["questions"])
    assert len(raw["questions"]) == 3

  def test_reads_and_fills_metrics_none(self) -> None:
    record = read_record(SCHEMA1 / "run.json")
    assert record["schema"] == 1
    assert record["metrics"] is None
    assert [q["metrics"] for q in record["questions"]] == [None, None, None]

  def test_render_is_byte_for_byte_what_the_old_code_wrote(self) -> None:
    expected = (SCHEMA1 / "report.html").read_text(encoding="utf-8")
    assert render_run(read_record(SCHEMA1 / "run.json")) == expected

  def test_render_does_not_depend_on_the_filled_metrics_keys(self) -> None:
    raw = json.loads((SCHEMA1 / "run.json").read_text(encoding="utf-8"))
    filled = read_record(SCHEMA1 / "run.json")
    assert render_run(raw) == render_run(filled)

  def test_compare_accepts_schema_1_next_to_schema_2(
    self, make_config: Callable[..., RunConfig]
  ) -> None:
    old = read_record(SCHEMA1 / "run.json")
    new = quiet(make_config()).record
    page = render_compare([("old/report.html", old), ("new/report.html", new)])
    assert "<html" in page
    assert page == render_compare([("old/report.html", old), ("new/report.html", new)])
    assert "Comparación de 2 ejecuciones" in page

  def test_write_then_read_back_a_schema_1_record_keeps_it_readable(
    self, tmp_path: Path
  ) -> None:
    from hermes_report.record import write_record

    record = read_record(SCHEMA1 / "run.json")
    path = tmp_path / "again.json"
    write_record(record, path)
    again = read_record(path)
    assert again == record
    assert render_run(again) == (SCHEMA1 / "report.html").read_text(encoding="utf-8")

  def test_render_command_on_the_schema_1_file(self, tmp_path: Path) -> None:
    target = tmp_path / "run.json"
    shutil.copy(SCHEMA1 / "run.json", target)
    code = cli.main(["render", str(target)])
    assert code == 0
    assert (tmp_path / "report.html").read_text("utf-8") == (
      SCHEMA1 / "report.html"
    ).read_text("utf-8")

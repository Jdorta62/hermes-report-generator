"""Tests for ``hermes_report.metrics``: core, judge session, deterministic families.

Everything is offline: scripted fake judge clients, ``CharRatioTokenCounter``
(4 characters per token, so token counts are hand-computable) and plain data.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
from hermes_core.generation import PromptMessage
from hermes_enrich.tokens import CharRatioTokenCounter

import hermes_report.metrics as metrics_pkg
from hermes_report.metrics import (
  METRICS,
  Citation,
  JudgeSession,
  MetricInput,
  MetricResult,
  MetricSelection,
  MetricServices,
  RankedChunk,
  ScoreSummary,
  aggregate,
  evaluate_metrics,
)
from hermes_report.metrics.base import (
  INPUT_FIELDS,
  METRIC_STATUSES,
  JudgeCall,
  Metric,
  MetricError,
  describe,
)
from hermes_report.metrics.deterministic import (
  Cost,
  DocRetrieval,
  EvidenceRetrieval,
  Refusal,
)
from hermes_report.metrics.judge import Claim, ClaimVerdict
from hermes_report.metrics.selection import MetricSelectionError
from hermes_report.providers import ProviderError
from hermes_report.questions import Question

COUNTER = CharRatioTokenCounter(4)
SERVICES = MetricServices(judge=None, embedder=None, counter=COUNTER)


def chunk(
  n: int, text: str = "texto", source: str | None = "a.md", score: float = 0.5
) -> RankedChunk:
  return RankedChunk(f"c{n}", text, source, score)


def make_item(**changes: Any) -> MetricInput:
  base = MetricInput(
    question_id="q1",
    question="¿Qué?",
    expected="Esto.",
    refusal=False,
    answer="Una respuesta.",
    refused=False,
    retrieved=(),
    context=(),
  )
  return replace(base, **changes)


def user(text: str) -> PromptMessage:
  return PromptMessage(role="user", content=text)


# --- fakes --------------------------------------------------------------------


class ScriptedJudge:
  """Replies from a script; an ``Exception`` item is raised instead."""

  model = "scripted"

  def __init__(self, *replies: str | Exception) -> None:
    self.replies = list(replies)
    self.requests: list[Sequence[PromptMessage]] = []

  def complete_json(self, messages: Sequence[PromptMessage]) -> str:
    self.requests.append(messages)
    reply = self.replies.pop(0)
    if isinstance(reply, Exception):
      raise reply
    return reply


def session(*replies: str | Exception) -> tuple[JudgeSession, ScriptedJudge]:
  client = ScriptedJudge(*replies)
  return JudgeSession(client, make_item()), client


class Fake:
  """A configurable Metric for the runner."""

  def __init__(
    self,
    name: str,
    *,
    scores: tuple[str, ...] = ("s",),
    needs: frozenset[str] = frozenset(),
    requires: frozenset[str] = frozenset(),
    default_enabled: bool = True,
    raises: BaseException | None = None,
    value: float | None = 1.0,
  ) -> None:
    self.name = name
    self.default_enabled = default_enabled
    self.scores = scores
    self.needs = needs
    self.requires = requires
    self._raises = raises
    self._value = value
    self.runs = 0

  def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
    self.runs += 1
    if self._raises is not None:
      raise self._raises
    return MetricResult(self.name, "ok", {s: self._value for s in self.scores})


# --- base ---------------------------------------------------------------------


class TestMetricInputHas:
  @pytest.mark.parametrize(
    ("changes", "field", "expected"),
    [
      ({"answer": "x"}, "answer", True),
      ({"answer": None}, "answer", False),
      ({"answer": ""}, "answer", False),
      ({"answer": "  \n\t"}, "answer", False),
      ({"expected": "   "}, "expected", False),
      ({"expected": "x"}, "expected", True),
      ({"retrieved": ()}, "retrieved", False),
      ({"retrieved": (chunk(1),)}, "retrieved", True),
      ({"context": ()}, "context", False),
      ({"context": ("",)}, "context", True),
      ({"documents": ()}, "documents", False),
      ({"documents": ("a.md",)}, "documents", True),
      ({"evidence": ()}, "evidence", False),
      ({"evidence": ("x",)}, "evidence", True),
      ({"citations": None}, "citations", False),
      ({"citations": ()}, "citations", False),
      ({"citations": (Citation(1, 1),)}, "citations", True),
    ],
  )
  def test_usable_data(
    self, changes: dict[str, Any], field: str, expected: bool
  ) -> None:
    assert make_item(**changes).has(field) is expected

  @pytest.mark.parametrize("field", ["seconds", "question", "", "ANSWER", "refused"])
  def test_unknown_field_raises(self, field: str) -> None:
    with pytest.raises(ValueError, match="campo de entrada desconocido"):
      make_item().has(field)

  def test_every_input_field_is_accepted(self) -> None:
    item = make_item()
    for field in INPUT_FIELDS:
      assert isinstance(item.has(field), bool)

  def test_input_is_frozen(self) -> None:
    with pytest.raises(FrozenInstanceError):
      make_item().answer = "otra"  # type: ignore[misc]


class TestDescribe:
  def test_with_message(self) -> None:
    assert describe(ValueError("mal")) == "ValueError: mal"

  def test_without_message_is_never_empty(self) -> None:
    assert describe(KeyError()) == "KeyError"
    assert describe(ValueError()) == "ValueError"


class TestMetricResult:
  def test_not_applicable_sorts_and_accepts_any_collection(self) -> None:
    result = MetricResult.not_applicable("m", {"retrieved", "documents"})
    assert result.status == "not_applicable"
    assert result.scores == {}
    assert result.details == {"missing": ["documents", "retrieved"]}

  def test_skipped(self) -> None:
    result = MetricResult.skipped("m", "sin juez")
    assert (result.status, result.scores, result.details) == (
      "skipped",
      {},
      {"reason": "sin juez"},
    )

  def test_failed_keeps_calls_and_error(self) -> None:
    call = JudgeCall("p", (), None, "boom", 0.1)
    result = MetricResult.failed("m", "X: y", [call])
    assert result.status == "error"
    assert result.error == "X: y"
    assert result.calls == (call,)
    assert result.scores == {}

  def test_statuses_cover_every_constructor(self) -> None:
    built = [
      MetricResult.not_applicable("m", []),
      MetricResult.skipped("m", "r"),
      MetricResult.failed("m", "e", []),
      MetricResult("m", "ok", {}),
    ]
    assert {r.status for r in built} == set(METRIC_STATUSES)

  def test_ok_record_round_trips_through_json(self) -> None:
    call = JudgeCall(
      "p", ({"role": "user", "content": "¿hola?"},), '{"a": 1}', None, 0.25
    )
    result = MetricResult(
      "m",
      "ok",
      {"a": 1.0, "b": None},
      {
        "claims": (Claim("x", (1, 2)),),
        "verdicts": [ClaimVerdict(1, True, (1,))],
        "ids": {3, 1, 2},
        "nested": {"t": (1, (2, 3))},
        "key": {1: "uno"},
        "citations": (Citation(1, 0),),
        "none": None,
      },
      (call,),
      None,
      0.5,
    )
    record = result.to_record()
    assert json.loads(json.dumps(record)) == record
    assert record["details"]["claims"] == [{"text": "x", "markers": [1, 2]}]
    assert record["details"]["key"] == {"1": "uno"}
    assert sorted(record["details"]["ids"]) == [1, 2, 3]
    assert record["calls"][0]["messages"] == [{"role": "user", "content": "¿hola?"}]
    assert record["scores"] == {"a": 1.0, "b": None}

  @pytest.mark.parametrize(
    "result",
    [
      MetricResult.not_applicable("m", ["a"]),
      MetricResult.skipped("m", "sin juez"),
      MetricResult.failed("m", "E: x", [JudgeCall("p", (), None, "e", 0.0)]),
    ],
  )
  def test_other_statuses_round_trip(self, result: MetricResult) -> None:
    record = result.to_record()
    assert json.loads(json.dumps(record)) == record
    assert record["status"] == result.status

  def test_record_does_not_alias_scores(self) -> None:
    result = MetricResult("m", "ok", {"a": 1.0})
    record = result.to_record()
    record["scores"]["a"] = 9.0
    assert result.scores == {"a": 1.0}


# --- judge session ------------------------------------------------------------


def parse_int(data: Any) -> int:
  value = data["n"]
  if not isinstance(value, int):
    raise MetricError("n debe ser entero")
  return int(value)


class TestJudgeSession:
  def test_ask_parses_and_logs_one_call(self) -> None:
    judge, client = session('{"n": 4}')
    assert judge.ask("p", [user("hola")], parse_int) == 4
    assert len(client.requests) == 1
    (call,) = judge.calls
    assert call.purpose == "p"
    assert call.messages == ({"role": "user", "content": "hola"},)
    assert call.raw == '{"n": 4}'
    assert call.error is None
    assert call.seconds >= 0

  def test_memo_by_purpose_does_not_call_again(self) -> None:
    judge, client = session('{"n": 4}')
    assert judge.ask("p", [user("a")], parse_int) == 4
    assert judge.ask("p", [user("otra cosa")], parse_int) == 4
    assert len(client.requests) == 1
    assert len(judge.calls) == 1

  def test_memo_by_explicit_key(self) -> None:
    judge, client = session('{"n": 1}', '{"n": 2}')
    assert judge.ask("verify", [user("a")], parse_int, key="k1") == 1
    assert judge.ask("verify", [user("b")], parse_int, key="k2") == 2
    assert judge.ask("verify", [user("a")], parse_int, key="k1") == 1
    assert len(client.requests) == 2

  def test_same_key_other_purpose_hits_memo(self) -> None:
    judge, client = session('{"n": 1}')
    judge.ask("one", [user("a")], parse_int, key="k")
    assert judge.ask("two", [user("a")], parse_int, key="k") == 1
    assert len(client.requests) == 1

  @pytest.mark.parametrize("value", ["0", "[]", "null", "false", '""'])
  def test_falsy_values_are_memoised(self, value: str) -> None:
    judge, client = session(f'{{"v": {value}}}')
    first = judge.ask("p", [user("a")], lambda d: d["v"])
    assert judge.ask("p", [user("a")], lambda d: d["v"]) == first
    assert len(client.requests) == 1

  def test_provider_error_is_memoised_and_reraised_same_instance(self) -> None:
    judge, client = session(ProviderError("sin red"))
    with pytest.raises(MetricError, match="ProviderError: sin red") as first:
      judge.ask("p", [user("a")], parse_int)
    with pytest.raises(MetricError) as second:
      judge.ask("p", [user("a")], parse_int)
    assert second.value is first.value
    assert len(client.requests) == 1
    (call,) = judge.calls
    assert call.raw is None
    assert call.error is not None
    assert "sin red" in call.error

  def test_provider_error_is_chained(self) -> None:
    judge, _ = session(ProviderError("sin red"))
    with pytest.raises(MetricError) as info:
      judge.ask("p", [user("a")], parse_int)
    assert isinstance(info.value.__cause__, ProviderError)

  @pytest.mark.parametrize("raw", ["", "no es json", '{"n": ', "{'n': 1}"])
  def test_invalid_json_is_memoised_with_its_raw_text(self, raw: str) -> None:
    judge, client = session(raw)
    with pytest.raises(MetricError):
      judge.ask("p", [user("a")], parse_int)
    with pytest.raises(MetricError):
      judge.ask("p", [user("a")], parse_int)
    assert len(client.requests) == 1
    assert judge.calls[0].raw == raw
    assert judge.calls[0].error

  def test_parse_metric_error_is_logged_once_with_raw_and_memoised(self) -> None:
    judge, client = session('{"n": "x"}')
    with pytest.raises(MetricError, match="n debe ser entero"):
      judge.ask("p", [user("a")], parse_int)
    with pytest.raises(MetricError, match="n debe ser entero"):
      judge.ask("p", [user("a")], parse_int)
    assert len(client.requests) == 1
    assert len(judge.calls) == 1
    assert judge.calls[0].raw == '{"n": "x"}'
    assert judge.calls[0].error == "n debe ser entero"

  def test_parse_value_error_becomes_metric_error(self) -> None:
    def parse(data: Any) -> int:
      raise ValueError("forma rara")

    judge, _ = session("{}")
    with pytest.raises(MetricError, match="ValueError: forma rara"):
      judge.ask("p", [user("a")], parse)

  def test_failure_of_one_key_does_not_poison_another(self) -> None:
    judge, client = session(ProviderError("x"), '{"n": 2}')
    with pytest.raises(MetricError):
      judge.ask("p", [user("a")], parse_int, key="bad")
    assert judge.ask("p", [user("a")], parse_int, key="good") == 2
    assert len(client.requests) == 2

  def test_empty_key_falls_back_to_purpose(self) -> None:
    judge, client = session('{"n": 1}')
    judge.ask("p", [user("a")], parse_int, key="")
    judge.ask("p", [user("a")], parse_int)
    assert len(client.requests) == 1

  def test_calls_since(self) -> None:
    judge, _ = session('{"n": 1}', '{"n": 2}', '{"n": 3}')
    judge.ask("a", [user("x")], parse_int)
    mark = len(judge.calls)
    judge.ask("b", [user("x")], parse_int)
    judge.ask("c", [user("x")], parse_int)
    assert [c.purpose for c in judge.calls_since(mark)] == ["b", "c"]
    assert judge.calls_since(len(judge.calls)) == ()
    assert judge.calls_since(99) == ()
    assert len(judge.calls_since(0)) == 3

  def test_calls_since_returns_a_snapshot(self) -> None:
    judge, _ = session('{"n": 1}', '{"n": 2}')
    judge.ask("a", [user("x")], parse_int)
    snapshot = judge.calls_since(0)
    judge.ask("b", [user("x")], parse_int)
    assert len(snapshot) == 1

  def test_every_network_attempt_is_logged_even_when_it_fails(self) -> None:
    judge, _ = session(ProviderError("x"), "{bad", '{"n": 1}')
    for key in ("a", "b", "c"):
      try:
        judge.ask("p", [user("a")], parse_int, key=key)
      except MetricError:
        pass
    assert [c.error is None for c in judge.calls] == [False, False, True]


# --- selection ----------------------------------------------------------------


class TestSelection:
  def test_registry_order_and_names(self) -> None:
    assert list(METRICS) == ["doc_retrieval", "evidence_retrieval", "refusal", "cost"]
    for key, metric in METRICS.items():
      assert metric.name == key

  def test_registry_is_read_only(self) -> None:
    with pytest.raises(TypeError):
      METRICS["x"] = Cost()  # type: ignore[index]

  def test_every_metric_honours_the_contract(self) -> None:
    for metric in METRICS.values():
      assert metric.default_enabled is True
      assert len(set(metric.scores)) == len(metric.scores) > 0
      assert metric.requires <= set(INPUT_FIELDS)
      assert metric.needs <= {"judge", "embedder"}

  def test_default_runs_every_default_on_family_in_order(self) -> None:
    assert [m.name for m in MetricSelection().resolve()] == list(METRICS)

  def test_disabled_runs_nothing(self) -> None:
    assert MetricSelection(enabled=False).resolve() == ()
    assert MetricSelection(enabled=False, include=("cost",)).resolve() == ()

  def test_exclude(self) -> None:
    names = [m.name for m in MetricSelection(exclude=("refusal", "cost")).resolve()]
    assert names == ["doc_retrieval", "evidence_retrieval"]

  def test_exclude_everything_is_empty(self) -> None:
    assert MetricSelection(exclude=tuple(METRICS)).resolve() == ()

  def test_include_of_a_default_on_family_is_a_no_op(self) -> None:
    assert [m.name for m in MetricSelection(include=("cost",)).resolve()] == list(
      METRICS
    )

  def test_exclude_wins_over_include(self) -> None:
    names = [
      m.name for m in MetricSelection(include=("cost",), exclude=("cost",)).resolve()
    ]
    assert "cost" not in names

  def test_duplicates_are_harmless(self) -> None:
    selection = MetricSelection(
      include=("cost", "cost"), exclude=("refusal", "refusal")
    )
    assert [m.name for m in selection.resolve()] == [
      "doc_retrieval",
      "evidence_retrieval",
      "cost",
    ]

  def test_include_turns_on_an_off_by_default_family_in_registry_order(self) -> None:
    registry = {
      "a": Fake("a"),
      "b": Fake("b", default_enabled=False),
      "c": Fake("c"),
    }
    assert [m.name for m in MetricSelection().resolve(registry)] == ["a", "c"]
    selection = MetricSelection(include=("b",))
    assert [m.name for m in selection.resolve(registry)] == ["a", "b", "c"]
    only = MetricSelection(include=("b",), exclude=("a", "c"))
    assert [m.name for m in only.resolve(registry)] == ["b"]

  @pytest.mark.parametrize(
    "selection",
    [
      MetricSelection(include=("nope",)),
      MetricSelection(exclude=("nope",)),
      MetricSelection(enabled=False, include=("nope",)),
      MetricSelection(include=("cost", "nope"), exclude=("tambien",)),
      MetricSelection(include=("",)),
      MetricSelection(include=("COST",)),
    ],
  )
  def test_unknown_names_are_rejected(self, selection: MetricSelection) -> None:
    with pytest.raises(MetricSelectionError) as info:
      selection.resolve()
    assert info.value.names
    assert info.value.available == tuple(METRICS)

  def test_error_lists_every_unknown_name_in_spanish(self) -> None:
    selection = MetricSelection(include=("uno",), exclude=("dos",))
    with pytest.raises(MetricSelectionError) as info:
      selection.resolve()
    message = str(info.value)
    assert message.startswith("Métrica desconocida: uno, dos")
    assert "Disponibles: doc_retrieval" in message
    assert info.value.names == ("uno", "dos")
    assert isinstance(info.value, ValueError)


# --- deterministic: doc_retrieval ---------------------------------------------


def docs(sources: Sequence[str | None], documents: Sequence[str]) -> MetricInput:
  return make_item(
    retrieved=tuple(chunk(i, source=s) for i, s in enumerate(sources, 1)),
    documents=tuple(documents),
  )


def doc_scores(item: MetricInput) -> dict[str, float | None]:
  result = DocRetrieval().compute(item, SERVICES)
  assert result.status == "ok"
  return result.scores


class TestDocRetrieval:
  @pytest.mark.parametrize(
    ("sources", "documents", "hit", "recall", "mrr"),
    [
      (["a.md", "b.md", "c.md"], ["a.md"], 1.0, 1.0, 1.0),
      (["a.md", "b.md", "c.md"], ["b.md"], 1.0, 1.0, 0.5),
      (["a.md", "b.md", "c.md"], ["c.md"], 1.0, 1.0, 1 / 3),
      (["a.md", "b.md", "c.md"], ["z.md"], 0.0, 0.0, 0.0),
      (["a.md", "b.md", "c.md"], ["a.md", "z.md"], 1.0, 0.5, 1.0),
      (["a.md", "b.md", "c.md"], ["c.md", "b.md"], 1.0, 1.0, 0.5),
      (["x.md", "y.md", "z.md", "a.md"], ["a.md", "q.md", "r.md"], 1.0, 1 / 3, 0.25),
      (["a.md", "a.md", "a.md"], ["a.md"], 1.0, 1.0, 1.0),
      (["a.md", "a.md", "b.md"], ["a.md", "b.md"], 1.0, 1.0, 1.0),
      (["a.md"], ["a.md", "a.md"], 1.0, 1.0, 1.0),
      ([None, "b.md"], ["b.md"], 1.0, 1.0, 0.5),
      ([None, "b.md"], ["z.md"], 0.0, 0.0, 0.0),
    ],
  )
  def test_hand_computed(
    self,
    sources: list[str | None],
    documents: list[str],
    hit: float,
    recall: float,
    mrr: float,
  ) -> None:
    scores = doc_scores(docs(sources, documents))
    assert scores == {
      "doc_hit": hit,
      "doc_recall": pytest.approx(recall),
      "doc_mrr": pytest.approx(mrr),
    }

  def test_details(self) -> None:
    result = DocRetrieval().compute(docs(["a.md", "b.md"], ["b.md", "z.md"]), SERVICES)
    assert result.details == {
      "expected": ["b.md", "z.md"],
      "found": ["b.md"],
      "first_rank": 2,
    }

  def test_no_hit_has_no_first_rank(self) -> None:
    result = DocRetrieval().compute(docs(["a.md"], ["z.md"]), SERVICES)
    assert result.details["first_rank"] is None

  def test_recall_is_never_above_one(self) -> None:
    scores = doc_scores(docs(["a.md"] * 5, ["a.md"]))
    assert scores["doc_recall"] == 1.0

  def test_matching_is_exact_not_by_suffix_or_case(self) -> None:
    scores = doc_scores(docs(["docs/a.md", "A.MD"], ["a.md"]))
    assert scores["doc_hit"] == 0.0

  def test_scores_match_the_declared_names(self) -> None:
    scores = doc_scores(docs(["a.md"], ["a.md"]))
    assert tuple(scores) == DocRetrieval.scores

  def test_no_source_anywhere_is_not_applicable(self) -> None:
    result = DocRetrieval().compute(docs([None, None], ["a.md"]), SERVICES)
    assert result.status == "not_applicable"
    assert result.details == {"missing": ["source"]}
    assert result.scores == {}

  def test_no_source_through_the_runner(self) -> None:
    (result,) = evaluate_metrics(docs([None], ["a.md"]), [DocRetrieval()], SERVICES)
    assert result.status == "not_applicable"
    assert result.details["missing"] == ["source"]

  @pytest.mark.parametrize(
    ("item", "missing"),
    [
      (docs(["a.md"], []), ["documents"]),
      (docs([], ["a.md"]), ["retrieved"]),
      (docs([], []), ["documents", "retrieved"]),
    ],
  )
  def test_missing_inputs_through_the_runner(
    self, item: MetricInput, missing: list[str]
  ) -> None:
    (result,) = evaluate_metrics(item, [DocRetrieval()], SERVICES)
    assert result.status == "not_applicable"
    assert result.details["missing"] == missing

  def test_empty_string_source_is_a_source_not_missing(self) -> None:
    # An empty ``source`` is data, not "unknown": it must not match a document.
    result = DocRetrieval().compute(docs([""], ["a.md"]), SERVICES)
    assert result.status == "ok"
    assert result.scores["doc_hit"] == 0.0


# --- deterministic: evidence_retrieval ----------------------------------------

TAJO = "El río Tajo nace en Albarracín"
LISBOA = "y desemboca en Lisboa"


def ev(
  texts: Sequence[str], evidence: Sequence[str], context: Sequence[str] | None = None
) -> MetricInput:
  return make_item(
    retrieved=tuple(chunk(i, text=t) for i, t in enumerate(texts, 1)),
    evidence=tuple(evidence),
    context=tuple(texts if context is None else context),
  )


def ev_scores(item: MetricInput) -> dict[str, float | None]:
  result = EvidenceRetrieval().compute(item, SERVICES)
  assert result.status == "ok"
  return result.scores


class TestEvidenceRetrieval:
  def test_hit_at_rank_one(self) -> None:
    assert ev_scores(ev([TAJO, LISBOA, "otro"], ["Tajo"])) == {
      "evidence_hit": 1.0,
      "evidence_recall": 1.0,
      "evidence_mrr": 1.0,
      "evidence_recall_context": 1.0,
      "budget_loss": 0.0,
    }

  def test_hit_at_rank_k(self) -> None:
    scores = ev_scores(ev(["uno", "dos", TAJO], ["Albarracín"]))
    assert scores["evidence_hit"] == 1.0
    assert scores["evidence_mrr"] == pytest.approx(1 / 3)

  def test_no_hit(self) -> None:
    assert ev_scores(ev([TAJO, LISBOA], ["Zaragoza"])) == {
      "evidence_hit": 0.0,
      "evidence_recall": 0.0,
      "evidence_mrr": 0.0,
      "evidence_recall_context": 0.0,
      "budget_loss": 0.0,
    }

  def test_two_phrases_in_two_chunks(self) -> None:
    result = EvidenceRetrieval().compute(
      ev([TAJO, LISBOA], ["Tajo", "Lisboa"]), SERVICES
    )
    assert result.scores["evidence_hit"] == 1.0
    assert result.scores["evidence_recall"] == 1.0
    assert result.scores["evidence_mrr"] == 0.5
    assert result.details == {"found_at": {"Tajo": 1, "Lisboa": 2}}

  def test_partial_phrases(self) -> None:
    result = EvidenceRetrieval().compute(
      ev([TAJO, LISBOA], ["Tajo", "Zaragoza"]), SERVICES
    )
    assert result.scores["evidence_hit"] == 0.0
    assert result.scores["evidence_recall"] == 0.5
    assert result.scores["evidence_mrr"] == 0.0
    assert result.scores["evidence_recall_context"] == 0.5
    assert result.details["found_at"] == {"Tajo": 1, "Zaragoza": None}

  def test_one_phrase_split_across_two_chunks(self) -> None:
    result = EvidenceRetrieval().compute(
      ev(["...el río Tajo nace en", "Albarracín y baja..."], ["nace en Albarracín"]),
      SERVICES,
    )
    assert result.scores["evidence_hit"] == 1.0
    assert result.scores["evidence_mrr"] == 0.5
    assert result.details["found_at"] == {"nace en Albarracín": 2}
    assert result.scores["evidence_recall_context"] == 1.0

  def test_split_phrase_with_a_chunk_between_is_not_found(self) -> None:
    # Ranks 1 and 3 hold the halves, but rank 2 sits between them in the join.
    result = EvidenceRetrieval().compute(
      ev(["nace en", "ruido", "Albarracín"], ["nace en Albarracín"]), SERVICES
    )
    assert result.details["found_at"] == {"nace en Albarracín": None}

  def test_matching_ignores_case_accents_and_whitespace(self) -> None:
    result = EvidenceRetrieval().compute(
      ev(["EL   RIO\nTAJO"], ["el río tajo"]), SERVICES
    )
    assert result.scores["evidence_hit"] == 1.0

  def test_evidence_retrieved_but_not_in_context_is_budget_loss(self) -> None:
    item = ev([TAJO, LISBOA], ["Tajo", "Lisboa"], context=[TAJO])
    scores = ev_scores(item)
    assert scores["evidence_recall"] == 1.0
    assert scores["evidence_recall_context"] == 0.5
    assert scores["budget_loss"] == 0.5

  def test_everything_lost_in_the_context(self) -> None:
    scores = ev_scores(ev([TAJO], ["Tajo"], context=["otra cosa"]))
    assert scores["evidence_recall_context"] == 0.0
    assert scores["budget_loss"] == 1.0

  def test_empty_context_loses_everything_retrieved(self) -> None:
    scores = ev_scores(ev([TAJO], ["Tajo"], context=[]))
    assert scores["evidence_recall_context"] == 0.0
    assert scores["budget_loss"] == 1.0

  def test_budget_loss_is_floored_at_zero(self) -> None:
    # The evidence is in the context but was never retrieved (e.g. enrichment added it).
    scores = ev_scores(ev(["nada"], ["Tajo"], context=[TAJO]))
    assert scores["evidence_recall"] == 0.0
    assert scores["evidence_recall_context"] == 1.0
    assert scores["budget_loss"] == 0.0

  def test_phrase_split_across_context_pieces_counts_like_the_rules(self) -> None:
    scores = ev_scores(
      ev(
        [TAJO + " " + LISBOA],
        ["Albarracín y desemboca"],
        context=["Albarracín", "y desemboca"],
      )
    )
    assert scores["evidence_recall_context"] == 1.0

  def test_duplicate_phrases_count_once(self) -> None:
    scores = ev_scores(ev([TAJO], ["Tajo", "Tajo", "Zaragoza"]))
    assert scores["evidence_recall"] == 0.5

  def test_blank_phrase_is_never_found(self) -> None:
    scores = ev_scores(ev([TAJO], ["", "Tajo"]))
    assert scores["evidence_recall"] == 0.5
    assert scores["evidence_hit"] == 0.0

  def test_single_blank_phrase_does_not_crash_or_match(self) -> None:
    (result,) = evaluate_metrics(ev([TAJO], [""]), [EvidenceRetrieval()], SERVICES)
    assert result.status == "ok"
    assert result.scores["evidence_hit"] == 0.0

  def test_scores_match_the_declared_names(self) -> None:
    assert tuple(ev_scores(ev([TAJO], ["Tajo"]))) == EvidenceRetrieval.scores

  def test_found_at_is_json_friendly(self) -> None:
    result = EvidenceRetrieval().compute(ev([TAJO], ["Tajo", "Zaragoza"]), SERVICES)
    record = result.to_record()
    assert json.loads(json.dumps(record)) == record

  @pytest.mark.parametrize(
    ("item", "missing"),
    [
      (ev([TAJO], []), ["evidence"]),
      (ev([], ["Tajo"]), ["retrieved"]),
      (ev([], []), ["evidence", "retrieved"]),
    ],
  )
  def test_missing_inputs_through_the_runner(
    self, item: MetricInput, missing: list[str]
  ) -> None:
    (result,) = evaluate_metrics(item, [EvidenceRetrieval()], SERVICES)
    assert result.status == "not_applicable"
    assert result.details["missing"] == missing

  def test_agrees_with_the_rules_on_random_inputs(self) -> None:
    # The metric must give the same verdict as Question.reachable over the
    # "\n"-joined chunks, which is what the rules use for ``in_retrieval``.
    rng = random.Random(7)
    vocabulary = ["río", "Tajo", "nace", "Albarracín", "mar", "Lisboa", "x", "Ñu"]
    for _ in range(300):
      texts = [
        " ".join(rng.choices(vocabulary, k=rng.randint(0, 4)))
        for _ in range(rng.randint(1, 4))
      ]
      phrases = [
        " ".join(rng.choices(vocabulary, k=rng.randint(1, 3)))
        for _ in range(rng.randint(1, 3))
      ]
      context = rng.sample(texts, k=rng.randint(0, len(texts)))
      question = Question("q", "¿?", "r", evidence=tuple(phrases))
      scores = ev_scores(ev(texts, phrases, context=context))
      assert scores["evidence_hit"] == float(
        question.reachable("\n".join(texts)) is True
      )
      assert (scores["evidence_recall_context"] == 1.0) == (
        question.reachable("\n".join(context)) is True
      )
      assert scores["evidence_recall"] is not None
      assert 0.0 <= scores["evidence_recall"] <= 1.0
      assert scores["evidence_mrr"] is not None
      assert (scores["evidence_mrr"] > 0) == (scores["evidence_hit"] == 1.0)
      if scores["evidence_hit"] == 1.0:
        assert scores["evidence_recall"] == 1.0


# --- deterministic: refusal ---------------------------------------------------


def refusal_scores(item: MetricInput) -> dict[str, float | None]:
  result = Refusal().compute(item, SERVICES)
  assert result.status == "ok"
  return result.scores


class TestRefusal:
  def test_answerable_and_answered(self) -> None:
    assert refusal_scores(make_item(refusal=False, refused=False)) == {
      "false_refusal": 0.0,
      "invented_answer": None,
    }

  def test_answerable_refused_with_evidence_in_context_is_a_false_refusal(self) -> None:
    item = make_item(refusal=False, refused=True, evidence=("Tajo",), context=(TAJO,))
    assert refusal_scores(item) == {"false_refusal": 1.0, "invented_answer": None}

  def test_answerable_refused_with_evidence_missing_is_not_scored(self) -> None:
    item = make_item(refusal=False, refused=True, evidence=("Tajo",), context=("nada",))
    assert refusal_scores(item)["false_refusal"] is None

  def test_refused_with_only_part_of_the_evidence_is_not_scored(self) -> None:
    item = make_item(
      refusal=False, refused=True, evidence=("Tajo", "Lisboa"), context=(TAJO,)
    )
    assert refusal_scores(item)["false_refusal"] is None

  def test_refused_with_evidence_split_across_pieces_is_a_false_refusal(self) -> None:
    item = make_item(
      refusal=False,
      refused=True,
      evidence=("Albarracín y desemboca",),
      context=("Albarracín", "y desemboca"),
    )
    assert refusal_scores(item)["false_refusal"] == 1.0

  def test_refused_without_any_evidence_is_not_scored(self) -> None:
    item = make_item(refusal=False, refused=True, evidence=(), context=(TAJO,))
    assert refusal_scores(item)["false_refusal"] is None

  def test_unanswerable_and_refused(self) -> None:
    assert refusal_scores(make_item(refusal=True, refused=True)) == {
      "false_refusal": None,
      "invented_answer": 0.0,
    }

  def test_unanswerable_and_answered_is_an_invented_answer(self) -> None:
    assert refusal_scores(make_item(refusal=True, refused=False)) == {
      "false_refusal": None,
      "invented_answer": 1.0,
    }

  def test_unanswerable_with_evidence_in_context_never_scores_false_refusal(
    self,
  ) -> None:
    item = make_item(refusal=True, refused=True, evidence=("Tajo",), context=(TAJO,))
    assert refusal_scores(item)["false_refusal"] is None

  def test_unknown_refused_flag_scores_nothing(self) -> None:
    for refusal in (False, True):
      assert refusal_scores(make_item(refusal=refusal, refused=None)) == {
        "false_refusal": None,
        "invented_answer": None,
      }

  def test_scores_match_the_declared_names(self) -> None:
    assert tuple(refusal_scores(make_item())) == Refusal.scores

  @pytest.mark.parametrize("answer", [None, "", "   "])
  def test_no_answer_is_not_applicable_through_the_runner(
    self, answer: str | None
  ) -> None:
    (result,) = evaluate_metrics(
      make_item(answer=answer, refused=None), [Refusal()], SERVICES
    )
    assert result.status == "not_applicable"
    assert result.details["missing"] == ["answer"]

  def test_the_two_scores_are_never_both_set(self) -> None:
    for refusal in (False, True):
      for refused in (None, False, True):
        scores = refusal_scores(make_item(refusal=refusal, refused=refused))
        assert not (
          scores["false_refusal"] is not None and scores["invented_answer"] is not None
        )


# --- deterministic: cost ------------------------------------------------------


def cost_scores(item: MetricInput) -> dict[str, float | None]:
  result = Cost().compute(item, SERVICES)
  assert result.status == "ok"
  return result.scores


class TestCost:
  def test_all_data_hand_computed(self) -> None:
    item = make_item(
      seconds={"retrieval": 0.5, "enrich": 0.25, "generation": 2.0},
      context=("abcd", "efgh"),  # "abcd\nefgh" = 9 chars -> 3 tokens
      prompt=(user("abcdefgh"), user("abcd")),  # 2 + 1
      replies=("abcd", "ab"),  # 1 + 1
    )
    assert cost_scores(item) == {
      "seconds_retrieval": 0.5,
      "seconds_enrich": 0.25,
      "seconds_generation": 2.0,
      "context_tokens": 3.0,
      "prompt_tokens": 3.0,
      "reply_tokens": 2.0,
    }

  def test_nothing_known_is_none_not_zero(self) -> None:
    assert cost_scores(make_item()) == {
      "seconds_retrieval": None,
      "seconds_enrich": None,
      "seconds_generation": None,
      "context_tokens": None,
      "prompt_tokens": None,
      "reply_tokens": None,
    }

  def test_a_phase_at_zero_seconds_is_kept(self) -> None:
    scores = cost_scores(make_item(seconds={"retrieval": 0.0}))
    assert scores["seconds_retrieval"] == 0.0
    assert scores["seconds_enrich"] is None

  def test_unknown_phase_keys_are_ignored(self) -> None:
    assert cost_scores(make_item(seconds={"judge": 9.0}))["seconds_generation"] is None

  def test_empty_texts_count_zero_tokens(self) -> None:
    scores = cost_scores(make_item(context=("",), prompt=(user(""),), replies=("",)))
    assert (
      scores["context_tokens"],
      scores["prompt_tokens"],
      scores["reply_tokens"],
    ) == (0.0, 0.0, 0.0)

  def test_values_are_floats(self) -> None:
    scores = cost_scores(make_item(context=("abcd",), seconds={"retrieval": 1}))
    assert isinstance(scores["context_tokens"], float)

  def test_uses_the_counter_of_the_services(self) -> None:
    services = MetricServices(None, None, CharRatioTokenCounter(1))
    result = Cost().compute(make_item(context=("abcd",)), services)
    assert result.scores["context_tokens"] == 4.0
    assert result.details == {"counter": "char_ratio"}

  def test_never_not_applicable(self) -> None:
    (result,) = evaluate_metrics(
      make_item(answer=None, refused=None), [Cost()], SERVICES
    )
    assert result.status == "ok"

  def test_scores_match_the_declared_names(self) -> None:
    assert tuple(cost_scores(make_item())) == Cost.scores


# --- runner -------------------------------------------------------------------


class TestEvaluateMetrics:
  def test_a_raising_metric_is_an_error_and_the_next_still_runs(self) -> None:
    boom = Fake("boom", raises=RuntimeError("mal"))
    after = Fake("after")
    results = evaluate_metrics(make_item(), [boom, after], SERVICES)
    assert [r.name for r in results] == ["boom", "after"]
    assert results[0].status == "error"
    assert results[0].error == "RuntimeError: mal"
    assert results[0].scores == {}
    assert results[1].status == "ok"
    assert after.runs == 1

  @pytest.mark.parametrize(
    "exc",
    [
      MetricError("x"),
      ZeroDivisionError(),
      KeyError("k"),
      ValueError(),
      AssertionError("a"),
    ],
  )
  def test_any_exception_is_isolated(self, exc: Exception) -> None:
    (result,) = evaluate_metrics(make_item(), [Fake("m", raises=exc)], SERVICES)
    assert result.status == "error"
    assert result.error

  def test_not_applicable_does_not_run_the_metric(self) -> None:
    metric = Fake("m", requires=frozenset({"retrieved", "answer"}))
    item = make_item(answer=None)
    (result,) = evaluate_metrics(item, [metric], SERVICES)
    assert result.status == "not_applicable"
    assert result.details["missing"] == ["answer", "retrieved"]
    assert metric.runs == 0

  def test_unknown_required_field_is_an_error_not_an_exception(self) -> None:
    results = evaluate_metrics(
      make_item(), [Fake("bad", requires=frozenset({"nope"})), Fake("next")], SERVICES
    )
    assert results[0].status == "error"
    assert "nope" in (results[0].error or "")
    assert results[1].status == "ok"

  def test_needs_judge_without_judge_is_skipped(self) -> None:
    metric = Fake("m", needs=frozenset({"judge"}))
    (result,) = evaluate_metrics(make_item(), [metric], SERVICES)
    assert (result.status, result.details) == ("skipped", {"reason": "sin juez"})
    assert metric.runs == 0

  def test_needs_embedder_without_embedder_is_skipped(self) -> None:
    judge, _ = session()
    services = MetricServices(judge, None, COUNTER)
    metric = Fake("m", needs=frozenset({"judge", "embedder"}))
    (result,) = evaluate_metrics(make_item(), [metric], services)
    assert (result.status, result.details) == ("skipped", {"reason": "sin embedder"})

  def test_missing_input_wins_over_missing_service(self) -> None:
    metric = Fake("m", needs=frozenset({"judge"}), requires=frozenset({"retrieved"}))
    (result,) = evaluate_metrics(make_item(), [metric], SERVICES)
    assert result.status == "not_applicable"

  def test_service_present_runs_the_metric(self) -> None:
    judge, _ = session()
    services = MetricServices(judge, None, COUNTER)
    metric = Fake("m", needs=frozenset({"judge"}))
    (result,) = evaluate_metrics(make_item(), [metric], services)
    assert result.status == "ok"

  def test_error_carries_only_the_calls_of_the_failed_metric(self) -> None:
    judge, _ = session('{"n": 1}', '{"n": 2}', ProviderError("x"))

    class Asking:
      name = "asking"
      default_enabled = True
      scores: tuple[str, ...] = ("s",)
      needs = frozenset({"judge"})
      requires: frozenset[str] = frozenset()

      def __init__(self, purpose: str, fail: bool) -> None:
        self.purpose = purpose
        self.fail = fail

      def compute(self, item: MetricInput, services: MetricServices) -> MetricResult:
        assert services.judge is not None
        services.judge.ask(self.purpose, [user("x")], parse_int)
        if self.fail:
          services.judge.ask(self.purpose + "2", [user("y")], parse_int)
        return MetricResult(self.name, "ok", {"s": 1.0})

    services = MetricServices(judge, None, COUNTER)
    first = Asking("first", fail=False)
    second = Asking("second", fail=True)
    results = evaluate_metrics(make_item(), [first, second], services)
    assert results[0].status == "ok"
    assert results[1].status == "error"
    assert [c.purpose for c in results[1].calls] == ["second", "second2"]
    assert results[1].calls[-1].error is not None
    assert len(judge.calls) == 3

  def test_seconds_are_measured(self) -> None:
    (result,) = evaluate_metrics(make_item(), [Fake("m")], SERVICES)
    assert result.seconds >= 0.0
    (failed,) = evaluate_metrics(
      make_item(), [Fake("m", raises=ValueError("x"))], SERVICES
    )
    assert failed.seconds >= 0.0

  def test_no_metrics_gives_an_empty_tuple(self) -> None:
    assert evaluate_metrics(make_item(), [], SERVICES) == ()

  def test_keyboard_interrupt_is_not_swallowed(self) -> None:
    with pytest.raises(KeyboardInterrupt):
      evaluate_metrics(make_item(), [Fake("m", raises=KeyboardInterrupt())], SERVICES)

  def test_real_registry_over_a_complete_item(self) -> None:
    item = make_item(
      retrieved=(chunk(1, TAJO, "a.md"), chunk(2, LISBOA, "b.md")),
      context=(TAJO,),
      documents=("b.md",),
      evidence=("Tajo", "Lisboa"),
      seconds={"retrieval": 0.1},
    )
    results = evaluate_metrics(item, MetricSelection().resolve(), SERVICES)
    assert [r.name for r in results] == list(METRICS)
    assert all(r.status == "ok" for r in results)
    for result, metric in zip(results, METRICS.values(), strict=True):
      assert tuple(result.scores) == metric.scores
      assert json.loads(json.dumps(result.to_record())) == result.to_record()

  def test_real_registry_over_a_bare_item_never_raises(self) -> None:
    bare = make_item(answer=None, refused=None)
    results = evaluate_metrics(bare, list(METRICS.values()), SERVICES)
    statuses = {r.name: r.status for r in results}
    assert statuses == {
      "doc_retrieval": "not_applicable",
      "evidence_retrieval": "not_applicable",
      "refusal": "not_applicable",
      "cost": "ok",
    }


class TestAggregate:
  def metrics(self) -> list[Metric]:
    return [Fake("a", scores=("a1", "a2")), Fake("b", scores=("b1",))]

  def ok(self, name: str, **scores: float | None) -> MetricResult:
    return MetricResult(name, "ok", dict(scores))

  def test_mean_of_real_values_only(self) -> None:
    results = [
      [self.ok("a", a1=1.0, a2=0.0), self.ok("b", b1=2.0)],
      [self.ok("a", a1=0.0, a2=None), self.ok("b", b1=4.0)],
      [self.ok("a", a1=0.5, a2=0.5), self.ok("b", b1=None)],
    ]
    summary = aggregate(results, self.metrics())
    assert summary["a1"] == ScoreSummary(0.5, 3, 0, 0, 0)
    assert summary["a2"] == ScoreSummary(0.25, 2, 1, 0, 0)
    assert summary["b1"] == ScoreSummary(3.0, 2, 1, 0, 0)

  def test_statuses_are_counted_apart_and_not_averaged(self) -> None:
    results = [
      [self.ok("a", a1=1.0, a2=1.0)],
      [MetricResult.not_applicable("a", ["x"])],
      [MetricResult.skipped("a", "sin juez")],
      [MetricResult.skipped("a", "sin juez")],
      [MetricResult.failed("a", "E", [])],
      [MetricResult.failed("a", "E", [])],
      [MetricResult.failed("a", "E", [])],
    ]
    summary = aggregate(results, [Fake("a", scores=("a1",))])
    assert summary == {"a1": ScoreSummary(1.0, 1, 1, 2, 3)}

  def test_zero_is_a_real_value(self) -> None:
    summary = aggregate([[self.ok("a", a1=0.0)]], [Fake("a", scores=("a1",))])
    assert summary["a1"] == ScoreSummary(0.0, 1, 0, 0, 0)

  def test_no_real_value_gives_no_mean(self) -> None:
    summary = aggregate([[self.ok("a", a1=None)]], [Fake("a", scores=("a1",))])
    assert summary["a1"] == ScoreSummary(None, 0, 1, 0, 0)

  def test_metric_absent_from_a_question_counts_as_not_applicable(self) -> None:
    summary = aggregate([[], [self.ok("a", a1=2.0)]], [Fake("a", scores=("a1",))])
    assert summary["a1"] == ScoreSummary(2.0, 1, 1, 0, 0)

  def test_empty_run_still_lists_every_score(self) -> None:
    summary = aggregate([], self.metrics())
    assert list(summary) == ["a1", "a2", "b1"]
    assert all(s == ScoreSummary(None, 0, 0, 0, 0) for s in summary.values())

  def test_no_metrics_gives_empty_summary(self) -> None:
    assert aggregate([[self.ok("a", a1=1.0)]], []) == {}

  def test_score_missing_from_an_ok_result_counts_as_not_applicable(self) -> None:
    summary = aggregate([[self.ok("a")]], [Fake("a", scores=("a1",))])
    assert summary["a1"] == ScoreSummary(None, 0, 1, 0, 0)

  def test_order_of_results_inside_a_question_is_irrelevant(self) -> None:
    results = [[self.ok("b", b1=1.0), self.ok("a", a1=3.0, a2=3.0)]]
    summary = aggregate(results, self.metrics())
    assert summary["a1"].mean == 3.0
    assert summary["b1"].mean == 1.0

  def test_counts_add_up_to_the_number_of_questions(self) -> None:
    results = [
      [self.ok("a", a1=1.0)],
      [MetricResult.not_applicable("a", [])],
      [MetricResult.skipped("a", "r")],
      [MetricResult.failed("a", "e", [])],
      [self.ok("a", a1=None)],
    ]
    s = aggregate(results, [Fake("a", scores=("a1",))])["a1"]
    assert s.n + s.not_applicable + s.skipped + s.errors == len(results)

  def test_end_to_end_with_the_real_registry(self) -> None:
    good = make_item(
      retrieved=(chunk(1, TAJO, "a.md"),),
      documents=("a.md",),
      evidence=("Tajo",),
      context=(TAJO,),
    )
    bad = make_item(
      retrieved=(chunk(1, "nada", "b.md"),),
      documents=("a.md",),
      evidence=("Tajo",),
      context=(),
    )
    bare = make_item(answer=None, refused=None)
    selected = MetricSelection().resolve()
    results = [evaluate_metrics(i, selected, SERVICES) for i in (good, bad, bare)]
    summary = aggregate(results, selected)
    assert summary["doc_hit"] == ScoreSummary(0.5, 2, 1, 0, 0)
    assert summary["evidence_hit"] == ScoreSummary(0.5, 2, 1, 0, 0)
    assert (
      summary["false_refusal"] == ScoreSummary(0.0, 2, 0, 0, 0)
      or summary["false_refusal"].n == 2
    )
    assert summary["false_refusal"].not_applicable == 1
    assert summary["seconds_retrieval"] == ScoreSummary(None, 0, 3, 0, 0)
    assert summary["context_tokens"].n == 1
    assert set(summary) == {s for m in selected for s in m.scores}


# --- package surface ----------------------------------------------------------


class TestPackage:
  def test_public_names_are_importable(self) -> None:
    for name in (
      "METRICS",
      "Citation",
      "JudgeSession",
      "Metric",
      "MetricInput",
      "MetricResult",
      "MetricSelection",
      "MetricServices",
      "RankedChunk",
      "ScoreSummary",
      "aggregate",
      "evaluate_metrics",
    ):
      assert hasattr(metrics_pkg, name), name

  def test_all_lists_only_existing_names(self) -> None:
    for name in getattr(metrics_pkg, "__all__", []):
      assert hasattr(metrics_pkg, name), name


class TestSelectionErrorMessage:
  def test_without_available_names(self) -> None:
    error = MetricSelectionError(["uno"])
    assert str(error) == "Métrica desconocida: uno"
    assert error.available == ()

"""Tests for the judge-based metric families, their prompts and validators (task 5.5.2).

Everything is offline: a scripted fake judge that replies by purpose,
``HashEmbedder`` or a table embedder with hand-picked vectors, and
``EchoJudgeClient``. Every formula is checked by hand on small cases.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import pytest
from hermes_core.generation import PromptMessage
from hermes_enrich.tokens import CharRatioTokenCounter

from hermes_report.metrics import (
  METRICS,
  Citation,
  Claim,
  ClaimVerdict,
  JudgeSession,
  MetricInput,
  MetricResult,
  MetricServices,
  RankedChunk,
  evaluate_metrics,
)
from hermes_report.metrics.base import MetricError
from hermes_report.metrics.judged import (
  AnswerCorrectness,
  AnswerRelevance,
  Citations,
  ContextPrecision,
  ContextRecall,
  Faithfulness,
  average_precision,
  citation_targets,
  cosine,
)
from hermes_report.metrics.prompts import (
  claims_messages,
  parse_claims,
  parse_questions,
  parse_relevance,
  parse_verdicts,
  questions_messages,
  relevance_messages,
  verify_messages,
)
from hermes_report.providers import EchoJudgeClient, HashEmbedder, ProviderError

COUNTER = CharRatioTokenCounter(4)
CONTEXT = (
  "Pieza uno sobre el Tajo",
  "Pieza dos sobre Lisboa",
  "Pieza tres sobre Toledo",
)


# --- fakes and helpers --------------------------------------------------------


def make_item(**changes: Any) -> MetricInput:
  base = MetricInput(
    question_id="q1",
    question="¿Dónde nace el Tajo?",
    expected="Nace en Albarracín. Desemboca en Lisboa.",
    refusal=False,
    answer="El Tajo nace en Albarracín [1].",
    refused=False,
    retrieved=(
      RankedChunk("c1", "chunk uno", "a.md", 0.9),
      RankedChunk("c2", "chunk dos", "b.md", 0.8),
      RankedChunk("c3", "chunk tres", "c.md", 0.7),
    ),
    context=CONTEXT,
  )
  return replace(base, **changes)


def claims_reply(*claims: tuple[str, Sequence[Any]]) -> str:
  return json.dumps(
    {"claims": [{"text": t, "citations": list(c)} for t, c in claims]},
    ensure_ascii=False,
  )


def verdicts_reply(*verdicts: tuple[bool, Sequence[int]]) -> str:
  return json.dumps(
    {
      "verdicts": [
        {"claim": n, "supported": s, "sources": list(src)}
        for n, (s, src) in enumerate(verdicts, 1)
      ]
    }
  )


def relevance_reply(*flags: bool) -> str:
  return json.dumps(
    {"verdicts": [{"chunk": n, "relevant": f} for n, f in enumerate(flags, 1)]}
  )


def questions_reply(*questions: str, noncommittal: bool = False) -> str:
  return json.dumps(
    {"questions": list(questions), "noncommittal": noncommittal}, ensure_ascii=False
  )


Script = str | Exception | Callable[[str], str] | dict[str, str]


class PurposeJudge:
  """A judge client that replies by what the prompt asks for.

  ``verify`` may be a dict: the reply whose key occurs in the user message is used.
  """

  model = "purpose"

  def __init__(
    self,
    *,
    answer_claims: Script | None = None,
    expected_claims: Script | None = None,
    verify: Script | None = None,
    relevance: Script | None = None,
    questions: Script | None = None,
  ) -> None:
    self.script: dict[str, Script | None] = {
      "answer_claims": answer_claims,
      "expected_claims": expected_claims,
      "verify": verify,
      "relevance": relevance,
      "questions": questions,
    }
    self.requests: list[tuple[str, str]] = []

  @staticmethod
  def purpose_of(messages: Sequence[PromptMessage]) -> str:
    system = messages[0].content
    if "atómicas" in system:
      return "expected_claims" if "no lleva marcadores" in system else "answer_claims"
    if "verificador" in system:
      return "verify"
    if "recuperación" in system:
      return "relevance"
    if "generador de preguntas" in system:
      return "questions"
    raise AssertionError(f"unknown prompt: {system[:40]}")

  def count(self, purpose: str) -> int:
    return sum(1 for p, _ in self.requests if p == purpose)

  def complete_json(self, messages: Sequence[PromptMessage]) -> str:
    purpose = self.purpose_of(messages)
    user = messages[-1].content
    self.requests.append((purpose, user))
    script = self.script[purpose]
    if script is None:
      raise AssertionError(f"unexpected judge call: {purpose}")
    if isinstance(script, Exception):
      raise script
    if isinstance(script, dict):
      for key, reply in script.items():
        if key in user:
          return reply
      raise AssertionError(f"no scripted verify reply for: {user}")
    if callable(script):
      return script(user)
    return script


class TableEmbedder:
  """Embeds known texts to fixed vectors; counts the calls."""

  def __init__(self, table: dict[str, list[float]]) -> None:
    self.table = table
    self.calls: list[str] = []

  def get_query_embedding(self, query: str) -> list[float]:
    self.calls.append(query)
    return self.table[query]

  def get_text_embedding(self, text: str) -> list[float]:
    return self.table[text]

  def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
    return [self.table[t] for t in texts]


def services_for(
  client: Any, item: MetricInput, embedder: Any = None
) -> tuple[MetricServices, JudgeSession]:
  judge = JudgeSession(client, item)
  return MetricServices(judge=judge, embedder=embedder, counter=COUNTER), judge


def run(
  metric: Any, client: Any, item: MetricInput, embedder: Any = None
) -> MetricResult:
  services, _ = services_for(client, item, embedder)
  return metric.compute(item, services)  # type: ignore[no-any-return]


def evaluate_one(
  metric: Any, client: Any, item: MetricInput, embedder: Any = None
) -> MetricResult:
  services, _ = services_for(client, item, embedder)
  (result,) = evaluate_metrics(item, [metric], services)
  return result


# --- registry -----------------------------------------------------------------

DESIGN_ORDER = [
  "doc_retrieval",
  "evidence_retrieval",
  "refusal",
  "cost",
  "faithfulness",
  "answer_relevance",
  "context_precision",
  "context_recall",
  "answer_correctness",
  "citations",
]


class TestRegistry:
  def test_order_is_the_design_table(self) -> None:
    assert list(METRICS) == DESIGN_ORDER

  @pytest.mark.parametrize(
    ("name", "scores", "needs", "requires"),
    [
      ("faithfulness", ("faithfulness",), {"judge"}, {"answer", "context"}),
      ("answer_relevance", ("answer_relevance",), {"judge", "embedder"}, {"answer"}),
      (
        "context_precision",
        ("context_precision",),
        {"judge"},
        {"retrieved", "expected"},
      ),
      ("context_recall", ("context_recall",), {"judge"}, {"expected", "context"}),
      (
        "answer_correctness",
        ("correctness_precision", "correctness_recall", "correctness_f1"),
        {"judge"},
        {"answer", "expected"},
      ),
      (
        "citations",
        ("citation_precision", "citation_recall"),
        {"judge"},
        {"answer", "context"},
      ),
    ],
  )
  def test_family_matches_the_design_table(
    self, name: str, scores: tuple[str, ...], needs: set[str], requires: set[str]
  ) -> None:
    metric = METRICS[name]
    assert metric.default_enabled is True
    assert metric.scores == scores
    assert set(metric.needs) == needs
    assert set(metric.requires) == requires

  def test_classes_are_registered_instances(self) -> None:
    kinds = {
      "faithfulness": Faithfulness,
      "answer_relevance": AnswerRelevance,
      "context_precision": ContextPrecision,
      "context_recall": ContextRecall,
      "answer_correctness": AnswerCorrectness,
      "citations": Citations,
    }
    for name, kind in kinds.items():
      assert isinstance(METRICS[name], kind)


# --- pure helpers -------------------------------------------------------------


class TestCosine:
  def test_hand_values(self) -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / math.sqrt(2))
    assert cosine([3.0, 4.0], [6.0, 8.0]) == pytest.approx(1.0)

  @pytest.mark.parametrize(
    ("a", "b"), [([], []), ([0.0, 0.0], [1.0, 0.0]), ([1.0], [0.0])]
  )
  def test_zero_norm_or_empty_is_zero(self, a: list[float], b: list[float]) -> None:
    assert cosine(a, b) == 0.0

  def test_length_mismatch_is_a_metric_error(self) -> None:
    with pytest.raises(MetricError, match="difieren"):
      cosine([1.0], [1.0, 2.0])

  def test_empty_against_non_empty_is_a_metric_error(self) -> None:
    with pytest.raises(MetricError):
      cosine([], [1.0])


class TestAveragePrecision:
  @pytest.mark.parametrize(
    ("flags", "expected"),
    [
      ([], 0.0),
      ([False], 0.0),
      ([False, False, False], 0.0),
      ([True], 1.0),
      ([True, True, True], 1.0),
      ([True, False, True], (1 + 2 / 3) / 2),
      ([False, False, True], 1 / 3),
      ([False, True, True], (1 / 2 + 2 / 3) / 2),
      ([True, False, False], 1.0),
      ([True, False, False, True], (1 + 2 / 4) / 2),
    ],
  )
  def test_hand_values(self, flags: list[bool], expected: float) -> None:
    assert average_precision(flags) == pytest.approx(expected)

  def test_relevant_first_beats_relevant_last(self) -> None:
    assert average_precision([True, False, False]) > average_precision(
      [False, False, True]
    )


class TestCitationTargets:
  def test_marker_n_is_piece_n(self) -> None:
    item = make_item()
    claims = [Claim("a", (1,)), Claim("b", (3, 2))]
    assert citation_targets(item, claims) == (
      Citation(1, 1),
      Citation(3, 3),
      Citation(2, 2),
    )

  @pytest.mark.parametrize("marker", [0, -1, 4, 99])
  def test_marker_outside_the_context_keeps_piece_zero(self, marker: int) -> None:
    item = make_item()
    assert citation_targets(item, [Claim("a", (marker,))]) == (Citation(marker, 0),)

  def test_boundaries_one_and_last_are_valid(self) -> None:
    item = make_item()
    assert citation_targets(item, [Claim("a", (1, 3))]) == (
      Citation(1, 1),
      Citation(3, 3),
    )

  def test_empty_context_points_nowhere(self) -> None:
    item = make_item(context=())
    assert citation_targets(item, [Claim("a", (1,))]) == (Citation(1, 0),)

  def test_no_claims_and_no_markers(self) -> None:
    item = make_item()
    assert citation_targets(item, []) == ()
    assert citation_targets(item, [Claim("a"), Claim("b")]) == ()

  def test_duplicate_markers_of_one_claim_collapse(self) -> None:
    item = make_item()
    assert citation_targets(item, [Claim("a", (2, 2, 2))]) == (Citation(2, 2),)

  def test_same_marker_in_two_claims_counts_per_claim(self) -> None:
    item = make_item()
    assert citation_targets(item, [Claim("a", (1,)), Claim("b", (1,))]) == (
      Citation(1, 1),
      Citation(1, 1),
    )

  def test_explicit_citations_override_the_one_to_one_mapping(self) -> None:
    item = make_item(citations=(Citation(1, 3), Citation(2, 1)))
    claims = [Claim("a", (1, 2)), Claim("b", (3,))]
    assert citation_targets(item, claims) == (
      Citation(1, 3),
      Citation(2, 1),
      Citation(3, 0),
    )

  def test_empty_explicit_citations_are_not_the_default_mapping(self) -> None:
    item = make_item(citations=())
    assert citation_targets(item, [Claim("a", (1,))]) == (Citation(1, 0),)


# --- prompts ------------------------------------------------------------------


class TestPrompts:
  def test_claims_messages_answer_vs_expected(self) -> None:
    answer = claims_messages("El Tajo [1].", "answer", "¿Q?")
    expected = claims_messages("El Tajo.", "expected", "¿Q?")
    assert [m.role for m in answer] == ["system", "user"]
    assert "El Tajo [1]." in answer[1].content
    assert "¿Q?" in answer[1].content
    assert "claims" in answer[0].content
    assert "no lleva marcadores" in expected[0].content
    assert "no lleva marcadores" not in answer[0].content
    assert answer[0].content != expected[0].content

  def test_verify_messages_number_sources_and_claims_from_one(self) -> None:
    messages = verify_messages([Claim("c uno"), Claim("c dos")], ["s uno", "s dos"])
    user = messages[1].content
    assert "[1] s uno" in user
    assert "[2] s dos" in user
    assert "[1] c uno" in user
    assert "[2] c dos" in user
    assert "[0]" not in user
    assert "verdicts" in messages[0].content

  def test_relevance_messages_carry_question_expected_and_chunks(self) -> None:
    messages = relevance_messages("la pregunta", "la esperada", ["x", "y", "z"])
    user = messages[1].content
    assert "la pregunta" in user
    assert "la esperada" in user
    assert "[1] x" in user
    assert "[3] z" in user
    assert "[4]" not in user

  def test_questions_messages_carry_n_and_answer(self) -> None:
    messages = questions_messages("la respuesta", 5)
    assert "5" in messages[1].content
    assert "la respuesta" in messages[1].content
    assert "3" in questions_messages("r")[1].content

  def test_prompts_are_in_spanish(self) -> None:
    systems = [
      claims_messages("t", "answer", "q")[0],
      verify_messages([Claim("c")], ["s"])[0],
      relevance_messages("q", "e", ["c"])[0],
      questions_messages("a")[0],
    ]
    for message in systems:
      assert "Responde únicamente" in message.content
      assert "JSON" in message.content


# --- strict validators --------------------------------------------------------


class TestParseClaims:
  def test_valid(self) -> None:
    data = {
      "claims": [
        {"text": " uno ", "citations": [1, 3]},
        {"text": "dos", "citations": []},
      ]
    }
    assert parse_claims(data) == (Claim("uno", (1, 3)), Claim("dos", ()))

  def test_no_claims_is_valid(self) -> None:
    assert parse_claims({"claims": []}) == ()

  def test_duplicated_markers_collapse(self) -> None:
    assert parse_claims({"claims": [{"text": "a", "citations": [2, 1, 2]}]}) == (
      Claim("a", (2, 1)),
    )

  @pytest.mark.parametrize(
    "data",
    [
      None,
      [],
      "claims",
      3,
      True,
      {},
      {"claim": []},
      {"claims": None},
      {"claims": {}},
      {"claims": "x"},
      {"claims": ["texto"]},
      {"claims": [None]},
      {"claims": [[]]},
      {"claims": [{"text": "a"}]},
      {"claims": [{"citations": []}]},
      {"claims": [{"text": "a", "citations": None}]},
      {"claims": [{"text": "a", "citations": "1"}]},
      {"claims": [{"text": "a", "citations": ["1"]}]},
      {"claims": [{"text": "a", "citations": [1.0]}]},
      {"claims": [{"text": "a", "citations": [True]}]},
      {"claims": [{"text": "a", "citations": [None]}]},
      {"claims": [{"text": "a", "citations": [[1]]}]},
      {"claims": [{"text": 1, "citations": []}]},
      {"claims": [{"text": None, "citations": []}]},
      {"claims": [{"text": "", "citations": []}]},
      {"claims": [{"text": "   ", "citations": []}]},
      {"claims": [{"text": ["a"], "citations": []}]},
    ],
  )
  def test_malformed_raises_metric_error(self, data: Any) -> None:
    with pytest.raises(MetricError):
      parse_claims(data)


class TestParseVerdicts:
  def good(self) -> dict[str, Any]:
    return {
      "verdicts": [
        {"claim": 1, "supported": True, "sources": [2, 1]},
        {"claim": 2, "supported": False, "sources": []},
      ]
    }

  def test_valid(self) -> None:
    assert parse_verdicts(self.good(), 2, 2) == (
      ClaimVerdict(1, True, (2, 1)),
      ClaimVerdict(2, False, ()),
    )

  def test_out_of_order_verdicts_are_returned_in_claim_order(self) -> None:
    data = self.good()
    data["verdicts"].reverse()
    assert [v.claim for v in parse_verdicts(data, 2, 2)] == [1, 2]

  def test_extra_keys_are_ignored(self) -> None:
    data = self.good()
    data["verdicts"][0]["why"] = "porque sí"
    data["extra"] = 1
    assert len(parse_verdicts(data, 2, 2)) == 2

  def test_boundary_indexes(self) -> None:
    data = {"verdicts": [{"claim": 1, "supported": True, "sources": [1, 5]}]}
    assert parse_verdicts(data, 1, 5) == (ClaimVerdict(1, True, (1, 5)),)

  @pytest.mark.parametrize(
    "mutate",
    [
      lambda d: d["verdicts"].pop(),  # fewer than the claims
      lambda d: d["verdicts"].append({"claim": 3, "supported": True, "sources": []}),
      lambda d: d["verdicts"][1].update(claim=1),  # repeated claim index
      lambda d: d["verdicts"][0].update(claim=0),  # 0-based index
      lambda d: d["verdicts"][1].update(claim=3),  # out of range
      lambda d: d["verdicts"][1].update(claim=-1),
      lambda d: d["verdicts"][0].update(claim="1"),
      lambda d: d["verdicts"][0].update(claim=True),
      lambda d: d["verdicts"][0].update(claim=1.0),
      lambda d: d["verdicts"][0].update(claim=None),
      lambda d: d["verdicts"][0].pop("claim"),
      lambda d: d["verdicts"][0].pop("supported"),
      lambda d: d["verdicts"][0].pop("sources"),
      lambda d: d["verdicts"][0].update(supported="true"),
      lambda d: d["verdicts"][0].update(supported=1),
      lambda d: d["verdicts"][0].update(supported=None),
      lambda d: d["verdicts"][0].update(sources=[0]),  # 0-based source
      lambda d: d["verdicts"][0].update(sources=[3]),  # out of range
      lambda d: d["verdicts"][0].update(sources=[1, 1]),  # repeated source
      lambda d: d["verdicts"][0].update(sources=["1"]),
      lambda d: d["verdicts"][0].update(sources=[True]),
      lambda d: d["verdicts"][0].update(sources=1),
      lambda d: d["verdicts"][0].update(sources=None),
      lambda d: d["verdicts"].__setitem__(0, "veredicto"),
      lambda d: d["verdicts"].__setitem__(0, None),
      lambda d: d.update(verdicts={}),
      lambda d: d.update(verdicts=None),
      lambda d: d.pop("verdicts"),
    ],
  )
  def test_malformed_raises_metric_error(self, mutate: Callable[[Any], Any]) -> None:
    data = self.good()
    mutate(data)
    with pytest.raises(MetricError):
      parse_verdicts(data, 2, 2)

  @pytest.mark.parametrize("data", [None, [], "x", 1, True])
  def test_not_an_object(self, data: Any) -> None:
    with pytest.raises(MetricError):
      parse_verdicts(data, 1, 1)

  def test_count_depends_on_the_input_not_on_the_reply(self) -> None:
    with pytest.raises(MetricError):
      parse_verdicts(self.good(), 3, 2)
    with pytest.raises(MetricError):
      parse_verdicts({"verdicts": []}, 1, 1)


class TestParseRelevance:
  def test_valid_in_chunk_order(self) -> None:
    data = {
      "verdicts": [
        {"chunk": 2, "relevant": False},
        {"chunk": 1, "relevant": True},
        {"chunk": 3, "relevant": True},
      ]
    }
    assert parse_relevance(data, 3) == (True, False, True)

  @pytest.mark.parametrize(
    "verdicts",
    [
      [],
      [{"chunk": 1, "relevant": True}],  # fewer
      [{"chunk": 1, "relevant": True}] * 3,  # repeated
      [
        {"chunk": 1, "relevant": True},
        {"chunk": 2, "relevant": True},
        {"chunk": 3, "relevant": True},
        {"chunk": 4, "relevant": True},
      ],
      [
        {"chunk": 0, "relevant": True},
        {"chunk": 1, "relevant": True},
        {"chunk": 2, "relevant": True},
      ],
      [
        {"chunk": 1, "relevant": True},
        {"chunk": 2, "relevant": True},
        {"chunk": 4, "relevant": True},
      ],
      [
        {"chunk": 1, "relevant": "true"},
        {"chunk": 2, "relevant": True},
        {"chunk": 3, "relevant": True},
      ],
      [
        {"chunk": 1, "relevant": 1},
        {"chunk": 2, "relevant": True},
        {"chunk": 3, "relevant": True},
      ],
      [
        {"chunk": 1},
        {"chunk": 2, "relevant": True},
        {"chunk": 3, "relevant": True},
      ],
      [
        {"relevant": True},
        {"chunk": 2, "relevant": True},
        {"chunk": 3, "relevant": True},
      ],
      [
        {"chunk": "1", "relevant": True},
        {"chunk": 2, "relevant": True},
        {"chunk": 3, "relevant": True},
      ],
      [True, False, True],
      "x",
      None,
    ],
  )
  def test_malformed_raises_metric_error(self, verdicts: Any) -> None:
    with pytest.raises(MetricError):
      parse_relevance({"verdicts": verdicts}, 3)

  @pytest.mark.parametrize("data", [None, [], "x", {}, {"chunks": []}])
  def test_wrong_envelope(self, data: Any) -> None:
    with pytest.raises(MetricError):
      parse_relevance(data, 1)


class TestParseQuestions:
  def test_valid(self) -> None:
    data = {"questions": [" a ", "b"], "noncommittal": False}
    assert parse_questions(data) == (("a", "b"), False)

  def test_noncommittal_with_no_questions(self) -> None:
    assert parse_questions({"questions": [], "noncommittal": True}) == ((), True)

  @pytest.mark.parametrize(
    "data",
    [
      None,
      [],
      "x",
      {},
      {"questions": ["a"]},
      {"noncommittal": False},
      {"questions": "a", "noncommittal": False},
      {"questions": None, "noncommittal": False},
      {"questions": [1], "noncommittal": False},
      {"questions": [None], "noncommittal": False},
      {"questions": [""], "noncommittal": False},
      {"questions": ["  "], "noncommittal": False},
      {"questions": [["a"]], "noncommittal": False},
      {"questions": ["a"], "noncommittal": "false"},
      {"questions": ["a"], "noncommittal": 0},
      {"questions": ["a"], "noncommittal": None},
    ],
  )
  def test_malformed_raises_metric_error(self, data: Any) -> None:
    with pytest.raises(MetricError):
      parse_questions(data)


class TestValidatorsNeverRaiseAnythingElse:
  """A fuzz over arbitrary JSON values: only MetricError may escape."""

  def value(self, rng: random.Random, depth: int = 0) -> Any:
    kinds = ["int", "str", "bool", "none", "float"]
    if depth < 3:
      kinds += ["list", "dict", "dict"]
    kind = rng.choice(kinds)
    if kind == "int":
      return rng.choice([0, 1, 2, 3, -1, 99, 2**70])
    if kind == "str":
      return rng.choice(["", " ", "1", "x", "claims", "verdicts"])
    if kind == "bool":
      return rng.choice([True, False])
    if kind == "none":
      return None
    if kind == "float":
      return rng.choice([0.0, 1.5, float("nan"), float("inf")])
    if kind == "list":
      return [self.value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    keys = ["claims", "verdicts", "questions", "noncommittal", "claim", "chunk"]
    keys += ["text", "citations", "supported", "sources", "relevant"]
    return {
      rng.choice(keys): self.value(rng, depth + 1) for _ in range(rng.randint(0, 4))
    }

  def test_random_shapes(self) -> None:
    rng = random.Random(20240522)
    parsers: list[Callable[[Any], Any]] = [
      parse_claims,
      lambda d: parse_verdicts(d, 2, 2),
      lambda d: parse_relevance(d, 2),
      parse_questions,
    ]
    for _ in range(3000):
      data = self.value(rng)
      for parse in parsers:
        try:
          parse(data)
        except MetricError:
          pass

  def test_random_mutations_of_valid_replies(self) -> None:
    rng = random.Random(7)
    valid: list[tuple[Callable[[Any], Any], Any]] = [
      (parse_claims, {"claims": [{"text": "a", "citations": [1]}]}),
      (
        lambda d: parse_verdicts(d, 1, 2),
        {"verdicts": [{"claim": 1, "supported": True, "sources": [1]}]},
      ),
      (
        lambda d: parse_relevance(d, 1),
        {"verdicts": [{"chunk": 1, "relevant": True}]},
      ),
      (parse_questions, {"questions": ["a"], "noncommittal": False}),
    ]
    for parse, good in valid:
      assert parse(good) is not None
      for _ in range(500):
        data = json.loads(json.dumps(good))
        node: Any = data
        while True:
          children = (
            list(node.items()) if isinstance(node, dict) else list(enumerate(node))
          )
          container = [c for c in children if isinstance(c[1], dict | list)]
          if not container or rng.random() < 0.3:
            break
          node = rng.choice(container)[1]
        if isinstance(node, dict) and node:
          key = rng.choice(list(node))
          node[key] = self.value(rng, 2)
        elif isinstance(node, list):
          node.append(self.value(rng, 2))
        try:
          parse(data)
        except MetricError:
          pass


# --- JudgeSession: new operations ---------------------------------------------


class TestJudgeSessionOperations:
  def test_answer_claims_parses_and_is_memoised(self) -> None:
    client = PurposeJudge(answer_claims=claims_reply(("a", [1]), ("b", [])))
    judge = JudgeSession(client, make_item())
    first = judge.answer_claims()
    assert first == (Claim("a", (1,)), Claim("b", ()))
    assert judge.answer_claims() is first
    assert client.count("answer_claims") == 1
    (call,) = judge.calls
    assert call.purpose == "answer_claims"
    assert call.error is None

  @pytest.mark.parametrize("answer", [None, ""])
  def test_answer_claims_without_answer_raises_without_calling(
    self, answer: str | None
  ) -> None:
    client = PurposeJudge()
    judge = JudgeSession(client, make_item(answer=answer))
    with pytest.raises(MetricError):
      judge.answer_claims()
    assert client.requests == []
    assert judge.calls == []

  def test_expected_claims_use_the_expected_prompt(self) -> None:
    client = PurposeJudge(expected_claims=claims_reply(("e", [])))
    judge = JudgeSession(client, make_item())
    assert judge.expected_claims() == (Claim("e", ()),)
    assert client.requests[0][0] == "expected_claims"
    assert "Albarracín" in client.requests[0][1]

  @pytest.mark.parametrize("expected", ["", "   ", "\n"])
  def test_expected_claims_without_expected_raises_without_calling(
    self, expected: str
  ) -> None:
    client = PurposeJudge()
    judge = JudgeSession(client, make_item(expected=expected))
    with pytest.raises(MetricError):
      judge.expected_claims()
    assert client.requests == []

  def test_whitespace_answer_reaches_the_judge(self) -> None:
    # answer_claims only checks for a falsy answer; the has() guard is the runner's.
    client = PurposeJudge(answer_claims=claims_reply())
    judge = JudgeSession(client, make_item(answer="   "))
    assert judge.answer_claims() == ()

  def test_verify_without_claims_is_empty_and_free(self) -> None:
    client = PurposeJudge()
    judge = JudgeSession(client, make_item())
    assert judge.verify([], ["s"], "k") == ()
    assert client.requests == []

  def test_verify_without_sources_is_all_unsupported_and_free(self) -> None:
    client = PurposeJudge()
    judge = JudgeSession(client, make_item())
    verdicts = judge.verify([Claim("a"), Claim("b")], [], "k")
    assert verdicts == (ClaimVerdict(1, False, ()), ClaimVerdict(2, False, ()))
    assert client.requests == []
    assert judge.calls == []

  def test_verify_memo_by_key(self) -> None:
    client = PurposeJudge(verify=verdicts_reply((True, [1])))
    judge = JudgeSession(client, make_item())
    first = judge.verify([Claim("a")], ["s"], "k1")
    assert judge.verify([Claim("otra")], ["otra"], "k1") is first
    assert client.count("verify") == 1
    judge.verify([Claim("a")], ["s"], "k2")
    assert client.count("verify") == 2
    assert [c.purpose for c in judge.calls] == ["verify:k1", "verify:k2"]

  def test_verify_key_never_collides_with_other_purposes(self) -> None:
    client = PurposeJudge(
      answer_claims=claims_reply(("a", [])), verify=verdicts_reply((True, [1]))
    )
    judge = JudgeSession(client, make_item())
    claims = judge.answer_claims()
    assert judge.verify(claims, ["s"], "answer_claims") == (
      ClaimVerdict(1, True, (1,)),
    )
    assert client.count("verify") == 1

  def test_verify_wrong_verdict_count_is_a_logged_memoised_error(self) -> None:
    client = PurposeJudge(verify=verdicts_reply((True, [1])))
    judge = JudgeSession(client, make_item())
    claims = [Claim("a"), Claim("b")]
    with pytest.raises(MetricError):
      judge.verify(claims, ["s"], "k")
    with pytest.raises(MetricError):
      judge.verify(claims, ["s"], "k")
    assert client.count("verify") == 1
    assert judge.calls[0].raw == verdicts_reply((True, [1]))
    assert judge.calls[0].error

  def test_verify_source_index_out_of_range_is_an_error(self) -> None:
    client = PurposeJudge(verify=verdicts_reply((True, [2])))
    judge = JudgeSession(client, make_item())
    with pytest.raises(MetricError):
      judge.verify([Claim("a")], ["solo una"], "k")

  def test_chunk_relevance_one_call_for_all_chunks(self) -> None:
    client = PurposeJudge(relevance=relevance_reply(True, False, True))
    judge = JudgeSession(client, make_item())
    assert judge.chunk_relevance() == (True, False, True)
    assert judge.chunk_relevance() == (True, False, True)
    assert client.count("relevance") == 1
    user = client.requests[0][1]
    for text in ("chunk uno", "chunk dos", "chunk tres"):
      assert text in user

  def test_chunk_relevance_without_chunks_is_empty_and_free(self) -> None:
    client = PurposeJudge()
    judge = JudgeSession(client, make_item(retrieved=()))
    assert judge.chunk_relevance() == ()
    assert client.requests == []

  def test_chunk_relevance_scales_to_many_chunks_in_one_call(self) -> None:
    chunks = tuple(RankedChunk(f"c{n}", f"t{n}", "a.md", 0.1) for n in range(50))
    client = PurposeJudge(relevance=relevance_reply(*([False] * 50)))
    judge = JudgeSession(client, make_item(retrieved=chunks))
    assert judge.chunk_relevance() == (False,) * 50
    assert client.count("relevance") == 1

  def test_calls_since_covers_new_operations(self) -> None:
    client = PurposeJudge(
      answer_claims=claims_reply(("a", [])), verify=verdicts_reply((True, [1]))
    )
    judge = JudgeSession(client, make_item())
    claims = judge.answer_claims()
    mark = len(judge.calls)
    judge.verify(claims, ["s"], "k")
    assert [c.purpose for c in judge.calls_since(mark)] == ["verify:k"]


# --- JudgeSession.ask: carry-over from 5.5.1 ----------------------------------


class RawClient:
  """A client that returns whatever it is given, with no str guarantee."""

  model = "raw"

  def __init__(self, reply: Any) -> None:
    self.reply = reply
    self.requests = 0

  def complete_json(self, messages: Sequence[PromptMessage]) -> Any:
    self.requests += 1
    if isinstance(self.reply, BaseException):
      raise self.reply
    return self.reply


def user(text: str) -> PromptMessage:
  return PromptMessage(role="user", content=text)


class TestAskHardening:
  @pytest.mark.parametrize(
    "exc",
    [
      KeyError("n"),
      TypeError("tipo"),
      IndexError(0),
      AttributeError("x"),
      ZeroDivisionError(),
    ],
  )
  def test_parse_raising_a_bare_exception_is_logged_memoised_and_wrapped(
    self, exc: Exception
  ) -> None:
    def parse(data: Any) -> int:
      raise exc

    client = RawClient('{"n": 1}')
    judge = JudgeSession(client, make_item())
    with pytest.raises(MetricError) as first:
      judge.ask("p", [user("a")], parse)
    assert isinstance(first.value.__cause__, type(exc))
    assert type(exc).__name__ in str(first.value)
    with pytest.raises(MetricError) as second:
      judge.ask("p", [user("a")], parse)
    assert second.value is first.value
    assert client.requests == 1
    (call,) = judge.calls
    assert call.raw == '{"n": 1}'
    assert call.error == str(first.value)

  def test_parse_key_error_through_a_plain_dict_lookup(self) -> None:
    client = RawClient('{"otra": 1}')
    judge = JudgeSession(client, make_item())
    with pytest.raises(MetricError, match="KeyError"):
      judge.ask("p", [user("a")], lambda data: data["n"])
    assert judge.calls[0].raw == '{"otra": 1}'

  @pytest.mark.parametrize(
    "reply",
    [
      42,
      None,
      b'{"n": 1}',
      {"n": 1},
      ["n"],
      3.5,
      True,
    ],
  )
  def test_non_str_reply_is_logged_memoised_and_a_metric_error(
    self, reply: Any
  ) -> None:
    client = RawClient(reply)
    judge = JudgeSession(client, make_item())
    with pytest.raises(MetricError):
      judge.ask("p", [user("a")], lambda d: d)
    with pytest.raises(MetricError):
      judge.ask("p", [user("a")], lambda d: d)
    assert client.requests == 1
    (call,) = judge.calls
    assert call.error
    if reply is None:
      assert call.raw is None
    else:
      assert call.raw == repr(reply)

  def test_client_raising_an_arbitrary_exception_is_wrapped_and_memoised(self) -> None:
    client = RawClient(RuntimeError("boom"))
    judge = JudgeSession(client, make_item())
    with pytest.raises(MetricError, match="RuntimeError: boom"):
      judge.ask("p", [user("a")], lambda d: d)
    with pytest.raises(MetricError):
      judge.ask("p", [user("a")], lambda d: d)
    assert client.requests == 1
    assert judge.calls[0].raw is None

  def test_client_raising_provider_error_still_chains(self) -> None:
    client = RawClient(ProviderError("sin red"))
    judge = JudgeSession(client, make_item())
    with pytest.raises(MetricError) as info:
      judge.ask("p", [user("a")], lambda d: d)
    assert isinstance(info.value.__cause__, ProviderError)

  @pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
  def test_interrupts_are_not_swallowed(self, exc: type[BaseException]) -> None:
    client = RawClient(exc())
    judge = JudgeSession(client, make_item())
    with pytest.raises(exc):
      judge.ask("p", [user("a")], lambda d: d)

  def test_a_metric_error_from_parse_keeps_its_own_message_and_instance(self) -> None:
    error = MetricError("forma mala")

    def parse(data: Any) -> int:
      raise error

    judge = JudgeSession(RawClient("{}"), make_item())
    with pytest.raises(MetricError) as info:
      judge.ask("p", [user("a")], parse)
    assert info.value is error
    assert judge.calls[0].error == "forma mala"

  def test_valid_json_that_is_not_an_object_reaches_parse_and_fails_cleanly(
    self,
  ) -> None:
    for raw in ("null", "[]", "3", '"x"', "true"):
      judge = JudgeSession(PurposeJudge(answer_claims=raw), make_item())
      with pytest.raises(MetricError):
        judge.answer_claims()
      assert judge.calls[0].raw == raw

  def test_deeply_nested_json_is_a_metric_error_not_a_crash(self) -> None:
    raw = "[" * 100000 + "]" * 100000
    judge = JudgeSession(PurposeJudge(answer_claims=raw), make_item())
    with pytest.raises(MetricError):
      judge.answer_claims()
    assert judge.calls[0].raw == raw


# --- faithfulness -------------------------------------------------------------


class TestFaithfulness:
  def test_two_of_three_supported(self) -> None:
    client = PurposeJudge(
      answer_claims=claims_reply(("a", []), ("b", []), ("c", [])),
      verify=verdicts_reply((True, [1]), (False, []), (True, [2, 3])),
    )
    result = run(Faithfulness(), client, make_item())
    assert result.status == "ok"
    assert result.scores == {"faithfulness": pytest.approx(2 / 3)}
    assert [c.purpose for c in result.calls] == [
      "answer_claims",
      "verify:answer_vs_context",
    ]
    verify_user = client.requests[1][1]
    for n, piece in enumerate(CONTEXT, 1):
      assert f"[{n}] {piece}" in verify_user

  def test_all_and_none_supported(self) -> None:
    all_ok = PurposeJudge(
      answer_claims=claims_reply(("a", []), ("b", [])),
      verify=verdicts_reply((True, [1]), (True, [2])),
    )
    none_ok = PurposeJudge(
      answer_claims=claims_reply(("a", []), ("b", [])),
      verify=verdicts_reply((False, []), (False, [])),
    )
    assert run(Faithfulness(), all_ok, make_item()).scores["faithfulness"] == 1.0
    assert run(Faithfulness(), none_ok, make_item()).scores["faithfulness"] == 0.0

  def test_no_claims_scores_one_without_verifying(self) -> None:
    client = PurposeJudge(answer_claims=claims_reply())
    result = run(Faithfulness(), client, make_item())
    assert result.status == "ok"
    assert result.scores == {"faithfulness": 1.0}
    assert client.count("verify") == 0

  def test_refused_answer_is_not_applicable_without_calls(self) -> None:
    client = PurposeJudge()
    result = run(Faithfulness(), client, make_item(refused=True))
    assert result.status == "not_applicable"
    assert result.scores == {}
    assert client.requests == []

  def test_empty_context_is_zero_without_a_verify_call(self) -> None:
    client = PurposeJudge(answer_claims=claims_reply(("a", [])))
    result = run(Faithfulness(), client, make_item(context=()))
    assert result.scores["faithfulness"] == 0.0
    assert client.count("verify") == 0

  def test_through_the_runner_missing_context_is_not_applicable(self) -> None:
    result = evaluate_one(Faithfulness(), PurposeJudge(), make_item(context=()))
    assert result.status == "not_applicable"
    assert result.details["missing"] == ["context"]

  def test_through_the_runner_without_judge_is_skipped(self) -> None:
    services = MetricServices(judge=None, embedder=None, counter=COUNTER)
    (result,) = evaluate_metrics(make_item(), [Faithfulness()], services)
    assert result.status == "skipped"

  def test_direct_compute_without_judge_is_a_metric_error(self) -> None:
    services = MetricServices(judge=None, embedder=None, counter=COUNTER)
    with pytest.raises(MetricError):
      Faithfulness().compute(make_item(), services)

  def test_details_hold_claims_and_verdicts(self) -> None:
    client = PurposeJudge(
      answer_claims=claims_reply(("a", [])), verify=verdicts_reply((True, [1]))
    )
    result = run(Faithfulness(), client, make_item())
    assert result.details["claims"] == (Claim("a", ()),)
    assert result.details["verdicts"] == (ClaimVerdict(1, True, (1,)),)


# --- answer relevance ---------------------------------------------------------


class TestAnswerRelevance:
  Q = "¿Dónde nace el Tajo?"

  def embedder(self, **vectors: list[float]) -> TableEmbedder:
    table = {self.Q: [1.0, 0.0]}
    table.update({k.replace("_", " "): v for k, v in vectors.items()})
    return TableEmbedder(table)

  def test_mean_cosine_of_three_questions(self) -> None:
    embedder = TableEmbedder(
      {self.Q: [1.0, 0.0], "g1": [1.0, 0.0], "g2": [0.0, 1.0], "g3": [1.0, 1.0]}
    )
    client = PurposeJudge(questions=questions_reply("g1", "g2", "g3"))
    result = run(AnswerRelevance(), client, make_item(), embedder)
    assert result.status == "ok"
    assert result.scores["answer_relevance"] == pytest.approx(
      (1.0 + 0.0 + 1 / math.sqrt(2)) / 3
    )
    assert result.details["similarities"] == pytest.approx([1.0, 0.0, 0.7071067811865])
    assert [c.purpose for c in result.calls] == ["questions"]

  def test_asks_for_three_questions(self) -> None:
    embedder = TableEmbedder({self.Q: [1.0], "g": [1.0]})
    client = PurposeJudge(questions=questions_reply("g"))
    run(AnswerRelevance(), client, make_item(), embedder)
    assert "Número de preguntas: 3" in client.requests[0][1]

  def test_negative_mean_is_floored_at_zero(self) -> None:
    embedder = TableEmbedder({self.Q: [1.0, 0.0], "g": [-1.0, 0.0]})
    client = PurposeJudge(questions=questions_reply("g"))
    result = run(AnswerRelevance(), client, make_item(), embedder)
    assert result.scores["answer_relevance"] == 0.0
    assert result.details["similarities"] == pytest.approx([-1.0])

  def test_noncommittal_is_zero_and_embeds_nothing(self) -> None:
    embedder = TableEmbedder({})
    client = PurposeJudge(questions=questions_reply("g", noncommittal=True))
    result = run(AnswerRelevance(), client, make_item(), embedder)
    assert result.status == "ok"
    assert result.scores == {"answer_relevance": 0.0}
    assert result.details["noncommittal"] is True
    assert embedder.calls == []

  def test_no_questions_is_zero(self) -> None:
    embedder = TableEmbedder({})
    client = PurposeJudge(questions=questions_reply())
    result = run(AnswerRelevance(), client, make_item(), embedder)
    assert result.scores == {"answer_relevance": 0.0}
    assert embedder.calls == []

  def test_refused_answer_to_a_normal_question_scores_zero_without_calls(self) -> None:
    embedder = TableEmbedder({})
    client = PurposeJudge()
    result = run(AnswerRelevance(), client, make_item(refused=True), embedder)
    assert result.status == "ok"
    assert result.scores == {"answer_relevance": 0.0}
    assert result.details == {"refused": True}
    assert client.requests == []

  def test_refused_answer_to_a_refusal_question_is_not_applicable(self) -> None:
    result = run(
      AnswerRelevance(),
      PurposeJudge(),
      make_item(refused=True, refusal=True),
      TableEmbedder({}),
    )
    assert result.status == "not_applicable"

  def test_refusal_question_answered_normally_is_scored(self) -> None:
    embedder = TableEmbedder({self.Q: [1.0], "g": [1.0]})
    client = PurposeJudge(questions=questions_reply("g"))
    result = run(AnswerRelevance(), client, make_item(refusal=True), embedder)
    assert result.scores["answer_relevance"] == pytest.approx(1.0)

  def test_zero_vectors_score_zero_not_nan(self) -> None:
    embedder = TableEmbedder({self.Q: [0.0, 0.0], "g": [0.0, 0.0]})
    client = PurposeJudge(questions=questions_reply("g"))
    result = run(AnswerRelevance(), client, make_item(), embedder)
    assert result.scores["answer_relevance"] == 0.0

  def test_embedding_length_mismatch_is_an_error_status_with_the_raw_reply(
    self,
  ) -> None:
    embedder = TableEmbedder({self.Q: [1.0, 0.0], "g": [1.0]})
    client = PurposeJudge(questions=questions_reply("g"))
    result = evaluate_one(AnswerRelevance(), client, make_item(), embedder)
    assert result.status == "error"
    assert result.calls[0].raw == questions_reply("g")

  def test_missing_embedder_is_skipped_by_the_runner_and_error_when_direct(
    self,
  ) -> None:
    client = PurposeJudge(questions=questions_reply("g"))
    result = evaluate_one(AnswerRelevance(), client, make_item())
    assert result.status == "skipped"
    with pytest.raises(MetricError):
      run(AnswerRelevance(), PurposeJudge(), make_item())

  def test_hash_embedder_identical_question_is_one(self) -> None:
    client = PurposeJudge(questions=questions_reply(self.Q, self.Q, self.Q))
    result = run(AnswerRelevance(), client, make_item(), HashEmbedder())
    assert result.scores["answer_relevance"] == pytest.approx(1.0)

  def test_hash_embedder_unrelated_question_is_lower(self) -> None:
    same = PurposeJudge(questions=questions_reply(self.Q))
    other = PurposeJudge(questions=questions_reply("zzz qqq www"))
    high = run(AnswerRelevance(), same, make_item(), HashEmbedder())
    low = run(AnswerRelevance(), other, make_item(), HashEmbedder())
    assert 0.0 <= low.scores["answer_relevance"] < high.scores["answer_relevance"]  # type: ignore[operator]

  def test_score_is_within_unit_interval(self) -> None:
    rng = random.Random(3)
    for _ in range(30):
      vectors = {f"g{n}": [rng.uniform(-1, 1), rng.uniform(-1, 1)] for n in range(3)}
      embedder = TableEmbedder(
        {self.Q: [rng.uniform(-1, 1), rng.uniform(-1, 1)]} | vectors
      )
      client = PurposeJudge(questions=questions_reply(*vectors))
      score = run(AnswerRelevance(), client, make_item(), embedder).scores[
        "answer_relevance"
      ]
      assert score is not None
      assert 0.0 <= score <= 1.0 + 1e-9

  def test_no_answer_is_not_applicable_through_the_runner(self) -> None:
    result = evaluate_one(
      AnswerRelevance(),
      PurposeJudge(),
      make_item(answer=None, refused=None),
      HashEmbedder(),
    )
    assert result.status == "not_applicable"


# --- context precision --------------------------------------------------------


class TestContextPrecision:
  @pytest.mark.parametrize(
    ("flags", "expected"),
    [
      ((True, False, True), (1 + 2 / 3) / 2),
      ((False, False, True), 1 / 3),
      ((True, True, True), 1.0),
      ((False, False, False), 0.0),
      ((False, True, True), (1 / 2 + 2 / 3) / 2),
    ],
  )
  def test_hand_values(self, flags: tuple[bool, ...], expected: float) -> None:
    client = PurposeJudge(relevance=relevance_reply(*flags))
    result = run(ContextPrecision(), client, make_item())
    assert result.status == "ok"
    assert result.scores["context_precision"] == pytest.approx(expected)
    assert result.details["relevant"] == list(flags)
    assert client.count("relevance") == 1

  def test_all_irrelevant_is_a_zero_not_none(self) -> None:
    client = PurposeJudge(relevance=relevance_reply(False, False, False))
    result = run(ContextPrecision(), client, make_item())
    assert result.scores == {"context_precision": 0.0}

  def test_single_chunk(self) -> None:
    item = make_item(retrieved=(RankedChunk("c1", "t", "a.md", 1.0),))
    relevant = PurposeJudge(relevance=relevance_reply(True))
    irrelevant = PurposeJudge(relevance=relevance_reply(False))
    assert run(ContextPrecision(), relevant, item).scores["context_precision"] == 1.0
    assert run(ContextPrecision(), irrelevant, item).scores["context_precision"] == 0.0

  def test_misaligned_verdicts_are_an_error_with_the_raw_reply(self) -> None:
    reply = relevance_reply(True, False)
    result = evaluate_one(
      ContextPrecision(), PurposeJudge(relevance=reply), make_item()
    )
    assert result.status == "error"
    assert result.calls[0].raw == reply
    assert result.calls[0].error

  def test_missing_expected_or_retrieved_is_not_applicable(self) -> None:
    for item in (make_item(expected=""), make_item(retrieved=())):
      assert evaluate_one(ContextPrecision(), PurposeJudge(), item).status == (
        "not_applicable"
      )


# --- context recall -----------------------------------------------------------


class TestContextRecall:
  def test_one_of_two_expected_claims_supported(self) -> None:
    client = PurposeJudge(
      expected_claims=claims_reply(
        ("nace en Albarracín", []), ("desemboca en Lisboa", [])
      ),
      verify=verdicts_reply((True, [1]), (False, [])),
    )
    result = run(ContextRecall(), client, make_item())
    assert result.status == "ok"
    assert result.scores == {"context_recall": 0.5}
    assert [c.purpose for c in result.calls] == [
      "expected_claims",
      "verify:expected_vs_context",
    ]
    assert "[1] Pieza uno" in client.requests[1][1]

  def test_no_expected_claims_is_none(self) -> None:
    client = PurposeJudge(expected_claims=claims_reply())
    result = run(ContextRecall(), client, make_item())
    assert result.status == "ok"
    assert result.scores == {"context_recall": None}
    assert client.count("verify") == 0

  def test_empty_context_scores_zero_without_verify(self) -> None:
    client = PurposeJudge(expected_claims=claims_reply(("a", [])))
    result = run(ContextRecall(), client, make_item(context=()))
    assert result.scores == {"context_recall": 0.0}
    assert client.count("verify") == 0

  def test_never_asks_for_answer_claims(self) -> None:
    client = PurposeJudge(
      expected_claims=claims_reply(("a", [])), verify=verdicts_reply((True, [1]))
    )
    run(ContextRecall(), client, make_item(answer=None, refused=None))
    assert client.count("answer_claims") == 0

  def test_blank_expected_through_the_runner_is_not_applicable(self) -> None:
    result = evaluate_one(ContextRecall(), PurposeJudge(), make_item(expected="  "))
    assert result.status == "not_applicable"


# --- answer correctness -------------------------------------------------------


class TestAnswerCorrectness:
  def client(
    self, answer_n: int, expected_n: int, forward: str, backward: str
  ) -> PurposeJudge:
    return PurposeJudge(
      answer_claims=claims_reply(*[(f"a{n}", []) for n in range(answer_n)]),
      expected_claims=claims_reply(*[(f"e{n}", []) for n in range(expected_n)]),
      verify={"[1] " + "Nace en Albarracín": forward, "[1] El Tajo nace": backward},
    )

  def test_hand_values(self) -> None:
    # 4 answer claims, 3 supported by the expected answer: P = 0.75.
    # 2 expected claims, 1 supported by the answer: R = 0.5. F1 = 0.6.
    client = self.client(
      4,
      2,
      verdicts_reply((True, [1]), (True, [1]), (True, [1]), (False, [])),
      verdicts_reply((True, [1]), (False, [])),
    )
    result = run(AnswerCorrectness(), client, make_item())
    assert result.status == "ok"
    assert result.scores == {
      "correctness_precision": 0.75,
      "correctness_recall": 0.5,
      "correctness_f1": pytest.approx(0.6),
    }

  def test_forward_is_against_the_expected_text_and_backward_against_the_answer(
    self,
  ) -> None:
    client = self.client(1, 1, verdicts_reply((True, [1])), verdicts_reply((True, [1])))
    run(AnswerCorrectness(), client, make_item())
    verify_requests = [user for p, user in client.requests if p == "verify"]
    assert len(verify_requests) == 2
    forward, backward = verify_requests
    assert "[1] Nace en Albarracín. Desemboca en Lisboa." in forward
    assert "[1] a0" in forward
    assert "[1] El Tajo nace en Albarracín [1]." in backward
    assert "[1] e0" in backward

  def test_perfect(self) -> None:
    client = self.client(
      2,
      2,
      verdicts_reply((True, [1]), (True, [1])),
      verdicts_reply((True, [1]), (True, [1])),
    )
    scores = run(AnswerCorrectness(), client, make_item()).scores
    assert scores == {
      "correctness_precision": 1.0,
      "correctness_recall": 1.0,
      "correctness_f1": 1.0,
    }

  def test_both_zero_gives_f1_zero_not_a_division_error(self) -> None:
    client = self.client(1, 1, verdicts_reply((False, [])), verdicts_reply((False, [])))
    scores = run(AnswerCorrectness(), client, make_item()).scores
    assert scores == {
      "correctness_precision": 0.0,
      "correctness_recall": 0.0,
      "correctness_f1": 0.0,
    }

  def test_precision_only(self) -> None:
    # P = 1, R = 0 -> F1 = 0
    client = self.client(
      1, 2, verdicts_reply((True, [1])), verdicts_reply((False, []), (False, []))
    )
    scores = run(AnswerCorrectness(), client, make_item()).scores
    assert scores["correctness_precision"] == 1.0
    assert scores["correctness_recall"] == 0.0
    assert scores["correctness_f1"] == 0.0

  def test_f1_is_the_harmonic_mean(self) -> None:
    # P = 1/2, R = 1/3 -> F1 = 0.4
    client = self.client(
      2,
      3,
      verdicts_reply((True, [1]), (False, [])),
      verdicts_reply((True, [1]), (False, []), (False, [])),
    )
    scores = run(AnswerCorrectness(), client, make_item()).scores
    assert scores["correctness_f1"] == pytest.approx(0.4)

  def test_no_expected_claims_gives_three_nones(self) -> None:
    client = PurposeJudge(
      answer_claims=claims_reply(("a", [])), expected_claims=claims_reply()
    )
    result = run(AnswerCorrectness(), client, make_item())
    assert result.status == "ok"
    assert result.scores == {
      "correctness_precision": None,
      "correctness_recall": None,
      "correctness_f1": None,
    }
    assert client.count("verify") == 0

  def test_no_answer_claims_gives_precision_none_and_zero_recall(self) -> None:
    client = PurposeJudge(
      answer_claims=claims_reply(),
      expected_claims=claims_reply(("e", [])),
      verify=verdicts_reply((False, [])),
    )
    scores = run(AnswerCorrectness(), client, make_item()).scores
    assert scores == {
      "correctness_precision": None,
      "correctness_recall": 0.0,
      "correctness_f1": 0.0,
    }

  def test_scores_keep_the_declared_order(self) -> None:
    client = self.client(1, 1, verdicts_reply((True, [1])), verdicts_reply((True, [1])))
    result = run(AnswerCorrectness(), client, make_item())
    assert tuple(result.scores) == AnswerCorrectness.scores

  def test_to_record_round_trips_through_json(self) -> None:
    client = self.client(1, 1, verdicts_reply((True, [1])), verdicts_reply((True, [1])))
    record = run(AnswerCorrectness(), client, make_item()).to_record()
    assert json.loads(json.dumps(record)) == record

  def test_answer_claims_failure_is_an_error(self) -> None:
    client = PurposeJudge(answer_claims="no es json", expected_claims=claims_reply())
    result = evaluate_one(AnswerCorrectness(), client, make_item())
    assert result.status == "error"
    assert result.calls[0].raw == "no es json"

  def test_backward_verdict_misalignment_is_an_error_with_both_raw_replies(
    self,
  ) -> None:
    client = self.client(1, 2, verdicts_reply((True, [1])), verdicts_reply((True, [1])))
    result = evaluate_one(AnswerCorrectness(), client, make_item())
    assert result.status == "error"
    assert result.calls[-1].raw == verdicts_reply((True, [1]))
    assert result.calls[-1].error


# --- citations ----------------------------------------------------------------


def cited_client(claims: str, verdicts: str) -> PurposeJudge:
  return PurposeJudge(answer_claims=claims, verify=verdicts)


class TestCitations:
  def test_hand_values(self) -> None:
    # c1 cites [1] (supported by 1): good. c2 cites [2][3], supported by 3 only:
    # one good, one bad. c3 cites nothing but is supported.
    # precision = 2 / 3. recall = (c1 covered, c2 covered, c3 not) / 3 = 2 / 3.
    client = cited_client(
      claims_reply(("c1", [1]), ("c2", [2, 3]), ("c3", [])),
      verdicts_reply((True, [1]), (True, [3]), (True, [2])),
    )
    result = run(Citations(), client, make_item())
    assert result.status == "ok"
    assert result.scores == {
      "citation_precision": pytest.approx(2 / 3),
      "citation_recall": pytest.approx(2 / 3),
    }
    assert result.details["citations"] == [
      Citation(1, 1),
      Citation(2, 2),
      Citation(3, 3),
    ]

  def test_perfect(self) -> None:
    client = cited_client(
      claims_reply(("c1", [1]), ("c2", [2])),
      verdicts_reply((True, [1]), (True, [2])),
    )
    assert run(Citations(), client, make_item()).scores == {
      "citation_precision": 1.0,
      "citation_recall": 1.0,
    }

  def test_no_citations_precision_none_recall_zero(self) -> None:
    client = cited_client(
      claims_reply(("c1", []), ("c2", [])), verdicts_reply((True, [1]), (False, []))
    )
    assert run(Citations(), client, make_item()).scores == {
      "citation_precision": None,
      "citation_recall": 0.0,
    }

  def test_no_claims_gives_two_nones(self) -> None:
    client = PurposeJudge(answer_claims=claims_reply())
    result = run(Citations(), client, make_item())
    assert result.status == "ok"
    assert result.scores == {"citation_precision": None, "citation_recall": None}
    assert client.count("verify") == 0

  def test_cited_but_nothing_supported_precision_zero_recall_none(self) -> None:
    client = cited_client(
      claims_reply(("c1", [1]), ("c2", [2])), verdicts_reply((False, []), (False, []))
    )
    assert run(Citations(), client, make_item()).scores == {
      "citation_precision": 0.0,
      "citation_recall": None,
    }

  def test_marker_outside_the_context_is_cited_but_never_good(self) -> None:
    # [7] has no piece: it counts as a citation and as a miss.
    client = cited_client(claims_reply(("c1", [1, 7])), verdicts_reply((True, [1])))
    result = run(Citations(), client, make_item())
    assert result.scores["citation_precision"] == 0.5
    assert result.scores["citation_recall"] == 1.0

  def test_marker_zero_never_matches_even_if_the_judge_lists_source_zero_is_rejected(
    self,
  ) -> None:
    # A [0] marker is piece 0; the verdict parser rejects source 0, so it can never
    # be good.
    client = cited_client(claims_reply(("c1", [0])), verdicts_reply((True, [1])))
    result = run(Citations(), client, make_item())
    assert result.scores["citation_precision"] == 0.0
    assert result.scores["citation_recall"] == 0.0

  def test_duplicate_markers_in_a_claim_count_once(self) -> None:
    client = cited_client(claims_reply(("c1", [1, 1, 1])), verdicts_reply((True, [1])))
    assert run(Citations(), client, make_item()).scores["citation_precision"] == 1.0

  def test_same_marker_on_two_claims_counts_per_claim(self) -> None:
    client = cited_client(
      claims_reply(("c1", [1]), ("c2", [1])), verdicts_reply((True, [1]), (False, []))
    )
    result = run(Citations(), client, make_item())
    assert result.scores["citation_precision"] == 0.5
    assert result.scores["citation_recall"] == 1.0

  def test_a_citation_on_an_unsupported_claim_is_never_good(self) -> None:
    client = cited_client(claims_reply(("c1", [1])), verdicts_reply((False, [1])))
    assert run(Citations(), client, make_item()).scores["citation_precision"] == 0.0

  def test_explicit_citations_change_the_target_piece(self) -> None:
    # marker 1 is piece 3 here; the judge says piece 3 supports the claim.
    item = make_item(citations=(Citation(1, 3),))
    client = cited_client(claims_reply(("c1", [1])), verdicts_reply((True, [3])))
    assert run(Citations(), client, item).scores == {
      "citation_precision": 1.0,
      "citation_recall": 1.0,
    }
    default = make_item()
    client = cited_client(claims_reply(("c1", [1])), verdicts_reply((True, [3])))
    assert run(Citations(), client, default).scores["citation_precision"] == 0.0

  def test_result_is_json_round_trippable(self) -> None:
    client = cited_client(claims_reply(("c1", [1])), verdicts_reply((True, [1])))
    record = run(Citations(), client, make_item()).to_record()
    assert json.loads(json.dumps(record)) == record

  def test_every_claim_marker_is_a_reachable_edge_with_empty_context(self) -> None:
    client = PurposeJudge(answer_claims=claims_reply(("c1", [1])))
    result = run(Citations(), client, make_item(context=()))
    assert result.scores == {"citation_precision": 0.0, "citation_recall": None}


# --- shared claims and failure isolation --------------------------------------


def full_client() -> PurposeJudge:
  return PurposeJudge(
    answer_claims=claims_reply(("a1", [1]), ("a2", [2])),
    expected_claims=claims_reply(("e1", []), ("e2", [])),
    verify={
      "[1] Pieza uno": verdicts_reply((True, [1]), (True, [2])),
      "[1] Nace en": verdicts_reply((True, [1]), (False, [])),
      "[1] El Tajo nace": verdicts_reply((True, [1]), (True, [1])),
    },
    relevance=relevance_reply(True, False, True),
    questions=questions_reply("g1", "g2", "g3"),
  )


JUDGED = [
  "faithfulness",
  "answer_relevance",
  "context_precision",
  "context_recall",
  "answer_correctness",
  "citations",
]


def judged_metrics(*names: str) -> list[Any]:
  return [METRICS[n] for n in names]


class TestSharedWork:
  def test_answer_claims_requested_once_for_the_three_families(self) -> None:
    client = full_client()
    item = make_item()
    services, judge = services_for(client, item)
    results = evaluate_metrics(
      item, judged_metrics("faithfulness", "answer_correctness", "citations"), services
    )
    assert [r.status for r in results] == ["ok", "ok", "ok"]
    assert client.count("answer_claims") == 1
    assert [c.purpose for c in judge.calls].count("answer_claims") == 1
    # The same verify over the context serves faithfulness and citations.
    assert [c.purpose for c in judge.calls].count("verify:answer_vs_context") == 1

  def test_calls_are_attributed_to_the_metric_that_made_them(self) -> None:
    client = full_client()
    item = make_item()
    services, _ = services_for(client, item)
    faith, correct, cite = evaluate_metrics(
      item, judged_metrics("faithfulness", "answer_correctness", "citations"), services
    )
    assert [c.purpose for c in faith.calls] == [
      "answer_claims",
      "verify:answer_vs_context",
    ]
    assert "answer_claims" not in [c.purpose for c in correct.calls]
    assert cite.calls == ()

  def test_total_judge_calls_of_every_family(self) -> None:
    client = full_client()
    item = make_item()
    services, judge = services_for(client, item, HashEmbedder())
    results = evaluate_metrics(item, judged_metrics(*JUDGED), services)
    assert [r.status for r in results] == ["ok"] * 6
    purposes = sorted(c.purpose for c in judge.calls)
    assert purposes == sorted(
      [
        "answer_claims",
        "expected_claims",
        "verify:answer_vs_context",
        "verify:expected_vs_context",
        "verify:answer_vs_expected",
        "verify:expected_vs_answer",
        "chunk_relevance",
        "questions",
      ]
    )
    # Every call is attributed to exactly one result.
    assert sum(len(r.calls) for r in results) == len(judge.calls)

  def test_order_of_families_does_not_change_the_scores(self) -> None:
    item = make_item()
    forward = evaluate_metrics(
      item,
      judged_metrics(*JUDGED),
      services_for(full_client(), item, HashEmbedder())[0],
    )
    backward = evaluate_metrics(
      item,
      judged_metrics(*reversed(JUDGED)),
      services_for(full_client(), item, HashEmbedder())[0],
    )
    by_name = {r.name: r.scores for r in backward}
    for result in forward:
      assert by_name[result.name] == result.scores

  def test_every_result_keeps_the_declared_score_order(self) -> None:
    item = make_item()
    services, _ = services_for(full_client(), item, HashEmbedder())
    for result, metric in zip(
      evaluate_metrics(item, judged_metrics(*JUDGED), services),
      judged_metrics(*JUDGED),
      strict=True,
    ):
      assert result.status == "ok", result.error
      assert tuple(result.scores) == metric.scores

  def test_every_ok_result_survives_a_json_round_trip(self) -> None:
    item = make_item()
    services, _ = services_for(full_client(), item, HashEmbedder())
    for result in evaluate_metrics(item, judged_metrics(*JUDGED), services):
      record = result.to_record()
      assert json.loads(json.dumps(record)) == record

  def test_a_shared_failure_is_memoised_and_does_not_call_the_judge_again(
    self,
  ) -> None:
    client = full_client()
    client.script["answer_claims"] = "no es json"
    item = make_item()
    services, judge = services_for(client, item)
    faith, correct, cite = evaluate_metrics(
      item, judged_metrics("faithfulness", "answer_correctness", "citations"), services
    )
    assert [r.status for r in (faith, correct, cite)] == ["error"] * 3
    assert client.count("answer_claims") == 1
    # The raw reply is kept on the call that failed.
    assert faith.calls[0].raw == "no es json"
    assert [c.raw for c in judge.calls if c.purpose == "answer_claims"] == [
      "no es json"
    ]


MALFORMED = {
  "answer_claims": [
    "no es json",
    "",
    "[]",
    "null",
    '{"claims": "x"}',
    '{"claims": [{"text": "a"}]}',
    '{"claims": [{"text": "a", "citations": ["1"]}]}',
  ],
  "verify": [
    "no es json",
    '{"verdicts": []}',
    '{"verdicts": [{"claim": 5, "supported": true, "sources": []}]}',
    '{"verdicts": [{"claim": 1, "supported": "yes", "sources": []}]}',
    '{"verdicts": [{"claim": 1, "supported": true, "sources": [9]}]}',
  ],
}


class TestMalformedRepliesIsolateTheFailure:
  @pytest.mark.parametrize("raw", MALFORMED["answer_claims"])
  def test_bad_answer_claims_fail_their_families_only(self, raw: str) -> None:
    client = full_client()
    client.script["answer_claims"] = raw
    item = make_item()
    services, _ = services_for(client, item, HashEmbedder())
    results = {
      r.name: r for r in evaluate_metrics(item, judged_metrics(*JUDGED), services)
    }
    for name in ("faithfulness", "answer_correctness", "citations"):
      assert results[name].status == "error"
      assert results[name].error
    for name in ("answer_relevance", "context_precision", "context_recall"):
      assert results[name].status == "ok", results[name].error
    assert results["faithfulness"].calls[0].raw == raw
    assert results["faithfulness"].calls[0].error

  @pytest.mark.parametrize("raw", MALFORMED["verify"])
  def test_bad_verify_fails_the_families_that_verify_and_keeps_the_rest(
    self, raw: str
  ) -> None:
    client = PurposeJudge(
      answer_claims=claims_reply(("a1", [1]), ("a2", [2])),
      expected_claims=claims_reply(("e1", []), ("e2", [])),
      verify=raw,
      relevance=relevance_reply(True, False, True),
      questions=questions_reply("g1"),
    )
    item = make_item()
    services, _ = services_for(client, item, HashEmbedder())
    results = {
      r.name: r for r in evaluate_metrics(item, judged_metrics(*JUDGED), services)
    }
    for name in ("faithfulness", "context_recall", "answer_correctness", "citations"):
      assert results[name].status == "error", name
    assert results["context_precision"].status == "ok"
    assert results["answer_relevance"].status == "ok"
    raws = [c.raw for r in results.values() for c in r.calls if c.error]
    assert raw in raws

  @pytest.mark.parametrize(
    "raw",
    [
      "no es json",
      '{"verdicts": [{"chunk": 1, "relevant": true}]}',
      '{"verdicts": [{"chunk": 1, "relevant": true}, {"chunk": 1, "relevant": true}, {"chunk": 3, "relevant": true}]}',
    ],
  )
  def test_bad_relevance(self, raw: str) -> None:
    client = full_client()
    client.script["relevance"] = raw
    item = make_item()
    services, _ = services_for(client, item, HashEmbedder())
    results = {
      r.name: r for r in evaluate_metrics(item, judged_metrics(*JUDGED), services)
    }
    assert results["context_precision"].status == "error"
    assert results["context_precision"].calls[0].raw == raw
    for name in JUDGED:
      if name != "context_precision":
        assert results[name].status == "ok", (name, results[name].error)

  @pytest.mark.parametrize(
    "raw",
    [
      "no es json",
      '{"questions": ["a"]}',
      '{"questions": "a", "noncommittal": false}',
      '{"questions": [1], "noncommittal": false}',
    ],
  )
  def test_bad_questions(self, raw: str) -> None:
    client = full_client()
    client.script["questions"] = raw
    item = make_item()
    services, _ = services_for(client, item, HashEmbedder())
    results = {
      r.name: r for r in evaluate_metrics(item, judged_metrics(*JUDGED), services)
    }
    assert results["answer_relevance"].status == "error"
    assert results["answer_relevance"].calls[0].raw == raw
    for name in JUDGED:
      if name != "answer_relevance":
        assert results[name].status == "ok", (name, results[name].error)

  def test_a_client_that_raises_fails_every_family_it_serves_and_nothing_crashes(
    self,
  ) -> None:
    client = PurposeJudge(
      answer_claims=ProviderError("sin red"),
      expected_claims=ProviderError("sin red"),
      verify=ProviderError("sin red"),
      relevance=ProviderError("sin red"),
      questions=ProviderError("sin red"),
    )
    item = make_item()
    services, _ = services_for(client, item, HashEmbedder())
    results = evaluate_metrics(item, judged_metrics(*JUDGED), services)
    assert [r.status for r in results] == ["error"] * 6
    assert all("sin red" in (r.error or "") for r in results)

  def test_a_failing_judge_family_does_not_stop_the_deterministic_ones(self) -> None:
    client = PurposeJudge(
      answer_claims="no es json",
      expected_claims="no es json",
      verify="no es json",
      relevance="no es json",
      questions="no es json",
    )
    item = make_item(
      documents=("a.md",), evidence=("Tajo",), seconds={"retrieval": 0.1}
    )
    services, _ = services_for(client, item, HashEmbedder())
    results = evaluate_metrics(item, list(METRICS.values()), services)
    status = {r.name: r.status for r in results}
    assert status["doc_retrieval"] == "ok"
    assert status["evidence_retrieval"] == "ok"
    assert status["cost"] == "ok"
    assert all(status[n] == "error" for n in JUDGED)


# --- EchoJudgeClient ----------------------------------------------------------


class TestWithEchoJudge:
  def test_every_judge_family_ends_with_a_valid_status(self) -> None:
    item = make_item()
    services, judge = services_for(EchoJudgeClient(), item, HashEmbedder())
    results = evaluate_metrics(item, judged_metrics(*JUDGED), services)
    assert [r.name for r in results] == JUDGED
    for result in results:
      assert result.status in {"ok", "error", "not_applicable", "skipped"}
      record = result.to_record()
      assert json.loads(json.dumps(record)) == record
    # The echo reply fits no judge shape, so the families error out, never crash.
    assert {r.status for r in results} <= {"ok", "error"}
    assert all(r.error for r in results if r.status == "error")
    assert judge.calls
    assert all(c.raw for c in judge.calls)

  def test_the_echo_judge_with_the_whole_registry(self) -> None:
    item = make_item(documents=("a.md",), evidence=("Tajo",))
    services, _ = services_for(EchoJudgeClient(), item, HashEmbedder())
    results = evaluate_metrics(item, list(METRICS.values()), services)
    assert [r.name for r in results] == DESIGN_ORDER
    assert all(r.status in {"ok", "error", "not_applicable"} for r in results)

  def test_bare_item_with_a_judge_never_raises(self) -> None:
    bare = make_item(answer=None, refused=None, expected="", retrieved=(), context=())
    services, judge = services_for(EchoJudgeClient(), bare, HashEmbedder())
    results = evaluate_metrics(bare, list(METRICS.values()), services)
    status = {r.name: r.status for r in results}
    for name in JUDGED:
      assert status[name] == "not_applicable", name
    assert judge.calls == []

"""Tests for ``hermes_report.evaluate``: the rule verdict and the LLM judge.

Everything runs over hand-built ``QuestionRecord``s and fake judge clients; no
pipeline, Qdrant or Ollama is involved.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any, cast

import pytest
from hermes_core import PromptMessage

from hermes_report.evaluate import (
  JUDGE_VERDICTS,
  RULE_VERDICTS,
  judge,
  judge_messages,
  rule_verdict,
)
from hermes_report.providers import EchoJudgeClient, JudgeClient, ProviderError
from hermes_report.questions import Question
from hermes_report.record import QuestionRecord

RULE_KEYS = {
  "verdict",
  "in_retrieval",
  "in_context",
  "in_cited",
  "answer_ok",
  "retrieved_rank",
  "expected_docs_retrieved",
}
ERROR_RECORD: dict[str, Any] = {
  "verdict": "error",
  "in_retrieval": None,
  "in_context": None,
  "in_cited": None,
  "answer_ok": None,
  "retrieved_rank": None,
  "expected_docs_retrieved": None,
}


def question(**kwargs: Any) -> Question:
  """A question; defaults to keyword ``alfa`` and evidence ``beta``."""
  data: dict[str, Any] = {
    "id": "q1",
    "question": "¿Qué dice el texto?",
    "expected": "Dice alfa.",
    "keywords": ("alfa",),
    "evidence": ("beta",),
  }
  data.update(kwargs)
  return Question(**data)


def chunk(text: str, source: str = "a.md", index: int = 0) -> dict[str, Any]:
  return {
    "chunk_id": f"{source}-{index}",
    "text": text,
    "score": 0.5,
    "vector_score": 0.5,
    "collection_name": "c",
    "metadata": {"source": source, "chunk_index": index},
  }


def piece(text: str) -> dict[str, Any]:
  return {"text": text, "transforms": [], "chunks": []}


def make_record(
  *,
  chunks: Sequence[str] | None = ("beta",),
  pieces: Sequence[str] | None = ("beta",),
  status: str = "answered",
  answer: str = "alfa",
  citations: Sequence[int] = (),
  error: str | None = None,
  result: bool = True,
  sources: Sequence[str] | None = None,
) -> QuestionRecord:
  """A record; ``None`` for chunks/pieces drops the phase entirely."""
  retrieval = (
    None
    if chunks is None
    else {
      "seconds": 0.0,
      "chunks": [
        chunk(t, sources[i] if sources else "a.md", i) for i, t in enumerate(chunks)
      ],
      "queries": [],
    }
  )
  enrich = (
    None
    if pieces is None
    else {
      "seconds": 0.0,
      "budget": {},
      "steps": [],
      "context": {
        "pieces": [piece(t) for t in pieces],
        "enrichments": [],
        "dropped_chunk_ids": [],
        "query_id": None,
      },
    }
  )
  generation = {
    "seconds": 0.0,
    "llm_calls": [],
    "result": (
      {
        "status": status,
        "answer": answer,
        "citations": [{"number": n, "chunk_ids": []} for n in citations],
      }
      if result
      else None
    ),
  }
  return cast(
    "QuestionRecord",
    {
      "spec": {},
      "query_id": "x",
      "seconds": 0.0,
      "error": error,
      "retrieval": retrieval,
      "enrich": enrich,
      "generation": generation,
      "rules": None,
      "judge": None,
    },
  )


class FakeJudge:
  """A judge client that replies a scripted text or raises a scripted error."""

  model = "fake-judge"

  def __init__(self, reply: str = "", error: BaseException | None = None) -> None:
    self.reply = reply
    self.error = error
    self.calls: list[Sequence[PromptMessage]] = []

  def complete_json(self, messages: Sequence[PromptMessage]) -> str:
    self.calls.append(messages)
    if self.error is not None:
      raise self.error
    return self.reply


def verdict_json(verdict: str = "correct", reason: str = "Dice lo mismo.") -> str:
  return json.dumps({"verdict": verdict, "reason": reason})


# --- constants -----------------------------------------------------------------


class TestConstants:
  def test_rule_verdicts_in_table_order(self) -> None:
    assert RULE_VERDICTS == (
      "error",
      "correct_refusal",
      "invented_answer",
      "correct",
      "correct_without_evidence",
      "budget_failure",
      "retrieval_failure",
      "generation_refused",
      "generation_wrong",
      "unrated",
    )

  def test_judge_verdicts(self) -> None:
    assert JUDGE_VERDICTS == ("correct", "partial", "incorrect")


# --- rule_verdict: the verdict table ---------------------------------------------


class TestVerdictTable:
  def test_correct(self) -> None:
    rec = rule_verdict(question(), make_record())
    assert rec == {
      "verdict": "correct",
      "in_retrieval": True,
      "in_context": True,
      "in_cited": False,
      "answer_ok": True,
      "retrieved_rank": 1,
      "expected_docs_retrieved": None,
    }

  def test_correct_when_no_evidence_is_given(self) -> None:
    q = question(evidence=())
    rec = rule_verdict(q, make_record(chunks=["x"], pieces=["x"]))
    assert rec["verdict"] == "correct"
    assert rec["in_context"] is None
    assert rec["in_retrieval"] is None
    assert rec["in_cited"] is None
    assert rec["retrieved_rank"] is None

  def test_correct_without_evidence(self) -> None:
    rec = rule_verdict(question(), make_record(chunks=["beta"], pieces=["zzz"]))
    assert rec["verdict"] == "correct_without_evidence"
    assert rec["answer_ok"] is True
    assert rec["in_context"] is False

  def test_budget_failure(self) -> None:
    rec = rule_verdict(
      question(), make_record(chunks=["beta"], pieces=["zzz"], answer="nada")
    )
    assert rec["verdict"] == "budget_failure"
    assert rec["in_retrieval"] is True
    assert rec["in_context"] is False
    assert rec["answer_ok"] is False

  def test_retrieval_failure(self) -> None:
    rec = rule_verdict(
      question(), make_record(chunks=["zzz"], pieces=["zzz"], answer="nada")
    )
    assert rec["verdict"] == "retrieval_failure"
    assert rec["in_retrieval"] is False
    assert rec["retrieved_rank"] is None

  def test_generation_refused(self) -> None:
    rec = rule_verdict(question(), make_record(status="insufficient", answer="alfa"))
    assert rec["verdict"] == "generation_refused"
    assert rec["answer_ok"] is False

  def test_generation_wrong(self) -> None:
    rec = rule_verdict(question(), make_record(answer="nada"))
    assert rec["verdict"] == "generation_wrong"
    assert rec["answer_ok"] is False

  def test_generation_wrong_with_no_evidence_given(self) -> None:
    rec = rule_verdict(question(evidence=()), make_record(answer="nada"))
    assert rec["verdict"] == "generation_wrong"
    assert rec["in_context"] is None

  def test_generation_refused_with_no_evidence_given(self) -> None:
    rec = rule_verdict(
      question(evidence=()), make_record(status="no_context", answer="")
    )
    assert rec["verdict"] == "generation_refused"

  def test_unrated(self) -> None:
    rec = rule_verdict(question(keywords=()), make_record(answer="lo que sea"))
    assert rec["verdict"] == "unrated"
    assert rec["answer_ok"] is None

  def test_unrated_without_keywords_or_evidence(self) -> None:
    rec = rule_verdict(question(keywords=(), evidence=()), make_record())
    assert rec["verdict"] == "unrated"
    assert rec["in_context"] is None

  def test_unrated_even_when_refused_if_evidence_is_in_context(self) -> None:
    rec = rule_verdict(question(keywords=()), make_record(status="insufficient"))
    assert rec["verdict"] == "unrated"
    assert rec["answer_ok"] is None

  def test_no_keywords_and_evidence_dropped_is_budget_failure(self) -> None:
    rec = rule_verdict(
      question(keywords=()), make_record(chunks=["beta"], pieces=["zzz"])
    )
    assert rec["verdict"] == "budget_failure"
    assert rec["answer_ok"] is None

  def test_no_keywords_and_evidence_never_retrieved_is_retrieval_failure(
    self,
  ) -> None:
    rec = rule_verdict(
      question(keywords=()), make_record(chunks=["zzz"], pieces=["zzz"])
    )
    assert rec["verdict"] == "retrieval_failure"

  def test_every_verdict_in_the_table_is_reachable(self) -> None:
    seen = {
      rule_verdict(q, r)["verdict"]
      for q, r in [
        (question(), make_record(error="boom")),
        (question(refusal=True), make_record(status="insufficient")),
        (question(refusal=True), make_record()),
        (question(), make_record()),
        (question(), make_record(pieces=["z"])),
        (question(), make_record(pieces=["z"], answer="n")),
        (question(), make_record(chunks=["z"], pieces=["z"], answer="n")),
        (question(), make_record(status="insufficient")),
        (question(), make_record(answer="n")),
        (question(keywords=()), make_record()),
      ]
    }
    assert seen == set(RULE_VERDICTS)

  @pytest.mark.parametrize("status", ["insufficient", "no_context"])
  def test_both_refusal_statuses_count_as_refused(self, status: str) -> None:
    rec = rule_verdict(question(), make_record(status=status))
    assert rec["verdict"] == "generation_refused"

  @pytest.mark.parametrize("status", ["answered", "partial", "", "INSUFFICIENT"])
  def test_other_statuses_are_not_refused(self, status: str) -> None:
    rec = rule_verdict(question(), make_record(status=status, answer="nada"))
    assert rec["verdict"] == "generation_wrong"

  def test_keywords_are_matched_without_accents_or_case(self) -> None:
    q = question(keywords=("Rocín",))
    assert rule_verdict(q, make_record(answer="el ROCIN era flaco"))["answer_ok"]

  def test_all_keywords_are_required(self) -> None:
    q = question(keywords=("alfa", "omega"))
    rec = rule_verdict(q, make_record(answer="solo alfa"))
    assert rec["answer_ok"] is False
    assert rec["verdict"] == "generation_wrong"

  def test_all_evidence_phrases_are_required_in_the_context(self) -> None:
    q = question(evidence=("beta", "gamma"))
    rec = rule_verdict(q, make_record(chunks=["beta gamma"], pieces=["beta"]))
    assert rec["in_context"] is False
    assert rec["in_retrieval"] is True

  def test_evidence_may_span_several_pieces(self) -> None:
    q = question(evidence=("beta", "gamma"))
    rec = rule_verdict(
      q, make_record(chunks=["beta", "gamma"], pieces=["beta", "gamma"])
    )
    assert rec["in_context"] is True
    assert rec["in_retrieval"] is True

  def test_result_keys_are_exactly_the_rule_record(self) -> None:
    assert set(rule_verdict(question(), make_record())) == RULE_KEYS


# --- rule_verdict: refusal questions ---------------------------------------------


class TestRefusalQuestions:
  @pytest.mark.parametrize("status", ["insufficient", "no_context"])
  def test_correct_refusal(self, status: str) -> None:
    q = question(refusal=True, keywords=(), evidence=())
    rec = rule_verdict(q, make_record(status=status, answer=""))
    assert rec["verdict"] == "correct_refusal"
    assert rec["answer_ok"] is True

  def test_invented_answer(self) -> None:
    q = question(refusal=True, keywords=(), evidence=())
    rec = rule_verdict(q, make_record(status="answered", answer="algo inventado"))
    assert rec["verdict"] == "invented_answer"
    assert rec["answer_ok"] is False

  def test_keywords_do_not_rescue_an_invented_answer(self) -> None:
    q = question(refusal=True)
    rec = rule_verdict(q, make_record(status="answered", answer="alfa"))
    assert rec["verdict"] == "invented_answer"

  def test_keywords_do_not_spoil_a_correct_refusal(self) -> None:
    q = question(refusal=True)
    rec = rule_verdict(q, make_record(status="insufficient", answer="no lo sé"))
    assert rec["verdict"] == "correct_refusal"

  def test_evidence_failures_do_not_override_a_refusal_verdict(self) -> None:
    q = question(refusal=True)
    rec = rule_verdict(
      q, make_record(chunks=["z"], pieces=["z"], status="insufficient")
    )
    assert rec["verdict"] == "correct_refusal"
    assert rec["in_context"] is False

  def test_error_beats_refusal(self) -> None:
    q = question(refusal=True)
    assert rule_verdict(q, make_record(error="boom"))["verdict"] == "error"


# --- rule_verdict: error and missing result ---------------------------------------


class TestErrorRecords:
  def test_error_set(self) -> None:
    assert rule_verdict(question(), make_record(error="boom")) == ERROR_RECORD

  def test_error_set_even_with_a_result(self) -> None:
    rec = make_record(error="boom")
    assert rec["generation"] is not None
    assert rec["generation"]["result"] is not None
    assert rule_verdict(question(), rec) == ERROR_RECORD

  def test_no_result(self) -> None:
    assert rule_verdict(question(), make_record(result=False)) == ERROR_RECORD

  def test_no_generation_at_all(self) -> None:
    rec = make_record()
    rec["generation"] = None
    assert rule_verdict(question(), rec) == ERROR_RECORD

  def test_generation_key_missing(self) -> None:
    rec = cast("dict[str, Any]", make_record())
    del rec["generation"]
    assert rule_verdict(question(), cast("QuestionRecord", rec)) == ERROR_RECORD

  def test_error_record_does_not_share_state_between_calls(self) -> None:
    first = rule_verdict(question(), make_record(error="a"))
    first["verdict"] = "mutated"
    assert rule_verdict(question(), make_record(error="a")) == ERROR_RECORD

  def test_record_is_not_mutated(self) -> None:
    rec = make_record(citations=(1,))
    before = copy.deepcopy(rec)
    rule_verdict(question(), rec)
    assert rec == before


# --- rule_verdict: missing phases --------------------------------------------------


class TestMissingPhases:
  def test_missing_retrieval_is_empty(self) -> None:
    rec = rule_verdict(question(), make_record(chunks=None, answer="nada"))
    assert rec["in_retrieval"] is False
    assert rec["retrieved_rank"] is None

  def test_missing_enrich_is_empty(self) -> None:
    rec = rule_verdict(question(), make_record(pieces=None))
    assert rec["in_context"] is False
    assert rec["verdict"] == "correct_without_evidence"

  def test_enrich_without_context_is_empty(self) -> None:
    rec = make_record()
    assert rec["enrich"] is not None
    rec["enrich"]["context"] = None
    out = rule_verdict(question(), rec)
    assert out["in_context"] is False

  def test_missing_phases_without_evidence_stay_none(self) -> None:
    q = question(evidence=())
    rec = rule_verdict(q, make_record(chunks=None, pieces=None))
    assert rec["in_retrieval"] is None
    assert rec["in_context"] is None
    assert rec["verdict"] == "correct"

  def test_empty_chunks_and_pieces(self) -> None:
    rec = rule_verdict(question(), make_record(chunks=[], pieces=[], answer="n"))
    assert rec["verdict"] == "retrieval_failure"

  def test_missing_citations_key_is_no_citation(self) -> None:
    rec = cast("dict[str, Any]", make_record())
    del rec["generation"]["result"]["citations"]
    out = rule_verdict(question(), cast("QuestionRecord", rec))
    assert out["in_cited"] is False


# --- rule_verdict: citations ------------------------------------------------------


class TestCitations:
  def test_citation_number_is_one_based_position_of_the_piece(self) -> None:
    rec = make_record(pieces=["zzz", "beta"], citations=(2,))
    assert rule_verdict(question(), rec)["in_cited"] is True

  def test_citing_the_wrong_piece_is_not_cited(self) -> None:
    rec = make_record(pieces=["zzz", "beta"], citations=(1,))
    out = rule_verdict(question(), rec)
    assert out["in_cited"] is False
    assert out["in_context"] is True

  def test_zero_is_not_a_position(self) -> None:
    rec = make_record(pieces=["beta", "zzz"], citations=(0,))
    assert rule_verdict(question(), rec)["in_cited"] is False

  def test_negative_number_does_not_wrap_around(self) -> None:
    rec = make_record(pieces=["zzz", "beta"], citations=(-1,))
    assert rule_verdict(question(), rec)["in_cited"] is False

  def test_out_of_range_number_is_ignored(self) -> None:
    rec = make_record(pieces=["beta"], citations=(7, 1))
    assert rule_verdict(question(), rec)["in_cited"] is True

  def test_only_out_of_range_numbers(self) -> None:
    rec = make_record(pieces=["beta"], citations=(2, 99))
    assert rule_verdict(question(), rec)["in_cited"] is False

  def test_no_citations(self) -> None:
    assert rule_verdict(question(), make_record())["in_cited"] is False

  def test_duplicate_citations_count_once(self) -> None:
    rec = make_record(pieces=["beta gamma"], citations=(1, 1, 1))
    q = question(evidence=("beta", "gamma"))
    assert rule_verdict(q, rec)["in_cited"] is True

  def test_evidence_split_across_cited_pieces(self) -> None:
    q = question(evidence=("beta", "gamma"))
    rec = make_record(pieces=["beta", "zzz", "gamma"], citations=(1, 3))
    assert rule_verdict(q, rec)["in_cited"] is True

  def test_evidence_only_partly_cited(self) -> None:
    q = question(evidence=("beta", "gamma"))
    rec = make_record(pieces=["beta", "zzz", "gamma"], citations=(1,))
    assert rule_verdict(q, rec)["in_cited"] is False

  def test_cited_is_none_without_evidence(self) -> None:
    rec = make_record(pieces=["beta"], citations=(1,))
    assert rule_verdict(question(evidence=()), rec)["in_cited"] is None

  def test_citation_does_not_change_the_verdict(self) -> None:
    uncited = rule_verdict(question(), make_record())
    cited = rule_verdict(question(), make_record(citations=(1,)))
    assert uncited["verdict"] == cited["verdict"] == "correct"

  def test_citations_map_to_pieces_not_to_chunks(self) -> None:
    # Chunk 1 holds the evidence; the pieces were reordered so piece 1 does not.
    rec = make_record(chunks=["beta"], pieces=["zzz", "beta"], citations=(1,))
    assert rule_verdict(question(), rec)["in_cited"] is False


# --- rule_verdict: rank and documents ------------------------------------------


class TestRankAndDocuments:
  def test_rank_is_one_based(self) -> None:
    rec = make_record(chunks=["zzz", "beta", "beta"])
    assert rule_verdict(question(), rec)["retrieved_rank"] == 2

  def test_rank_requires_a_single_chunk_to_hold_all_evidence(self) -> None:
    q = question(evidence=("beta", "gamma"))
    rec = rule_verdict(q, make_record(chunks=["beta", "gamma"], pieces=["beta gamma"]))
    assert rec["in_retrieval"] is True
    assert rec["retrieved_rank"] is None

  def test_rank_finds_the_chunk_holding_all_evidence(self) -> None:
    q = question(evidence=("beta", "gamma"))
    rec = rule_verdict(
      q, make_record(chunks=["beta", "gamma", "beta gamma"], pieces=["x"])
    )
    assert rec["retrieved_rank"] == 3

  def test_rank_is_none_without_evidence(self) -> None:
    rec = rule_verdict(question(evidence=()), make_record())
    assert rec["retrieved_rank"] is None

  def test_rank_is_none_without_retrieval(self) -> None:
    assert rule_verdict(question(), make_record(chunks=None))["retrieved_rank"] is None

  def test_documents_none_when_not_given(self) -> None:
    out = rule_verdict(question(documents=()), make_record())
    assert out["expected_docs_retrieved"] is None

  def test_documents_retrieved(self) -> None:
    q = question(documents=("a.md",))
    out = rule_verdict(q, make_record(chunks=["x", "y"], sources=["b.md", "a.md"]))
    assert out["expected_docs_retrieved"] is True

  def test_documents_not_retrieved(self) -> None:
    q = question(documents=("a.md",))
    out = rule_verdict(q, make_record(chunks=["x"], sources=["b.md"]))
    assert out["expected_docs_retrieved"] is False

  def test_documents_with_nothing_retrieved(self) -> None:
    q = question(documents=("a.md",))
    assert rule_verdict(q, make_record(chunks=[]))["expected_docs_retrieved"] is False
    assert rule_verdict(q, make_record(chunks=None))["expected_docs_retrieved"] is False

  def test_documents_any_of_several(self) -> None:
    q = question(documents=("a.md", "c.md"))
    out = rule_verdict(q, make_record(chunks=["x"], sources=["c.md"]))
    assert out["expected_docs_retrieved"] is True

  def test_chunk_without_source_metadata_does_not_match(self) -> None:
    q = question(documents=("a.md",))
    rec = make_record(chunks=["x"])
    assert rec["retrieval"] is not None
    rec["retrieval"]["chunks"][0]["metadata"] = {}
    assert rule_verdict(q, rec)["expected_docs_retrieved"] is False

  def test_documents_are_compared_exactly(self) -> None:
    q = question(documents=("a.md",))
    out = rule_verdict(q, make_record(chunks=["x"], sources=["A.md"]))
    assert out["expected_docs_retrieved"] is False

  def test_documents_do_not_change_the_verdict(self) -> None:
    q = question(documents=("zzz.md",))
    assert rule_verdict(q, make_record())["verdict"] == "correct"
    assert rule_verdict(q, make_record(sources=["b.md"]))["verdict"] == "correct"

  def test_fields_are_filled_for_refusal_questions_too(self) -> None:
    q = question(refusal=True, documents=("a.md",))
    out = rule_verdict(q, make_record(status="insufficient"))
    assert out["in_retrieval"] is True
    assert out["retrieved_rank"] == 1
    assert out["expected_docs_retrieved"] is True


# --- judge_messages ------------------------------------------------------------------


class TestJudgeMessages:
  def test_system_then_user(self) -> None:
    msgs = judge_messages(question(), "alfa", "answered")
    assert isinstance(msgs, tuple)
    assert [m.role for m in msgs] == ["system", "user"]
    assert all(isinstance(m, PromptMessage) for m in msgs)

  def test_system_message_defines_the_grader_in_spanish(self) -> None:
    system = judge_messages(question(), "a", "answered")[0].content
    assert "estricto" in system
    assert "significado" in system
    assert "redacción" in system
    assert "únicamente" in system
    assert "JSON" in system
    assert "español" in system

  def test_system_message_gives_the_reply_shape(self) -> None:
    system = judge_messages(question(), "a", "answered")[0].content
    assert '"verdict"' in system
    assert '"reason"' in system
    for verdict in JUDGE_VERDICTS:
      assert f'"{verdict}"' in system

  def test_system_message_defines_partial_and_refusal(self) -> None:
    system = judge_messages(question(), "a", "answered")[0].content
    assert "incompleta" in system
    assert "detalle erróneo" in system
    assert "negarse" in system
    assert "las fuentes no son suficientes" in system

  def test_user_message_holds_every_labelled_field(self) -> None:
    q = question(question="¿Quién llegó?", expected="Llegó Sancho.")
    user = judge_messages(q, "Sancho llegó tarde", "insufficient")[1].content
    assert "Pregunta: ¿Quién llegó?" in user
    assert "Respuesta esperada: Llegó Sancho." in user
    assert "Estado del generador: insufficient" in user
    assert "Respuesta candidata: Sancho llegó tarde" in user
    assert "se niegue" in user

  def test_refusal_flag_is_yes_or_no(self) -> None:
    no = judge_messages(question(), "a", "answered")[1].content
    yes = judge_messages(question(refusal=True), "a", "answered")[1].content
    assert "se niegue a responder?: no" in no
    assert "se niegue a responder?: sí" in yes

  def test_candidate_answer_is_last_and_verbatim(self) -> None:
    answer = 'línea 1\nlínea 2 {"verdict": "correct"}'
    user = judge_messages(question(), answer, "answered")[1].content
    assert user.endswith(f"Respuesta candidata: {answer}")

  def test_empty_answer_is_allowed(self) -> None:
    user = judge_messages(question(), "", "no_context")[1].content
    assert user.endswith("Respuesta candidata: ")

  def test_deterministic(self) -> None:
    assert judge_messages(question(), "a", "s") == judge_messages(question(), "a", "s")


# --- judge --------------------------------------------------------------------------


class TestJudge:
  def test_valid_reply(self) -> None:
    client = FakeJudge(verdict_json("partial", "Le falta un dato."))
    out = judge(question(), make_record(), client)
    assert out is not None
    assert out["verdict"] == "partial"
    assert out["reason"] == "Le falta un dato."
    assert out["model"] == "fake-judge"
    assert out["error"] is None
    assert out["raw"] == client.reply
    assert out["seconds"] >= 0.0
    assert set(out) == {
      "verdict",
      "reason",
      "model",
      "seconds",
      "messages",
      "raw",
      "error",
    }

  @pytest.mark.parametrize("verdict", JUDGE_VERDICTS)
  def test_every_verdict_is_accepted(self, verdict: str) -> None:
    out = judge(question(), make_record(), FakeJudge(verdict_json(verdict)))
    assert out is not None
    assert out["verdict"] == verdict

  def test_client_gets_the_judge_messages_of_the_answer(self) -> None:
    client = FakeJudge(verdict_json())
    rec = make_record(status="insufficient", answer="no lo sé")
    out = judge(question(), rec, client)
    expected = judge_messages(question(), "no lo sé", "insufficient")
    assert client.calls == [expected]
    assert out is not None
    assert out["messages"] == [
      {"role": "system", "content": expected[0].content},
      {"role": "user", "content": expected[1].content},
    ]

  def test_called_exactly_once(self) -> None:
    client = FakeJudge(verdict_json())
    judge(question(), make_record(), client)
    assert len(client.calls) == 1

  @pytest.mark.parametrize(
    "wrap",
    [
      "```json\n{}\n```",
      "```\n{}\n```",
      "```JSON\n{}\n```",
      "  \n```json\n{}\n```  \n",
      "```json {}```",
      "```json\r\n{}\r\n```",
      "\n\n  {}\n\t",
    ],
  )
  def test_fence_and_whitespace_are_tolerated(self, wrap: str) -> None:
    reply = wrap.replace("{}", verdict_json("incorrect", "Se equivoca."))
    out = judge(question(), make_record(), FakeJudge(reply))
    assert out is not None
    assert out["error"] is None, out["error"]
    assert out["verdict"] == "incorrect"
    assert out["raw"] == reply

  def test_raw_is_kept_unstripped(self) -> None:
    reply = "  " + verdict_json() + "\n"
    out = judge(question(), make_record(), FakeJudge(reply))
    assert out is not None
    assert out["raw"] == reply

  def test_extra_keys_are_ignored(self) -> None:
    reply = json.dumps({"verdict": "correct", "reason": "ok", "score": 9})
    out = judge(question(), make_record(), FakeJudge(reply))
    assert out is not None
    assert out["verdict"] == "correct"
    assert out["error"] is None

  def test_reason_is_stripped(self) -> None:
    out = judge(question(), make_record(), FakeJudge(verdict_json(reason="  ok \n")))
    assert out is not None
    assert out["reason"] == "ok"

  def test_reason_may_hold_a_backtick_fence_inside(self) -> None:
    reply = verdict_json(reason="usa ``` en el texto")
    out = judge(question(), make_record(), FakeJudge(reply))
    assert out is not None
    assert out["verdict"] == "correct"

  def test_no_result_returns_none_and_skips_the_client(self) -> None:
    client = FakeJudge(verdict_json())
    assert judge(question(), make_record(result=False), client) is None
    assert client.calls == []

  def test_no_generation_returns_none(self) -> None:
    rec = make_record()
    rec["generation"] = None
    assert judge(question(), rec, FakeJudge(verdict_json())) is None

  def test_errored_record_with_a_result_is_still_judged(self) -> None:
    # The task only returns None without a generation result.
    client = FakeJudge(verdict_json())
    out = judge(question(), make_record(error="boom"), client)
    assert out is not None
    assert out["verdict"] == "correct"

  def test_provider_error(self) -> None:
    client = FakeJudge(error=ProviderError("Ollama no responde"))
    out = judge(question(), make_record(), client)
    assert out is not None
    assert out["verdict"] is None
    assert out["reason"] is None
    assert out["raw"] is None
    assert out["error"] == "Ollama no responde"
    assert out["model"] == "fake-judge"
    assert len(out["messages"]) == 2
    assert out["seconds"] >= 0.0

  def test_other_exceptions_propagate(self) -> None:
    client = FakeJudge(error=RuntimeError("bug"))
    with pytest.raises(RuntimeError, match="bug"):
      judge(question(), make_record(), client)

  def test_seconds_times_the_client_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([10.0, 12.5, 99.0, 100.0])
    monkeypatch.setattr("hermes_report.evaluate.time.perf_counter", lambda: next(ticks))
    out = judge(question(), make_record(), FakeJudge(verdict_json()))
    assert out is not None
    assert out["seconds"] == 2.5

  def test_record_is_not_mutated(self) -> None:
    rec = make_record()
    before = copy.deepcopy(rec)
    judge(question(), rec, FakeJudge(verdict_json()))
    assert rec == before

  def test_output_is_json_serialisable(self) -> None:
    out = judge(question(), make_record(), FakeJudge(verdict_json()))
    assert json.loads(json.dumps(out)) == out

  def test_echo_judge_client_round_trip(self) -> None:
    client: JudgeClient = EchoJudgeClient()
    out = judge(question(), make_record(), client)
    assert out is not None
    assert out["verdict"] == "partial"
    assert out["error"] is None
    assert out["model"] == "echo"


class TestJudgeInvalidReplies:
  PREFIX = "respuesta del juez no válida: "

  def assert_invalid(self, reply: str, detail: str | None = None) -> None:
    out = judge(question(), make_record(), FakeJudge(reply))
    assert out is not None
    assert out["verdict"] is None
    assert out["reason"] is None
    assert out["raw"] == reply
    assert out["error"] is not None
    assert out["error"].startswith(self.PREFIX)
    if detail:
      assert detail in out["error"]
    assert len(out["messages"]) == 2
    assert out["model"] == "fake-judge"

  def test_not_json(self) -> None:
    self.assert_invalid("correct, porque sí", "JSON")

  def test_empty_reply(self) -> None:
    self.assert_invalid("")

  def test_whitespace_reply(self) -> None:
    self.assert_invalid("   \n ")

  def test_truncated_json(self) -> None:
    self.assert_invalid('{"verdict": "correct", "reason": "ok')

  def test_unclosed_fence(self) -> None:
    self.assert_invalid("```json\n" + verdict_json())

  @pytest.mark.parametrize("reply", ["[]", '"correct"', "42", "null", "true", "[1, 2]"])
  def test_json_that_is_not_an_object(self, reply: str) -> None:
    self.assert_invalid(reply, "objeto")

  def test_list_of_verdicts(self) -> None:
    self.assert_invalid("[" + verdict_json() + "]", "objeto")

  def test_wrong_verdict(self) -> None:
    self.assert_invalid(verdict_json("excellent"), "verdict")

  @pytest.mark.parametrize(
    "verdict", ["Correct", "CORRECT", " correct", "correct ", ""]
  )
  def test_verdict_must_match_exactly(self, verdict: str) -> None:
    self.assert_invalid(verdict_json(verdict), "verdict")

  @pytest.mark.parametrize("verdict", [None, True, 1, 0, ["correct"], {"a": 1}, 1.5])
  def test_verdict_of_a_wrong_type(self, verdict: Any) -> None:
    self.assert_invalid(json.dumps({"verdict": verdict, "reason": "ok"}), "verdict")

  def test_missing_verdict(self) -> None:
    self.assert_invalid(json.dumps({"reason": "ok"}), "verdict")

  def test_missing_reason(self) -> None:
    self.assert_invalid(json.dumps({"verdict": "correct"}), "reason")

  @pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
  def test_empty_reason(self, reason: str) -> None:
    self.assert_invalid(verdict_json(reason=reason), "reason")

  @pytest.mark.parametrize("reason", [None, 3, True, ["ok"], {"a": "b"}])
  def test_reason_of_a_wrong_type(self, reason: Any) -> None:
    self.assert_invalid(json.dumps({"verdict": "correct", "reason": reason}), "reason")

  def test_invalid_verdict_with_empty_reason_reports_the_verdict_first(self) -> None:
    self.assert_invalid(verdict_json("bad", ""), "verdict")

  def test_deeply_nested_json_does_not_crash(self) -> None:
    self.assert_invalid("[" * 100_000 + "]" * 100_000)

  def test_deeply_nested_object_does_not_crash(self) -> None:
    self.assert_invalid('{"a":' * 100_000 + "1" + "}" * 100_000)

  def test_json_with_nan_constants_is_not_a_valid_verdict(self) -> None:
    self.assert_invalid('{"verdict": NaN, "reason": "ok"}')

  def test_prose_around_the_json_is_not_accepted(self) -> None:
    self.assert_invalid("Aquí tienes: " + verdict_json())

  def test_two_objects_are_not_accepted(self) -> None:
    self.assert_invalid(verdict_json() + verdict_json())

  def test_non_ascii_garbage(self) -> None:
    self.assert_invalid("﻿\x00ñ")

  def test_invalid_reply_is_json_serialisable(self) -> None:
    out = judge(question(), make_record(), FakeJudge("no es json"))
    assert json.loads(json.dumps(out)) == out

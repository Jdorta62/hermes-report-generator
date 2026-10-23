"""Tests for ``hermes_report.record``: converters, JSON round trip, file I/O."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
from hermes_core.enrichment import (
  ChunkPosition,
  ContextChunk,
  ContextPiece,
  EnrichedContext,
)
from hermes_core.generation import (
  Citation,
  GenerationResult,
  PromptMessage,
  SourceChunk,
  SourcePiece,
)
from hermes_core.retrieval import RetrievedChunk, SearchedQuery
from hermes_enrich import ChunkRange

from hermes_report.record import (
  READABLE_SCHEMAS,
  SCHEMA_VERSION,
  RecordError,
  RunRecord,
  chunk_range_record,
  context_record,
  generation_result_record,
  json_safe,
  message_record,
  point_record,
  read_record,
  retrieved_chunk_record,
  searched_query_record,
  vector_summary,
  write_record,
)


def roundtrip(value: Any) -> Any:
  """The value after a JSON dump and load."""
  return json.loads(json.dumps(value))


# --- builders of real Hermes objects ------------------------------------------


def make_chunk(
  chunk_id: str = "c1",
  *,
  metadata: dict[str, Any] | None = None,
  origin: tuple[str, ...] = (),
  score: float | None = 0.75,
  position: ChunkPosition | None = None,
  text: str = "texto uno",
) -> ContextChunk:
  meta = {"source": "a.md", "doc_id": "d1", "chunk_index": 3}
  if metadata is not None:
    meta = metadata
  return ContextChunk(
    chunk_id=chunk_id,
    text=text,
    collection_name="col",
    position=position if position is not None else ChunkPosition("d1", 3, 10, 20, 2),
    score=score,
    metadata=meta,
    origin=origin,
  )


def make_context() -> EnrichedContext:
  first = ContextPiece(
    (make_chunk("c1"), make_chunk("c2", origin=("neighbors",), score=None)),
    transforms=("merge",),
  )
  second = ContextPiece(
    (make_chunk("c3", metadata={"source": 7}, position=None),), transforms=()
  )
  # ContextChunk without position: build directly (a falsy position).
  bare = ContextChunk("c4", "x", "col", None, 0.1, {}, ())
  third = ContextPiece((bare,))
  return EnrichedContext(
    pieces=(first, second, third),
    enrichments=("neighbors", "merge"),
    dropped_chunk_ids=("c9", "c8"),
    query_id="q-1",
  )


def make_result() -> GenerationResult:
  return GenerationResult(
    query_id="q-1",
    status="answered",
    answer="Respuesta [1]",
    strategy="default",
    response_format="plain",
    model="m",
    prompt_tokens=10,
    completion_tokens=5,
    citations=(Citation(1, ("c1", "c2")),),
    sources=(
      SourcePiece(
        1,
        (
          SourceChunk("c1", "col", "d1", "a.md", 3, None, None, 0.5, ()),
          SourceChunk("c2", "col", None, None, None, 1, 2, None, ("neighbors",)),
        ),
        transforms=("merge",),
      ),
    ),
    invalid_citations=("[7]",),
    enrichments=("neighbors",),
    dropped_chunk_ids=("c9",),
    prompt=(PromptMessage("system", "sé breve"), PromptMessage("user", "¿Qué?")),
  )


# --- vector_summary -----------------------------------------------------------


class TestVectorSummary:
  def test_fields(self) -> None:
    out = vector_summary([3.0, 4.0])
    assert out == {"dimension": 2, "norm": 5.0, "head": [3.0, 4.0]}
    assert roundtrip(out) == out

  def test_head_is_first_eight(self) -> None:
    out = vector_summary([float(i) for i in range(20)])
    assert out["dimension"] == 20
    assert out["head"] == [float(i) for i in range(8)]
    assert out["norm"] == pytest.approx(math.sqrt(sum(i * i for i in range(20))))

  def test_exactly_eight_and_nine(self) -> None:
    assert len(vector_summary([1.0] * 8)["head"]) == 8
    assert len(vector_summary([1.0] * 9)["head"]) == 8

  def test_empty(self) -> None:
    assert vector_summary([]) == {"dimension": 0, "norm": 0.0, "head": []}

  def test_accepts_tuple_and_ints_and_gives_floats(self) -> None:
    out = vector_summary((1, 2, 2))
    assert out["norm"] == 3.0
    assert all(isinstance(v, float) for v in out["head"])

  def test_head_is_a_list(self) -> None:
    assert isinstance(vector_summary((1.0,))["head"], list)

  def test_large_values_do_not_overflow(self) -> None:
    out = vector_summary([1e200, 1e200])
    assert math.isfinite(out["norm"])
    assert out["norm"] == pytest.approx(math.sqrt(2) * 1e200)

  def test_negative_values(self) -> None:
    assert vector_summary([-3.0, -4.0])["norm"] == 5.0


# --- point_record -------------------------------------------------------------


class TestPointRecord:
  def test_fields_with_vector(self) -> None:
    out = point_record(
      "abc", {"text": "t", "tags": ("a", "b"), "n": {"x": (1, 2)}}, [3.0, 4.0]
    )
    assert out == {
      "id": "abc",
      "payload": {"text": "t", "tags": ["a", "b"], "n": {"x": [1, 2]}},
      "vector": {"dimension": 2, "norm": 5.0, "head": [3.0, 4.0]},
    }
    assert roundtrip(out) == out

  def test_int_id_becomes_str(self) -> None:
    assert point_record(42, {}, None)["id"] == "42"

  def test_zero_id(self) -> None:
    assert point_record(0, {}, None)["id"] == "0"

  def test_no_vector(self) -> None:
    assert point_record("a", {}, None)["vector"] is None

  def test_empty_vector_is_summarised_not_none(self) -> None:
    assert point_record("a", {}, [])["vector"] == {
      "dimension": 0,
      "norm": 0.0,
      "head": [],
    }

  def test_payload_is_a_fresh_copy(self) -> None:
    payload: dict[str, Any] = {"nested": {"k": [1]}}
    out = point_record("a", payload, None)
    assert out["payload"] == payload
    assert out["payload"] is not payload
    assert out["payload"]["nested"] is not payload["nested"]
    out["payload"]["nested"]["k"].append(2)
    assert payload["nested"]["k"] == [1]


# --- retrieved_chunk_record / searched_query_record ---------------------------


class TestRetrievedChunkRecord:
  def test_fields_and_text_key_removed(self) -> None:
    chunk = RetrievedChunk(
      "c1", "cuerpo", 0.5, "col", {"text": "cuerpo", "source": "a.md", "t": (1,)}, 0.9
    )
    out = retrieved_chunk_record(chunk)
    assert out == {
      "chunk_id": "c1",
      "text": "cuerpo",
      "score": 0.5,
      "vector_score": 0.9,
      "collection_name": "col",
      "metadata": {"source": "a.md", "t": [1]},
    }
    assert roundtrip(out) == out

  def test_vector_score_none_is_present(self) -> None:
    out = retrieved_chunk_record(RetrievedChunk("c1", "t", 1.0, "col", {}))
    assert "vector_score" in out
    assert out["vector_score"] is None
    assert out["metadata"] == {}

  def test_metadata_is_a_copy_and_source_untouched(self) -> None:
    meta = {"text": "t", "k": {"a": 1}}
    chunk = RetrievedChunk("c1", "t", 1.0, "col", meta)
    out = retrieved_chunk_record(chunk)
    out["metadata"]["k"]["a"] = 2
    assert meta == {"text": "t", "k": {"a": 1}}

  def test_only_exact_text_key_removed(self) -> None:
    chunk = RetrievedChunk("c1", "t", 1.0, "col", {"Text": 1, "text_x": 2})
    assert retrieved_chunk_record(chunk)["metadata"] == {"Text": 1, "text_x": 2}


class TestSearchedQueryRecord:
  def test_fields(self) -> None:
    out = searched_query_record(SearchedQuery("hola", ("hyde", "multi")))
    assert out == {"text": "hola", "chain": ["hyde", "multi"]}
    assert isinstance(out["chain"], list)
    assert roundtrip(out) == out

  def test_original_query_empty_chain(self) -> None:
    assert searched_query_record(SearchedQuery("hola", ())) == {
      "text": "hola",
      "chain": [],
    }


# --- context_record -----------------------------------------------------------


class TestContextRecord:
  def test_field_by_field(self) -> None:
    out = context_record(make_context())
    assert out["enrichments"] == ["neighbors", "merge"]
    assert out["dropped_chunk_ids"] == ["c9", "c8"]
    assert out["query_id"] == "q-1"
    assert len(out["pieces"]) == 3
    first = out["pieces"][0]
    assert first["text"] == make_context().pieces[0].text
    assert first["transforms"] == ["merge"]
    assert first["chunks"] == [
      {
        "chunk_id": "c1",
        "collection_name": "col",
        "source": "a.md",
        "doc_id": "d1",
        "chunk_index": 3,
        "score": 0.75,
        "origin": [],
      },
      {
        "chunk_id": "c2",
        "collection_name": "col",
        "source": "a.md",
        "doc_id": "d1",
        "chunk_index": 3,
        "score": None,
        "origin": ["neighbors"],
      },
    ]

  def test_chunk_refs_carry_no_text(self) -> None:
    out = context_record(make_context())
    for piece in out["pieces"]:
      for chunk in piece["chunks"]:
        assert "text" not in chunk
        assert "metadata" not in chunk

  def test_non_str_source_is_none(self) -> None:
    # Second piece: metadata source is the int 7 and the position is given.
    chunk = context_record(make_context())["pieces"][1]["chunks"][0]
    assert chunk["source"] is None

  def test_missing_source_key_is_none(self) -> None:
    chunk = context_record(make_context())["pieces"][2]["chunks"][0]
    assert chunk["source"] is None

  def test_no_position_gives_none_doc_and_index(self) -> None:
    chunk = context_record(make_context())["pieces"][2]["chunks"][0]
    assert chunk["doc_id"] is None
    assert chunk["chunk_index"] is None
    assert chunk["score"] == 0.1

  def test_chunk_index_zero_is_kept(self) -> None:
    chunk = ContextChunk(
      "c", "t", "col", ChunkPosition("d", 0, None, None, None), 1, {}, ()
    )
    out = context_record(EnrichedContext((ContextPiece((chunk,)),)))
    assert out["pieces"][0]["chunks"][0]["chunk_index"] == 0

  def test_empty_doc_id_is_kept(self) -> None:
    chunk = ContextChunk(
      "c", "t", "col", ChunkPosition("", 0, None, None, None), 1, {}, ()
    )
    out = context_record(EnrichedContext((ContextPiece((chunk,)),)))
    assert out["pieces"][0]["chunks"][0]["doc_id"] == ""

  def test_empty_context(self) -> None:
    out = context_record(EnrichedContext())
    assert out == {
      "pieces": [],
      "enrichments": [],
      "dropped_chunk_ids": [],
      "query_id": None,
    }

  def test_json_round_trip(self) -> None:
    out = context_record(make_context())
    assert roundtrip(out) == out

  def test_lists_are_lists(self) -> None:
    out = context_record(make_context())
    assert isinstance(out["pieces"], list)
    assert isinstance(out["enrichments"], list)
    assert isinstance(out["dropped_chunk_ids"], list)
    assert isinstance(out["pieces"][0]["transforms"], list)
    assert isinstance(out["pieces"][0]["chunks"], list)
    assert isinstance(out["pieces"][0]["chunks"][1]["origin"], list)

  def test_does_not_mutate_input_and_is_fresh(self) -> None:
    ctx = make_context()
    a = context_record(ctx)
    b = context_record(ctx)
    assert a == b
    assert a is not b
    assert a["pieces"] is not b["pieces"]


# --- chunk_range_record / message_record --------------------------------------


class TestChunkRangeRecord:
  def test_fields(self) -> None:
    out = chunk_range_record(ChunkRange("col", "d1", 2, 5))
    assert out == {"collection_name": "col", "doc_id": "d1", "first": 2, "last": 5}
    assert roundtrip(out) == out

  def test_single_chunk_range(self) -> None:
    out = chunk_range_record(ChunkRange("col", "d1", 0, 0))
    assert (out["first"], out["last"]) == (0, 0)


class TestMessageRecord:
  @pytest.mark.parametrize("role", ["system", "user", "assistant"])
  def test_fields(self, role: Any) -> None:
    out = message_record(PromptMessage(role, "hola ñ"))
    assert out == {"role": role, "content": "hola ñ"}
    assert roundtrip(out) == out

  def test_empty_content(self) -> None:
    assert message_record(PromptMessage("user", ""))["content"] == ""


# --- generation_result_record / json_safe -------------------------------------


class TestGenerationResultRecord:
  def test_field_by_field(self) -> None:
    out = generation_result_record(make_result())
    assert out["query_id"] == "q-1"
    assert out["status"] == "answered"
    assert out["answer"] == "Respuesta [1]"
    assert out["strategy"] == "default"
    assert out["response_format"] == "plain"
    assert out["model"] == "m"
    assert out["prompt_tokens"] == 10
    assert out["completion_tokens"] == 5
    assert out["citations"] == [{"number": 1, "chunk_ids": ["c1", "c2"]}]
    assert out["invalid_citations"] == ["[7]"]
    assert out["enrichments"] == ["neighbors"]
    assert out["dropped_chunk_ids"] == ["c9"]
    assert out["prompt"] == [
      {"role": "system", "content": "sé breve"},
      {"role": "user", "content": "¿Qué?"},
    ]
    sources = out["sources"]
    assert len(sources) == 1
    assert sources[0]["number"] == 1
    assert sources[0]["transforms"] == ["merge"]
    assert sources[0]["chunks"][0] == {
      "chunk_id": "c1",
      "collection_name": "col",
      "doc_id": "d1",
      "source": "a.md",
      "chunk_index": 3,
      "page_start": None,
      "page_end": None,
      "score": 0.5,
      "origin": [],
    }
    assert sources[0]["chunks"][1]["origin"] == ["neighbors"]
    assert sources[0]["chunks"][1]["page_start"] == 1

  def test_no_tuple_anywhere(self) -> None:
    def walk(value: Any) -> None:
      assert not isinstance(value, tuple)
      if isinstance(value, dict):
        for item in value.values():
          walk(item)
      elif isinstance(value, list):
        for item in value:
          walk(item)

    walk(generation_result_record(make_result()))

  def test_equals_json_round_trip(self) -> None:
    out = generation_result_record(make_result())
    assert roundtrip(out) == out

  def test_minimal_result_with_prompt_none(self) -> None:
    result = GenerationResult(
      query_id=None,
      status="no_context",
      answer="",
      strategy="s",
      response_format="plain",
    )
    out = generation_result_record(result)
    assert out["prompt"] is None
    assert out["citations"] == []
    assert out["sources"] == []
    assert out["model"] is None
    assert out["prompt_tokens"] is None
    assert roundtrip(out) == out

  def test_empty_prompt_tuple_is_empty_list(self) -> None:
    result = GenerationResult(None, "answered", "a", "s", "f", prompt=())
    assert generation_result_record(result)["prompt"] == []


class TestJsonSafe:
  def test_tuple_to_list_recursively(self) -> None:
    value = {"a": (1, (2, 3)), "b": [(4,), {"c": (5,)}], "d": "x", "e": None}
    assert json_safe(value) == {
      "a": [1, [2, 3]],
      "b": [[4], {"c": [5]}],
      "d": "x",
      "e": None,
    }

  def test_result_has_no_tuples(self) -> None:
    out = json_safe({"a": ((1,),)})
    assert type(out["a"]) is list
    assert type(out["a"][0]) is list

  def test_scalars_pass_through(self) -> None:
    for value in (1, 1.5, "s", None, True):
      assert json_safe(value) == value

  def test_empty_containers(self) -> None:
    assert json_safe(()) == []
    assert json_safe({}) == {}
    assert json_safe([]) == []

  def test_input_not_mutated(self) -> None:
    inner = [1]
    value = {"a": inner}
    out = json_safe(value)
    assert out is not value
    assert out["a"] is not inner

  def test_dict_keys_are_kept(self) -> None:
    assert json_safe({"k": (1,)}).keys() == {"k"}


# --- full record, write / read ------------------------------------------------


def make_record() -> RunRecord:
  ctx = context_record(make_context())
  messages = [message_record(PromptMessage("user", "¿Hola? ñ 日本"))]
  result = generation_result_record(make_result())
  return {
    "schema": SCHEMA_VERSION,
    "title": "Prueba",
    "description": "descripción con acentos: áéí",
    "started_at": "2026-10-21T10:00:00+01:00",
    "finished_at": "2026-10-21T10:01:00+01:00",
    "seconds": 60.5,
    "config": {"title": "Prueba", "retrieval": {"top_k": 5}},
    "environment": {
      "python": "3.12.0",
      "packages": {"hermes-core": "0.6.0"},
      "ollama_host": "http://localhost:11434",
      "qdrant_url": ":memory:",
    },
    "corpus": {
      "dir": "corpus",
      "glob": "**/*.md",
      "fingerprint": "abc",
      "documents": [{"source": "a.md", "sha256": "ff", "chars": 10, "chunks": 2}],
    },
    "ingest": {
      "seconds": 1.5,
      "chunker": {"size": 100},
      "embedding": {"provider": "hash", "model": "h", "dimension": 8},
      "ingester_version": "1",
      "documents": [{"source": "a.md", "chunks": 2, "seconds": 0.5, "cached": False}],
    },
    "collection": {
      "name": "hermes_report_x",
      "url": ":memory:",
      "reloaded": True,
      "points": 2,
      "vector_size": 8,
      "distance": "Cosine",
      "indexes": ["doc_id"],
    },
    "points": [point_record("p1", {"text": "t", "tags": ("a",)}, [1.0, 2.0])],
    "questions": [
      {
        "spec": {"id": "q1", "question": "¿Qué?"},
        "query_id": "q-1",
        "seconds": 3.25,
        "error": None,
        "retrieval": {
          "seconds": 0.5,
          "chunks": [
            retrieved_chunk_record(
              RetrievedChunk("c1", "t", 0.5, "col", {"source": "a.md"}, 0.4)
            )
          ],
          "queries": [searched_query_record(SearchedQuery("q", ("hyde",)))],
        },
        "enrich": {
          "seconds": 0.1,
          "budget": {"max_context_tokens": 100, "max_chunks": 5},
          "steps": [
            {
              "name": "neighbors",
              "phase": "EXPAND",
              "seconds": 0.05,
              "before": ctx,
              "after": ctx,
              "reads": [
                {
                  "ranges": [chunk_range_record(ChunkRange("col", "d1", 0, 2))],
                  "returned": ["c1"],
                  "seconds": 0.01,
                }
              ],
            }
          ],
          "context": ctx,
        },
        "generation": {
          "seconds": 2.0,
          "llm_calls": [
            {
              "messages": messages,
              "params": {"temperature": 0.0},
              "reply": {
                "text": "r",
                "model": "m",
                "prompt_tokens": 1,
                "completion_tokens": 2,
              },
              "error": None,
              "seconds": 1.9,
            }
          ],
          "result": result,
        },
        "rules": {
          "verdict": "ok",
          "in_retrieval": True,
          "in_context": True,
          "in_cited": False,
          "answer_ok": None,
          "retrieved_rank": 1,
          "expected_docs_retrieved": True,
        },
        "judge": {
          "verdict": "correct",
          "reason": "bien",
          "model": "j",
          "seconds": 0.7,
          "messages": messages,
          "raw": None,
          "error": None,
        },
        "metrics": [
          {
            "name": "doc_retrieval",
            "status": "ok",
            "scores": {"doc_hit": 1.0, "doc_mrr": None},
            "details": {"ranks": [1]},
            "calls": [
              {
                "purpose": "answer_claims",
                "messages": messages,
                "raw": "{}",
                "error": None,
                "seconds": 0.2,
              }
            ],
            "error": None,
            "seconds": 0.01,
          }
        ],
      },
      {
        "spec": {},
        "query_id": "q-2",
        "seconds": 0.0,
        "error": "boom",
        "retrieval": None,
        "enrich": None,
        "generation": None,
        "rules": None,
        "judge": None,
        "metrics": None,
      },
    ],
    "metrics": {
      "selection": ["doc_retrieval"],
      "scores": ["doc_hit", "doc_mrr"],
      "summary": {
        "doc_hit": {
          "mean": 1.0,
          "n": 1,
          "not_applicable": 0,
          "skipped": 0,
          "errors": 0,
        },
        "doc_mrr": {
          "mean": None,
          "n": 0,
          "not_applicable": 1,
          "skipped": 0,
          "errors": 0,
        },
      },
    },
  }


class TestWriteRecord:
  def test_full_round_trip(self, tmp_path: Path) -> None:
    record = make_record()
    path = tmp_path / "run.json"
    write_record(record, path)
    assert read_record(path) == record
    assert read_record(path) == roundtrip(record)

  def test_record_equals_its_json_round_trip(self) -> None:
    record = make_record()
    assert roundtrip(record) == record

  def test_creates_parent_folders(self, tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "run.json"
    write_record(make_record(), path)
    assert path.is_file()

  def test_overwrites_existing(self, tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text("old", encoding="utf-8")
    record = make_record()
    write_record(record, path)
    assert read_record(path) == record

  def test_format_utf8_not_ascii_escaped_indent_one(self, tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    write_record(make_record(), path)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert "日本" in text
    assert "áéí" in text
    assert "\\u" not in text
    assert text.startswith('{\n "schema": 2,\n "title": "Prueba",')
    assert text == json.dumps(make_record(), ensure_ascii=False, indent=1)

  def test_bad_record_leaves_no_file_and_no_folder(self, tmp_path: Path) -> None:
    path = tmp_path / "new" / "run.json"
    record: Any = make_record()
    record["seconds"] = object()
    with pytest.raises(RecordError) as info:
      write_record(record, path)
    assert str(path) in str(info.value)
    assert not path.exists()
    assert not path.parent.exists()

  def test_bad_record_keeps_previous_file(self, tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    good = make_record()
    write_record(good, path)
    bad: Any = make_record()
    bad["seconds"] = float("nan")
    with pytest.raises(RecordError):
      write_record(bad, path)
    assert read_record(path) == good

  @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
  def test_non_finite_floats_rejected(self, tmp_path: Path, value: float) -> None:
    record: Any = make_record()
    record["seconds"] = value
    path = tmp_path / "run.json"
    with pytest.raises(RecordError):
      write_record(record, path)
    assert not path.exists()

  def test_circular_reference_rejected(self, tmp_path: Path) -> None:
    record: Any = make_record()
    record["config"]["self"] = record["config"]
    path = tmp_path / "run.json"
    with pytest.raises(RecordError):
      write_record(record, path)
    assert not path.exists()

  def test_lone_surrogate_rejected(self, tmp_path: Path) -> None:
    record: Any = make_record()
    record["title"] = "\ud800"
    path = tmp_path / "run.json"
    with pytest.raises(RecordError):
      write_record(record, path)
    assert not path.exists()

  def test_deep_nesting_rejected(self, tmp_path: Path) -> None:
    deep: Any = []
    for _ in range(100_000):
      deep = [deep]
    record: Any = make_record()
    record["config"]["deep"] = deep
    with pytest.raises(RecordError):
      write_record(record, tmp_path / "run.json")

  def test_non_string_key_type_rejected(self, tmp_path: Path) -> None:
    record: Any = make_record()
    record["config"][object()] = 1
    with pytest.raises(RecordError):
      write_record(record, tmp_path / "run.json")

  def test_parent_is_a_file(self, tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    path = blocker / "run.json"
    with pytest.raises(RecordError) as info:
      write_record(make_record(), path)
    assert str(path) in str(info.value)

  def test_path_is_a_directory(self, tmp_path: Path) -> None:
    with pytest.raises(RecordError) as info:
      write_record(make_record(), tmp_path)
    assert str(tmp_path) in str(info.value)

  def test_nul_in_path(self, tmp_path: Path) -> None:
    with pytest.raises(RecordError):
      write_record(make_record(), tmp_path / "a\x00b.json")

  def test_unwritable_folder(self, tmp_path: Path) -> None:
    if os.geteuid() == 0:
      pytest.skip("root ignores permissions")
    folder = tmp_path / "ro"
    folder.mkdir()
    folder.chmod(0o500)
    try:
      with pytest.raises(RecordError):
        write_record(make_record(), folder / "run.json")
    finally:
      folder.chmod(0o700)

  def test_error_message_is_spanish(self, tmp_path: Path) -> None:
    record: Any = make_record()
    record["seconds"] = math.nan
    with pytest.raises(RecordError) as info:
      write_record(record, tmp_path / "run.json")
    assert info.value.message
    assert "serializar" in info.value.message


class TestReadRecord:
  def raises(self, path: Path) -> RecordError:
    with pytest.raises(RecordError) as info:
      read_record(path)
    assert str(path) in str(info.value)
    return info.value

  def put(self, tmp_path: Path, content: str | bytes) -> Path:
    path = tmp_path / "run.json"
    if isinstance(content, bytes):
      path.write_bytes(content)
    else:
      path.write_text(content, encoding="utf-8")
    return path

  def test_missing_file(self, tmp_path: Path) -> None:
    err = self.raises(tmp_path / "nope.json")
    assert "no existe" in err.message

  def test_directory(self, tmp_path: Path) -> None:
    self.raises(tmp_path)

  def test_nul_in_path(self, tmp_path: Path) -> None:
    self.raises(tmp_path / "a\x00b.json")

  def test_missing_parent_folder(self, tmp_path: Path) -> None:
    self.raises(tmp_path / "x" / "run.json")

  def test_empty_file(self, tmp_path: Path) -> None:
    err = self.raises(self.put(tmp_path, ""))
    assert "JSON" in err.message

  @pytest.mark.parametrize("content", ["{", '{"schema": 1,}', "not json", "{'a': 1}"])
  def test_invalid_json(self, tmp_path: Path, content: str) -> None:
    err = self.raises(self.put(tmp_path, content))
    assert "JSON" in err.message

  def test_invalid_utf8(self, tmp_path: Path) -> None:
    self.raises(self.put(tmp_path, b'{"schema": 1, "x": "\xff\xfe"}'))

  def test_deep_nesting(self, tmp_path: Path) -> None:
    err = self.raises(self.put(tmp_path, "[" * 200_000 + "]" * 200_000))
    assert "JSON" in err.message

  def test_huge_int(self, tmp_path: Path) -> None:
    self.raises(self.put(tmp_path, '{"schema": 1, "x": ' + "9" * 10_000 + "}"))

  @pytest.mark.parametrize(
    "content", ["[]", '[{"schema": 1}]', '"x"', "1", "null", "true", "1.5"]
  )
  def test_not_an_object(self, tmp_path: Path, content: str) -> None:
    err = self.raises(self.put(tmp_path, content))
    assert "objeto" in err.message

  def test_missing_schema(self, tmp_path: Path) -> None:
    err = self.raises(self.put(tmp_path, '{"title": "x"}'))
    assert "schema" in err.message

  def test_empty_object_missing_schema(self, tmp_path: Path) -> None:
    err = self.raises(self.put(tmp_path, "{}"))
    assert "schema" in err.message

  @pytest.mark.parametrize("schema", [0, 3, -1, 99])
  def test_other_version(self, tmp_path: Path, schema: int) -> None:
    err = self.raises(self.put(tmp_path, json.dumps({"schema": schema})))
    assert str(schema) in err.message

  @pytest.mark.parametrize("schema", ['"1"', "null", "true", "false", "[1]", "{}"])
  def test_schema_of_wrong_type(self, tmp_path: Path, schema: str) -> None:
    self.raises(self.put(tmp_path, '{"schema": ' + schema + "}"))

  def test_schema_float_one_rejected(self, tmp_path: Path) -> None:
    # 1.0 == 1 in Python, but the file declares a float: not the schema version.
    self.raises(self.put(tmp_path, '{"schema": 1.0}'))

  @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
  def test_non_finite_literals_rejected(self, tmp_path: Path, literal: str) -> None:
    self.raises(self.put(tmp_path, '{"schema": 1, "seconds": ' + literal + "}"))

  def test_minimal_valid_returns_object_untouched(self, tmp_path: Path) -> None:
    out = read_record(self.put(tmp_path, '{"schema": 1, "extra": [1, 2]}'))
    # A schema-1 record only gains ``metrics=None`` on the run.
    assert out == {"schema": 1, "extra": [1, 2], "metrics": None}

  def test_accepts_utf8_with_non_ascii(self, tmp_path: Path) -> None:
    path = self.put(tmp_path, '{"schema": 1, "title": "áéí 日本"}')
    assert read_record(path)["title"] == "áéí 日本"

  def test_error_is_record_error_not_leaking_others(self, tmp_path: Path) -> None:
    for content in ("{", "[]", "{}", b"\xff"):
      path = self.put(tmp_path, content)
      with pytest.raises(RecordError):
        read_record(path)

  def test_read_does_not_deep_validate(self, tmp_path: Path) -> None:
    out: Any = read_record(self.put(tmp_path, '{"schema": 1, "points": "nope"}'))
    assert out["points"] == "nope"


class TestSchemaMigration:
  """Schema 1 stays readable (task 5.5.4); schema 2 is read as written."""

  def put(self, tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path

  def test_readable_schemas(self) -> None:
    assert READABLE_SCHEMAS == (1, 2)
    assert SCHEMA_VERSION in READABLE_SCHEMAS

  def test_schema_1_gets_metrics_none_on_run_and_every_question(
    self, tmp_path: Path
  ) -> None:
    data = {"schema": 1, "questions": [{"query_id": "a"}, {"query_id": "b"}]}
    out: Any = read_record(self.put(tmp_path, data))
    assert out["metrics"] is None
    assert [q["metrics"] for q in out["questions"]] == [None, None]
    assert [q["query_id"] for q in out["questions"]] == ["a", "b"]

  def test_schema_1_with_no_questions_or_odd_questions_does_not_crash(
    self, tmp_path: Path
  ) -> None:
    empty = read_record(self.put(tmp_path, {"schema": 1, "questions": []}))
    assert empty["metrics"] is None
    for questions in (None, "x", 3, {"a": 1}, [1, "x", None, {"q": 1}]):
      out: Any = read_record(self.put(tmp_path, {"schema": 1, "questions": questions}))
      assert out["metrics"] is None

  def test_schema_1_odd_questions_only_dicts_are_filled(self, tmp_path: Path) -> None:
    out: Any = read_record(
      self.put(tmp_path, {"schema": 1, "questions": [1, {"q": 1}]})
    )
    assert out["questions"] == [1, {"q": 1, "metrics": None}]

  def test_schema_2_is_not_touched(self, tmp_path: Path) -> None:
    data = {"schema": 2, "questions": [{"query_id": "a"}], "extra": 1}
    assert read_record(self.put(tmp_path, data)) == data

  def test_schema_2_keeps_the_metrics(self, tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    record = make_record()
    write_record(record, path)
    out = read_record(path)
    assert out == record
    assert out["metrics"] is not None
    assert out["questions"][0]["metrics"] is not None
    assert out["questions"][1]["metrics"] is None

  def test_schema_3_is_rejected_naming_both_readable_versions(
    self, tmp_path: Path
  ) -> None:
    with pytest.raises(RecordError) as info:
      read_record(self.put(tmp_path, {"schema": 3}))
    assert "3" in info.value.message
    assert "1 o 2" in info.value.message

  @pytest.mark.parametrize("schema", [True, False, 2.0, 1.0, "2", None])
  def test_schema_of_other_types_is_rejected(self, tmp_path: Path, schema: Any) -> None:
    with pytest.raises(RecordError):
      read_record(self.put(tmp_path, {"schema": schema}))


class TestRecordError:
  def test_attributes_and_str(self) -> None:
    err = RecordError(Path("/x/run.json"), "el archivo no existe")
    assert (err.path, err.message) == (Path("/x/run.json"), "el archivo no existe")
    assert str(err) == "/x/run.json: el archivo no existe"
    assert isinstance(err, Exception)

  def test_schema_version_is_two(self) -> None:
    assert SCHEMA_VERSION == 2

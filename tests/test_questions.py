"""Tests for ``hermes_report.questions``: normalise, rules, file validation."""

from __future__ import annotations

import dataclasses
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hermes_report.questions import (
  Question,
  QuestionsError,
  QuestionSet,
  load_questions,
  normalise,
)

Write = Callable[[object], Path]


@pytest.fixture
def write(tmp_path: Path) -> Write:
  """Write ``data`` (or a raw string) as the questions file."""

  def _write(data: object) -> Path:
    path = tmp_path / "preguntas.json"
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return path

  return _write


def q(**fields: Any) -> dict[str, Any]:
  """A valid raw question, with ``fields`` merged in."""
  return {"question": "¿P?", "expected": "R", **fields}


def fails(
  path: Path, key: str | None, sources: list[str] | None = None
) -> QuestionsError:
  with pytest.raises(QuestionsError) as info:
    load_questions(path, sources)
  err = info.value
  assert err.key == key
  assert str(path) in str(err)
  if key is not None:
    assert key in str(err)
  return err


# --- normalise ------------------------------------------------------------------


class TestNormalise:
  @pytest.mark.parametrize(
    ("text", "expected"),
    [
      ("", ""),
      ("   ", ""),
      ("Hola", "hola"),
      ("ÁÉÍÓÚ áéíóú ñ Ñ ü", "aeiou aeiou n n u"),
      ("Rocinante", "rocinante"),
      ("a  b", "a b"),
      ("a\nb", "a b"),
      ("a\r\nb\t\tc", "a b c"),
      ("  a \n\n b  ", "a b"),
      ("a b", "a b"),  # no-break space is whitespace for str.split
      ("a b", "a b"),
      ("é", "e"),  # already decomposed
      ("é", "e"),  # precomposed
      ("ç", "c"),
      ("ß", "ß"),  # no decomposition: left alone
      ("İ", "i"),  # lower() gives "i" + combining dot, which is stripped
      ("¿Qué?", "¿que?"),
      ("😀", "😀"),
    ],
  )
  def test_cases(self, text: str, expected: str) -> None:
    assert normalise(text) == expected

  def test_idempotent(self) -> None:
    text = "  ÁRBOL\n\tde   la  Vida  "
    assert normalise(normalise(text)) == normalise(text)

  def test_nfc_and_nfd_inputs_agree(self) -> None:
    assert normalise("Manchaña") == normalise("Manchaña")

  def test_long_input(self) -> None:
    assert normalise("á " * 50_000) == " ".join(["a"] * 50_000)


# --- Question rules ---------------------------------------------------------------


class TestMatches:
  def test_none_without_keywords(self) -> None:
    assert Question("q", "?", "r").matches("anything") is None
    assert Question("q", "?", "r").matches("") is None

  def test_all_keywords_must_be_present(self) -> None:
    question = Question("q", "?", "r", keywords=("cuatro", "Rocinante"))
    assert question.matches("Cuatro días; lo llamó ROCINANTE") is True
    assert question.matches("Cuatro días") is False
    assert question.matches("") is False

  def test_accents_and_case_ignored_on_both_sides(self) -> None:
    question = Question("q", "?", "r", keywords=("Año", "MÁS"))
    assert question.matches("el ano y mas") is True
    assert question.matches("EL AÑO Y MÁS") is True

  def test_whitespace_collapsed_in_keyword_and_answer(self) -> None:
    question = Question("q", "?", "r", keywords=("la  Mancha",))
    assert question.matches("en la\nMancha") is True

  def test_substring_semantics(self) -> None:
    # The contract is "is in the normalised answer": a plain substring.
    assert (
      Question("q", "?", "r", keywords=("cuatro",)).matches("cuatrocientos") is True
    )

  def test_empty_answer_with_keywords(self) -> None:
    assert Question("q", "?", "r", keywords=("x",)).matches("") is False

  def test_keyword_of_only_combining_marks_does_not_match_everything(self) -> None:
    # "́" is non-blank for the loader but normalises to "" and would be
    # "in" every answer: a keyword that can never fail is a silent pass.
    question = Question("q", "?", "r", keywords=("́",))
    assert question.matches("zzz") is False


class TestReachable:
  def test_none_without_evidence(self) -> None:
    assert Question("q", "?", "r").reachable("text") is None

  def test_all_phrases_required(self) -> None:
    question = Question("q", "?", "r", evidence=("uno dos", "tres"))
    assert question.reachable("Uno dos y TRES") is True
    assert question.reachable("uno dos") is False

  def test_phrase_matches_across_line_breaks(self) -> None:
    # The reason for collapsing whitespace: chunk text is wrapped.
    question = Question(
      "q", "?", "r", evidence=("Cuatro días se le pasaron en imaginar",)
    )
    assert question.reachable("Cuatro días se le\npasaron   en\n\nimaginar qué") is True

  def test_accents_ignored(self) -> None:
    assert (
      Question("q", "?", "r", evidence=("qué nombre",)).reachable("que NOMBRE") is True
    )

  def test_empty_text(self) -> None:
    assert Question("q", "?", "r", evidence=("x",)).reachable("") is False


class TestQuestionData:
  def test_defaults(self) -> None:
    question = Question(id="a", question="?", expected="r")
    assert (question.keywords, question.evidence, question.documents) == ((), (), ())
    assert question.refusal is False

  def test_to_dict_has_every_field_and_lists(self) -> None:
    question = Question(
      "q1",
      "¿?",
      "r",
      keywords=("a",),
      evidence=("b", "c"),
      documents=("d.md",),
      refusal=True,
    )
    assert question.to_dict() == {
      "id": "q1",
      "question": "¿?",
      "expected": "r",
      "keywords": ["a"],
      "evidence": ["b", "c"],
      "documents": ["d.md"],
      "refusal": True,
    }

  def test_to_dict_empty_tuples_are_lists(self) -> None:
    data = Question("q1", "?", "r").to_dict()
    assert data["keywords"] == [] and data["evidence"] == [] and data["documents"] == []
    assert set(data) == {f.name for f in dataclasses.fields(Question)}
    assert json.loads(json.dumps(data)) == data

  def test_frozen(self) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
      Question("q", "?", "r").id = "x"  # type: ignore[misc]

  def test_hashable(self) -> None:
    assert len({Question("q", "?", "r"), Question("q", "?", "r")}) == 1


class TestQuestionSet:
  def test_ids_and_get(self) -> None:
    a, b = Question("a", "?", "r"), Question("b", "?", "r")
    qs = QuestionSet(path=Path("x.json"), title=None, questions=(a, b))
    assert qs.ids == ("a", "b")
    assert qs.get("b") is b

  def test_get_unknown_raises_keyerror(self) -> None:
    qs = QuestionSet(path=Path("x.json"), title=None, questions=())
    assert qs.ids == ()
    with pytest.raises(KeyError):
      qs.get("nope")

  def test_get_is_exact(self) -> None:
    qs = QuestionSet(Path("x"), None, (Question("q01", "?", "r"),))
    with pytest.raises(KeyError):
      qs.get("Q01")
    with pytest.raises(KeyError):
      qs.get("")


class TestQuestionsError:
  def test_str_and_attributes(self) -> None:
    err = QuestionsError(
      Path("/x/p.json"), "debe ser un texto", "questions[2].keywords"
    )
    assert str(err) == "/x/p.json: questions[2].keywords: debe ser un texto"
    assert (err.path, err.message, err.key) == (
      Path("/x/p.json"),
      "debe ser un texto",
      "questions[2].keywords",
    )

  def test_str_without_key(self) -> None:
    assert (
      str(QuestionsError(Path("/x/p.json"), "JSON inválido"))
      == "/x/p.json: JSON inválido"
    )

  def test_pickle(self) -> None:
    err = pickle.loads(pickle.dumps(QuestionsError(Path("/x"), "m", "k")))  # noqa: S301
    assert str(err) == "/x: k: m"


# --- load_questions: happy paths ---------------------------------------------------


def test_sample_file_loads(data_dir: Path) -> None:
  qs = load_questions(data_dir / "preguntas.json")
  assert qs.path == data_dir / "preguntas.json"
  assert qs.title == "Batería de prueba"
  assert qs.ids == ("q01", "q02", "q11")
  assert qs.get("q11").refusal is True
  assert qs.get("q01").keywords == ("cuatro", "Rocinante")
  assert qs.get("q01").documents == ("parte-1/capitulo-01.md",)
  assert qs.get("q02").keywords == ()


def test_sample_documents_exist_in_sample_corpus(data_dir: Path) -> None:
  sources = ["notas.md", "parte-1/capitulo-01.md", "parte-1/capitulo-02.md"]
  assert load_questions(data_dir / "preguntas.json", sources).ids == (
    "q01",
    "q02",
    "q11",
  )


def test_sample_evidence_is_in_the_sample_corpus(data_dir: Path) -> None:
  qs = load_questions(data_dir / "preguntas.json")
  text = (data_dir / "corpus" / "parte-1" / "capitulo-01.md").read_text(
    encoding="utf-8"
  )
  assert qs.get("q01").reachable(text) is True


def test_minimal_question(write: Write) -> None:
  qs = load_questions(write({"questions": [q()]}))
  assert qs.title is None
  only = qs.questions[0]
  assert only == Question("q01", "¿P?", "R")


def test_default_ids_by_position_padded(write: Write) -> None:
  qs = load_questions(write({"questions": [q() for _ in range(101)]}))
  assert qs.ids[0] == "q01"
  assert qs.ids[8] == "q09"
  assert qs.ids[9] == "q10"
  assert qs.ids[98] == "q99"
  assert qs.ids[99] == "q100"
  assert qs.ids[100] == "q101"


def test_default_id_uses_position_even_after_explicit_ids(write: Write) -> None:
  qs = load_questions(write({"questions": [q(id="a"), q(), q(id="z"), q()]}))
  assert qs.ids == ("a", "q02", "z", "q04")


def test_explicit_ids_may_look_like_defaults_elsewhere(write: Write) -> None:
  qs = load_questions(write({"questions": [q(id="q02"), q(id="q01")]}))
  assert qs.ids == ("q02", "q01")


@pytest.mark.parametrize(
  "question_id", ["a", "Q-1", "q_1", "A1b2", "-", "_", "0", "x" * 500]
)
def test_valid_ids(write: Write, question_id: str) -> None:
  assert load_questions(write({"questions": [q(id=question_id)]})).ids == (question_id,)


def test_order_and_title_are_kept(write: Write) -> None:
  qs = load_questions(write({"title": "T", "questions": [q(id="b"), q(id="a")]}))
  assert qs.title == "T"
  assert qs.ids == ("b", "a")


def test_unicode_content_survives(write: Write) -> None:
  qs = load_questions(write({"questions": [q(question="¿Ñandú 😀?", keywords=["ñ"])]}))
  assert qs.questions[0].question == "¿Ñandú 😀?"
  assert qs.questions[0].keywords == ("ñ",)


def test_lists_become_tuples(write: Write) -> None:
  question = load_questions(
    write({"questions": [q(keywords=["a"], evidence=["b"], documents=["c.md"])]})
  ).questions[0]
  assert question.keywords == ("a",)
  assert question.evidence == ("b",)
  assert question.documents == ("c.md",)


def test_empty_lists_are_allowed(write: Write) -> None:
  question = load_questions(
    write({"questions": [q(keywords=[], evidence=[], documents=[])]})
  ).questions[0]
  assert question.matches("x") is None
  assert question.reachable("x") is None


def test_text_is_not_stripped(write: Write) -> None:
  # Non-blank is the only rule: the text is kept as written.
  assert (
    load_questions(write({"questions": [q(expected="  r  ")]})).questions[0].expected
    == "  r  "
  )


def test_utf8_file_with_non_ascii_and_escaped_form(write: Write) -> None:
  path = write('{"questions": [{"question": "\\u00bfQu\\u00e9?", "expected": "r"}]}')
  assert load_questions(path).questions[0].question == "¿Qué?"


# --- load_questions: file-level errors ----------------------------------------------


class TestFileErrors:
  def test_missing_file(self, tmp_path: Path) -> None:
    err = fails(tmp_path / "nope.json", None)
    assert "no existe" in err.message

  def test_directory(self, tmp_path: Path) -> None:
    fails(tmp_path, None)

  def test_invalid_json(self, write: Write) -> None:
    err = fails(write('{"questions": [}'), None)
    assert "JSON" in err.message

  def test_empty_file(self, write: Write) -> None:
    fails(write(""), None)

  def test_invalid_utf8(self, tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_bytes(b'{"questions": [{"question": "\xff", "expected": "r"}]}')
    fails(path, None)

  @pytest.mark.parametrize(
    ("bad", "key"),
    [
      ('{"questions": [{"question": "\\ud800", "expected": "r"}]}', "question"),
      ('{"questions": [{"question": "q", "expected": "\\udfff"}]}', "expected"),
      (
        '{"questions": [{"question": "q", "expected": "r", "keywords": ["\\ud800"]}]}',
        "keywords[0]",
      ),
      ('{"questions": [{"question": "q", "expected": "r", "id": "\\ud800"}]}', "id"),
      (
        '{"title": "\\ud800", "questions": [{"question": "q", "expected": "r"}]}',
        "title",
      ),
    ],
  )
  def test_lone_surrogate_is_a_questions_error(
    self, tmp_path: Path, bad: str, key: str
  ) -> None:
    path = tmp_path / "p.json"
    path.write_text(bad, encoding="utf-8")
    full_key = key if key == "title" else f"questions[0].{key}"
    err = fails(path, full_key)
    assert str(path) in str(err)

  def test_valid_surrogate_pair_is_accepted(self, tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(
      '{"questions": [{"question": "\\ud83d\\ude00", "expected": "r"}]}',
      encoding="utf-8",
    )
    assert load_questions(path).questions[0].question == "\U0001f600"

  def test_over_long_path(self, tmp_path: Path) -> None:
    with pytest.raises(QuestionsError):
      load_questions(tmp_path / ("a" * 5000))

  def test_nul_byte_in_path(self) -> None:
    with pytest.raises(QuestionsError):
      load_questions(Path("a\0b.json"))

  def test_bom_is_an_error_not_a_crash(self, tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_bytes(b'\xef\xbb\xbf{"questions": []}')
    with pytest.raises(QuestionsError):
      load_questions(path)

  @pytest.mark.parametrize("text", ["[]", "[{}]", '"x"', "3", "null", "true"])
  def test_root_must_be_an_object(self, write: Write, text: str) -> None:
    err = fails(write(text), None)
    assert "objeto" in err.message

  def test_nan_literal_does_not_crash(self, write: Write) -> None:
    # json.loads accepts NaN; it lands in a text field, which must reject it.
    fails(
      write('{"questions": [{"question": NaN, "expected": "r"}]}'),
      "questions[0].question",
    )

  def test_deeply_nested_json_does_not_crash(self, write: Write) -> None:
    with pytest.raises(QuestionsError):
      load_questions(write("[" * 100_000))

  def test_huge_integer_does_not_crash(self, write: Write) -> None:
    # json.loads raises a bare ValueError past 4300 digits (int conversion limit).
    with pytest.raises(QuestionsError):
      load_questions(write('{"title": 1' + "0" * 5000 + ', "questions": []}'))

  def test_huge_integer_in_a_question_does_not_crash(self, write: Write) -> None:
    with pytest.raises(QuestionsError):
      load_questions(
        write(
          '{"questions": [{"question": "a", "expected": "b", "id": '
          + "9" * 5000
          + "}]}"
        )
      )

  def test_unknown_top_level_key(self, write: Write) -> None:
    err = fails(write({"questions": [q()], "version": 1}), "version")
    assert "desconocida" in err.message

  def test_questions_required(self, write: Write) -> None:
    fails(write({"title": "T"}), "questions")

  def test_empty_object(self, write: Write) -> None:
    fails(write({}), "questions")

  @pytest.mark.parametrize("value", [None, {}, "x", 3, []])
  def test_questions_must_be_a_non_empty_list(
    self, write: Write, value: object
  ) -> None:
    fails(write({"questions": value}), "questions")

  @pytest.mark.parametrize("title", [None, 3, ["x"], True])
  def test_title_must_be_text(self, write: Write, title: object) -> None:
    fails(write({"title": title, "questions": [q()]}), "title")

  def test_empty_title_is_allowed(self, write: Write) -> None:
    assert load_questions(write({"title": "", "questions": [q()]})).title == ""


# --- load_questions: per-question errors ---------------------------------------------


class TestQuestionErrors:
  @pytest.mark.parametrize("item", [None, "x", 3, [], [1]])
  def test_item_must_be_an_object(self, write: Write, item: object) -> None:
    fails(write({"questions": [q(), item]}), "questions[1]")

  def test_key_is_zero_based(self, write: Write) -> None:
    # questions[2] is the third question, while default ids are 1-based.
    err = fails(
      write({"questions": [q(), q(), q(keywords="nope")]}), "questions[2].keywords"
    )
    assert "q03" not in err.message

  @pytest.mark.parametrize("field", ["question", "expected"])
  def test_required_fields(self, write: Write, field: str) -> None:
    item = q()
    del item[field]
    fails(write({"questions": [item]}), f"questions[0].{field}")

  @pytest.mark.parametrize("field", ["question", "expected"])
  @pytest.mark.parametrize("value", ["", "   ", "\n\t", None, 3, True, ["x"], {}])
  def test_required_fields_must_be_non_blank_text(
    self, write: Write, field: str, value: object
  ) -> None:
    fails(write({"questions": [q(**{field: value})]}), f"questions[0].{field}")

  def test_unknown_question_key(self, write: Write) -> None:
    err = fails(write({"questions": [q(), q(keyword=["x"])]}), "questions[1].keyword")
    assert "desconocida" in err.message

  def test_other_cased_key_is_unknown(self, write: Write) -> None:
    fails(write({"questions": [q(Refusal=True)]}), "questions[0].Refusal")

  @pytest.mark.parametrize("value", ["", "  ", None, 3, True, ["a"]])
  def test_bad_id_type_or_blank(self, write: Write, value: object) -> None:
    fails(write({"questions": [q(id=value)]}), "questions[0].id")

  @pytest.mark.parametrize(
    "value",
    [
      "has space",
      "a/b",
      "a.b",
      "ñ",
      "q#1",
      "a\n",
      "\na",
      "q 1",
      "a?b",
      "a&b",
      "x٠",
      "Ａ",
    ],
  )
  def test_id_pattern(self, write: Write, value: str) -> None:
    err = fails(write({"questions": [q(id=value)]}), "questions[0].id")
    assert "id" in err.message

  def test_id_with_trailing_newline_is_rejected(self, write: Write) -> None:
    # ``$`` would accept "a\n"; the id ends up in a URL anchor.
    fails(write({"questions": [q(id="a\n")]}), "questions[0].id")

  @pytest.mark.parametrize("field", ["keywords", "evidence", "documents"])
  @pytest.mark.parametrize("value", ["x", 3, None, {"a": 1}, True])
  def test_lists_must_be_lists(self, write: Write, field: str, value: object) -> None:
    fails(write({"questions": [q(**{field: value})]}), f"questions[0].{field}")

  @pytest.mark.parametrize("field", ["keywords", "evidence", "documents"])
  @pytest.mark.parametrize("item", ["", "  ", None, 3, ["x"], True])
  def test_list_items_must_be_non_blank_text(
    self, write: Write, field: str, item: object
  ) -> None:
    fails(
      write({"questions": [q(**{field: ["ok", item]})]}), f"questions[0].{field}[1]"
    )

  @pytest.mark.parametrize("value", [1, 0, "true", "false", None, [], "yes"])
  def test_refusal_must_be_bool(self, write: Write, value: object) -> None:
    fails(write({"questions": [q(refusal=value)]}), "questions[0].refusal")

  @pytest.mark.parametrize("value", [True, False])
  def test_refusal_bool(self, write: Write, value: bool) -> None:
    assert (
      load_questions(write({"questions": [q(refusal=value)]})).questions[0].refusal
      is value
    )

  def test_error_is_reported_on_the_offending_question_only(self, write: Write) -> None:
    fails(
      write({"questions": [q(), q(), q(), q(expected="")]}), "questions[3].expected"
    )


class TestDuplicateIds:
  def test_duplicate_explicit_ids_name_both_positions(self, write: Write) -> None:
    err = fails(
      write({"questions": [q(id="a"), q(id="b"), q(id="a")]}), "questions[2].id"
    )
    assert "questions[0]" in err.message
    assert "'a'" in err.message

  def test_explicit_id_colliding_with_a_later_default(self, write: Write) -> None:
    # The second question defaults to q02, which the first one already took.
    err = fails(write({"questions": [q(id="q02"), q()]}), "questions[1].id")
    assert "questions[0]" in err.message

  def test_default_colliding_with_a_later_explicit(self, write: Write) -> None:
    err = fails(write({"questions": [q(), q(id="q01")]}), "questions[1].id")
    assert "questions[0]" in err.message

  def test_ids_are_case_sensitive(self, write: Write) -> None:
    assert load_questions(write({"questions": [q(id="a"), q(id="A")]})).ids == (
      "a",
      "A",
    )


class TestDocuments:
  SOURCES = ["a.md", "dir/b.md"]

  def test_not_checked_without_sources(self, write: Write) -> None:
    assert load_questions(write({"questions": [q(documents=["whatever.md"])]})).ids == (
      "q01",
    )

  def test_known_documents_pass(self, write: Write) -> None:
    path = write({"questions": [q(documents=["a.md", "dir/b.md"])]})
    assert load_questions(path, self.SOURCES).ids == ("q01",)

  def test_unknown_documents_are_all_listed(self, write: Write) -> None:
    path = write({"questions": [q(), q(documents=["a.md", "x.md", "y.md"])]})
    err = fails(path, "questions[1].documents", self.SOURCES)
    assert "x.md" in err.message and "y.md" in err.message
    assert "a.md" not in err.message

  def test_empty_sources_reject_any_document(self, write: Write) -> None:
    fails(write({"questions": [q(documents=["a.md"])]}), "questions[0].documents", [])

  def test_empty_sources_accept_no_documents(self, write: Write) -> None:
    assert load_questions(write({"questions": [q()]}), []).ids == ("q01",)

  def test_match_is_exact(self, write: Write) -> None:
    for name in ("A.md", "./a.md", "a.md ", "dir\\b.md", "/a.md"):
      fails(
        write({"questions": [q(documents=[name])]}),
        "questions[0].documents",
        self.SOURCES,
      )

  def test_any_collection_type_is_accepted(self, write: Write) -> None:
    path = write({"questions": [q(documents=["a.md"])]})
    assert load_questions(path, frozenset(self.SOURCES)).ids == ("q01",)
    assert load_questions(path, tuple(self.SOURCES)).ids == ("q01",)
    assert load_questions(path, {"a.md": 1}).ids == ("q01",)

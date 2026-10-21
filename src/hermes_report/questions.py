"""Questions file: the JSON battery with expected answers and its rule checks."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
  "Question",
  "QuestionSet",
  "QuestionsError",
  "load_questions",
  "normalise",
]

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
# Everything reading or parsing user-given input can raise.
_INPUT_ERRORS = (
  OSError,
  ValueError,
  RecursionError,
  RuntimeError,
  OverflowError,
  UnicodeError,
)
_QUESTION_KEYS = frozenset(
  {"id", "question", "expected", "keywords", "evidence", "documents", "refusal"}
)


def normalise(text: str) -> str:
  """Lower-case, strip accents and collapse whitespace.

  Whitespace runs become one space, so a phrase matches across line breaks.
  """
  decomposed = unicodedata.normalize("NFD", text.lower())
  stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
  return " ".join(stripped.split())


def _contained(term: str, normalised_text: str) -> bool:
  """Whether ``term``, normalised, is in the text; an empty term never is."""
  needle = normalise(term)
  return bool(needle) and needle in normalised_text


class QuestionsError(Exception):
  """The questions file is invalid (message in Spanish).

  Attributes:
    path: The questions file.
    message: What is wrong.
    key: Where, like ``questions[2].keywords``; ``None`` for the whole file.
  """

  def __init__(self, path: Path, message: str, key: str | None = None) -> None:
    super().__init__(path, message, key)
    self.path = path
    self.message = message
    self.key = key

  def __str__(self) -> str:
    if self.key is None:
      return f"{self.path}: {self.message}"
    return f"{self.path}: {self.key}: {self.message}"


@dataclass(frozen=True)
class Question:
  """One question with its expected answer and rule data.

  Attributes:
    id: Unique identifier, also the ``--only`` value and the URL anchor.
    question: The user question.
    expected: The right answer, for the reader and the judge.
    keywords: Words that must all appear in a right answer.
    evidence: Corpus phrases that must all be in the context for the answer
      to be reachable.
    documents: Corpus files (relative to the corpus folder) holding the answer.
    refusal: Whether the right outcome is to say the sources are not enough.
  """

  id: str
  question: str
  expected: str
  keywords: tuple[str, ...] = ()
  evidence: tuple[str, ...] = ()
  documents: tuple[str, ...] = ()
  refusal: bool = False

  def matches(self, answer: str) -> bool | None:
    """Whether every keyword is in ``answer``; ``None`` without keywords."""
    if not self.keywords:
      return None
    norm = normalise(answer)
    return all(_contained(k, norm) for k in self.keywords)

  def reachable(self, text: str) -> bool | None:
    """Whether every evidence phrase is in ``text``; ``None`` without evidence."""
    if not self.evidence:
      return None
    norm = normalise(text)
    return all(_contained(e, norm) for e in self.evidence)

  def to_dict(self) -> dict[str, Any]:
    """Every field as plain data, tuples as lists."""
    return {
      "id": self.id,
      "question": self.question,
      "expected": self.expected,
      "keywords": list(self.keywords),
      "evidence": list(self.evidence),
      "documents": list(self.documents),
      "refusal": self.refusal,
    }


@dataclass(frozen=True)
class QuestionSet:
  """A loaded questions file.

  Attributes:
    path: The questions file.
    title: Title given in the file, if any.
    questions: The questions, in file order.
  """

  path: Path
  title: str | None
  questions: tuple[Question, ...]

  @property
  def ids(self) -> tuple[str, ...]:
    """The id of every question, in order."""
    return tuple(q.id for q in self.questions)

  def get(self, question_id: str) -> Question:
    """The question with ``question_id``.

    Raises:
      KeyError: if no question has that id.
    """
    for question in self.questions:
      if question.id == question_id:
        return question
    raise KeyError(question_id)


def _text(path: Path, key: str, value: object) -> str:
  if not isinstance(value, str):
    raise QuestionsError(path, "debe ser un texto", key)
  if not value.strip():
    raise QuestionsError(path, "no puede estar vacío", key)
  _require_encodable(path, key, value)
  return value


def _require_encodable(path: Path, key: str, value: str) -> None:
  """Reject lone surrogates (valid in JSON), which cannot be written as UTF-8."""
  try:
    value.encode("utf-8")
  except UnicodeEncodeError:
    raise QuestionsError(
      path, "contiene caracteres que no son UTF-8 válido", key
    ) from None


def _text_list(path: Path, key: str, value: object) -> tuple[str, ...]:
  if not isinstance(value, list):
    raise QuestionsError(path, "debe ser una lista de textos", key)
  return tuple(_text(path, f"{key}[{i}]", item) for i, item in enumerate(value))


def _term_list(path: Path, key: str, value: object) -> tuple[str, ...]:
  """Keywords or evidence: texts that are not empty once normalised."""
  terms = _text_list(path, key, value)
  for i, term in enumerate(terms):
    if not normalise(term):
      raise QuestionsError(
        path, "no puede estar vacío una vez normalizado (solo acentos)", f"{key}[{i}]"
      )
  return terms


def load_questions(
  path: Path, corpus_sources: Collection[str] | None = None
) -> QuestionSet:
  """Load and validate the questions file.

  Args:
    path: The JSON file.
    corpus_sources: When given, every ``documents`` entry must be one of them.

  Raises:
    QuestionsError: if the file is missing, is not valid JSON or breaks the
      format; the error names the file and the key.
  """
  try:
    is_file = path.is_file()
  except _INPUT_ERRORS as exc:
    raise QuestionsError(path, f"no se puede leer: {exc}") from None
  if not is_file:
    raise QuestionsError(path, "el archivo no existe")
  try:
    text = path.read_bytes().decode("utf-8")
  except _INPUT_ERRORS as exc:
    raise QuestionsError(path, f"no se puede leer: {exc}") from None
  try:
    data: object = json.loads(text)
  except json.JSONDecodeError as exc:
    raise QuestionsError(path, f"JSON inválido: {exc}") from exc
  except RecursionError:
    raise QuestionsError(path, "JSON inválido: anidamiento excesivo") from None
  except _INPUT_ERRORS as exc:
    raise QuestionsError(path, f"JSON inválido: {exc}") from None
  if not isinstance(data, dict):
    raise QuestionsError(path, "debe ser un objeto JSON")
  for key in data:
    if key not in ("title", "questions"):
      raise QuestionsError(path, "clave desconocida", str(key))
  title: str | None = None
  if "title" in data:
    if not isinstance(data["title"], str):
      raise QuestionsError(path, "debe ser un texto", "title")
    title = data["title"]
    _require_encodable(path, "title", title)
  if "questions" not in data:
    raise QuestionsError(path, "falta la clave obligatoria", "questions")
  raw_questions = data["questions"]
  if not isinstance(raw_questions, list):
    raise QuestionsError(path, "debe ser una lista", "questions")
  if not raw_questions:
    raise QuestionsError(path, "no puede estar vacía", "questions")
  known = None if corpus_sources is None else set(corpus_sources)
  questions: list[Question] = []
  first_seen: dict[str, int] = {}
  for index, raw in enumerate(raw_questions):
    base = f"questions[{index}]"
    if not isinstance(raw, dict):
      raise QuestionsError(path, "debe ser un objeto", base)
    for key in raw:
      if key not in _QUESTION_KEYS:
        raise QuestionsError(path, "clave desconocida", f"{base}.{key}")
    for key in ("question", "expected"):
      if key not in raw:
        raise QuestionsError(path, "falta la clave obligatoria", f"{base}.{key}")
    if "id" in raw:
      question_id = _text(path, f"{base}.id", raw["id"])
      if _ID_PATTERN.fullmatch(question_id) is None:
        raise QuestionsError(
          path,
          f"id inválido {question_id!r}: solo letras, dígitos, '_' y '-'",
          f"{base}.id",
        )
    else:
      question_id = f"q{index + 1:02d}"
    if question_id in first_seen:
      raise QuestionsError(
        path,
        f"id duplicado {question_id!r}: ya usado en questions[{first_seen[question_id]}]",
        f"{base}.id",
      )
    first_seen[question_id] = index
    documents = (
      _text_list(path, f"{base}.documents", raw["documents"])
      if "documents" in raw
      else ()
    )
    if known is not None:
      unknown = [d for d in documents if d not in known]
      if unknown:
        raise QuestionsError(
          path,
          "documentos que no están en el corpus: " + ", ".join(unknown),
          f"{base}.documents",
        )
    refusal = raw.get("refusal", False)
    if not isinstance(refusal, bool):
      raise QuestionsError(path, "debe ser verdadero o falso", f"{base}.refusal")
    questions.append(
      Question(
        id=question_id,
        question=_text(path, f"{base}.question", raw["question"]),
        expected=_text(path, f"{base}.expected", raw["expected"]),
        keywords=(
          _term_list(path, f"{base}.keywords", raw["keywords"])
          if "keywords" in raw
          else ()
        ),
        evidence=(
          _term_list(path, f"{base}.evidence", raw["evidence"])
          if "evidence" in raw
          else ()
        ),
        documents=documents,
        refusal=refusal,
      )
    )
  return QuestionSet(path=path, title=title, questions=tuple(questions))

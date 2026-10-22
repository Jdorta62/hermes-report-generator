"""Run record: the schema of ``run.json``, its converters and its file I/O."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, NoReturn, TypedDict, cast

from hermes_core.enrichment import EnrichedContext
from hermes_core.generation import GenerationResult, PromptMessage
from hermes_core.retrieval import RetrievedChunk, SearchedQuery
from hermes_enrich import ChunkRange

__all__ = [
  "SCHEMA_VERSION",
  "ChunkRangeRecord",
  "ChunkRefRecord",
  "CollectionRecord",
  "ContextRecord",
  "CorpusRecord",
  "DocumentRecord",
  "EnrichRecord",
  "EnvironmentRecord",
  "GenerationRecord",
  "IngestDocumentRecord",
  "IngestRecord",
  "JudgeCallRecord",
  "JudgeRecord",
  "LLMCallRecord",
  "MessageRecord",
  "MetricResultRecord",
  "PieceRecord",
  "PointRecord",
  "QuestionRecord",
  "RecordError",
  "RetrievalRecord",
  "RetrievedChunkRecord",
  "RuleRecord",
  "RunRecord",
  "SearchedQueryRecord",
  "StepRecord",
  "StoreReadRecord",
  "VectorSummary",
  "chunk_range_record",
  "context_record",
  "generation_result_record",
  "json_safe",
  "message_record",
  "point_record",
  "read_record",
  "retrieved_chunk_record",
  "searched_query_record",
  "vector_summary",
  "write_record",
]

SCHEMA_VERSION: Final = 1

_HEAD_LENGTH = 8
# Everything reading, parsing or writing a user-given path or file can raise.
_INPUT_ERRORS = (
  OSError,
  ValueError,
  RecursionError,
  RuntimeError,
  OverflowError,
  UnicodeError,
)


class RecordError(Exception):
  """The run record cannot be read or written (message in Spanish).

  Attributes:
    path: The record file.
    message: What is wrong.
  """

  def __init__(self, path: Path, message: str) -> None:
    super().__init__(path, message)
    self.path = path
    self.message = message

  def __str__(self) -> str:
    return f"{self.path}: {self.message}"


class VectorSummary(TypedDict):
  """A vector reduced to its dimension, Euclidean norm and first values."""

  dimension: int
  norm: float
  head: list[float]


class PointRecord(TypedDict):
  """One Qdrant point: its id as a string, payload and vector summary."""

  id: str
  payload: dict[str, Any]
  vector: VectorSummary | None


class DocumentRecord(TypedDict):
  """One corpus document with its chunk count."""

  source: str
  sha256: str
  chars: int
  chunks: int


class CorpusRecord(TypedDict):
  """The corpus folder and its documents."""

  dir: str
  glob: str
  fingerprint: str
  documents: list[DocumentRecord]


class IngestDocumentRecord(TypedDict):
  """The ingestion of one document."""

  source: str
  chunks: int
  seconds: float
  cached: bool


class IngestRecord(TypedDict):
  """The ingestion phase."""

  seconds: float
  chunker: dict[str, Any]
  embedding: dict[str, Any]
  ingester_version: str
  documents: list[IngestDocumentRecord]


class CollectionRecord(TypedDict):
  """The Qdrant collection the run used."""

  name: str
  url: str
  reloaded: bool
  points: int
  vector_size: int
  distance: str
  indexes: list[str]


class EnvironmentRecord(TypedDict):
  """Where the run happened: interpreter, packages and services."""

  python: str
  packages: dict[str, str]
  ollama_host: str
  qdrant_url: str


class RetrievedChunkRecord(TypedDict):
  """A retrieved chunk; ``metadata`` is the payload without its ``text`` key."""

  chunk_id: str
  text: str
  score: float
  vector_score: float | None
  collection_name: str
  metadata: dict[str, Any]


class SearchedQueryRecord(TypedDict):
  """A query that was embedded and searched."""

  text: str
  chain: list[str]


class RetrievalRecord(TypedDict):
  """The retrieval phase of one question."""

  seconds: float
  chunks: list[RetrievedChunkRecord]
  queries: list[SearchedQueryRecord]


class ChunkRefRecord(TypedDict):
  """A chunk inside a context piece, with its provenance and no text."""

  chunk_id: str
  collection_name: str
  source: str | None
  doc_id: str | None
  chunk_index: int | None
  score: float | None
  origin: list[str]


class PieceRecord(TypedDict):
  """A contiguous block of context text."""

  text: str
  transforms: list[str]
  chunks: list[ChunkRefRecord]


class ContextRecord(TypedDict):
  """An enriched context."""

  pieces: list[PieceRecord]
  enrichments: list[str]
  dropped_chunk_ids: list[str]
  query_id: str | None


class ChunkRangeRecord(TypedDict):
  """An inclusive range of chunk indexes of one document."""

  collection_name: str
  doc_id: str
  first: int
  last: int


class StoreReadRecord(TypedDict):
  """One read of the vector store made by an enrichment step."""

  ranges: list[ChunkRangeRecord]
  returned: list[str]
  seconds: float


class StepRecord(TypedDict):
  """One enrichment step with the context before and after it."""

  name: str
  phase: str
  seconds: float
  before: ContextRecord
  after: ContextRecord
  reads: list[StoreReadRecord]


class EnrichRecord(TypedDict):
  """The enrichment phase of one question."""

  seconds: float
  budget: dict[str, Any]
  steps: list[StepRecord]
  context: ContextRecord | None


class MessageRecord(TypedDict):
  """One chat message."""

  role: str
  content: str


class LLMCallRecord(TypedDict):
  """One call to a language model."""

  messages: list[MessageRecord]
  params: dict[str, Any]
  reply: dict[str, Any] | None
  error: str | None
  seconds: float


class GenerationRecord(TypedDict):
  """The generation phase of one question."""

  seconds: float
  llm_calls: list[LLMCallRecord]
  result: dict[str, Any] | None


class RuleRecord(TypedDict):
  """The rule-based verdict of one answer."""

  verdict: str
  in_retrieval: bool | None
  in_context: bool | None
  in_cited: bool | None
  answer_ok: bool | None
  retrieved_rank: int | None
  expected_docs_retrieved: bool | None


class JudgeRecord(TypedDict):
  """The LLM judge verdict of one answer."""

  verdict: str | None
  reason: str | None
  model: str
  seconds: float
  messages: list[MessageRecord]
  raw: str | None
  error: str | None


class JudgeCallRecord(TypedDict):
  """One judge call made by a metric operation."""

  purpose: str
  messages: list[MessageRecord]
  raw: str | None
  error: str | None
  seconds: float


class MetricResultRecord(TypedDict):
  """The outcome of one metric family for one question."""

  name: str
  status: str
  scores: dict[str, float | None]
  details: dict[str, Any]
  calls: list[JudgeCallRecord]
  error: str | None
  seconds: float


class QuestionRecord(TypedDict):
  """Everything that happened to one question."""

  spec: dict[str, Any]
  query_id: str
  seconds: float
  error: str | None
  retrieval: RetrievalRecord | None
  enrich: EnrichRecord | None
  generation: GenerationRecord | None
  rules: RuleRecord | None
  judge: JudgeRecord | None


class RunRecord(TypedDict):
  """The whole run: the content of ``run.json``."""

  schema: int
  title: str
  description: str
  started_at: str
  finished_at: str
  seconds: float
  config: dict[str, Any]
  environment: EnvironmentRecord
  corpus: CorpusRecord
  ingest: IngestRecord
  collection: CollectionRecord
  points: list[PointRecord]
  questions: list[QuestionRecord]


# --- converters ---------------------------------------------------------------


def json_safe(value: Any) -> Any:
  """Copy of ``value`` with every tuple turned into a list, recursively."""
  if isinstance(value, dict):
    return {key: json_safe(item) for key, item in value.items()}
  if isinstance(value, list | tuple):
    return [json_safe(item) for item in value]
  return value


def vector_summary(vector: Sequence[float]) -> VectorSummary:
  """Summarise a vector as its dimension, Euclidean norm and first 8 values."""
  values = [float(item) for item in vector]
  return {
    "dimension": len(values),
    "norm": math.hypot(*values),
    "head": values[:_HEAD_LENGTH],
  }


def point_record(
  point_id: str | int,
  payload: Mapping[str, Any],
  vector: Sequence[float] | None,
) -> PointRecord:
  """Record a Qdrant point with its id as a string and a summarised vector."""
  return {
    "id": str(point_id),
    "payload": json_safe(dict(payload)),
    "vector": None if vector is None else vector_summary(vector),
  }


def retrieved_chunk_record(chunk: RetrievedChunk) -> RetrievedChunkRecord:
  """Record a retrieved chunk; its metadata loses any ``text`` key."""
  return {
    "chunk_id": chunk.chunk_id,
    "text": chunk.text,
    "score": chunk.score,
    "vector_score": chunk.vector_score,
    "collection_name": chunk.collection_name,
    "metadata": json_safe(
      {key: value for key, value in chunk.metadata.items() if key != "text"}
    ),
  }


def searched_query_record(query: SearchedQuery) -> SearchedQueryRecord:
  """Record a searched query with its transform chain as a list."""
  return {"text": query.text, "chain": list(query.chain)}


def context_record(context: EnrichedContext) -> ContextRecord:
  """Record an enriched context: pieces with their chunks, without chunk text."""
  pieces: list[PieceRecord] = []
  for piece in context.pieces:
    chunks: list[ChunkRefRecord] = []
    for chunk in piece.chunks:
      source = chunk.metadata.get("source")
      position = chunk.position
      chunks.append(
        {
          "chunk_id": chunk.chunk_id,
          "collection_name": chunk.collection_name,
          "source": source if isinstance(source, str) else None,
          "doc_id": None if position is None else position.doc_id,
          "chunk_index": None if position is None else position.chunk_index,
          "score": chunk.score,
          "origin": list(chunk.origin),
        }
      )
    pieces.append(
      {"text": piece.text, "transforms": list(piece.transforms), "chunks": chunks}
    )
  return {
    "pieces": pieces,
    "enrichments": list(context.enrichments),
    "dropped_chunk_ids": list(context.dropped_chunk_ids),
    "query_id": context.query_id,
  }


def chunk_range_record(chunk_range: ChunkRange) -> ChunkRangeRecord:
  """Record a chunk range."""
  return {
    "collection_name": chunk_range.collection_name,
    "doc_id": chunk_range.doc_id,
    "first": chunk_range.first,
    "last": chunk_range.last,
  }


def message_record(message: PromptMessage) -> MessageRecord:
  """Record a prompt message."""
  return {"role": message.role, "content": message.content}


def generation_result_record(result: GenerationResult) -> dict[str, Any]:
  """Record a generation result as plain data that equals its JSON round trip."""
  data: dict[str, Any] = json_safe(asdict(result))
  return data


# --- file I/O -----------------------------------------------------------------


def write_record(record: RunRecord, path: Path) -> None:
  """Write the record as UTF-8 JSON, creating the parent folder.

  Raises:
    RecordError: if the record is not JSON-serialisable (``NaN`` and infinity
      included) or the file cannot be written; the error names the file.

  Effects: io
  """
  try:
    data = json.dumps(record, ensure_ascii=False, indent=1, allow_nan=False).encode(
      "utf-8"
    )
  except _INPUT_ERRORS + (TypeError,) as exc:
    raise RecordError(path, f"el registro no se puede serializar: {exc}") from None
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
  except _INPUT_ERRORS as exc:
    raise RecordError(path, f"no se puede escribir: {exc}") from None


def _reject_constant(name: str) -> NoReturn:
  raise ValueError(f"valor no permitido {name}")


def read_record(path: Path) -> RunRecord:
  """Read a run record; the object loaded is cast, not deeply validated.

  Raises:
    RecordError: if the file is missing or unreadable, is not valid JSON or
      not an object, or its ``schema`` is missing or differs from
      ``SCHEMA_VERSION``; the error names the file.

  Effects: io
  """
  try:
    text = path.read_bytes().decode("utf-8")
  except FileNotFoundError:
    raise RecordError(path, "el archivo no existe") from None
  except _INPUT_ERRORS as exc:
    raise RecordError(path, f"no se puede leer: {exc}") from None
  try:
    data: object = json.loads(text, parse_constant=_reject_constant)
  except RecursionError:
    raise RecordError(path, "JSON inválido: anidamiento excesivo") from None
  except _INPUT_ERRORS as exc:
    raise RecordError(path, f"JSON inválido: {exc}") from None
  if not isinstance(data, dict):
    raise RecordError(path, "el contenido debe ser un objeto JSON")
  if "schema" not in data:
    raise RecordError(path, "falta la clave obligatoria 'schema'")
  schema = data["schema"]
  if isinstance(schema, bool) or not isinstance(schema, int):
    raise RecordError(path, "'schema' debe ser un número entero")
  if schema != SCHEMA_VERSION:
    raise RecordError(
      path,
      f"versión de esquema {schema} no soportada; se esperaba {SCHEMA_VERSION}",
    )
  return cast("RunRecord", data)

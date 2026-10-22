"""Ingestion of a corpus into Qdrant points, with a per-document cache on disk."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hermes_core import QdrantPayload, QdrantPoint
from hermes_core.protocols import Embedder
from hermes_ingest_md import MarkdownIngester, MarkdownInput
from hermes_ingest_md.chunking import MarkdownStructuralChunker

from hermes_report.config import EmbeddingConfig, IngestConfig
from hermes_report.corpus import Corpus, CorpusDocument

__all__ = ["IngestError", "IngestedDocument", "Ingestion", "ingest_corpus"]

_INGESTER_PACKAGE = "hermes-ingest-md"


class IngestError(Exception):
  """A document could not be ingested or cached (message in Spanish)."""


@dataclass(frozen=True)
class IngestedDocument:
  """The points of one corpus document.

  Attributes:
    source: The ``source`` of the document in the corpus.
    points: Its Qdrant points, in chunk order.
    seconds: Time spent ingesting it, or reading its cache file.
    cached: Whether the points came from the cache.
  """

  source: str
  points: tuple[QdrantPoint, ...]
  seconds: float
  cached: bool


@dataclass(frozen=True)
class Ingestion:
  """The ingested corpus and how it was produced.

  Attributes:
    documents: One entry per corpus document, in corpus order.
    seconds: Total time of the ingestion.
    chunker: Chunker arguments: ``name``, ``max_chunk_size``,
      ``hard_boundary_levels`` (a list) and ``strip_frontmatter``.
    embedding: ``provider``, ``model`` and ``dimension``; the dimension is the
      length of the first vector, or ``None`` without points.
    ingester_version: Installed version of ``hermes-ingest-md``.
    key: Hex sha256 of the corpus fingerprint and the ingestion setup.
  """

  documents: tuple[IngestedDocument, ...]
  seconds: float
  chunker: dict[str, Any]
  embedding: dict[str, Any]
  ingester_version: str
  key: str

  @property
  def points(self) -> tuple[QdrantPoint, ...]:
    """Every point of every document, in document order."""
    return tuple(point for doc in self.documents for point in doc.points)


def ingest_corpus(
  corpus: Corpus,
  ingest: IngestConfig,
  embedding: EmbeddingConfig,
  embedder: Embedder,
  cache_dir: Path,
  force: bool = False,
  progress: Callable[[str], None] | None = None,
) -> Ingestion:
  """Ingest every document of ``corpus``, reusing the cached points that still apply.

  A document is read from ``cache_dir`` while its content, the chunker
  arguments, the embedding provider and model and the ingester version stay
  the same; ``force`` ignores the cache but rewrites it. An unreadable cache
  file counts as a miss.

  Effects: io

  Raises:
    IngestError: if a document cannot be read, chunked or embedded, the cache
      cannot be written, or two documents yield the same point id.
  """
  started = time.perf_counter()
  version = importlib.metadata.version(_INGESTER_PACKAGE)
  chunker_args: dict[str, Any] = {
    "max_chunk_size": ingest.max_chunk_size,
    "hard_boundary_levels": list(ingest.hard_boundary_levels),
    "strip_frontmatter": ingest.strip_frontmatter,
  }
  chunker = MarkdownStructuralChunker(**chunker_args)
  chunker_info: dict[str, Any] = {"name": chunker.name, **chunker_args}
  ingester = MarkdownIngester(embedder, chunker=chunker)
  owners: dict[str, str] = {}
  documents: list[IngestedDocument] = []
  for doc in corpus.documents:
    doc_started = time.perf_counter()
    cache_file = (
      cache_dir / f"{_document_key(doc, chunker_info, embedding, version)}.jsonl"
    )
    points = None if force else _read_cache(cache_file)
    cached = points is not None
    if points is None:
      points = _ingest_document(ingester, doc)
      _write_cache(cache_file, points, doc.source)
    for point in points:
      other = owners.setdefault(point.id, doc.source)
      if other != doc.source:
        raise IngestError(
          f"{other} y {doc.source}: producen el mismo punto ({point.id}); "
          "el contenido está repetido en dos rutas"
        )
    seconds = time.perf_counter() - doc_started
    documents.append(IngestedDocument(doc.source, tuple(points), seconds, cached))
    if progress is not None:
      origin = "caché" if cached else "ingerido"
      noun = "chunk" if len(points) == 1 else "chunks"
      progress(f"{doc.source}: {len(points)} {noun} ({origin})")
  first = next((p for d in documents for p in d.points), None)
  return Ingestion(
    documents=tuple(documents),
    seconds=time.perf_counter() - started,
    chunker=chunker_info,
    embedding={
      "provider": embedding.provider,
      "model": embedding.model,
      "dimension": None if first is None else len(first.vector),
    },
    ingester_version=version,
    key=_digest(
      {
        "corpus": corpus.fingerprint,
        "chunker": chunker_info,
        "embedding": {"provider": embedding.provider, "model": embedding.model},
        "ingester_version": version,
      }
    ),
  )


def _digest(data: dict[str, Any]) -> str:
  return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _document_key(
  doc: CorpusDocument,
  chunker: dict[str, Any],
  embedding: EmbeddingConfig,
  version: str,
) -> str:
  return _digest(
    {
      "source": doc.source,
      "sha256": doc.sha256,
      "chunker": chunker,
      "provider": embedding.provider,
      "model": embedding.model,
      "ingester_version": version,
    }
  )


def _ingest_document(
  ingester: MarkdownIngester, doc: CorpusDocument
) -> list[QdrantPoint]:
  try:
    result = ingester.ingest(MarkdownInput(content=doc.read_bytes()), doc.source)
  except Exception as exc:
    raise IngestError(f"{doc.source}: no se pudo ingerir: {exc}") from exc
  return list(result.points)


def _read_cache(path: Path) -> list[QdrantPoint] | None:
  """The cached points, or ``None`` when the file is missing or malformed."""
  try:
    points: list[QdrantPoint] = []
    with path.open(encoding="utf-8") as lines:
      for line in lines:
        if not line.strip():
          continue
        item = json.loads(line)
        point_id, vector, payload = item["id"], item["vector"], item["payload"]
        if (
          not isinstance(point_id, str)
          or not isinstance(vector, list)
          or not all(
            isinstance(x, int | float) and not isinstance(x, bool) for x in vector
          )
          or not isinstance(payload, dict)
        ):
          return None
        points.append(
          QdrantPoint(
            id=point_id,
            vector=[float(x) for x in vector],
            payload=cast(QdrantPayload, payload),
          )
        )
  except (OSError, ValueError, KeyError, TypeError):
    return None
  return points


def _write_cache(path: Path, points: list[QdrantPoint], source: str) -> None:
  """Write the cache file atomically: a temp file in the folder, then a rename."""
  temp: str | None = None
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
      temp = handle.name
      for point in points:
        handle.write(
          json.dumps(
            {"id": point.id, "vector": point.vector, "payload": point.payload},
            ensure_ascii=False,
          )
          + "\n"
        )
    os.replace(temp, path)
  except (OSError, TypeError, ValueError) as exc:
    if temp is not None:
      Path(temp).unlink(missing_ok=True)
    raise IngestError(f"{source}: no se pudo escribir la caché: {exc}") from exc

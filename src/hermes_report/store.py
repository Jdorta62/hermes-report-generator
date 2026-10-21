"""The Qdrant collection of a run: create or reuse it, and read it back as stored."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from hermes_core import QdrantPoint
from qdrant_client import QdrantClient, models

from hermes_report.questions import normalise
from hermes_report.record import PointRecord, point_record

__all__ = [
  "PREFIX",
  "CollectionInfo",
  "StoreError",
  "collection_name",
  "ensure_collection",
  "make_client",
  "read_points",
  "slug",
]

PREFIX: Final = "hermes_report_"

_BATCH = 256
_SLUG_MAX = 40
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")
_LOCAL_INDEXES = ("chunk_index", "doc_id")


class StoreError(Exception):
  """Qdrant cannot be used as asked (message in Spanish)."""


@dataclass(frozen=True)
class CollectionInfo:
  """The Qdrant collection of a run.

  Attributes:
    name: Collection name.
    url: Qdrant URL it lives in.
    reloaded: Whether it was (re)created and loaded, instead of reused.
    points: Exact number of points it holds.
    vector_size: Dimension of its vectors.
    distance: Distance metric, as Qdrant names it.
    indexes: Names of the indexed payload fields, sorted.
  """

  name: str
  url: str
  reloaded: bool
  points: int
  vector_size: int
  distance: str
  indexes: tuple[str, ...]


def slug(text: str) -> str:
  """Reduce ``text`` to at most 40 characters of ``[a-z0-9_]``; ``run`` if empty."""
  words = _NOT_ALNUM.sub("_", normalise(text)).strip("_")
  return words[:_SLUG_MAX].strip("_") or "run"


def collection_name(title: str, ingestion_key: str, explicit: str = "") -> str:
  """Return ``explicit``, else ``hermes_report_<slug(title)>_<first 8 of key>``."""
  return explicit or f"{PREFIX}{slug(title)}_{ingestion_key[:8]}"


def make_client(url: str) -> QdrantClient:
  """Return an in-process client for ``":memory:"``, else a client for ``url``."""
  if url == ":memory:":
    return QdrantClient(":memory:")
  return QdrantClient(url=url, timeout=60)


def ensure_collection(
  client: QdrantClient,
  name: str,
  points: Sequence[QdrantPoint],
  url: str,
  force: bool = False,
) -> CollectionInfo:
  """Reuse the collection ``name`` if it holds exactly ``points``, else rebuild it.

  Only collections whose name starts with ``PREFIX`` are touched. The rebuilt
  collection uses cosine distance and indexes ``doc_id`` and ``chunk_index``.

  Effects: net, mut

  Raises:
    StoreError: if the name lacks the prefix, there are no points, or Qdrant
      fails.
  """
  if not name.startswith(PREFIX):
    raise StoreError(
      f"la colección {name!r} no empieza por {PREFIX!r}: solo se tocan "
      "colecciones propias de la herramienta"
    )
  if not points:
    raise StoreError("el corpus no produjo ningún chunk")
  try:
    reloaded = force or not _holds_exactly(client, name, points)
    if reloaded:
      _rebuild(client, name, points)
    return _describe(client, name, url, reloaded)
  except StoreError:
    raise
  except Exception as exc:
    raise StoreError(f"Qdrant no responde en {url} ({exc})") from exc


def read_points(client: QdrantClient, name: str) -> list[PointRecord]:
  """Read every point of ``name`` as stored, ordered by source and chunk index.

  Effects: net

  Raises:
    StoreError: if Qdrant fails.
  """
  records: list[Any] = []
  try:
    offset: Any = None
    while True:
      page, offset = client.scroll(
        name, limit=_BATCH, offset=offset, with_payload=True, with_vectors=True
      )
      records.extend(page)
      if offset is None:
        break
  except Exception as exc:
    raise StoreError(f"no se pudo leer la colección {name} de Qdrant ({exc})") from exc
  records.sort(key=_order)
  return [
    point_record(
      record.id,
      record.payload or {},
      record.vector if isinstance(record.vector, list) else None,
    )
    for record in records
  ]


def _order(record: Any) -> tuple[str, int, str]:
  payload = record.payload or {}
  source = payload.get("source")
  index = payload.get("chunk_index")
  return (
    source if isinstance(source, str) else "",
    index if isinstance(index, int) and not isinstance(index, bool) else -1,
    str(record.id),
  )


def _holds_exactly(
  client: QdrantClient, name: str, points: Sequence[QdrantPoint]
) -> bool:
  if not client.collection_exists(name):
    return False
  if client.count(name, exact=True).count != len(points):
    return False
  ids = [point.id for point in points]
  for start in range(0, len(ids), _BATCH):
    batch = ids[start : start + _BATCH]
    found = client.retrieve(name, batch, with_payload=False, with_vectors=False)
    if {str(record.id) for record in found} != set(batch):
      return False
  return True


def _rebuild(client: QdrantClient, name: str, points: Sequence[QdrantPoint]) -> None:
  if client.collection_exists(name):
    client.delete_collection(name)
  client.create_collection(
    name,
    vectors_config=models.VectorParams(
      size=len(points[0].vector), distance=models.Distance.COSINE
    ),
  )
  client.create_payload_index(name, "doc_id", models.PayloadSchemaType.KEYWORD)
  client.create_payload_index(name, "chunk_index", models.PayloadSchemaType.INTEGER)
  for start in range(0, len(points), _BATCH):
    client.upsert(
      name,
      points=[
        models.PointStruct(id=p.id, vector=p.vector, payload=dict(p.payload))
        for p in points[start : start + _BATCH]
      ],
      wait=True,
    )


def _describe(
  client: QdrantClient, name: str, url: str, reloaded: bool
) -> CollectionInfo:
  info = client.get_collection(name)
  vectors = info.config.params.vectors
  if not isinstance(vectors, models.VectorParams):
    raise StoreError(f"la colección {name} usa vectores con nombre, no soportados")
  indexes = tuple(sorted(info.payload_schema or {})) or _LOCAL_INDEXES
  return CollectionInfo(
    name=name,
    url=url,
    reloaded=reloaded,
    points=client.count(name, exact=True).count,
    vector_size=vectors.size,
    distance=str(getattr(vectors.distance, "value", vectors.distance)),
    indexes=indexes,
  )

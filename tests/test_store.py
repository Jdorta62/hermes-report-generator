"""Tests for ``hermes_report.store``: naming, the collection life cycle and ``read_points``.

Every test runs on ``QdrantClient(":memory:")`` or on a fake client; none reaches
a real Qdrant.
"""

from __future__ import annotations

import dataclasses
import uuid
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from hermes_core import QdrantPoint
from qdrant_client import QdrantClient, models

from hermes_report import store
from hermes_report.config import EmbeddingConfig, IngestConfig
from hermes_report.corpus import load_corpus
from hermes_report.ingest import ingest_corpus
from hermes_report.providers import HashEmbedder
from hermes_report.store import (
  PREFIX,
  CollectionInfo,
  StoreError,
  collection_name,
  ensure_collection,
  make_client,
  read_points,
  slug,
)

NAME = "hermes_report_test_12345678"
URL = ":memory:"


@pytest.fixture(autouse=True)
def _local_index_warning() -> Iterator[None]:
  """Local mode warns that payload indexes have no effect; that is expected."""
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Payload indexes have no effect")
    warnings.filterwarnings("ignore", message="Failed to obtain server version")
    yield


def _point(
  n: int, *, source: str = "a.md", index: int | None = None, dim: int = 4
) -> QdrantPoint:
  return QdrantPoint(
    id=str(uuid.UUID(int=n)),
    vector=[float(n + 1)] + [0.5] * (dim - 1),
    payload={  # type: ignore[typeddict-item]
      "text": f"chunk {n}",
      "source": source,
      "doc_id": f"doc-{source}",
      "chunk_index": n if index is None else index,
    },
  )


def _points(count: int, dim: int = 4) -> list[QdrantPoint]:
  return [_point(n, dim=dim) for n in range(count)]


@pytest.fixture
def client() -> QdrantClient:
  return QdrantClient(":memory:")


# --------------------------------------------------------------------------- slug


@pytest.mark.parametrize(
  ("text", "expected"),
  [
    ("Mi Informe", "mi_informe"),
    ("Don Quijote: ¿qué?", "don_quijote_que"),
    ("ÁÉÍÓÚ ñandú", "aeiou_nandu"),
    ("  --Hello   World--  ", "hello_world"),
    ("a.b/c\\d", "a_b_c_d"),
    ("", "run"),
    ("   ", "run"),
    ("¿¡!?", "run"),
    ("日本語", "run"),
    ("___", "run"),
    ("x", "x"),
    ("UPPER_lower_123", "upper_lower_123"),
    ("a" * 40, "a" * 40),
    ("a" * 41, "a" * 40),
    ("a" * 500, "a" * 40),
  ],
)
def test_slug(text: str, expected: str) -> None:
  assert slug(text) == expected


def test_slug_never_ends_or_starts_with_underscore_after_the_cut() -> None:
  text = "a" * 39 + " bbbb"
  assert slug(text) == "a" * 39
  assert len(slug("word " * 30)) <= 40
  assert not slug("word " * 30).endswith("_")


@pytest.mark.parametrize(
  "text", ["Ünï cödé ✓ título", "a  b\tc\nd", "%%%a%%%", "x" * 80]
)
def test_slug_only_uses_the_allowed_alphabet(text: str) -> None:
  value = slug(text)
  assert value
  assert len(value) <= 40
  assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in value)
  assert "__" not in value


# ----------------------------------------------------------------- collection_name


def test_collection_name_derived() -> None:
  assert (
    collection_name("Mi Informe", "abcdef0123456789")
    == "hermes_report_mi_informe_abcdef01"
  )
  assert collection_name("Mi Informe", "abcdef0123456789").startswith(PREFIX)


def test_collection_name_empty_title_uses_run() -> None:
  assert collection_name("", "1234567890") == "hermes_report_run_12345678"


def test_collection_name_explicit_wins_and_is_not_prefixed() -> None:
  assert collection_name("T", "abcdef0123", "mi_coleccion") == "mi_coleccion"


def test_collection_name_short_key() -> None:
  assert collection_name("T", "ab") == "hermes_report_t_ab"


def test_prefix_constant() -> None:
  assert PREFIX == "hermes_report_"


# --------------------------------------------------------------------- make_client


def test_make_client_memory() -> None:
  assert isinstance(make_client(":memory:"), QdrantClient)
  # Two memory clients do not share data.
  a, b = make_client(":memory:"), make_client(":memory:")
  a.create_collection(
    "x", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
  )
  assert not b.collection_exists("x")


def test_make_client_url_passes_url_and_timeout(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  seen: dict[str, Any] = {}

  class Spy:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      seen["args"], seen["kwargs"] = args, kwargs

  monkeypatch.setattr(store, "QdrantClient", Spy)
  make_client("http://qdrant.example:6333")
  assert seen == {
    "args": (),
    "kwargs": {"url": "http://qdrant.example:6333", "timeout": 60},
  }


def test_make_client_memory_is_exact_match_only(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  seen: list[Any] = []

  class Spy:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      seen.append((args, kwargs))

  monkeypatch.setattr(store, "QdrantClient", Spy)
  make_client(":MEMORY:")
  assert seen[0][1].get("url") == ":MEMORY:"


# ------------------------------------------------------------------ prefix / guard


class Exploding:
  """A client where any attribute access fails: proves no call was made."""

  def __getattr__(self, name: str) -> Any:
    raise AssertionError(f"client.{name} must not be called")


@pytest.mark.parametrize(
  "name", ["other", "", "hermes_report", "Hermes_Report_x", "x_hermes_report_"]
)
def test_name_without_prefix_raises_before_any_call(name: str) -> None:
  with pytest.raises(StoreError, match="empieza"):
    ensure_collection(_exploding(), name, _points(1), URL)


def test_prefix_guard_comes_before_the_empty_points_check() -> None:
  with pytest.raises(StoreError, match="empieza"):
    ensure_collection(_exploding(), "other", [], URL)


def test_prefix_guard_protects_an_existing_foreign_collection(
  client: QdrantClient,
) -> None:
  client.create_collection(
    "precious",
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
  )
  client.upsert(
    "precious", points=[models.PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0])]
  )
  with pytest.raises(StoreError):
    ensure_collection(client, "precious", _points(2), URL, force=True)
  assert client.count("precious").count == 1


def test_only_prefixed_collections_are_touched(client: QdrantClient) -> None:
  client.create_collection(
    "precious",
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
  )
  ensure_collection(client, NAME, _points(3), URL)
  ensure_collection(client, NAME, _points(2), URL)
  ensure_collection(client, NAME, _points(2), URL, force=True)
  names = {c.name for c in client.get_collections().collections}
  assert names == {"precious", NAME}


def test_empty_points_raise(client: QdrantClient) -> None:
  with pytest.raises(StoreError, match="ningún chunk"):
    ensure_collection(client, NAME, [], URL)
  assert not client.collection_exists(NAME)


def test_empty_points_do_not_delete_an_existing_collection(
  client: QdrantClient,
) -> None:
  ensure_collection(client, NAME, _points(2), URL)
  with pytest.raises(StoreError):
    ensure_collection(client, NAME, [], URL, force=True)
  assert client.count(NAME).count == 2


# --------------------------------------------------------- ensure_collection cycle


def test_first_call_creates_and_loads(client: QdrantClient) -> None:
  info = ensure_collection(client, NAME, _points(5, dim=8), URL)
  assert info == CollectionInfo(
    name=NAME,
    url=URL,
    reloaded=True,
    points=5,
    vector_size=8,
    distance=models.Distance.COSINE,
    indexes=("chunk_index", "doc_id"),
  )
  assert client.count(NAME, exact=True).count == 5


def test_second_call_reuses(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(5), URL)
  # Poison a stored payload: a reuse must leave it alone.
  client.set_payload(NAME, {"text": "poisoned"}, points=[str(uuid.UUID(int=0))])
  info = ensure_collection(client, NAME, _points(5), URL)
  assert info.reloaded is False
  assert info.points == 5
  assert read_points(client, NAME)[0]["payload"]["text"] == "poisoned"


def test_force_rebuilds_even_when_identical(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(3), URL)
  client.set_payload(NAME, {"text": "poisoned"}, points=[str(uuid.UUID(int=0))])
  info = ensure_collection(client, NAME, _points(3), URL, force=True)
  assert info.reloaded is True
  assert read_points(client, NAME)[0]["payload"]["text"] == "chunk 0"


def test_fewer_points_in_the_collection_reloads(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(3), URL)
  client.delete(
    NAME, points_selector=models.PointIdsList(points=[str(uuid.UUID(int=2))])
  )
  info = ensure_collection(client, NAME, _points(3), URL)
  assert info.reloaded is True and info.points == 3


def test_extra_stale_points_reload_and_are_dropped(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(4), URL)
  info = ensure_collection(client, NAME, _points(3), URL)
  assert info.reloaded is True
  assert info.points == 3
  assert len(read_points(client, NAME)) == 3


def test_same_count_but_different_ids_reloads(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(3), URL)
  other = [_point(n) for n in (10, 11, 12)]
  info = ensure_collection(client, NAME, other, URL)
  assert info.reloaded is True
  ids = {r["id"] for r in read_points(client, NAME)}
  assert ids == {p.id for p in other}


def test_same_count_one_id_differs_reloads(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(3), URL)
  swapped = _points(2) + [_point(99)]
  assert ensure_collection(client, NAME, swapped, URL).reloaded is True


def test_changed_vector_size_recreates_the_collection(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(3, dim=4), URL)
  info = ensure_collection(client, NAME, _points(3, dim=6), URL, force=True)
  assert info.vector_size == 6
  assert read_points(client, NAME)[0]["vector"]["dimension"] == 6  # type: ignore[index]


def test_same_ids_other_dimension_without_force_is_a_stale_reuse(
  client: QdrantClient,
) -> None:
  """Documented gap: reuse checks ids and count, not the vector size.

  The ingestion key (which names the collection) changes with the embedding
  model, so a different dimension normally lands in a different collection.
  """
  ensure_collection(client, NAME, _points(3, dim=4), URL)
  info = ensure_collection(client, NAME, _points(3, dim=6), URL)
  assert info.reloaded is False and info.vector_size == 4


def test_batches_of_256_load_and_check_every_point(client: QdrantClient) -> None:
  points = _points(600, dim=3)
  info = ensure_collection(client, NAME, points, URL)
  assert info.points == 600 and info.reloaded is True
  again = ensure_collection(client, NAME, points, URL)
  assert again.reloaded is False and again.points == 600
  # A single missing id deep in the last batch is noticed.
  client.delete(NAME, points_selector=models.PointIdsList(points=[points[599].id]))
  client.upsert(
    NAME,
    points=[models.PointStruct(id=str(uuid.UUID(int=9999)), vector=[1.0, 1.0, 1.0])],
  )
  assert ensure_collection(client, NAME, points, URL).reloaded is True


@pytest.mark.parametrize("count", [255, 256, 257, 512, 513])
def test_batch_boundaries(client: QdrantClient, count: int) -> None:
  points = _points(count, dim=2)
  assert ensure_collection(client, NAME, points, URL).points == count
  assert ensure_collection(client, NAME, points, URL).reloaded is False
  assert len(read_points(client, NAME)) == count


class Recorder:
  """Wrap a real client and record the calls that matter."""

  def __init__(self, inner: QdrantClient) -> None:
    self.inner = inner
    self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

  def __getattr__(self, name: str) -> Any:
    target = getattr(self.inner, name)
    if not callable(target):
      return target

    def call(*args: Any, **kwargs: Any) -> Any:
      self.calls.append((name, args, kwargs))
      return target(*args, **kwargs)

    return call


def test_loading_uses_batches_of_256_with_wait(client: QdrantClient) -> None:
  rec = _recorder(client)
  ensure_collection(rec, NAME, _points(600, dim=2), URL)
  upserts = [c for c in rec.calls if c[0] == "upsert"]
  assert [len(c[2]["points"]) for c in upserts] == [256, 256, 88]
  assert all(c[2]["wait"] is True for c in upserts)


def test_reuse_check_retrieves_in_batches_without_payload_or_vectors(
  client: QdrantClient,
) -> None:
  ensure_collection(client, NAME, _points(600, dim=2), URL)
  rec = _recorder(client)
  ensure_collection(rec, NAME, _points(600, dim=2), URL)
  retrieves = [c for c in rec.calls if c[0] == "retrieve"]
  assert [len(c[1][1]) for c in retrieves] == [256, 256, 88]
  assert all(c[2] == {"with_payload": False, "with_vectors": False} for c in retrieves)
  assert not any(
    c[0] in {"upsert", "delete_collection", "create_collection"} for c in rec.calls
  )


def test_rebuild_creates_cosine_collection_and_both_indexes(
  client: QdrantClient,
) -> None:
  rec = _recorder(client)
  ensure_collection(rec, NAME, _points(2, dim=5), URL)
  create = next(c for c in rec.calls if c[0] == "create_collection")
  vectors = create[2]["vectors_config"]
  assert vectors.size == 5 and vectors.distance == models.Distance.COSINE
  indexes = {c[1][1]: c[1][2] for c in rec.calls if c[0] == "create_payload_index"}
  assert indexes == {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "chunk_index": models.PayloadSchemaType.INTEGER,
  }
  names = [c[0] for c in rec.calls]
  assert (
    names.index("create_collection")
    < names.index("create_payload_index")
    < names.index("upsert")
  )


def test_force_deletes_the_existing_collection_first(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(2), URL)
  rec = _recorder(client)
  ensure_collection(rec, NAME, _points(2), URL, force=True)
  names = [c[0] for c in rec.calls]
  assert names.index("delete_collection") < names.index("create_collection")
  assert rec.calls[names.index("delete_collection")][1] == (NAME,)


def test_indexes_come_from_the_payload_schema_sorted() -> None:
  info = models.CollectionInfo.model_construct(
    config=models.CollectionConfig.model_construct(
      params=models.CollectionParams.model_construct(
        vectors=models.VectorParams(size=3, distance=models.Distance.COSINE)
      )
    ),
    payload_schema={"z_field": object(), "a_field": object(), "doc_id": object()},
  )

  class Fake:
    def collection_exists(self, name: str) -> bool:
      return True

    def count(self, name: str, exact: bool = True) -> Any:
      return models.CountResult(count=1)

    def retrieve(self, name: str, ids: Any, **kwargs: Any) -> list[Any]:
      return [models.Record(id=i) for i in ids]

    def get_collection(self, name: str) -> Any:
      return info

  result = ensure_collection(_as_client(Fake()), NAME, _points(1, dim=3), "http://q")
  assert result.indexes == ("a_field", "doc_id", "z_field")
  assert result.url == "http://q" and result.reloaded is False


def test_named_vector_collection_raises_store_error(client: QdrantClient) -> None:
  client.create_collection(
    NAME,
    vectors_config={
      "dense": models.VectorParams(size=4, distance=models.Distance.COSINE)
    },
  )
  points = _points(1)
  client.upsert(
    NAME,
    points=[models.PointStruct(id=points[0].id, vector={"dense": points[0].vector})],
  )
  with pytest.raises(StoreError, match="nombre"):
    ensure_collection(client, NAME, points, URL)


def test_collection_info_is_frozen() -> None:
  info = CollectionInfo(NAME, URL, True, 1, 4, "Cosine", ("chunk_index", "doc_id"))
  with pytest.raises(dataclasses.FrozenInstanceError):
    info.points = 2  # type: ignore[misc]


# ------------------------------------------------------------------- StoreError wrap


class Broken:
  """A fake client whose chosen method raises ``exc``; the others delegate."""

  def __init__(self, inner: QdrantClient, method: str, exc: BaseException) -> None:
    self.inner, self.method, self.exc = inner, method, exc

  def __getattr__(self, name: str) -> Any:
    if name == self.method:

      def boom(*args: Any, **kwargs: Any) -> Any:
        raise self.exc

      return boom
    return getattr(self.inner, name)


@pytest.mark.parametrize(
  "method",
  [
    "collection_exists",
    "create_collection",
    "create_payload_index",
    "upsert",
    "get_collection",
    "count",
  ],
)
@pytest.mark.parametrize(
  "exc",
  [
    ConnectionError("refused"),
    RuntimeError("weird"),
    OSError("net down"),
    ValueError("v"),
  ],
)
def test_any_client_failure_becomes_store_error_with_the_url(
  client: QdrantClient, method: str, exc: Exception
) -> None:
  fake = _broken(client, method, exc)
  with pytest.raises(StoreError) as info:
    ensure_collection(fake, NAME, _points(2), "http://q:6333")
  text = str(info.value)
  assert text.startswith("Qdrant no responde en http://q:6333 (")
  assert str(exc) in text
  assert info.value.__cause__ is exc


def test_delete_and_retrieve_failures_are_wrapped(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(2), URL)
  with pytest.raises(StoreError, match="Qdrant no responde en http://q"):
    ensure_collection(
      _broken(client, "delete_collection", OSError("x")),
      NAME,
      _points(2),
      "http://q",
      force=True,
    )
  with pytest.raises(StoreError, match="Qdrant no responde en http://q"):
    ensure_collection(
      _broken(client, "retrieve", OSError("x")), NAME, _points(2), "http://q"
    )


def test_qdrant_client_exception_types_are_wrapped(client: QdrantClient) -> None:
  from httpx import Headers
  from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
  )

  for exc in (
    ResponseHandlingException(ConnectionError("down")),
    UnexpectedResponse(500, "err", b"", Headers([])),
  ):
    with pytest.raises(StoreError, match="Qdrant no responde"):
      ensure_collection(
        _broken(client, "collection_exists", exc), NAME, _points(1), "http://q"
      )


def test_keyboard_interrupt_is_not_swallowed(client: QdrantClient) -> None:
  with pytest.raises(KeyboardInterrupt):
    ensure_collection(
      _broken(client, "count", KeyboardInterrupt()), NAME, _points(1), URL
    )


def test_a_failed_load_does_not_report_success(client: QdrantClient) -> None:
  with pytest.raises(StoreError):
    ensure_collection(
      _broken(client, "upsert", RuntimeError("half")), NAME, _points(3), URL
    )
  # The half-loaded collection must not be taken as complete on the next call.
  info = ensure_collection(client, NAME, _points(3), URL)
  assert info.reloaded is True and info.points == 3


def test_unreachable_server_end_to_end() -> None:
  """A real client pointed at a closed port: wrapped, not a raw httpx error."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    closed = QdrantClient(
      url="http://127.0.0.1:1", timeout=1, check_compatibility=False
    )
  with pytest.raises(
    StoreError, match=r"Qdrant no responde en http://127\.0\.0\.1:1 \("
  ):
    ensure_collection(closed, NAME, _points(1), "http://127.0.0.1:1")


# ------------------------------------------------------------------------ read_points


def test_read_points_orders_by_source_chunk_index_then_id(client: QdrantClient) -> None:
  pts = [
    _point(1, source="b.md", index=0),
    _point(2, source="a.md", index=10),
    _point(3, source="a.md", index=2),
    _point(4, source="a.md", index=2),
    _point(5, source="b.md", index=-5),
  ]
  ensure_collection(client, NAME, pts, URL)
  got = [
    (r["payload"]["source"], r["payload"]["chunk_index"], r["id"])
    for r in read_points(client, NAME)
  ]
  assert got == [
    ("a.md", 2, pts[2].id),
    ("a.md", 2, pts[3].id),
    ("a.md", 10, pts[1].id),
    ("b.md", -5, pts[4].id),
    ("b.md", 0, pts[0].id),
  ]


def test_read_points_orders_numerically_not_lexically(client: QdrantClient) -> None:
  pts = [_point(n, index=n) for n in (0, 1, 2, 9, 10, 11, 100)]
  ensure_collection(client, NAME, list(reversed(pts)), URL)
  assert [r["payload"]["chunk_index"] for r in read_points(client, NAME)] == [
    0,
    1,
    2,
    9,
    10,
    11,
    100,
  ]


def test_read_points_missing_values_sort_first(client: QdrantClient) -> None:
  client.create_collection(
    NAME, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
  )
  ids = [str(uuid.UUID(int=n)) for n in range(1, 6)]
  client.upsert(
    NAME,
    points=[
      models.PointStruct(
        id=ids[0], vector=[1.0, 0.0], payload={"source": "a.md", "chunk_index": 0}
      ),
      models.PointStruct(id=ids[1], vector=[1.0, 0.0], payload={"source": "a.md"}),
      models.PointStruct(id=ids[2], vector=[1.0, 0.0], payload={"chunk_index": 3}),
      models.PointStruct(id=ids[3], vector=[1.0, 0.0], payload={}),
      models.PointStruct(
        id=ids[4], vector=[1.0, 0.0], payload={"source": 7, "chunk_index": "x"}
      ),
    ],
  )
  got = [r["id"] for r in read_points(client, NAME)]
  # ("", -1, id1) is not guaranteed relative to wrong types; the two typed gaps are.
  assert got == [ids[3], ids[4], ids[2], ids[1], ids[0]]


def test_read_points_pages_through_the_whole_collection(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(600, dim=2), URL)
  rec = _recorder(client)
  records = read_points(rec, NAME)
  assert len(records) == 600
  assert len({r["id"] for r in records}) == 600
  scrolls = [c for c in rec.calls if c[0] == "scroll"]
  assert len(scrolls) >= 3
  assert all(
    c[2]["limit"] == 256 and c[2]["with_payload"] and c[2]["with_vectors"]
    for c in scrolls
  )
  assert [r["payload"]["chunk_index"] for r in records] == list(range(600))


def test_read_points_vector_summary_and_string_id(client: QdrantClient) -> None:
  pts = _points(2, dim=4)
  ensure_collection(client, NAME, pts, URL)
  first = read_points(client, NAME)[0]
  assert isinstance(first["id"], str) and first["id"] == pts[0].id
  summary = first["vector"]
  assert summary is not None
  assert summary["dimension"] == 4
  assert set(summary) == {"dimension", "norm", "head"}
  assert summary["head"]
  assert summary["norm"] == pytest.approx(1.0, abs=1e-4)  # cosine stores unit vectors
  assert first["payload"]["text"] == "chunk 0"


def test_read_points_integer_ids_become_strings(client: QdrantClient) -> None:
  client.create_collection(
    NAME, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
  )
  client.upsert(
    NAME, points=[models.PointStruct(id=7, vector=[1.0, 0.0], payload={"source": "a"})]
  )
  assert read_points(client, NAME)[0]["id"] == "7"


def test_read_points_named_vectors_give_none(client: QdrantClient) -> None:
  client.create_collection(
    NAME,
    vectors_config={
      "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
    },
  )
  client.upsert(NAME, points=[models.PointStruct(id=1, vector={"dense": [1.0, 0.0]})])
  [record] = read_points(client, NAME)
  assert record["vector"] is None and record["payload"] == {}


def test_read_points_empty_collection(client: QdrantClient) -> None:
  client.create_collection(
    NAME, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
  )
  assert read_points(client, NAME) == []


def test_read_points_missing_collection_is_a_store_error(client: QdrantClient) -> None:
  with pytest.raises(StoreError) as info:
    read_points(client, "hermes_report_absent_00000000")
  assert "hermes_report_absent_00000000" in str(info.value)


def test_read_points_wraps_client_failures(client: QdrantClient) -> None:
  exc = ConnectionError("refused")
  with pytest.raises(StoreError, match=NAME) as info:
    read_points(_broken(client, "scroll", exc), NAME)
  assert "refused" in str(info.value) and info.value.__cause__ is exc


def test_read_points_failure_on_a_later_page_is_wrapped(client: QdrantClient) -> None:
  ensure_collection(client, NAME, _points(600, dim=2), URL)

  class FailSecondPage:
    def __init__(self) -> None:
      self.pages = 0

    def scroll(self, *args: Any, **kwargs: Any) -> Any:
      self.pages += 1
      if self.pages == 2:
        raise OSError("lost")
      return client.scroll(*args, **kwargs)

  with pytest.raises(StoreError, match="lost"):
    read_points(_as_client(FailSecondPage()), NAME)


def test_read_points_payload_is_json_safe(client: QdrantClient) -> None:
  import json

  ensure_collection(client, NAME, _points(2), URL)
  json.dumps(read_points(client, NAME))


# ------------------------------------------------------------ end to end with ingest


def test_ingested_corpus_round_trips_through_qdrant(
  project: Path, tmp_path: Path
) -> None:
  corpus = load_corpus(project / "corpus")
  ingestion = ingest_corpus(
    corpus,
    IngestConfig(),
    EmbeddingConfig(provider="hash", model="h"),
    HashEmbedder(32),
    tmp_path / "cache",
  )
  client = make_client(":memory:")
  name = collection_name("Quijote", ingestion.key)
  assert name.startswith(PREFIX) and name.endswith(ingestion.key[:8])
  info = ensure_collection(client, name, ingestion.points, ":memory:")
  assert info.points == len(ingestion.points) == 4
  assert info.vector_size == 32 and info.reloaded is True
  records = read_points(client, name)
  assert [r["id"] for r in records] == [
    p.id
    for p in sorted(
      ingestion.points,
      key=lambda p: (p.payload["source"], p.payload["chunk_index"], p.id),
    )
  ]
  assert ensure_collection(client, name, ingestion.points, ":memory:").reloaded is False


def _broken(client: QdrantClient, method: str, exc: BaseException) -> Any:
  return Broken(client, method, exc)


def _recorder(client: QdrantClient) -> Any:
  return Recorder(client)


def _exploding() -> Any:
  return Exploding()


def _as_client(fake: object) -> QdrantClient:
  return fake  # type: ignore[return-value]

"""Tests for ``hermes_report.ingest``: the per-document cache, the key and the error paths.

Every ingestion uses ``HashEmbedder`` over copies of ``tests/data`` in ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from hermes_report.config import EmbeddingConfig, IngestConfig
from hermes_report.corpus import Corpus, load_corpus
from hermes_report.ingest import IngestError, Ingestion, ingest_corpus
from hermes_report.providers import HashEmbedder, ProviderError

EMB = EmbeddingConfig(provider="hash", model="h")
ING = IngestConfig()


class CountingEmbedder(HashEmbedder):
  """A ``HashEmbedder`` that counts how many texts it embeds."""

  def __init__(self, dimension: int = 16) -> None:
    super().__init__(dimension)
    self.texts = 0

  def get_text_embedding(self, text: str) -> list[float]:
    self.texts += 1
    return super().get_text_embedding(text)

  def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
    self.texts += len(texts)
    return super().get_text_embedding_batch(texts)


class FailingEmbedder(HashEmbedder):
  """An embedder whose every call raises ``exc``."""

  def __init__(self, exc: Exception) -> None:
    super().__init__(8)
    self.exc = exc

  def get_text_embedding(self, text: str) -> list[float]:
    raise self.exc

  def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
    raise self.exc


def _run(
  corpus: Corpus,
  cache: Path,
  *,
  ingest: IngestConfig = ING,
  embedding: EmbeddingConfig = EMB,
  embedder: Any = None,
  force: bool = False,
  progress: Any = None,
) -> Ingestion:
  return ingest_corpus(
    corpus,
    ingest,
    embedding,
    embedder if embedder is not None else HashEmbedder(16),
    cache,
    force=force,
    progress=progress,
  )


@pytest.fixture
def corpus_dir(project: Path) -> Path:
  return project / "corpus"


@pytest.fixture
def corpus(corpus_dir: Path) -> Corpus:
  return load_corpus(corpus_dir)


@pytest.fixture
def cache(tmp_path: Path) -> Path:
  return tmp_path / "cache" / "nested"


def test_miss_then_hit_returns_the_same_points(corpus: Corpus, cache: Path) -> None:
  first_embedder = CountingEmbedder()
  first = _run(corpus, cache, embedder=first_embedder)
  assert [d.source for d in first.documents] == list(corpus.sources)
  assert not any(d.cached for d in first.documents)
  assert first_embedder.texts == len(first.points) > 0
  assert len(list(cache.glob("*.jsonl"))) == len(corpus.documents)
  assert list(cache.glob("*.tmp")) == []

  second_embedder = CountingEmbedder()
  second = _run(corpus, cache, embedder=second_embedder)
  assert all(d.cached for d in second.documents)
  assert second_embedder.texts == 0
  # ``ingested_at`` is in the payload: equality proves the points were not re-made.
  assert second.points == first.points
  assert second.key == first.key


def test_cache_lines_have_id_vector_payload(corpus: Corpus, cache: Path) -> None:
  _run(corpus, cache)
  for file in cache.glob("*.jsonl"):
    for line in file.read_text(encoding="utf-8").splitlines():
      assert set(json.loads(line)) == {"id", "vector", "payload"}


def test_force_reingests_and_rewrites_the_cache(corpus: Corpus, cache: Path) -> None:
  _run(corpus, cache)
  counter = CountingEmbedder()
  forced = _run(corpus, cache, embedder=counter, force=True)
  assert not any(d.cached for d in forced.documents)
  assert counter.texts == len(forced.points)
  after = CountingEmbedder()
  again = _run(corpus, cache, embedder=after)
  assert all(d.cached for d in again.documents)
  assert again.points == forced.points


def test_progress_has_one_spanish_line_per_document(
  corpus: Corpus, cache: Path
) -> None:
  lines: list[str] = []
  _run(corpus, cache, progress=lines.append)
  assert len(lines) == len(corpus.documents)
  assert lines[0].startswith(f"{corpus.sources[0]}: ")
  counts = [re.search(r": (\d+) (chunks?) \(ingerido\)$", line) for line in lines]
  assert all(counts)
  # The noun agrees with the number: "1 chunk", never "1 chunks".
  assert all(
    m and m.group(2) == ("chunk" if m.group(1) == "1" else "chunks") for m in counts
  )
  lines.clear()
  _run(corpus, cache, progress=lines.append)
  assert all(re.search(r": \d+ chunks? \(caché\)$", line) for line in lines)
  assert "capitulo-01.md: 2 chunks (caché)" in " ".join(lines).replace("parte-1/", "")


@pytest.mark.parametrize(
  "garbage",
  [
    "not json at all\n",
    '{"id": "a"}\n',
    '{"id": 1, "vector": [0.1], "payload": {}}\n',
    '{"id": "a", "vector": "oops", "payload": {}}\n',
    '{"id": "a", "vector": [0.1, "x"], "payload": {}}\n',
    '{"id": "a", "vector": [true], "payload": {}}\n',
    '{"id": "a", "vector": [0.1], "payload": []}\n',
    "[1, 2]\n",
    "null\n",
    '{"id": "a", "vector": null, "payload": {}}\n',
    "\xff\xfe binary \x00\n",
  ],
)
def test_corrupt_cache_file_is_a_miss_and_is_rewritten(
  corpus: Corpus, cache: Path, garbage: str
) -> None:
  first = _run(corpus, cache)
  victim = sorted(cache.glob("*.jsonl"))[0]
  victim.write_bytes(garbage.encode("utf-8", errors="surrogateescape"))
  counter = CountingEmbedder()
  second = _run(corpus, cache, embedder=counter)
  assert sum(not d.cached for d in second.documents) == 1
  assert counter.texts > 0
  assert [p.id for p in second.points] == [p.id for p in first.points]
  third = _run(corpus, cache)
  assert all(d.cached for d in third.documents)


def test_invalid_utf8_cache_file_is_a_miss(corpus: Corpus, cache: Path) -> None:
  _run(corpus, cache)
  sorted(cache.glob("*.jsonl"))[0].write_bytes(b"\xff\xfe\x00 not utf-8\n")
  again = _run(corpus, cache)
  assert sum(not d.cached for d in again.documents) == 1


def test_cache_directory_that_is_a_file_raises_ingest_error(
  corpus: Corpus, tmp_path: Path
) -> None:
  blocker = tmp_path / "blocker"
  blocker.write_text("x", encoding="utf-8")
  with pytest.raises(IngestError, match="caché"):
    _run(corpus, blocker / "cache")


def test_cache_write_failure_names_the_source_and_leaves_no_temp_file(
  corpus: Corpus, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def boom(src: Any, dst: Any) -> None:
    raise OSError("disk full")

  monkeypatch.setattr(os, "replace", boom)
  with pytest.raises(IngestError, match="disk full") as info:
    _run(corpus, cache)
  assert corpus.sources[0] in str(info.value)
  assert list(cache.glob("*.tmp")) == []
  assert list(cache.glob("*.jsonl")) == []


def test_cache_write_is_atomic_through_os_replace(
  corpus: Corpus, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  calls: list[tuple[str, str]] = []
  real = os.replace

  def spy(src: Any, dst: Any) -> None:
    calls.append((str(src), str(dst)))
    real(src, dst)

  monkeypatch.setattr(os, "replace", spy)
  _run(corpus, cache)
  assert len(calls) == len(corpus.documents)
  for src, dst in calls:
    assert Path(src).parent == Path(dst).parent == cache
    assert src.endswith(".tmp") and dst.endswith(".jsonl")


def _cache_files(cache: Path) -> set[str]:
  return {p.name for p in cache.glob("*.jsonl")}


def test_document_cache_key_changes_with_each_input(
  corpus: Corpus, corpus_dir: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _run(corpus, cache)
  base = _cache_files(cache)
  n = len(base)

  def new_files(**kwargs: Any) -> int:
    before = _cache_files(cache)
    ingestion = _run(kwargs.pop("corpus", corpus), cache, **kwargs)
    assert not any(d.cached for d in ingestion.documents), kwargs
    return len(_cache_files(cache) - before)

  assert new_files(ingest=IngestConfig(max_chunk_size=400)) == n
  assert new_files(ingest=IngestConfig(hard_boundary_levels=(1,))) == n
  assert new_files(ingest=IngestConfig(strip_frontmatter=False)) == n
  assert new_files(embedding=EmbeddingConfig(provider="hash", model="other")) == n
  assert new_files(embedding=EmbeddingConfig(provider="ollama", model="h")) == n
  monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
  assert new_files() == n
  monkeypatch.undo()
  # Unchanged inputs still hit the first generation of files.
  assert all(d.cached for d in _run(corpus, cache).documents)

  # Changed content: only that document misses.
  (corpus_dir / "notas.md").write_text("# Notas\n\nOtro texto.\n", encoding="utf-8")
  changed = _run(load_corpus(corpus_dir), cache)
  assert {d.source: d.cached for d in changed.documents}["notas.md"] is False
  assert sum(d.cached for d in changed.documents) == n - 1


def test_cache_key_depends_on_the_source_path(
  corpus_dir: Path, cache: Path, tmp_path: Path
) -> None:
  """A file moved to another path with identical bytes must not reuse the cache."""
  _run(load_corpus(corpus_dir), cache)
  (corpus_dir / "notas.md").rename(corpus_dir / "renombrado.md")
  moved = _run(load_corpus(corpus_dir), cache)
  assert {d.source: d.cached for d in moved.documents}["renombrado.md"] is False


def test_ingestion_key_changes_with_each_input(
  corpus: Corpus, corpus_dir: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  base = _run(corpus, cache).key
  assert len(base) == 64 and int(base, 16) >= 0
  keys = {
    base,
    _run(corpus, cache, ingest=IngestConfig(max_chunk_size=400)).key,
    _run(corpus, cache, ingest=IngestConfig(hard_boundary_levels=(1,))).key,
    _run(corpus, cache, ingest=IngestConfig(strip_frontmatter=False)).key,
    _run(corpus, cache, embedding=EmbeddingConfig(provider="hash", model="x")).key,
    _run(corpus, cache, embedding=EmbeddingConfig(provider="ollama", model="h")).key,
  }
  monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
  keys.add(_run(corpus, cache).key)
  monkeypatch.undo()
  (corpus_dir / "notas.md").write_text("# Notas\n\nOtro.\n", encoding="utf-8")
  keys.add(_run(load_corpus(corpus_dir), cache).key)
  assert len(keys) == 8


def test_ingestion_key_ignores_what_does_not_shape_the_points(
  corpus: Corpus, cache: Path
) -> None:
  a = _run(corpus, cache, embedder=HashEmbedder(16))
  b = _run(corpus, cache, embedder=HashEmbedder(32), force=True)
  assert a.key == b.key  # documented: the key has provider and model, not dimension
  assert b.embedding["dimension"] == 32


def test_ingestion_metadata(corpus: Corpus, cache: Path) -> None:
  import hashlib

  ing = IngestConfig(max_chunk_size=900, hard_boundary_levels=(1, 2, 3))
  result = _run(corpus, cache, ingest=ing, embedder=HashEmbedder(24))
  assert result.chunker == {
    "name": "markdown_structural",
    "max_chunk_size": 900,
    "hard_boundary_levels": [1, 2, 3],
    "strip_frontmatter": True,
  }
  assert isinstance(result.chunker["hard_boundary_levels"], list)
  assert result.embedding == {"provider": "hash", "model": "h", "dimension": 24}
  version = importlib.metadata.version("hermes-ingest-md")
  assert result.ingester_version == version
  expected = hashlib.sha256(
    json.dumps(
      {
        "corpus": corpus.fingerprint,
        "chunker": result.chunker,
        "embedding": {"provider": "hash", "model": "h"},
        "ingester_version": version,
      },
      sort_keys=True,
    ).encode()
  ).hexdigest()
  assert result.key == expected
  assert result.seconds >= 0
  assert all(d.seconds >= 0 for d in result.documents)


def test_points_property_keeps_document_order(corpus: Corpus, cache: Path) -> None:
  result = _run(corpus, cache)
  assert result.points == tuple(p for d in result.documents for p in d.points)
  sources = [p.payload["source"] for p in result.points]
  assert sources == sorted(sources)
  assert len(result.points) == 4


def test_dataclasses_are_frozen(corpus: Corpus, cache: Path) -> None:
  result = _run(corpus, cache)
  with pytest.raises(dataclasses.FrozenInstanceError):
    result.key = "x"  # type: ignore[misc]
  with pytest.raises(dataclasses.FrozenInstanceError):
    result.documents[0].cached = True  # type: ignore[misc]


def test_duplicate_content_under_two_paths_names_both_sources(
  corpus_dir: Path, cache: Path
) -> None:
  text = (corpus_dir / "notas.md").read_text(encoding="utf-8")
  (corpus_dir / "copia.md").write_text(text, encoding="utf-8")
  with pytest.raises(IngestError) as info:
    _run(load_corpus(corpus_dir), cache)
  assert "copia.md" in str(info.value) and "notas.md" in str(info.value)


def test_duplicate_ids_are_detected_on_cache_hits_too(
  corpus_dir: Path, cache: Path
) -> None:
  text = (corpus_dir / "notas.md").read_text(encoding="utf-8")
  (corpus_dir / "copia.md").write_text(text, encoding="utf-8")
  corpus = load_corpus(corpus_dir)
  with pytest.raises(IngestError):
    _run(corpus, cache)
  # Both documents are now cached (written before the check) and still collide.
  with pytest.raises(IngestError, match="copia.md"):
    _run(corpus, cache)


def test_embedder_failure_is_wrapped_with_source_and_message(
  corpus: Corpus, cache: Path
) -> None:
  with pytest.raises(IngestError) as info:
    _run(corpus, cache, embedder=FailingEmbedder(RuntimeError("kaput")))
  assert corpus.sources[0] in str(info.value)
  assert "kaput" in str(info.value)
  assert isinstance(info.value.__cause__, RuntimeError)
  assert list(cache.glob("*.jsonl")) == []


def test_provider_error_message_is_kept(corpus: Corpus, cache: Path) -> None:
  err = ProviderError("Ollama no responde en http://x (boom)")
  with pytest.raises(IngestError) as info:
    _run(corpus, cache, embedder=FailingEmbedder(err))
  assert "Ollama no responde en http://x (boom)" in str(info.value)
  assert corpus.sources[0] in str(info.value)


def test_unreadable_document_is_an_ingest_error(
  corpus: Corpus, corpus_dir: Path, cache: Path
) -> None:
  (corpus_dir / "notas.md").unlink()
  with pytest.raises(IngestError, match="notas.md"):
    _run(corpus, cache)


def test_failure_in_a_later_document_keeps_earlier_cache_files(
  corpus: Corpus, cache: Path
) -> None:
  class FailOnSecond(CountingEmbedder):
    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
      if self.texts > 0:
        raise RuntimeError("second failed")
      return super().get_text_embedding_batch(texts)

  with pytest.raises(IngestError, match="second failed"):
    _run(corpus, cache, embedder=FailOnSecond())
  assert len(list(cache.glob("*.jsonl"))) == 1
  recovered = _run(corpus, cache)
  assert sum(d.cached for d in recovered.documents) == 1


def test_empty_corpus_has_no_points_and_a_none_dimension(
  corpus: Corpus, cache: Path
) -> None:
  empty = dataclasses.replace(corpus, documents=())
  result = _run(empty, cache)
  assert result.documents == () and result.points == ()
  assert result.embedding["dimension"] is None


def test_document_without_chunks_is_cached_and_hit(
  corpus_dir: Path, cache: Path
) -> None:
  (corpus_dir / "vacio.md").write_text("", encoding="utf-8")
  corpus = load_corpus(corpus_dir)
  first = _run(corpus, cache)
  empty = {d.source: d for d in first.documents}["vacio.md"]
  assert empty.points == ()
  second = _run(corpus, cache)
  assert {d.source: d for d in second.documents}["vacio.md"].cached is True


def test_unicode_payload_survives_the_cache_round_trip(
  corpus: Corpus, cache: Path
) -> None:
  first = _run(corpus, cache)
  second = _run(corpus, cache)
  assert any("í" in str(p.payload["text"]) for p in second.points)
  assert second.points == first.points
  assert all(isinstance(x, float) for p in second.points for x in p.vector)


def test_cache_dir_is_created_only_when_needed(corpus: Corpus, cache: Path) -> None:
  assert not cache.exists()
  _run(corpus, cache)
  assert cache.is_dir()


def test_blank_lines_in_a_cache_file_are_skipped(corpus: Corpus, cache: Path) -> None:
  first = _run(corpus, cache)
  for file in cache.glob("*.jsonl"):
    file.write_text(
      "\n" + file.read_text(encoding="utf-8") + "\n  \n", encoding="utf-8"
    )
  second = _run(corpus, cache)
  assert all(d.cached for d in second.documents)
  assert second.points == first.points

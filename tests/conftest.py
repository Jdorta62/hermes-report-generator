"""Shared fixtures: a throw-away project folder and clean environment."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.common import client_warnings

from hermes_report.config import RunConfig, load_config
from hermes_report.corpus import load_corpus
from hermes_report.ingest import ingest_corpus
from hermes_report.providers import HashEmbedder
from hermes_report.questions import QuestionSet, load_questions
from hermes_report.store import ensure_collection

DATA = Path(__file__).parent / "data"

MINIMAL = 'title = "T"\n[corpus]\ndir = "corpus"\n[questions]\nfile = "q.json"\n'


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
  """No test sees the developer's QDRANT_URL / OLLAMA_MODEL."""
  monkeypatch.delenv("QDRANT_URL", raising=False)
  monkeypatch.delenv("OLLAMA_MODEL", raising=False)


@pytest.fixture
def fresh_qdrant_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
  """Forget which one-time qdrant-client warnings were already shown.

  The client warns about payload indexes once per process, so a test that
  asserts "no such warning" would pass vacuously if an earlier test used it.
  """
  monkeypatch.setattr(client_warnings, "SEEN_MESSAGES", set())


@pytest.fixture
def data_dir() -> Path:
  """The checked-in sample project (read-only use)."""
  return DATA


@pytest.fixture
def project(tmp_path: Path) -> Path:
  """A copy of the sample project that a test may modify."""
  target = tmp_path / "proj"
  shutil.copytree(DATA, target)
  return target


@pytest.fixture
def write_run(tmp_path: Path) -> Callable[..., Path]:
  """Write a run file next to an empty corpus folder and a questions file."""
  folder = tmp_path / "run"
  (folder / "corpus").mkdir(parents=True)
  (folder / "corpus" / "a.md").write_text("# A\n", encoding="utf-8")
  (folder / "q.json").write_text("{}", encoding="utf-8")

  def write(extra: str = "", *, head: str = MINIMAL, name: str = "run.toml") -> Path:
    path = folder / name
    path.write_text(head + extra, encoding="utf-8")
    return path

  return write


COLLECTION = "hermes_report_pipeline_test"


@dataclass(frozen=True)
class PipelineEnv:
  """The sample corpus ingested into an in-memory Qdrant, ready to be queried."""

  config: RunConfig
  qdrant: QdrantClient
  collection: str
  embedder: HashEmbedder
  questions: QuestionSet


@pytest.fixture(scope="module")
def pipeline_env(tmp_path_factory: pytest.TempPathFactory) -> PipelineEnv:
  """``tests/data`` ingested with ``HashEmbedder`` into ``QdrantClient(":memory:")``."""
  config = load_config(DATA / "run.toml")
  embedder = HashEmbedder(256)
  ingestion = ingest_corpus(
    load_corpus(DATA / "corpus"),
    config.ingest,
    config.embedding,
    embedder,
    tmp_path_factory.mktemp("cache"),
  )
  qdrant = QdrantClient(":memory:")
  ensure_collection(qdrant, COLLECTION, ingestion.points, ":memory:")
  return PipelineEnv(
    config, qdrant, COLLECTION, embedder, load_questions(config.questions.file)
  )

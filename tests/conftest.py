"""Shared fixtures: a throw-away project folder and clean environment."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"

MINIMAL = 'title = "T"\n[corpus]\ndir = "corpus"\n[questions]\nfile = "q.json"\n'


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
  """No test sees the developer's QDRANT_URL / OLLAMA_MODEL."""
  monkeypatch.delenv("QDRANT_URL", raising=False)
  monkeypatch.delenv("OLLAMA_MODEL", raising=False)


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

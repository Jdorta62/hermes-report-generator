"""Tests for ``hermes_report.config``: defaults, validation, overrides, TOML."""

from __future__ import annotations

import dataclasses
import json
import pickle
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import MINIMAL

from hermes_report.config import (
  ConfigError,
  CorpusConfig,
  EmbeddingConfig,
  EnrichConfig,
  GenerationConfig,
  IngestConfig,
  JudgeConfig,
  MetricsConfig,
  OutputConfig,
  Overrides,
  QdrantConfig,
  QuestionsConfig,
  RetrievalConfig,
  load_config,
)

WriteRun = Callable[..., Path]


def fails(
  path: Path, key: str | None, overrides: Overrides | None = None
) -> ConfigError:
  """Load ``path``, expect a ConfigError, check it names file and key."""
  with pytest.raises(ConfigError) as info:
    load_config(path, overrides)
  err = info.value
  assert err.key == key
  assert str(path.resolve()) in str(err)
  if key is not None:
    assert key in str(err)
  return err


# --- ConfigError --------------------------------------------------------------


class TestConfigError:
  def test_attributes_and_str_with_key(self) -> None:
    err = ConfigError(Path("/x/run.toml"), "debe ser un texto", "retrieval.top_k")
    assert (err.path, err.message, err.key) == (
      Path("/x/run.toml"),
      "debe ser un texto",
      "retrieval.top_k",
    )
    assert str(err) == "/x/run.toml: retrieval.top_k: debe ser un texto"

  def test_str_without_key(self) -> None:
    err = ConfigError(Path("/x/run.toml"), "el archivo no existe")
    assert err.key is None
    assert str(err) == "/x/run.toml: el archivo no existe"

  def test_is_exception_and_survives_pickle(self) -> None:
    err: ConfigError = pickle.loads(pickle.dumps(ConfigError(Path("/x"), "m", "k")))  # noqa: S301
    assert isinstance(err, Exception)
    assert (err.path, err.message, err.key) == (Path("/x"), "m", "k")
    assert str(err) == "/x: k: m"


# --- the sample project ---------------------------------------------------------


def test_sample_run_file_loads(data_dir: Path) -> None:
  config = load_config(data_dir / "run.toml")
  assert config.path == (data_dir / "run.toml").resolve()
  assert config.title == "Don Quijote"
  assert config.description == "Batería de prueba"
  assert config.corpus.dir == (data_dir / "corpus").resolve()
  assert config.questions.file == (data_dir / "preguntas.json").resolve()
  assert config.embedding == EmbeddingConfig(provider="hash", model="hash-256")
  assert config.qdrant == QdrantConfig(url=":memory:", collection="")
  assert config.retrieval == RetrievalConfig(
    top_k=5, candidate_factor=2, score_threshold=0.3
  )
  assert config.enrich.max_chunks == 20
  assert config.generation.provider == "echo"
  assert config.generation.insufficient_message == "No lo sé con estas fuentes."
  assert config.judge == JudgeConfig(enabled=True, provider="echo", model="echo")
  assert config.output.dir == (data_dir / "out").resolve()
  assert config.output.cache_dir == (data_dir / ".hermes-report-cache").resolve()


# --- defaults -------------------------------------------------------------------


def test_every_default(write_run: WriteRun) -> None:
  path = write_run()
  folder = path.parent
  config = load_config(path)
  assert config.title == "T"
  assert config.description == ""
  assert config.corpus == CorpusConfig(dir=folder / "corpus", glob="**/*.md")
  assert config.questions == QuestionsConfig(file=folder / "q.json")
  assert config.ingest == IngestConfig(
    max_chunk_size=1500, hard_boundary_levels=(1, 2), strip_frontmatter=True
  )
  assert config.embedding == EmbeddingConfig(
    provider="ollama", model="nomic-embed-text"
  )
  assert config.qdrant == QdrantConfig(url="http://localhost:6333", collection="")
  assert config.retrieval == RetrievalConfig(
    top_k=10, candidate_factor=1, score_threshold=None
  )
  assert config.enrich == EnrichConfig(
    chain=("neighbors", "merge", "doc_order"), max_context_tokens=3000, max_chunks=None
  )
  assert config.generation == GenerationConfig(
    provider="ollama",
    model="qwen3:4b",
    temperature=0.0,
    seed=42,
    response_format="plain",
    insufficient_message=None,
  )
  assert config.judge == JudgeConfig(enabled=True, provider="ollama", model="qwen3:4b")
  assert config.output == OutputConfig(
    dir=folder / "out", cache_dir=folder / ".hermes-report-cache"
  )
  assert config.only == ()
  assert config.reingest is False


def test_dataclass_constructor_defaults() -> None:
  assert CorpusConfig(dir=Path("x")).glob == "**/*.md"
  assert EnrichConfig().chain == ("neighbors", "merge", "doc_order")
  assert GenerationConfig(provider="echo", model="m").seed == 42
  overrides = Overrides()
  assert dataclasses.astuple(overrides) == (
    None,
    None,
    False,
    None,
    None,
    None,
    (),
    False,
    False,
    (),
  )


def test_sections_are_frozen() -> None:
  with pytest.raises(dataclasses.FrozenInstanceError):
    IngestConfig().max_chunk_size = 2  # type: ignore[misc]
  with pytest.raises(dataclasses.FrozenInstanceError):
    Overrides().no_judge = True  # type: ignore[misc]


def test_run_config_is_frozen_and_hashable_fields(write_run: WriteRun) -> None:
  config = load_config(write_run())
  with pytest.raises(dataclasses.FrozenInstanceError):
    config.title = "other"  # type: ignore[misc]


# --- the file itself ------------------------------------------------------------


class TestFile:
  def test_missing_file(self, tmp_path: Path) -> None:
    err = fails(tmp_path / "nope.toml", None)
    assert "no existe" in err.message

  def test_directory_instead_of_file(self, tmp_path: Path) -> None:
    fails(tmp_path, None)

  def test_invalid_toml_carries_tomllib_message(self, write_run: WriteRun) -> None:
    path = write_run("[oops\n")
    err = fails(path, None)
    with pytest.raises(tomllib.TOMLDecodeError) as info:
      tomllib.loads(path.read_text())
    assert str(info.value) in err.message
    assert "TOML" in err.message

  def test_duplicate_key_is_invalid_toml(self, write_run: WriteRun) -> None:
    fails(write_run(head='title = "a"\ntitle = "b"\n'), None)

  def test_invalid_utf8(self, write_run: WriteRun) -> None:
    path = write_run()
    path.write_bytes(b'title = "\xff\xfe"\n')
    fails(path, None)

  def test_empty_file_reports_missing_title(self, write_run: WriteRun) -> None:
    fails(write_run(head=""), "title")

  def test_relative_run_path_is_made_absolute(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    path = write_run()
    monkeypatch.chdir(path.parent)
    config = load_config(Path("run.toml"))
    assert config.path.is_absolute()
    assert config.path == path.resolve()


# --- required keys and paths ----------------------------------------------------


class TestRequired:
  def test_missing_title(self, write_run: WriteRun) -> None:
    head = '[corpus]\ndir = "corpus"\n[questions]\nfile = "q.json"\n'
    fails(write_run(head=head), "title")

  @pytest.mark.parametrize("title", ['""', '"   "', '"\\t\\n"'])
  def test_blank_title(self, write_run: WriteRun, title: str) -> None:
    head = f'title = {title}\n[corpus]\ndir = "corpus"\n[questions]\nfile = "q.json"\n'
    fails(write_run(head=head), "title")

  @pytest.mark.parametrize("title", ["1", "true", "['a']", "1.5"])
  def test_non_string_title(self, write_run: WriteRun, title: str) -> None:
    head = f'title = {title}\n[corpus]\ndir = "corpus"\n[questions]\nfile = "q.json"\n'
    fails(write_run(head=head), "title")

  def test_missing_corpus_section(self, write_run: WriteRun) -> None:
    fails(write_run(head='title = "T"\n[questions]\nfile = "q.json"\n'), "corpus.dir")

  def test_missing_corpus_dir(self, write_run: WriteRun) -> None:
    head = 'title = "T"\n[corpus]\nglob = "*.md"\n[questions]\nfile = "q.json"\n'
    fails(write_run(head=head), "corpus.dir")

  def test_missing_questions_section(self, write_run: WriteRun) -> None:
    fails(write_run(head='title = "T"\n[corpus]\ndir = "corpus"\n'), "questions.file")

  @pytest.mark.parametrize("value", ['""', '"  "', "3", "true"])
  def test_bad_corpus_dir_value(self, write_run: WriteRun, value: str) -> None:
    head = f'title = "T"\n[corpus]\ndir = {value}\n[questions]\nfile = "q.json"\n'
    fails(write_run(head=head), "corpus.dir")

  @pytest.mark.parametrize("value", ['""', '"  "', "3", "true"])
  def test_bad_questions_file_value(self, write_run: WriteRun, value: str) -> None:
    head = f'title = "T"\n[corpus]\ndir = "corpus"\n[questions]\nfile = {value}\n'
    fails(write_run(head=head), "questions.file")

  def test_corpus_dir_must_exist(self, write_run: WriteRun) -> None:
    head = 'title = "T"\n[corpus]\ndir = "missing"\n[questions]\nfile = "q.json"\n'
    err = fails(write_run(head=head), "corpus.dir")
    assert "missing" in err.message

  def test_corpus_dir_must_be_a_folder(self, write_run: WriteRun) -> None:
    head = 'title = "T"\n[corpus]\ndir = "q.json"\n[questions]\nfile = "q.json"\n'
    fails(write_run(head=head), "corpus.dir")

  def test_questions_file_must_exist(self, write_run: WriteRun) -> None:
    head = 'title = "T"\n[corpus]\ndir = "corpus"\n[questions]\nfile = "nada.json"\n'
    err = fails(write_run(head=head), "questions.file")
    assert "nada.json" in err.message

  def test_questions_file_must_be_a_file(self, write_run: WriteRun) -> None:
    head = 'title = "T"\n[corpus]\ndir = "corpus"\n[questions]\nfile = "corpus"\n'
    fails(write_run(head=head), "questions.file")

  def test_blank_glob(self, write_run: WriteRun) -> None:
    head = 'title = "T"\n[corpus]\ndir = "corpus"\nglob = " "\n[questions]\nfile = "q.json"\n'
    fails(write_run(head=head), "corpus.glob")

  def test_non_string_glob(self, write_run: WriteRun) -> None:
    head = (
      'title = "T"\n[corpus]\ndir = "corpus"\nglob = 1\n[questions]\nfile = "q.json"\n'
    )
    fails(write_run(head=head), "corpus.glob")

  def test_custom_glob(self, write_run: WriteRun) -> None:
    head = 'title = "T"\n[corpus]\ndir = "corpus"\nglob = "*.md"\n[questions]\nfile = "q.json"\n'
    assert load_config(write_run(head=head)).corpus.glob == "*.md"

  def test_description_must_be_text(self, write_run: WriteRun) -> None:
    head = 'title = "T"\ndescription = 3\n[corpus]\ndir = "corpus"\n[questions]\nfile = "q.json"\n'
    fails(write_run(head=head), "description")


class TestPaths:
  def test_relative_paths_resolve_against_run_file_not_cwd(
    self, write_run: WriteRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    path = write_run('[output]\ndir = "sub/out"\ncache_dir = "c"\n')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_config(path)
    assert config.corpus.dir == path.parent / "corpus"
    assert config.questions.file == path.parent / "q.json"
    assert config.output.dir == path.parent / "sub" / "out"
    assert config.output.cache_dir == path.parent / "c"

  def test_absolute_paths_are_kept(self, write_run: WriteRun, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    out = tmp_path / "elsewhere-out"
    path = write_run(f'[output]\ndir = "{out}"\n')
    head = f'title = "T"\n[corpus]\ndir = "{other}"\n[questions]\nfile = "q.json"\n'
    path.write_text(head + f'[output]\ndir = "{out}"\n', encoding="utf-8")
    config = load_config(path)
    assert config.corpus.dir == other
    assert config.output.dir == out

  def test_dotdot_is_normalised(self, write_run: WriteRun) -> None:
    path = write_run()
    head = (
      'title = "T"\n[corpus]\ndir = "../run/corpus"\n[questions]\nfile = "./q.json"\n'
    )
    path.write_text(head, encoding="utf-8")
    config = load_config(path)
    assert config.corpus.dir == path.parent / "corpus"
    assert config.questions.file == path.parent / "q.json"

  def test_all_paths_are_absolute(self, write_run: WriteRun) -> None:
    config = load_config(write_run())
    for p in (
      config.path,
      config.corpus.dir,
      config.questions.file,
      config.output.dir,
      config.output.cache_dir,
    ):
      assert p.is_absolute()

  @pytest.mark.parametrize("key", ["dir", "cache_dir"])
  @pytest.mark.parametrize("value", ['""', '" "', "1"])
  def test_bad_output_path(self, write_run: WriteRun, key: str, value: str) -> None:
    fails(write_run(f"[output]\n{key} = {value}\n"), f"output.{key}")


# --- unknown keys ---------------------------------------------------------------


class TestUnknownKeys:
  def test_unknown_top_level_key(self, write_run: WriteRun) -> None:
    path = write_run(
      head='titel = "x"\n' + 'title = "T"\n[corpus]\ndir = "corpus"\n'
      '[questions]\nfile = "q.json"\n'
    )
    err = fails(path, "titel")
    assert "desconocida" in err.message

  def test_unknown_section(self, write_run: WriteRun) -> None:
    err = fails(write_run("[extra]\nx = 1\n"), "extra")
    assert "sección" in err.message

  @pytest.mark.parametrize(
    "section",
    [
      "corpus",
      "questions",
      "ingest",
      "embedding",
      "qdrant",
      "retrieval",
      "enrich",
      "generation",
      "judge",
      "output",
    ],
  )
  def test_unknown_key_in_every_section(
    self, write_run: WriteRun, section: str
  ) -> None:
    body = MINIMAL_SECTIONS.copy()
    body.setdefault(section, "")
    head = 'title = "T"\n' + "".join(
      f"[{name}]\n{text}" for name, text in body.items() if name != section
    )
    text = head + f"[{section}]\n{body[section]}typo_key = 1\n"
    fails(write_run(head=text), f"{section}.typo_key")

  def test_section_that_is_not_a_table(self, write_run: WriteRun) -> None:
    head = (
      'title = "T"\nretrieval = 3\n[corpus]\ndir = "corpus"\n'
      '[questions]\nfile = "q.json"\n'
    )
    fails(write_run(head=head), "retrieval")

  def test_misspelled_key_is_not_silently_defaulted(self, write_run: WriteRun) -> None:
    fails(write_run("[retrieval]\ntopk = 3\n"), "retrieval.topk")


MINIMAL_SECTIONS = {
  "corpus": 'dir = "corpus"\n',
  "questions": 'file = "q.json"\n',
}


# --- type strictness and value rules -------------------------------------------

# (section, key, bad TOML literal) -> must be rejected naming "section.key"
BAD_VALUES = [
  ("ingest", "max_chunk_size", "0"),
  ("ingest", "max_chunk_size", "-5"),
  ("ingest", "max_chunk_size", "true"),
  ("ingest", "max_chunk_size", "10.0"),
  ("ingest", "max_chunk_size", '"10"'),
  ("ingest", "hard_boundary_levels", "[]"),
  ("ingest", "hard_boundary_levels", "[0]"),
  ("ingest", "hard_boundary_levels", "[7]"),
  ("ingest", "hard_boundary_levels", "[1, 7]"),
  ("ingest", "hard_boundary_levels", "[true]"),
  ("ingest", "hard_boundary_levels", "[1.0]"),
  ("ingest", "hard_boundary_levels", '["1"]'),
  ("ingest", "hard_boundary_levels", "2"),
  ("ingest", "strip_frontmatter", "1"),
  ("ingest", "strip_frontmatter", '"true"'),
  ("embedding", "provider", '"echo"'),
  ("embedding", "provider", '"OLLAMA"'),
  ("embedding", "provider", '""'),
  ("embedding", "provider", "1"),
  ("embedding", "model", '""'),
  ("embedding", "model", '"  "'),
  ("embedding", "model", "1"),
  ("generation", "provider", '"hash"'),
  ("generation", "provider", '""'),
  ("generation", "provider", "true"),
  ("generation", "model", '""'),
  ("generation", "model", '" "'),
  ("generation", "model", "1"),
  ("generation", "temperature", "-0.1"),
  ("generation", "temperature", "-1"),
  ("generation", "temperature", "nan"),
  ("generation", "temperature", "-inf"),
  ("generation", "temperature", "true"),
  ("generation", "temperature", '"0.5"'),
  ("generation", "seed", "true"),
  ("generation", "seed", "1.5"),
  ("generation", "seed", '"42"'),
  ("generation", "response_format", '"html"'),
  ("generation", "response_format", '"Plain"'),
  ("generation", "response_format", "1"),
  ("generation", "insufficient_message", '""'),
  ("generation", "insufficient_message", '"   "'),
  ("generation", "insufficient_message", "1"),
  ("judge", "enabled", "1"),
  ("judge", "enabled", '"yes"'),
  ("judge", "provider", '"hash"'),
  ("judge", "provider", "3"),
  ("judge", "model", "3"),
  ("qdrant", "url", '""'),
  ("qdrant", "url", '"  "'),
  ("qdrant", "url", "1"),
  ("qdrant", "collection", '"mi_coleccion"'),
  ("qdrant", "collection", '"Hermes_report_x"'),
  ("qdrant", "collection", '"hermes_report"'),
  ("qdrant", "collection", '" hermes_report_x"'),
  ("qdrant", "collection", "1"),
  ("retrieval", "top_k", "0"),
  ("retrieval", "top_k", "-1"),
  ("retrieval", "top_k", "true"),
  ("retrieval", "top_k", "3.0"),
  ("retrieval", "top_k", '"3"'),
  ("retrieval", "candidate_factor", "0"),
  ("retrieval", "candidate_factor", "false"),
  ("retrieval", "candidate_factor", "2.5"),
  ("retrieval", "score_threshold", "true"),
  ("retrieval", "score_threshold", '"0.3"'),
  ("retrieval", "score_threshold", "nan"),
  ("enrich", "chain", '["neighbors", "magic"]'),
  ("enrich", "chain", '["Neighbors"]'),
  ("enrich", "chain", '"neighbors"'),
  ("enrich", "chain", "[1]"),
  ("enrich", "max_context_tokens", "-1"),
  ("enrich", "max_context_tokens", "true"),
  ("enrich", "max_context_tokens", "1.5"),
  ("enrich", "max_context_tokens", '"0"'),
  ("enrich", "max_chunks", "0"),
  ("enrich", "max_chunks", "-3"),
  ("enrich", "max_chunks", "true"),
  ("enrich", "max_chunks", "2.0"),
]


@pytest.mark.parametrize(("section", "key", "literal"), BAD_VALUES)
def test_bad_value_names_the_key(
  write_run: WriteRun, section: str, key: str, literal: str
) -> None:
  path = write_run(f"[{section}]\n{key} = {literal}\n", head=_head_without(section))
  with pytest.raises(ConfigError) as info:
    load_config(path)
  err = info.value
  # List elements are reported as ``section.key[i]``.
  assert err.key is not None
  assert err.key == f"{section}.{key}" or err.key.startswith(f"{section}.{key}[")
  assert err.key in str(err)
  assert str(path.resolve()) in str(err)
  assert err.message


def _head_without(section: str) -> str:
  """Required keys of the other sections, so only ``section`` is under test."""
  head = 'title = "T"\n'
  if section != "corpus":
    head += '[corpus]\ndir = "corpus"\n'
  if section != "questions":
    head += '[questions]\nfile = "q.json"\n'
  return head


def test_bad_value_in_list_names_key_or_element(write_run: WriteRun) -> None:
  path = write_run("[ingest]\nhard_boundary_levels = [1, 9]\n")
  with pytest.raises(ConfigError) as info:
    load_config(path)
  assert info.value.key is not None
  assert info.value.key.startswith("ingest.hard_boundary_levels")


class TestValidBoundaries:
  def test_boundaries_accepted(self, write_run: WriteRun) -> None:
    path = write_run(
      "[ingest]\nmax_chunk_size = 1\nhard_boundary_levels = [1, 6]\n"
      "[retrieval]\ntop_k = 1\ncandidate_factor = 1\nscore_threshold = 0\n"
      "[enrich]\nmax_context_tokens = 0\nmax_chunks = 1\nchain = []\n"
      "[generation]\ntemperature = 0\nseed = -1\n"
    )
    config = load_config(path)
    assert config.ingest.max_chunk_size == 1
    assert config.ingest.hard_boundary_levels == (1, 6)
    assert config.retrieval.top_k == 1
    assert config.retrieval.score_threshold == 0.0
    assert config.enrich.max_context_tokens == 0
    assert config.enrich.max_chunks == 1
    assert config.enrich.chain == ()
    assert config.generation.temperature == 0.0
    assert config.generation.seed == -1

  def test_int_is_accepted_as_float_and_becomes_float(
    self, write_run: WriteRun
  ) -> None:
    config = load_config(
      write_run("[generation]\ntemperature = 1\n[retrieval]\nscore_threshold = 1\n")
    )
    assert config.generation.temperature == 1.0
    assert isinstance(config.generation.temperature, float)
    assert isinstance(config.retrieval.score_threshold, float)

  def test_negative_score_threshold_is_allowed(self, write_run: WriteRun) -> None:
    # No lower bound is specified for score_threshold (cosine scores can be < 0).
    assert (
      load_config(
        write_run("[retrieval]\nscore_threshold = -0.5\n")
      ).retrieval.score_threshold
      == -0.5
    )

  def test_levels_keep_order_and_duplicates(self, write_run: WriteRun) -> None:
    config = load_config(write_run("[ingest]\nhard_boundary_levels = [3, 1, 3]\n"))
    assert config.ingest.hard_boundary_levels == (3, 1, 3)

  def test_chain_order_is_kept(self, write_run: WriteRun) -> None:
    config = load_config(write_run('[enrich]\nchain = ["doc_order", "merge"]\n'))
    assert config.enrich.chain == ("doc_order", "merge")

  @pytest.mark.parametrize("provider", ["ollama", "hash"])
  def test_embedding_providers(self, write_run: WriteRun, provider: str) -> None:
    config = load_config(write_run(f'[embedding]\nprovider = "{provider}"\n'))
    assert config.embedding.provider == provider

  @pytest.mark.parametrize("provider", ["ollama", "echo"])
  def test_llm_providers(self, write_run: WriteRun, provider: str) -> None:
    config = load_config(write_run(f'[generation]\nprovider = "{provider}"\n'))
    assert config.generation.provider == provider

  @pytest.mark.parametrize("fmt", ["plain", "markdown", "bullets"])
  def test_response_formats(self, write_run: WriteRun, fmt: str) -> None:
    config = load_config(write_run(f'[generation]\nresponse_format = "{fmt}"\n'))
    assert config.generation.response_format == fmt

  @pytest.mark.parametrize("name", ["hermes_report_", "hermes_report_demo"])
  def test_collection_prefix_accepted(self, write_run: WriteRun, name: str) -> None:
    assert (
      load_config(write_run(f'[qdrant]\ncollection = "{name}"\n')).qdrant.collection
      == name
    )

  def test_insufficient_message_kept(self, write_run: WriteRun) -> None:
    config = load_config(write_run('[generation]\ninsufficient_message = "Ni idea"\n'))
    assert config.generation.insufficient_message == "Ni idea"

  def test_explicit_url_is_kept(self, write_run: WriteRun) -> None:
    config = load_config(write_run('[qdrant]\nurl = ":memory:"\n'))
    assert config.qdrant.url == ":memory:"


# --- environment defaults -------------------------------------------------------


class TestEnvironment:
  def test_qdrant_url_from_env(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example:1234")
    assert load_config(write_run()).qdrant.url == "http://qdrant.example:1234"

  def test_file_url_wins_over_env(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://env:1")
    assert (
      load_config(write_run('[qdrant]\nurl = ":memory:"\n')).qdrant.url == ":memory:"
    )

  def test_empty_env_falls_back_to_default(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("OLLAMA_MODEL", "")
    config = load_config(write_run())
    assert config.qdrant.url == "http://localhost:6333"
    assert config.generation.model == "qwen3:4b"

  def test_ollama_model_from_env(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3:8b")
    config = load_config(write_run())
    assert config.generation.model == "llama3:8b"
    # The judge follows the generation model, env default included.
    assert config.judge.model == "llama3:8b"

  def test_file_model_wins_over_env(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3:8b")
    config = load_config(write_run('[generation]\nmodel = "gemma"\n'))
    assert config.generation.model == "gemma"
    assert config.judge.model == "gemma"

  def test_env_is_read_at_load_time_not_import_time(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    path = write_run()
    monkeypatch.setenv("OLLAMA_MODEL", "first")
    assert load_config(path).generation.model == "first"
    monkeypatch.setenv("OLLAMA_MODEL", "second")
    assert load_config(path).generation.model == "second"

  def test_env_values_are_stripped(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setenv("QDRANT_URL", "  http://q:1  ")
    monkeypatch.setenv("OLLAMA_MODEL", "\tllama3:8b\n")
    config = load_config(write_run())
    assert config.qdrant.url == "http://q:1"
    assert config.generation.model == "llama3:8b"
    assert config.judge.model == "llama3:8b"

  def test_blank_env_values_fall_back_to_defaults(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setenv("QDRANT_URL", " \t ")
    monkeypatch.setenv("OLLAMA_MODEL", "  ")
    config = load_config(write_run())
    assert config.qdrant.url == "http://localhost:6333"
    assert config.generation.model == "qwen3:4b"

  def test_whitespace_env_url_is_not_used_blindly(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    # "non-empty" means non-blank everywhere: a blank QDRANT_URL must not
    # end up as the Qdrant URL (it must fall back or raise ConfigError).
    monkeypatch.setenv("QDRANT_URL", "   ")
    try:
      config = load_config(write_run())
    except ConfigError:
      return
    assert config.qdrant.url.strip()


# --- judge ----------------------------------------------------------------------


class TestJudge:
  def test_defaults_follow_generation(self, write_run: WriteRun) -> None:
    config = load_config(write_run('[generation]\nprovider = "echo"\nmodel = "m1"\n'))
    assert config.judge == JudgeConfig(enabled=True, provider="echo", model="m1")

  def test_empty_strings_mean_default(self, write_run: WriteRun) -> None:
    config = load_config(
      write_run(
        '[generation]\nprovider = "echo"\nmodel = "m1"\n[judge]\nprovider = ""\nmodel = ""\n'
      )
    )
    assert (config.judge.provider, config.judge.model) == ("echo", "m1")

  def test_explicit_judge(self, write_run: WriteRun) -> None:
    config = load_config(
      write_run(
        '[generation]\nprovider = "echo"\nmodel = "m1"\n'
        '[judge]\nenabled = false\nprovider = "ollama"\nmodel = "big"\n'
      )
    )
    assert config.judge == JudgeConfig(enabled=False, provider="ollama", model="big")

  def test_judge_provider_alone(self, write_run: WriteRun) -> None:
    config = load_config(
      write_run('[generation]\nprovider = "echo"\n[judge]\nprovider = "ollama"\n')
    )
    assert config.judge.provider == "ollama"
    assert config.judge.model == config.generation.model

  def test_invalid_judge_provider_names_key(self, write_run: WriteRun) -> None:
    fails(write_run('[judge]\nprovider = "hash"\n'), "judge.provider")

  def test_blank_judge_model_does_not_resolve_to_blank(
    self, write_run: WriteRun
  ) -> None:
    # A whitespace-only judge.model is not a usable model name.
    try:
      config = load_config(write_run('[judge]\nmodel = "   "\n'))
    except ConfigError:
      return
    assert config.judge.model.strip()


# --- overrides ------------------------------------------------------------------


class TestOverrides:
  def test_none_and_default_overrides_change_nothing(self, write_run: WriteRun) -> None:
    path = write_run()
    base = load_config(path)
    assert load_config(path, None) == base
    assert load_config(path, Overrides()) == base

  def test_model_sets_generation_and_default_judge(self, write_run: WriteRun) -> None:
    config = load_config(write_run(), Overrides(model="phi"))
    assert config.generation.model == "phi"
    assert config.judge.model == "phi"

  def test_model_keeps_judge_model_from_file(self, write_run: WriteRun) -> None:
    config = load_config(write_run('[judge]\nmodel = "big"\n'), Overrides(model="phi"))
    assert config.generation.model == "phi"
    assert config.judge.model == "big"

  def test_model_keeps_judge_model_even_if_it_equals_generation_default(
    self, write_run: WriteRun
  ) -> None:
    # The judge model was set explicitly (to the same text as the default
    # generation model): it must not be treated as "left to its default".
    config = load_config(
      write_run('[judge]\nmodel = "qwen3:4b"\n'), Overrides(model="phi")
    )
    assert config.generation.model == "phi"
    assert config.judge.model == "qwen3:4b"

  def test_model_keeps_judge_model_when_it_equals_explicit_generation_model(
    self, write_run: WriteRun
  ) -> None:
    config = load_config(
      write_run('[generation]\nmodel = "x"\n[judge]\nmodel = "x"\n'),
      Overrides(model="phi"),
    )
    assert config.judge.model == "x"

  def test_judge_model_override_wins_over_model_override(
    self, write_run: WriteRun
  ) -> None:
    config = load_config(write_run(), Overrides(model="phi", judge_model="big"))
    assert config.generation.model == "phi"
    assert config.judge.model == "big"

  def test_judge_model_alone(self, write_run: WriteRun) -> None:
    config = load_config(write_run(), Overrides(judge_model="big"))
    assert config.judge.model == "big"
    assert config.generation.model == "qwen3:4b"

  def test_no_judge(self, write_run: WriteRun) -> None:
    assert load_config(write_run(), Overrides(no_judge=True)).judge.enabled is False

  def test_no_judge_false_does_not_enable_a_disabled_judge(
    self, write_run: WriteRun
  ) -> None:
    config = load_config(
      write_run("[judge]\nenabled = false\n"), Overrides(no_judge=False)
    )
    assert config.judge.enabled is False

  def test_top_k_and_budget(self, write_run: WriteRun) -> None:
    config = load_config(write_run(), Overrides(top_k=3, budget=0))
    assert config.retrieval.top_k == 3
    assert config.enrich.max_context_tokens == 0

  def test_override_replaces_file_value(self, write_run: WriteRun) -> None:
    config = load_config(
      write_run("[retrieval]\ntop_k = 20\n[enrich]\nmax_context_tokens = 100\n"),
      Overrides(top_k=2, budget=50),
    )
    assert (config.retrieval.top_k, config.enrich.max_context_tokens) == (2, 50)

  def test_out_is_resolved_against_cwd_not_run_file(
    self, write_run: WriteRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    path = write_run()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = load_config(path, Overrides(out=Path("results")))
    assert config.output.dir == (cwd / "results").resolve()
    assert config.output.cache_dir == path.parent / ".hermes-report-cache"

  def test_absolute_out(self, write_run: WriteRun, tmp_path: Path) -> None:
    target = tmp_path / "abs-out"
    config = load_config(write_run(), Overrides(out=target))
    assert config.output.dir == target

  def test_only_and_reingest_are_copied(self, write_run: WriteRun) -> None:
    config = load_config(write_run(), Overrides(only=("q01", "q03"), reingest=True))
    assert config.only == ("q01", "q03")
    assert config.reingest is True

  def test_only_given_as_list_is_normalised_to_tuple(self, write_run: WriteRun) -> None:
    overrides = Overrides(only=["q02"])  # type: ignore[arg-type]
    assert load_config(write_run(), overrides).only == ("q02",)

  @pytest.mark.parametrize(
    ("overrides", "key"),
    [
      (Overrides(top_k=0), "retrieval.top_k"),
      (Overrides(top_k=-3), "retrieval.top_k"),
      (Overrides(top_k=True), "retrieval.top_k"),
      (Overrides(budget=-1), "enrich.max_context_tokens"),
      (Overrides(budget=True), "enrich.max_context_tokens"),
      (Overrides(model=""), "generation.model"),
      (Overrides(model="  "), "generation.model"),
      (Overrides(judge_model=""), "judge.model"),
      (Overrides(judge_model="\t"), "judge.model"),
    ],
  )
  def test_invalid_override_is_validated(
    self, write_run: WriteRun, overrides: Overrides, key: str
  ) -> None:
    err = fails(write_run(), key, overrides)
    assert err.message.endswith("(opción de la línea de comandos)")

  def test_file_errors_do_not_carry_the_cli_suffix(self, write_run: WriteRun) -> None:
    err = fails(write_run("[retrieval]\ntop_k = 0\n"), "retrieval.top_k")
    assert "línea de comandos" not in err.message

  def test_file_error_wins_over_valid_override(self, write_run: WriteRun) -> None:
    fails(write_run("[retrieval]\ntop_k = 0\n"), "retrieval.top_k", Overrides(top_k=3))

  def test_valid_file_with_huge_override_value(self, write_run: WriteRun) -> None:
    assert load_config(write_run(), Overrides(top_k=10**30)).retrieval.top_k == 10**30

  def test_overrides_do_not_mutate_other_sections(self, write_run: WriteRun) -> None:
    path = write_run(
      "[generation]\ntemperature = 0.7\n[retrieval]\nscore_threshold = 0.2\n"
    )
    base = load_config(path)
    over = load_config(path, Overrides(model="m", top_k=2, budget=9, no_judge=True))
    assert over.generation.temperature == 0.7
    assert over.retrieval.score_threshold == 0.2
    assert over.corpus == base.corpus
    assert over.ingest == base.ingest

  # --- robustness: bad input must always surface as ConfigError -------------------

  @pytest.mark.parametrize("blank", ["   ", "\t"])
  def test_blank_out_override_is_rejected(
    self, write_run: WriteRun, blank: str
  ) -> None:
    fails(write_run(), "output.dir", Overrides(out=Path(blank)))

  @pytest.mark.parametrize("current", [".", ""])
  def test_dot_out_override_resolves_to_the_current_directory(
    self, write_run: WriteRun, current: str
  ) -> None:
    # Path("") is Path("."): both mean the current directory (the CLI rejects an
    # empty --out before a Path exists).
    config = load_config(write_run(), Overrides(out=Path(current)))
    assert config.output.dir == Path.cwd()

  def test_explicit_out_override_is_accepted(
    self, write_run: WriteRun, tmp_path: Path
  ) -> None:
    config = load_config(write_run(), Overrides(out=tmp_path / "o"))
    assert config.output.dir == tmp_path / "o"

  @pytest.mark.parametrize("value", ["q01", "", b"q01", 5, None])
  def test_only_must_not_be_a_scalar(self, write_run: WriteRun, value: Any) -> None:
    err = fails(write_run(), "only", Overrides(only=value))
    assert "lista" in err.message

  @pytest.mark.parametrize("bad", ["", "  ", 3, None, b"x"])
  def test_only_items_must_be_non_blank_strings(
    self, write_run: WriteRun, bad: Any
  ) -> None:
    # The override wrapper reports the field name, not the item index.
    fails(write_run(), "only", Overrides(only=("q01", bad)))

  def test_only_list_is_accepted_and_stored_as_tuple(self, write_run: WriteRun) -> None:
    config = load_config(write_run(), Overrides(only=["q01", "q02"]))  # type: ignore[arg-type]
    assert config.only == ("q01", "q02")


class TestRobustness:
  def test_huge_integer_for_a_float_key(self, write_run: WriteRun) -> None:
    # float(10**400) raises OverflowError; the contract is ConfigError.
    fails(
      write_run(f"[generation]\ntemperature = {'9' * 400}\n"), "generation.temperature"
    )

  def test_huge_integer_for_score_threshold(self, write_run: WriteRun) -> None:
    fails(
      write_run(f"[retrieval]\nscore_threshold = {'9' * 400}\n"),
      "retrieval.score_threshold",
    )

  def test_huge_integer_in_the_file_does_not_crash(self, write_run: WriteRun) -> None:
    # tomllib raises a bare ValueError past 4300 digits (int conversion limit).
    fails(write_run(f"[ingest]\nmax_chunk_size = {'9' * 5000}\n"), None)

  @pytest.mark.parametrize("closed", [False, True])
  def test_deeply_nested_array_does_not_crash(
    self, write_run: WriteRun, closed: bool
  ) -> None:
    # tomllib raises RecursionError; the contract is ConfigError.
    tail = "]" * 5000 if closed else ""
    fails(write_run(f"[ingest]\nhard_boundary_levels = {'[' * 5000}{tail}\n"), None)

  def test_nul_byte_in_run_file_path(self) -> None:
    with pytest.raises(ConfigError):
      load_config(Path("a\0b.toml"))

  def test_nul_byte_in_corpus_dir(self, write_run: WriteRun) -> None:
    head = 'title = "T"\n[corpus]\ndir = "co\\u0000r"\n[questions]\nfile = "q.json"\n'
    fails(write_run(head=head), "corpus.dir")

  def test_nul_byte_in_output_dir(self, write_run: WriteRun) -> None:
    fails(write_run('[output]\ndir = "a\\u0000b"\n'), "output.dir")

  def test_over_long_path_component(self, write_run: WriteRun) -> None:
    head = (
      f'title = "T"\n[corpus]\ndir = "{"a" * 5000}"\n[questions]\nfile = "q.json"\n'
    )
    fails(write_run(head=head), "corpus.dir")

  def test_nul_byte_in_out_override(self, write_run: WriteRun) -> None:
    with pytest.raises(ConfigError):
      load_config(write_run(), Overrides(out=Path("a\0b")))

  def test_directory_as_run_file(self, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
      load_config(tmp_path)

  def test_over_long_run_file_name(self, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
      load_config(tmp_path / ("a" * 5000))

  def test_nul_byte_inside_the_toml_text(self, tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_bytes(b'title = "a\x00b"\n')
    fails(path, None)

  def test_invalid_utf8_run_file(self, tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_bytes(b'title = "\xff"\n')
    fails(path, None)

  def test_datetime_where_int_expected(self, write_run: WriteRun) -> None:
    fails(write_run("[retrieval]\ntop_k = 1979-05-27\n"), "retrieval.top_k")

  def test_nested_table_where_scalar_expected(self, write_run: WriteRun) -> None:
    fails(write_run("[retrieval.top_k]\nx = 1\n"), "retrieval.top_k")

  def test_inline_table_for_section_key(self, write_run: WriteRun) -> None:
    fails(write_run("[generation]\nmodel = { a = 1 }\n"), "generation.model")


# --- to_dict / to_toml ---------------------------------------------------------


FULL_RUN = """
[ingest]
max_chunk_size = 800
hard_boundary_levels = [1, 2, 3]
strip_frontmatter = false

[embedding]
provider = "hash"
model = "hash-256"

[qdrant]
url = ":memory:"
collection = "hermes_report_demo"

[retrieval]
top_k = 4
candidate_factor = 3
score_threshold = 0.25

[enrich]
chain = ["merge", "doc_order"]
max_context_tokens = 0
max_chunks = 7

[generation]
provider = "echo"
model = "m-ñ"
temperature = 0.7
seed = 7
response_format = "bullets"
insufficient_message = "No sé: \\t¿y tú?"

[judge]
enabled = false
provider = "echo"
model = "judge-m"

[output]
dir = "salida"
cache_dir = "cache"
"""

FULL_HEAD = (
  'title = "T"\ndescription = "Línea 1\\nLínea 2 con \\"comillas\\" y \\\\ barra"\n'
  '[corpus]\ndir = "corpus"\n[questions]\nfile = "q.json"\n'
)


def full(write_run: WriteRun) -> Path:
  return write_run(FULL_RUN, head=FULL_HEAD)


class TestToDict:
  def test_layout_of_defaults(self, write_run: WriteRun) -> None:
    config = load_config(write_run())
    folder = str(config.path.parent)
    assert config.to_dict() == {
      "title": "T",
      "description": "",
      "corpus": {"dir": f"{folder}/corpus", "glob": "**/*.md"},
      "questions": {"file": f"{folder}/q.json"},
      "ingest": {
        "max_chunk_size": 1500,
        "hard_boundary_levels": [1, 2],
        "strip_frontmatter": True,
      },
      "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
      "qdrant": {"url": "http://localhost:6333", "collection": ""},
      "retrieval": {"top_k": 10, "candidate_factor": 1},
      "enrich": {
        "chain": ["neighbors", "merge", "doc_order"],
        "max_context_tokens": 3000,
      },
      "generation": {
        "provider": "ollama",
        "model": "qwen3:4b",
        "temperature": 0.0,
        "seed": 42,
        "response_format": "plain",
      },
      "judge": {"enabled": True, "provider": "ollama", "model": "qwen3:4b"},
      "output": {"dir": f"{folder}/out", "cache_dir": f"{folder}/.hermes-report-cache"},
      "metrics": {"enabled": True, "include": [], "exclude": []},
    }

  def test_is_plain_json_data(self, write_run: WriteRun) -> None:
    data = load_config(full(write_run)).to_dict()
    assert json.loads(json.dumps(data)) == data

  def test_none_values_are_left_out_and_set_ones_kept(
    self, write_run: WriteRun
  ) -> None:
    data = load_config(full(write_run)).to_dict()
    assert data["retrieval"]["score_threshold"] == 0.25
    assert data["enrich"]["max_chunks"] == 7
    data = load_config(write_run()).to_dict()
    assert "score_threshold" not in data["retrieval"]
    assert "max_chunks" not in data["enrich"]
    assert "insufficient_message" not in data["generation"]

  def test_only_and_reingest_are_not_included(self, write_run: WriteRun) -> None:
    data = load_config(write_run(), Overrides(only=("q1",), reingest=True)).to_dict()
    assert "only" not in data and "reingest" not in data
    assert all(
      "only" not in v and "reingest" not in v
      for v in data.values()
      if isinstance(v, dict)
    )

  def test_no_tuple_or_path_objects_remain(self, write_run: WriteRun) -> None:
    def walk(value: Any) -> None:
      assert not isinstance(value, tuple | Path)
      if isinstance(value, dict):
        for item in value.values():
          walk(item)
      if isinstance(value, list):
        for item in value:
          walk(item)

    walk(load_config(full(write_run)).to_dict())

  def test_paths_are_absolute_strings(self, write_run: WriteRun) -> None:
    data = load_config(full(write_run)).to_dict()
    for section, key in (
      ("corpus", "dir"),
      ("questions", "file"),
      ("output", "dir"),
      ("output", "cache_dir"),
    ):
      assert isinstance(data[section][key], str)
      assert Path(data[section][key]).is_absolute()

  def test_overrides_are_reflected(self, write_run: WriteRun) -> None:
    data = load_config(
      write_run(), Overrides(model="phi", top_k=2, budget=5, no_judge=True)
    ).to_dict()
    assert data["generation"]["model"] == "phi"
    assert data["judge"]["model"] == "phi"
    assert data["judge"]["enabled"] is False
    assert data["retrieval"]["top_k"] == 2
    assert data["enrich"]["max_context_tokens"] == 5

  def test_hand_built_none_seed_is_left_out(self, write_run: WriteRun) -> None:
    config = load_config(write_run())
    built = dataclasses.replace(
      config, generation=dataclasses.replace(config.generation, seed=None)
    )
    assert "seed" not in built.to_dict()["generation"]


class TestToToml:
  def test_output_is_valid_toml_equal_to_to_dict(self, write_run: WriteRun) -> None:
    for path in (write_run(), full(write_run)):
      config = load_config(path)
      assert tomllib.loads(config.to_toml()) == config.to_dict()

  def test_ends_with_one_newline(self, write_run: WriteRun) -> None:
    text = load_config(write_run()).to_toml()
    assert text.endswith("\n") and not text.endswith("\n\n")

  def test_round_trip_defaults(self, write_run: WriteRun) -> None:
    config = load_config(write_run())
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    again = load_config(saved)
    assert again == config
    assert again.path == saved.resolve()
    assert again.path != config.path

  def test_round_trip_every_option(self, write_run: WriteRun) -> None:
    config = load_config(full(write_run))
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    again = load_config(saved)
    assert again == config
    assert again.generation.insufficient_message == "No sé: \t¿y tú?"
    assert again.description == 'Línea 1\nLínea 2 con "comillas" y \\ barra'

  def test_round_trip_is_a_fixed_point(self, write_run: WriteRun) -> None:
    config = load_config(full(write_run))
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    assert load_config(saved).to_toml() == config.to_toml()

  def test_round_trip_sample_project(self, data_dir: Path, project: Path) -> None:
    config = load_config(project / "run.toml")
    saved = project / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    assert load_config(saved) == config

  def test_round_trip_with_overrides_applied(
    self, write_run: WriteRun, tmp_path: Path
  ) -> None:
    config = load_config(
      write_run(),
      Overrides(
        model="phi",
        judge_model="big",
        top_k=2,
        budget=0,
        no_judge=True,
        out=tmp_path / "o",
      ),
    )
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    again = load_config(saved)
    assert again == config  # only/reingest are empty here

  def test_round_trip_does_not_depend_on_the_environment(
    self, write_run: WriteRun, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://a:1")
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")
    config = load_config(write_run())
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    monkeypatch.setenv("QDRANT_URL", "http://b:2")
    monkeypatch.setenv("OLLAMA_MODEL", "other-model")
    assert load_config(saved) == config

  def test_path_is_not_part_of_equality(self, write_run: WriteRun) -> None:
    config = load_config(write_run())
    assert dataclasses.replace(config, path=Path("/elsewhere/run.toml")) == config

  def test_content_still_part_of_equality(self, write_run: WriteRun) -> None:
    config = load_config(write_run())
    assert dataclasses.replace(config, title="other") != config
    assert dataclasses.replace(config, only=("q1",)) != config

  @pytest.mark.parametrize(
    "text",
    [
      "plain",
      'quote " inside',
      "back\\slash",
      "tab\tnewline\nreturn\r",
      "ctl\x01\x1f\x7f",
      "bell\x07 backspace\x08 formfeed\x0c",
      "ñandú ¿qué? ✓ 😀",
      "'''",
      '"""',
      "# not a comment",
      "[table] = x",
      "a b",
      "x" * 10_000,
    ],
  )
  def test_hostile_strings_round_trip(self, write_run: WriteRun, text: str) -> None:
    config = load_config(write_run())
    config = dataclasses.replace(
      config,
      title=text + "t",
      description=text,
      generation=dataclasses.replace(config.generation, insufficient_message=text),
      embedding=dataclasses.replace(config.embedding, model=text + "m"),
    )
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    assert load_config(saved) == config

  def test_nul_in_a_string_round_trips(self, write_run: WriteRun) -> None:
    config = dataclasses.replace(load_config(write_run()), description="a\x00b")
    assert tomllib.loads(config.to_toml())["description"] == "a\x00b"

  @pytest.mark.parametrize(
    "value", [0.0, 0.1, 0.7, 1e-7, 1e22, 123456789.123456789, float("inf")]
  )
  def test_floats_round_trip(self, write_run: WriteRun, value: float) -> None:
    config = load_config(write_run())
    config = dataclasses.replace(
      config,
      generation=dataclasses.replace(config.generation, temperature=value),
      retrieval=dataclasses.replace(config.retrieval, score_threshold=value),
    )
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    again = load_config(saved)
    assert again.generation.temperature == value
    assert again.retrieval.score_threshold == value

  def test_path_with_awkward_characters_round_trips(self, tmp_path: Path) -> None:
    folder = tmp_path / 'we"ird \\ dir ñ'
    (folder / "corpus").mkdir(parents=True)
    (folder / "q.json").write_text("{}", encoding="utf-8")
    run = folder / "run.toml"
    run.write_text(
      'title = "T"\n[corpus]\ndir = "corpus"\n[questions]\nfile = "q.json"\n',
      encoding="utf-8",
    )
    config = load_config(run)
    saved = folder / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    assert load_config(saved) == config

  def test_empty_chain_round_trips(self, write_run: WriteRun) -> None:
    config = load_config(write_run("[enrich]\nchain = []\n"))
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    assert load_config(saved).enrich.chain == ()

  def test_seed_round_trips_when_set(self, write_run: WriteRun) -> None:
    config = load_config(write_run("[generation]\nseed = -3\n"))
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    assert load_config(saved).generation.seed == -3

  def test_to_toml_saved_elsewhere_still_points_to_the_same_inputs(
    self, write_run: WriteRun, tmp_path: Path
  ) -> None:
    # Paths are absolute in the saved file, so it also works from another folder.
    config = load_config(write_run())
    other = tmp_path / "another-folder"
    other.mkdir()
    saved = other / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    again = load_config(saved)
    assert again.corpus == config.corpus
    assert again.questions == config.questions
    assert again.output == config.output


# --- [metrics] (task 5.5.4) ----------------------------------------------------


class TestMetricsConfig:
  def test_defaults(self, write_run: WriteRun) -> None:
    config = load_config(write_run())
    assert config.metrics == MetricsConfig(enabled=True, include=(), exclude=())

  def test_section_is_frozen(self) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
      MetricsConfig().enabled = False  # type: ignore[misc]

  def test_reads_every_key(self, write_run: WriteRun) -> None:
    config = load_config(
      write_run(
        '[metrics]\nenabled = true\ninclude = ["bleu", "rouge_l"]\nexclude = ["cost"]\n'
      )
    )
    assert config.metrics == MetricsConfig(True, ("bleu", "rouge_l"), ("cost",))

  def test_disabled(self, write_run: WriteRun) -> None:
    assert (
      load_config(write_run("[metrics]\nenabled = false\n")).metrics.enabled is False
    )

  @pytest.mark.parametrize("key", ["include", "exclude"])
  def test_unknown_name_names_the_key_and_lists_options(
    self, write_run: WriteRun, key: str
  ) -> None:
    err = fails(write_run(f'[metrics]\n{key} = ["bleu", "nope"]\n'), f"metrics.{key}")
    assert "nope" in err.message
    assert "faithfulness" in err.message  # the options are listed

  def test_unknown_name_is_rejected_even_when_disabled(
    self, write_run: WriteRun
  ) -> None:
    fails(
      write_run('[metrics]\nenabled = false\ninclude = ["nope"]\n'), "metrics.include"
    )

  @pytest.mark.parametrize("name", ["BLEU", " bleu", "bleu ", "rouge-l", "", "Bleu"])
  def test_names_are_exact(self, write_run: WriteRun, name: str) -> None:
    with pytest.raises(ConfigError) as info:
      load_config(write_run(f'[metrics]\ninclude = ["{name}"]\n'))
    assert "metrics.include" in str(info.value)

  @pytest.mark.parametrize(
    "value", ['"bleu"', "1", "true", "{}", "[1]", "[true]", "[[]]"]
  )
  def test_include_must_be_a_list_of_texts(
    self, write_run: WriteRun, value: str
  ) -> None:
    with pytest.raises(ConfigError) as info:
      load_config(write_run(f"[metrics]\ninclude = {value}\n"))
    assert "metrics.include" in str(info.value)

  @pytest.mark.parametrize("value", ['"yes"', "1", "0", '"true"', "[]"])
  def test_enabled_must_be_a_bool(self, write_run: WriteRun, value: str) -> None:
    fails(write_run(f"[metrics]\nenabled = {value}\n"), "metrics.enabled")

  def test_unknown_key_in_section(self, write_run: WriteRun) -> None:
    fails(write_run("[metrics]\nselect = []\n"), "metrics.select")

  def test_section_must_be_a_table(self, write_run: WriteRun) -> None:
    fails(write_run(head='metrics = "all"\n' + MINIMAL), "metrics")

  def test_name_in_both_lists_loads_and_exclude_wins_at_run_time(
    self, write_run: WriteRun
  ) -> None:
    config = load_config(
      write_run('[metrics]\ninclude = ["bleu"]\nexclude = ["bleu"]\n')
    )
    assert config.metrics.include == ("bleu",) == config.metrics.exclude

  def test_overrides_add_to_include_without_duplicates_in_order(
    self, write_run: WriteRun
  ) -> None:
    config = load_config(
      write_run('[metrics]\ninclude = ["bleu"]\n'),
      Overrides(metrics=("rouge_l", "bleu", "rouge_l", "bias")),
    )
    assert config.metrics.include == ("bleu", "rouge_l", "bias")

  def test_overrides_do_not_touch_exclude_or_enabled(self, write_run: WriteRun) -> None:
    config = load_config(
      write_run('[metrics]\nexclude = ["cost"]\n'), Overrides(metrics=("bleu",))
    )
    assert config.metrics == MetricsConfig(True, ("bleu",), ("cost",))

  def test_no_metrics_override_disables_and_keeps_the_lists(
    self, write_run: WriteRun
  ) -> None:
    config = load_config(
      write_run('[metrics]\ninclude = ["bleu"]\n'), Overrides(no_metrics=True)
    )
    assert config.metrics == MetricsConfig(False, ("bleu",), ())

  def test_no_metrics_override_beats_enabled_true_and_default_is_off(
    self, write_run: WriteRun
  ) -> None:
    on = load_config(write_run("[metrics]\nenabled = true\n"), Overrides())
    off = load_config(
      write_run("[metrics]\nenabled = true\n"), Overrides(no_metrics=True)
    )
    assert on.metrics.enabled and not off.metrics.enabled

  @pytest.mark.parametrize("bad", [("nope",), ("bleu", "nope"), ("",), ("BLEU",)])
  def test_unknown_override_is_a_config_error_with_the_key(
    self, write_run: WriteRun, bad: tuple[str, ...]
  ) -> None:
    err = fails(write_run(), "metrics.include", Overrides(metrics=bad))
    assert "línea de comandos" in err.message

  def test_override_must_not_be_a_bare_string(self, write_run: WriteRun) -> None:
    fails(write_run(), "metrics.include", Overrides(metrics="bleu"))  # type: ignore[arg-type]

  def test_override_non_string_item(self, write_run: WriteRun) -> None:
    fails(write_run(), "metrics.include", Overrides(metrics=(3,)))  # type: ignore[arg-type]

  def test_to_dict_layout(self, write_run: WriteRun) -> None:
    config = load_config(
      write_run('[metrics]\nenabled = false\ninclude = ["bleu"]\nexclude = ["cost"]\n')
    )
    assert config.to_dict()["metrics"] == {
      "enabled": False,
      "include": ["bleu"],
      "exclude": ["cost"],
    }

  def test_round_trip_through_to_toml(self, write_run: WriteRun) -> None:
    config = load_config(
      write_run(
        '[metrics]\nenabled = false\ninclude = ["bleu", "bias"]\nexclude = ["cost"]\n'
      )
    )
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    assert load_config(saved).metrics == config.metrics
    assert load_config(saved).to_toml() == config.to_toml()

  def test_round_trip_with_overrides_applied(self, write_run: WriteRun) -> None:
    config = load_config(write_run(), Overrides(metrics=("bleu",), no_metrics=True))
    saved = config.path.parent / "saved.toml"
    saved.write_text(config.to_toml(), encoding="utf-8")
    assert load_config(saved).metrics == MetricsConfig(False, ("bleu",), ())

  def test_runconfig_built_by_hand_without_metrics_still_works(
    self, write_run: WriteRun
  ) -> None:
    config = load_config(write_run())
    fields = {
      f.name: getattr(config, f.name)
      for f in dataclasses.fields(config)
      if f.name != "metrics"
    }
    assert type(config)(**fields).metrics == MetricsConfig()

  def test_unknown_names_in_the_metrics_import_do_not_break_load_without_section(
    self, write_run: WriteRun
  ) -> None:
    # The registry import is lazy; loading must work with no [metrics] at all.
    assert load_config(write_run()).metrics.include == ()

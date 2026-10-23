"""Tests for ``hermes_report.runner``: a whole run over the offline providers.

Every run uses the sample project (``tests/data``): an in-memory Qdrant, the hash
embedder, the echo LLM and the echo judge. Only ``check_ollama`` is monkeypatched,
so no test touches a real service.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import DATA
from hermes_core.generation import PromptMessage
from hermes_generator import GenerationParams, LLMReply
from qdrant_client import QdrantClient

from hermes_report import runner
from hermes_report.config import ConfigError, Overrides, RunConfig, load_config
from hermes_report.corpus import CorpusError
from hermes_report.ingest import IngestError, ingest_corpus
from hermes_report.providers import (
  EchoJudgeClient,
  EchoLLM,
  HashEmbedder,
  ProviderError,
)
from hermes_report.questions import QuestionsError
from hermes_report.record import RecordError, read_record
from hermes_report.render import render_run
from hermes_report.runner import RunError, RunOutcome, run
from hermes_report.store import StoreError, slug


@pytest.fixture(autouse=True)
def quiet_qdrant() -> Iterator[None]:
  """The local Qdrant warns that payload indexes have no effect; not under test."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    yield


@pytest.fixture
def out(tmp_path: Path) -> Path:
  return tmp_path / "out"


@pytest.fixture
def make_config(project: Path, out: Path) -> Callable[..., RunConfig]:
  """Load the sample run file with ``out`` as output folder."""

  def make(**overrides: Any) -> RunConfig:
    overrides.setdefault("out", out)
    return load_config(project / "run.toml", Overrides(**overrides))

  return make


@pytest.fixture
def config(make_config: Callable[..., RunConfig]) -> RunConfig:
  return make_config()


@pytest.fixture
def forbid_ollama(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
  """``check_ollama`` fails the test when called; returns the (empty) call list."""
  calls: list[list[str]] = []

  def refuse(models: Sequence[str], *args: Any, **kwargs: Any) -> None:
    calls.append(list(models))
    raise AssertionError("check_ollama must not be called for an offline run")

  monkeypatch.setattr(runner, "check_ollama", refuse)
  return calls


def lines_of(config: RunConfig) -> tuple[RunOutcome, list[str]]:
  lines: list[str] = []
  outcome = run(config, progress=lines.append)
  return outcome, lines


# ------------------------------------------------------------- folder layout


def test_folder_layout(config: RunConfig, out: Path, forbid_ollama: Any) -> None:
  outcome = run(config, progress=lambda line: None)
  assert outcome.folder.parent == out
  assert re.fullmatch(r"\d{8}-\d{6}-don_quijote", outcome.folder.name)
  assert sorted(p.name for p in outcome.folder.iterdir()) == [
    "report.html",
    "run.json",
    "run.toml",
  ]
  assert outcome.failed == 0


def test_outcome_is_frozen(config: RunConfig) -> None:
  outcome = run(config, progress=lambda line: None)
  with pytest.raises(dataclasses.FrozenInstanceError):
    outcome.failed = 3  # type: ignore[misc]


def test_folder_timestamp_is_the_start_of_the_run(
  config: RunConfig, forbid_ollama: Any
) -> None:
  before = datetime.now().astimezone().replace(microsecond=0)
  outcome = run(config, progress=lambda line: None)
  record = outcome.record
  started = datetime.fromisoformat(record["started_at"])
  stamp = outcome.folder.name[:15]
  assert stamp == f"{started:%Y%m%d-%H%M%S}"
  assert started >= before
  assert datetime.fromisoformat(record["finished_at"]) >= started


def test_timestamps_carry_an_offset(config: RunConfig) -> None:
  record = run(config, progress=lambda line: None).record
  for key in ("started_at", "finished_at"):
    parsed = datetime.fromisoformat(record[key])
    assert parsed.tzinfo is not None
    assert re.search(r"[+-]\d\d:\d\d$", record[key])


def test_title_is_slugged_in_the_folder_name(
  make_config: Callable[..., RunConfig], project: Path, out: Path
) -> None:
  text = (project / "run.toml").read_text(encoding="utf-8")
  (project / "run.toml").write_text(
    text.replace('title = "Don Quijote"', 'title = "¡Año Ñandú / 2: la vuelta!"'),
    encoding="utf-8",
  )
  outcome = run(make_config(), progress=lambda line: None)
  title = outcome.record["title"]
  assert outcome.folder.name.endswith("-" + slug(title))
  assert "/" not in outcome.folder.name[16:]


def test_missing_output_folders_are_created(
  make_config: Callable[..., RunConfig], tmp_path: Path
) -> None:
  deep = tmp_path / "a" / "b" / "c"
  outcome = run(make_config(out=deep), progress=lambda line: None)
  assert outcome.folder.parent == deep
  assert (outcome.folder / "run.json").is_file()


def test_out_dot_writes_into_the_current_directory(
  make_config: Callable[..., RunConfig],
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  here = tmp_path / "here"
  here.mkdir()
  monkeypatch.chdir(here)
  outcome = run(make_config(out=Path(".")), progress=lambda line: None)
  assert outcome.folder.parent == here.resolve()


class _FixedDatetime(datetime):
  @classmethod
  def now(cls, tz: Any = None) -> _FixedDatetime:
    return cls(2026, 1, 2, 3, 4, 5, tzinfo=UTC).astimezone(tz)


def test_existing_folder_gets_a_numeric_suffix(
  config: RunConfig, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(runner, "datetime", _FixedDatetime)
  folders = [run(config, progress=lambda line: None).folder for _ in range(3)]
  base = folders[0].name
  assert [f.name for f in folders] == [base, f"{base}-2", f"{base}-3"]
  for folder in folders:
    assert (folder / "run.json").is_file() and (folder / "run.toml").is_file()
  # A run never writes into a folder that is already there.
  assert len(list(out.iterdir())) == 3


def test_suffix_skips_over_a_taken_gap(
  config: RunConfig, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(runner, "datetime", _FixedDatetime)
  first = run(config, progress=lambda line: None).folder
  (out / f"{first.name}-2").mkdir()
  assert run(config, progress=lambda line: None).folder.name == f"{first.name}-3"


def test_output_dir_that_is_a_file_is_a_run_error(
  make_config: Callable[..., RunConfig], tmp_path: Path
) -> None:
  blocker = tmp_path / "blocker"
  blocker.write_text("x", encoding="utf-8")
  with pytest.raises(RunError) as info:
    run(make_config(out=blocker), progress=lambda line: None)
  assert str(blocker) in str(info.value)


def test_output_dir_under_a_file_is_a_run_error(
  make_config: Callable[..., RunConfig], tmp_path: Path
) -> None:
  blocker = tmp_path / "blocker"
  blocker.write_text("x", encoding="utf-8")
  with pytest.raises(RunError):
    run(make_config(out=blocker / "sub"), progress=lambda line: None)


def test_failing_run_toml_write_is_a_run_error(
  config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
  def broken(self: Path, *args: Any, **kwargs: Any) -> int:
    raise PermissionError("denied")

  monkeypatch.setattr(Path, "write_text", broken)
  with pytest.raises(RunError, match="denied"):
    run(config, progress=lambda line: None)


def test_failing_record_write_stays_a_record_error(
  config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
  def broken(record: Any, path: Path) -> None:
    raise RecordError(path, "no se puede escribir")

  monkeypatch.setattr(runner, "write_record", broken)
  with pytest.raises(RecordError):
    run(config, progress=lambda line: None)


def test_files_are_written_in_the_order_run_json_run_toml_report_html(
  config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
  present: dict[str, list[str]] = {}
  original_write = Path.write_text
  original_replace = os.replace

  def spy_write(self: Path, *args: Any, **kwargs: Any) -> int:
    if self.name == "run.toml":
      present[self.name] = sorted(p.name for p in self.parent.iterdir())
    return original_write(self, *args, **kwargs)

  # The report is written through a temporary file and moved over its name: the
  # moment it appears is the ``os.replace``, whatever the temporary is called.
  def spy_replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
    if Path(dst).name == "report.html":
      present["report.html"] = sorted(
        p.name for p in Path(dst).parent.iterdir() if p != Path(src)
      )
    original_replace(src, dst, *args, **kwargs)

  monkeypatch.setattr(Path, "write_text", spy_write)
  monkeypatch.setattr(os, "replace", spy_replace)
  run(config, progress=lambda line: None)
  assert present == {
    "run.toml": ["run.json"],
    "report.html": ["run.json", "run.toml"],
  }


def test_a_failing_report_write_keeps_run_json_and_run_toml(
  config: RunConfig, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def broken(record: Any, path: Path) -> None:
    raise RecordError(path, "no se puede escribir")

  monkeypatch.setattr(runner, "write_report", broken)
  with pytest.raises(RecordError, match="report.html"):
    run(config, progress=lambda line: None)
  [folder] = list(out.iterdir())
  assert sorted(p.name for p in folder.iterdir()) == ["run.json", "run.toml"]
  assert read_record(folder / "run.json")["questions"]


def test_a_failing_run_toml_write_keeps_run_json(
  config: RunConfig, out: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  original = Path.write_text

  def selective(self: Path, *args: Any, **kwargs: Any) -> int:
    if self.name == "run.toml":
      raise PermissionError("denied")
    return original(self, *args, **kwargs)

  monkeypatch.setattr(Path, "write_text", selective)
  with pytest.raises(RunError, match="run.toml"):
    run(config, progress=lambda line: None)
  [folder] = list(out.iterdir())
  assert [p.name for p in folder.iterdir()] == ["run.json"]


def test_report_html_is_the_render_of_run_json(config: RunConfig) -> None:
  outcome = run(config, progress=lambda line: None)
  assert (outcome.folder / "report.html").read_text(encoding="utf-8") == render_run(
    read_record(outcome.folder / "run.json")
  )


def test_in_memory_run_emits_no_payload_index_warning(
  config: RunConfig, fresh_qdrant_warnings: None
) -> None:
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    run(config, progress=lambda line: None)
  assert [str(w.message) for w in caught if "Payload indexes" in str(w.message)] == []


# ---------------------------------------------------------------- run.json


def test_run_json_reads_back_equal_to_the_outcome(config: RunConfig) -> None:
  outcome = run(config, progress=lambda line: None)
  assert read_record(outcome.folder / "run.json") == outcome.record
  json.loads((outcome.folder / "run.json").read_text(encoding="utf-8"))


def test_record_header(config: RunConfig) -> None:
  record = run(config, progress=lambda line: None).record
  assert record["schema"] == 2
  assert record["title"] == "Don Quijote"
  assert record["description"] == "Batería de prueba"
  assert record["seconds"] > 0
  assert record["config"] == config.to_dict()


def test_record_questions_in_file_order_with_everything_graded(
  config: RunConfig,
) -> None:
  record = run(config, progress=lambda line: None).record
  assert [q["query_id"] for q in record["questions"]] == ["q01", "q02", "q11"]
  for question in record["questions"]:
    assert question["error"] is None
    assert question["rules"] is not None
    assert question["judge"] is not None
    assert question["judge"]["verdict"] == "partial"
    assert question["generation"] is not None


def test_record_corpus_ingest_and_collection_agree(config: RunConfig) -> None:
  record = run(config, progress=lambda line: None).record
  corpus = record["corpus"]
  assert [d["source"] for d in corpus["documents"]] == [
    "notas.md",
    "parte-1/capitulo-01.md",
    "parte-1/capitulo-02.md",
  ]
  chunks = {d["source"]: d["chunks"] for d in corpus["documents"]}
  assert all(count > 0 for count in chunks.values())
  assert chunks == {d["source"]: d["chunks"] for d in record["ingest"]["documents"]}
  assert record["collection"]["points"] == sum(chunks.values())
  assert len(record["points"]) == record["collection"]["points"]
  assert record["collection"]["url"] == ":memory:"
  assert record["collection"]["name"].startswith("hermes_report_don_quijote_")
  assert record["collection"]["vector_size"] == 256


def test_environment_lists_every_package(
  config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("OLLAMA_HOST", "http://elsewhere:1234")
  env = run(config, progress=lambda line: None).record["environment"]
  assert set(env["packages"]) == {
    "hermes-core",
    "hermes-ingest-md",
    "hermes-retrieval",
    "hermes-enrich",
    "hermes-generator",
    "hermes-report",
    "qdrant-client",
    "ollama",
  }
  assert re.fullmatch(r"\d+\.\d+\.\d+", env["python"])
  assert env["packages"]["qdrant-client"] == importlib.metadata.version("qdrant-client")
  assert env["ollama_host"] == "http://elsewhere:1234"
  assert env["qdrant_url"] == ":memory:"


def test_missing_package_version_is_a_question_mark(
  config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
  real = importlib.metadata.version

  def version(name: str) -> str:
    if name == "hermes-enrich":
      raise importlib.metadata.PackageNotFoundError(name)
    return real(name)

  monkeypatch.setattr(importlib.metadata, "version", version)
  env = run(config, progress=lambda line: None).record["environment"]
  assert env["packages"]["hermes-enrich"] == "?"
  assert env["packages"]["ollama"] != "?"


# ---------------------------------------------------------------- run.toml


def test_run_toml_loads_back_to_an_equal_config(config: RunConfig, out: Path) -> None:
  outcome = run(config, progress=lambda line: None)
  again = load_config(outcome.folder / "run.toml", Overrides(out=out))
  assert again == config


def test_run_toml_keeps_the_overrides(
  make_config: Callable[..., RunConfig], out: Path
) -> None:
  config = make_config(top_k=2, budget=1000, no_judge=True, only=("q11",))
  outcome = run(config, progress=lambda line: None)
  again = load_config(outcome.folder / "run.toml", Overrides(out=out))
  assert again.retrieval.top_k == 2
  assert again.enrich.max_context_tokens == 1000
  assert again.judge.enabled is False


# --------------------------------------------------------- progress lines


def test_progress_question_lines(config: RunConfig) -> None:
  _, lines = lines_of(config)
  question_lines = [line for line in lines if re.match(r"· q\d+ ", line)]
  assert len(question_lines) == 3
  pattern = r"· (q\d+) (.*) … (\S+) / juez: (\S+) \(\d+\.\d s\)"
  parsed = [re.fullmatch(pattern, line) for line in question_lines]
  assert all(parsed)
  assert [m.group(1) for m in parsed if m] == ["q01", "q02", "q11"]
  assert all(m.group(4) == "partial" for m in parsed if m)
  assert (
    parsed[0]
    and parsed[0].group(2) == "¿Cuántos días tardó en ponerle nombre a su rocín?"
  )
  assert parsed[0] and parsed[0].group(3) == "correct"


def test_progress_codes_match_the_recorded_rules(config: RunConfig) -> None:
  outcome, lines = lines_of(config)
  for record in outcome.record["questions"]:
    rules = record["rules"]
    assert rules is not None
    expected = f" … {rules['verdict']} / juez: "
    assert any(
      line.startswith(f"· {record['query_id']} ") and expected in line for line in lines
    )


def test_progress_question_text_is_cut_at_60_characters(
  make_config: Callable[..., RunConfig], project: Path
) -> None:
  path = project / "preguntas.json"
  data = json.loads(path.read_text(encoding="utf-8"))
  long = "¿" + "palabra " * 20 + "fin?"
  data["questions"][0]["question"] = long
  path.write_text(json.dumps(data), encoding="utf-8")
  _, lines = lines_of(make_config(only=("q01",)))
  line = next(line for line in lines if line.startswith("· q01 "))
  assert line.startswith(f"· q01 {long[:60]} … ")
  assert long[:61] not in line


def test_progress_has_corpus_ingestion_and_collection_lines(config: RunConfig) -> None:
  _, lines = lines_of(config)
  assert lines[0] == "· Corpus: 3 documentos"
  assert any("parte-1/capitulo-01.md" in line and "chunks" in line for line in lines)
  qdrant = [line for line in lines if line.startswith("· Qdrant ")]
  assert len(qdrant) == 1
  assert ":memory:" in qdrant[0] and "cargada" in qdrant[0]
  # Question lines come after the loading steps.
  assert lines.index(qdrant[0]) < next(
    i for i, line in enumerate(lines) if line.startswith("· q01 ")
  )


def test_default_progress_is_print(
  config: RunConfig, capsys: pytest.CaptureFixture[str]
) -> None:
  run(config)
  assert "· q01 " in capsys.readouterr().out


def test_no_judge_shows_a_dash(make_config: Callable[..., RunConfig]) -> None:
  outcome, lines = lines_of(make_config(no_judge=True))
  assert all(q["judge"] is None for q in outcome.record["questions"])
  assert all("juez: —" in line for line in lines if re.match(r"· q\d+ ", line))


# -------------------------------------------------------------- --only


def test_only_selects_and_keeps_file_order(
  make_config: Callable[..., RunConfig],
) -> None:
  outcome, lines = lines_of(make_config(only=("q11", "q01")))
  assert [q["query_id"] for q in outcome.record["questions"]] == ["q01", "q11"]
  assert [line[2:5] for line in lines if re.match(r"· q\d+ ", line)] == ["q01", "q11"]


def test_only_with_repeated_id_runs_it_once(
  make_config: Callable[..., RunConfig],
) -> None:
  outcome = run(make_config(only=("q02", "q02")), progress=lambda line: None)
  assert [q["query_id"] for q in outcome.record["questions"]] == ["q02"]


def test_only_unknown_id_is_a_run_error_listing_it(
  make_config: Callable[..., RunConfig],
  out: Path,
  forbid_ollama: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def no_ingest(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("must fail before ingesting")

  monkeypatch.setattr(runner, "ingest_corpus", no_ingest)
  with pytest.raises(RunError) as info:
    run(make_config(only=("q01", "zz", "yy", "zz")), progress=lambda line: None)
  message = str(info.value)
  assert "zz" in message and "yy" in message
  assert message.count("zz") == 1
  assert "q11" in message  # the ids that do exist
  assert not out.exists()


def test_only_matching_is_exact_not_by_prefix(
  make_config: Callable[..., RunConfig],
) -> None:
  with pytest.raises(RunError, match="q0"):
    run(make_config(only=("q0",)), progress=lambda line: None)


def test_only_ids_are_case_sensitive(make_config: Callable[..., RunConfig]) -> None:
  with pytest.raises(RunError, match="Q01"):
    run(make_config(only=("Q01",)), progress=lambda line: None)


# --------------------------------------------------------- check_ollama


@pytest.fixture
def ollama_run(project: Path, out: Path) -> Callable[..., RunConfig]:
  """A run file whose three providers are ``ollama``, with the given models."""

  def make(
    embedding: str = "emb",
    generation: str = "gen",
    judge: str = "jud",
    **overrides: Any,
  ) -> RunConfig:
    text = (DATA / "run.toml").read_text(encoding="utf-8")
    text = text.replace(
      'provider = "hash"\nmodel = "hash-256"',
      f'provider = "ollama"\nmodel = "{embedding}"',
    )
    text = text.replace(
      'provider = "echo"\nmodel = "echo"',
      f'provider = "ollama"\nmodel = "{generation}"',
    )
    text = text.replace(
      'provider = "echo"\nmodel = ""', f'provider = "ollama"\nmodel = "{judge}"'
    )
    (project / "run.toml").write_text(text, encoding="utf-8")
    overrides.setdefault("out", out)
    return load_config(project / "run.toml", Overrides(**overrides))

  return make


@pytest.fixture
def offline_factories(monkeypatch: pytest.MonkeyPatch) -> list[str]:
  """Replace the provider factories with offline ones; log their order."""
  order: list[str] = []

  def embedder(cfg: Any) -> HashEmbedder:
    order.append("make_embedder")
    return HashEmbedder(256)

  def llm(cfg: Any) -> EchoLLM:
    order.append("make_llm")
    return EchoLLM()

  def judge(cfg: Any) -> EchoJudgeClient:
    order.append("make_judge")
    return EchoJudgeClient()

  monkeypatch.setattr(runner, "make_embedder", embedder)
  monkeypatch.setattr(runner, "make_llm", llm)
  monkeypatch.setattr(runner, "make_judge", judge)
  return order


def record_calls(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> list[list[str]]:
  calls: list[list[str]] = []

  def check(models: Sequence[str], *args: Any, **kwargs: Any) -> None:
    order.append("check_ollama")
    calls.append(list(models))

  monkeypatch.setattr(runner, "check_ollama", check)
  return calls


def test_check_ollama_gets_every_model_once_in_one_call(
  ollama_run: Callable[..., RunConfig],
  offline_factories: list[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = record_calls(monkeypatch, offline_factories)
  run(ollama_run(), progress=lambda line: None)
  assert calls == [["emb", "gen", "jud"]]


def test_check_ollama_runs_before_anything_is_built_or_ingested(
  ollama_run: Callable[..., RunConfig],
  offline_factories: list[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  record_calls(monkeypatch, offline_factories)
  original = ingest_corpus

  def ingest(*args: Any, **kwargs: Any) -> Any:
    offline_factories.append("ingest")
    return original(*args, **kwargs)

  monkeypatch.setattr(runner, "ingest_corpus", ingest)
  run(ollama_run(), progress=lambda line: None)
  assert offline_factories[0] == "check_ollama"
  assert offline_factories.index("ingest") > offline_factories.index("make_judge")


def test_check_ollama_failure_stops_the_run_before_ingestion(
  ollama_run: Callable[..., RunConfig],
  offline_factories: list[str],
  monkeypatch: pytest.MonkeyPatch,
  out: Path,
) -> None:
  def check(models: Sequence[str], *args: Any, **kwargs: Any) -> None:
    raise ProviderError("el modelo gen no está descargado")

  def no_ingest(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("ingestion must not start")

  monkeypatch.setattr(runner, "check_ollama", check)
  monkeypatch.setattr(runner, "ingest_corpus", no_ingest)
  with pytest.raises(ProviderError, match="gen"):
    run(ollama_run(), progress=lambda line: None)
  assert offline_factories == []
  assert not out.exists()


def test_check_ollama_deduplicates_repeated_models(
  ollama_run: Callable[..., RunConfig],
  offline_factories: list[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = record_calls(monkeypatch, offline_factories)
  run(ollama_run("m", "m", "m"), progress=lambda line: None)
  assert calls == [["m"]]
  calls.clear()
  run(ollama_run("e", "g", "e"), progress=lambda line: None)
  assert calls == [["e", "g"]]


def test_check_ollama_skips_the_judge_model_without_judge(
  ollama_run: Callable[..., RunConfig],
  offline_factories: list[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = record_calls(monkeypatch, offline_factories)
  run(ollama_run(no_judge=True), progress=lambda line: None)
  assert calls == [["emb", "gen"]]
  assert "make_judge" not in offline_factories


def test_check_ollama_only_lists_ollama_providers(
  project: Path,
  out: Path,
  offline_factories: list[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  text = (project / "run.toml").read_text(encoding="utf-8")
  # Only the judge is on Ollama; embedding stays hash and generation echo.
  text = text.replace(
    'provider = "echo"\nmodel = ""', 'provider = "ollama"\nmodel = "jud"'
  )
  (project / "run.toml").write_text(text, encoding="utf-8")
  calls = record_calls(monkeypatch, offline_factories)
  run(load_config(project / "run.toml", Overrides(out=out)), progress=lambda line: None)
  assert calls == [["jud"]]


def test_check_ollama_sees_the_model_overrides(
  ollama_run: Callable[..., RunConfig],
  offline_factories: list[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = record_calls(monkeypatch, offline_factories)
  run(
    ollama_run(model="other-gen", judge_model="other-jud"), progress=lambda line: None
  )
  assert calls == [["emb", "other-gen", "other-jud"]]


def test_offline_run_never_calls_check_ollama(
  config: RunConfig, forbid_ollama: list[list[str]]
) -> None:
  run(config, progress=lambda line: None)
  assert forbid_ollama == []


# ------------------------------------------------------- failing questions


class FlakyLLM:
  """Fails on questions that mention one of ``broken``; echoes otherwise."""

  model = "flaky"

  def __init__(self, broken: tuple[str, ...]) -> None:
    self.broken = broken
    self.echo = EchoLLM()

  def complete(
    self, messages: Sequence[PromptMessage], params: GenerationParams
  ) -> LLMReply:
    if any(word in messages[-1].content for word in self.broken):
      raise RuntimeError("model down")
    return self.echo.complete(messages, params)


class CountingJudge:
  model = "counting"

  def __init__(self) -> None:
    self.calls = 0

  def complete_json(self, messages: Sequence[PromptMessage]) -> str:
    self.calls += 1
    return '{"verdict": "correct", "reason": "bien"}'


def test_failing_questions_are_counted_and_do_not_stop_the_run(
  make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
  # Metrics off: the counting judge counts only the verdict calls.
  config = make_config(no_metrics=True)
  judge = CountingJudge()
  monkeypatch.setattr(runner, "make_llm", lambda cfg: FlakyLLM(("rocín", "Quijote")))
  monkeypatch.setattr(runner, "make_judge", lambda cfg: judge)
  outcome, lines = lines_of(config)
  by_id = {q["query_id"]: q for q in outcome.record["questions"]}
  assert outcome.failed == sum(1 for q in by_id.values() if q["error"] is not None)
  assert outcome.failed >= 1
  assert len(by_id) == 3
  # Judge only for the questions that have a result and no error.
  graded = [q for q in by_id.values() if q["error"] is None]
  assert judge.calls == len(graded)
  for question in by_id.values():
    assert question["rules"] is not None  # computed for every record
    if question["error"] is not None:
      assert question["judge"] is None
  assert (outcome.folder / "run.json").is_file()
  failed_lines = [
    line for line in lines if re.match(r"· q\d+ ", line) and "juez: —" in line
  ]
  assert len(failed_lines) == outcome.failed


def test_every_question_failing_counts_them_all(
  make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
  config = make_config(no_metrics=True)
  judge = CountingJudge()
  monkeypatch.setattr(runner, "make_llm", lambda cfg: FlakyLLM(("",)))
  monkeypatch.setattr(runner, "make_judge", lambda cfg: judge)
  outcome = run(config, progress=lambda line: None)
  # q11 is a refusal with no sources: the echo LLM is not even asked.
  by_id = {q["query_id"]: q for q in outcome.record["questions"]}
  assert outcome.failed == 2
  assert by_id["q01"]["error"] is not None and by_id["q02"]["error"] is not None
  assert by_id["q11"]["error"] is None
  assert judge.calls == 1
  assert read_record(outcome.folder / "run.json")["questions"][0]["error"] is not None


def test_judge_exception_is_recorded_not_raised(
  config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
  def boom(question: Any, record: Any, client: Any) -> None:
    raise RuntimeError("kaput")

  monkeypatch.setattr(runner, "judge", boom)
  outcome, lines = lines_of(config)
  for question in outcome.record["questions"]:
    judged = question["judge"]
    assert judged is not None
    assert judged["error"] == "RuntimeError: kaput"
    assert judged["verdict"] is None and judged["reason"] is None
    assert judged["raw"] is None
    assert judged["model"] == "echo"
    assert judged["seconds"] >= 0
    assert [m["role"] for m in judged["messages"]] == ["system", "user"]
  # A judge failure is not a question failure.
  assert outcome.failed == 0
  assert all("juez: —" in line for line in lines if re.match(r"· q\d+ ", line))
  read_record(outcome.folder / "run.json")


def test_judge_provider_error_escaping_judge_propagates(
  config: RunConfig, monkeypatch: pytest.MonkeyPatch, out: Path
) -> None:
  def down(question: Any, record: Any, client: Any) -> None:
    raise ProviderError("ollama caído")

  monkeypatch.setattr(runner, "judge", down)
  with pytest.raises(ProviderError):
    run(config, progress=lambda line: None)
  assert not out.exists()


def test_invalid_judge_reply_is_recorded_and_not_a_failure(
  config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
  class Garbage:
    model = "garbage"

    def complete_json(self, messages: Sequence[PromptMessage]) -> str:
      return "not json"

  monkeypatch.setattr(runner, "make_judge", lambda cfg: Garbage())
  outcome = run(config, progress=lambda line: None)
  assert outcome.failed == 0
  for question in outcome.record["questions"]:
    assert question["judge"] is not None
    assert question["judge"]["verdict"] is None
    assert question["judge"]["raw"] == "not json"
    assert question["judge"]["error"]


def test_judge_is_not_built_without_judge(
  make_config: Callable[..., RunConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
  def refuse(cfg: Any) -> None:
    raise AssertionError("no judge expected")

  monkeypatch.setattr(runner, "make_judge", refuse)
  assert run(make_config(no_judge=True), progress=lambda line: None).failed == 0


# ------------------------------------------------- propagated errors, client


class SpyClient:
  """Wraps the real in-memory client and records ``close``."""

  def __init__(self) -> None:
    self.inner = QdrantClient(":memory:")
    self.closed = 0

  def close(self) -> None:
    self.closed += 1
    self.inner.close()

  def __getattr__(self, name: str) -> Any:
    return getattr(self.inner, name)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> SpyClient:
  client = SpyClient()
  monkeypatch.setattr(runner, "make_client", lambda url: client)
  return client


def test_client_is_closed_after_a_good_run(config: RunConfig, spy: SpyClient) -> None:
  run(config, progress=lambda line: None)
  assert spy.closed == 1


@pytest.mark.parametrize("stage", ["ensure_collection", "read_points", "build_context"])
def test_client_is_closed_when_a_stage_raises(
  config: RunConfig, spy: SpyClient, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
  def boom(*args: Any, **kwargs: Any) -> None:
    raise StoreError("qdrant caído")

  monkeypatch.setattr(runner, stage, boom)
  with pytest.raises(StoreError):
    run(config, progress=lambda line: None)
  assert spy.closed == 1


def test_client_is_closed_when_the_progress_callback_raises(
  config: RunConfig, spy: SpyClient
) -> None:
  def angry(line: str) -> None:
    if line.startswith("· q01"):
      raise KeyboardInterrupt

  with pytest.raises(KeyboardInterrupt):
    run(config, progress=angry)
  assert spy.closed == 1


@pytest.mark.parametrize(
  ("target", "error"),
  [
    ("load_corpus", CorpusError(Path("c"), "no hay documentos")),
    ("load_questions", QuestionsError(Path("q"), "mal formato")),
    ("ingest_corpus", IngestError("fallo al ingerir")),
    ("make_client", StoreError("sin conexión")),
    ("make_embedder", ProviderError("sin embedder")),
  ],
)
def test_expected_errors_propagate_unchanged(
  config: RunConfig,
  monkeypatch: pytest.MonkeyPatch,
  out: Path,
  target: str,
  error: Exception,
) -> None:
  def boom(*args: Any, **kwargs: Any) -> None:
    raise error

  monkeypatch.setattr(runner, target, boom)
  with pytest.raises(type(error)) as info:
    run(config, progress=lambda line: None)
  assert info.value is error
  assert not out.exists()  # a failed run leaves no folder behind


def test_questions_naming_a_missing_document_is_a_questions_error(
  make_config: Callable[..., RunConfig], project: Path
) -> None:
  path = project / "preguntas.json"
  data = json.loads(path.read_text(encoding="utf-8"))
  data["questions"][0]["documents"] = ["no-existe.md"]
  path.write_text(json.dumps(data), encoding="utf-8")
  with pytest.raises(QuestionsError):
    run(make_config(), progress=lambda line: None)


def test_corpus_removed_after_loading_the_config_is_a_corpus_error(
  config: RunConfig, project: Path
) -> None:
  import shutil

  shutil.rmtree(project / "corpus")
  with pytest.raises(CorpusError):
    run(config, progress=lambda line: None)


def test_empty_corpus_is_an_error_not_a_crash(config: RunConfig, project: Path) -> None:
  for md in (project / "corpus").rglob("*.md"):
    md.unlink()
  with pytest.raises((CorpusError, QuestionsError)):
    run(config, progress=lambda line: None)


def test_bad_config_error_type_is_not_swallowed(project: Path) -> None:
  with pytest.raises(ConfigError):
    load_config(project / "missing.toml")


# ------------------------------------------------------------ re-ingestion


def test_second_run_uses_the_ingestion_cache_and_reingest_ignores_it(
  make_config: Callable[..., RunConfig],
) -> None:
  first = run(make_config(), progress=lambda line: None).record
  second = run(make_config(), progress=lambda line: None).record
  third = run(make_config(reingest=True), progress=lambda line: None).record
  assert not any(d["cached"] for d in first["ingest"]["documents"])
  assert all(d["cached"] for d in second["ingest"]["documents"])
  assert not any(d["cached"] for d in third["ingest"]["documents"])
  assert first["collection"]["name"] == second["collection"]["name"]


def test_run_does_not_mutate_the_config(config: RunConfig) -> None:
  before = config.to_toml()
  run(config, progress=lambda line: None)
  assert config.to_toml() == before

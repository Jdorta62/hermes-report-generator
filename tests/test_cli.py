"""Tests for ``hermes_report.cli`` and ``python -m hermes_report``.

Real runs use the sample project (in-memory Qdrant, hash embedder, echo LLM and
judge). Tests of the flag mapping and of the error mapping replace ``cli.run`` so
they do not depend on the pipeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from conftest import DATA
from hermes_core.generation import PromptMessage
from hermes_generator import GenerationParams, LLMReply

from hermes_report import cli, runner
from hermes_report.cli import main
from hermes_report.config import ConfigError, RunConfig
from hermes_report.corpus import CorpusError
from hermes_report.ingest import IngestError
from hermes_report.providers import ProviderError
from hermes_report.questions import QuestionsError
from hermes_report.record import RecordError, read_record
from hermes_report.render import render_compare
from hermes_report.runner import RunError, RunOutcome
from hermes_report.store import StoreError

ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def quiet_qdrant() -> Iterator[None]:
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    yield


@pytest.fixture
def run_toml(project: Path) -> Path:
  return project / "run.toml"


@pytest.fixture
def out(tmp_path: Path) -> Path:
  return tmp_path / "out"


@pytest.fixture
def forbid_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
  def refuse(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("an offline run must not call check_ollama")

  monkeypatch.setattr(runner, "check_ollama", refuse)


class Captured:
  """The configs that ``cli.run`` received."""

  def __init__(self, folder: Path) -> None:
    self.configs: list[RunConfig] = []
    self.folder = folder
    self.failed = 0


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Captured:
  captured = Captured(tmp_path / "fake-folder")

  def fake(config: RunConfig, *args: Any, **kwargs: Any) -> RunOutcome:
    captured.configs.append(config)
    return RunOutcome(record={}, folder=captured.folder, failed=captured.failed)  # type: ignore[typeddict-item]

  monkeypatch.setattr(cli, "run", fake)
  return captured


# ------------------------------------------------------------ real runs


def test_run_prints_folder_and_record_and_returns_0(
  run_toml: Path,
  out: Path,
  capsys: pytest.CaptureFixture[str],
  forbid_ollama: None,
) -> None:
  code = main(["run", str(run_toml), "--out", str(out)])
  captured = capsys.readouterr()
  assert code == 0
  assert captured.err == ""
  [folder] = list(out.iterdir())
  assert f"Carpeta: {folder}" in captured.out.splitlines()
  assert f"Registro: {folder / 'run.json'}" in captured.out.splitlines()
  assert captured.out.splitlines()[-3:] == [
    f"Carpeta: {folder}",
    f"Registro: {folder / 'run.json'}",
    f"Informe: {folder / 'report.html'}",
  ]
  assert (folder / "report.html").is_file()
  assert [q["query_id"] for q in read_record(folder / "run.json")["questions"]] == [
    "q01",
    "q02",
    "q11",
  ]
  assert "· q01 " in captured.out


def test_only_flags_run_just_those_questions(
  run_toml: Path, out: Path, forbid_ollama: None
) -> None:
  code = main(
    ["run", str(run_toml), "--out", str(out), "--only", "q11", "--only", "q01"]
  )
  assert code == 0
  [folder] = list(out.iterdir())
  record = read_record(folder / "run.json")
  assert [q["query_id"] for q in record["questions"]] == ["q01", "q11"]


def test_no_judge_and_top_k_and_budget_reach_the_record(
  run_toml: Path, out: Path, forbid_ollama: None
) -> None:
  code = main(
    [
      "run",
      str(run_toml),
      "--out",
      str(out),
      "--no-judge",
      "--top-k",
      "2",
      "--budget",
      "900",
    ]
  )
  assert code == 0
  [folder] = list(out.iterdir())
  record = read_record(folder / "run.json")
  assert all(q["judge"] is None for q in record["questions"])
  assert record["config"]["retrieval"]["top_k"] == 2
  assert record["config"]["enrich"]["max_context_tokens"] == 900
  assert record["config"]["judge"]["enabled"] is False


def test_out_dot_is_accepted_and_means_the_current_directory(
  run_toml: Path,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  forbid_ollama: None,
) -> None:
  here = tmp_path / "here"
  here.mkdir()
  monkeypatch.chdir(here)
  assert main(["run", str(run_toml), "--out", "."]) == 0
  [folder] = list(here.iterdir())
  assert (folder / "run.json").is_file()


def test_out_default_comes_from_the_run_file(
  run_toml: Path, forbid_ollama: None
) -> None:
  assert main(["run", str(run_toml)]) == 0
  [folder] = list((run_toml.parent / "out").iterdir())
  assert (folder / "run.toml").is_file()


def test_reingest_flag_ignores_the_cache(
  run_toml: Path, out: Path, forbid_ollama: None
) -> None:
  assert main(["run", str(run_toml), "--out", str(out)]) == 0
  assert main(["run", str(run_toml), "--out", str(out)]) == 0
  assert main(["run", str(run_toml), "--out", str(out), "--reingest"]) == 0
  cached = [
    all(d["cached"] for d in read_record(f / "run.json")["ingest"]["documents"])
    for f in sorted(out.iterdir())
  ]
  assert sorted(cached) == [False, False, True]


def test_argv_defaults_to_sys_argv(
  run_toml: Path,
  out: Path,
  monkeypatch: pytest.MonkeyPatch,
  forbid_ollama: None,
) -> None:
  monkeypatch.setattr(
    sys, "argv", ["hermes-report", "run", str(run_toml), "--out", str(out)]
  )
  assert main() == 0
  assert len(list(out.iterdir())) == 1


# ------------------------------------------------------------ exit code 1


class Failing:
  model = "failing"

  def complete(
    self, messages: Sequence[PromptMessage], params: GenerationParams
  ) -> LLMReply:
    raise RuntimeError("model down")


def test_failed_question_returns_1_but_still_reports_the_folder(
  run_toml: Path,
  out: Path,
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  forbid_ollama: None,
) -> None:
  monkeypatch.setattr(runner, "make_llm", lambda cfg: Failing())
  code = main(["run", str(run_toml), "--out", str(out)])
  captured = capsys.readouterr()
  assert code == 1
  [folder] = list(out.iterdir())
  assert f"Carpeta: {folder}" in captured.out
  assert (folder / "run.json").is_file()
  assert "2 pregunta(s) con error" in captured.err


def test_judge_failure_alone_does_not_change_the_exit_code(
  run_toml: Path,
  out: Path,
  monkeypatch: pytest.MonkeyPatch,
  forbid_ollama: None,
) -> None:
  def boom(question: Any, record: Any, client: Any) -> None:
    raise RuntimeError("judge down")

  monkeypatch.setattr(runner, "judge", boom)
  assert main(["run", str(run_toml), "--out", str(out)]) == 0


def test_exit_1_when_outcome_reports_failures(
  fake_run: Captured, run_toml: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  fake_run.failed = 1
  assert main(["run", str(run_toml)]) == 1
  assert "1 pregunta(s) con error" in capsys.readouterr().err


# ------------------------------------------------------------ exit code 2

EXPECTED_ERRORS = [
  ConfigError(Path("run.toml"), "falta el título", "title"),
  QuestionsError(Path("q.json"), "formato inválido"),
  CorpusError(Path("corpus"), "sin documentos"),
  ProviderError("Ollama no responde"),
  IngestError("no se puede ingerir"),
  StoreError("Qdrant no responde"),
  RecordError(Path("run.json"), "no se puede escribir"),
  RunError("pregunta(s) desconocida(s)"),
]


@pytest.mark.parametrize("error", EXPECTED_ERRORS, ids=lambda e: type(e).__name__)
def test_expected_errors_print_and_return_2(
  error: Exception,
  run_toml: Path,
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def boom(*args: Any, **kwargs: Any) -> None:
    raise error

  monkeypatch.setattr(cli, "run", boom)
  code = main(["run", str(run_toml)])
  captured = capsys.readouterr()
  assert code == 2
  assert captured.out == ""
  assert captured.err == f"error: {error}\n"


def test_unexpected_exceptions_are_not_swallowed(
  run_toml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def boom(*args: Any, **kwargs: Any) -> None:
    raise ZeroDivisionError("bug")

  monkeypatch.setattr(cli, "run", boom)
  with pytest.raises(ZeroDivisionError):
    main(["run", str(run_toml)])


def test_missing_run_file_returns_2(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  missing = tmp_path / "nope.toml"
  assert main(["run", str(missing)]) == 2
  err = capsys.readouterr().err
  assert err.startswith("error: ")
  assert str(missing) in err


def test_run_file_that_is_not_toml_returns_2(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  bad = tmp_path / "bad.toml"
  bad.write_text("title = [", encoding="utf-8")
  assert main(["run", str(bad)]) == 2
  assert capsys.readouterr().err.startswith("error: ")


def test_run_file_that_is_a_directory_returns_2(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(["run", str(tmp_path)]) == 2
  assert capsys.readouterr().err.startswith("error: ")


def test_unknown_only_id_returns_2_and_lists_it(
  run_toml: Path,
  out: Path,
  capsys: pytest.CaptureFixture[str],
  forbid_ollama: None,
) -> None:
  assert (
    main(["run", str(run_toml), "--out", str(out), "--only", "q01", "--only", "zz"])
    == 2
  )
  captured = capsys.readouterr()
  assert captured.err.startswith("error: ")
  assert "zz" in captured.err
  assert "Carpeta" not in captured.out
  assert not out.exists()


@pytest.mark.parametrize(
  ("flag", "value"), [("--top-k", "0"), ("--top-k", "-1"), ("--budget", "-1")]
)
def test_non_positive_numbers_return_2_not_a_traceback(
  run_toml: Path,
  capsys: pytest.CaptureFixture[str],
  forbid_ollama: None,
  flag: str,
  value: str,
) -> None:
  assert main(["run", str(run_toml), flag, value]) == 2
  assert capsys.readouterr().err != ""


def test_budget_zero_means_no_limit_and_is_accepted(
  run_toml: Path, out: Path, forbid_ollama: None
) -> None:
  assert main(["run", str(run_toml), "--out", str(out), "--budget", "0"]) == 0


def test_blank_model_returns_2(
  run_toml: Path, capsys: pytest.CaptureFixture[str], forbid_ollama: None
) -> None:
  assert main(["run", str(run_toml), "--model", "   "]) == 2
  assert capsys.readouterr().err.startswith("error: ")


def test_unreachable_ollama_returns_2_before_ingestion(
  project: Path,
  out: Path,
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  text = (DATA / "run.toml").read_text(encoding="utf-8")
  text = text.replace(
    'provider = "echo"\nmodel = "echo"', 'provider = "ollama"\nmodel = "gen"'
  )
  (project / "run.toml").write_text(text, encoding="utf-8")

  def check(models: Sequence[str], *args: Any, **kwargs: Any) -> None:
    assert list(models) == ["gen"]
    raise ProviderError("Ollama no responde en http://localhost:11434")

  def no_ingest(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("ingestion must not start")

  monkeypatch.setattr(runner, "check_ollama", check)
  monkeypatch.setattr(runner, "ingest_corpus", no_ingest)
  assert main(["run", str(project / "run.toml"), "--out", str(out)]) == 2
  assert "Ollama no responde" in capsys.readouterr().err
  assert not out.exists()


def test_check_ollama_receives_the_model_flags(
  project: Path,
  out: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  text = (DATA / "run.toml").read_text(encoding="utf-8")
  text = text.replace(
    'provider = "echo"\nmodel = "echo"', 'provider = "ollama"\nmodel = "gen"'
  ).replace('provider = "echo"\nmodel = ""', 'provider = "ollama"\nmodel = "jud"')
  (project / "run.toml").write_text(text, encoding="utf-8")
  seen: list[list[str]] = []

  def check(models: Sequence[str], *args: Any, **kwargs: Any) -> None:
    seen.append(list(models))
    raise ProviderError("stop")

  monkeypatch.setattr(runner, "check_ollama", check)
  args = ["run", str(project / "run.toml"), "--out", str(out)]
  assert main([*args, "--model", "M", "--judge-model", "J"]) == 2
  assert main([*args, "--model", "M", "--no-judge"]) == 2
  assert seen == [["M", "J"], ["M"]]


# -------------------------------------------------------- argument handling


def test_empty_out_is_rejected_before_running(
  run_toml: Path, capsys: pytest.CaptureFixture[str], fake_run: Captured
) -> None:
  assert main(["run", str(run_toml), "--out", ""]) == 2
  err = capsys.readouterr().err
  assert "--out" in err
  assert fake_run.configs == []  # run() was never reached


def test_blank_out_is_rejected_too(
  run_toml: Path, capsys: pytest.CaptureFixture[str], fake_run: Captured
) -> None:
  # Not an empty string, so argparse accepts it; load_config rejects it (exit 2).
  assert main(["run", str(run_toml), "--out", "   "]) == 2
  assert fake_run.configs == []
  assert capsys.readouterr().err.startswith("error: ")


def test_no_subcommand_prints_help_and_returns_2(
  capsys: pytest.CaptureFixture[str],
) -> None:
  assert main([]) == 2
  out = capsys.readouterr().out
  assert "usage: hermes-report" in out
  assert "run" in out


@pytest.mark.parametrize(
  "argv",
  [
    ["run"],
    ["bogus"],
    ["run", "x.toml", "--unknown"],
    ["run", "x.toml", "--top-k", "abc"],
    ["run", "x.toml", "--top-k", "1.5"],
    ["run", "x.toml", "--budget", ""],
    ["run", "x.toml", "--model"],
    ["run", "x.toml", "--only"],
  ],
)
def test_usage_errors_return_2(
  argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(argv) == 2
  assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--help"], ["-h"], ["run", "--help"]])
def test_help_returns_0_and_is_in_spanish(
  argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(argv) == 0
  out = capsys.readouterr().out
  assert "hermes-report" in out
  if argv[0] == "run":
    for flag in (
      "--model",
      "--judge-model",
      "--no-judge",
      "--top-k",
      "--budget",
      "--out",
      "--only",
      "--reingest",
    ):
      assert flag in out
    assert "ejecuta" in out.lower()


def test_main_always_returns_an_int(capsys: pytest.CaptureFixture[str]) -> None:
  for argv in ([], ["--help"], ["nope"]):
    code = main(argv)
    assert type(code) is int
  capsys.readouterr()


def test_flags_map_to_overrides(
  fake_run: Captured, run_toml: Path, tmp_path: Path
) -> None:
  target = tmp_path / "target"
  code = main(
    [
      "run",
      str(run_toml),
      "--model",
      "M",
      "--judge-model",
      "J",
      "--top-k",
      "7",
      "--budget",
      "1234",
      "--out",
      str(target),
      "--only",
      "q01",
      "--only",
      "q11",
      "--reingest",
    ]
  )
  assert code == 0
  [config] = fake_run.configs
  assert config.generation.model == "M"
  assert config.judge.model == "J"
  assert config.retrieval.top_k == 7
  assert config.enrich.max_context_tokens == 1234
  assert config.output.dir == target.resolve()
  assert config.only == ("q01", "q11")
  assert config.reingest is True
  assert config.judge.enabled is True


def test_defaults_map_to_no_overrides(fake_run: Captured, run_toml: Path) -> None:
  assert main(["run", str(run_toml)]) == 0
  [config] = fake_run.configs
  assert config.only == ()
  assert config.reingest is False
  assert config.judge.enabled is True
  assert config.generation.model == "echo"
  assert config.retrieval.top_k == 5


def test_no_judge_flag_disables_the_judge(fake_run: Captured, run_toml: Path) -> None:
  assert main(["run", str(run_toml), "--no-judge"]) == 0
  assert fake_run.configs[0].judge.enabled is False


def test_judge_model_with_no_judge_is_still_accepted(
  fake_run: Captured, run_toml: Path
) -> None:
  assert main(["run", str(run_toml), "--no-judge", "--judge-model", "J"]) == 0
  [config] = fake_run.configs
  assert config.judge.enabled is False
  assert config.judge.model == "J"


# ------------------------------------------------------ entry points


def test_pyproject_declares_the_console_script() -> None:
  data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
  assert data["project"]["scripts"] == {"hermes-report": "hermes_report.cli:main"}


def test_python_dash_m_runs_the_command(project: Path, out: Path) -> None:
  result = subprocess.run(
    [
      sys.executable,
      "-W",
      "ignore",
      "-m",
      "hermes_report",
      "run",
      str(project / "run.toml"),
      "--out",
      str(out),
      "--only",
      "q01",
    ],
    capture_output=True,
    text=True,
    check=False,
    timeout=120,
  )
  assert result.returncode == 0, result.stderr
  [folder] = list(out.iterdir())
  assert f"Carpeta: {folder}" in result.stdout
  assert (folder / "run.json").is_file()


def test_python_dash_m_propagates_the_exit_code(project: Path) -> None:
  result = subprocess.run(
    [sys.executable, "-m", "hermes_report", "run", str(project / "missing.toml")],
    capture_output=True,
    text=True,
    check=False,
    timeout=120,
  )
  assert result.returncode == 2
  assert result.stderr.startswith("error: ")


def test_python_dash_m_without_arguments_returns_2() -> None:
  result = subprocess.run(
    [sys.executable, "-m", "hermes_report"],
    capture_output=True,
    text=True,
    check=False,
    timeout=120,
  )
  assert result.returncode == 2
  assert "usage: hermes-report" in result.stdout


def test_importing_main_module_does_not_run_anything() -> None:
  import hermes_report.__main__ as entry  # noqa: F401


def test_console_script_is_installed(project: Path, out: Path) -> None:
  script = Path(sys.executable).parent / "hermes-report"
  if not script.exists():
    pytest.skip("console script not installed in this environment")
  result = subprocess.run(
    [str(script), "run", str(project / "run.toml"), "--out", str(out), "--no-judge"],
    capture_output=True,
    text=True,
    check=False,
    timeout=120,
  )
  assert result.returncode == 0, result.stderr


# ------------------------------------------------------------ render


@pytest.fixture
def run_json(run_toml: Path, out: Path, forbid_ollama: None) -> Path:
  """The ``run.json`` of a real offline run, with its ``report.html`` removed."""
  assert main(["run", str(run_toml), "--out", str(out)]) == 0
  [folder] = list(out.iterdir())
  (folder / "report.html").unlink()
  return folder / "run.json"


def test_run_writes_report_html_into_the_run_folder(
  run_toml: Path, out: Path, forbid_ollama: None
) -> None:
  assert main(["run", str(run_toml), "--out", str(out)]) == 0
  [folder] = list(out.iterdir())
  assert sorted(p.name for p in folder.iterdir()) == [
    "report.html",
    "run.json",
    "run.toml",
  ]
  text = (folder / "report.html").read_text(encoding="utf-8")
  assert text.startswith("<!doctype html>")
  assert 'data-key="q-q01"' in text


def test_run_with_failed_questions_still_writes_the_report(
  run_toml: Path,
  out: Path,
  monkeypatch: pytest.MonkeyPatch,
  forbid_ollama: None,
) -> None:
  monkeypatch.setattr(runner, "make_llm", lambda cfg: Failing())
  assert main(["run", str(run_toml), "--out", str(out)]) == 1
  [folder] = list(out.iterdir())
  assert "model down" in (folder / "report.html").read_text(encoding="utf-8")


def test_run_that_cannot_write_the_report_returns_2(
  run_toml: Path,
  out: Path,
  capsys: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  forbid_ollama: None,
) -> None:
  def broken(record: Any, path: Path) -> None:
    raise RecordError(path, "no se puede escribir: disco lleno")

  monkeypatch.setattr(runner, "write_report", broken)
  assert main(["run", str(run_toml), "--out", str(out)]) == 2
  assert "disco lleno" in capsys.readouterr().err
  [folder] = list(out.iterdir())
  assert (folder / "run.json").is_file()  # the results are not lost


def test_render_writes_report_html_next_to_run_json(
  run_json: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(["render", str(run_json)]) == 0
  captured = capsys.readouterr()
  target = run_json.parent / "report.html"
  assert captured.out == f"Informe: {target}\n"
  assert captured.err == ""
  assert target.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_render_output_equals_the_report_of_the_run(
  run_toml: Path, out: Path, forbid_ollama: None
) -> None:
  assert main(["run", str(run_toml), "--out", str(out)]) == 0
  [folder] = list(out.iterdir())
  original = (folder / "report.html").read_bytes()
  (folder / "report.html").unlink()
  assert main(["render", str(folder / "run.json")]) == 0
  assert (folder / "report.html").read_bytes() == original


def test_render_out_writes_elsewhere_and_creates_folders(
  run_json: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  target = tmp_path / "x" / "y" / "informe.html"
  assert main(["render", str(run_json), "--out", str(target)]) == 0
  assert capsys.readouterr().out == f"Informe: {target}\n"
  assert target.is_file()
  assert not (run_json.parent / "report.html").exists()


def test_render_overwrites_an_existing_report(run_json: Path) -> None:
  target = run_json.parent / "report.html"
  target.write_text("old", encoding="utf-8")
  assert main(["render", str(run_json)]) == 0
  assert target.read_text(encoding="utf-8") != "old"


def test_render_twice_gives_the_same_bytes(run_json: Path, tmp_path: Path) -> None:
  first, second = tmp_path / "1.html", tmp_path / "2.html"
  assert main(["render", str(run_json), "--out", str(first)]) == 0
  assert main(["render", str(run_json), "--out", str(second)]) == 0
  assert first.read_bytes() == second.read_bytes()


def test_render_does_not_touch_run_json_or_run_toml(run_json: Path) -> None:
  before = {p.name: p.read_bytes() for p in run_json.parent.iterdir()}
  assert main(["render", str(run_json)]) == 0
  after = {p.name: p.read_bytes() for p in run_json.parent.iterdir()}
  assert after.pop("report.html")
  assert after == before


def test_render_works_with_a_relative_path(
  run_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.chdir(run_json.parent)
  assert main(["render", "run.json"]) == 0
  assert (run_json.parent / "report.html").is_file()


def test_render_missing_file_returns_2(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  missing = tmp_path / "nope.json"
  assert main(["render", str(missing)]) == 2
  captured = capsys.readouterr()
  assert captured.out == ""
  assert captured.err == f"error: {missing}: el archivo no existe\n"
  assert not (tmp_path / "report.html").exists()


@pytest.mark.parametrize(
  ("content", "fragment"),
  [
    ("{", "JSON inválido"),
    ("", "JSON inválido"),
    ("[1, 2]", "debe ser un objeto"),
    ("{}", "schema"),
    ('{"schema": 99}', "no soportada"),
    ('{"schema": "1"}', "entero"),
    ('{"schema": true}', "entero"),
    ('{"schema": 1, "x": NaN}', "JSON inválido"),
  ],
)
def test_render_invalid_run_json_returns_2_and_writes_nothing(
  tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, fragment: str
) -> None:
  path = tmp_path / "run.json"
  path.write_text(content, encoding="utf-8")
  assert main(["render", str(path)]) == 2
  captured = capsys.readouterr()
  assert captured.err.startswith(f"error: {path}: ")
  assert fragment in captured.err
  assert captured.out == ""
  assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_render_non_utf8_run_json_returns_2(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  path = tmp_path / "run.json"
  path.write_bytes(b'{"schema": 1, "t": "\xff"}')
  assert main(["render", str(path)]) == 2
  assert capsys.readouterr().err.startswith("error: ")


def test_render_directory_as_run_json_returns_2(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(["render", str(tmp_path)]) == 2
  assert capsys.readouterr().err.startswith("error: ")


def test_render_schema_valid_but_empty_record_returns_2_not_a_traceback(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  path = tmp_path / "run.json"
  path.write_text('{"schema": 1}', encoding="utf-8")
  assert main(["render", str(path)]) == 2
  err = capsys.readouterr().err
  assert err.startswith(f"error: {path}: el registro no tiene la forma esperada")
  assert "KeyError" in err
  assert not (tmp_path / "report.html").exists()


@pytest.mark.parametrize("break_with", [None, "x", 5, []])
def test_render_a_record_with_a_wrongly_shaped_section_returns_2(
  run_json: Path, capsys: pytest.CaptureFixture[str], break_with: Any
) -> None:
  data = json.loads(run_json.read_text(encoding="utf-8"))
  data["questions"][0]["spec"] = break_with
  run_json.write_text(json.dumps(data), encoding="utf-8")
  assert main(["render", str(run_json)]) == 2
  assert capsys.readouterr().err.startswith("error: ")
  assert not (run_json.parent / "report.html").exists()


def test_render_a_lone_surrogate_in_the_record_returns_2_not_a_traceback(
  run_json: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  text = run_json.read_text(encoding="utf-8")
  run_json.write_text(
    text.replace('"title": "Don Quijote"', '"title": "bad \\ud800"'), encoding="utf-8"
  )
  assert main(["render", str(run_json)]) == 2
  assert capsys.readouterr().err.startswith("error: ")


def test_render_unwritable_out_returns_2(
  run_json: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(["render", str(run_json), "--out", str(tmp_path)]) == 2  # a directory
  err = capsys.readouterr().err
  assert err.startswith("error: ")
  assert "no se puede escribir" in err


def test_render_empty_out_is_rejected_by_the_parser(
  run_json: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(["render", str(run_json), "--out", ""]) == 2
  assert "--out" in capsys.readouterr().err


@pytest.mark.parametrize(
  "argv", [["render"], ["render", "a.json", "b.json"], ["render", "a.json", "--nope"]]
)
def test_render_usage_errors_return_2(
  argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(argv) == 2
  assert "usage:" in capsys.readouterr().err


def test_render_help_is_in_spanish(capsys: pytest.CaptureFixture[str]) -> None:
  assert main(["render", "--help"]) == 0
  out = capsys.readouterr().out
  assert "--out" in out
  assert "report.html" in out
  assert "RUN_JSON" in out


def test_top_level_help_lists_render(capsys: pytest.CaptureFixture[str]) -> None:
  assert main(["--help"]) == 0
  assert "render" in capsys.readouterr().out


def test_unexpected_exceptions_while_rendering_are_not_all_swallowed(
  run_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def boom(*args: Any, **kwargs: Any) -> None:
    raise ZeroDivisionError("bug")

  monkeypatch.setattr(cli, "write_report", boom)
  with pytest.raises(ZeroDivisionError):
    main(["render", str(run_json)])


def test_render_through_python_dash_m(run_json: Path) -> None:
  result = subprocess.run(
    [sys.executable, "-m", "hermes_report", "render", str(run_json)],
    capture_output=True,
    text=True,
    check=False,
    timeout=120,
  )
  assert result.returncode == 0, result.stderr
  assert result.stdout == f"Informe: {run_json.parent / 'report.html'}\n"


# ------------------------------------------------------------ compare


@pytest.fixture
def three_runs(
  run_toml: Path, tmp_path: Path, forbid_ollama: None
) -> tuple[Path, Path, Path]:
  """Three real offline runs (default, ``--top-k 1``, ``--budget 30``) -> run.json."""
  paths = []
  for name, flags in (("a", []), ("b", ["--top-k", "1"]), ("c", ["--budget", "30"])):
    out_dir = tmp_path / "runs" / name
    assert main(["run", str(run_toml), "--out", str(out_dir), *flags]) in (0, 1)
    [folder] = list(out_dir.iterdir())
    paths.append(folder / "run.json")
  return paths[0], paths[1], paths[2]


def errors_of(stderr: str) -> list[str]:
  """The ``error:`` lines of stderr (warnings about missing reports are ignored)."""
  return [line for line in stderr.splitlines() if line.startswith("error: ")]


def hrefs_of(text: str) -> list[str]:
  return re.findall(r'<a class="raw-link" href="([^"]*)"', text)


def test_compare_writes_compare_html_in_the_current_directory(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  cwd = tmp_path / "work"
  cwd.mkdir()
  monkeypatch.chdir(cwd)
  a, b, _ = three_runs
  assert main(["compare", str(a), str(b)]) == 0
  captured = capsys.readouterr()
  assert captured.out == f"Comparación: {Path('compare.html')}\n"
  assert captured.err == ""
  text = (cwd / "compare.html").read_text(encoding="utf-8")
  assert text.startswith("<!doctype html>")
  assert "Comparación de 2 ejecuciones" in text
  assert [p.name for p in cwd.iterdir()] == ["compare.html"]


def test_compare_page_equals_render_compare_of_the_records_with_relative_links(
  three_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
  target = tmp_path / "cmp" / "compare.html"
  a, b, c = three_runs
  assert main(["compare", str(a), str(b), str(c), "--out", str(target)]) == 0
  expected = render_compare(
    [
      (
        os.path.relpath(p.parent / "report.html", target.parent).replace(os.sep, "/"),
        read_record(p),
      )
      for p in (a, b, c)
    ]
  )
  assert target.read_text(encoding="utf-8") == expected


def test_compare_links_are_relative_to_the_output_folder_with_slashes(
  three_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
  a, b, _ = three_runs
  for out_dir in (
    tmp_path / "runs",
    tmp_path / "runs" / "a",
    tmp_path / "x" / "y" / "z",
  ):
    target = out_dir / "compare.html"
    assert main(["compare", str(a), str(b), "--out", str(target)]) == 0
    links = hrefs_of(target.read_text(encoding="utf-8"))
    assert links == [
      Path(os.path.relpath(p.parent / "report.html", out_dir)).as_posix()
      for p in (a, b)
    ]
    assert not any("\\" in link or link.startswith("/") for link in links)
    # following the link from the comparison lands on the report of that run
    for link, run in zip(links, (a, b), strict=True):
      assert (out_dir / link).resolve() == (run.parent / "report.html").resolve()


def test_compare_links_with_up_levels_when_the_output_is_deeper(
  three_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
  a, b, _ = three_runs
  target = tmp_path / "runs" / "a" / "deeper" / "compare.html"
  assert main(["compare", str(a), str(b), "--out", str(target)]) == 0
  links = hrefs_of(target.read_text(encoding="utf-8"))
  assert links[0].startswith("../") and links[1].startswith("../../b/")


def test_compare_with_relative_paths_and_the_default_output(
  three_runs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  a, b, _ = three_runs
  monkeypatch.chdir(tmp_path)
  rel_a, rel_b = a.relative_to(tmp_path), b.relative_to(tmp_path)
  assert main(["compare", str(rel_a), str(rel_b)]) == 0
  links = hrefs_of((tmp_path / "compare.html").read_text(encoding="utf-8"))
  assert links == [
    (rel_a.parent / "report.html").as_posix(),
    (rel_b.parent / "report.html").as_posix(),
  ]
  for link in links:
    assert (tmp_path / link).is_file()


def test_compare_run_json_in_the_current_directory(
  three_runs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
  a, b, _ = three_runs
  monkeypatch.chdir(a.parent)
  assert main(["compare", "run.json", str(b), "--out", "cmp.html"]) == 0
  links = hrefs_of(Path("cmp.html").read_text(encoding="utf-8"))
  assert links[0] == "report.html"


def test_compare_shows_the_differences_of_the_flags_end_to_end(
  three_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
  target = tmp_path / "compare.html"
  a, b, c = three_runs
  assert main(["compare", str(a), str(b), str(c), "--out", str(target)]) == 0
  text = target.read_text(encoding="utf-8")
  assert "<code>retrieval.top_k</code>" in text
  assert "<code>enrich.max_context_tokens</code>" in text
  assert '<tr class="diff" data-q="q01">' in text  # the 30-token budget fails q01


def test_compare_does_not_touch_the_run_folders(
  three_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
  a, b, _ = three_runs
  before = {p: p.read_bytes() for r in (a, b) for p in r.parent.iterdir()}
  assert main(["compare", str(a), str(b), "--out", str(tmp_path / "c.html")]) == 0
  after = {p: p.read_bytes() for r in (a, b) for p in r.parent.iterdir()}
  assert after == before


def test_compare_twice_gives_the_same_bytes_and_overwrites(
  three_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
  a, b, _ = three_runs
  target = tmp_path / "compare.html"
  target.write_text("old", encoding="utf-8")
  assert main(["compare", str(a), str(b), "--out", str(target)]) == 0
  first = target.read_bytes()
  assert first != b"old"
  assert main(["compare", str(a), str(b), "--out", str(target)]) == 0
  assert target.read_bytes() == first


def test_compare_the_same_run_twice_is_allowed_and_has_no_differences(
  three_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
  a, _, _ = three_runs
  target = tmp_path / "compare.html"
  assert main(["compare", str(a), str(a), "--out", str(target)]) == 0
  text = target.read_text(encoding="utf-8")
  assert "no se diferencian" in text
  assert "Comparación de 2 ejecuciones" in text


def test_compare_a_missing_report_html_warns_but_keeps_the_link(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  a, b, _ = three_runs
  (b.parent / "report.html").unlink()
  target = tmp_path / "compare.html"
  assert main(["compare", str(a), str(b), "--out", str(target)]) == 0
  captured = capsys.readouterr()
  missing = b.parent / "report.html"
  assert captured.err == f"aviso: no existe {missing}; el enlace quedará roto\n"
  assert str(a.parent / "report.html") not in captured.err
  assert captured.out == f"Comparación: {target}\n"
  links = hrefs_of(target.read_text(encoding="utf-8"))
  assert links[1] == Path(os.path.relpath(missing, tmp_path)).as_posix()


def test_compare_warns_once_per_missing_report(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  for run in three_runs:
    (run.parent / "report.html").unlink()
  assert (
    main(["compare", *map(str, three_runs), "--out", str(tmp_path / "c.html")]) == 0
  )
  assert capsys.readouterr().err.count("aviso:") == 3


def test_compare_with_all_reports_present_prints_no_warning(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  assert (
    main(["compare", *map(str, three_runs), "--out", str(tmp_path / "c.html")]) == 0
  )
  assert capsys.readouterr().err == ""


def test_compare_run_folder_with_spaces_and_accents(
  three_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
  a, b, _ = three_runs
  odd = tmp_path / "mi carpeta ñ" / "ejecución 1"
  odd.parent.mkdir()
  shutil.copytree(a.parent, odd)
  target = tmp_path / "compare.html"
  assert main(["compare", str(odd / "run.json"), str(b), "--out", str(target)]) == 0
  links = hrefs_of(target.read_text(encoding="utf-8"))
  assert links[0] == "mi carpeta ñ/ejecución 1/report.html"


def test_compare_with_no_files_is_a_usage_error(
  capsys: pytest.CaptureFixture[str],
) -> None:
  assert main(["compare"]) == 2
  captured = capsys.readouterr()
  assert "usage:" in captured.err
  assert captured.out == ""


def test_compare_with_one_file_returns_2_and_writes_nothing(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  target = tmp_path / "compare.html"
  assert main(["compare", str(three_runs[0]), "--out", str(target)]) == 2
  captured = capsys.readouterr()
  assert captured.err.startswith("error: ")
  assert "2" in captured.err
  assert captured.out == ""
  assert not target.exists()


def test_compare_with_one_file_that_does_not_exist_still_returns_2(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(["compare", str(tmp_path / "nope.json")]) == 2
  assert capsys.readouterr().err.startswith("error: ")


def test_compare_missing_file_returns_2_naming_it_and_writes_nothing(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  missing = tmp_path / "nope.json"
  target = tmp_path / "compare.html"
  assert main(["compare", str(three_runs[0]), str(missing), "--out", str(target)]) == 2
  captured = capsys.readouterr()
  assert f"error: {missing}: el archivo no existe\n" in captured.err
  assert captured.out == ""
  assert not target.exists()


@pytest.mark.parametrize(
  ("content", "fragment"),
  [
    ("{", "JSON inválido"),
    ("", "JSON inválido"),
    ("[1]", "debe ser un objeto"),
    ("{}", "schema"),
    ('{"schema": 99}', "no soportada"),
  ],
)
def test_compare_invalid_run_json_returns_2_whichever_position_it_is_in(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
  content: str,
  fragment: str,
) -> None:
  bad = tmp_path / "bad.json"
  bad.write_text(content, encoding="utf-8")
  target = tmp_path / "compare.html"
  good = str(three_runs[0])
  for order in ([good, str(bad)], [str(bad), good]):
    assert main(["compare", *order, "--out", str(target)]) == 2
    [error] = errors_of(capsys.readouterr().err)
    assert error.startswith(f"error: {bad}: ")
    assert fragment in error
    assert not target.exists()


def test_compare_non_utf8_run_json_returns_2(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  bad = tmp_path / "bad.json"
  bad.write_bytes(b'{"schema": 1, "t": "\xff"}')
  assert (
    main(["compare", str(three_runs[0]), str(bad), "--out", str(tmp_path / "c.html")])
    == 2
  )
  assert len(errors_of(capsys.readouterr().err)) == 1


def test_compare_directory_as_run_json_returns_2(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  assert main(["compare", str(three_runs[0]), str(tmp_path)]) == 2
  assert len(errors_of(capsys.readouterr().err)) == 1


@pytest.mark.parametrize("break_with", [None, "x", 5])
@pytest.mark.parametrize("key", ["environment", "questions"])
def test_compare_a_record_with_a_wrongly_shaped_section_returns_2_not_a_traceback(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
  key: str,
  break_with: Any,
) -> None:
  a, b, _ = three_runs
  data = json.loads(b.read_text(encoding="utf-8"))
  data[key] = break_with
  b.write_text(json.dumps(data), encoding="utf-8")
  target = tmp_path / "compare.html"
  assert main(["compare", str(a), str(b), "--out", str(target)]) == 2
  assert len(errors_of(capsys.readouterr().err)) == 1
  assert not target.exists()


def test_compare_schema_valid_but_empty_record_returns_2(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  empty = tmp_path / "run.json"
  empty.write_text('{"schema": 1}', encoding="utf-8")
  assert (
    main(["compare", str(three_runs[0]), str(empty), "--out", str(tmp_path / "c.html")])
    == 2
  )
  [error] = errors_of(capsys.readouterr().err)
  assert "KeyError" in error
  assert not (tmp_path / "c.html").exists()


def test_compare_lone_surrogate_in_the_record_returns_2_and_keeps_the_old_output(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  a, b, _ = three_runs
  text = b.read_text(encoding="utf-8")
  b.write_text(
    text.replace('"title": "Don Quijote"', '"title": "bad \\ud800"'), encoding="utf-8"
  )
  target = tmp_path / "compare.html"
  target.write_text("old", encoding="utf-8")
  assert main(["compare", str(a), str(b), "--out", str(target)]) == 2
  assert capsys.readouterr().err.startswith("error: ")
  assert target.read_text(encoding="utf-8") == "old"


def test_compare_unwritable_out_returns_2(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  assert (
    main(["compare", str(three_runs[0]), str(three_runs[1]), "--out", str(tmp_path)])
    == 2
  )
  err = capsys.readouterr().err
  assert err.startswith("error: ")
  assert "no se puede escribir" in err


def test_compare_out_under_a_file_returns_2(
  three_runs: tuple[Path, Path, Path],
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
) -> None:
  blocker = tmp_path / "blocker"
  blocker.write_text("x", encoding="utf-8")
  code = main(
    [
      "compare",
      str(three_runs[0]),
      str(three_runs[1]),
      "--out",
      str(blocker / "c.html"),
    ]
  )
  assert code == 2
  assert capsys.readouterr().err.startswith("error: ")


def test_compare_empty_out_is_rejected_by_the_parser(
  three_runs: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(["compare", str(three_runs[0]), str(three_runs[1]), "--out", ""]) == 2
  assert "--out" in capsys.readouterr().err


def test_compare_unknown_flag_is_a_usage_error(
  three_runs: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
  assert main(["compare", str(three_runs[0]), str(three_runs[1]), "--nope"]) == 2
  assert "usage:" in capsys.readouterr().err


def test_compare_help_is_in_spanish_and_top_level_help_lists_it(
  capsys: pytest.CaptureFixture[str],
) -> None:
  assert main(["compare", "--help"]) == 0
  out = capsys.readouterr().out
  for fragment in ("RUN_JSON", "--out", "compare.html", "run.json"):
    assert fragment in out
  assert main(["--help"]) == 0
  assert "compare" in capsys.readouterr().out


def test_unexpected_exceptions_while_comparing_are_not_swallowed(
  three_runs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  def boom(*args: Any, **kwargs: Any) -> None:
    raise ZeroDivisionError("bug")

  monkeypatch.setattr(cli, "write_compare", boom)
  with pytest.raises(ZeroDivisionError):
    main(
      [
        "compare",
        str(three_runs[0]),
        str(three_runs[1]),
        "--out",
        str(tmp_path / "c.html"),
      ]
    )


def test_compare_through_python_dash_m(
  three_runs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
  target = tmp_path / "compare.html"
  result = subprocess.run(
    [
      sys.executable,
      "-m",
      "hermes_report",
      "compare",
      str(three_runs[0]),
      str(three_runs[1]),
      "--out",
      str(target),
    ],
    capture_output=True,
    text=True,
    check=False,
    timeout=120,
  )
  assert result.returncode == 0, result.stderr
  assert result.stdout == f"Comparación: {target}\n"
  assert target.is_file()


def test_compare_one_file_through_python_dash_m_exits_2(
  three_runs: tuple[Path, Path, Path],
) -> None:
  result = subprocess.run(
    [sys.executable, "-m", "hermes_report", "compare", str(three_runs[0])],
    capture_output=True,
    text=True,
    check=False,
    timeout=120,
  )
  assert result.returncode == 2
  assert result.stderr.startswith("error: ")

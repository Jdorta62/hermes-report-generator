"""Tests for ``hermes_report.cli`` and ``python -m hermes_report``.

Real runs use the sample project (in-memory Qdrant, hash embedder, echo LLM and
judge). Tests of the flag mapping and of the error mapping replace ``cli.run`` so
they do not depend on the pipeline.
"""

from __future__ import annotations

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
  assert captured.out.splitlines()[-2:] == [
    f"Carpeta: {folder}",
    f"Registro: {folder / 'run.json'}",
  ]
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

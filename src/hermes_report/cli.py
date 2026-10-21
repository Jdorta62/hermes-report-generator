"""Command line of the tool: ``hermes-report run RUN_TOML``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hermes_report.config import ConfigError, Overrides, load_config
from hermes_report.corpus import CorpusError
from hermes_report.ingest import IngestError
from hermes_report.providers import ProviderError
from hermes_report.questions import QuestionsError
from hermes_report.record import RecordError
from hermes_report.runner import RunError, run
from hermes_report.store import StoreError

__all__ = ["main"]

_EXPECTED_ERRORS = (
  ConfigError,
  QuestionsError,
  CorpusError,
  ProviderError,
  IngestError,
  StoreError,
  RecordError,
  RunError,
)


def _non_empty_path(text: str) -> Path:
  """An argparse type: a path that is not an empty string."""
  if not text:
    raise argparse.ArgumentTypeError("no puede estar vacío")
  return Path(text)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="hermes-report",
    description="Informes de calidad del pipeline Hermes sobre un corpus Markdown.",
  )
  commands = parser.add_subparsers(dest="command", metavar="COMANDO")
  run_parser = commands.add_parser(
    "run",
    help="ejecuta una batería de preguntas y guarda el registro de la ejecución",
    description="Ejecuta las preguntas de un archivo de ejecución (TOML) "
    "y guarda run.toml y run.json en una carpeta nueva.",
  )
  run_parser.add_argument("run_toml", metavar="RUN_TOML", type=Path)
  run_parser.add_argument(
    "--model", help="modelo de generación (sustituye al del archivo)"
  )
  run_parser.add_argument("--judge-model", help="modelo del juez")
  run_parser.add_argument(
    "--no-judge", action="store_true", help="no evaluar con el juez LLM"
  )
  run_parser.add_argument("--top-k", type=int, metavar="N", help="chunks recuperados")
  run_parser.add_argument(
    "--budget", type=int, metavar="N", help="tokens máximos del contexto"
  )
  run_parser.add_argument(
    "--out",
    type=_non_empty_path,
    metavar="DIR",
    help="carpeta donde crear la ejecución",
  )
  run_parser.add_argument(
    "--only",
    action="append",
    metavar="ID",
    help="ejecuta solo esta pregunta (se puede repetir)",
  )
  run_parser.add_argument(
    "--reingest",
    action="store_true",
    help="ignora la caché de ingesta y recarga la colección",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  """Run the ``hermes-report`` command line and return its exit code.

  Exit codes: 0 when every question ran, 1 when some question raised, 2 for
  invalid input, an unreachable service or a missing subcommand. Errors are
  printed on stderr as ``error: <message>``.

  Effects: io, net, mut
  """
  parser = _parser()
  try:
    args = parser.parse_args(argv)
  except SystemExit as exc:
    return exc.code if isinstance(exc.code, int) else 2
  if args.command != "run":
    parser.print_help()
    return 2
  overrides = Overrides(
    model=args.model,
    judge_model=args.judge_model,
    no_judge=args.no_judge,
    top_k=args.top_k,
    budget=args.budget,
    out=args.out,
    only=tuple(args.only or ()),
    reingest=args.reingest,
  )
  try:
    outcome = run(load_config(args.run_toml, overrides))
  except _EXPECTED_ERRORS as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  print(f"Carpeta: {outcome.folder}")
  print(f"Registro: {outcome.folder / 'run.json'}")
  if outcome.failed:
    print(f"{outcome.failed} pregunta(s) con error", file=sys.stderr)
    return 1
  return 0

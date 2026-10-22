"""Command line of the tool: ``hermes-report`` ``run``, ``render`` and ``compare``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from hermes_report.config import ConfigError, Overrides, load_config
from hermes_report.corpus import CorpusError
from hermes_report.ingest import IngestError
from hermes_report.providers import ProviderError
from hermes_report.questions import QuestionsError
from hermes_report.record import RecordError, read_record
from hermes_report.render import write_compare, write_report
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
  """An argparse type: a path that is neither empty nor only whitespace."""
  if not text.strip():
    raise argparse.ArgumentTypeError("no puede estar vacío ni ser solo espacios")
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
    description="Ejecuta las preguntas de un archivo de ejecución (TOML) y guarda "
    "run.json, run.toml y report.html en una carpeta nueva.",
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
  render_parser = commands.add_parser(
    "render",
    help="genera el informe HTML a partir de un run.json",
    description="Lee un run.json y escribe el informe report.html, sin ejecutar nada.",
  )
  render_parser.add_argument("run_json", metavar="RUN_JSON", type=Path)
  render_parser.add_argument(
    "--out",
    type=_non_empty_path,
    metavar="PATH",
    help="archivo HTML de salida (por defecto report.html junto al run.json)",
  )
  compare_parser = commands.add_parser(
    "compare",
    help="compara varias ejecuciones en un informe HTML",
    description="Lee dos o más run.json y escribe compare.html con sus diferencias, "
    "puntuación y una matriz de preguntas, sin ejecutar nada.",
  )
  compare_parser.add_argument("run_jsons", metavar="RUN_JSON", type=Path, nargs="+")
  compare_parser.add_argument(
    "--out",
    type=_non_empty_path,
    metavar="PATH",
    help="archivo HTML de salida (por defecto compare.html en la carpeta actual)",
  )
  return parser


def _compare(args: argparse.Namespace) -> int:
  """The ``compare`` command: write the comparison of several ``run.json``."""
  paths: list[Path] = args.run_jsons
  if len(paths) < 2:
    print("error: compare necesita al menos 2 archivos run.json", file=sys.stderr)
    return 2
  out: Path = args.out or Path("compare.html")
  try:
    runs = []
    for current in paths:
      report = current.parent / "report.html"
      if not report.is_file():
        print(f"aviso: no existe {report}; el enlace quedará roto", file=sys.stderr)
      href = Path(os.path.relpath(report, out.parent)).as_posix()
      runs.append((href, read_record(current)))
    write_compare(runs, out)
  except RecordError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
    print(
      f"error: algún registro no tiene la forma esperada ({type(exc).__name__}: {exc})",
      file=sys.stderr,
    )
    return 2
  print(f"Comparación: {out}")
  return 0


def _render(args: argparse.Namespace) -> int:
  """The ``render`` command: write the report of a ``run.json``."""
  out: Path = args.out or args.run_json.parent / "report.html"
  try:
    write_report(read_record(args.run_json), out)
  except RecordError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
  except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
    print(
      f"error: {args.run_json}: el registro no tiene la forma esperada "
      f"({type(exc).__name__}: {exc})",
      file=sys.stderr,
    )
    return 2
  print(f"Informe: {out}")
  return 0


def main(argv: list[str] | None = None) -> int:
  """Run the ``hermes-report`` command line and return its exit code.

  Exit codes: 0 when every question ran (or the report was written), 1 when
  some question raised, 2 for invalid input, an unreachable service, a run
  record that cannot be read or rendered, or a missing subcommand. Errors are
  printed on stderr as ``error: <message>``.

  Effects: io, net, mut
  """
  parser = _parser()
  try:
    args = parser.parse_args(argv)
  except SystemExit as exc:
    return exc.code if isinstance(exc.code, int) else 2
  if args.command == "render":
    return _render(args)
  if args.command == "compare":
    return _compare(args)
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
  print(f"Informe: {outcome.folder / 'report.html'}")
  if outcome.failed:
    print(f"{outcome.failed} pregunta(s) con error", file=sys.stderr)
    return 1
  return 0

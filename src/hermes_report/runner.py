"""Run orchestration: corpus to graded questions, written as a run folder."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_core import Embedder

from hermes_report.config import RunConfig
from hermes_report.corpus import Corpus, load_corpus
from hermes_report.evaluate import judge, judge_messages, rule_verdict
from hermes_report.ingest import Ingestion, ingest_corpus
from hermes_report.metrics.adapter import metric_input
from hermes_report.metrics.base import (
  Metric,
  MetricResult,
  MetricServices,
  describe,
)
from hermes_report.metrics.judge import JudgeSession
from hermes_report.metrics.runner import ScoreSummary, aggregate, evaluate_metrics
from hermes_report.metrics.selection import MetricSelection
from hermes_report.pipeline import PipelineContext, answer_question, build_context
from hermes_report.providers import (
  JudgeClient,
  ProviderError,
  check_ollama,
  make_embedder,
  make_judge,
  make_llm,
  ollama_host,
)
from hermes_report.questions import Question, load_questions
from hermes_report.record import (
  SCHEMA_VERSION,
  CollectionRecord,
  CorpusRecord,
  DocumentRecord,
  EnvironmentRecord,
  IngestDocumentRecord,
  IngestRecord,
  JudgeRecord,
  MetricsRunRecord,
  QuestionRecord,
  RunRecord,
  ScoreSummaryRecord,
  message_record,
  write_record,
)
from hermes_report.render import write_report
from hermes_report.store import (
  CollectionInfo,
  collection_name,
  ensure_collection,
  make_client,
  read_points,
  slug,
)

__all__ = ["RunError", "RunOutcome", "run"]

_PACKAGES = (
  "hermes-core",
  "hermes-ingest-md",
  "hermes-retrieval",
  "hermes-enrich",
  "hermes-generator",
  "hermes-report",
  "qdrant-client",
  "ollama",
)
_QUESTION_PREVIEW = 60
_NO_VERDICT = "—"


class RunError(Exception):
  """The run cannot start or finish as asked (message in Spanish)."""


@dataclass(frozen=True)
class RunOutcome:
  """The result of a run: the record, where it was written and how many failed.

  Attributes:
    record: The run record, as written to ``run.json``.
    folder: The output folder of the run.
    failed: The number of questions whose pipeline raised (``error`` set).
  """

  record: RunRecord
  folder: Path
  failed: int


def run(config: RunConfig, progress: Callable[[str], None] = print) -> RunOutcome:
  """Run every selected question through the pipeline and write the run folder.

  The folder holds ``run.json`` (the run record), ``run.toml`` (the resolved
  run file) and ``report.html`` (the report), written in that order so that a
  late failure never loses the results. Pipeline errors of one question are
  recorded and do not stop the run; the judge runs only for a question that has
  a generation result and no error, and an unexpected failure of the judge is
  recorded in its record. Then the selected metric families run on every
  question (also on one whose pipeline raised) with a fresh judge session, or
  without a judge when it is disabled; a metric never changes the verdicts or
  the number of failed questions. ``metrics`` stays ``None`` when
  ``config.metrics.enabled`` is false.

  Args:
    config: The resolved run, with its overrides applied.
    progress: Receives one line of text per step and per question.

  Raises:
    RunError: if ``config.only`` names an unknown question or the folder
      cannot be written.
    QuestionsError: if the questions file is invalid.
    CorpusError: if the corpus folder cannot be read.
    ProviderError: if Ollama or another provider fails or lacks a model.
    IngestError: if a document cannot be ingested.
    StoreError: if Qdrant cannot be used.
    RecordError: if ``run.json`` or ``report.html`` cannot be written.

  Effects: io, net, mut
  """
  start = datetime.now().astimezone()
  begin = time.perf_counter()
  corpus = load_corpus(config.corpus.dir, config.corpus.glob)
  question_set = load_questions(config.questions.file, corpus.sources)
  questions = _select(question_set.questions, config.only)
  selection = config.metrics
  selected: tuple[Metric, ...] = ()
  try:
    selected = MetricSelection(
      selection.enabled, selection.include, selection.exclude
    ).resolve()
  except Exception as exc:
    progress(f"· AVISO: no se pudo resolver la selección de métricas ({describe(exc)})")

  models = _ollama_models(config)
  if models:
    check_ollama(models)

  embedder = make_embedder(config.embedding)
  llm = make_llm(config.generation)
  judge_client = make_judge(config.judge) if config.judge.enabled else None

  progress(f"· Corpus: {len(corpus.documents)} documentos")
  ingestion = ingest_corpus(
    corpus,
    config.ingest,
    config.embedding,
    embedder,
    config.output.cache_dir,
    force=config.reingest,
    progress=progress,
  )
  client = make_client(config.qdrant.url)
  try:
    name = collection_name(config.title, ingestion.key, config.qdrant.collection)
    info = ensure_collection(
      client, name, ingestion.points, config.qdrant.url, force=config.reingest
    )
    progress(
      f"· Qdrant {config.qdrant.url} / {name}: "
      f"{'cargada' if info.reloaded else 'ya cargada'} ({info.points} puntos)"
    )
    points = read_points(client, name)
    ctx = build_context(config, client, name, embedder, llm)
    records: list[QuestionRecord] = []
    results: list[tuple[MetricResult, ...]] = []
    for question in questions:
      answered, metric_results = _answer(
        question, ctx, judge_client, embedder, selected, progress
      )
      records.append(answered)
      results.append(metric_results)
  finally:
    client.close()

  finished = datetime.now().astimezone()
  metrics_record: MetricsRunRecord | None = None
  if selection.enabled:
    metrics_record = _metrics_record(selected, results, progress)
  record: RunRecord = {
    "schema": SCHEMA_VERSION,
    "title": config.title,
    "description": config.description,
    "started_at": start.isoformat(),
    "finished_at": finished.isoformat(),
    "seconds": time.perf_counter() - begin,
    "config": config.to_dict(),
    "environment": _environment(config),
    "corpus": _corpus_record(corpus, ingestion),
    "ingest": _ingest_record(ingestion),
    "collection": _collection_record(info),
    "points": points,
    "questions": records,
    "metrics": metrics_record,
  }
  folder = _write(config, record, start)
  return RunOutcome(
    record=record,
    folder=folder,
    failed=sum(1 for item in records if item["error"] is not None),
  )


def _select(questions: tuple[Question, ...], only: tuple[str, ...]) -> list[Question]:
  """The questions named in ``only`` in file order, or all of them."""
  if not only:
    return list(questions)
  known = {question.id for question in questions}
  unknown = [item for item in dict.fromkeys(only) if item not in known]
  if unknown:
    raise RunError(
      "pregunta(s) desconocida(s) en --only: "
      + ", ".join(unknown)
      + "; las del archivo son: "
      + ", ".join(question.id for question in questions)
    )
  wanted = set(only)
  return [question for question in questions if question.id in wanted]


def _ollama_models(config: RunConfig) -> list[str]:
  """The Ollama models the run will use, without repeats."""
  models: list[str] = []
  if config.embedding.provider == "ollama":
    models.append(config.embedding.model)
  if config.generation.provider == "ollama":
    models.append(config.generation.model)
  if config.judge.enabled and config.judge.provider == "ollama":
    models.append(config.judge.model)
  return list(dict.fromkeys(models))


def _answer(
  question: Question,
  ctx: PipelineContext,
  judge_client: JudgeClient | None,
  embedder: Embedder,
  selected: tuple[Metric, ...],
  progress: Callable[[str], None],
) -> tuple[QuestionRecord, tuple[MetricResult, ...]]:
  """Answer, grade and measure one question, and report it."""
  begin = time.perf_counter()
  record = answer_question(question, ctx)
  record["rules"] = rule_verdict(question, record)
  generation = record["generation"]
  if (
    judge_client is not None
    and record["error"] is None
    and generation is not None
    and generation["result"] is not None
  ):
    record["judge"] = _grade(question, record, generation["result"], judge_client)
  results: tuple[MetricResult, ...] = ()
  if ctx.config.metrics.enabled:
    # Metrics never change the run: a failure leaves this question without them.
    try:
      item = metric_input(question, record)
      services = MetricServices(
        judge=None if judge_client is None else JudgeSession(judge_client, item),
        embedder=embedder,
        counter=ctx.token_counter,
      )
      results = evaluate_metrics(item, selected, services)
      record["metrics"] = [result.to_record() for result in results]
    except Exception as exc:
      results = ()
      record["metrics"] = None
      progress(
        f"· AVISO {question.id}: las métricas fallaron y se omiten ({describe(exc)})"
      )
  judged = record["judge"]
  verdict = (judged["verdict"] if judged else None) or _NO_VERDICT
  rules = record["rules"]
  code = rules["verdict"] if rules else _NO_VERDICT
  seconds = time.perf_counter() - begin
  progress(
    f"· {question.id} {question.question[:_QUESTION_PREVIEW]} … "
    f"{code} / juez: {verdict} ({seconds:.1f} s)"
  )
  return record, results


def _score_record(item: ScoreSummary) -> ScoreSummaryRecord:
  return {
    "mean": item.mean,
    "n": item.n,
    "not_applicable": item.not_applicable,
    "skipped": item.skipped,
    "errors": item.errors,
  }


def _metrics_record(
  selected: tuple[Metric, ...],
  results: list[tuple[MetricResult, ...]],
  progress: Callable[[str], None],
) -> MetricsRunRecord:
  """The metrics of the run: the selection and the aggregate of every score.

  A failing aggregate leaves the summary empty and is reported to ``progress``.
  """
  try:
    summary = aggregate(results, selected)
    records = {score: _score_record(item) for score, item in summary.items()}
  except Exception as exc:
    progress(f"· AVISO: no se pudo agregar las métricas ({describe(exc)})")
    records = {}
  return {
    "selection": [metric.name for metric in selected],
    "scores": [score for metric in selected for score in metric.scores],
    "summary": records,
  }


def _grade(
  question: Question,
  record: QuestionRecord,
  result: dict[str, Any],
  client: JudgeClient,
) -> JudgeRecord | None:
  """Judge one answer; a failure that is not a ``ProviderError`` is recorded."""
  begin = time.perf_counter()
  try:
    return judge(question, record, client)
  except ProviderError:
    raise
  except Exception as exc:
    seconds = time.perf_counter() - begin
    messages = judge_messages(question, result["answer"], result["status"])
    return {
      "verdict": None,
      "reason": None,
      "model": client.model,
      "seconds": seconds,
      "messages": [message_record(m) for m in messages],
      "raw": None,
      "error": f"{type(exc).__name__}: {exc}",
    }


def _version(package: str) -> str:
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return "?"


def _environment(config: RunConfig) -> EnvironmentRecord:
  return {
    "python": platform.python_version(),
    "packages": {package: _version(package) for package in _PACKAGES},
    "ollama_host": ollama_host(),
    "qdrant_url": config.qdrant.url,
  }


def _corpus_record(corpus: Corpus, ingestion: Ingestion) -> CorpusRecord:
  chunks = {doc.source: len(doc.points) for doc in ingestion.documents}
  documents: list[DocumentRecord] = [
    {
      "source": doc.source,
      "sha256": doc.sha256,
      "chars": doc.chars,
      "chunks": chunks.get(doc.source, 0),
    }
    for doc in corpus.documents
  ]
  return {
    "dir": str(corpus.dir),
    "glob": corpus.glob,
    "fingerprint": corpus.fingerprint,
    "documents": documents,
  }


def _ingest_record(ingestion: Ingestion) -> IngestRecord:
  documents: list[IngestDocumentRecord] = [
    {
      "source": doc.source,
      "chunks": len(doc.points),
      "seconds": doc.seconds,
      "cached": doc.cached,
    }
    for doc in ingestion.documents
  ]
  return {
    "seconds": ingestion.seconds,
    "chunker": ingestion.chunker,
    "embedding": ingestion.embedding,
    "ingester_version": ingestion.ingester_version,
    "documents": documents,
  }


def _collection_record(info: CollectionInfo) -> CollectionRecord:
  return {
    "name": info.name,
    "url": info.url,
    "reloaded": info.reloaded,
    "points": info.points,
    "vector_size": info.vector_size,
    "distance": info.distance,
    "indexes": list(info.indexes),
  }


def _write(config: RunConfig, record: RunRecord, start: datetime) -> Path:
  """Create the run folder (a ``-2``, ``-3``... suffix if taken) and fill it.

  ``run.json`` goes first: it holds the results, the other files are copies.
  """
  base = config.output.dir
  name = f"{start:%Y%m%d-%H%M%S}-{slug(config.title)}"
  try:
    base.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
      folder = base / (name if number == 1 else f"{name}-{number}")
      try:
        folder.mkdir()
      except FileExistsError:
        number += 1
        continue
      break
  except OSError as exc:
    raise RunError(f"no se puede escribir la carpeta de salida {base}: {exc}") from exc
  write_record(record, folder / "run.json")
  try:
    (folder / "run.toml").write_text(config.to_toml(), encoding="utf-8")
  except OSError as exc:
    raise RunError(f"no se puede escribir {folder / 'run.toml'}: {exc}") from exc
  write_report(record, folder / "report.html")
  return folder

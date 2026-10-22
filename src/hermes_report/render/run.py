"""The ``report.html`` of a run: a pure function of the run record."""

from __future__ import annotations

import contextlib
import json
import os
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from hermes_enrich.tokens import TokenCounter, default_token_counter

from hermes_report.record import (
  PointRecord,
  QuestionRecord,
  RecordError,
  RunRecord,
)
from hermes_report.render.html import (
  NO_DATA,
  RULE_TONES,
  card,
  esc,
  grid,
  json_block,
  json_script,
  page,
  seconds,
  status_pill,
  table,
  yes_no,
)
from hermes_report.render.question import (
  QuestionContext,
  chunk_usage,
  disagree,
  question_panel,
)

__all__ = ["render_run", "write_report"]

# With more questions than this the tabs show only the id.
_COMPACT_ABOVE: Final = 6
_TAB_TEXT: Final = 40
_MARK: Final = '<span class="status mark" title="reglas y juez discrepan">≠</span>'
_BUDGET_FAILURES: Final = ("budget_failure",)
_RETRIEVAL_FAILURES: Final = ("retrieval_failure",)
_GENERATION_FAILURES: Final = ("generation_refused", "generation_wrong")


def render_run(record: RunRecord, *, token_counter: TokenCounter | None = None) -> str:
  """Render the whole report of ``record`` as one self-contained HTML document.

  The result depends only on ``record`` and the token counter: the same record
  and counter always give the same bytes, and nothing in the page refers to an
  external URL. The Qdrant points travel once, as JSON in the page, and the
  explorer and the raw documents of each question are drawn from them by the
  script.

  Args:
    record: The run record.
    token_counter: Measures the context texts; ``default_token_counter()``,
      created once for the whole render, when ``None``.

  Effects: net
  """
  counter = default_token_counter() if token_counter is None else token_counter
  context = QuestionContext(record["points"], counter)
  questions = record["questions"]
  ids = [_question_id(item, number) for number, item in enumerate(questions, 1)]
  compact = len(questions) > _COMPACT_ABOVE
  tabs = [
    ("resumen", "Resumen", "Resumen de la ejecución"),
    ("corpus", "Corpus e ingesta", "Corpus e ingesta"),
    ("chunks", "Chunks en Qdrant", "Chunks en Qdrant"),
  ]
  panels = [
    ("resumen", _summary_panel(record, ids)),
    ("corpus", _corpus_panel(record)),
    ("chunks", _chunks_panel()),
  ]
  for question_id, item in zip(ids, questions, strict=True):
    text = str(item["spec"].get("question", ""))
    key = f"q-{question_id}"
    tabs.append(
      (
        key,
        question_id if compact else _clip(text, _TAB_TEXT),
        f"{question_id} · {text}",
      )
    )
    panels.append((key, question_panel(item, question_id, context)))
  # The points are written once and drawn by ``report.js``; the usage is one
  # list of chunk ids per state and question, so a corpus is never copied per tab.
  data = json_script("hr-points", record["points"]) + json_script(
    "hr-usage",
    {qid: chunk_usage(item) for qid, item in zip(ids, questions, strict=True)},
  )
  return page(record["title"], _meta(record, context.counter_name), tabs, panels, data)


def write_report(record: RunRecord, path: Path) -> None:
  """Render ``record`` and write it as UTF-8 HTML, creating the parent folder.

  The report is written to a temporary file in the same folder and then moved
  over ``path``, so a failure keeps the previous report intact.

  Raises:
    RecordError: if the file cannot be written or the report cannot be encoded
      as UTF-8 (a lone surrogate in the record); the error names the file.

  Effects: io, net
  """
  html = render_run(record)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
      handle.write(html)
    os.replace(temporary, path)
  except UnicodeError as exc:
    raise RecordError(
      path, f"el informe contiene texto que no se puede codificar en UTF-8: {exc}"
    ) from None
  except OSError as exc:
    raise RecordError(path, f"no se puede escribir: {exc}") from None
  finally:
    with contextlib.suppress(OSError):
      temporary.unlink(missing_ok=True)


# --- shared bits ---------------------------------------------------------------


def _question_id(item: QuestionRecord, number: int) -> str:
  return str(item["spec"].get("id") or f"q{number:02d}")


def _clip(text: str, limit: int) -> str:
  return text if len(text) <= limit else text[: limit - 1] + "…"


def _number(value: float) -> str:
  return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _head(title: str, goal: str = "") -> str:
  return f'<div class="sc-head"><h2>{esc(title)}</h2></div>' + (
    f'<p class="goal">{esc(goal)}</p>' if goal else ""
  )


def _meta(record: RunRecord, counter_name: str) -> str:
  config = record["config"]
  judge = config.get("judge", {})
  judge_text = esc(judge.get("model", "?")) if judge.get("enabled") else "sin juez"
  collection = record["collection"]
  return (
    f"Generación: <b>{esc(config.get('generation', {}).get('model', '?'))}</b>"
    f" · Juez: <b>{judge_text}</b>"
    f" · Qdrant: <b>{esc(collection['url'])}</b> / {esc(collection['name'])}"
    f" · Tokens: <b>{esc(counter_name)}</b>"
    f" · {esc(record['started_at'])} · {esc(seconds(record['seconds']))}"
  )


# --- Chunks en Qdrant ---------------------------------------------------------------


def _chunks_panel() -> str:
  return (
    _head(
      "Chunks en Qdrant",
      "Todos los chunks de la colección tal como están guardados: texto, payload y "
      "vector. Marca una pregunta para ver cómo los usó.",
    )
    + '<div id="explorer"><noscript><p class="muted">El explorador de chunks '
    "necesita JavaScript. El resto del informe se puede leer sin él.</p>"
    "</noscript></div>"
  )


# --- Resumen --------------------------------------------------------------------


def _summary_panel(record: RunRecord, ids: Sequence[str]) -> str:
  questions = record["questions"]
  codes = [item["rules"]["verdict"] if item["rules"] else None for item in questions]
  rated = [code for code in codes if code and RULE_TONES.get(code, "info") != "info"]
  good = sum(1 for code in rated if RULE_TONES.get(code) == "good")
  judged = [item["judge"] for item in questions if item["judge"] is not None]
  judge_good = sum(1 for item in judged if item["verdict"] == "correct")
  mean = (
    sum(item["seconds"] for item in questions) / len(questions) if questions else 0.0
  )
  tiles = (
    ("Correctas por reglas", f"{good}/{len(rated)}" if rated else "—"),
    ("Correctas por juez", f"{judge_good}/{len(judged)}" if judged else "—"),
    ("Fallos de recuperación", _count(codes, _RETRIEVAL_FAILURES)),
    ("Fallos de presupuesto", _count(codes, _BUDGET_FAILURES)),
    ("Fallos de generación", _count(codes, _GENERATION_FAILURES)),
    ("Reglas y juez discrepan", str(sum(1 for item in questions if disagree(item)))),
    ("Tiempo medio por pregunta", seconds(mean)),
  )
  tile_html = "".join(
    f'<div class="tile"><span class="k">{esc(label)}</span>'
    f'<span class="big">{esc(value)}</span></div>'
    for label, value in tiles
  )
  rows = [
    _summary_row(item, question_id)
    for question_id, item in zip(ids, questions, strict=True)
  ]
  results = table(
    [
      ("id", False),
      ("pregunta", False),
      ("reglas", False),
      ("juez", False),
      ("discrepan", False),
      ("1.ª evidencia", False),
      ("tiempo", False),
    ],
    rows,
  )
  environment = record["environment"]
  setup = grid(
    [("Python", esc(environment["python"]))]
    + [(package, esc(version)) for package, version in environment["packages"].items()]
    + [
      ("Ollama", esc(environment["ollama_host"])),
      ("Qdrant", esc(environment["qdrant_url"])),
    ]
  )
  return (
    _head(record["title"], record["description"])
    + f'<div class="tiles">{tile_html}</div>'
    + card("Resultados", results if rows else '<p class="muted">Sin preguntas.</p>')
    + card("Archivo de ejecución resuelto", json_block(record["config"]))
    + card("Entorno", setup)
  )


def _count(codes: Sequence[str | None], wanted: Sequence[str]) -> str:
  return str(sum(1 for code in codes if code in wanted))


def _summary_row(item: QuestionRecord, question_id: str) -> str:
  key = f"q-{question_id}"
  rules = item["rules"]
  judge = item["judge"]
  rank = rules["retrieved_rank"] if rules else None
  return (
    f'<tr><td><a href="#{esc(key)}" data-goto="{esc(key)}">'
    f"<b>{esc(question_id)}</b></a></td>"
    f"<td>{esc(item['spec'].get('question', ''))}</td>"
    f"<td>{status_pill(rules['verdict'] if rules else None)}</td>"
    f"<td>{status_pill(judge['verdict'] if judge else None, 'judge')}</td>"
    f"<td>{_MARK if disagree(item) else ''}</td>"
    f"<td>{f'#{rank}' if rank else NO_DATA}</td>"
    f"<td>{esc(seconds(item['seconds']))}</td></tr>"
  )


# --- Corpus e ingesta -------------------------------------------------------------


def _corpus_panel(record: RunRecord) -> str:
  corpus = record["corpus"]
  ingest = record["ingest"]
  collection = record["collection"]
  ingested = {item["source"]: item for item in ingest["documents"]}
  rows: list[str] = []
  for document in corpus["documents"]:
    done = ingested.get(document["source"])
    rows.append(
      f"<tr><td><code>{esc(document['source'])}</code></td>"
      f'<td class="num-cell">{document["chars"]}</td>'
      f'<td class="num-cell">{document["chunks"]}</td>'
      f"<td>{yes_no(done['cached']) if done else NO_DATA}</td>"
      f'<td class="num-cell">{esc(seconds(done["seconds"])) if done else "—"}</td></tr>'
    )
  documents = table(
    [
      ("fuente", False),
      ("caracteres", True),
      ("chunks", True),
      ("en caché", False),
      ("segundos", True),
    ],
    rows,
  )
  chunker = grid([(key, _value(value)) for key, value in ingest["chunker"].items()])
  embedding = ingest["embedding"]
  embedding_rows = [
    ("Proveedor", esc(embedding.get("provider", "?"))),
    ("Modelo", esc(embedding.get("model", "?"))),
    ("Dimensión", esc(embedding.get("dimension", collection["vector_size"]))),
  ]
  state = (
    "recargada en esta ejecución" if collection["reloaded"] else "reutilizada tal cual"
  )
  collection_rows = [
    ("Nombre", f"<code>{esc(collection['name'])}</code>"),
    ("URL", esc(collection["url"])),
    ("Puntos", esc(collection["points"])),
    ("Tamaño del vector", esc(collection["vector_size"])),
    ("Distancia", esc(collection["distance"])),
    ("Índices", esc(", ".join(collection["indexes"]) or "—")),
    ("Estado", esc(state)),
  ]
  intro = (
    f"<code>{esc(corpus['dir'])}</code> · {esc(corpus['glob'])} · "
    f"{len(corpus['documents'])} documentos · ingesta {esc(seconds(ingest['seconds']))} · "
    f"ingester {esc(ingest['ingester_version'])}"
  )
  return (
    _head("Corpus e ingesta")
    + f'<p class="goal">{intro}</p>'
    + card("Documentos", documents)
    + '<div class="grid2">'
    + card("Argumentos del chunker", chunker)
    + card("Embeddings", grid(embedding_rows))
    + "</div>"
    + card("Tamaño de los chunks (caracteres)", _size_table(record["points"]))
    + card("Colección de Qdrant", grid(collection_rows))
  )


def _value(value: Any) -> str:
  if isinstance(value, bool):
    return "sí" if value else "no"
  if isinstance(value, str):
    return esc(value)
  return esc(json.dumps(value, ensure_ascii=False))


def _size_table(points: Sequence[PointRecord]) -> str:
  """Min, median and max of ``chunk_size`` per document, with a bar each."""
  by_source: dict[str, list[int]] = {}
  for point in points:
    size = point["payload"].get("chunk_size")
    if isinstance(size, int) and not isinstance(size, bool):
      by_source.setdefault(str(point["payload"].get("source", "?")), []).append(size)
  if not by_source:
    return '<p class="muted">Sin datos de tamaño.</p>'
  everything = [size for sizes in by_source.values() for size in sizes]
  scale = max(everything) or 1
  groups = [*by_source.items(), ("todos", everything)]
  rows: list[str] = []
  for source, sizes in groups:
    low, mid, high = min(sizes), statistics.median(sizes), max(sizes)
    rows.append(
      f"<tr><td><code>{esc(source)}</code></td>"
      f'<td class="num-cell">{len(sizes)}</td>'
      f'<td class="num-cell">{low}</td><td class="num-cell">{_number(mid)}</td>'
      f'<td class="num-cell">{high}</td>'
      f'<td><div class="sbar" title="mediana {_number(mid)}, máximo {high}">'
      f'<div class="sbar-max" style="width:{100 * high / scale:.1f}%"></div>'
      f'<div class="sbar-med" style="width:{100 * mid / scale:.1f}%"></div>'
      "</div></td></tr>"
    )
  return table(
    [
      ("fuente", False),
      ("chunks", True),
      ("mín", True),
      ("mediana", True),
      ("máx", True),
      ("mediana y máximo", False),
    ],
    rows,
  )

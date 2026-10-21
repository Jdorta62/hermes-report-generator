"""The ``report.html`` of a run: a pure function of the run record."""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from hermes_report.record import (
  PointRecord,
  QuestionRecord,
  RecordError,
  RuleRecord,
  RunRecord,
)
from hermes_report.render.html import (
  RULE_TONES,
  answer_html,
  esc,
  json_block,
  page,
  seconds,
  status_pill,
)

__all__ = ["render_run", "write_report"]

# With more questions than this the tabs show only the id.
_COMPACT_ABOVE: Final = 6
_TAB_TEXT: Final = 40
_NO_DATA: Final = '<span class="muted">—</span>'
_MARK: Final = '<span class="status mark" title="reglas y juez discrepan">≠</span>'
_BUDGET_FAILURES: Final = ("budget_failure",)
_RETRIEVAL_FAILURES: Final = ("retrieval_failure",)
_GENERATION_FAILURES: Final = ("generation_refused", "generation_wrong")


def render_run(record: RunRecord) -> str:
  """Render the whole report of ``record`` as one self-contained HTML document.

  The result depends only on ``record``: the same record always gives the same
  bytes, and nothing in the page refers to an external URL.
  """
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
    ("chunks", '<div id="explorer"></div>'),
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
    panels.append((key, _question_panel(item, question_id)))
  return page(record["title"], _meta(record), tabs, panels)


def write_report(record: RunRecord, path: Path) -> None:
  """Render ``record`` and write it as UTF-8 HTML, creating the parent folder.

  Raises:
    RecordError: if the file cannot be written; the error names the file.

  Effects: io
  """
  html = render_run(record)
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
  except OSError as exc:
    raise RecordError(path, f"no se puede escribir: {exc}") from None


# --- shared bits ---------------------------------------------------------------


def _question_id(item: QuestionRecord, number: int) -> str:
  return str(item["spec"].get("id") or f"q{number:02d}")


def _clip(text: str, limit: int) -> str:
  return text if len(text) <= limit else text[: limit - 1] + "…"


def _number(value: float) -> str:
  return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _yes_no(value: bool | None) -> str:
  if value is None:
    return _NO_DATA
  return (
    '<span class="status good">sí</span>'
    if value
    else '<span class="status bad">no</span>'
  )


def _rows(rows: Sequence[tuple[str, str]]) -> str:
  """A label/value grid; labels are plain text, values are HTML."""
  return (
    '<div class="verdict">'
    + "".join(
      f'<span class="k">{esc(label)}</span><span>{value}</span>'
      for label, value in rows
    )
    + "</div>"
  )


def _card(title: str, body: str) -> str:
  return f'<div class="card"><h3>{esc(title)}</h3>{body}</div>'


def _head(title: str, goal: str = "") -> str:
  return f'<div class="sc-head"><h2>{esc(title)}</h2></div>' + (
    f'<p class="goal">{esc(goal)}</p>' if goal else ""
  )


def _table(headers: Sequence[tuple[str, bool]], rows: Sequence[str]) -> str:
  """A table; ``headers`` are ``(text, right-aligned)``, ``rows`` are ``<tr>``s."""
  head = "".join(
    f'<th class="num-cell">{esc(text)}</th>' if right else f"<th>{esc(text)}</th>"
    for text, right in headers
  )
  return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def _tone(rules: RuleRecord | None) -> str | None:
  return None if rules is None else RULE_TONES.get(rules["verdict"], "info")


def _disagree(item: QuestionRecord) -> bool:
  """The rules say right and the judge wrong, or the other way round."""
  tone = _tone(item["rules"])
  judged = item["judge"]["verdict"] if item["judge"] else None
  return (tone == "good" and judged == "incorrect") or (
    tone == "bad" and judged == "correct"
  )


def _meta(record: RunRecord) -> str:
  config = record["config"]
  judge = config.get("judge", {})
  judge_text = esc(judge.get("model", "?")) if judge.get("enabled") else "sin juez"
  collection = record["collection"]
  return (
    f"Generación: <b>{esc(config.get('generation', {}).get('model', '?'))}</b>"
    f" · Juez: <b>{judge_text}</b>"
    f" · Qdrant: <b>{esc(collection['url'])}</b> / {esc(collection['name'])}"
    f" · {esc(record['started_at'])} · {esc(seconds(record['seconds']))}"
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
    ("Reglas y juez discrepan", str(sum(1 for item in questions if _disagree(item)))),
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
  table = _table(
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
  setup = _rows(
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
    + _card("Resultados", table if rows else '<p class="muted">Sin preguntas.</p>')
    + _card("Archivo de ejecución resuelto", json_block(record["config"]))
    + _card("Entorno", setup)
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
    f"<td>{_MARK if _disagree(item) else ''}</td>"
    f"<td>{f'#{rank}' if rank else _NO_DATA}</td>"
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
      f"<td>{_yes_no(done['cached']) if done else _NO_DATA}</td>"
      f'<td class="num-cell">{esc(seconds(done["seconds"])) if done else "—"}</td></tr>'
    )
  documents = _table(
    [
      ("fuente", False),
      ("caracteres", True),
      ("chunks", True),
      ("en caché", False),
      ("segundos", True),
    ],
    rows,
  )
  chunker = _rows([(key, _value(value)) for key, value in ingest["chunker"].items()])
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
    + _card("Documentos", documents)
    + '<div class="grid2">'
    + _card("Argumentos del chunker", chunker)
    + _card("Embeddings", _rows(embedding_rows))
    + "</div>"
    + _card("Tamaño de los chunks (caracteres)", _size_table(record["points"]))
    + _card("Colección de Qdrant", _rows(collection_rows))
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
  return _table(
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


# --- one question -----------------------------------------------------------------


def _question_panel(item: QuestionRecord, question_id: str) -> str:
  spec = item["spec"]
  generation = item["generation"]
  result = generation["result"] if generation else None
  pill = status_pill(result["status"] if result else "error", "generation")
  error = f'<div class="error">{esc(item["error"])}</div>' if item["error"] else ""
  return (
    f'<div class="sc-head"><h2>{esc(spec.get("question", ""))}</h2>{pill}'
    f'<span class="muted">{esc(question_id)}</span>'
    f'<span class="muted">{esc(seconds(item["seconds"]))}</span></div>'
    + error
    + _evaluation_card(item)
  )


def _evaluation_card(item: QuestionRecord) -> str:
  spec = item["spec"]
  rules = item["rules"]
  judge = item["judge"]
  generation = item["generation"]
  result = generation["result"] if generation else None
  retrieval = item["retrieval"]
  rows: list[tuple[str, str]] = [
    ("Veredicto", status_pill(rules["verdict"] if rules else None))
  ]
  if result is not None:
    rows.append(
      ("Respuesta", f'<div class="answer">{answer_html(str(result["answer"]))}</div>')
    )
  rows.append(("Esperada", esc(spec.get("expected", ""))))
  if spec.get("refusal"):
    rows.append(("Se espera rechazo", "sí"))
  rows.append(("Palabras clave", esc(", ".join(spec.get("keywords", ())) or "—")))
  evidence = spec.get("evidence", ())
  if evidence:
    rows.append(("Evidencia", f"«{esc(' … '.join(evidence))}»"))
  documents = spec.get("documents", ())
  if documents:
    rows.append(
      (
        "Documentos esperados",
        f"{esc(', '.join(documents))} "
        f"{_yes_no(rules['expected_docs_retrieved'] if rules else None)}",
      )
    )
  if rules is not None:
    rank = rules["retrieved_rank"]
    total = len(retrieval["chunks"]) if retrieval else 0
    rows += [
      (
        "¿Recuperada?",
        _yes_no(rules["in_retrieval"])
        + (
          f' <span class="muted">primer chunk con la evidencia: #{rank} de {total}'
          "</span>"
          if rank
          else ""
        ),
      ),
      ("¿En el contexto final?", _yes_no(rules["in_context"])),
      ("¿En una pieza citada?", _yes_no(rules["in_cited"])),
      ("¿Respuesta correcta?", _yes_no(rules["answer_ok"])),
    ]
  if retrieval is not None:
    rows.append(
      (
        "Recuperación",
        f"{len(retrieval['chunks'])} chunks · {esc(seconds(retrieval['seconds']))}",
      )
    )
  rows.append(
    (
      "Juez",
      status_pill(judge["verdict"], "judge")
      if judge
      else "sin juez para esta pregunta",
    )
  )
  if judge is not None:
    rows += [
      ("Motivo del juez", esc(judge["reason"] or "—")),
      ("Modelo del juez", esc(judge["model"])),
      ("Tiempo del juez", esc(seconds(judge["seconds"]))),
    ]
    if _disagree(item):
      rows.append(
        ("Discrepancia", '<span class="status mark">reglas y juez discrepan</span>')
      )
  body = _rows(rows)
  if judge is not None and judge["error"]:
    body += f'<div class="error">{esc(judge["error"])}</div>'
  if judge is not None and judge["raw"] is not None:
    body += (
      '<details class="raw-reply"><summary>Respuesta cruda del juez</summary>'
      f'<pre class="text raw">{esc(judge["raw"])}</pre></details>'
    )
  return _card("Evaluación", body)

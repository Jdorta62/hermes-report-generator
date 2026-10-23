"""The ``compare.html`` of several runs: a pure function of their run records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hermes_report.record import QuestionRecord, RunRecord
from hermes_report.render.html import (
  NO_DATA,
  card,
  esc,
  link,
  metric_label,
  page,
  seconds,
  status_pill,
  table,
  write_html,
)
from hermes_report.render.score import (
  format_score,
  metric_tiles,
  question_id,
  score_cell,
  score_tiles,
)

__all__ = ["render_compare", "write_compare"]

_PACKAGES_KEY = "environment.packages"


def render_compare(runs: Sequence[tuple[str, RunRecord]]) -> str:
  """Render several runs side by side as one self-contained HTML document.

  Each pair is the link to that run's ``report.html``, relative to the
  comparison file, and its record. The page shows the runs, the run-file keys
  and package versions that differ, the summary tiles of each run and a matrix
  with one row per question id. The result depends only on ``runs``: the same
  input always gives the same bytes, and the generation time is the latest
  ``finished_at``, never the clock.

  Raises:
    ValueError: if ``runs`` is empty.
  """
  if not runs:
    raise ValueError("render_compare needs at least one run")
  count = len(runs)
  title = f"Comparación de {count} {'ejecución' if count == 1 else 'ejecuciones'}"
  labels = [_label(index) for index in range(count)]
  latest = max(record["finished_at"] for _, record in runs)
  meta = f"Generada el {esc(latest)}"
  tabs = [
    ("resumen", "Resumen", "Ejecuciones, diferencias y puntuación"),
    ("matriz", "Matriz", "Cada pregunta en cada ejecución"),
  ]
  panels = [
    ("resumen", _summary_panel(runs, labels)),
    ("matriz", _matrix_panel(runs, labels)),
  ]
  return page(title, meta, tabs, panels)


def write_compare(runs: Sequence[tuple[str, RunRecord]], path: Path) -> None:
  """Render the comparison of ``runs`` and write it as UTF-8 HTML to ``path``.

  Raises:
    RecordError: if the file cannot be written or the page cannot be encoded
      as UTF-8; the error names the file.
    ValueError: if ``runs`` is empty.

  Effects: io
  """
  write_html(render_compare(runs), path)


# --- shared bits ---------------------------------------------------------------


def _label(index: int) -> str:
  """``A``..``Z``, then ``AA``, ``AB``... like spreadsheet columns."""
  name = ""
  number = index + 1
  while number:
    number, rest = divmod(number - 1, 26)
    name = chr(ord("A") + rest) + name
  return name


def _head(title: str) -> str:
  return f'<div class="sc-head"><h2>{esc(title)}</h2></div>'


def _columns(labels: Sequence[str], first: str = "") -> list[tuple[str, bool]]:
  return [(first, False), *((label, False) for label in labels)]


# --- Resumen ------------------------------------------------------------------------


def _summary_panel(runs: Sequence[tuple[str, RunRecord]], labels: Sequence[str]) -> str:
  return (
    _head("Resumen")
    + card("Ejecuciones", _runs_table(runs, labels))
    + card("Diferencias de configuración", _differences(runs, labels))
    + card("Puntuación", _score_table(runs, labels))
  )


def _runs_table(runs: Sequence[tuple[str, RunRecord]], labels: Sequence[str]) -> str:
  rows = [
    "<tr><td>Título</td>"
    + "".join(f"<td>{esc(record['title'])}</td>" for _, record in runs)
    + "</tr>",
    "<tr><td>Inicio</td>"
    + "".join(f"<td>{esc(record['started_at'])}</td>" for _, record in runs)
    + "</tr>",
    "<tr><td>Informe</td>"
    + "".join(
      f"<td>{link(report, 'Abrir informe', 'raw-link')}</td>" for report, _ in runs
    )
    + "</tr>",
  ]
  return table(_columns(labels), rows)


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
  """``value`` as dotted keys to text: strings as they are, the rest as JSON."""
  if isinstance(value, Mapping) and value:
    flat: dict[str, str] = {}
    for key, item in value.items():
      flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    return flat
  if not prefix:
    return {}
  text = (
    value
    if isinstance(value, str)
    else json.dumps(value, ensure_ascii=False, default=str)
  )
  return {prefix: text}


def _differing(
  flats: Sequence[Mapping[str, str]],
) -> list[tuple[str, list[str | None]]]:
  """The keys whose values are not the same in every run, in first-seen order."""
  keys = list(dict.fromkeys(key for flat in flats for key in flat))
  rows = [(key, [flat.get(key) for flat in flats]) for key in keys]
  return [(key, values) for key, values in rows if len(set(values)) > 1]


def _differences(runs: Sequence[tuple[str, RunRecord]], labels: Sequence[str]) -> str:
  config = _differing([_flatten(record["config"]) for _, record in runs])
  packages = _differing(
    [
      _flatten(record["environment"].get("packages", {}), _PACKAGES_KEY)
      for _, record in runs
    ]
  )
  rows = [
    f"<tr><td><code>{esc(key)}</code></td>"
    + "".join(
      f"<td>{NO_DATA if value is None else f'<code>{esc(value)}</code>'}</td>"
      for value in values
    )
    + "</tr>"
    for key, values in config + packages
  ]
  if not rows:
    return (
      '<p class="muted">Las ejecuciones no se diferencian en la configuración '
      "ni en las versiones de los paquetes.</p>"
    )
  return table(_columns(labels, "clave"), rows)


def _score_table(runs: Sequence[tuple[str, RunRecord]], labels: Sequence[str]) -> str:
  tiles = [score_tiles(record["questions"]) for _, record in runs]
  rows = [
    f"<tr><td>{esc(tiles[0][position][0])}</td>"
    + "".join(f'<td class="num-cell">{esc(run[position][1])}</td>' for run in tiles)
    + "</tr>"
    for position in range(len(tiles[0]))
  ]
  # The metric means: one row per score any run has, a dash where a run lacks it.
  metrics = [dict(metric_tiles(record)) for _, record in runs]
  rows += [
    f"<tr><td>{esc(label)}</td>"
    + "".join(
      f'<td class="num-cell">{esc(run[label]) if label in run else NO_DATA}</td>'
      for run in metrics
    )
    + "</tr>"
    for label in dict.fromkeys(label for run in metrics for label in run)
  ]
  return table(
    [("", False), *((label, True) for label in labels)],
    rows,
  )


# --- Matriz ---------------------------------------------------------------------------


def _matrix_panel(runs: Sequence[tuple[str, RunRecord]], labels: Sequence[str]) -> str:
  by_run: list[dict[str, QuestionRecord]] = []
  texts: dict[str, str] = {}
  for _, record in runs:
    found: dict[str, QuestionRecord] = {}
    for number, item in enumerate(record["questions"], 1):
      qid = question_id(item, number)
      found.setdefault(qid, item)
      texts.setdefault(qid, str(item["spec"].get("question", "")))
    by_run.append(found)
  ids = list(dict.fromkeys(qid for found in by_run for qid in found))
  rows = [_matrix_row(qid, texts[qid], runs, by_run) for qid in ids]
  if rows:
    rows += _score_groups(ids, texts, runs, by_run)
  body = (
    table(
      [("id", False), ("pregunta", False), *((label, False) for label in labels)], rows
    )
    if rows
    else '<p class="muted">Sin preguntas.</p>'
  )
  return (
    _head("Matriz")
    + '<label class="matrix-filter" hidden><input type="checkbox" id="only-diff"> '
    "solo filas con diferencias</label>" + card("Preguntas por ejecución", body)
  )


def _matrix_row(
  qid: str,
  text: str,
  runs: Sequence[tuple[str, RunRecord]],
  by_run: Sequence[Mapping[str, QuestionRecord]],
) -> str:
  verdicts = {_rule_verdict(found[qid]) for found in by_run if qid in found}
  cells = "".join(
    f"<td>{_cell(report, qid, found[qid])}</td>"
    if qid in found
    else f"<td>{NO_DATA}</td>"
    for (report, _), found in zip(runs, by_run, strict=True)
  )
  css = ' class="diff"' if len(verdicts) > 1 else ""
  return (
    f'<tr{css} data-q="{esc(qid)}"><td><b>{esc(qid)}</b></td><td>{esc(text)}</td>'
    f"{cells}</tr>"
  )


def _scores(runs: Sequence[tuple[str, RunRecord]]) -> list[str]:
  """The scores any run recorded, in the order of their first appearance."""
  found: dict[str, None] = {}
  for _, record in runs:
    metrics = record.get("metrics")
    for score in metrics["scores"] if metrics else ():
      found.setdefault(score)
  return list(found)


def _mean(record: RunRecord, score: str) -> str:
  metrics = record.get("metrics")
  entry = metrics["summary"].get(score) if metrics else None
  if entry is None or entry["mean"] is None:
    return NO_DATA
  return esc(format_score(entry["mean"]))


def _score_groups(
  ids: Sequence[str],
  texts: Mapping[str, str],
  runs: Sequence[tuple[str, RunRecord]],
  by_run: Sequence[Mapping[str, QuestionRecord]],
) -> list[str]:
  """Per score, a row with its mean in each run, then one row per question.

  A run without metrics, or without the question, shows a dash.
  """
  rows: list[str] = []
  for score in _scores(runs):
    rows.append(
      f'<tr><td colspan="2"><b>{esc(metric_label(score))}</b> '
      '<span class="muted">media</span></td>'
      + "".join(
        f'<td class="num-cell"><b>{_mean(record, score)}</b></td>' for _, record in runs
      )
      + "</tr>"
    )
    for qid in ids:
      cells = "".join(
        f'<td class="num-cell">{score_cell(found[qid], score)}</td>'
        if qid in found
        else f"<td>{NO_DATA}</td>"
        for found in by_run
      )
      rows.append(f"<tr><td>{esc(qid)}</td><td>{esc(texts[qid])}</td>{cells}</tr>")
  return rows


def _rule_verdict(item: QuestionRecord) -> str | None:
  rules = item["rules"]
  return rules["verdict"] if rules else None


def _cell(report: str, qid: str, item: QuestionRecord) -> str:
  judge = item["judge"]
  return link(
    f"{report}#q-{qid}",
    f"{status_pill(_rule_verdict(item))} "
    f"{status_pill(judge['verdict'] if judge else None, 'judge')} "
    f'<span class="muted">{esc(seconds(item["seconds"]))}</span>',
  )

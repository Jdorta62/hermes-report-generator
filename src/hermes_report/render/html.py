"""HTML helpers shared by every renderer of the report, and the page itself."""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Sequence
from html import escape
from importlib import resources
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import quote

from hermes_report.record import RecordError

__all__ = [
  "GENERATION_LABELS",
  "JUDGE_LABELS",
  "NO_DATA",
  "RULE_LABELS",
  "RULE_TONES",
  "answer_html",
  "card",
  "chip",
  "deep_link",
  "esc",
  "json_block",
  "json_script",
  "page",
  "grid",
  "link",
  "section",
  "seconds",
  "status_pill",
  "table",
  "write_html",
  "yes_no",
]

RULE_LABELS: Final = {
  "error": "error",
  "correct_refusal": "rechazo correcto",
  "invented_answer": "inventa una respuesta",
  "correct": "acierto",
  "correct_without_evidence": (
    "acierto sin evidencia en el contexto (¿memoria del modelo?)"
  ),
  "budget_failure": "fallo de presupuesto: se recuperó y se descartó",
  "retrieval_failure": "fallo de recuperación",
  "generation_refused": "fallo de generación: se niega con la evidencia delante",
  "generation_wrong": "fallo de generación: responde mal",
  "unrated": "sin valorar: la pregunta no tiene palabras clave",
}
RULE_TONES: Final = {
  "error": "bad",
  "correct_refusal": "good",
  "invented_answer": "bad",
  "correct": "good",
  "correct_without_evidence": "meh",
  "budget_failure": "bad",
  "retrieval_failure": "bad",
  "generation_refused": "bad",
  "generation_wrong": "bad",
  "unrated": "info",
}
JUDGE_LABELS: Final = {
  "correct": "correcta",
  "partial": "parcial",
  "incorrect": "incorrecta",
}
GENERATION_LABELS: Final = {
  "answered": "respondida",
  "insufficient": "sin información suficiente",
  "no_context": "sin contexto",
  "error": "error",
}

NO_DATA: Final = '<span class="muted">—</span>'

_JUDGE_TONES: Final = {"correct": "good", "partial": "meh", "incorrect": "bad"}
_GENERATION_TONES: Final = {
  "answered": "good",
  "insufficient": "meh",
  "no_context": "info",
  "error": "bad",
}
_CITATION = re.compile(r"\[(\d+)\]")
_BOLD = re.compile(r"\*\*(.+?)\*\*")

_ASSETS = resources.files("hermes_report.render") / "assets"


def esc(text: object) -> str:
  """Escape ``text`` (``str()`` of it when it is not a string) for HTML."""
  return escape(text if isinstance(text, str) else str(text))


def section(anchor: str, step: str, title: str, summary: str, body: str) -> str:
  """A collapsible phase card: a step mark, a title and a summary above its body.

  ``anchor``, ``step``, ``title`` and ``summary`` are plain text and are
  escaped; ``body`` is HTML the caller has already escaped.
  """
  return (
    f'<details class="phase" id="{esc(anchor)}" open><summary>'
    f'<span class="step">{esc(step)}</span>'
    f'<span class="ph-title">{esc(title)}</span>'
    f'<span class="ph-sum">{esc(summary)}</span></summary>'
    f'<div class="ph-body">{body}</div></details>'
  )


def chip(anchor: str, title: str, summary: str) -> str:
  """A flow chip linking to the phase card ``anchor``; its texts are escaped."""
  return (
    f'<a class="chip" href="#{esc(anchor)}"><b>{esc(title)}</b>'
    f"<span>{esc(summary)}</span></a>"
  )


def card(title: str, body: str) -> str:
  """A titled card; ``title`` is plain text, ``body`` is HTML already escaped."""
  return f'<div class="card"><h3>{esc(title)}</h3>{body}</div>'


def grid(items: Sequence[tuple[str, str]]) -> str:
  """A label/value grid; the labels are plain text, the values are HTML."""
  return (
    '<div class="verdict">'
    + "".join(
      f'<span class="k">{esc(label)}</span><span>{value}</span>'
      for label, value in items
    )
    + "</div>"
  )


def table(headers: Sequence[tuple[str, bool]], body_rows: Sequence[str]) -> str:
  """A table; ``headers`` are ``(text, right-aligned)``, rows are ``<tr>`` HTML."""
  head = "".join(
    f'<th class="num-cell">{esc(text)}</th>' if right else f"<th>{esc(text)}</th>"
    for text, right in headers
  )
  return f"<table><tr>{head}</tr>{''.join(body_rows)}</table>"


def yes_no(value: bool | None) -> str:
  """A green ``sí``, a red ``no`` or a dash for ``None``."""
  if value is None:
    return NO_DATA
  return (
    '<span class="status good">sí</span>'
    if value
    else '<span class="status bad">no</span>'
  )


def status_pill(
  code: str | None, kind: Literal["rule", "judge", "generation"] = "rule"
) -> str:
  """A coloured pill with the Spanish label of ``code``; a dash for ``None``.

  ``kind`` says which family the code belongs to: a rule verdict, a judge
  verdict or a generation status. A code of that family that is not known
  keeps its own text and the neutral tone.
  """
  if code is None:
    return NO_DATA
  labels, tones = {
    "rule": (RULE_LABELS, RULE_TONES),
    "judge": (JUDGE_LABELS, _JUDGE_TONES),
    "generation": (GENERATION_LABELS, _GENERATION_TONES),
  }[kind]
  tone = tones.get(code, "info")
  return f'<span class="status {tone}">{esc(labels.get(code, code))}</span>'


def answer_html(text: str) -> str:
  """The answer text escaped, with bold and the ``[n]`` markers as chips."""
  html = esc(text)
  html = _BOLD.sub(r"<strong>\1</strong>", html)
  return _CITATION.sub(r'<span class="cite">\1</span>', html)


def json_block(value: Any) -> str:
  """``value`` as an indented, escaped JSON text block."""
  text = json.dumps(value, ensure_ascii=False, indent=2)
  return f'<pre class="text json">{esc(text)}</pre>'


def json_script(element_id: str, value: Any) -> str:
  """``value`` as a JSON ``<script>`` block that cannot close itself.

  Every ``<`` of the JSON text is written as ``\\u003c``, so no text of the value
  (``</script>``, ``<!--``) can end or confuse the block; ``JSON.parse`` reads it back.
  ``element_id`` is plain text and is escaped.
  """
  text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
  safe = text.replace("<", "\\u003c")
  return f'<script type="application/json" id="{esc(element_id)}">{safe}</script>\n'


def deep_link(
  *,
  doc: str | None = None,
  question: str | None = None,
  chunk: str | None = None,
  used: bool = False,
) -> str:
  """The href ``#chunks?doc=<source>&q=<id>&chunk=<chunk_id>`` of the explorer.

  Only the parts given appear; ``used`` adds ``used=1`` (the explorer then shows
  only the chunks the question used). Values are percent-encoded; the result is
  not HTML-escaped, ``link`` does that.
  """
  params = [
    ("doc", doc),
    ("q", question),
    ("chunk", chunk),
    ("used", "1" if used else None),
  ]
  query = "&".join(f"{key}={quote(value, safe='')}" for key, value in params if value)
  return f"#chunks?{query}" if query else "#chunks"


def link(href: str, inner: str, css: str = "chunk-link") -> str:
  """An anchor to ``href`` (plain text, escaped) around ``inner`` (HTML)."""
  return f'<a class="{esc(css)}" href="{esc(href)}">{inner}</a>'


def seconds(value: float) -> str:
  """A duration with one decimal, as ``"4.2 s"``."""
  return f"{value:.1f} s"


def write_html(html: str, path: Path) -> None:
  """Write ``html`` as UTF-8 to ``path``, creating the parent folder.

  The text goes to a temporary file in the same folder and is then moved over
  ``path``, so a failure keeps the previous file intact.

  Raises:
    RecordError: if the file cannot be written or the text cannot be encoded
      as UTF-8 (a lone surrogate); the error names the file.

  Effects: io
  """
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


def page(
  title: str,
  meta: str,
  tabs: Sequence[tuple[str, str, str]],
  panels: Sequence[tuple[str, str]],
  data: str = "",
) -> str:
  """Assemble the self-contained document: styles and script inlined.

  Args:
    title: The report title, plain text.
    meta: The meta line under the title, HTML the caller has already escaped.
    tabs: One ``(key, label, hover text)`` per tab button, in order.
    panels: One ``(key, body HTML)`` per tab, wrapped in a ``panel`` section
      with ``id="tab-<key>"``.
    data: HTML placed before the script, such as ``json_script`` blocks that
      the script reads; the caller has already escaped it.

  Without JavaScript every panel is shown one after the other.
  """
  buttons = "".join(
    f'<button class="tab" data-key="{esc(key)}" title="{esc(hover)}">'
    f"{esc(label)}</button>"
    for key, label, hover in tabs
  )
  sections = "".join(
    f'<section class="panel" id="tab-{esc(key)}" data-key="{esc(key)}">{body}</section>'
    for key, body in panels
  )
  css = (_ASSETS / "report.css").read_text(encoding="utf-8")
  script = (_ASSETS / "report.js").read_text(encoding="utf-8")
  return (
    '<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    f"<title>Hermes Report · {esc(title)}</title>\n<style>\n{css}</style>\n"
    "<noscript><style>.panel { display: block; } nav { display: none; }</style></noscript>\n"
    "</head>\n<body>\n<header>\n"
    f"  <h1>Hermes Report <span>· {esc(title)}</span></h1>\n"
    f'  <div class="meta">{meta}</div>\n  <nav>{buttons}</nav>\n</header>\n'
    f"<main>{sections}</main>\n{data}<script>\n{script}</script>\n</body>\n</html>\n"
  )

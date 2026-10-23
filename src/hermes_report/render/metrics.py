"""The "Métricas" section of a question tab: scores, details and judge calls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from hermes_report.record import JudgeCallRecord, MetricResultRecord
from hermes_report.render.html import (
  NO_DATA,
  esc,
  grid,
  json_block,
  messages_block,
  metric_label,
  seconds,
  table,
)

__all__ = ["metrics_body", "metrics_summary"]

_STATUS: Final = {
  "ok": ("calculada", "good"),
  "not_applicable": ("no aplicable", "info"),
  "skipped": ("omitida", "meh"),
  "error": ("error", "bad"),
}
_DETAIL_LABELS: Final = {
  "claims": "Afirmaciones",
  "verdicts": "Veredictos",
  "answer_claims": "Afirmaciones de la respuesta",
  "expected_claims": "Afirmaciones de la respuesta esperada",
  "answer_vs_expected": "Respuesta frente a la esperada",
  "expected_vs_answer": "Esperada frente a la respuesta",
  "answer_vs_retrieved": "Respuesta frente a lo recuperado",
  "noisy_claims": "Afirmaciones respaldadas solo por ruido",
  "questions": "Preguntas generadas",
  "similarities": "Similitudes con la pregunta",
  "noncommittal": "Respuesta evasiva",
  "refused": "Se negó a responder",
  "refusal": "Pregunta que exige rechazo",
  "relevant": "Chunks relevantes (por rango)",
  "citations": "Citas",
  "support": "Respaldo de las afirmaciones",
  "found_at": "Rango donde aparece cada frase",
  "expected": "Documentos esperados",
  "found": "Encontrados",
  "first_rank": "Primer rango",
  "missing": "Datos que faltan",
  "reason": "Motivo",
  "counter": "Contador de tokens",
  "entities": "Entidades",
  "no_expected_claims": "La respuesta esperada no tiene afirmaciones",
  "candidate_tokens": "Palabras de la respuesta",
  "reference_tokens": "Palabras de la esperada",
}
_FIELD_LABELS: Final = {
  "text": "texto",
  "markers": "marcadores",
  "claim": "afirmación",
  "supported": "respaldada",
  "sources": "fuentes",
  "marker": "marcador",
  "piece": "pieza",
}


def metrics_summary(results: Sequence[MetricResultRecord]) -> str:
  """One line for the flow chip: how many families were computed and how many failed."""
  done = sum(1 for result in results if result["status"] == "ok")
  failed = sum(1 for result in results if result["status"] == "error")
  text = f"{done}/{len(results)} calculadas"
  return f"{text}, {failed} con error" if failed else text


def metrics_body(results: Sequence[MetricResultRecord]) -> str:
  """The body of the section: one block per metric family, in record order."""
  return "".join(_family(result) for result in results)


def _family(result: MetricResultRecord) -> str:
  label, tone = _STATUS.get(result["status"], (result["status"], "info"))
  head = (
    f"<h4><code>{esc(result['name'])}</code> "
    f'<span class="status {tone}">{esc(label)}</span> '
    f'<span class="muted">{esc(seconds(result["seconds"]))}</span></h4>'
  )
  body = ""
  if result["error"]:
    body += f'<div class="error">{esc(result["error"])}</div>'
  if result["scores"]:
    body += grid(
      [
        (metric_label(score), _value(value))
        for score, value in result["scores"].items()
      ]
    )
  if result["details"]:
    body += _details(result["details"])
  if result["calls"]:
    body += _calls(result["calls"])
  return head + body


def _details(details: Mapping[str, Any]) -> str:
  return (
    "<details><summary>Detalles</summary>"
    + grid(
      [(_DETAIL_LABELS.get(key, key), _render(value)) for key, value in details.items()]
    )
    + "</details>"
  )


def _calls(calls: Sequence[JudgeCallRecord]) -> str:
  items = "".join(_call(call) for call in calls)
  return f"<details><summary>Llamadas al juez ({len(calls)})</summary>{items}</details>"


def _call(call: JudgeCallRecord) -> str:
  raw = (
    f'<pre class="text raw">{esc(call["raw"])}</pre>'
    if call["raw"] is not None
    else '<p class="muted">Sin respuesta.</p>'
  )
  error = f'<div class="error">{esc(call["error"])}</div>' if call["error"] else ""
  return (
    f"<h4>{esc(call['purpose'])} "
    f'<span class="muted">{esc(seconds(call["seconds"]))}</span></h4>'
    + error
    + messages_block(call["messages"])
    + "<h4>Respuesta cruda</h4>"
    + raw
  )


# --- values -------------------------------------------------------------------------


def _value(value: float | None) -> str:
  return NO_DATA if value is None else esc(f"{value:.2f}")


def _scalar(value: object) -> str:
  """Plain text of a scalar value (not escaped)."""
  if value is None:
    return "—"
  if isinstance(value, bool):
    return "sí" if value else "no"
  if isinstance(value, float):
    return f"{value:.2f}"
  return value if isinstance(value, str) else str(value)


def _is_scalar(value: object) -> bool:
  return value is None or isinstance(value, str | int | float | bool)


def _text(value: Any) -> str:
  """A table cell: a scalar, a list of scalars joined, or compact JSON."""
  if _is_scalar(value):
    return _scalar(value)
  if isinstance(value, list) and all(_is_scalar(item) for item in value):
    return ", ".join(_scalar(item) for item in value) or "—"
  return json.dumps(value, ensure_ascii=False)


def _render(value: Any) -> str:
  """The HTML of a detail value, readable for the shapes the metrics record."""
  if _is_scalar(value):
    return esc(_scalar(value))
  if isinstance(value, list):
    if not value:
      return '<span class="muted">ninguno</span>'
    if all(isinstance(item, dict) for item in value):
      keys = list(dict.fromkeys(key for item in value for key in item))
      rows = [
        "<tr>"
        + "".join(f"<td>{esc(_text(item.get(key)))}</td>" for key in keys)
        + "</tr>"
        for item in value
      ]
      return table([(_FIELD_LABELS.get(key, key), False) for key in keys], rows)
    if all(_is_scalar(item) for item in value):
      return esc(_text(value))
  if (
    isinstance(value, dict)
    and value
    and all(_is_scalar(item) for item in value.values())
  ):
    rows = [
      f"<tr><td>{esc(key)}</td><td>{esc(_scalar(item))}</td></tr>"
      for key, item in value.items()
    ]
    return table([("clave", False), ("valor", False)], rows)
  return json_block(value)

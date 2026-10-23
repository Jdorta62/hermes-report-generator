"""Metrics in the report and the comparison (task 5.5.5).

The reference record comes from a real offline run over ``tests/data`` (hash
embedder, echo LLM, echo judge). Tests that need another shape deep-copy it and
edit the copy: rendering is a pure function of the record.
"""

from __future__ import annotations

import copy
import html as html_lib
import json
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

import pytest
from conftest import DATA
from test_render_run import Page, panel_of, parse

from hermes_report import runner
from hermes_report.config import Overrides, load_config
from hermes_report.metrics import METRICS
from hermes_report.record import QuestionRecord, RunRecord, read_record
from hermes_report.render import html as h
from hermes_report.render import render_compare, render_run
from hermes_report.render.metrics import metrics_body, metrics_summary
from hermes_report.render.score import (
  format_score,
  metric_tiles,
  primary_scores,
  score_cell,
  score_tiles,
)

HOSTILE = '<script>alert("x")</script><img src=x onerror=alert(1)>&amp;"\''
SCHEMA1 = Path(__file__).parent / "fixtures" / "schema1"
DEFAULT_ON = [name for name, metric in METRICS.items() if metric.default_enabled]

Runs = list[tuple[str, RunRecord]]


# ------------------------------------------------------------------ helpers


@pytest.fixture(scope="module")
def real_record(tmp_path_factory: pytest.TempPathFactory) -> RunRecord:
  project = tmp_path_factory.mktemp("proj") / "proj"
  shutil.copytree(DATA, project)
  config = load_config(project / "run.toml", Overrides(out=project / "out"))
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    return runner.run(config, progress=lambda line: None).record


@pytest.fixture
def record(real_record: RunRecord) -> RunRecord:
  return copy.deepcopy(real_record)


def schema1() -> RunRecord:
  return read_record(SCHEMA1 / "run.json")


def metrics_of(record: RunRecord) -> Any:
  found = record["metrics"]
  assert found is not None
  return found


def results_of(item: QuestionRecord) -> Any:
  assert item["metrics"] is not None
  return item["metrics"]


def result(item: QuestionRecord, name: str) -> Any:
  return next(r for r in results_of(item) if r["name"] == name)


def tile_pairs(text: str) -> dict[str, str]:
  found = re.findall(
    r'<div class="tile"><span class="k">([^<]*)</span><span class="big small">([^<]*)</span>',
    text,
  )
  return {html_lib.unescape(k): html_lib.unescape(v) for k, v in found}


def outside_script(text: str) -> str:
  """The page without the inlined style and script (assets are not user text)."""
  body = text[text.index("</style>") :]
  return body[: body.index("<script>")] + body[body.rindex("</script>") :]


def assert_no_injection(text: str) -> None:
  page = outside_script(text)
  assert HOSTILE not in page
  assert "<script>alert" not in page
  assert "<img" not in page
  parsed = parse(text)
  assert len(parsed.tags("img")) == 0
  # only the report script (and the JSON data blocks) may exist
  for attrs in parsed.tags("script"):
    assert attrs.get("type") in {None, "application/json"}
  for _, attrs in parsed.elements:
    assert not any(name.startswith("on") for name in attrs), attrs


# ------------------------------------------------------------------ labels


def test_every_score_of_every_family_has_a_spanish_label() -> None:
  scores = [score for metric in METRICS.values() for score in metric.scores]
  assert set(scores) <= set(h.METRIC_LABELS)
  assert set(h.METRIC_LABELS) <= set(scores)  # no stale label
  assert all(h.METRIC_LABELS[s].strip() for s in scores)
  assert len(set(h.METRIC_LABELS.values())) == len(h.METRIC_LABELS)
  assert h.LOWER_IS_BETTER <= set(h.METRIC_LABELS)


def test_lower_is_better_is_what_the_design_marks() -> None:
  assert h.LOWER_IS_BETTER == {
    "budget_loss",
    "false_refusal",
    "invented_answer",
    "seconds_retrieval",
    "seconds_enrich",
    "seconds_generation",
    "context_tokens",
    "prompt_tokens",
    "reply_tokens",
    "noise_sensitivity",
    "toxicity",
    "bias",
  }


def test_metric_label_marks_only_the_lower_is_better_scores() -> None:
  for score, label in h.METRIC_LABELS.items():
    shown = h.metric_label(score)
    if score in h.LOWER_IS_BETTER:
      assert shown == f"{label} ↓"
    else:
      assert shown == label
  assert h.metric_label("not_a_score") == "not_a_score"
  assert h.metric_label("") == ""


# ------------------------------------------------------------------ score.py


@pytest.mark.parametrize(
  ("value", "shown"),
  [
    (0.0, "0.00"),
    (1.0, "1.00"),
    (0.8333333, "0.83"),
    (0.999, "1.00"),
    (0.004, "0.00"),
    (1234.5678, "1234.57"),
    (3, "3.00"),
  ],
)
def test_format_score(value: float, shown: str) -> None:
  assert format_score(value) == shown


def test_metric_tiles_of_a_schema_1_record_is_empty() -> None:
  assert metric_tiles(schema1()) == ()
  assert primary_scores(schema1()) == ()
  raw = json.loads((SCHEMA1 / "run.json").read_text("utf-8"))
  assert metric_tiles(raw) == ()  # no ``metrics`` key at all
  assert primary_scores(raw) == ()


def test_metric_tiles_of_the_reference_run(real_record: RunRecord) -> None:
  summary = metrics_of(real_record)["summary"]
  tiles = metric_tiles(real_record)
  assert [label for label, _ in tiles] == [
    h.metric_label(s)
    for s in metrics_of(real_record)["scores"]
    if summary[s]["mean"] is not None
  ]
  for _, value in tiles:
    assert re.fullmatch(r"-?\d+\.\d\d \(n=\d+(, \d+ errores?)?\)", value), value


def test_metric_tile_value_has_mean_n_and_errors(record: RunRecord) -> None:
  metrics = metrics_of(record)
  metrics["scores"] = ["doc_hit", "faithfulness", "bleu", "bias"]
  metrics["summary"] = {
    "doc_hit": {"mean": 5 / 6, "n": 5, "not_applicable": 0, "skipped": 0, "errors": 0},
    "faithfulness": {
      "mean": 0.5,
      "n": 3,
      "not_applicable": 0,
      "skipped": 0,
      "errors": 1,
    },
    "bleu": {"mean": 0.0, "n": 1, "not_applicable": 0, "skipped": 0, "errors": 2},
    "bias": {"mean": 0.25, "n": 4, "not_applicable": 1, "skipped": 0, "errors": 0},
  }
  assert metric_tiles(record) == (
    ("Documento esperado recuperado", "0.83 (n=5)"),
    ("Fidelidad", "0.50 (n=3, 1 error)"),
    ("BLEU", "0.00 (n=1, 2 errores)"),
    ("Sesgo ↓", "0.25 (n=4)"),
  )


def test_a_score_without_a_mean_has_no_tile(record: RunRecord) -> None:
  metrics = metrics_of(record)
  metrics["scores"] = ["doc_hit", "doc_recall", "doc_mrr"]
  metrics["summary"] = {
    "doc_hit": {"mean": None, "n": 0, "not_applicable": 3, "skipped": 0, "errors": 0},
    "doc_recall": {"mean": 0.5, "n": 1, "not_applicable": 0, "skipped": 0, "errors": 0},
    # doc_mrr has no summary entry at all
  }
  assert [label for label, _ in metric_tiles(record)] == [
    "Cobertura de documentos esperados"
  ]


def test_metric_tiles_with_metrics_but_an_empty_summary(record: RunRecord) -> None:
  metrics_of(record)["summary"] = {}
  assert metric_tiles(record) == ()
  text = render_run(record)
  assert tile_pairs(text) == {}
  assert "menor es mejor" not in text


def test_primary_score_is_the_first_one_of_each_family(real_record: RunRecord) -> None:
  found = primary_scores(real_record)
  assert found == tuple(METRICS[name].scores[0] for name in DEFAULT_ON)
  assert found[:4] == ("doc_hit", "evidence_hit", "false_refusal", "seconds_retrieval")
  assert "correctness_precision" in found
  assert "correctness_f1" not in found


def test_primary_scores_follow_the_selection_with_optional_families(
  record: RunRecord,
) -> None:
  metrics_of(record)["selection"] = ["bleu", "doc_retrieval", "toxicity"]
  assert primary_scores(record) == ("bleu", "doc_hit", "toxicity")


def test_primary_score_of_an_unknown_family_falls_back_to_a_recorded_score(
  record: RunRecord,
) -> None:
  metrics_of(record)["selection"] = ["mystery", "ghost", "doc_retrieval"]
  results_of(record["questions"][1]).append(
    {
      "name": "mystery",
      "status": "ok",
      "scores": {"zz_score": 0.5, "zz_other": 1.0},
      "details": {},
      "calls": [],
      "error": None,
      "seconds": 0.0,
    }
  )
  # ``ghost`` recorded nothing anywhere: it has no column
  assert primary_scores(record) == ("zz_score", "doc_hit")


def test_score_cell_value_error_and_dashes(record: RunRecord) -> None:
  item = record["questions"][0]
  assert score_cell(item, "doc_hit") == "1.00"
  found = result(item, "doc_retrieval")
  found["scores"]["doc_hit"] = None
  assert score_cell(item, "doc_hit") == h.NO_DATA
  found["status"] = "not_applicable"
  assert score_cell(item, "doc_hit") == h.NO_DATA
  found["status"] = "skipped"
  assert score_cell(item, "doc_hit") == h.NO_DATA
  # a score nobody emits, or a question with no metrics at all
  assert score_cell(item, "nope") == h.NO_DATA
  item["metrics"] = None
  assert score_cell(item, "doc_hit") == h.NO_DATA
  item.pop("metrics")  # type: ignore[misc]
  assert score_cell(item, "doc_hit") == h.NO_DATA


def test_score_cell_error_is_marked_and_its_message_is_escaped(
  record: RunRecord,
) -> None:
  item = record["questions"][0]
  found = result(item, "doc_retrieval")
  found["status"] = "error"
  found["error"] = HOSTILE
  cell = score_cell(item, "doc_hit")
  assert 'class="status bad"' in cell and ">error<" in cell
  assert HOSTILE not in cell and "<script>" not in cell
  assert html_lib.escape(HOSTILE) in cell
  assert 'title=""' not in cell
  found["error"] = None
  assert 'title=""' in score_cell(item, "doc_hit")


def test_score_cell_error_wins_over_a_stale_value(record: RunRecord) -> None:
  item = record["questions"][0]
  found = result(item, "doc_retrieval")
  found["status"] = "error"
  found["scores"]["doc_hit"] = 0.7
  assert "0.70" not in score_cell(item, "doc_hit")


def test_score_cell_of_a_zero_is_a_value_not_a_dash(record: RunRecord) -> None:
  item = record["questions"][0]
  result(item, "doc_retrieval")["scores"]["doc_hit"] = 0.0
  assert score_cell(item, "doc_hit") == "0.00"


# ------------------------------------------------------------------ Resumen


def test_summary_shows_the_metric_tiles_below_the_question_ones(
  real_record: RunRecord,
) -> None:
  text = render_run(real_record)
  panel = panel_of(text, "resumen")
  assert tile_pairs(panel) == dict(metric_tiles(real_record))
  first_metric = panel.index('class="big small"')
  last_plain = panel.rindex('<span class="big">')
  assert last_plain < first_metric
  # both tile blocks sit above the results table
  assert first_metric < panel.index("<table>")
  # the plain tiles are exactly the question ones
  plain = re.findall(
    r'<div class="tile"><span class="k">([^<]*)</span><span class="big">', panel
  )
  assert plain == [label for label, _ in score_tiles(real_record["questions"])]


def test_summary_notes_that_lower_is_better_only_when_such_a_tile_exists(
  record: RunRecord,
) -> None:
  panel = panel_of(render_run(record), "resumen")
  assert "↓ menor es mejor" in panel
  metrics_of(record)["scores"] = [
    s for s in metrics_of(record)["scores"] if s not in h.LOWER_IS_BETTER
  ]
  panel = panel_of(render_run(record), "resumen")
  assert "menor es mejor" not in panel
  assert "Media sobre las preguntas" in panel


def test_lower_is_better_tiles_and_columns_carry_the_arrow(
  real_record: RunRecord,
) -> None:
  panel = panel_of(render_run(real_record), "resumen")
  tiles = tile_pairs(panel)
  for score in h.LOWER_IS_BETTER & set(metrics_of(real_record)["scores"]):
    label = h.metric_label(score)
    assert label.endswith("↓")
    assert label in tiles or metrics_of(real_record)["summary"][score]["mean"] is None
  assert "Fidelidad" in tiles and "Fidelidad ↓" not in tiles
  headers = [html_lib.unescape(x) for x in re.findall(r"<th[^>]*>(.*?)</th>", panel)]
  assert "Rechazos injustificados ↓" in headers
  assert "Documento esperado recuperado" in headers


def test_question_table_has_one_cell_per_primary_score_and_row(
  real_record: RunRecord,
) -> None:
  panel = panel_of(render_run(real_record), "resumen")
  scores = primary_scores(real_record)
  headers = re.findall(r"<th[^>]*>(.*?)</th>", panel)
  assert headers[-len(scores) :] == [h.esc(h.metric_label(s)) for s in scores]
  rows = re.findall(r"<tr>(?!<th).*?</tr>", panel, re.S)
  assert len(rows) == len(real_record["questions"])
  for row, item in zip(rows, real_record["questions"], strict=True):
    cells = re.findall(r'<td class="num-cell">(.*?)</td>', row, re.S)
    assert cells == [score_cell(item, s) for s in scores]


def test_metrics_off_leaves_no_trace_in_the_summary(record: RunRecord) -> None:
  record["metrics"] = None
  for item in record["questions"]:
    item["metrics"] = None
  text = render_run(record)
  assert 'class="big small"' not in text
  assert "Métricas" not in text
  assert 'class="num-cell"' not in panel_of(text, "resumen").split("<table>")[1]


def test_a_record_with_metrics_but_no_question_metrics_renders(
  record: RunRecord,
) -> None:
  for item in record["questions"]:
    item["metrics"] = None
  text = render_run(record)
  assert "Métricas</span>" not in text  # no section
  # the columns remain and every cell is a dash
  panel = panel_of(text, "resumen")
  for row in re.findall(r"<tr>(?!<th).*?</tr>", panel, re.S):
    cells = re.findall(r'<td class="num-cell">(.*?)</td>', row, re.S)
    assert cells and set(cells) == {h.NO_DATA}


def test_a_hand_built_record_without_the_metrics_keys_renders(
  record: RunRecord,
) -> None:
  record.pop("metrics")  # type: ignore[misc]
  for item in record["questions"]:
    item.pop("metrics")  # type: ignore[misc]
  text = render_run(record)
  assert "Métricas" not in text and 'class="big small"' not in text


def test_zero_questions_with_metrics_renders(record: RunRecord) -> None:
  record["questions"] = []
  text = render_run(record)
  assert "Sin preguntas." in text
  assert render_compare([("a/report.html", record)])


# ------------------------------------------------------------------ the question section


def test_every_question_tab_has_one_metrics_section_after_the_judge(
  real_record: RunRecord,
) -> None:
  text = render_run(real_record)
  for item in real_record["questions"]:
    panel = panel_of(text, f"q-{item['query_id']}")
    ids = re.findall(r'<details class="phase" id="q-[^"]*?-([a-z0-9]+)" open>', panel)
    assert ids[-1] == "metrics" and ids.count("metrics") == 1
    assert ids.index("judge") < ids.index("metrics")
    flow = re.search(r'<div class="flow">(.*?)</div>', panel, re.S)
    assert flow and flow.group(1).count('href="#') == len(ids)
    body = panel.split('-metrics" open>')[1]
    for result_ in results_of(item):
      assert f"<code>{result_['name']}</code>" in body


def test_the_section_summary_counts_families(record: RunRecord) -> None:
  results = results_of(record["questions"][0])
  n = len(results)
  assert (
    metrics_summary(results)
    == f"{sum(r['status'] == 'ok' for r in results)}/{n} calculadas"
  )
  results[0]["status"] = "error"
  results[1]["status"] = "error"
  assert metrics_summary(results).endswith(", 2 con error")
  assert metrics_summary([]) == "0/0 calculadas"
  assert metrics_body([]) == ""


def test_status_labels_and_tones(record: RunRecord) -> None:
  item = record["questions"][0]
  for status, label, tone in [
    ("ok", "calculada", "good"),
    ("not_applicable", "no aplicable", "info"),
    ("skipped", "omitida", "meh"),
    ("error", "error", "bad"),
  ]:
    results_of(item)[0]["status"] = status
    body = metrics_body(results_of(item))
    assert f'<span class="status {tone}">{label}</span>' in body.split("</h4>")[0]


def test_an_unknown_status_keeps_its_text_and_is_escaped(record: RunRecord) -> None:
  item = record["questions"][0]
  results_of(item)[0]["status"] = HOSTILE
  body = metrics_body(results_of(item))
  assert HOSTILE not in body
  assert f'<span class="status info">{html_lib.escape(HOSTILE)}</span>' in body


def test_a_failed_family_shows_its_error_and_the_other_families_survive(
  record: RunRecord,
) -> None:
  item = record["questions"][0]
  found = result(item, "faithfulness")
  found["status"] = "error"
  found["error"] = "MetricError: el juez devolvió basura"
  found["scores"] = {"faithfulness": None}
  body = metrics_body(results_of(item))
  assert '<div class="error">MetricError: el juez devolvió basura</div>' in body
  assert body.count("<h4><code>") == len(results_of(item))
  # the None score is a dash, not "None" and not an exception
  assert ">None<" not in body


def test_the_scores_grid_shows_labels_and_two_decimals(record: RunRecord) -> None:
  item = record["questions"][0]
  found = result(item, "refusal")
  found["scores"] = {"false_refusal": 0.0, "invented_answer": None}
  body = metrics_body([found])
  assert '<span class="k">Rechazos injustificados ↓</span><span>0.00</span>' in body
  assert (
    f'<span class="k">Respuestas inventadas ↓</span><span>{h.NO_DATA}</span>' in body
  )


def test_judge_calls_are_shown_raw_with_their_messages(record: RunRecord) -> None:
  item = record["questions"][0]
  with_calls = [r for r in results_of(item) if r["calls"]]
  assert with_calls, "the echo judge makes metric calls"
  body = metrics_body(with_calls)
  assert body.count("<details><summary>Llamadas al juez (") == len(with_calls)
  for found in with_calls:
    assert f"Llamadas al juez ({len(found['calls'])})" in body
    for call in found["calls"]:
      assert html_lib.escape(call["raw"]) in body or call["raw"] is None
      assert html_lib.escape(call["purpose"]) in body
  assert "<h4>Respuesta cruda</h4>" in body
  assert '<div class="msg system">' in body and '<div class="msg user">' in body


def test_a_call_without_a_reply_or_messages_says_so(record: RunRecord) -> None:
  item = record["questions"][0]
  found = next(r for r in results_of(item) if r["calls"])
  found["calls"] = [
    {
      "purpose": "answer_claims",
      "messages": [],
      "raw": None,
      "error": "ProviderError: caído",
      "seconds": 0.25,
    }
  ]
  body = metrics_body([found])
  assert "Sin respuesta." in body and "Sin mensajes." in body
  assert '<div class="error">ProviderError: caído</div>' in body
  assert "0.2 s" in body or "0.3 s" in body


def test_a_family_without_scores_details_or_calls_is_just_a_header(
  record: RunRecord,
) -> None:
  found = copy.deepcopy(result(record["questions"][0], "cost"))
  found.update(scores={}, details={}, calls=[], error=None)
  body = metrics_body([found])
  assert body.count("<h4>") == 1
  assert "<details>" not in body and 'class="verdict"' not in body


def test_hostile_text_in_every_place_of_the_section_is_escaped(
  record: RunRecord,
) -> None:
  item = record["questions"][0]
  found = result(item, "faithfulness")
  found["name"] = HOSTILE
  found["status"] = HOSTILE
  found["error"] = HOSTILE
  found["scores"] = {HOSTILE: 0.5, "faithfulness": 0.25}
  found["details"] = {
    HOSTILE: HOSTILE,
    "claims": [{"text": HOSTILE, "markers": [1], HOSTILE: HOSTILE}],
    "questions": [HOSTILE, HOSTILE],
    "verdicts": {HOSTILE: HOSTILE, "n": 3},
    "nested": {"a": [HOSTILE, {HOSTILE: [HOSTILE]}]},
    "mixed": [HOSTILE, 1, None, True, 0.5],
    "reason": HOSTILE,
  }
  found["calls"] = [
    {
      "purpose": HOSTILE,
      "messages": [{"role": HOSTILE, "content": HOSTILE}],
      "raw": HOSTILE,
      "error": HOSTILE,
      "seconds": 0.1,
    }
  ]
  text = render_run(record)
  assert_no_injection(text)
  panel = panel_of(text, "q-q01")
  assert html_lib.escape(HOSTILE) in panel
  # in the summary and the tile area as well when the hostile text is a score name
  metrics_of(record)["scores"].append(HOSTILE)
  metrics_of(record)["selection"].append(HOSTILE)
  metrics_of(record)["summary"][HOSTILE] = {
    "mean": 0.5,
    "n": 1,
    "not_applicable": 0,
    "skipped": 0,
    "errors": 0,
  }
  assert_no_injection(render_run(record))
  runs: Runs = [(HOSTILE, record), ("b/report.html", copy.deepcopy(record))]
  assert_no_injection(render_compare(runs))


def test_details_render_every_shape_without_raising(record: RunRecord) -> None:
  item = record["questions"][0]
  found = copy.deepcopy(result(item, "faithfulness"))
  found["details"] = {
    "none": None,
    "bool": True,
    "int": 3,
    "float": 0.123456,
    "str": "texto\ncon salto",
    "empty_list": [],
    "empty_dict": {},
    "scalars": ["a", 1, 2.5, None, False],
    "dicts_ragged": [{"a": 1}, {"b": 2, "c": None}],
    "dicts_nested": [{"a": {"b": [1, "ñ"]}}],
    "dict_of_scalars": {"x": 1, "y": "z"},
    "dict_with_list": {"x": [1, 2]},
    "list_of_lists": [[1, 2], [3]],
    "mixed_list": [{"a": 1}, "s"],
    "unicode": "ñandú — 日本語",
    "big": 10**30,
  }
  body = metrics_body([found])
  assert "None" not in body.replace("NoneType", "")
  assert "0.12" in body and "sí" in body
  assert "ninguno" in body
  parse(body)  # parseable


def test_a_very_large_details_value_renders(record: RunRecord) -> None:
  found = copy.deepcopy(result(record["questions"][0], "faithfulness"))
  found["details"] = {"claims": [{"text": "x" * 50, "i": i} for i in range(2000)]}
  assert metrics_body([found]).count("<tr>") >= 2000


def test_failed_question_keeps_its_metrics_section(record: RunRecord) -> None:
  item = record["questions"][0]
  item["error"] = "RuntimeError: se cayó"
  item["retrieval"] = item["enrich"] = item["generation"] = None
  item["rules"] = item["judge"] = None
  panel = panel_of(render_run(record), "q-q01")
  assert '-metrics" open>' in panel
  assert '<div class="error">RuntimeError: se cayó</div>' in panel


# ------------------------------------------------------------------ schema 1


def test_schema_1_report_is_byte_identical_to_the_golden() -> None:
  expected = (SCHEMA1 / "report.html").read_text(encoding="utf-8")
  assert render_run(schema1()) == expected
  raw = json.loads((SCHEMA1 / "run.json").read_text("utf-8"))
  assert render_run(raw) == expected


def test_schema_1_report_has_none_of_the_new_markup() -> None:
  text = render_run(schema1())
  for needle in ("big small", "Métricas", "menor es mejor"):
    assert needle not in text


# ------------------------------------------------------------------ determinism


def test_report_is_byte_deterministic_and_survives_a_json_round_trip(
  real_record: RunRecord,
) -> None:
  first = render_run(real_record)
  assert render_run(real_record) == first
  assert render_run(copy.deepcopy(real_record)) == first
  assert render_run(json.loads(json.dumps(real_record))) == first


def test_compare_is_byte_deterministic(real_record: RunRecord) -> None:
  runs: Runs = [
    ("a/report.html", real_record),
    ("old/report.html", schema1()),
    ("b/report.html", copy.deepcopy(real_record)),
  ]
  first = render_compare(runs)
  assert render_compare(runs) == first
  rebuilt: Runs = [(href, json.loads(json.dumps(r))) for href, r in runs]
  assert render_compare(rebuilt) == first


def test_rendering_does_not_mutate_the_record(real_record: RunRecord) -> None:
  before = copy.deepcopy(real_record)
  render_run(real_record)
  render_compare([("a", real_record), ("b", schema1())])
  assert real_record == before


def test_the_report_does_not_depend_on_the_clock(
  real_record: RunRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
  import time

  first = render_run(real_record)
  monkeypatch.setattr(time, "time", lambda: 4_000_000_000.0)
  assert render_run(real_record) == first


def test_the_report_works_offline(real_record: RunRecord) -> None:
  text = render_run(real_record)
  assert re.search(r"(?:src|href)=[\"']https?://", text) is None
  assert "<link" not in text


# ------------------------------------------------------------------ compare


def card_text(text: str, title: str) -> str:
  match = re.search(
    rf'<div class="card"><h3>{re.escape(title)}</h3>(.*?)</div>(?=<div class="card">|</section>)',
    text,
    re.S,
  )
  assert match, title
  return match.group(1)


def score_rows(text: str) -> dict[str, list[str]]:
  rows: dict[str, list[str]] = {}
  for row in re.findall(r"<tr>(?!<th).*?</tr>", card_text(text, "Puntuación"), re.S):
    cells = [
      html_lib.unescape(re.sub(r"<[^>]+>", "", c)).strip()
      for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
    ]
    rows[cells[0]] = cells[1:]
  return rows


def matrix(text: str) -> str:
  found = re.search(r'id="tab-matriz"[^>]*>(.*?)</section>', text, re.S)
  assert found
  return found.group(1)


def group_rows(text: str) -> list[str]:
  return [
    r
    for r in re.findall(r"<tr(?: [^>]*)?>(?!<th).*?</tr>", matrix(text), re.S)
    if "data-q=" not in r.split(">")[0]
  ]


def test_compare_of_schema_1_and_schema_2_shows_dashes_for_the_old_run(
  real_record: RunRecord,
) -> None:
  runs: Runs = [("new/report.html", real_record), ("old/report.html", schema1())]
  text = render_compare(runs)
  rows = score_rows(text)
  tiles = dict(metric_tiles(real_record))
  assert tiles
  for label, value in tiles.items():
    assert rows[label] == [value, "—"]
  # the old run's question tiles are still there
  assert rows["Correctas por reglas"][1] != "—"


def test_compare_order_does_not_matter_for_the_metric_rows(
  real_record: RunRecord,
) -> None:
  a = score_rows(render_compare([("n", real_record), ("o", schema1())]))
  b = score_rows(render_compare([("o", schema1()), ("n", real_record)]))
  assert list(a) == list(b)
  assert {k: v[::-1] for k, v in a.items()} == b


def test_compare_of_two_schema_1_runs_has_no_metric_rows() -> None:
  runs: Runs = [("a/report.html", schema1()), ("b/report.html", schema1())]
  text = render_compare(runs)
  assert list(score_rows(text)) == [label for label, _ in score_tiles([])]
  assert group_rows(text) == []


def test_compare_with_metrics_off_in_both_runs_has_no_metric_rows(
  record: RunRecord,
) -> None:
  record["metrics"] = None
  for item in record["questions"]:
    item["metrics"] = None
  text = render_compare([("a", record), ("b", copy.deepcopy(record))])
  assert group_rows(text) == []
  assert list(score_rows(text)) == [label for label, _ in score_tiles([])]


def test_matrix_has_one_group_per_score_with_a_mean_row_then_a_row_per_question(
  real_record: RunRecord,
) -> None:
  runs: Runs = [("new/report.html", real_record), ("old/report.html", schema1())]
  text = render_compare(runs)
  scores = metrics_of(real_record)["scores"]
  ids = [item["spec"]["id"] for item in real_record["questions"]]
  rows = group_rows(text)
  assert len(rows) == len(scores) * (1 + len(ids))
  for number, score in enumerate(scores):
    group = rows[number * (1 + len(ids)) : (number + 1) * (1 + len(ids))]
    assert html_lib.escape(h.metric_label(score)) in group[0]
    assert "media" in group[0]
    mean = metrics_of(real_record)["summary"][score]["mean"]
    expected_mean = h.NO_DATA if mean is None else format_score(mean)
    cells = re.findall(r'<td class="num-cell">(?:<b>)?(.*?)(?:</b>)?</td>', group[0])
    assert cells == [expected_mean, h.NO_DATA]
    for row, qid, item in zip(group[1:], ids, real_record["questions"], strict=True):
      assert f"<td>{html_lib.escape(qid)}</td>" in row
      cells = re.findall(
        r'<td class="num-cell">(.*?)</td>|<td>(<span class="muted">—</span>)</td>', row
      )
      assert [a or b for a, b in cells] == [score_cell(item, score), h.NO_DATA]


def test_every_matrix_row_has_the_same_number_of_columns(
  real_record: RunRecord,
) -> None:
  runs: Runs = [("a", real_record), ("b", schema1()), ("c", copy.deepcopy(real_record))]
  text = render_compare(runs)
  rows = group_rows(text)
  assert rows
  for row in rows:
    # the mean row joins id and question in one cell (colspan 2)
    expected = len(runs) + (1 if 'colspan="2"' in row else 2)
    assert len(re.findall(r"<td", row)) == expected, row


def test_a_question_missing_from_one_run_gets_a_dash_in_the_score_group(
  real_record: RunRecord,
) -> None:
  other = copy.deepcopy(real_record)
  other["questions"] = other["questions"][:1]
  text = render_compare([("a", real_record), ("b", other)])
  rows = group_rows(text)
  q11_rows = [r for r in rows if "<td>q11</td>" in r]
  assert q11_rows
  assert all(r.rstrip().endswith(f"<td>{h.NO_DATA}</td></tr>") for r in q11_rows)


def test_runs_with_different_selections_use_the_union_of_scores(
  record: RunRecord,
) -> None:
  other = copy.deepcopy(record)
  metrics_of(other)["scores"] = [*metrics_of(other)["scores"], "bleu"]
  metrics_of(other)["selection"].append("bleu")
  metrics_of(other)["summary"]["bleu"] = {
    "mean": 0.4,
    "n": 3,
    "not_applicable": 0,
    "skipped": 0,
    "errors": 0,
  }
  for item in other["questions"]:
    results_of(item).append(
      {
        "name": "bleu",
        "status": "ok",
        "scores": {"bleu": 0.4},
        "details": {},
        "calls": [],
        "error": None,
        "seconds": 0.0,
      }
    )
  text = render_compare([("a", record), ("b", other)])
  assert score_rows(text)["BLEU"] == ["—", "0.40 (n=3)"]
  rows = group_rows(text)
  [bleu_mean] = [r for r in rows if "<b>BLEU</b>" in r]
  assert re.findall(r'<td class="num-cell"><b>(.*?)</b></td>', bleu_mean) == [
    h.NO_DATA,
    "0.40",
  ]


def test_filter_leaves_score_groups_alone(real_record: RunRecord) -> None:
  text = render_compare([("a", real_record), ("b", schema1())])
  assert group_rows(text)
  assert all(
    "data-q" not in r and "class=" not in r.split(">")[0] for r in group_rows(text)
  )
  script = text[text.index("<script>") : text.rindex("</script>")]
  assert "tr[data-q]:not(.diff)" in script


def test_a_metric_difference_alone_does_not_highlight_the_row(
  real_record: RunRecord,
) -> None:
  other = copy.deepcopy(real_record)
  result(other["questions"][0], "doc_retrieval")["scores"]["doc_hit"] = 0.0
  text = render_compare([("a", real_record), ("b", other)])
  assert 'class="diff"' not in text


def test_lower_is_better_rows_carry_the_arrow_in_compare(
  real_record: RunRecord,
) -> None:
  text = render_compare([("a", real_record), ("b", copy.deepcopy(real_record))])
  rows = score_rows(text)
  assert "Rechazos injustificados ↓" in rows
  assert "Fidelidad" in rows and "Fidelidad ↓" not in rows
  assert "Rechazos injustificados ↓" in matrix(text).replace("&nbsp;", " ")


def test_compare_with_runs_without_questions_and_metrics(
  real_record: RunRecord,
) -> None:
  runs: Runs = [("a", copy.deepcopy(real_record)), ("b", copy.deepcopy(real_record))]
  for _, rec in runs:
    rec["questions"] = []
  text = render_compare(runs)
  assert "Sin preguntas." in text
  assert group_rows(text) == []  # no question, no group
  value = dict(metric_tiles(runs[0][1]))["Fidelidad"]
  assert score_rows(text)["Fidelidad"] == [value, value]


def test_a_hostile_question_id_is_escaped_in_the_score_groups(
  real_record: RunRecord,
) -> None:
  runs: Runs = [("a/report.html", copy.deepcopy(real_record)), ("b", schema1())]
  runs[0][1]["questions"][0]["spec"]["id"] = HOSTILE
  runs[0][1]["questions"][0]["spec"]["question"] = HOSTILE
  text = render_compare(runs)
  assert_no_injection(text)
  assert f"<td>{html_lib.escape(HOSTILE)}</td>" in "".join(group_rows(text))


def test_compare_page_is_still_self_contained(real_record: RunRecord) -> None:
  text = render_compare([("a", real_record), ("b", schema1())])
  assert re.search(r"(?:src|href)=[\"']https?://", text) is None
  assert "<link" not in text
  assert isinstance(parse(text), Page)


def test_compare_summary_missing_mean_entry_is_a_dash(real_record: RunRecord) -> None:
  other = copy.deepcopy(real_record)
  del metrics_of(other)["summary"]["faithfulness"]
  text = render_compare([("a", real_record), ("b", other)])
  assert score_rows(text)["Fidelidad"][1] == "—"
  [mean] = [r for r in group_rows(text) if "<b>Fidelidad</b>" in r]
  assert re.findall(r'<td class="num-cell"><b>(.*?)</b></td>', mean)[1] == h.NO_DATA

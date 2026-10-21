"""Tests for ``hermes_report.render``: the html helpers, the page and ``render_run``.

The reference record comes from a real in-memory run over ``tests/data`` (hash
embedder, echo LLM, echo judge). Tests that need another shape (hostile text,
more questions, failed questions) deep-copy it and edit the copy; the rendering
is a pure function of the record, so no service is involved.
"""

from __future__ import annotations

import copy
import html as html_lib
import json
import re
import shutil
import subprocess
import warnings
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from conftest import DATA

from hermes_report import runner
from hermes_report.config import Overrides, load_config
from hermes_report.evaluate import JUDGE_VERDICTS, RULE_VERDICTS
from hermes_report.record import RecordError, RunRecord
from hermes_report.render import html as h
from hermes_report.render import render_run, write_report
from hermes_report.render.run import render_run as render_run_direct

HOSTILE = '<script>alert("x")</script><img src=x onerror=alert(1)>&amp;"\''


# ------------------------------------------------------------------ helpers


class Page(HTMLParser):
  """Collects the structure of a page: elements, attributes and text per tag."""

  def __init__(self) -> None:
    super().__init__(convert_charrefs=True)
    self.elements: list[tuple[str, dict[str, str | None]]] = []
    self.text: list[tuple[str, str]] = []
    self._stack: list[str] = []

  def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    self.elements.append((tag, dict(attrs)))
    if tag not in {"meta", "br", "img", "link", "input"}:
      self._stack.append(tag)

  def handle_endtag(self, tag: str) -> None:
    if tag in self._stack:
      while self._stack and self._stack.pop() != tag:
        pass

  def handle_data(self, data: str) -> None:
    self.text.append((self._stack[-1] if self._stack else "", data))

  def tags(self, name: str) -> list[dict[str, str | None]]:
    return [attrs for tag, attrs in self.elements if tag == name]

  def body_text(self) -> str:
    """Visible text: everything but the inlined style and script."""
    return " ".join(d for tag, d in self.text if tag not in {"style", "script"})


def parse(text: str) -> Page:
  page = Page()
  page.feed(text)
  page.close()
  return page


def tab_keys(text: str) -> list[str]:
  return [a["data-key"] or "" for a in parse(text).tags("button")]


def panel_of(text: str, key: str) -> str:
  """The HTML of the panel ``key`` (panels are not nested)."""
  match = re.search(
    rf'<section class="panel" id="tab-{re.escape(key)}"[^>]*>(.*?)</section>',
    text,
    re.S,
  )
  assert match, f"no panel {key}"
  return match.group(1)


def tiles(text: str) -> dict[str, str]:
  found = re.findall(
    r'<div class="tile"><span class="k">(.*?)</span><span class="big">(.*?)</span>',
    text,
  )
  return {html_lib.unescape(k): html_lib.unescape(v) for k, v in found}


def rows_of(panel: str) -> list[str]:
  return re.findall(r"<tr>(?!<th).*?</tr>", panel, re.S)


@pytest.fixture(scope="module")
def real_record(tmp_path_factory: pytest.TempPathFactory) -> RunRecord:
  """A whole offline run over the sample project."""
  project = tmp_path_factory.mktemp("proj") / "proj"
  shutil.copytree(DATA, project)
  config = load_config(project / "run.toml", Overrides(out=project / "out"))
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    return runner.run(config, progress=lambda line: None).record


@pytest.fixture
def record(real_record: RunRecord) -> RunRecord:
  """A private copy of the reference record that a test may edit."""
  return copy.deepcopy(real_record)


@pytest.fixture
def report(real_record: RunRecord) -> str:
  return render_run(real_record)


def with_questions(record: RunRecord, count: int) -> RunRecord:
  """``record`` with ``count`` questions (copies of its first, ids q001...)."""
  base = record["questions"][0]
  questions = []
  for number in range(1, count + 1):
    item = copy.deepcopy(base)
    item["spec"]["id"] = f"q{number:03d}"
    item["query_id"] = f"q{number:03d}"
    questions.append(item)
  record["questions"] = questions
  return record


def with_verdicts(
  record: RunRecord, pairs: list[tuple[str | None, str | None]]
) -> RunRecord:
  """One question per ``(rule verdict, judge verdict)``; ``None`` drops the record."""
  base = record["questions"][0]
  questions = []
  for number, (rule, judged) in enumerate(pairs, 1):
    item = copy.deepcopy(base)
    item["spec"]["id"] = f"v{number}"
    item["query_id"] = f"v{number}"
    if rule is None:
      item["rules"] = None
    else:
      assert item["rules"] is not None
      item["rules"]["verdict"] = rule
    if judged is None:
      item["judge"] = None
    else:
      assert item["judge"] is not None
      item["judge"]["verdict"] = judged
    questions.append(item)
  record["questions"] = questions
  return record


# ------------------------------------------------------------------ the reference record


def test_reference_record_has_what_the_tests_rely_on(real_record: RunRecord) -> None:
  assert [q["query_id"] for q in real_record["questions"]] == ["q01", "q02", "q11"]
  assert all(q["judge"] is not None for q in real_record["questions"])
  assert real_record["collection"]["url"] == ":memory:"


# ------------------------------------------------------------------ the page: tabs


def test_render_run_is_exported_from_the_package() -> None:
  assert render_run is render_run_direct


def test_tabs_in_order(report: str) -> None:
  assert tab_keys(report) == [
    "resumen",
    "corpus",
    "chunks",
    "q-q01",
    "q-q02",
    "q-q11",
  ]


def test_every_tab_has_exactly_one_panel_with_the_same_key(report: str) -> None:
  panels = [a for a in parse(report).tags("section") if a.get("class") == "panel"]
  assert [p["data-key"] for p in panels] == tab_keys(report)
  assert [p["id"] for p in panels] == [f"tab-{k}" for k in tab_keys(report)]


def test_tab_labels(report: str) -> None:
  labels = re.findall(r'<button class="tab"[^>]*>(.*?)</button>', report)
  assert labels[:3] == ["Resumen", "Corpus e ingesta", "Chunks en Qdrant"]
  assert labels[3].startswith("¿Cuántos días tardó")


def test_tab_text_is_clipped_in_the_normal_case(record: RunRecord) -> None:
  record["questions"][0]["spec"]["question"] = "x" * 100
  labels = re.findall(r'<button class="tab"[^>]*>(.*?)</button>', render_run(record))
  assert labels[3] == "x" * 39 + "…"
  assert len(labels[3]) == 40


def test_tab_title_is_the_whole_question_with_its_id(record: RunRecord) -> None:
  record["questions"][0]["spec"]["question"] = "y" * 100
  text = render_run(record)
  assert f'title="q01 · {"y" * 100}"' in text


@pytest.mark.parametrize(("count", "compact"), [(6, False), (7, True), (1, False)])
def test_tabs_are_compact_above_six_questions(
  record: RunRecord, count: int, compact: bool
) -> None:
  text = render_run(with_questions(record, count))
  labels = re.findall(r'<button class="tab"[^>]*>(.*?)</button>', text)[3:]
  assert len(labels) == count
  if compact:
    assert labels == [f"q{n:03d}" for n in range(1, count + 1)]
  else:
    assert all(label.startswith("¿Cuántos") for label in labels)


def test_chunks_panel_is_only_the_empty_explorer(report: str) -> None:
  assert panel_of(report, "chunks") == '<div id="explorer"></div>'
  assert report.count('id="explorer"') == 1


def test_exactly_one_active_script_and_one_style(report: str) -> None:
  page = parse(report)
  assert len(page.tags("script")) == 1
  assert len(page.tags("style")) == 1
  assert page.tags("script")[0] == {}


def test_page_is_a_utf8_spanish_document(report: str) -> None:
  assert report.startswith("<!doctype html>")
  assert '<html lang="es">' in report
  assert '<meta charset="utf-8">' in report
  assert report.endswith("</html>\n")


# ------------------------------------------------------------------ header


def test_header_title_and_meta(report: str, real_record: RunRecord) -> None:
  assert "<title>Hermes Report · Don Quijote</title>" in report
  meta = re.search(r'<div class="meta">(.*?)</div>', report, re.S)
  assert meta
  text = html_lib.unescape(re.sub(r"<[^>]+>", "", meta.group(1)))
  assert "Generación: echo" in text
  assert "Juez: echo" in text
  assert f"Qdrant: {real_record['collection']['url']} / " in text
  assert real_record["collection"]["name"] in text
  assert real_record["started_at"] in text
  assert f"{real_record['seconds']:.1f} s" in text


def test_meta_says_sin_juez_when_the_judge_is_disabled(record: RunRecord) -> None:
  record["config"]["judge"]["enabled"] = False
  for item in record["questions"]:
    item["judge"] = None
  text = render_run(record)
  assert "Juez: <b>sin juez</b>" in text
  assert tiles(text)["Correctas por juez"] == "—"


def test_meta_survives_a_config_without_judge_or_generation(record: RunRecord) -> None:
  record["config"] = {}
  text = render_run(record)
  assert "Generación: <b>?</b>" in text
  assert "Juez: <b>sin juez</b>" in text


# ------------------------------------------------------------------ Resumen


def test_all_seven_tiles(report: str) -> None:
  assert list(tiles(panel_of(report, "resumen"))) == [
    "Correctas por reglas",
    "Correctas por juez",
    "Fallos de recuperación",
    "Fallos de presupuesto",
    "Fallos de generación",
    "Reglas y juez discrepan",
    "Tiempo medio por pregunta",
  ]


def test_tile_values_of_the_reference_run(report: str, real_record: RunRecord) -> None:
  values = tiles(report)
  rules = [q["rules"]["verdict"] for q in real_record["questions"] if q["rules"]]
  rated = [v for v in rules if h.RULE_TONES[v] != "info"]
  good = [v for v in rated if h.RULE_TONES[v] == "good"]
  assert values["Correctas por reglas"] == f"{len(good)}/{len(rated)}"
  # the echo judge always says "partial": none is right
  assert values["Correctas por juez"] == "0/3"
  mean = sum(q["seconds"] for q in real_record["questions"]) / 3
  assert values["Tiempo medio por pregunta"] == f"{mean:.1f} s"


def test_summary_table_one_row_per_question_linking_to_its_tab(report: str) -> None:
  panel = panel_of(report, "resumen")
  rows = rows_of(panel)
  assert len(rows) == 3
  gotos = re.findall(r'<a href="#(q-\w+)" data-goto="(q-\w+)">', panel)
  assert gotos == [("q-q01", "q-q01"), ("q-q02", "q-q02"), ("q-q11", "q-q11")]
  assert "¿Cuántos días tardó" in rows[0]
  headers = re.findall(r"<th[^>]*>(.*?)</th>", panel)
  assert headers == [
    "id",
    "pregunta",
    "reglas",
    "juez",
    "discrepan",
    "1.ª evidencia",
    "tiempo",
  ]


def test_summary_row_shows_pills_rank_and_seconds(
  report: str, real_record: RunRecord
) -> None:
  rows = rows_of(panel_of(report, "resumen"))
  for row, item in zip(rows, real_record["questions"], strict=True):
    assert item["rules"] and item["judge"]
    assert h.status_pill(item["rules"]["verdict"]) in row
    assert h.status_pill(item["judge"]["verdict"], "judge") in row
    rank = item["rules"]["retrieved_rank"]
    assert (f"#{rank}" in row) if rank else ('<span class="muted">—</span>' in row)
    assert f"{item['seconds']:.1f} s" in row


def test_summary_has_the_resolved_run_file_and_environment(
  report: str, real_record: RunRecord
) -> None:
  panel = panel_of(report, "resumen")
  block = re.search(r'<pre class="text json">(.*?)</pre>', panel, re.S)
  assert block
  assert json.loads(html_lib.unescape(block.group(1))) == real_record["config"]
  env = real_record["environment"]
  assert env["python"] in panel
  for package, version in env["packages"].items():
    assert package in panel
    assert version in panel
  assert html_lib.escape(env["ollama_host"]) in panel
  assert html_lib.escape(env["qdrant_url"]) in panel


def test_summary_with_no_questions(record: RunRecord) -> None:
  record["questions"] = []
  text = render_run(record)
  assert tab_keys(text) == ["resumen", "corpus", "chunks"]
  values = tiles(text)
  assert values["Tiempo medio por pregunta"] == "0.0 s"
  assert values["Correctas por reglas"] == "—"
  assert values["Correctas por juez"] == "—"
  assert "Sin preguntas." in text


# ------------------------------------------------------------------ tiles: counting


def test_every_rule_verdict_has_label_tone_and_every_judge_verdict_a_label() -> None:
  assert set(h.RULE_LABELS) == set(RULE_VERDICTS)
  assert set(h.RULE_TONES) == set(RULE_VERDICTS)
  assert set(h.JUDGE_LABELS) == set(JUDGE_VERDICTS)
  assert set(h.RULE_TONES.values()) <= {"good", "meh", "bad", "info"}
  assert h.RULE_LABELS["retrieval_failure"] == "fallo de recuperación"
  assert all(label and label == label.strip() for label in h.RULE_LABELS.values())


@pytest.mark.parametrize("verdict", RULE_VERDICTS)
def test_every_rule_verdict_renders_its_own_pill(
  record: RunRecord, verdict: str
) -> None:
  text = render_run(with_verdicts(record, [(verdict, "correct")]))
  assert h.status_pill(verdict) in panel_of(text, "resumen")
  assert h.status_pill(verdict) in panel_of(text, "q-v1")
  assert f'<span class="status {h.RULE_TONES[verdict]}">' in text


def test_failure_tiles_count_their_own_codes(record: RunRecord) -> None:
  pairs: list[tuple[str | None, str | None]] = [
    ("retrieval_failure", None),
    ("retrieval_failure", None),
    ("budget_failure", None),
    ("generation_refused", None),
    ("generation_wrong", None),
    ("generation_wrong", None),
    ("correct", None),
  ]
  values = tiles(render_run(with_verdicts(record, pairs)))
  assert values["Fallos de recuperación"] == "2"
  assert values["Fallos de presupuesto"] == "1"
  assert values["Fallos de generación"] == "3"
  assert values["Correctas por reglas"] == "1/7"


def test_unrated_is_left_out_of_the_denominator_of_the_rules_tile(
  record: RunRecord,
) -> None:
  pairs: list[tuple[str | None, str | None]] = [
    ("correct", None),
    ("unrated", None),
    ("unrated", None),
  ]
  assert (
    tiles(render_run(with_verdicts(record, pairs)))["Correctas por reglas"] == "1/1"
  )


def test_only_unrated_questions_gives_a_dash_not_a_division_by_zero(
  record: RunRecord,
) -> None:
  values = tiles(render_run(with_verdicts(record, [("unrated", None)])))
  assert values["Correctas por reglas"] == "—"


def test_judge_tile_counts_correct_over_judged_questions_only(
  record: RunRecord,
) -> None:
  pairs: list[tuple[str | None, str | None]] = [
    ("correct", "correct"),
    ("correct", "partial"),
    ("correct", "incorrect"),
    ("correct", None),
  ]
  assert tiles(render_run(with_verdicts(record, pairs)))["Correctas por juez"] == "1/3"


def test_an_errored_judge_counts_as_not_correct_but_stays_in_the_denominator(
  record: RunRecord,
) -> None:
  record = with_verdicts(record, [("correct", "correct"), ("correct", "correct")])
  judge = record["questions"][1]["judge"]
  assert judge is not None
  judge["verdict"] = None
  judge["error"] = "boom"
  text = render_run(record)
  assert tiles(text)["Correctas por juez"] == "1/2"
  assert "boom" in panel_of(text, "q-v2")


@pytest.mark.parametrize(
  ("rule", "judged", "disagree"),
  [
    ("correct", "incorrect", True),
    ("correct_refusal", "incorrect", True),
    ("correct", "correct", False),
    ("correct", "partial", False),
    ("correct_without_evidence", "incorrect", False),
    ("correct_without_evidence", "correct", False),
    ("retrieval_failure", "correct", True),
    ("budget_failure", "correct", True),
    ("generation_refused", "correct", True),
    ("generation_wrong", "correct", True),
    ("invented_answer", "correct", True),
    ("error", "correct", True),
    ("retrieval_failure", "incorrect", False),
    ("retrieval_failure", "partial", False),
    ("unrated", "correct", False),
    ("unrated", "incorrect", False),
    ("correct", None, False),
    (None, "correct", False),
    (None, "incorrect", False),
  ],
)
def test_disagreement_rule(
  record: RunRecord, rule: str | None, judged: str | None, disagree: bool
) -> None:
  text = render_run(with_verdicts(record, [(rule, judged)]))
  assert tiles(text)["Reglas y juez discrepan"] == ("1" if disagree else "0")
  assert ("≠" in panel_of(text, "resumen")) is disagree
  assert ("reglas y juez discrepan" in panel_of(text, "q-v1")) is disagree


def test_unknown_rule_code_is_shown_verbatim_and_does_not_crash(
  record: RunRecord,
) -> None:
  text = render_run(with_verdicts(record, [("brand_new_code", "correct")]))
  assert '<span class="status info">brand_new_code</span>' in text
  assert tiles(text)["Reglas y juez discrepan"] == "0"


def test_mean_seconds_tile(record: RunRecord) -> None:
  for item, value in zip(record["questions"], (1.0, 2.0, 6.0), strict=True):
    item["seconds"] = value
  assert tiles(render_run(record))["Tiempo medio por pregunta"] == "3.0 s"


# ------------------------------------------------------------------ Corpus e ingesta


def test_corpus_documents_table(report: str, real_record: RunRecord) -> None:
  rows = rows_of(panel_of(report, "corpus"))
  documents = real_record["corpus"]["documents"]
  ingested = {d["source"]: d for d in real_record["ingest"]["documents"]}
  table = [
    r for r in rows if "<code>" in r and any(d["source"] in r for d in documents)
  ]
  assert len(table) >= len(documents)
  for document in documents:
    row = next(r for r in rows if f"<code>{document['source']}</code>" in r)
    assert f'<td class="num-cell">{document["chars"]}</td>' in row
    assert f'<td class="num-cell">{document["chunks"]}</td>' in row
    done = ingested[document["source"]]
    assert f"{done['seconds']:.1f} s" in row
    assert ("sí" if done["cached"] else "no") in row


def test_corpus_document_missing_from_ingest_shows_dashes(record: RunRecord) -> None:
  record["ingest"]["documents"] = []
  panel = panel_of(render_run(record), "corpus")
  assert panel.count('<span class="muted">—</span>') >= len(
    record["corpus"]["documents"]
  )


def test_corpus_chunker_arguments(report: str, real_record: RunRecord) -> None:
  panel = panel_of(report, "corpus")
  assert "Argumentos del chunker" in panel
  for key, value in real_record["ingest"]["chunker"].items():
    assert f'<span class="k">{html_lib.escape(key)}</span>' in panel
    if isinstance(value, bool):
      assert ("sí" if value else "no") in panel
    elif isinstance(value, int):
      assert f"<span>{value}</span>" in panel


def test_corpus_embedding(report: str, real_record: RunRecord) -> None:
  panel = panel_of(report, "corpus")
  embedding = real_record["ingest"]["embedding"]
  assert "Embeddings" in panel
  assert str(embedding["provider"]) in panel
  assert str(embedding["model"]) in panel
  assert f"<span>{embedding['dimension']}</span>" in panel


def test_corpus_collection_card(report: str, real_record: RunRecord) -> None:
  panel = panel_of(report, "corpus")
  info = real_record["collection"]
  card = panel[panel.index("Colección de Qdrant") :]
  assert f"<code>{info['name']}</code>" in card
  assert f"<span>{info['points']}</span>" in card
  assert f"<span>{info['vector_size']}</span>" in card
  assert f"<span>{info['distance']}</span>" in card
  for index in info["indexes"]:
    assert index in card
  assert ("recargada" if info["reloaded"] else "reutilizada") in card


@pytest.mark.parametrize(
  ("reloaded", "expected"),
  [(True, "recargada en esta ejecución"), (False, "reutilizada tal cual")],
)
def test_collection_state_reloaded_or_reused(
  record: RunRecord, reloaded: bool, expected: str
) -> None:
  record["collection"]["reloaded"] = reloaded
  assert expected in panel_of(render_run(record), "corpus")


def test_collection_without_indexes_shows_a_dash(record: RunRecord) -> None:
  record["collection"]["indexes"] = []
  assert "<span>—</span>" in panel_of(render_run(record), "corpus")


def test_chunk_size_statistics_are_exact(record: RunRecord) -> None:
  sizes = {"a.md": [10, 20, 30, 41], "b.md": [7]}
  points = []
  for source, values in sizes.items():
    for number, size in enumerate(values):
      points.append(
        {
          "id": f"{source}-{number}",
          "payload": {"source": source, "chunk_size": size},
          "vector": None,
        }
      )
  record["points"] = points  # type: ignore[typeddict-item]
  panel = panel_of(render_run(record), "corpus")
  card = panel[panel.index("Tamaño de los chunks") :]
  card = card[: card.index("Colección de Qdrant")]
  rows = rows_of(card)
  assert len(rows) == 3  # a.md, b.md and the total
  a_row, b_row, all_row = rows
  assert "<code>a.md</code>" in a_row
  assert re.findall(r'<td class="num-cell">([^<]*)</td>', a_row) == [
    "4",
    "10",
    "25",
    "41",
  ]
  assert re.findall(r'<td class="num-cell">([^<]*)</td>', b_row) == [
    "1",
    "7",
    "7",
    "7",
  ]
  assert re.findall(r'<td class="num-cell">([^<]*)</td>', all_row) == [
    "5",
    "7",
    "20",
    "41",
  ]
  # one bar per document and one for the total; the largest chunk fills the scale
  assert card.count('class="sbar"') == 3
  assert 'style="width:100.0%"' in a_row


def test_chunk_size_bars_stay_within_the_track(report: str) -> None:
  widths = [
    float(w)
    for w in re.findall(r'class="sbar-(?:max|med)" style="width:([\d.]+)%', report)
  ]
  assert widths
  assert all(0 <= w <= 100 for w in widths)


def test_chunk_size_ignores_missing_or_non_integer_sizes(record: RunRecord) -> None:
  record["points"] = [
    {"id": "1", "payload": {"source": "a.md", "chunk_size": 5}, "vector": None},
    {"id": "2", "payload": {"source": "a.md"}, "vector": None},
    {"id": "3", "payload": {"source": "a.md", "chunk_size": "9"}, "vector": None},
    {"id": "4", "payload": {"source": "a.md", "chunk_size": True}, "vector": None},
    {"id": "5", "payload": {"source": "a.md", "chunk_size": None}, "vector": None},
  ]
  panel = panel_of(render_run(record), "corpus")
  assert "Tamaño de los chunks" in panel
  assert '<td class="num-cell">5</td>' in panel


def test_chunk_size_without_any_data(record: RunRecord) -> None:
  record["points"] = []
  assert "Sin datos de tamaño." in panel_of(render_run(record), "corpus")


def test_chunk_size_all_zero_does_not_divide_by_zero(record: RunRecord) -> None:
  record["points"] = [
    {"id": "1", "payload": {"source": "a.md", "chunk_size": 0}, "vector": None}
  ]
  assert 'style="width:0.0%"' in render_run(record)


def test_median_of_an_even_count_is_shown_with_one_decimal(record: RunRecord) -> None:
  record["points"] = [
    {"id": str(n), "payload": {"source": "a.md", "chunk_size": s}, "vector": None}
    for n, s in enumerate((1, 2))
  ]
  assert '<td class="num-cell">1.5</td>' in render_run(record)


# ------------------------------------------------------------------ question tabs


def test_question_header(report: str, real_record: RunRecord) -> None:
  panel = panel_of(report, "q-q01")
  head = re.search(r'<div class="sc-head">(.*?)</div>', panel, re.S)
  assert head
  item = real_record["questions"][0]
  assert f"<h2>{html_lib.escape(item['spec']['question'])}</h2>" in head.group(1)
  assert "q01" in head.group(1)
  assert f"{item['seconds']:.1f} s" in head.group(1)
  assert '<div class="error">' not in panel


def test_question_header_shows_the_generation_status_pill(
  report: str, real_record: RunRecord
) -> None:
  for item in real_record["questions"]:
    assert item["generation"] and item["generation"]["result"]
    status = item["generation"]["result"]["status"]
    assert h.status_pill(status, "generation") in panel_of(
      report, f"q-{item['query_id']}"
    )


def test_errored_question_shows_its_error_and_survives_missing_phases(
  record: RunRecord,
) -> None:
  item = record["questions"][0]
  item["error"] = "RuntimeError: model down"
  item["retrieval"] = None
  item["enrich"] = None
  item["generation"] = None
  item["rules"] = None
  item["judge"] = None
  text = render_run(record)
  panel = panel_of(text, "q-q01")
  assert '<div class="error">RuntimeError: model down</div>' in panel
  assert h.status_pill("error", "generation") in panel
  assert "sin juez para esta pregunta" in panel
  assert "Respuesta" not in panel
  # and the summary row falls back to dashes
  assert (
    rows_of(panel_of(text, "resumen"))[0].count('<span class="muted">—</span>') >= 3
  )


def test_evaluation_card_has_rules_answer_and_expected(
  report: str, real_record: RunRecord
) -> None:
  panel = panel_of(report, "q-q01")
  item = real_record["questions"][0]
  assert item["generation"] and item["generation"]["result"]
  assert "Evaluación" in panel
  assert '<div class="answer">' in panel
  answer = str(item["generation"]["result"]["answer"])
  assert html_lib.escape(answer.split("[")[0].strip()[:30]) in panel
  assert html_lib.escape(item["spec"]["expected"]) in panel
  for label in (
    "Veredicto",
    "Respuesta",
    "Esperada",
    "Palabras clave",
    "Evidencia",
    "Documentos esperados",
    "¿Recuperada?",
    "¿En el contexto final?",
    "¿En una pieza citada?",
    "¿Respuesta correcta?",
    "Recuperación",
  ):
    assert f'<span class="k">{label}</span>' in panel, label


def test_refusal_question_says_so(report: str) -> None:
  assert "Se espera rechazo" in panel_of(report, "q-q11")
  assert "Se espera rechazo" not in panel_of(report, "q-q01")


def test_evaluation_card_judge_section(report: str, real_record: RunRecord) -> None:
  panel = panel_of(report, "q-q01")
  judge = real_record["questions"][0]["judge"]
  assert judge
  assert h.status_pill(judge["verdict"], "judge") in panel
  assert (
    f'<span class="k">Motivo del juez</span><span>{html_lib.escape(judge["reason"] or "—")}</span>'
    in panel
  )
  assert f'<span class="k">Modelo del juez</span><span>{judge["model"]}</span>' in panel
  assert '<details class="raw-reply">' in panel
  assert "Respuesta cruda del juez" in panel
  raw = re.search(r'<pre class="text raw">(.*?)</pre>', panel, re.S)
  assert raw
  assert html_lib.unescape(raw.group(1)) == judge["raw"]


def test_judge_without_raw_reply_has_no_details(record: RunRecord) -> None:
  judge = record["questions"][0]["judge"]
  assert judge
  judge["raw"] = None
  assert "raw-reply" not in panel_of(render_run(record), "q-q01")


def test_judge_with_empty_raw_reply_still_shows_it(record: RunRecord) -> None:
  judge = record["questions"][0]["judge"]
  assert judge
  judge["raw"] = ""
  assert "raw-reply" in panel_of(render_run(record), "q-q01")


def test_judge_error_is_shown_in_an_error_box(record: RunRecord) -> None:
  judge = record["questions"][0]["judge"]
  assert judge
  judge["verdict"] = None
  judge["reason"] = None
  judge["error"] = "respuesta del juez no válida"
  panel = panel_of(render_run(record), "q-q01")
  assert '<div class="error">respuesta del juez no válida</div>' in panel


def test_answer_with_a_citation_marker_is_chipped(record: RunRecord) -> None:
  result = record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result
  result["answer"] = "Cuatro días **Rocinante** [1] y [12]."
  panel = panel_of(render_run(record), "q-q01")
  assert "<strong>Rocinante</strong>" in panel
  assert '<span class="cite">1</span>' in panel
  assert '<span class="cite">12</span>' in panel


def test_question_without_text_or_id_still_renders(record: RunRecord) -> None:
  record["questions"][0]["spec"] = {}
  text = render_run(record)
  assert "q-q01" in tab_keys(text)  # the id falls back to q<number>


def test_question_ids_fall_back_to_their_position(record: RunRecord) -> None:
  for item in record["questions"]:
    del item["spec"]["id"]
  assert tab_keys(render_run(record))[3:] == ["q-q01", "q-q02", "q-q03"]


# ------------------------------------------------------------------ escaping hostile text


def hostile_record(record: RunRecord) -> RunRecord:
  """Hostile text in every field of the record that the report prints."""
  record["title"] = HOSTILE
  record["description"] = HOSTILE
  record["started_at"] = HOSTILE
  record["config"]["generation"]["model"] = HOSTILE
  record["config"]["judge"]["model"] = HOSTILE
  record["config"]["extra"] = HOSTILE
  record["environment"]["python"] = HOSTILE
  record["environment"]["ollama_host"] = HOSTILE
  record["environment"]["qdrant_url"] = HOSTILE
  record["environment"]["packages"][HOSTILE] = HOSTILE
  record["corpus"]["dir"] = HOSTILE
  record["corpus"]["glob"] = HOSTILE
  record["corpus"]["documents"][0]["source"] = HOSTILE
  record["ingest"]["documents"][0]["source"] = HOSTILE
  record["ingest"]["chunker"][HOSTILE] = HOSTILE
  record["ingest"]["embedding"]["provider"] = HOSTILE
  record["ingest"]["embedding"]["model"] = HOSTILE
  record["ingest"]["ingester_version"] = HOSTILE
  record["collection"]["name"] = HOSTILE
  record["collection"]["url"] = HOSTILE
  record["collection"]["distance"] = HOSTILE
  record["collection"]["indexes"] = [HOSTILE]
  record["points"][0]["payload"]["source"] = HOSTILE
  record["points"][0]["payload"]["text"] = HOSTILE
  for item in record["questions"]:
    item["spec"]["id"] = HOSTILE.replace("/", "") + item["query_id"]
    item["spec"]["question"] = HOSTILE
    item["spec"]["expected"] = HOSTILE
    item["spec"]["keywords"] = [HOSTILE]
    item["spec"]["evidence"] = [HOSTILE]
    item["spec"]["documents"] = [HOSTILE]
    item["error"] = HOSTILE
    assert item["generation"] and item["generation"]["result"]
    item["generation"]["result"]["answer"] = HOSTILE + " **" + HOSTILE + "** [1]"
    assert item["judge"]
    item["judge"]["reason"] = HOSTILE
    item["judge"]["model"] = HOSTILE
    item["judge"]["raw"] = HOSTILE + "</pre></details>"
    item["judge"]["error"] = HOSTILE
  return record


def test_hostile_text_is_escaped_everywhere(record: RunRecord) -> None:
  text = render_run(hostile_record(record))
  page = parse(text)
  # No element or attribute that the hostile text could have introduced.
  assert page.tags("img") == []
  assert all(not any(name.startswith("on") for name in a) for _, a in page.elements)
  assert len(page.tags("script")) == 1
  assert "<script>alert" not in text
  assert "onerror=alert(1)>" not in text.replace("onerror=alert(1)&gt;", "")
  # The hostile text is present, as text, many times.
  assert html_lib.escape(HOSTILE) in text
  assert text.count(html_lib.escape(HOSTILE)) > 20
  # The document is still well formed: same tabs, panels and explorer.
  assert len(tab_keys(text)) == 6
  assert text.count('id="explorer"') == 1


def test_hostile_text_reaches_the_user_unchanged(record: RunRecord) -> None:
  page = parse(render_run(hostile_record(record)))
  assert HOSTILE in page.body_text()


def test_hostile_id_cannot_break_out_of_attributes(record: RunRecord) -> None:
  record["questions"][0]["spec"]["id"] = '"><img src=x onerror=alert(1)>'
  text = render_run(record)
  page = parse(text)
  assert page.tags("img") == []
  for attrs in page.tags("button"):
    assert set(attrs) == {"class", "data-key", "title"}
  anchors = [a for a in page.tags("a") if "data-goto" in a]
  assert anchors[0]["data-goto"] == 'q-"><img src=x onerror=alert(1)>'
  assert '"' not in re.findall(r'data-goto="([^"]*)"', text)[0]


def test_script_closing_sequence_in_data_cannot_end_the_inline_script(
  record: RunRecord,
) -> None:
  record["config"]["extra"] = "</script><script>alert(1)</script>"
  record["questions"][0]["judge"]["raw"] = "</script>"  # type: ignore[index]
  text = render_run(record)
  assert text.count("</script>") == 1
  assert text.count("<script>") == 1


def test_hostile_pairs_in_the_answer_do_not_unbalance_the_markup(
  record: RunRecord,
) -> None:
  result = record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result
  result["answer"] = "**<b>** [1] ** sin cerrar [x] [999999999999999999999]"
  panel = panel_of(render_run(record), "q-q01")
  assert "<b>" not in panel.split('<div class="answer">')[1].split("</div>")[0]


def test_bold_markup_with_script_inside_is_escaped(record: RunRecord) -> None:
  result = record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result
  result["answer"] = "**<script>alert(1)</script>**"
  panel = panel_of(render_run(record), "q-q01")
  assert "<strong>&lt;script&gt;alert(1)&lt;/script&gt;</strong>" in panel


# ------------------------------------------------------------------ determinism, no network


def test_same_record_gives_byte_identical_html(
  real_record: RunRecord, record: RunRecord
) -> None:
  first = render_run(real_record)
  assert render_run(real_record) == first
  assert render_run(record) == first  # also for an equal copy
  assert render_run(real_record).encode() == first.encode()


def test_render_does_not_mutate_the_record(real_record: RunRecord) -> None:
  before = json.dumps(real_record, sort_keys=True)
  render_run(real_record)
  assert json.dumps(real_record, sort_keys=True) == before


def test_render_after_a_json_round_trip_is_identical(
  real_record: RunRecord,
) -> None:
  again: RunRecord = json.loads(json.dumps(real_record))
  assert render_run(again) == render_run(real_record)


class References(HTMLParser):
  """Every attribute value that can make a browser fetch something."""

  def __init__(self) -> None:
    super().__init__()
    self.urls: list[str] = []
    self.tags: list[str] = []

  def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
    self.tags.append(tag)
    for name, value in attrs:
      if name in {"src", "href", "action", "data", "poster", "srcset", "formaction"}:
        self.urls.append(value or "")


def test_the_html_references_no_external_url(report: str) -> None:
  parser = References()
  parser.feed(report)
  assert not [u for u in parser.urls if not u.startswith("#")]
  assert "link" not in parser.tags
  assert "iframe" not in parser.tags
  assert "img" not in parser.tags
  assert re.search(r"url\(", report) is None
  assert re.search(r"@import", report) is None
  assert re.search(r"<link\b", report) is None
  assert re.search(r"\b(?:fetch|XMLHttpRequest|WebSocket|sendBeacon)\b", report) is None


def test_hostile_text_cannot_add_an_external_reference(record: RunRecord) -> None:
  text = render_run(hostile_record(record))
  parser = References()
  parser.feed(text)
  assert not [u for u in parser.urls if not u.startswith("#")]
  assert re.search(r"url\(", text) is None


def test_the_assets_themselves_are_offline() -> None:
  css = (h._ASSETS / "report.css").read_text(encoding="utf-8")
  js = (h._ASSETS / "report.js").read_text(encoding="utf-8")
  for text in (css, js):
    assert "http://" not in text
    assert "https://" not in text
    assert "</script" not in text.lower()
    assert "</style" not in text.lower()
  assert "@import" not in css
  assert "url(" not in css
  assert ".scenario" not in css
  assert ".panel" in css


def test_inlined_assets_are_the_files_of_the_package(report: str) -> None:
  css = (h._ASSETS / "report.css").read_text(encoding="utf-8")
  js = (h._ASSETS / "report.js").read_text(encoding="utf-8")
  assert f"<style>\n{css}</style>" in report
  assert f"<script>\n{js}</script>" in report


# ------------------------------------------------------------------ the tab script


def test_script_contract_by_inspection() -> None:
  js = (h._ASSETS / "report.js").read_text(encoding="utf-8")
  assert '"hermes-report-tab:"' in js
  assert ".panel" in js
  assert "location.hash" in js
  assert "scenario" not in js


NODE_HARNESS = r"""
const source = require('fs').readFileSync(process.argv[2], 'utf8');
const state = JSON.parse(process.argv[3]);
const makeEl = (key) => {
  const classes = new Set();
  return { dataset: { key }, classList: { toggle(c, on) { on ? classes.add(c) : classes.delete(c); }, has: (c) => classes.has(c) }, addEventListener() {} };
};
const keys = state.keys;
const tabs = keys.map(makeEl);
const panels = keys.map(makeEl);
const store = Object.assign({}, state.storage);
const document = {
  title: 'T',
  querySelectorAll: (sel) => (sel === '.tab' ? tabs : panels),
  addEventListener() {},
};
const location = { hash: state.hash };
const window = { addEventListener() {}, scrollTo() {} };
const localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = v; } };
const history = { replaceState(a, b, url) { location.hash = url; } };
new Function('document', 'location', 'window', 'localStorage', 'history', source)(
  document, location, window, localStorage, history);
console.log(JSON.stringify({
  active: keys.filter((k, i) => tabs[i].classList.has('active')),
  activePanels: keys.filter((k, i) => panels[i].classList.has('active')),
  store,
  hash: location.hash,
}));
"""


@pytest.fixture
def run_script(tmp_path: Path) -> Callable[..., dict[str, Any]]:
  node = shutil.which("node")
  if node is None:
    pytest.skip("node is not installed")
  harness = tmp_path / "harness.js"
  harness.write_text(NODE_HARNESS, encoding="utf-8")
  script = tmp_path / "report.js"
  script.write_text((h._ASSETS / "report.js").read_text(encoding="utf-8"))

  def run(hash_: str, storage: dict[str, str]) -> dict[str, Any]:
    state = {"keys": ["resumen", "corpus", "q-q05"], "hash": hash_, "storage": storage}
    done = subprocess.run(
      [node, str(harness), str(script), json.dumps(state)],
      capture_output=True,
      text=True,
      check=True,
      timeout=20,
    )
    result: dict[str, Any] = json.loads(done.stdout)
    return result

  return run


def test_script_opens_the_first_tab_by_default(
  run_script: Callable[..., dict[str, Any]],
) -> None:
  result = run_script("", {})
  assert result["active"] == ["resumen"]
  assert result["activePanels"] == ["resumen"]


def test_script_restores_the_saved_tab(
  run_script: Callable[..., dict[str, Any]],
) -> None:
  result = run_script("", {"hermes-report-tab:T": "corpus"})
  assert result["active"] == ["corpus"]


def test_a_hash_naming_a_tab_wins_over_the_saved_tab(
  run_script: Callable[..., dict[str, Any]],
) -> None:
  result = run_script("#q-q05", {"hermes-report-tab:T": "corpus"})
  assert result["active"] == ["q-q05"]
  assert result["activePanels"] == ["q-q05"]
  assert result["store"]["hermes-report-tab:T"] == "q-q05"


def test_a_hash_that_names_no_tab_falls_back_to_the_saved_tab(
  run_script: Callable[..., dict[str, Any]],
) -> None:
  result = run_script("#nonsense", {"hermes-report-tab:T": "corpus"})
  assert result["active"] == ["corpus"]


def test_an_unknown_saved_tab_falls_back_to_the_first(
  run_script: Callable[..., dict[str, Any]],
) -> None:
  result = run_script("", {"hermes-report-tab:T": "gone"})
  assert result["active"] == ["resumen"]


# ------------------------------------------------------------------ html helpers


def test_esc_escapes_and_accepts_non_strings() -> None:
  assert h.esc('<a href="x">&\'') == "&lt;a href=&quot;x&quot;&gt;&amp;&#x27;"
  assert h.esc(5) == "5"
  assert h.esc(None) == "None"
  assert h.esc("") == ""


@pytest.mark.parametrize(
  ("value", "expected"),
  [
    (0, "0.0 s"),
    (4.2, "4.2 s"),
    (4.25, "4.2 s"),
    (59.96, "60.0 s"),
    (123.456, "123.5 s"),
  ],
)
def test_seconds(value: float, expected: str) -> None:
  assert h.seconds(value) == expected


def test_json_block_is_indented_escaped_and_valid_json() -> None:
  value = {"a": [1, 2], "b": HOSTILE, "ñ": "á"}
  block = h.json_block(value)
  assert block.startswith('<pre class="text json">')
  assert "\n  " in block
  assert "<script>" not in block
  body = block.removeprefix('<pre class="text json">').removesuffix("</pre>")
  assert json.loads(html_lib.unescape(body)) == value
  assert "ñ" in block  # not \u-escaped


def test_json_block_of_non_serialisable_value_raises() -> None:
  with pytest.raises(TypeError):
    h.json_block({"x": object()})


def test_status_pill_known_none_and_unknown() -> None:
  assert h.status_pill("retrieval_failure") == (
    '<span class="status bad">fallo de recuperación</span>'
  )
  assert h.status_pill("correct") != h.status_pill("correct", "judge")
  assert h.status_pill("correct", "judge") == (
    '<span class="status good">correcta</span>'
  )
  assert h.status_pill(None) == '<span class="muted">—</span>'
  assert h.status_pill("<b>", "judge") == '<span class="status info">&lt;b&gt;</span>'
  assert h.status_pill("", "rule") == '<span class="status info"></span>'


def test_status_pill_rejects_an_unknown_kind() -> None:
  with pytest.raises(KeyError):
    h.status_pill("correct", "nope")  # type: ignore[arg-type]


def test_section_and_chip_escape_their_plain_text_but_not_the_body() -> None:
  out = h.section('a"b', "<1>", "<t>", "<s>", "<p>body</p>")
  assert 'id="a&quot;b"' in out
  assert "&lt;1&gt;" in out and "&lt;t&gt;" in out and "&lt;s&gt;" in out
  assert "<p>body</p>" in out
  chip = h.chip('a"b', "<t>", "<s>")
  assert chip == (
    '<a class="chip" href="#a&quot;b"><b>&lt;t&gt;</b><span>&lt;s&gt;</span></a>'
  )


def test_answer_html() -> None:
  assert h.answer_html("") == ""
  assert h.answer_html("a **b** [2]") == (
    'a <strong>b</strong> <span class="cite">2</span>'
  )
  assert h.answer_html("**a**\n**b**") == "<strong>a</strong>\n<strong>b</strong>"
  assert h.answer_html("[x] [] [1a]") == "[x] [] [1a]"
  assert "<" not in h.answer_html(HOSTILE).replace("<strong>", "")


def test_page_assembles_tabs_and_panels() -> None:
  out = h.page(
    "T<",
    "<i>meta</i>",
    [("a", "A", "hover a"), ("b<", 'B"', "hover b")],
    [("a", "<p>1</p>"), ("b<", "<p>2</p>")],
  )
  assert "<title>Hermes Report · T&lt;</title>" in out
  assert '<div class="meta"><i>meta</i></div>' in out
  assert '<button class="tab" data-key="a" title="hover a">A</button>' in out
  assert 'data-key="b&lt;"' in out and ">B&quot;</button>" in out
  assert '<section class="panel" id="tab-a" data-key="a"><p>1</p></section>' in out


def test_page_with_no_tabs_is_still_a_document() -> None:
  out = h.page("T", "", [], [])
  assert out.startswith("<!doctype html>") and out.endswith("</html>\n")


# ------------------------------------------------------------------ write_report


def test_write_report_creates_parents_and_writes_utf8(
  real_record: RunRecord, tmp_path: Path
) -> None:
  target = tmp_path / "a" / "b" / "report.html"
  write_report(real_record, target)
  assert target.read_text(encoding="utf-8") == render_run(real_record)
  assert "¿Cuántos días" in target.read_bytes().decode("utf-8")


def test_write_report_overwrites(real_record: RunRecord, tmp_path: Path) -> None:
  target = tmp_path / "report.html"
  target.write_text("old", encoding="utf-8")
  write_report(real_record, target)
  assert target.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_write_report_to_a_directory_raises_record_error(
  real_record: RunRecord, tmp_path: Path
) -> None:
  with pytest.raises(RecordError) as caught:
    write_report(real_record, tmp_path)
  assert str(tmp_path) in str(caught.value)
  assert "no se puede escribir" in str(caught.value)


def test_write_report_under_a_file_raises_record_error(
  real_record: RunRecord, tmp_path: Path
) -> None:
  blocker = tmp_path / "file"
  blocker.write_text("x")
  with pytest.raises(RecordError):
    write_report(real_record, blocker / "report.html")


# ------------------------------------------------------------------ malformed records


@pytest.mark.parametrize(
  "key", ["questions", "corpus", "ingest", "collection", "points"]
)
def test_a_record_missing_a_section_raises_a_lookup_error(
  record: RunRecord, key: str
) -> None:
  del record[key]  # type: ignore[misc]
  with pytest.raises(KeyError):
    render_run(record)


def test_no_warning_is_printed_while_rendering(record: RunRecord) -> None:
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    render_run(record)

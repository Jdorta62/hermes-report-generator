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
from hermes_report.render.score import primary_scores

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


def active_scripts(page: Page) -> list[dict[str, str | None]]:
  """The ``<script>`` elements that run: those without a ``type``."""
  return [attrs for attrs in page.tags("script") if "type" not in attrs]


def json_blocks(text: str) -> dict[str, Any]:
  """The embedded ``application/json`` blocks, parsed as the browser would.

  The pattern stops at the first ``</script>``, so a block that a payload could
  close early would come back truncated and fail to parse.
  """
  found = re.findall(
    r'<script type="application/json" id="([^"]+)">(.*?)</script>', text, re.S
  )
  assert len({name for name, _ in found}) == len(found), "a block appears twice"
  return {name: json.loads(body) for name, body in found}


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


def test_chunks_panel_is_a_heading_and_the_explorer_with_a_spanish_notice(
  report: str,
) -> None:
  panel = panel_of(report, "chunks")
  assert panel.startswith('<div class="sc-head"><h2>Chunks en Qdrant</h2></div>')
  # the explorer is empty server side: only the notice for browsers without script
  explorer = re.search(r'<div id="explorer">(.*?)</div>', panel, re.S)
  assert explorer
  assert re.fullmatch(
    r"<noscript><p [^>]*>[^<]*JavaScript[^<]*</p></noscript>", explorer[1]
  )
  assert "El resto del informe se puede leer sin él" in explorer[1]
  assert report.count('id="explorer"') == 1
  # the chunks themselves are not in the panel: the script draws them
  assert "chunk-card" not in panel


def test_three_scripts_two_of_them_json_and_one_style_that_matters(
  report: str,
) -> None:
  page = parse(report)
  scripts = page.tags("script")
  assert len(scripts) == 3
  assert [a.get("id") for a in scripts if a.get("type") == "application/json"] == [
    "hr-points",
    "hr-usage",
  ]
  assert len(active_scripts(page)) == 1
  assert active_scripts(page)[0] == {}
  # The data comes before the script that reads it, and each block is unique.
  assert report.index('id="hr-points"') < report.index('id="hr-usage"')
  assert report.index('id="hr-usage"') < report.index("<script>\n")
  # one stylesheet, plus the small one inside <noscript> that shows every panel
  assert len(page.tags("style")) == 2
  assert report.count("<noscript><style>") == 1


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


def test_summary_table_one_row_per_question_linking_to_its_tab(
  report: str, real_record: RunRecord
) -> None:
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
    *[h.metric_label(score) for score in primary_scores(real_record)],
  ]
  # one primary score per default-on family, and every row has a cell for each
  assert len(primary_scores(real_record)) == len(real_record["metrics"]["selection"])  # type: ignore[index]
  assert all(row.count("<td") == len(headers) for row in rows)


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
  item["metrics"] = None
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
  assert len(active_scripts(page)) == 1
  assert len(page.tags("script")) == 3
  assert "<script>alert" not in text
  # the data blocks carry the raw text, but inert: the page outside them is escaped
  markup = re.sub(r'<script type="application/json".*?</script>', "", text, flags=re.S)
  assert "onerror=alert(1)>" not in markup.replace("onerror=alert(1)&gt;", "")
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
  record["points"][0]["payload"]["text"] = "</script><script>alert(1)</script>"
  text = render_run(record)
  # the active script and the two data blocks: one closing tag each, nothing more
  assert text.count("</script>") == 3
  assert text.count("<script>") == 1
  assert text.count('<script type="application/json"') == 2
  assert len(json_blocks(text)) == 2


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


# ------------------------------------------------------------------ question tabs: phases
#
# The tests below read the phase sections of a question tab. Records come from
# real offline runs (module fixtures) or are built by hand with the ``pt`` /
# ``rf`` / ``pc`` / ``cx`` helpers, so that every branch of the renderer is
# reached from data alone, with no service involved.


class FakeCounter:
  """A token counter that counts words and never touches the network."""

  name = "fake-words"

  def count(self, text: str) -> int:
    return len(text.split())


FAKE = FakeCounter()


def offline_record(
  tmp_path_factory: pytest.TempPathFactory,
  *,
  threshold: float | None = None,
  budget: int | None = None,
) -> RunRecord:
  """A whole offline run over the sample project with another threshold / budget."""
  project = tmp_path_factory.mktemp("variant") / "proj"
  shutil.copytree(DATA, project)
  if threshold is not None:
    path = project / "run.toml"
    text = path.read_text(encoding="utf-8")
    assert "score_threshold = 0.3" in text
    path.write_text(
      text.replace("score_threshold = 0.3", f"score_threshold = {threshold}"),
      encoding="utf-8",
    )
  config = load_config(
    project / "run.toml", Overrides(out=project / "out", budget=budget)
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    return runner.run(config, progress=lambda line: None).record


@pytest.fixture(scope="module")
def budget_record(tmp_path_factory: pytest.TempPathFactory) -> RunRecord:
  """Every chunk retrieved and a 60-token budget: q01 drops chunks."""
  return offline_record(tmp_path_factory, threshold=0.0, budget=60)


@pytest.fixture(scope="module")
def empty_record(tmp_path_factory: pytest.TempPathFactory) -> RunRecord:
  """A threshold nothing reaches: every retrieval is empty."""
  return offline_record(tmp_path_factory, threshold=0.99)


@pytest.fixture(scope="module")
def failed_budget_record(tmp_path_factory: pytest.TempPathFactory) -> RunRecord:
  """A 30-token budget: q01 fails with ContextBudgetError after its steps."""
  return offline_record(tmp_path_factory, threshold=0.0, budget=30)


def phase_ids(panel: str) -> list[str]:
  """The anchor suffixes of the phase sections of a panel, in page order."""
  found = re.findall(r'<details class="phase" id="q-[^"]*?-([a-z0-9]+)" open>', panel)
  return found


def phase(panel: str, suffix: str) -> str:
  """The HTML of the phase section ``suffix`` (up to the next phase)."""
  match = re.search(rf'<details class="phase" id="q-[^"]*?-{suffix}" open>', panel)
  assert match, f"no phase {suffix} in {phase_ids(panel)}"
  start = match.end()
  after = panel.find('<details class="phase"', start)
  return panel[start : after if after != -1 else len(panel)]


def flow_targets(panel: str) -> list[str]:
  flow = re.search(r'<div class="flow">(.*?)</div>', panel, re.S)
  return re.findall(r'href="#([^"]*)"', flow.group(1)) if flow else []


def text_of(fragment: str) -> str:
  return html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment))


def pt(
  chunk_id: str,
  source: str | None,
  index: int | None,
  heading: list[str] | None = None,
  text: str = "texto del chunk",
) -> Any:
  payload: dict[str, Any] = {"text": text}
  if source is not None:
    payload["source"] = source
  if index is not None:
    payload["chunk_index"] = index
  if heading is not None:
    payload["heading_path"] = heading
  return {"id": chunk_id, "payload": payload, "vector": None}


def rf(
  chunk_id: str,
  score: float | None = 0.5,
  origin: tuple[str, ...] = (),
  source: str | None = None,
  index: int | None = None,
) -> Any:
  return {
    "chunk_id": chunk_id,
    "collection_name": "coll",
    "source": source,
    "doc_id": None,
    "chunk_index": index,
    "score": score,
    "origin": list(origin),
  }


def pc(text: str, *refs: Any, transforms: tuple[str, ...] = ()) -> Any:
  return {"text": text, "transforms": list(transforms), "chunks": list(refs)}


def cx(*pieces: Any, dropped: tuple[str, ...] = ()) -> Any:
  return {
    "pieces": list(pieces),
    "enrichments": [],
    "dropped_chunk_ids": list(dropped),
    "query_id": None,
  }


def rc(
  chunk_id: str,
  source: str | None = None,
  index: int | None = None,
  *,
  score: float = 0.5,
  vector: float | None = None,
  text: str = "texto",
  metadata: dict[str, Any] | None = None,
) -> Any:
  meta: dict[str, Any] = dict(metadata or {})
  if source is not None:
    meta["source"] = source
  if index is not None:
    meta["chunk_index"] = index
  return {
    "chunk_id": chunk_id,
    "text": text,
    "score": score,
    "vector_score": vector,
    "collection_name": "coll",
    "metadata": meta,
  }


def step(name: str, before: Any, after: Any, reads: list[Any] | None = None) -> Any:
  return {
    "name": name,
    "phase": "EXPAND",
    "seconds": 0.002,
    "before": before,
    "after": after,
    "reads": reads or [],
  }


def hand_record(
  record: RunRecord,
  *,
  points: list[Any],
  chunks: list[Any] | None = None,
  steps: list[Any] | None = None,
  final: Any = None,
  budget: dict[str, Any] | None = None,
  documents: list[str] | None = None,
  citations: list[Any] | None = None,
  result_dropped: list[str] | None = None,
) -> RunRecord:
  """Replace the phases of the first question with hand-built ones."""
  item = record["questions"][0]
  record["points"] = points
  if documents is not None:
    item["spec"]["documents"] = documents
  assert item["retrieval"] and item["enrich"]
  assert item["generation"] and item["generation"]["result"]
  item["retrieval"]["chunks"] = chunks or []
  item["enrich"]["steps"] = steps or []
  item["enrich"]["context"] = final
  if budget is not None:
    item["enrich"]["budget"] = budget
  result = item["generation"]["result"]
  if citations is not None:
    result["citations"] = citations
  if result_dropped is not None:
    result["dropped_chunk_ids"] = result_dropped
  return record


def q01(record: RunRecord, counter: Any = FAKE) -> str:
  return panel_of(render_run(record, token_counter=counter), "q-q01")


# ---- every section on the in-memory run


def test_real_run_has_every_section_in_pipeline_order(report: str) -> None:
  panel = panel_of(report, "q-q01")
  assert phase_ids(panel) == [
    "in",
    "seed",
    "e1",
    "e2",
    "e3",
    "budget",
    "prompt",
    "llm",
    "out",
    "judge",
    "metrics",
  ]


def test_flow_has_one_chip_per_section_each_linking_to_its_section(report: str) -> None:
  for key in ("q-q01", "q-q02", "q-q11"):
    panel = panel_of(report, key)
    ids = re.findall(r'<details class="phase" id="([^"]*)"', panel)
    assert flow_targets(panel) == ids
    assert len(ids) == len(set(ids))
    assert all(i.startswith(f"{key}-") for i in ids)


def test_section_anchors_are_unique_across_the_whole_page(report: str) -> None:
  ids = re.findall(r'\bid="([^"]*)"', report)
  assert len(ids) == len(set(ids))


def test_flow_chips_carry_the_step_names_and_summaries(report: str) -> None:
  panel = panel_of(report, "q-q01")
  flow = re.search(r'<div class="flow">(.*?)</div>', panel, re.S)
  assert flow
  titles = re.findall(r"<b>(.*?)</b>", flow.group(1))
  assert titles == [
    "Retrieval",
    "Semilla",
    "neighbors",
    "merge",
    "doc_order",
    "Presupuesto",
    "Prompt",
    "LLM",
    "Resultado",
    "Juez",
    "Métricas",
  ]


def test_section_step_marks_and_titles(report: str) -> None:
  panel = panel_of(report, "q-q01")
  marks = re.findall(
    r'<span class="step">(.*?)</span><span class="ph-title">(.*?)</span>', panel
  )
  assert marks == [
    ("0", "Entrada"),
    ("S", "Semilla del contexto"),
    ("1", "Enriquecedor: neighbors"),
    ("2", "Enriquecedor: merge"),
    ("3", "Enriquecedor: doc_order"),
    ("B", "Ajuste al presupuesto"),
    ("P", "Prompt enviado"),
    ("L", "LLM"),
    ("R", "GenerationResult"),
    ("J", "Juez"),
    ("M", "Métricas"),
  ]


def test_header_meta_names_the_token_counter(record: RunRecord) -> None:
  text = render_run(record, token_counter=FAKE)
  assert "Tokens: <b>fake-words</b>" in text
  assert "Tokens: <b>tiktoken</b>" in render_run(record)


def test_the_token_counter_is_created_once_per_render(
  record: RunRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
  made: list[FakeCounter] = []

  def factory() -> FakeCounter:
    made.append(FakeCounter())
    return made[-1]

  monkeypatch.setattr("hermes_report.render.run.default_token_counter", factory)
  render_run(record)
  assert len(made) == 1  # three questions, one counter


def test_an_explicit_counter_means_default_is_never_built(
  record: RunRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
  def boom() -> None:
    raise AssertionError("default_token_counter must not be called")

  monkeypatch.setattr("hermes_report.render.run.default_token_counter", boom)
  render_run(record, token_counter=FAKE)


def test_token_numbers_come_from_the_counter(record: RunRecord) -> None:
  class Ten:
    name = "ten"

    def count(self, text: str) -> int:
      return 10

  panel = q01(record, Ten())
  budget = phase(panel, "budget")
  final = record["questions"][0]["enrich"]["context"]  # type: ignore[index]
  assert final
  assert f"{10 * len(final['pieces'])} tok" in budget
  assert "tok" in phase(panel, "seed")


# ---- 0 Entrada


def test_entrada_shows_the_question_queries_and_retrieved_chunks_in_order(
  report: str, real_record: RunRecord
) -> None:
  item = real_record["questions"][0]
  assert item["retrieval"]
  body = phase(panel_of(report, "q-q01"), "in")
  assert (
    f'<div class="question">{html_lib.escape(item["spec"]["question"])}</div>' in body
  )
  assert "Consultas buscadas" in body
  queries = item["retrieval"]["queries"]
  for query in queries:
    assert html_lib.escape(query["text"]) in body
  rows = rows_of(body)
  chunks = item["retrieval"]["chunks"]
  assert len(rows) == len(queries) + len(chunks)
  chunk_rows = rows[len(queries) :]
  for number, (row, chunk) in enumerate(zip(chunk_rows, chunks, strict=True), 1):
    meta = chunk["metadata"]
    assert row.startswith(f"<tr><td>{number}</td>")
    assert (
      f"<code>{html_lib.escape(meta['source'])}#{meta['chunk_index']}</code>" in row
    )
    assert " › ".join(meta["heading_path"]) in html_lib.unescape(row)


def test_entrada_chunk_scores_and_vector_scores(
  report: str, real_record: RunRecord
) -> None:
  body = phase(panel_of(report, "q-q01"), "in")
  assert "<th>vector</th>" in body
  chunk = real_record["questions"][0]["retrieval"]["chunks"][0]  # type: ignore[index]
  assert chunk["vector_score"] is not None
  assert f"<td>{chunk['score']:.4f}</td>" in body  # RRF scores are below 0.1
  assert f"<td>{chunk['vector_score']:.2f}</td>" in body


def test_entrada_searched_queries_show_their_technique_chain(
  record: RunRecord,
) -> None:
  item = record["questions"][0]
  assert item["retrieval"]
  item["retrieval"]["queries"] = [
    {"text": "primera", "chain": ["multi_query", "hyde"]},
    {"text": "segunda", "chain": []},
  ]
  body = phase(q01(record), "in")
  rows = [r for r in rows_of(body) if "primera" in r or "segunda" in r]
  assert len(rows) == 2
  assert '<span class="tag">multi_query</span> <span class="tag">hyde</span>' in rows[0]
  assert '<span class="muted">—</span>' in rows[1]


def test_entrada_with_no_queries_says_so(record: RunRecord) -> None:
  item = record["questions"][0]
  assert item["retrieval"]
  item["retrieval"]["queries"] = []
  assert "Sin consultas." in phase(q01(record), "in")


def test_entrada_label_is_the_chunk_id_without_a_source_and_place_falls_back(
  record: RunRecord,
) -> None:
  chunks = [
    rc("c-with", "doc.md", 3, metadata={"heading_path": ["A", "B"]}),
    rc("c-nosource", None, 4, metadata={"char_start": 10, "char_end": 99}),
    rc("c-bare", None, None),
    rc("c-sourceonly", "doc.md", None, metadata={"heading_path": []}),
  ]
  body = phase(q01(hand_record(record, points=[], chunks=chunks)), "in")
  rows = rows_of(body)[1:]  # the first row is the query
  assert "<code>doc.md#3</code></td><td>A › B</td>" in rows[0]
  assert "<code>c-nosource</code></td><td>caracteres 10–99</td>" in rows[1]
  assert "<code>c-bare</code></td><td>—</td>" in rows[2]
  assert "<code>doc.md</code>" in rows[3]  # no index: just the source


@pytest.mark.parametrize(
  ("size", "ellipsis"),
  [(0, False), (139, False), (140, False), (141, True), (500, True)],
)
def test_entrada_snippet_is_cut_at_140_characters(
  record: RunRecord, size: int, ellipsis: bool
) -> None:
  text = "ñ" * size
  body = phase(q01(hand_record(record, points=[], chunks=[rc("c", text=text)])), "in")
  snippet = re.search(r'<td class="snippet">(.*?)</td>', body, re.S)
  assert snippet
  shown = html_lib.unescape(snippet.group(1))
  assert shown == "ñ" * min(size, 140) + ("…" if ellipsis else "")


def test_entrada_snippet_strips_surrounding_whitespace_before_cutting(
  record: RunRecord,
) -> None:
  chunk = rc("c", text="\n\n  " + "a" * 140 + "  \n")
  body = phase(q01(hand_record(record, points=[], chunks=[chunk])), "in")
  assert f'<td class="snippet">{"a" * 140}</td>' in body


def test_entrada_vector_column_is_left_out_when_no_chunk_has_a_vector_score(
  record: RunRecord,
) -> None:
  chunks = [rc("a", "d.md", 0, vector=None), rc("b", "d.md", 1, vector=None)]
  assert "<th>vector</th>" not in phase(
    q01(hand_record(record, points=[], chunks=chunks)), "in"
  )


def test_entrada_vector_column_shows_a_dash_for_a_chunk_without_one(
  record: RunRecord,
) -> None:
  chunks = [rc("a", "d.md", 0, vector=0.5), rc("b", "d.md", 1, vector=None)]
  body = phase(q01(hand_record(record, points=[], chunks=chunks)), "in")
  assert "<th>vector</th>" in body
  rows = rows_of(body)[1:]
  assert "<td>0.50</td>" in rows[0]
  assert '<td><span class="muted">—</span></td>' in rows[1]


@pytest.mark.parametrize(
  ("score", "shown"),
  [(0.5, "0.50"), (0.1, "0.10"), (0.0999, "0.0999"), (0.0, "0.0000"), (-0.25, "-0.25")],
)
def test_entrada_score_format(record: RunRecord, score: float, shown: str) -> None:
  body = phase(q01(hand_record(record, points=[], chunks=[rc("a", score=score)])), "in")
  assert f"<td>{shown}</td>" in body


# ---- empty retrieval


def test_empty_retrieval_has_input_and_empty_seed_and_no_enrichment_sections(
  empty_record: RunRecord,
) -> None:
  panel = panel_of(render_run(empty_record), "q-q01")
  assert phase_ids(panel) == ["in", "seed", "llm", "out", "judge", "metrics"]
  assert "Lista vacía." in phase(panel, "in")
  seed = phase(panel, "seed")
  assert "Sin chunks no hay semilla" in seed
  assert '<span class="ph-sum">vacía</span>' in seed
  assert flow_targets(panel)[1].endswith("-seed")


def test_empty_retrieval_never_calls_the_llm_and_says_so(
  empty_record: RunRecord,
) -> None:
  panel = panel_of(render_run(empty_record), "q-q01")
  llm = phase(panel, "llm")
  assert "sin llamar al LLM" in llm
  assert '<span class="ph-sum">no se llama</span>' in panel
  assert '<div class="error">' not in llm


def test_empty_retrieval_has_no_budget_and_no_prompt_section(
  empty_record: RunRecord,
) -> None:
  panel = panel_of(render_run(empty_record), "q-q01")
  assert "-budget" not in panel
  assert "-prompt" not in panel
  assert "Mapa de documentos" in panel  # the expected document still shows


def test_empty_retrieval_document_map_shows_the_expected_row_unused(
  empty_record: RunRecord,
) -> None:
  rows = doc_map(panel_of(render_run(empty_record), "q-q01"))
  assert list(rows) == ["parte-1/capitulo-01.md"]
  assert rows["parte-1/capitulo-01.md"]["expected"] is True
  assert {c["state"] for c in rows["parte-1/capitulo-01.md"]["cells"]} == {"unused"}


def test_empty_retrieval_result_section_has_no_citations(
  empty_record: RunRecord,
) -> None:
  out = phase(panel_of(render_run(empty_record), "q-q01"), "out")
  assert "Sin citas." in out
  assert h.status_pill("no_context", "generation") in out


# ---- S Semilla


def test_seed_is_the_before_of_the_first_step(
  report: str, real_record: RunRecord
) -> None:
  enrich = real_record["questions"][0]["enrich"]
  assert enrich
  seed = enrich["steps"][0]["before"]
  body = phase(panel_of(report, "q-q01"), "seed")
  assert body.count('<div class="piece">') == len(seed["pieces"])
  for piece in seed["pieces"]:
    assert html_lib.escape(piece["text"]) in body


def test_seed_is_not_the_after_of_the_last_step(record: RunRecord) -> None:
  seed = cx(pc("seed words only", rf("s1", 0.9, source="d.md", index=0)))
  grown = cx(
    pc("seed words only", rf("s1", 0.9, source="d.md", index=0)),
    pc("neighbour text", rf("s2", None, ("neighbors",), source="d.md", index=1)),
  )
  record = hand_record(
    record,
    points=[pt("s1", "d.md", 0), pt("s2", "d.md", 1)],
    chunks=[rc("s1", "d.md", 0)],
    steps=[step("neighbors", seed, grown)],
    final=grown,
  )
  body = phase(q01(record), "seed")
  assert "seed words only" in body
  assert "neighbour text" not in body
  assert body.count('<div class="piece">') == 1


# ---- E1…En


def test_step_section_shows_reads_pieces_and_marks_new_chunks(
  record: RunRecord,
) -> None:
  before = cx(pc("uno dos", rf("a", 0.9, source="d.md", index=1)))
  after = cx(
    pc(
      "cero uno dos tres",
      rf("z", None, ("neighbors",), "d.md", 0),
      rf("a", 0.9),
      rf("b", None, ("neighbors",), "d.md", 2),
    )
  )
  reads = [
    {
      "ranges": [
        {"collection_name": "coll", "doc_id": "d" * 40, "first": 0, "last": 0},
        {"collection_name": "coll", "doc_id": "d" * 40, "first": 2, "last": 2},
      ],
      "returned": ["b", "z"],
      "seconds": 0.0123,
    }
  ]
  record = hand_record(
    record,
    points=[pt("z", "d.md", 0), pt("a", "d.md", 1), pt("b", "d.md", 2)],
    chunks=[rc("a", "d.md", 1)],
    steps=[step("neighbors", before, after, reads)],
    final=after,
  )
  body = phase(q01(record), "e1")
  assert "Lecturas a Qdrant" in body
  assert f"coll / {'d' * 16} [0–0], coll / {'d' * 16} [2–2]" in body
  assert "Recibe</span> 2 chunk(s)" in body
  # returned chunks are shown by chunk index, not in the order of the reply
  assert body.index("<code>d.md#0</code>") < body.index("<code>d.md#2</code>")
  assert "12.3 ms" in body
  assert body.count('class="pill added new"') == 2  # z and b are new
  assert 'class="pill orig"' in body  # a was already there
  assert "+2 chunks" in body
  assert '<span class="ph-sum">+2 chunks</span>' in body


def test_step_summary_variants(record: RunRecord) -> None:
  a, b = rf("a", 0.9, source="d.md", index=0), rf("b", 0.8, source="d.md", index=1)
  steps = [
    step("grow", cx(pc("a", a)), cx(pc("a", a), pc("b", b))),  # +1 chunk, 1->2
    step("merge", cx(pc("a", a), pc("b", b)), cx(pc("a b", a, b))),  # 2->1
    step("swap", cx(pc("a b", a, b)), cx(pc("a b", b, a))),  # reorder
    step("noop", cx(pc("a b", a, b)), cx(pc("a b", a, b))),  # nothing
  ]
  record = hand_record(
    record,
    points=[pt("a", "d.md", 0), pt("b", "d.md", 1)],
    chunks=[rc("a", "d.md", 0)],
    steps=steps,
    final=steps[-1]["after"],
  )
  panel = q01(record)
  sums = dict(
    re.findall(r'id="q-q01-(e\d)" open><summary>.*?ph-sum">(.*?)</span>', panel)
  )
  assert sums["e1"] == "+1 chunk, 1→2 piezas"
  assert sums["e2"] == "2→1 piezas"
  assert sums["e3"] == "reordena"
  assert sums["e4"] == "sin cambios"
  assert "No consulta Qdrant." in phase(panel, "e4")


def test_step_reads_of_a_chunk_unknown_to_the_record_fall_back_to_its_id(
  record: RunRecord,
) -> None:
  ctx = cx(pc("t", rf("lonely", 0.5)))
  reads = [{"ranges": [], "returned": ["ghost"], "seconds": 0.0}]
  record = hand_record(
    record, points=[], chunks=[], steps=[step("n", ctx, ctx, reads)], final=ctx
  )
  body = phase(q01(record), "e1")
  assert "<code>ghost</code>" in body
  assert "<b>lonely</b>" in body  # a ref with no point is named by its id


def test_step_piece_with_no_chunks_does_not_crash(record: RunRecord) -> None:
  ctx = cx(pc("orphan text"))
  record = hand_record(record, points=[], steps=[step("n", ctx, ctx)], final=ctx)
  body = phase(q01(record), "e1")
  assert "orphan text" in body
  assert "1. ?" in body


# ---- B Presupuesto


def test_budget_without_drops_says_everything_fits(report: str) -> None:
  body = phase(panel_of(report, "q-q01"), "budget")
  assert "Todo cabe: no se descarta nada." in body
  assert 'class="dropped"' not in body
  assert "<h4>EnrichedContext (contrato de hermes-core)</h4>" in body
  assert "3000 tok · 20 chunks" in body


def test_budget_drop_with_a_tiny_budget(budget_record: RunRecord) -> None:
  counter = default_counter()
  item = budget_record["questions"][0]
  assert item["enrich"] and item["enrich"]["context"]
  final = item["enrich"]["context"]
  last = item["enrich"]["steps"][-1]["after"]
  dropped = final["dropped_chunk_ids"]
  assert len(dropped) == 2  # the premise of this test

  panel = panel_of(render_run(budget_record), "q-q01")
  body = phase(panel, "budget")
  before_tok = sum(counter.count(p["text"]) for p in last["pieces"])
  after_tok = sum(counter.count(p["text"]) for p in final["pieces"])
  assert before_tok > 60 >= after_tok  # something really had to go
  assert "60 tok · 20 chunks" in body
  assert f"Antes</span> {before_tok} tok" in body
  assert f"Después</span> {after_tok} tok" in body
  assert f'<span class="ph-sum">{after_tok}/60 tok · −2 chunks</span>' in panel
  # the dropped chunks, in the order the record gives them, each marked gone
  items = re.findall(r"<li>(.*?)</li>", body, re.S)
  assert len(items) == 2
  for entry, chunk_id in zip(items, dropped, strict=True):
    assert 'class="pill orig gone"' in entry or 'class="pill added gone"' in entry
    assert chunk_id in entry  # in the title attribute


def test_budget_dropped_chunks_keep_the_record_order(record: RunRecord) -> None:
  pieces = [pc(f"w{n}", rf(f"c{n}", 0.5, source="d.md", index=n)) for n in range(4)]
  final = cx(pieces[0], dropped=("c3", "c1", "c2"))
  record = hand_record(
    record,
    points=[pt(f"c{n}", "d.md", n) for n in range(4)],
    chunks=[rc("c0", "d.md", 0)],
    steps=[step("n", cx(*pieces), cx(*pieces))],
    final=final,
    budget={"max_context_tokens": 10, "max_chunks": 3},
  )
  body = phase(q01(record), "budget")
  names = re.findall(r"<li>.*?<b>(.*?)</b>", body, re.S)
  assert names == ["d.md#3", "d.md#1", "d.md#2"]
  assert "Antes</span> 4 tok · 4 chunks" in body
  assert "Después</span> 1 tok · 1 chunk</div>" in body


@pytest.mark.parametrize(("max_chunks", "noun"), [(1, "chunk"), (2, "chunks")])
def test_budget_limit_noun_agrees_with_the_chunk_cap(
  record: RunRecord, max_chunks: int, noun: str
) -> None:
  pieces = [pc(f"w{n}", rf(f"c{n}", 0.5, source="d.md", index=n)) for n in range(2)]
  record = hand_record(
    record,
    points=[pt(f"c{n}", "d.md", n) for n in range(2)],
    chunks=[rc("c0", "d.md", 0)],
    steps=[step("n", cx(*pieces), cx(*pieces))],
    final=cx(*pieces),
    budget={"max_context_tokens": 10, "max_chunks": max_chunks},
  )
  body = phase(q01(record), "budget")
  assert f"Límite</span> 10 tok · {max_chunks} {noun}" in body
  assert f"{max_chunks} {noun}<span" in body
  assert "1 chunks" not in body


def test_budget_dropped_id_missing_from_the_last_step_still_listed(
  record: RunRecord,
) -> None:
  a = pc("a", rf("a", 0.5, source="d.md", index=0))
  record = hand_record(
    record,
    points=[pt("a", "d.md", 0)],
    chunks=[rc("a", "d.md", 0)],
    steps=[step("n", cx(a), cx(a))],
    final=cx(a, dropped=("ghost-id",)),
  )
  body = phase(q01(record), "budget")
  assert '<li><span class="pill gone"><b>ghost-id</b></span></li>' in body


@pytest.mark.parametrize(
  "budget",
  [{}, {"max_context_tokens": None, "max_chunks": None}, {"max_context_tokens": 0}],
)
def test_budget_without_limits_is_unlimited_and_has_an_empty_bar(
  record: RunRecord, budget: dict[str, Any]
) -> None:
  a = pc("uno dos tres", rf("a", 0.5, source="d.md", index=0))
  record = hand_record(
    record,
    points=[pt("a", "d.md", 0)],
    chunks=[rc("a", "d.md", 0)],
    steps=[step("n", cx(a), cx(a))],
    final=cx(a),
    budget=budget,
  )
  panel = q01(record)
  body = phase(panel, "budget")
  assert "sin límite tok · sin límite chunks" in body
  assert 'style="width:0.0%"' in body
  assert '<span class="ph-sum">3 tok</span>' in panel


def test_budget_bar_never_overflows_when_over_the_limit(record: RunRecord) -> None:
  a = pc("uno dos tres cuatro cinco", rf("a", 0.5, source="d.md", index=0))
  record = hand_record(
    record,
    points=[pt("a", "d.md", 0)],
    chunks=[rc("a", "d.md", 0)],
    steps=[step("n", cx(a), cx(a))],
    final=cx(a),
    budget={"max_context_tokens": 2, "max_chunks": 1},
  )
  widths = [
    float(w) for w in re.findall(r"width:([\d.]+)%", phase(q01(record), "budget"))
  ]
  assert widths == [100.0, 100.0]


def test_budget_final_context_shows_texts_open_for_small_contexts_only(
  record: RunRecord,
) -> None:
  def build(count: int) -> str:
    pieces = [
      pc(f"t{n}", rf(f"c{n}", 0.5, source="d.md", index=n)) for n in range(count)
    ]
    ctx = cx(*pieces)
    rec = hand_record(
      copy.deepcopy(record),
      points=[pt(f"c{n}", "d.md", n) for n in range(count)],
      chunks=[rc("c0", "d.md", 0)],
      steps=[step("n", ctx, ctx)],
      final=ctx,
    )
    body = phase(q01(rec), "budget")
    return body[body.index("EnrichedContext") :]

  assert build(4).count("<details open>") == 4
  assert build(5).count("<details open>") == 0
  assert build(5).count("<details>") == 5


def test_a_run_that_fails_the_budget_shows_its_steps_but_no_budget_section(
  failed_budget_record: RunRecord,
) -> None:
  text = render_run(failed_budget_record)
  panel = panel_of(text, "q-q01")
  assert phase_ids(panel)[:5] == ["in", "seed", "e1", "e2", "e3"]
  assert "budget" not in phase_ids(panel)
  assert not {"prompt", "llm", "out"} & set(phase_ids(panel))
  assert '<div class="error">ContextBudgetError' in panel
  assert h.status_pill("error", "generation") in panel


# ---- P Prompt


def test_prompt_shows_every_message_with_its_role(
  report: str, real_record: RunRecord
) -> None:
  result = real_record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result and result["prompt"]
  body = phase(panel_of(report, "q-q01"), "prompt")
  assert body.count('<div class="msg ') == len(result["prompt"])
  for message in result["prompt"]:
    assert f'<div class="msg {message["role"]}">' in body
    assert html_lib.escape(message["content"]) in body
  assert f"{len(result['prompt'])} mensajes" in report


@pytest.mark.parametrize("prompt", [None, []])
def test_no_prompt_means_no_prompt_section_and_no_chip(
  record: RunRecord, prompt: list[Any] | None
) -> None:
  result = record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result
  result["prompt"] = prompt
  panel = q01(record)
  assert "prompt" not in phase_ids(panel)
  assert not [t for t in flow_targets(panel) if t.endswith("-prompt")]


def test_result_without_a_prompt_key_at_all(record: RunRecord) -> None:
  result = record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result
  del result["prompt"]
  assert "prompt" not in phase_ids(q01(record))


# ---- L LLM


def llm_call(
  *,
  text: str | None = "respuesta",
  error: str | None = None,
  seconds: float = 0.25,
  params: dict[str, Any] | None = None,
  model: str | None = "m1",
  prompt_tokens: int | None = 11,
  completion_tokens: int | None = 7,
) -> Any:
  reply = (
    None
    if text is None
    else {
      "text": text,
      "model": model,
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
    }
  )
  return {
    "messages": [{"role": "user", "content": "hola"}],
    "params": params if params is not None else {"temperature": 0.0, "seed": 42},
    "reply": reply,
    "error": error,
    "seconds": seconds,
  }


def test_llm_section_of_the_real_run(report: str, real_record: RunRecord) -> None:
  generation = real_record["questions"][0]["generation"]
  assert generation
  call = generation["llm_calls"][0]
  body = phase(panel_of(report, "q-q01"), "llm")
  assert call["reply"]
  assert f"Modelo</span> {html_lib.escape(call['reply']['model'])}" in body
  assert html_lib.escape(call["reply"]["text"]) in body
  assert f"{call['seconds']:.2f} s" in body
  assert "Llamada" not in body  # one call: no per-call heading


def test_llm_section_lists_every_call_not_only_the_last(record: RunRecord) -> None:
  generation = record["questions"][0]["generation"]
  assert generation
  generation["llm_calls"] = [
    llm_call(text="primera respuesta", seconds=1.0),
    llm_call(
      text="segunda respuesta", seconds=2.0, params={"temperature": 0.7, "top_p": None}
    ),
    llm_call(text=None, error="boom: connection reset", seconds=0.5),
  ]
  panel = q01(record)
  body = phase(panel, "llm")
  assert (
    "Llamada 1 de 3" in body and "Llamada 2 de 3" in body and "Llamada 3 de 3" in body
  )
  assert "primera respuesta" in body and "segunda respuesta" in body
  assert '<div class="error">boom: connection reset</div>' in body
  assert "temperature=0.7" in body
  assert "top_p" not in body  # None parameters are not shown
  assert "<code>temperature=0.0, seed=42</code>" in body
  assert '<span class="ph-sum">3 llamadas · 3.50 s · con error</span>' in panel


def test_llm_call_without_reply_shows_the_error_box_and_summary(
  record: RunRecord,
) -> None:
  generation = record["questions"][0]["generation"]
  assert generation
  generation["llm_calls"] = [llm_call(text=None, error="timeout")]
  panel = q01(record)
  assert '<div class="error">timeout</div>' in phase(panel, "llm")
  assert '<span class="ph-sum">error</span>' in panel.split('id="q-q01-llm"')[1]


def test_llm_call_with_missing_model_and_tokens_shows_placeholders(
  record: RunRecord,
) -> None:
  generation = record["questions"][0]["generation"]
  assert generation
  generation["llm_calls"] = [
    llm_call(model=None, prompt_tokens=None, completion_tokens=None)
  ]
  body = phase(q01(record), "llm")
  assert "Modelo</span> ?" in body
  assert "prompt — · respuesta —" in body


def test_llm_zero_token_counts_are_real_numbers_not_missing(record: RunRecord) -> None:
  generation = record["questions"][0]["generation"]
  assert generation
  generation["llm_calls"] = [llm_call(prompt_tokens=0, completion_tokens=0)]
  body = phase(q01(record), "llm")
  assert "prompt 0 · respuesta 0" in body


def test_llm_section_of_a_generation_that_failed_before_any_call(
  record: RunRecord,
) -> None:
  item = record["questions"][0]
  assert item["generation"]
  item["generation"]["llm_calls"] = []
  item["generation"]["result"] = None
  item["error"] = "RuntimeError: prompt build failed"
  panel = q01(record)
  assert '<div class="error">RuntimeError: prompt build failed</div>' in phase(
    panel, "llm"
  )
  assert not {"prompt", "out"} & set(phase_ids(panel))


def test_llm_section_with_an_empty_raw_reply(record: RunRecord) -> None:
  generation = record["questions"][0]["generation"]
  assert generation
  generation["llm_calls"] = [llm_call(text="")]
  assert '<pre class="text raw"></pre>' in phase(q01(record), "llm")


# ---- R Resultado


def test_result_section_fields_and_json(report: str, real_record: RunRecord) -> None:
  result = real_record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result
  body = phase(panel_of(report, "q-q01"), "out")
  assert '<div class="answer">' in body
  assert h.status_pill(result["status"], "generation") in body
  assert f"Estrategia</span> {result['strategy']}" in body
  assert f"Formato</span> {result['response_format']}" in body
  assert f"{len(result['sources'])} piezas" in body
  assert "Enriquecimientos</span> " + " → ".join(result["enrichments"]) in body
  block = re.search(r'<pre class="text json">(.*?)</pre>', body, re.S)
  assert block
  assert "<details><summary>GenerationResult completo (JSON)</summary>" in body
  assert json.loads(html_lib.unescape(block.group(1))) == result


def test_result_citations_table_maps_numbers_to_chunk_labels_and_source(
  record: RunRecord,
) -> None:
  result = record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result
  result["citations"] = [
    {"number": 1, "chunk_ids": ["a", "b"]},
    {"number": 2, "chunk_ids": ["c"]},
    {"number": 3, "chunk_ids": ["nobody"]},
  ]
  result["sources"] = [
    {
      "number": 1,
      "chunks": [
        {"chunk_id": "a", "source": "x.md"},
        {"chunk_id": "b", "source": "x.md"},
      ],
    },
    {"number": 2, "chunks": [{"chunk_id": "c", "source": "y.md"}]},
  ]
  record["points"] = [pt("a", "x.md", 2), pt("b", "x.md", 3), pt("c", "y.md", 0)]
  body = phase(q01(record), "out")
  rows = rows_of(body)
  assert len(rows) == 3
  assert '<span class="cite">1</span>' in rows[0]
  # each label is a link to the explorer card; the labels still read as before
  assert re.findall(r"<code>(.*?)</code>", rows[0]) == ["x.md#2", "x.md#3"]
  assert rows[0].count("</a> <a ") == 1
  assert 'href="#chunks?doc=x.md&amp;q=q01&amp;chunk=a"' in rows[0]
  assert 'href="#chunks?doc=x.md&amp;q=q01&amp;chunk=b"' in rows[0]
  assert "<td>x.md</td>" in rows[0]
  assert "<code>y.md#0</code>" in rows[1] and "<td>y.md</td>" in rows[1]
  # a citation whose number has no source piece, and an unknown chunk id, survive
  assert "<code>nobody</code>" in rows[2]
  assert "<td></td>" in rows[2]


def test_result_without_citations_says_so_and_a_source_without_a_name_gets_a_mark(
  record: RunRecord,
) -> None:
  result = record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result
  result["citations"] = []
  assert "Sin citas." in phase(q01(record), "out")
  result["citations"] = [{"number": 1, "chunk_ids": ["a"]}]
  result["sources"] = [{"number": 1, "chunks": [{"chunk_id": "a"}]}]
  assert "<td>?</td>" in phase(q01(record), "out")


def test_result_survives_missing_optional_keys(record: RunRecord) -> None:
  item = record["questions"][0]
  assert item["generation"]
  item["generation"]["result"] = {"status": "answered", "answer": "hola"}
  body = phase(q01(record), "out")
  assert "Estrategia</span> ?" in body
  assert "Formato</span> ?" in body
  assert "Enriquecimientos</span> —" in body
  assert "0 piezas" in body
  assert "Sin citas." in body


# ---- J Juez


def test_judge_section_shows_messages_raw_reply_verdict_and_reason(
  report: str, real_record: RunRecord
) -> None:
  judge = real_record["questions"][0]["judge"]
  assert judge
  body = phase(panel_of(report, "q-q01"), "judge")
  assert h.status_pill(judge["verdict"], "judge") in body
  assert html_lib.escape(judge["reason"] or "") in body
  assert html_lib.escape(judge["model"]) in body
  assert body.count('<div class="msg ') == len(judge["messages"])
  for message in judge["messages"]:
    assert html_lib.escape(message["content"]) in body
  assert html_lib.escape(judge["raw"] or "") in body


def test_no_judge_means_no_judge_section_and_chip(record: RunRecord) -> None:
  record["questions"][0]["judge"] = None
  panel = q01(record)
  assert "judge" not in phase_ids(panel)
  assert not [t for t in flow_targets(panel) if t.endswith("-judge")]


def test_judge_that_failed_shows_the_error_box_and_no_reply(record: RunRecord) -> None:
  judge = record["questions"][0]["judge"]
  assert judge
  judge["verdict"] = None
  judge["reason"] = None
  judge["raw"] = None
  judge["error"] = "juez caído"
  panel = q01(record)
  body = phase(panel, "judge")
  assert '<div class="error">juez caído</div>' in body
  assert "Sin respuesta." in body
  assert '<span class="ph-sum">error</span>' in panel.split('id="q-q01-judge"')[1]


def test_judge_with_no_messages(record: RunRecord) -> None:
  judge = record["questions"][0]["judge"]
  assert judge
  judge["messages"] = []
  assert "Sin mensajes." in phase(q01(record), "judge")


# ---- a question whose run failed in each phase


def failed_at(record: RunRecord, where: str) -> RunRecord:
  """The first question as it is recorded when the pipeline died in ``where``."""
  item = record["questions"][0]
  item["error"] = f"{where}: se cayó"
  item["rules"] = None
  item["judge"] = None
  item["metrics"] = None
  if where == "retrieval":
    item["retrieval"] = item["enrich"] = item["generation"] = None
  elif where == "enrich":
    item["enrich"] = item["generation"] = None
  elif where == "enrich-late":
    assert item["enrich"]
    item["enrich"]["context"] = None
    item["generation"] = None
  elif where == "generation":
    assert item["generation"]
    item["generation"]["result"] = None
    item["generation"]["llm_calls"] = [llm_call(text=None, error="llm: se cayó")]
  elif where == "generation-early":
    assert item["generation"]
    item["generation"]["result"] = None
    item["generation"]["llm_calls"] = []
  elif where == "judge":
    item["judge"] = {
      "verdict": None,
      "reason": None,
      "model": "m",
      "seconds": 0.1,
      "messages": [],
      "raw": None,
      "error": "judge: se cayó",
    }
  return record


@pytest.mark.parametrize(
  ("where", "sections"),
  [
    ("retrieval", []),
    ("enrich", ["in"]),
    ("enrich-late", ["in", "seed", "e1", "e2", "e3"]),
    ("generation", ["in", "seed", "e1", "e2", "e3", "budget", "llm"]),
    ("generation-early", ["in", "seed", "e1", "e2", "e3", "budget", "llm"]),
    (
      "judge",
      ["in", "seed", "e1", "e2", "e3", "budget", "prompt", "llm", "out", "judge"],
    ),
  ],
)
def test_a_question_that_failed_in_one_phase_keeps_the_phases_before_it(
  record: RunRecord, where: str, sections: list[str]
) -> None:
  text = render_run(failed_at(record, where), token_counter=FAKE)
  panel = panel_of(text, "q-q01")
  assert phase_ids(panel) == sections
  assert len(flow_targets(panel)) == len(sections)
  assert f'<div class="error">{where}: se cayó</div>' in panel
  assert "Mapa de documentos" in panel
  assert tab_keys(text)[3] == "q-q01"


def test_failed_generation_shows_the_failed_call_and_not_a_fake_result(
  record: RunRecord,
) -> None:
  panel = q01(failed_at(record, "generation"))
  assert '<div class="error">llm: se cayó</div>' in phase(panel, "llm")
  assert h.status_pill("error", "generation") in panel


def test_failed_generation_before_any_call_shows_the_question_error_in_l(
  record: RunRecord,
) -> None:
  panel = q01(failed_at(record, "generation-early"))
  assert '<div class="error">generation-early: se cayó</div>' in phase(panel, "llm")


def test_failed_judge_keeps_every_other_section(record: RunRecord) -> None:
  panel = q01(failed_at(record, "judge"))
  assert '<div class="error">judge: se cayó</div>' in phase(panel, "judge")
  assert "-out" in panel


def test_a_phase_that_is_none_is_left_out_not_rendered_empty(record: RunRecord) -> None:
  panel = q01(failed_at(record, "retrieval"))
  assert '<details class="phase"' not in panel
  assert '<div class="flow">' not in panel
  assert "Entrada" not in panel


def test_every_combination_of_missing_phases_renders(record: RunRecord) -> None:
  """Each phase record is None in turn, and all together: nothing raises."""
  base = record
  for keys in (
    ("retrieval",),
    ("enrich",),
    ("generation",),
    ("rules",),
    ("judge",),
    ("retrieval", "enrich"),
    ("enrich", "generation"),
    ("retrieval", "generation", "judge"),
    ("retrieval", "enrich", "generation", "rules", "judge"),
  ):
    rec = copy.deepcopy(base)
    for key in keys:
      rec["questions"][0][key] = None  # type: ignore[literal-required]
    render_run(rec, token_counter=FAKE)


def test_retrieval_present_but_enrich_none_has_no_seed(record: RunRecord) -> None:
  record["questions"][0]["enrich"] = None
  ids = phase_ids(q01(record))
  assert "seed" not in ids and "budget" not in ids and "in" in ids


def test_enrich_present_but_retrieval_none_shows_no_phase_that_needs_it(
  record: RunRecord,
) -> None:
  record["questions"][0]["retrieval"] = None
  ids = phase_ids(q01(record))
  assert "in" not in ids and "seed" not in ids


# ---- document map


def doc_map(panel: str) -> dict[str, dict[str, Any]]:
  """``source -> {expected, count, cells: [{index, state, classes, title}]}``."""
  start = panel.index("Mapa de documentos")
  end = panel.find('<details class="phase"', start)
  card_html = panel[start : end if end != -1 else len(panel)]
  rows: dict[str, dict[str, Any]] = {}
  parts = card_html.split('<div class="maprow')[1:]
  for part in parts:
    head = re.match(
      r'( expected)?"><div class="mapdoc"><b>(.*?)</b><span>(.*?)</span>', part
    )
    assert head, part[:200]
    cells = []
    for classes, title, index in re.findall(
      r'<span class="(cell[^"]*)" title="([^"]*)">(\d+)</span>', part
    ):
      names = classes.split()
      states = {"orig", "added", "unused"} & set(names)
      assert len(states) == 1
      cells.append(
        {
          "index": int(index),
          "state": next(iter(states)),
          "classes": set(names),
          "title": html_lib.unescape(title),
        }
      )
    rows[html_lib.unescape(head.group(2))] = {
      "expected": bool(head.group(1)),
      "count": html_lib.unescape(head.group(3)),
      "cells": cells,
      "empty": "sin chunks en la colección" in part,
    }
  return rows


def map_record(record: RunRecord) -> RunRecord:
  """Documents with every cell class: orig, added, unused, dropped, cited."""
  points = [
    pt("z2", "z.md", 2),
    pt("z0", "z.md", 0),
    pt("a3", "a.md", 3),
    pt("a0", "a.md", 0, ["Parte", "Capítulo"], "  hola\n\n   mundo " + "x" * 300),
    pt("a2", "a.md", 2),
    pt("a1", "a.md", 1),
    pt("a4", "a.md", 4),
    pt("b0", "b.md", 0),
    pt("b1", "b.md", 1),
    pt("n0", None, None),  # no source and no index: never a cell
    pt("d0", "d.md", 0),
  ]
  a1 = rf("a1", 0.9, source="a.md", index=1)
  a2 = rf("a2", None, ("neighbors",), "a.md", 2)
  d0 = rf("d0", None, ("neighbors",), "d.md", 0)
  seed = cx(pc("uno", a1))
  grown = cx(pc("uno dos", a1, a2), pc("d", d0))
  final = cx(pc("uno", a1), dropped=("a2",))
  return hand_record(
    record,
    points=points,
    chunks=[rc("a1", "a.md", 1), rc("n0", None, None), rc("z0", "z.md", 0)],
    steps=[step("neighbors", seed, grown)],
    final=final,
    documents=["b.md", "ghost.md"],
    citations=[{"number": 1, "chunk_ids": ["a1", "z0"]}],
    result_dropped=["z0"],
  )


def test_document_map_rows_are_the_retrieved_added_and_expected_documents_sorted(
  record: RunRecord,
) -> None:
  rows = doc_map(q01(map_record(record)))
  # a.md retrieved, b.md and ghost.md expected, d.md only added, z.md retrieved;
  # "nosource" has a retrieved chunk but no source: no row.
  assert list(rows) == ["a.md", "b.md", "d.md", "ghost.md", "z.md"]


def test_document_map_has_one_cell_per_chunk_index_sorted_by_index(
  record: RunRecord,
) -> None:
  rows = doc_map(q01(map_record(record)))
  assert [c["index"] for c in rows["a.md"]["cells"]] == [0, 1, 2, 3, 4]
  assert [c["index"] for c in rows["z.md"]["cells"]] == [0, 2]
  assert rows["a.md"]["count"] == "5 chunks"


def test_document_map_cell_states(record: RunRecord) -> None:
  rows = doc_map(q01(map_record(record)))
  state = {c["index"]: c["state"] for c in rows["a.md"]["cells"]}
  assert state == {0: "unused", 1: "orig", 2: "added", 3: "unused", 4: "unused"}
  assert {c["state"] for c in rows["b.md"]["cells"]} == {"unused"}
  assert [c["state"] for c in rows["d.md"]["cells"]] == ["added"]
  assert {c["index"]: c["state"] for c in rows["z.md"]["cells"]} == {
    0: "orig",
    2: "unused",
  }


def test_document_map_dropped_and_cited_flags(record: RunRecord) -> None:
  rows = doc_map(q01(map_record(record)))
  cells = {c["index"]: c["classes"] for c in rows["a.md"]["cells"]}
  assert "dropped" in cells[2]  # dropped by the final context
  assert "dropped" not in cells[1] and "cited" in cells[1]
  assert "cited" not in cells[2]
  assert not ({"dropped", "cited"} & cells[0])
  z = {c["index"]: c["classes"] for c in rows["z.md"]["cells"]}
  # dropped only by the result's own list, and cited
  assert {"orig", "dropped", "cited"} <= z[0]
  assert not ({"dropped", "cited"} & z[2])


def test_document_map_dropped_is_the_union_of_context_and_result_lists(
  record: RunRecord,
) -> None:
  record = map_record(record)
  item = record["questions"][0]
  assert item["enrich"] and item["enrich"]["context"]
  assert item["generation"] and item["generation"]["result"]
  item["enrich"]["context"]["dropped_chunk_ids"] = ["a1"]
  item["generation"]["result"]["dropped_chunk_ids"] = ["a4", "a3"]
  cells = {c["index"]: c["classes"] for c in doc_map(q01(record))["a.md"]["cells"]}
  assert [i for i, c in cells.items() if "dropped" in c] == [1, 3, 4]


def test_document_map_expected_rows_and_missing_collection(record: RunRecord) -> None:
  rows = doc_map(q01(map_record(record)))
  assert rows["b.md"]["expected"] and rows["ghost.md"]["expected"]
  assert not rows["a.md"]["expected"] and not rows["z.md"]["expected"]
  assert rows["b.md"]["count"] == "2 chunks · esperado"
  assert rows["ghost.md"]["cells"] == [] and rows["ghost.md"]["empty"]
  assert rows["ghost.md"]["count"] == "0 chunks · esperado"
  assert not rows["a.md"]["empty"]


def test_document_map_tooltip_is_index_heading_path_and_160_char_preview(
  record: RunRecord,
) -> None:
  rows = doc_map(q01(map_record(record)))
  zero = rows["a.md"]["cells"][0]["title"]
  parts = zero.split(" · ")
  assert parts[0] == "#0"
  assert parts[1] == "Parte › Capítulo"
  assert parts[2] == "hola mundo " + "x" * (160 - len("hola mundo "))
  assert len(parts[2]) == 160
  # no heading path and short text: only the index and the preview
  assert rows["a.md"]["cells"][3]["title"] == "#3 · texto del chunk"


def test_document_map_tooltip_survives_odd_payloads(record: RunRecord) -> None:
  points = [
    pt("p0", "a.md", 0, text=""),
    {
      "id": "p1",
      "payload": {"source": "a.md", "chunk_index": 1, "heading_path": None},
      "vector": None,
    },
    {
      "id": "p2",
      "payload": {
        "source": "a.md",
        "chunk_index": 2,
        "heading_path": "texto",
        "text": 5,
      },
      "vector": None,
    },
    {
      "id": "p3",
      "payload": {"source": "a.md", "chunk_index": 3, "text": None},
      "vector": None,
    },
  ]
  rows = doc_map(q01(hand_record(record, points=points, chunks=[rc("p0", "a.md", 0)])))
  titles = [c["title"] for c in rows["a.md"]["cells"]]
  assert titles[0] == "#0"
  assert titles[1] == "#1"
  assert titles[2] == "#2 · 5"
  assert "None" not in titles[3], titles[3]


def test_document_map_ignores_points_without_a_usable_chunk_index(
  record: RunRecord,
) -> None:
  points = [
    pt("p0", "a.md", 0),
    {"id": "p1", "payload": {"source": "a.md", "chunk_index": "1"}, "vector": None},
    {"id": "p2", "payload": {"source": "a.md", "chunk_index": True}, "vector": None},
    {"id": "p3", "payload": {"source": "a.md", "chunk_index": None}, "vector": None},
    {"id": "p4", "payload": {"source": "", "chunk_index": 4}, "vector": None},
    {"id": "p5", "payload": {"source": 7, "chunk_index": 5}, "vector": None},
  ]
  rows = doc_map(q01(hand_record(record, points=points, chunks=[rc("p0", "a.md", 0)])))
  assert [c["index"] for c in rows["a.md"]["cells"]] == [0]


@pytest.mark.parametrize(("count", "compact"), [(400, False), (401, True)])
def test_document_map_cells_are_compact_above_400_chunks(
  record: RunRecord, count: int, compact: bool
) -> None:
  points = [pt(f"p{n}", "big.md", n) for n in range(count)] + [pt("o", "small.md", 0)]
  rows = doc_map(
    q01(
      hand_record(
        record, points=points, chunks=[rc("p0", "big.md", 0), rc("o", "small.md", 0)]
      )
    )
  )
  assert len(rows["big.md"]["cells"]) == count
  assert all(("sm" in c["classes"]) is compact for c in rows["big.md"]["cells"])
  assert all(
    "sm" not in c["classes"] for c in rows["small.md"]["cells"]
  )  # per document


def test_document_map_legend_names_every_class(record: RunRecord) -> None:
  panel = q01(map_record(record))
  legend = re.search(r'<div class="legend">(.*?)</div>', panel, re.S)
  assert legend
  text = text_of(legend.group(1))
  for word in ("recuperado", "añadido", "no usado", "descartado", "citado", "esperado"):
    assert word in text
  classes = re.findall(r'<span class="(cell[^"]*)">', legend.group(1))
  assert {
    "cell orig",
    "cell added",
    "cell unused",
    "cell orig dropped",
    "cell orig cited",
  } <= set(classes)


def test_document_map_real_run_marks_the_expected_document_and_the_cited_chunk(
  report: str, real_record: RunRecord
) -> None:
  panel = panel_of(report, "q-q01")
  rows = doc_map(panel)
  assert rows["parte-1/capitulo-01.md"]["expected"]
  item = real_record["questions"][0]
  assert item["generation"] and item["generation"]["result"]
  cited = {
    cid for c in item["generation"]["result"]["citations"] for cid in c["chunk_ids"]
  }
  by_id = {p["id"]: p["payload"] for p in real_record["points"]}
  for cid in cited:
    source, index = by_id[cid]["source"], by_id[cid]["chunk_index"]
    cell = next(c for c in rows[source]["cells"] if c["index"] == index)
    assert "cited" in cell["classes"]
  assert list(rows) == sorted(rows)


def test_document_map_budget_run_marks_the_dropped_chunks(
  budget_record: RunRecord,
) -> None:
  item = budget_record["questions"][0]
  assert item["enrich"] and item["enrich"]["context"]
  dropped = set(item["enrich"]["context"]["dropped_chunk_ids"])
  assert dropped
  by_id = {p["id"]: p["payload"] for p in budget_record["points"]}
  rows = doc_map(panel_of(render_run(budget_record), "q-q01"))
  flagged = {
    (source, c["index"])
    for source, row in rows.items()
    for c in row["cells"]
    if "dropped" in c["classes"]
  }
  assert flagged == {
    (by_id[cid]["source"], by_id[cid]["chunk_index"]) for cid in dropped
  }


def test_document_map_without_any_document(record: RunRecord) -> None:
  item = record["questions"][0]
  item["spec"]["documents"] = []
  hand = hand_record(record, points=[], chunks=[])
  assert "No hay documentos que mostrar." in q01(hand)


def test_document_map_is_built_strictly_from_points(record: RunRecord) -> None:
  """A retrieved chunk whose point is missing from the record has no cell."""
  rec = hand_record(
    record,
    points=[pt("known", "a.md", 0)],
    chunks=[rc("known", "a.md", 0), rc("missing", "a.md", 1)],
  )
  rows = doc_map(q01(rec))
  assert [c["index"] for c in rows["a.md"]["cells"]] == [0]


# ---- escaping in the phase sections


def hostile_phases(record: RunRecord) -> RunRecord:
  """Hostile text in every field that the question sections print."""
  item = record["questions"][0]
  assert item["retrieval"] and item["enrich"] and item["generation"]
  assert item["generation"]["result"] and item["judge"]
  for point in record["points"]:
    point["payload"]["text"] = HOSTILE
    point["payload"]["heading_path"] = [HOSTILE, HOSTILE]
  item["retrieval"]["queries"] = [{"text": HOSTILE, "chain": [HOSTILE]}]
  for chunk in item["retrieval"]["chunks"]:
    chunk["text"] = HOSTILE
    chunk["collection_name"] = HOSTILE
    chunk["metadata"]["source"] = HOSTILE
    chunk["metadata"]["heading_path"] = [HOSTILE]
  for number, one in enumerate(item["enrich"]["steps"]):
    one["name"] = HOSTILE + str(number)
    one["phase"] = HOSTILE
    one["reads"] = [
      {
        "ranges": [
          {"collection_name": HOSTILE, "doc_id": HOSTILE, "first": 0, "last": 1}
        ],
        "returned": [HOSTILE],
        "seconds": 0.0,
      }
    ]
    for context in (one["before"], one["after"]):
      for piece in context["pieces"]:
        piece["text"] = HOSTILE
        piece["transforms"] = [HOSTILE]
        for ref in piece["chunks"]:
          ref["chunk_id"] = HOSTILE
          ref["collection_name"] = HOSTILE
          ref["source"] = HOSTILE
          ref["origin"] = [HOSTILE]
  final = item["enrich"]["context"]
  assert final
  final["dropped_chunk_ids"] = [HOSTILE]
  for piece in final["pieces"]:
    piece["text"] = HOSTILE
    piece["transforms"] = [HOSTILE]
    for ref in piece["chunks"]:
      ref["chunk_id"] = HOSTILE
      ref["source"] = HOSTILE
  item["generation"]["llm_calls"] = [
    llm_call(text=HOSTILE, params={HOSTILE: HOSTILE}, model=HOSTILE),
    llm_call(text=None, error=HOSTILE),
  ]
  result = item["generation"]["result"]
  result["strategy"] = HOSTILE
  result["response_format"] = HOSTILE
  result["enrichments"] = [HOSTILE]
  result["citations"] = [{"number": 1, "chunk_ids": [HOSTILE]}]
  result["sources"] = [
    {"number": 1, "chunks": [{"chunk_id": HOSTILE, "source": HOSTILE}]}
  ]
  result["prompt"] = [{"role": HOSTILE, "content": HOSTILE}]
  result["dropped_chunk_ids"] = [HOSTILE]
  item["judge"]["messages"] = [{"role": HOSTILE, "content": HOSTILE}]
  return record


def test_hostile_text_in_every_phase_section_is_escaped(record: RunRecord) -> None:
  text = render_run(hostile_phases(record), token_counter=FAKE)
  page = parse(text)
  assert page.tags("img") == []
  assert len(active_scripts(page)) == 1
  assert len(page.tags("script")) == 3
  assert all(not any(name.startswith("on") for name in a) for _, a in page.elements)
  assert "<script>alert" not in text
  assert text.count(html_lib.escape(HOSTILE)) > 40
  assert len(tab_keys(text)) == 6
  panel = panel_of(text, "q-q01")
  assert phase_ids(panel)[:2] == ["in", "seed"]
  assert "llm" in phase_ids(panel)


def test_hostile_text_in_phase_sections_reaches_the_user_unchanged(
  record: RunRecord,
) -> None:
  page = parse(render_run(hostile_phases(record), token_counter=FAKE))
  assert page.body_text().count(HOSTILE) > 20


def test_hostile_map_cells_and_tooltips_cannot_break_out(record: RunRecord) -> None:
  points = [pt("p", HOSTILE, 0, [HOSTILE], HOSTILE)]
  rec = hand_record(
    record, points=points, chunks=[rc("p", HOSTILE, 0)], documents=[HOSTILE]
  )
  text = q01(rec)
  page = parse(text)
  assert page.tags("img") == []
  cells = [a for a in page.tags("span") if (a.get("class") or "").startswith("cell")]
  assert cells
  assert all(set(a) <= {"class", "title"} for a in cells)
  assert any(HOSTILE in (a.get("title") or "") for a in cells)


def test_hostile_json_block_cannot_close_the_script_or_pre(record: RunRecord) -> None:
  result = record["questions"][0]["generation"]["result"]  # type: ignore[index]
  assert result
  result["answer"] = "</pre></details></script><script>alert(1)</script>"
  result["extra"] = "</pre><script>alert(2)</script>"
  text = render_run(record, token_counter=FAKE)
  assert text.count("<script>") == 1
  assert text.count("</script>") == 3
  assert len(json_blocks(text)) == 2


def test_hostile_question_id_cannot_break_section_anchors_or_chips(
  record: RunRecord,
) -> None:
  record["questions"][0]["spec"]["id"] = '"><img src=x onerror=alert(1)>'
  text = render_run(record, token_counter=FAKE)
  page = parse(text)
  assert page.tags("img") == []
  for attrs in page.tags("a"):
    if attrs.get("class") == "chip":
      assert set(attrs) == {"class", "href"}
  for attrs in page.tags("details"):
    assert set(attrs) <= {"class", "id", "open", "data-doc"}


# ---- byte-identical output


def test_phase_sections_are_byte_identical_for_the_same_record(
  budget_record: RunRecord,
) -> None:
  first = render_run(budget_record, token_counter=FAKE)
  assert render_run(budget_record, token_counter=FAKE) == first
  again: RunRecord = json.loads(json.dumps(budget_record))
  assert render_run(again, token_counter=FAKE) == first
  assert (
    render_run(copy.deepcopy(budget_record), token_counter=FAKE).encode()
    == first.encode()
  )


def test_hand_built_record_is_byte_identical_and_not_mutated(record: RunRecord) -> None:
  rec = map_record(record)
  before = json.dumps(rec, sort_keys=True)
  first = render_run(rec, token_counter=FAKE)
  assert render_run(rec, token_counter=FAKE) == first
  assert json.dumps(rec, sort_keys=True) == before


def test_output_does_not_depend_on_the_order_the_sets_are_iterated(
  record: RunRecord,
) -> None:
  """Sets of chunk ids drive the map; PYTHONHASHSEED must not leak into the page."""
  script = (
    "import json,sys\n"
    "from hermes_report.render import render_run\n"
    "class C:\n"
    "  name='c'\n"
    "  def count(self, t): return len(t.split())\n"
    "rec=json.load(open(sys.argv[1]))\n"
    "sys.stdout.write(render_run(rec, token_counter=C()))\n"
  )
  rec = map_record(copy.deepcopy(record))
  outputs = set()
  import os
  import sys
  import tempfile

  with tempfile.TemporaryDirectory() as tmp:
    data = Path(tmp) / "r.json"
    data.write_text(json.dumps(rec), encoding="utf-8")
    for seed in ("0", "1", "12345"):
      done = subprocess.run(
        [sys.executable, "-c", script, str(data)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env={**os.environ, "PYTHONHASHSEED": seed},
      )
      outputs.add(done.stdout)
  assert len(outputs) == 1


def test_question_panel_and_default_counter_do_not_change_other_tabs(
  real_record: RunRecord,
) -> None:
  """The question tabs add content but the summary/corpus panels stay as before."""
  text = render_run(real_record, token_counter=FAKE)
  other = render_run(real_record)
  for key in ("resumen", "corpus", "chunks"):
    assert panel_of(text, key) == panel_of(other, key)


def default_counter() -> Any:
  from hermes_enrich.tokens import default_token_counter

  return default_token_counter()


# ---- remaining branches


def test_step_with_no_pieces_says_so(record: RunRecord) -> None:
  empty = cx()
  record = hand_record(record, points=[], steps=[step("n", empty, empty)], final=empty)
  panel = q01(record)
  assert "Sin piezas." in phase(panel, "e1")
  assert "Sin piezas." in phase(panel, "seed")
  assert '<span class="ph-sum">0 piezas · 0 tok</span>' in panel


def test_added_chunk_without_a_known_source_adds_no_document_row(
  record: RunRecord,
) -> None:
  ghost = rf("ghost", None, ("neighbors",))
  before, after = (
    cx(pc("a", rf("a", 0.5, source="a.md", index=0))),
    cx(pc("a", rf("a", 0.5, source="a.md", index=0)), pc("g", ghost)),
  )
  record = hand_record(
    record,
    points=[pt("a", "a.md", 0)],
    chunks=[rc("a", "a.md", 0)],
    steps=[step("n", before, after)],
    final=after,
    documents=[],
  )
  assert list(doc_map(q01(record))) == ["a.md"]


def test_write_report_with_a_lone_surrogate_raises_record_error(
  record: RunRecord, tmp_path: Path
) -> None:
  record["description"] = "mal \ud800 codificado"
  target = tmp_path / "report.html"
  with pytest.raises(RecordError) as caught:
    write_report(record, target)
  assert str(target) in str(caught.value)
  assert "UTF-8" in str(caught.value)


def test_a_surrogate_in_the_question_sections_also_gives_record_error(
  record: RunRecord, tmp_path: Path
) -> None:
  item = record["questions"][0]
  assert item["retrieval"]
  item["retrieval"]["chunks"][0]["text"] = "\ud800"
  with pytest.raises(RecordError):
    write_report(record, tmp_path / "x" / "report.html")


def test_a_failed_encoding_leaves_the_previous_report_intact(
  record: RunRecord, tmp_path: Path
) -> None:
  target = tmp_path / "report.html"
  target.write_text("previous good report", encoding="utf-8")
  record["description"] = "\ud800"
  with pytest.raises(RecordError):
    write_report(record, target)
  assert target.read_text(encoding="utf-8") == "previous good report"


# ------------------------------------------------------------------ explorer data, links, raw documents


def usage_of(item: Any) -> dict[str, set[str]]:
  """The five chunk-id sets of a question, worked out from the record on its own.

  This repeats the definitions of the task (and of the document map) without
  calling ``chunk_usage``, so a mistake in that function cannot hide itself.
  """
  found: dict[str, set[str]] = {
    key: set() for key in ("retrieved", "added", "context", "cited", "dropped")
  }
  if item["retrieval"]:
    for chunk in item["retrieval"]["chunks"]:
      found["retrieved"].add(chunk["chunk_id"])
  contexts = []
  if item["enrich"]:
    contexts = [one["after"] for one in item["enrich"]["steps"]]
    final = item["enrich"]["context"]
    if final:
      contexts.append(final)
      found["dropped"].update(final["dropped_chunk_ids"])
      for piece in final["pieces"]:
        for ref in piece["chunks"]:
          found["context"].add(ref["chunk_id"])
  for context in contexts:
    for piece in context["pieces"]:
      for ref in piece["chunks"]:
        if ref["origin"]:
          found["added"].add(ref["chunk_id"])
  result = item["generation"]["result"] if item["generation"] else None
  if result:
    found["dropped"].update(result.get("dropped_chunk_ids") or [])
    for cite in result.get("citations") or []:
      found["cited"].update(cite["chunk_ids"])
  return found


def question_ids(rec: RunRecord) -> list[str]:
  return [
    str(item["spec"].get("id") or f"q{number:02d}")
    for number, item in enumerate(rec["questions"], 1)
  ]


def assert_embedded_data_matches(rec: RunRecord) -> None:
  blocks = json_blocks(render_run(rec, token_counter=FAKE))
  assert blocks["hr-points"] == json.loads(json.dumps(rec["points"]))
  usage = blocks["hr-usage"]
  assert list(usage) == question_ids(rec)
  for qid, item in zip(question_ids(rec), rec["questions"], strict=True):
    assert list(usage[qid]) == ["retrieved", "added", "context", "cited", "dropped"]
    for key, ids in usage[qid].items():
      assert ids == sorted(set(ids)), f"{qid}.{key} is not sorted and unique"
      assert set(ids) == usage_of(item)[key], f"{qid}.{key}"


def test_embedded_points_and_usage_equal_the_record_of_the_real_run(
  real_record: RunRecord,
) -> None:
  assert real_record["points"]
  assert_embedded_data_matches(real_record)
  # in record order, not sorted by id or by source
  points = json_blocks(render_run(real_record))["hr-points"]
  assert [p["id"] for p in points] == [p["id"] for p in real_record["points"]]
  assert all(set(p) == {"id", "payload", "vector"} for p in points)


@pytest.mark.parametrize(
  "fixture", ["budget_record", "empty_record", "failed_budget_record"]
)
def test_embedded_data_equals_the_record_of_the_variant_runs(
  fixture: str, request: pytest.FixtureRequest
) -> None:
  assert_embedded_data_matches(request.getfixturevalue(fixture))


def test_budget_run_has_dropped_chunks_in_the_usage(budget_record: RunRecord) -> None:
  usage = json_blocks(render_run(budget_record))["hr-usage"]
  assert any(entry["dropped"] for entry in usage.values())
  assert any(entry["context"] for entry in usage.values())


def test_embedded_usage_of_the_hand_built_record_is_exactly_the_five_sets(
  record: RunRecord,
) -> None:
  usage = json_blocks(render_run(map_record(record), token_counter=FAKE))["hr-usage"]
  first = usage["q01"]
  assert first == {
    "retrieved": ["a1", "n0", "z0"],
    "added": ["a2", "d0"],
    "context": ["a1"],
    "cited": ["a1", "z0"],
    "dropped": ["a2", "z0"],
  }


@pytest.mark.parametrize(
  "where",
  ["retrieval", "enrich", "enrich-late", "generation", "generation-early", "judge"],
)
def test_embedded_usage_of_a_question_that_died_midway(
  record: RunRecord, where: str
) -> None:
  rec = failed_at(record, where)
  assert_embedded_data_matches(rec)
  if where == "retrieval":
    usage = json_blocks(render_run(rec, token_counter=FAKE))["hr-usage"]["q01"]
    assert usage == {
      "retrieved": [],
      "added": [],
      "context": [],
      "cited": [],
      "dropped": [],
    }


def test_the_data_travels_once_however_many_questions_there_are(
  record: RunRecord,
) -> None:
  one = render_run(with_questions(copy.deepcopy(record), 1), token_counter=FAKE)
  many = render_run(with_questions(copy.deepcopy(record), 7), token_counter=FAKE)
  for text in (one, many):
    assert text.count('id="hr-points"') == 1
    assert text.count('id="hr-usage"') == 1
  assert len(json_blocks(many)["hr-usage"]) == 7

  # the points block is the same bytes whatever the number of questions
  def block(text: str) -> str:
    return re.search(r'id="hr-points">(.*?)</script>', text, re.S)[0]  # type: ignore[index]

  assert block(one) == block(many)
  # a point text appears in the page only in the points block and in the cards the
  # server renders for it (never once per question)
  needle = record["points"][0]["payload"]["text"]
  assert needle
  assert (
    many.count(json.dumps(needle, ensure_ascii=False)[1:-1])
    <= one.count(json.dumps(needle, ensure_ascii=False)[1:-1]) * 7
  )


NASTY = (
  '</script><!-- <script>alert(1)</script> -->    "q" \\ ]]> <\\/script> </SCRIPT >'
)


def test_a_payload_with_a_closing_script_tag_cannot_leave_its_block(
  record: RunRecord,
) -> None:
  point = record["points"][0]
  point["id"] = NASTY
  point["payload"] = {
    "source": NASTY,
    "text": NASTY,
    "heading_path": [NASTY, "</script>"],
    NASTY: NASTY,
    "chunk_index": 0,
  }
  point["vector"] = {"dimension": 1, "norm": 1.0, "head": [0.5]}
  record["questions"][0]["spec"]["id"] = NASTY
  record["questions"][0]["generation"]["result"]["citations"] = [  # type: ignore[index]
    {"number": 1, "chunk_ids": [NASTY]}
  ]
  text = render_run(record, token_counter=FAKE)
  # the page has the three scripts and no stray tag opened by the data
  assert text.count("</script>") == 3
  assert text.count("<script") == 3
  assert len(active_scripts(parse(text))) == 1
  assert len(parse(text).tags("script")) == 3
  blocks = json_blocks(text)
  assert blocks["hr-points"][0] == json.loads(json.dumps(point))
  assert blocks["hr-points"][0]["payload"]["text"] == NASTY
  assert next(iter(blocks["hr-usage"])) == NASTY
  assert blocks["hr-usage"][NASTY]["cited"] == [NASTY]
  # no "<" survives inside a data block, so nothing in it can be parsed as a tag
  for body in re.findall(
    r'<script type="application/json"[^>]*>(.*?)</script>', text, re.S
  ):
    assert "<" not in body


def test_json_script_writes_every_less_than_sign_as_an_escape() -> None:
  out = h.json_script("hr-x", {"k": "a<b</script><!--", "<": ["<"]})
  assert out.startswith('<script type="application/json" id="hr-x">')
  assert out.endswith("</script>\n")
  body = out[len('<script type="application/json" id="hr-x">') : -len("</script>\n")]
  assert "<" not in body
  assert json.loads(body) == {"k": "a<b</script><!--", "<": ["<"]}
  # the id is plain text and is escaped
  assert 'id="a&quot;b&lt;"' in h.json_script('a"b<', None)
  assert json.loads(re.search(r">(.*)</script>", h.json_script("i", 0))[1]) == 0  # type: ignore[index]


def test_json_script_non_serialisable_value_raises() -> None:
  with pytest.raises(TypeError):
    h.json_script("x", {"a": object()})


def test_embedded_data_is_byte_identical_for_the_same_record(record: RunRecord) -> None:
  rec = map_record(record)
  first = render_run(copy.deepcopy(rec), token_counter=FAKE)
  assert render_run(copy.deepcopy(rec), token_counter=FAKE) == first
  assert json_blocks(first) == json_blocks(render_run(rec, token_counter=FAKE))


# ---- deep links


def anchors(fragment: str) -> list[tuple[str, str, str]]:
  """``(class, unescaped href, inner HTML)`` of every ``<a class= href=>``."""
  return [
    (cls, html_lib.unescape(href), inner)
    for cls, href, inner in re.findall(
      r'<a class="([^"]*)" href="([^"]*)">(.*?)</a>', fragment, re.S
    )
  ]


def query_of(href: str) -> dict[str, str]:
  from urllib.parse import parse_qs

  assert href.startswith("#chunks")
  _, _, query = href.partition("?")
  return {k: v[0] for k, v in parse_qs(query, keep_blank_values=True).items()}


def test_deep_link_helper_encodes_and_omits_what_is_not_given() -> None:
  assert h.deep_link() == "#chunks"
  assert h.deep_link(doc="a.md") == "#chunks?doc=a.md"
  assert h.deep_link(question="q01", used=True) == "#chunks?q=q01&used=1"
  assert (
    h.deep_link(doc="d/é & x.md", question="q 1", chunk="c#1?")
    == "#chunks?doc=d%2F%C3%A9%20%26%20x.md&q=q%201&chunk=c%231%3F"
  )
  # empty or None values are left out, never written as "doc="
  assert h.deep_link(doc="", question=None, chunk="") == "#chunks"


def test_link_helper_escapes_the_href_and_keeps_the_inner_html() -> None:
  out = h.link('#a"b&c', "<code>x</code>")
  assert out == '<a class="chunk-link" href="#a&quot;b&amp;c"><code>x</code></a>'
  assert h.link("#x", "i", 'we"ird').startswith('<a class="we&quot;ird"')


def test_map_cells_link_to_their_chunk_in_the_explorer(record: RunRecord) -> None:
  panel = q01(map_record(record))
  start = panel.index("Mapa de documentos")
  card_html = panel[start : panel.index('<div class="card rawdocs"', start)]
  cells = anchors(card_html)
  assert cells and all(cls == "cl" for cls, _, _ in cells)
  # one anchor around every cell, none around a cell that has no point
  assert len(cells) == len(re.findall(r'<span class="cell[^"]*" title=', card_html))
  by_chunk = {query_of(href)["chunk"]: (href, inner) for _, href, inner in cells}
  assert set(by_chunk) == {"z2", "z0", "a3", "a0", "a2", "a1", "a4", "b0", "b1", "d0"}
  href, inner = by_chunk["a1"]
  assert query_of(href) == {"doc": "a.md", "q": "q01", "chunk": "a1"}
  assert re.fullmatch(r'<span class="cell orig[^"]*" title="[^"]*">1</span>', inner)
  # the anchor is outside the span, so the map still reads as before
  assert len(doc_map(panel)["a.md"]["cells"]) == 5


def test_chunk_labels_in_every_section_link_to_the_explorer(record: RunRecord) -> None:
  panel = q01(map_record(record))
  for suffix in ("in", "out"):
    for _, href, _ in anchors(phase(panel, suffix)):
      assert query_of(href)["q"] == "q01"
  # entrada: the three retrieved chunks, labels unchanged
  entrada = anchors(phase(panel, "in"))
  assert [(query_of(h_)["chunk"], inner) for _, h_, inner in entrada] == [
    ("a1", "<code>a.md#1</code>"),
    ("n0", "<code>n0</code>"),
    ("z0", "<code>z.md#0</code>"),
  ]
  # a chunk that has no source still opens the explorer, without a doc filter
  assert query_of(entrada[1][1]) == {"q": "q01", "chunk": "n0"}
  # salida: the citation labels
  salida = [
    (query_of(h_)["chunk"], inner) for _, h_, inner in anchors(phase(panel, "out"))
  ]
  assert salida == [("a1", "<code>a.md#1</code>"), ("z0", "<code>z.md#0</code>")]


def test_step_pills_and_reads_are_links_too(record: RunRecord) -> None:
  rec = map_record(record)
  item = rec["questions"][0]
  item["enrich"]["steps"][0]["reads"] = [  # type: ignore[index]
    {
      "ranges": [{"collection_name": "c", "doc_id": "d", "first": 1, "last": 2}],
      "returned": ["a2", "a1"],
      "seconds": 0.0,
    }
  ]
  panel = q01(rec)
  step_id = next(i for i in phase_ids(panel) if i not in {"in", "seed", "out", "llm"})
  body = phase(panel, step_id)
  targets = {
    query_of(href)["chunk"] for cls, href, _ in anchors(body) if "chunk-link" in cls
  }
  assert {"a1", "a2"} <= targets
  pills = [
    inner for cls, _, inner in anchors(body) if "chunk-link" in cls and "pill" in inner
  ]
  assert pills and all(inner.startswith('<span class="pill') for inner in pills)


def test_a_chunk_with_no_point_stays_plain_text(record: RunRecord) -> None:
  rec = map_record(record)
  rec["questions"][0]["generation"]["result"]["citations"] = [  # type: ignore[index]
    {"number": 1, "chunk_ids": ["a1", "nobody"]}
  ]
  body = phase(q01(rec), "out")
  assert "<code>nobody</code>" in body
  assert "chunk=nobody" not in body
  assert "chunk=a1" in body


def test_links_of_the_real_run_point_at_real_points_and_the_right_question(
  real_record: RunRecord,
) -> None:
  text = render_run(real_record)
  ids = {p["id"]: p["payload"].get("source") for p in real_record["points"]}
  found = 0
  for qid in question_ids(real_record):
    for _, href, _ in anchors(panel_of(text, f"q-{qid}")):
      if not href.startswith("#chunks") or "chunk=" not in href:
        continue
      query = query_of(href)
      found += 1
      assert query["q"] == qid
      assert query["chunk"] in ids
      assert query.get("doc") == ids[query["chunk"]]
  assert found > 10


def test_point_ids_are_the_retrieval_chunk_ids_in_the_real_run(
  real_record: RunRecord,
) -> None:
  """The deep link uses the point id as the chunk id: this is what makes it valid."""
  point_ids = {p["id"] for p in real_record["points"]}
  assert len(point_ids) == len(real_record["points"])
  touched = set()
  for item in real_record["questions"]:
    for ids in usage_of(item).values():
      touched |= ids
  assert touched
  assert touched <= point_ids
  # and the ids a question retrieved are exactly those the report links to
  text = render_run(real_record)
  for qid, item in zip(
    question_ids(real_record), real_record["questions"], strict=True
  ):
    linked = {
      query_of(h_)["chunk"]
      for _, h_, _ in anchors(panel_of(text, f"q-{qid}"))
      if h_.startswith("#chunks") and "chunk=" in h_
    }
    assert usage_of(item)["retrieved"] <= linked


def test_hostile_source_and_ids_cannot_break_out_of_the_href(record: RunRecord) -> None:
  points = [pt('c"><img src=x onerror=alert(1)>', HOSTILE, 0, [HOSTILE], HOSTILE)]
  rec = hand_record(
    record,
    points=points,
    chunks=[rc(points[0]["id"], HOSTILE, 0)],
    documents=[HOSTILE],
  )
  text = q01(rec)
  page = parse(text)
  assert page.tags("img") == []
  hrefs = [a["href"] or "" for a in page.tags("a") if "href" in a]
  deep = [href for href in hrefs if href.startswith("#chunks") and "chunk=" in href]
  assert deep
  for href in deep:
    assert re.fullmatch(r"#chunks\?[A-Za-z0-9._~%&=-]*", href), href
    query = query_of(href)
    assert query["doc"] == HOSTILE
    assert query["chunk"] == points[0]["id"]


# ---- raw documents section


def raw_section(panel: str) -> str:
  start = panel.index('<div class="card rawdocs"')
  end = panel.find('<details class="phase"', start)
  return panel[start : end if end != -1 else len(panel)]


def raw_docs(panel: str) -> list[str]:
  return [
    html_lib.unescape(d)
    for d in re.findall(r'<details class="rawdoc" data-doc="([^"]*)"', panel)
  ]


def test_every_question_tab_has_one_raw_section_with_its_documents(
  report: str, real_record: RunRecord
) -> None:
  with_docs = 0
  for qid in question_ids(real_record):
    panel = panel_of(report, f"q-{qid}")
    # the same documents as the map, in the same order; none, no section
    assert raw_docs(panel) == list(doc_map(panel))
    if not raw_docs(panel):
      assert "rawdocs" not in panel
      continue
    with_docs += 1
    assert panel.count(f'id="raw-{qid}"') == 1
    assert f'data-q="{qid}"' in panel
  assert with_docs >= 2


def test_raw_section_lists_the_documents_of_the_map(record: RunRecord) -> None:
  panel = q01(map_record(record))
  assert (
    raw_docs(panel)
    == list(doc_map(panel))
    == ["a.md", "b.md", "d.md", "ghost.md", "z.md"]
  )
  # a document of the question that the collection does not have is still listed
  assert "ghost.md" in raw_docs(panel)


def test_raw_document_summary_uses_the_singular_for_one_chunk(
  record: RunRecord,
) -> None:
  panel = q01(map_record(record))
  summaries = dict(
    re.findall(
      r'<details class="rawdoc" data-doc="([^"]*)"><summary>.*?'
      r'<span class="muted">([^<]*)</span>',
      panel,
    )
  )
  points: dict[str, int] = {}
  for point in record["points"]:
    source = point["payload"].get("source")
    if source is not None and point["payload"].get("chunk_index") is not None:
      points[source] = points.get(source, 0) + 1
  assert set(summaries) == {"a.md", "b.md", "d.md", "ghost.md", "z.md"}
  assert "1 chunk" in summaries.values()  # the case under test is present
  assert not any(v.startswith("1 chunks") for v in summaries.values())
  for source, text in summaries.items():
    count = points.get(source, 0)
    assert text == ("1 chunk" if count == 1 else f"{count} chunks"), source
  assert summaries["ghost.md"] == "0 chunks"


def test_raw_section_sits_between_the_map_and_the_phases_and_is_collapsed(
  record: RunRecord,
) -> None:
  panel = q01(map_record(record))
  assert panel.index("Mapa de documentos") < panel.index('id="raw-q01"')
  assert panel.index('id="raw-q01"') < panel.index('<details class="phase"')
  section = raw_section(panel)
  assert "Documentos tocados (crudo)" in section
  for tag in re.findall(r'<details class="rawdoc"[^>]*>', section):
    assert "open" not in tag
  # lazy: the chunks are not in the page once per document, the script draws them
  assert "chunk-card" not in section
  assert "texto del chunk" not in section
  assert section.count('<div class="rawbody"></div>') == 5
  # the phases are still where they were (the section is not a phase)
  assert phase_ids(panel)[:2] == ["in", "seed"]


def test_raw_section_links_to_the_explorer_filtered_by_the_question(
  record: RunRecord,
) -> None:
  section = raw_section(q01(map_record(record)))
  links = [a for a in anchors(section) if a[0] == "raw-link"]
  assert len(links) == 1
  assert query_of(links[0][1]) == {"q": "q01", "used": "1"}


def test_raw_section_and_map_agree_for_each_run_variant(
  budget_record: RunRecord, empty_record: RunRecord
) -> None:
  for rec in (budget_record, empty_record):
    text = render_run(rec)
    for qid in question_ids(rec):
      panel = panel_of(text, f"q-{qid}")
      if "No hay documentos que mostrar." in panel:
        continue
      assert raw_docs(panel) == list(doc_map(panel))


def test_no_documents_means_no_raw_section(record: RunRecord) -> None:
  rec = hand_record(record, points=[], documents=[])
  panel = q01(rec)
  assert "No hay documentos que mostrar." in panel
  assert "rawdocs" not in panel
  assert 'id="raw-q01"' not in panel


def test_raw_section_of_a_question_that_died_in_retrieval(
  record: RunRecord,
) -> None:
  rec = failed_at(record, "retrieval")
  rec["questions"][0]["spec"]["documents"] = ["x.md"]
  panel = q01(rec)
  # the expected document is still listed, and the section still has its link
  assert raw_docs(panel) == list(doc_map(panel)) == ["x.md"]
  assert 'id="raw-q01"' in panel
  rec["questions"][0]["spec"]["documents"] = []
  assert "rawdocs" not in q01(rec)


def test_hostile_document_names_and_question_ids_stay_inside_their_attributes(
  record: RunRecord,
) -> None:
  rec = hand_record(
    record,
    points=[pt("p", HOSTILE, 0)],
    chunks=[rc("p", HOSTILE, 0)],
    documents=[HOSTILE],
  )
  rec["questions"][0]["spec"]["id"] = '"><img src=x onerror=alert(1)>'
  text = render_run(rec, token_counter=FAKE)
  page = parse(text)
  assert page.tags("img") == []
  details = [a for a in page.tags("details") if "data-doc" in a]
  assert details[0]["data-doc"] == HOSTILE  # the first question's first document
  assert set(details[0]) == {"class", "data-doc"}
  holders = [a for a in page.tags("div") if "data-q" in a]
  assert holders[0]["data-q"] == '"><img src=x onerror=alert(1)>'
  assert set(holders[0]) == {"class", "id", "data-q"}


def test_the_noscript_notices_and_styles_are_present(report: str) -> None:
  assert (
    "<noscript><style>.panel { display: block; } nav { display: none; }</style></noscript>"
    in report
  )
  assert report.index("<noscript><style>") < report.index("</head>")
  assert "Se necesita JavaScript para ver los chunks." in report
  # without script every panel is shown: the rule has to follow the base stylesheet
  assert report.index("<style>\n") < report.index("<noscript><style>")


# ---- carry-over from 3.3.2 and the atomic write


def test_zero_token_counts_next_to_missing_ones(record: RunRecord) -> None:
  generation = record["questions"][0]["generation"]
  assert generation
  generation["llm_calls"] = [llm_call(prompt_tokens=0, completion_tokens=None)]
  assert "prompt 0 · respuesta —" in phase(q01(record), "llm")
  generation["llm_calls"] = [llm_call(prompt_tokens=None, completion_tokens=0)]
  assert "prompt — · respuesta 0" in phase(q01(record), "llm")
  generation["llm_calls"] = [llm_call(prompt_tokens=12, completion_tokens=7)]
  assert "prompt 12 · respuesta 7" in phase(q01(record), "llm")


def test_map_tooltip_never_prints_none_for_any_odd_payload_value(
  record: RunRecord,
) -> None:
  texts: list[Any] = [None, "", "   ", 0, False, [], {}, [None]]
  points = [
    {
      "id": f"p{n}",
      "payload": {"source": "a.md", "chunk_index": n, "text": t},
      "vector": None,
    }
    for n, t in enumerate(texts)
  ]
  rows = doc_map(q01(hand_record(record, points=points, chunks=[rc("p0", "a.md", 0)])))
  titles = [c["title"] for c in rows["a.md"]["cells"]]
  assert len(titles) == len(texts)
  # None is dropped; a falsy non-None value is data, so showing "False" or "0" is fine
  assert "None" not in titles[0]
  assert titles[0] == "#0"


def test_write_report_leaves_no_temporary_file_and_a_normal_mode(
  real_record: RunRecord, tmp_path: Path
) -> None:
  import os

  target = tmp_path / "out" / "report.html"
  write_report(real_record, target)
  assert [p.name for p in target.parent.iterdir()] == ["report.html"]
  assert target.read_text(encoding="utf-8") == render_run(real_record)
  umask = os.umask(0)
  os.umask(umask)
  assert target.stat().st_mode & 0o777 == 0o666 & ~umask


def test_write_report_twice_gives_the_same_bytes(
  real_record: RunRecord, tmp_path: Path
) -> None:
  first, second = tmp_path / "a.html", tmp_path / "b.html"
  write_report(real_record, first)
  write_report(real_record, second)
  assert first.read_bytes() == second.read_bytes() == render_run(real_record).encode()


def test_a_failed_encoding_leaves_no_temporary_file_behind(
  record: RunRecord, tmp_path: Path
) -> None:
  target = tmp_path / "report.html"
  target.write_text("previous good report", encoding="utf-8")
  record["description"] = "\ud800"
  with pytest.raises(RecordError):
    write_report(record, target)
  assert [p.name for p in tmp_path.iterdir()] == ["report.html"]


def test_a_failed_move_keeps_the_previous_report_and_names_the_file(
  real_record: RunRecord, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  import os

  target = tmp_path / "report.html"
  target.write_text("previous good report", encoding="utf-8")

  def broken(*args: Any, **kwargs: Any) -> None:
    raise PermissionError("denied")

  monkeypatch.setattr(os, "replace", broken)
  with pytest.raises(RecordError) as caught:
    write_report(real_record, target)
  assert str(target) in str(caught.value)
  assert target.read_text(encoding="utf-8") == "previous good report"
  assert [p.name for p in tmp_path.iterdir()] == ["report.html"]


def test_the_report_is_not_visible_half_written(
  real_record: RunRecord, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """While the temporary file is being moved, the old report is still whole."""
  import os

  target = tmp_path / "report.html"
  target.write_text("previous good report", encoding="utf-8")
  original = os.replace
  seen: list[str] = []

  def spy(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
    seen.append(target.read_text(encoding="utf-8"))
    assert Path(src).read_text(encoding="utf-8") == render_run(real_record)
    assert Path(src).parent == target.parent  # same folder: the move is atomic
    original(src, dst, *args, **kwargs)

  monkeypatch.setattr(os, "replace", spy)
  write_report(real_record, target)
  assert seen == ["previous good report"]
  assert target.read_text(encoding="utf-8") == render_run(real_record)

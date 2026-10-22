"""Tests for ``hermes_report.render.compare``: the side-by-side comparison page.

The runs are real offline runs over ``tests/data`` (hash embedder, echo LLM and
judge) with different flags: the default, ``--top-k 1``, ``--budget 30`` (which
makes q01 fail with an error) and ``--only``. The echo LLM gives the same rule
verdicts for every retrieval setting, so a row with differing verdicts comes
either from the budget run or from an edited deep copy of a record. The paths
inside ``config`` differ because each run lives in its own folder; the fixtures
make them equal so that only the flag under test shows up as a difference.
"""

from __future__ import annotations

import copy
import html as html_lib
import json
import re
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from conftest import DATA

from hermes_report import runner
from hermes_report.config import Overrides, load_config
from hermes_report.record import RecordError, RunRecord
from hermes_report.render import render_compare, render_run, write_compare
from hermes_report.render.compare import render_compare as render_compare_direct
from hermes_report.render.score import score_tiles

HOSTILE = '<script>alert("x")</script><img src=x onerror=alert(1)>&amp;"\''
PATH_KEYS = (("corpus", "dir"), ("questions", "file"), ("output", "dir"))

Runs = list[tuple[str, RunRecord]]


# ------------------------------------------------------------------ helpers


def _make_run(root: Path, name: str, **overrides: Any) -> RunRecord:
  project = root / name
  shutil.copytree(DATA, project)
  config = load_config(
    project / "run.toml", Overrides(out=project / "out", **overrides)
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    return runner.run(config, progress=lambda line: None).record


def _same_paths(record: RunRecord, reference: RunRecord) -> RunRecord:
  """``record`` with the folder-dependent config values of ``reference``."""
  for section, key in PATH_KEYS:
    record["config"][section][key] = reference["config"][section][key]
  record["config"]["output"]["cache_dir"] = reference["config"]["output"]["cache_dir"]
  return record


@pytest.fixture(scope="module")
def runs_by_name(tmp_path_factory: pytest.TempPathFactory) -> dict[str, RunRecord]:
  root = tmp_path_factory.mktemp("compare")
  a = _make_run(root, "a")
  found = {
    "a": a,
    "b": _same_paths(_make_run(root, "b", top_k=1), a),
    "c": _same_paths(_make_run(root, "c", budget=30), a),
    "only11": _same_paths(_make_run(root, "d", only=("q11",)), a),
    "only01": _same_paths(_make_run(root, "e", only=("q01",)), a),
  }
  return found


@pytest.fixture
def rec(runs_by_name: dict[str, RunRecord]) -> Callable[[str], RunRecord]:
  """``rec("b")``: a private deep copy of one of the reference runs."""

  def get(name: str) -> RunRecord:
    return copy.deepcopy(runs_by_name[name])

  return get


def links(*names: str) -> list[str]:
  return [f"{name}/report.html" for name in names]


def build(
  rec: Callable[[str], RunRecord], *names: str, hrefs: list[str] | None = None
) -> Runs:
  return list(zip(hrefs or links(*names), [rec(n) for n in names], strict=True))


def tag(text: str, name: str) -> list[str]:
  return re.findall(rf"<{name}[ >].*?</{name}>", text, re.S)


def card_of(text: str, title: str) -> str:
  """The HTML of the card titled ``title``."""
  match = re.search(
    rf'<div class="card"><h3>{re.escape(title)}</h3>(.*?)</div>(?=<div class="card">|</section>)',
    text,
    re.S,
  )
  assert match, f"no card {title}"
  return match.group(1)


def table_rows(html: str) -> list[list[str]]:
  """The cells of every body row of the first table in ``html``, tags stripped."""
  rows = []
  for row in re.findall(r"<tr[^>]*>(?!<th).*?</tr>", html, re.S):
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
    rows.append([html_lib.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells])
  return rows


def header_of(html: str) -> list[str]:
  head = re.search(r"<tr>(<th.*?)</tr>", html, re.S)
  assert head
  return [
    html_lib.unescape(re.sub(r"<[^>]+>", "", c))
    for c in re.findall(r"<th[^>]*>(.*?)</th>", head.group(1), re.S)
  ]


def matrix_rows(text: str) -> list[str]:
  """The raw ``<tr ...>`` HTML of the body rows of the matrix."""
  panel = re.search(r'id="tab-matriz"[^>]*>(.*?)</section>', text, re.S)
  assert panel
  return re.findall(r"<tr(?: [^>]*)?>(?!<th).*?</tr>", panel.group(1), re.S)


def row_for(text: str, qid: str) -> str:
  rows = [r for r in matrix_rows(text) if f'data-q="{html_lib.escape(qid)}"' in r]
  assert len(rows) == 1, qid
  return rows[0]


def diff_ids(text: str) -> list[str]:
  return [
    html_lib.unescape(m)
    for r in matrix_rows(text)
    if 'class="diff"' in r.split(">")[0]
    for m in re.findall(r'data-q="([^"]*)"', r)[:1]
  ]


def set_rule(record: RunRecord, qid: str, verdict: str | None) -> None:
  for item in record["questions"]:
    if item["spec"]["id"] == qid:
      if verdict is None:
        item["rules"] = None
      else:
        assert item["rules"] is not None
        item["rules"]["verdict"] = verdict
      return
  raise AssertionError(qid)


def set_judge(record: RunRecord, qid: str, verdict: str) -> None:
  for item in record["questions"]:
    if item["spec"]["id"] == qid:
      judge = item["judge"]
      assert judge is not None
      judge["verdict"] = verdict
      return
  raise AssertionError(qid)


def differences(text: str) -> list[list[str]]:
  return table_rows(card_of(text, "Diferencias de configuración"))


# ------------------------------------------------------------------ fixtures sanity


def test_reference_runs_have_what_the_tests_rely_on(
  runs_by_name: dict[str, RunRecord],
) -> None:
  def verdicts(name: str) -> dict[str, str | None]:
    return {
      q["spec"]["id"]: q["rules"]["verdict"] if q["rules"] else None
      for q in runs_by_name[name]["questions"]
    }

  assert (
    verdicts("a")
    == verdicts("b")
    == {
      "q01": "correct",
      "q02": "unrated",
      "q11": "correct_refusal",
    }
  )
  assert verdicts("c")["q01"] == "error"
  assert verdicts("only11") == {"q11": "correct_refusal"}
  assert runs_by_name["a"]["config"]["retrieval"]["top_k"] == 5
  assert runs_by_name["b"]["config"]["retrieval"]["top_k"] == 1
  assert runs_by_name["c"]["config"]["enrich"]["max_context_tokens"] == 30


# ------------------------------------------------------------------ the page


def test_render_compare_is_exported_by_the_package() -> None:
  assert render_compare is render_compare_direct


def test_page_is_a_document_with_the_two_tabs(rec: Callable[[str], RunRecord]) -> None:
  text = render_compare(build(rec, "a", "b", "c"))
  assert text.startswith("<!doctype html>")
  assert text.rstrip().endswith("</html>")
  assert re.findall(r'<button class="tab" data-key="([^"]+)"', text) == [
    "resumen",
    "matriz",
  ]
  assert 'id="tab-resumen"' in text and 'id="tab-matriz"' in text
  assert "<title>Hermes Report · Comparación de 3 ejecuciones</title>" in text
  assert "<span>· Comparación de 3 ejecuciones</span></h1>" in text


def test_page_works_offline(rec: Callable[[str], RunRecord]) -> None:
  text = render_compare(build(rec, "a", "b"))
  assert re.search(r"(?:src|href)=[\"']https?://", text) is None
  assert "url(http" not in text
  assert "<link" not in text


@pytest.mark.parametrize(("count", "word"), [(2, "ejecuciones"), (1, "ejecución")])
def test_header_counts_the_runs_and_uses_the_singular_for_one(
  rec: Callable[[str], RunRecord], count: int, word: str
) -> None:
  names = ["a", "b"][:count]
  text = render_compare(build(rec, *names))
  assert f"Comparación de {count} {word}" in text


def test_empty_input_is_a_value_error() -> None:
  with pytest.raises(ValueError, match="at least one run"):
    render_compare([])


def test_generation_time_is_the_latest_finished_at_in_any_order(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b", "c")
  runs[0][1]["finished_at"] = "2030-01-01T00:00:00+00:00"
  runs[1][1]["finished_at"] = "2031-06-30T12:00:00+00:00"
  runs[2][1]["finished_at"] = "2029-12-31T23:59:59+00:00"
  for ordering in (runs, runs[::-1], [runs[1], runs[2], runs[0]]):
    text = render_compare(ordering)
    assert "Generada el 2031-06-30T12:00:00+00:00" in text
    assert "2030-01-01T00:00:00+00:00" not in text.split("Ejecuciones")[0]


def test_generation_time_is_not_the_clock(
  rec: Callable[[str], RunRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
  import time

  runs = build(rec, "a", "b")
  first = render_compare(runs)
  monkeypatch.setattr(time, "time", lambda: 4_000_000_000.0)
  assert render_compare(runs) == first


# ------------------------------------------------------------------ Ejecuciones


def test_runs_table_has_one_column_per_run_with_title_start_and_link(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(
    rec, "a", "b", "c", hrefs=["../x/report.html", "y/report.html", "z.html"]
  )
  for index, label in enumerate("ABC"):
    runs[index][1]["title"] = f"Título {label}"
  text = render_compare(runs)
  body = card_of(text, "Ejecuciones")
  assert header_of(body) == ["", "A", "B", "C"]
  rows = table_rows(body)
  assert rows[0] == ["Título", "Título A", "Título B", "Título C"]
  assert rows[1] == ["Inicio", *[r[1]["started_at"] for r in runs]]
  assert re.findall(r'<a class="raw-link" href="([^"]*)"', body) == [
    "../x/report.html",
    "y/report.html",
    "z.html",
  ]


def test_labels_go_past_z_like_spreadsheet_columns(
  rec: Callable[[str], RunRecord],
) -> None:
  runs: Runs = [(f"r{n}/report.html", rec("a")) for n in range(28)]
  text = render_compare(runs)
  labels = header_of(card_of(text, "Ejecuciones"))[1:]
  assert labels[:3] == ["A", "B", "C"]
  assert labels[24:] == ["Y", "Z", "AA", "AB"]
  assert len(set(labels)) == 28


# ------------------------------------------------------------------ config differences


def test_top_k_is_the_only_difference_between_the_default_and_the_top_k_run(
  rec: Callable[[str], RunRecord],
) -> None:
  rows = differences(render_compare(build(rec, "a", "b")))
  assert rows == [["retrieval.top_k", "5", "1"]]


def test_budget_shows_as_enrich_max_context_tokens_with_three_runs(
  rec: Callable[[str], RunRecord],
) -> None:
  rows = differences(render_compare(build(rec, "a", "b", "c")))
  assert rows == [
    ["retrieval.top_k", "5", "1", "5"],
    ["enrich.max_context_tokens", "3000", "3000", "30"],
  ]


def test_keys_equal_in_every_run_are_not_listed(
  rec: Callable[[str], RunRecord],
) -> None:
  keys = {row[0] for row in differences(render_compare(build(rec, "a", "b", "c")))}
  assert "retrieval.candidate_factor" not in keys
  assert "retrieval.score_threshold" not in keys
  assert "title" not in keys
  assert not any(key.startswith("corpus.") for key in keys)


def test_identical_runs_say_so_instead_of_showing_an_empty_table(
  rec: Callable[[str], RunRecord],
) -> None:
  text = render_compare(build(rec, "a", "a"))
  body = card_of(text, "Diferencias de configuración")
  assert "<table" not in body
  assert "no se diferencian" in body


def test_a_run_compared_with_its_own_copy_has_no_differences_at_all(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "b", "b", "b")
  assert differences(render_compare(runs)) == []


def test_lists_are_compared_and_shown_as_json_text(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a", "a")
  runs[1][1]["config"]["enrich"]["chain"] = ["neighbors", "merge"]
  runs[2][1]["config"]["enrich"]["chain"] = []
  rows = differences(render_compare(runs))
  assert rows == [
    [
      "enrich.chain",
      '["neighbors", "merge", "doc_order"]',
      '["neighbors", "merge"]',
      "[]",
    ]
  ]


def test_non_string_scalars_are_shown_as_json(rec: Callable[[str], RunRecord]) -> None:
  runs = build(rec, "a", "a")
  runs[1][1]["config"]["judge"]["enabled"] = False
  runs[1][1]["config"]["generation"]["temperature"] = 0.7
  runs[1][1]["config"]["qdrant"]["collection"] = None
  rows = {row[0]: row[1:] for row in differences(render_compare(runs))}
  assert rows["judge.enabled"] == ["true", "false"]
  assert rows["generation.temperature"] == ["0.0", "0.7"]
  assert rows["qdrant.collection"] == ["", "null"]


def test_a_key_missing_from_one_run_shows_a_dash_and_counts_as_a_difference(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a", "a")
  del runs[1][1]["config"]["retrieval"]["score_threshold"]
  runs[2][1]["config"]["extra"] = {"deep": {"er": 1}}
  rows = {row[0]: row[1:] for row in differences(render_compare(runs))}
  assert rows["retrieval.score_threshold"] == ["0.3", "—", "0.3"]
  assert rows["extra.deep.er"] == ["—", "—", "1"]


def test_a_value_present_in_all_runs_but_missing_in_none_is_not_confused_with_empty(
  rec: Callable[[str], RunRecord],
) -> None:
  """An empty string and a missing key are two different values."""
  runs = build(rec, "a", "a")
  runs[0][1]["config"]["description"] = ""
  del runs[1][1]["config"]["description"]
  rows = {row[0]: row[1:] for row in differences(render_compare(runs))}
  assert rows["description"] == ["", "—"]


def test_a_whole_section_missing_from_one_run_lists_each_of_its_keys(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a")
  del runs[1][1]["config"]["judge"]
  rows = {row[0]: row[1:] for row in differences(render_compare(runs))}
  assert rows["judge.enabled"] == ["true", "—"]
  assert rows["judge.provider"] == ["echo", "—"]


def test_a_config_that_is_empty_in_every_run_is_not_an_error(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a")
  for _, record in runs:
    record["config"] = {}
  text = render_compare(runs)
  assert "no se diferencian" in card_of(text, "Diferencias de configuración")


def test_differing_package_versions_are_rows_after_the_config_rows(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b")
  runs[1][1]["environment"]["packages"]["hermes-core"] = "9.9.9"
  runs[1][1]["environment"]["packages"]["brand-new"] = "1.0"
  rows = differences(render_compare(runs))
  assert rows == [
    ["retrieval.top_k", "5", "1"],
    ["environment.packages.hermes-core", "0.6.0", "9.9.9"],
    ["environment.packages.brand-new", "—", "1.0"],
  ]


def test_equal_package_versions_are_not_listed(
  rec: Callable[[str], RunRecord],
) -> None:
  rows = differences(render_compare(build(rec, "a", "b")))
  assert not [r for r in rows if r[0].startswith("environment.")]


def test_only_packages_differ_and_nothing_else(rec: Callable[[str], RunRecord]) -> None:
  runs = build(rec, "a", "a")
  runs[1][1]["environment"]["packages"]["ollama"] = "0.0.1"
  runs[1][1]["environment"]["python"] = "2.7"  # only package versions are compared
  rows = differences(render_compare(runs))
  assert [r[0] for r in rows] == ["environment.packages.ollama"]


def test_an_environment_without_packages_is_tolerated(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a")
  cast(dict[str, Any], runs[0][1]["environment"]).pop("packages")
  rows = differences(render_compare(runs))
  assert rows
  assert all(r[0].startswith("environment.packages") for r in rows)
  by_key = {r[0]: r[1:] for r in rows}
  assert by_key["environment.packages.hermes-core"] == ["—", "0.6.0"]


def test_config_keys_and_values_are_escaped(rec: Callable[[str], RunRecord]) -> None:
  runs = build(rec, "a", "a")
  runs[0][1]["config"][HOSTILE] = HOSTILE
  runs[1][1]["config"][HOSTILE] = "other " + HOSTILE
  text = render_compare(runs)
  assert HOSTILE not in text
  assert "<img src=x" not in text
  rows = {row[0]: row[1:] for row in differences(text)}
  assert rows[HOSTILE] == [HOSTILE, "other " + HOSTILE]


def test_a_value_that_looks_like_html_is_shown_inside_code(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a")
  runs[1][1]["config"]["title"] = "<b>x</b>"
  body = card_of(render_compare(runs), "Diferencias de configuración")
  assert "&lt;b&gt;x&lt;/b&gt;" in body
  assert "<b>x</b>" not in body


# ------------------------------------------------------------------ Puntuación


def run_tiles(record: RunRecord) -> dict[str, str]:
  found = re.findall(
    r'<div class="tile"><span class="k">(.*?)</span><span class="big">(.*?)</span>',
    render_run(record),
  )
  return {html_lib.unescape(k): html_lib.unescape(v) for k, v in found}


def test_the_score_table_matches_the_tiles_of_each_run_report(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b", "c")
  body = card_of(render_compare(runs), "Puntuación")
  assert header_of(body) == ["", "A", "B", "C"]
  rows = table_rows(body)
  assert rows
  for index, (_, record) in enumerate(runs, 1):
    expected = run_tiles(record)
    assert {row[0]: row[index] for row in rows} == expected


def test_score_values_for_the_reference_runs(rec: Callable[[str], RunRecord]) -> None:
  body = card_of(render_compare(build(rec, "a", "c")), "Puntuación")
  rows = {row[0]: row[1:] for row in table_rows(body)}
  assert rows["Correctas por reglas"] == ["2/2", "1/2"]
  assert rows["Correctas por juez"] == ["0/3", "0/2"]  # the echo judge says partial
  assert rows["Fallos de generación"] == ["0", "0"]
  assert rows["Tiempo medio por pregunta"][0].endswith(" s")


def test_score_uses_the_same_function_as_the_run_report(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "c")
  rows = table_rows(card_of(render_compare(runs), "Puntuación"))
  assert [row[0] for row in rows] == [label for label, _ in score_tiles([])]


def test_score_shows_dashes_for_a_run_with_no_questions(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b")
  runs[1][1]["questions"] = []
  rows = {
    row[0]: row[1:] for row in table_rows(card_of(render_compare(runs), "Puntuación"))
  }
  assert rows["Correctas por reglas"] == ["2/2", "—"]
  assert rows["Correctas por juez"] == ["0/3", "—"]
  assert rows["Tiempo medio por pregunta"][1] == "0.0 s"


def test_score_counts_disagreements_and_failures_per_run(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a")
  set_judge(runs[1][1], "q01", "incorrect")  # the rules say correct
  set_rule(runs[1][1], "q11", "retrieval_failure")
  set_rule(runs[1][1], "q02", "budget_failure")
  rows = {
    row[0]: row[1:] for row in table_rows(card_of(render_compare(runs), "Puntuación"))
  }
  assert rows["Reglas y juez discrepan"] == ["0", "1"]
  assert rows["Fallos de recuperación"] == ["0", "1"]
  assert rows["Fallos de presupuesto"] == ["0", "1"]
  assert rows["Correctas por reglas"] == ["2/2", "1/3"]


# ------------------------------------------------------------------ matrix: structure


def test_matrix_has_one_row_per_question_and_one_column_per_run(
  rec: Callable[[str], RunRecord],
) -> None:
  text = render_compare(build(rec, "a", "b", "c"))
  rows = matrix_rows(text)
  assert [re.findall(r'data-q="([^"]*)"', r)[0] for r in rows] == ["q01", "q02", "q11"]
  panel = re.search(r'id="tab-matriz"[^>]*>(.*?)</section>', text, re.S)
  assert panel
  assert header_of(panel.group(1)) == ["id", "pregunta", "A", "B", "C"]
  for row in rows:
    assert len(re.findall(r"<td>", row)) == 5


def test_matrix_row_shows_id_question_pills_and_seconds(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "c")
  runs[0][1]["questions"][0]["seconds"] = 1.26
  text = render_compare(runs)
  row = row_for(text, "q01")
  assert "<b>q01</b>" in row
  assert "¿Cuántos días tardó en ponerle nombre a su rocín?" in row
  first, second = re.findall(r"<td>(<a .*?</a>)</td>", row, re.S)
  assert "acierto" in first and "error" in second
  assert "parcial" in first  # the echo judge
  assert "1.3 s" in first
  assert first.count('class="status') == 2


def test_cells_link_to_the_report_of_their_run_at_the_question(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b", hrefs=["../one/report.html", "two/report.html"])
  row = row_for(render_compare(runs), "q11")
  assert re.findall(r'<a class="chunk-link" href="([^"]*)"', row) == [
    "../one/report.html#q-q11",
    "two/report.html#q-q11",
  ]


def test_a_cell_link_matches_a_tab_key_of_the_run_report(
  rec: Callable[[str], RunRecord],
) -> None:
  """The anchor must name a tab that the linked report really has."""
  record = rec("a")
  report = render_run(record)
  keys = set(re.findall(r'<button class="tab" data-key="([^"]+)"', report))
  text = render_compare([("a/report.html", record)])
  for href in re.findall(r'href="a/report\.html#([^"]+)"', text):
    assert href in keys, href


def test_question_text_comes_from_the_first_run_that_has_it(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "only11", "a", "b")
  runs[0][1]["questions"][0]["spec"]["question"] = "texto del primero"
  runs[1][1]["questions"][2]["spec"]["question"] = "texto del segundo"
  runs[2][1]["questions"][0]["spec"]["question"] = "texto del tercero"
  text = render_compare(runs)
  assert table_rows(row_for(text, "q11"))[0][1] == "texto del primero"
  # q01 is in the second and the third run: the second one wins
  assert table_rows(row_for(text, "q01"))[0][1].startswith("¿Cuántos días")
  assert "texto del tercero" not in row_for(text, "q01")


def test_ids_are_the_union_in_first_appearance_order(
  rec: Callable[[str], RunRecord],
) -> None:
  text = render_compare(build(rec, "only11", "a"))
  assert [re.findall(r'data-q="([^"]*)"', r)[0] for r in matrix_rows(text)] == [
    "q11",
    "q01",
    "q02",
  ]
  text = render_compare(build(rec, "a", "only11"))
  assert [re.findall(r'data-q="([^"]*)"', r)[0] for r in matrix_rows(text)] == [
    "q01",
    "q02",
    "q11",
  ]


def test_union_order_survives_a_reordered_run(rec: Callable[[str], RunRecord]) -> None:
  runs = build(rec, "a", "a")
  runs[1][1]["questions"].reverse()
  runs[1][1]["questions"].append(copy.deepcopy(runs[0][1]["questions"][0]))
  runs[1][1]["questions"][-1]["spec"]["id"] = "late"
  text = render_compare(runs)
  assert [re.findall(r'data-q="([^"]*)"', r)[0] for r in matrix_rows(text)] == [
    "q01",
    "q02",
    "q11",
    "late",
  ]


def test_a_run_without_a_question_gets_a_dash_and_no_link(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "only11", "a", hrefs=links("x", "y", "z"))
  text = render_compare(runs)
  row = row_for(text, "q01")
  cells = re.findall(r"<td>(.*?)</td>", row, re.S)
  assert len(cells) == 5
  assert cells[3] == '<span class="muted">—</span>'
  assert 'href="y/report.html' not in row
  assert row.count("<a ") == 2
  assert 'href="y/report.html#q-q11"' in row_for(text, "q11")


def test_a_gap_alone_does_not_highlight_a_row(rec: Callable[[str], RunRecord]) -> None:
  text = render_compare(build(rec, "a", "only11"))
  assert diff_ids(text) == []
  assert "tr class=" not in text


def test_runs_without_any_question_give_an_empty_matrix_message(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b")
  for _, record in runs:
    record["questions"] = []
  text = render_compare(runs)
  assert "Sin preguntas." in text
  assert matrix_rows(text) == []


def test_a_duplicated_question_id_in_one_run_does_not_break_the_matrix(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b")
  runs[0][1]["questions"].append(copy.deepcopy(runs[0][1]["questions"][0]))
  text = render_compare(runs)
  assert len(matrix_rows(text)) == 3
  assert row_for(text, "q01").count("<a ") == 2


def test_questions_without_an_id_in_the_spec_get_the_report_numbering(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b")
  for _, record in runs:
    for item in record["questions"]:
      item["spec"].pop("id", None)
  text = render_compare(runs)
  assert [re.findall(r'data-q="([^"]*)"', r)[0] for r in matrix_rows(text)] == [
    "q01",
    "q02",
    "q03",
  ]
  keys = set(re.findall(r'data-key="(q-[^"]+)"', render_run(runs[0][1])))
  hrefs = set(re.findall(r'href="a/report\.html#(q-[^"]+)"', text))
  assert hrefs <= keys


def test_a_question_without_rules_or_judge_has_dash_pills(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b")
  runs[0][1]["questions"][0]["rules"] = None
  runs[0][1]["questions"][0]["judge"] = None
  row = row_for(render_compare(runs), "q01")
  first = re.findall(r"<td>(<a .*?</a>)</td>", row, re.S)[0]
  assert first.count("—") == 2


# ------------------------------------------------------------------ matrix: highlighted rows


def test_the_budget_run_highlights_only_the_row_whose_verdict_changed(
  rec: Callable[[str], RunRecord],
) -> None:
  text = render_compare(build(rec, "a", "c"))
  assert diff_ids(text) == ["q01"]
  assert '<tr class="diff" data-q="q01">' in text
  assert '<tr data-q="q02">' in text and '<tr data-q="q11">' in text


def test_identical_verdicts_highlight_nothing(rec: Callable[[str], RunRecord]) -> None:
  assert diff_ids(render_compare(build(rec, "a", "b"))) == []


def test_a_difference_between_two_of_three_runs_highlights_the_row(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a", "a")
  set_rule(runs[2][1], "q11", "invented_answer")
  assert diff_ids(render_compare(runs)) == ["q11"]


def test_only_the_rule_verdict_counts_not_the_judge_or_the_time(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a")
  set_judge(runs[1][1], "q01", "incorrect")
  runs[1][1]["questions"][0]["seconds"] = 99.0
  assert diff_ids(render_compare(runs)) == []


def test_a_gap_plus_a_real_difference_still_highlights(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "only01", "c")
  assert diff_ids(render_compare(runs)) == ["q01"]


def test_a_missing_rule_result_differs_from_a_verdict(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a")
  set_rule(runs[1][1], "q01", None)
  assert diff_ids(render_compare(runs)) == ["q01"]


def test_no_rule_result_in_any_run_is_not_a_difference(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "a")
  set_rule(runs[0][1], "q01", None)
  set_rule(runs[1][1], "q01", None)
  assert diff_ids(render_compare(runs)) == []


def test_every_row_can_be_highlighted_at_once(rec: Callable[[str], RunRecord]) -> None:
  runs = build(rec, "a", "a")
  for qid in ("q01", "q02", "q11"):
    set_rule(runs[1][1], qid, "retrieval_failure")
  assert diff_ids(render_compare(runs)) == ["q01", "q02", "q11"]


def test_the_page_carries_the_filter_hidden_until_the_script_shows_it(
  rec: Callable[[str], RunRecord],
) -> None:
  text = render_compare(build(rec, "a", "c"))
  assert re.search(
    r'<label class="matrix-filter" hidden><input type="checkbox" id="only-diff"> '
    r"solo filas con diferencias</label>",
    text,
  )
  assert text.count('id="only-diff"') == 1
  css = text[text.index("<style") : text.index("</style>")]
  assert "tr.diff" in css and "tr[hidden]" in css


def test_the_filter_script_and_the_page_markup_agree(
  rec: Callable[[str], RunRecord],
) -> None:
  """report.js looks for ``#only-diff`` and ``tr[data-q]:not(.diff)``: both exist."""
  text = render_compare(build(rec, "a", "c"))
  script = text[text.index("<script>") : text.rindex("</script>")]
  assert 'getElementById("only-diff")' in script
  assert "tr[data-q]:not(.diff)" in script
  rows = matrix_rows(text)
  assert rows and all("data-q=" in r.split(">")[0] for r in rows)
  assert not any(
    "data-q=" in r for r in re.findall(r"<tr[^>]*>", card_of(text, "Puntuación"))
  )


# ------------------------------------------------------------------ links and the report page


def test_links_are_used_verbatim_and_escaped(rec: Callable[[str], RunRecord]) -> None:
  hostile = '"><script>alert(1)</script>/a b&c/report.html'
  runs = build(rec, "a", "b", hrefs=[hostile, "ok/report.html"])
  text = render_compare(runs)
  assert "<script>alert(1)" not in text
  assert html_lib.escape(hostile) in text
  assert 'href="ok/report.html#q-q01"' in text


# ------------------------------------------------------------------ escaping


def hostile_runs(rec: Callable[[str], RunRecord]) -> Runs:
  runs = build(rec, "a", "c")
  for _, record in runs:
    record["title"] = HOSTILE
    record["started_at"] = HOSTILE
    record["finished_at"] = HOSTILE
    for item in record["questions"]:
      item["spec"]["question"] = HOSTILE
  runs[0][1]["questions"][0]["spec"]["id"] = HOSTILE
  runs[1][1]["questions"][0]["spec"]["id"] = HOSTILE
  return runs


def test_nothing_hostile_reaches_the_page_unescaped(
  rec: Callable[[str], RunRecord],
) -> None:
  text = render_compare(hostile_runs(rec))
  # the only script is the report script; no image, no handler, no raw payload
  body = text[text.index("</style>") :]
  script_start = body.index("<script>")
  script_end = body.rindex("</script>")
  outside = body[:script_start] + body[script_end:]
  assert HOSTILE not in outside
  assert "<img" not in outside
  assert "onerror=alert" not in outside.replace("&quot;", '"').replace(
    "onerror=alert(1)&gt;", ""
  )
  assert "<script>alert" not in outside


def test_a_hostile_id_is_escaped_in_the_row_and_the_link(
  rec: Callable[[str], RunRecord],
) -> None:
  text = render_compare(hostile_runs(rec))
  escaped = html_lib.escape(HOSTILE)
  assert f'data-q="{escaped}"' in text
  assert f'href="a/report.html#q-{escaped}"' in text
  rows = matrix_rows(text)
  assert len(rows) == 3
  [highlighted] = [r for r in rows if 'class="diff"' in r.split(">")[0]]
  assert f'data-q="{escaped}"' in highlighted
  assert f"<b>{escaped}</b>" in highlighted


def test_hostile_text_survives_the_round_trip_through_an_html_parser(
  rec: Callable[[str], RunRecord],
) -> None:
  from html.parser import HTMLParser

  class Collect(HTMLParser):
    def __init__(self) -> None:
      super().__init__(convert_charrefs=True)
      self.tags: list[str] = []
      self.attrs: list[tuple[str, str | None]] = []
      self.data: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
      self.tags.append(tag)
      self.attrs.extend(attrs)

    def handle_data(self, data: str) -> None:
      self.data.append(data)

  parsed = Collect()
  parsed.feed(render_compare(hostile_runs(rec)))
  assert "img" not in parsed.tags
  assert parsed.tags.count("script") == 1
  assert not any(name.startswith("on") for name, _ in parsed.attrs)
  assert HOSTILE in parsed.data


def test_unicode_survives_in_titles_ids_and_config(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "b")
  runs[0][1]["title"] = "Ñandú ✓ 日本語"
  runs[0][1]["config"]["description"] = "años — “comillas”"
  runs[1][1]["config"]["description"] = "otro"
  text = render_compare(runs)
  assert "Ñandú ✓ 日本語" in text
  assert "años — “comillas”" in text


# ------------------------------------------------------------------ determinism and purity


def test_same_input_gives_the_same_bytes(rec: Callable[[str], RunRecord]) -> None:
  runs = build(rec, "a", "b", "c")
  first = render_compare(runs)
  assert render_compare(runs) == first
  assert render_compare(build(rec, "a", "b", "c")) == first  # equal copies


def test_rendering_does_not_modify_the_records(rec: Callable[[str], RunRecord]) -> None:
  runs = build(rec, "a", "b", "c")
  before = copy.deepcopy(runs)
  render_compare(runs)
  assert runs == before


def test_the_input_may_be_any_sequence(rec: Callable[[str], RunRecord]) -> None:
  runs = build(rec, "a", "b")
  assert render_compare(tuple(runs)) == render_compare(runs)


def test_the_order_of_the_runs_changes_the_columns(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "c")
  assert render_compare(runs) != render_compare(runs[::-1])
  rows = differences(render_compare(runs[::-1]))
  assert rows == [["enrich.max_context_tokens", "30", "3000"]]


def test_no_run_specific_noise_in_the_output(rec: Callable[[str], RunRecord]) -> None:
  text = render_compare(build(rec, "a", "b"))
  assert str(Path.cwd()) not in text


# ------------------------------------------------------------------ malformed records (library level)


@pytest.mark.parametrize("key", ["config", "environment", "questions", "finished_at"])
def test_a_record_missing_a_section_raises_instead_of_rendering_garbage(
  rec: Callable[[str], RunRecord], key: str
) -> None:
  runs = build(rec, "a", "b")
  del runs[1][1][key]  # type: ignore[misc]
  with pytest.raises((KeyError, TypeError)):
    render_compare(runs)


# ------------------------------------------------------------------ write_compare


def test_write_compare_writes_the_same_text_as_render_compare(
  rec: Callable[[str], RunRecord], tmp_path: Path
) -> None:
  runs = build(rec, "a", "b")
  target = tmp_path / "deep" / "er" / "compare.html"
  write_compare(runs, target)
  assert target.read_text(encoding="utf-8") == render_compare(runs)
  assert [p.name for p in target.parent.iterdir()] == ["compare.html"]


def test_write_compare_overwrites_and_leaves_no_temporary_file(
  rec: Callable[[str], RunRecord], tmp_path: Path
) -> None:
  target = tmp_path / "compare.html"
  target.write_text("old", encoding="utf-8")
  write_compare(build(rec, "a", "b"), target)
  assert target.read_text(encoding="utf-8").startswith("<!doctype html>")
  assert [p.name for p in tmp_path.iterdir()] == ["compare.html"]


def test_write_compare_into_a_directory_raises_a_record_error(
  rec: Callable[[str], RunRecord], tmp_path: Path
) -> None:
  with pytest.raises(RecordError, match="no se puede escribir"):
    write_compare(build(rec, "a", "b"), tmp_path)


def test_write_compare_with_a_lone_surrogate_raises_and_keeps_the_old_file(
  rec: Callable[[str], RunRecord], tmp_path: Path
) -> None:
  target = tmp_path / "compare.html"
  target.write_text("old", encoding="utf-8")
  runs = build(rec, "a", "b")
  runs[0][1]["title"] = "bad \ud800"
  with pytest.raises(RecordError, match="UTF-8"):
    write_compare(runs, target)
  assert target.read_text(encoding="utf-8") == "old"
  assert [p.name for p in tmp_path.iterdir()] == ["compare.html"]


def test_write_compare_of_nothing_is_a_value_error_and_writes_nothing(
  tmp_path: Path,
) -> None:
  with pytest.raises(ValueError, match="at least one run"):
    write_compare([], tmp_path / "compare.html")
  assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ real records from json


def test_records_read_back_from_run_json_give_the_same_page(
  rec: Callable[[str], RunRecord],
) -> None:
  runs = build(rec, "a", "c")
  again = [(href, json.loads(json.dumps(record))) for href, record in runs]
  assert render_compare(again) == render_compare(runs)

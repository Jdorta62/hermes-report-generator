"""The experiment notebook ``notebooks/rag_evaluation.ipynb`` (task 5.5.6).

The offline tests run the whole notebook with ``nbclient`` in a real kernel,
with ``HERMES_NOTEBOOK_OFFLINE=1`` (hash embedder, echo LLM and judge, Qdrant in
memory) and ``HERMES_NOTEBOOK_OUT`` pointing into ``tmp_path``. The notebook
object is executed in memory and never saved, so the stored file stays free of
outputs. The integration test reads a real Quijote collection and skips when
Ollama or Qdrant do not answer.
"""

from __future__ import annotations

import csv
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "rag_evaluation.ipynb"

# Section 7 of the design: the columns that make two rows comparable.
FIXED_COLUMNS = [
  "timestamp",
  "run_id",
  "collection",
  "embedding_model",
  "llm_model",
  "judge_model",
  "top_k",
  "score_threshold",
  "filters",
  "n_questions",
  "seconds_mean",
  "config_hash",
  "note",
]
# Scores of the families that run offline on the demo questions.
SOME_SCORES = ["doc_hit", "evidence_recall", "faithfulness", "correctness_f1"]
WRITE_CALLS = (
  "create_collection",
  "recreate_collection",
  "delete_collection",
  "update_collection",
  "upsert",
  "upload_points",
  "upload_collection",
  "delete",
  "delete_vectors",
  "set_payload",
  "overwrite_payload",
  "delete_payload",
  "clear_payload",
  "update_vectors",
  "batch_update_points",
  "create_payload_index",
  "delete_payload_index",
  "create_snapshot",
  "recover_snapshot",
)


def load() -> nbformat.NotebookNode:
  notebook: nbformat.NotebookNode = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]
  return notebook


def cell_index(notebook: nbformat.NotebookNode, cell_id: str) -> int:
  return next(i for i, c in enumerate(notebook.cells) if c.get("id") == cell_id)


def insert_after(
  notebook: nbformat.NotebookNode, cell_id: str, source: str, new_id: str
) -> None:
  """Add a test-side code cell after ``cell_id`` (the file on disk is untouched)."""
  cell = nbformat.v4.new_code_cell(source, id=new_id)  # type: ignore[no-untyped-call]
  notebook.cells.insert(cell_index(notebook, cell_id) + 1, cell)


def execute(
  notebook: nbformat.NotebookNode, timeout: int = 300
) -> nbformat.NotebookNode:
  """Run every cell in a fresh kernel whose cwd is the repository."""
  NotebookClient(
    notebook,
    timeout=timeout,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
  ).execute()
  return notebook


def stream_text(notebook: nbformat.NotebookNode, cell_id: str) -> str:
  cell = notebook.cells[cell_index(notebook, cell_id)]
  return "".join(o.get("text", "") for o in cell.outputs if o.output_type == "stream")


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  lines = path.read_text(encoding="utf-8").splitlines()
  return [json.loads(line) for line in lines]


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
  """Offline mode into ``tmp_path/out``; returns that folder."""
  out = tmp_path / "out"
  monkeypatch.setenv("HERMES_NOTEBOOK_OFFLINE", "1")
  monkeypatch.setenv("HERMES_NOTEBOOK_OUT", str(out))
  for name in (
    "HERMES_NOTEBOOK_TOP_K",
    "HERMES_NOTEBOOK_COLLECTIONS",
    "HERMES_NOTEBOOK_DATASET",
    "HERMES_NOTEBOOK_QDRANT_URL",
  ):
    monkeypatch.delenv(name, raising=False)
  return out


@pytest.fixture
def run_offline(
  offline: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., nbformat.NotebookNode]:
  """Run the notebook offline with an optional ``top_k``; returns the executed copy."""

  def run(top_k: int | None = None) -> nbformat.NotebookNode:
    if top_k is None:
      monkeypatch.delenv("HERMES_NOTEBOOK_TOP_K", raising=False)
    else:
      monkeypatch.setenv("HERMES_NOTEBOOK_TOP_K", str(top_k))
    return execute(load())

  return run


# --------------------------------------------------------- the stored notebook


class TestStoredNotebook:
  def test_it_has_no_outputs_and_no_execution_counts(self) -> None:
    notebook = load()
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert code
    for cell in code:
      assert cell.outputs == [], cell.get("id")
      assert cell.execution_count is None, cell.get("id")

  def test_the_raw_json_has_no_outputs_either(self) -> None:
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in raw["cells"]:
      if cell["cell_type"] == "code":
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
    assert "image/png" not in NOTEBOOK.read_text(encoding="utf-8")

  def test_it_is_valid_nbformat_with_unique_cell_ids(self) -> None:
    notebook = load()
    nbformat.validate(notebook)
    ids = [c["id"] for c in notebook.cells]
    assert len(ids) == len(set(ids))

  def test_it_has_the_eight_sections_in_order(self) -> None:
    titles = [
      line
      for c in load().cells
      if c.cell_type == "markdown"
      for line in c.source.splitlines()
      if line.startswith("## ")
    ]
    assert [t.split(".")[0] for t in titles] == [f"## {n}" for n in range(1, 9)]

  def test_every_code_cell_starts_with_a_comment(self) -> None:
    for cell in load().cells:
      if cell.cell_type == "code":
        assert cell.source.lstrip().startswith("#"), cell.get("id")

  def test_it_does_not_call_any_collection_write_api(self) -> None:
    """Static guard: writes only go through ``ensure_collection`` (prefix check)."""
    source = "\n".join(c.source for c in load().cells if c.cell_type == "code")
    for call in WRITE_CALLS:
      assert f".{call}(" not in source, call
    assert "ensure_collection(" in source


# ------------------------------------------------------------- offline, whole run


class TestOfflineRun:
  def test_two_runs_with_different_top_k_give_two_comparable_rows(
    self, run_offline: Callable[..., Any], offline: Path
  ) -> None:
    run_offline(10)
    run_offline(3)

    rows = read_csv(offline / "history.csv")
    assert len(rows) == 2
    for column in FIXED_COLUMNS:
      assert column in rows[0], column
    for score in SOME_SCORES:
      assert f"{score}_mean" in rows[0], score
      assert f"{score}_errors" in rows[0], score
    # Comparable: same collection, same questions, different top_k and hash.
    assert {r["collection"] for r in rows} == {"hermes_report_notebook_demo"}
    assert [r["top_k"] for r in rows] == ["10", "3"]
    assert rows[0]["config_hash"] != rows[1]["config_hash"]
    assert rows[0]["run_id"] != rows[1]["run_id"]
    assert rows[0]["n_questions"] == rows[1]["n_questions"] == "7"
    assert {r["embedding_model"] for r in rows} == {"hash-256"}
    assert {r["llm_model"] for r in rows} == {"echo"}
    assert {r["judge_model"] for r in rows} == {"echo"}
    for row in rows:
      assert float(row["seconds_mean"]) >= 0
      assert 0.0 <= float(row["doc_hit_mean"]) <= 1.0
      assert row["doc_hit_errors"] == "0"

  def test_jsonl_and_detail_files_match_the_csv(
    self, run_offline: Callable[..., Any], offline: Path
  ) -> None:
    run_offline(10)
    run_offline(3)

    rows = read_csv(offline / "history.csv")
    lines = read_jsonl(offline / "history.jsonl")
    assert [x["run_id"] for x in lines] == [r["run_id"] for r in rows]
    assert [x["config_hash"] for x in lines] == [r["config_hash"] for r in rows]
    assert [x["top_k"] for x in lines] == [10, 3]
    assert lines[0]["filters"] == "{}"
    details = sorted(offline.glob("*-detail.csv"))
    assert len(details) == 2
    assert {d.name for d in details} == {f"{r['run_id']}-detail.csv" for r in rows}
    for detail in details:
      table = read_csv(detail)
      assert len(table) == 7
      assert {"run_id", "collection", "question_id", "answer", "retrieved"} <= set(
        table[0]
      )
      assert [r["question_id"] for r in table] == [f"d0{n}" for n in range(1, 8)]

  def test_an_identical_rerun_shares_the_config_hash(
    self, run_offline: Callable[..., Any], offline: Path
  ) -> None:
    run_offline(5)
    run_offline(5)
    rows = read_csv(offline / "history.csv")
    assert rows[0]["config_hash"] == rows[1]["config_hash"]
    assert rows[0]["run_id"] != rows[1]["run_id"]

  def test_no_cell_fails_and_the_run_leaves_the_repository_clean(
    self, run_offline: Callable[..., Any], offline: Path
  ) -> None:
    existed = (ROOT / "notebooks" / "out").exists()
    executed = run_offline(4)
    for cell in executed.cells:
      if cell.cell_type == "code":
        assert not any(o.output_type == "error" for o in cell.outputs)
        assert cell.outputs, f"{cell.get('id')} shows nothing"  # each cell shows
    assert (ROOT / "notebooks" / "out").exists() == existed
    assert (offline / "ingest-cache").is_dir()
    # The stored notebook is still clean after the run.
    assert all(c.outputs == [] for c in load().cells if c.cell_type == "code")

  def test_a_second_run_does_not_lose_the_first_row_or_its_values(
    self, run_offline: Callable[..., Any], offline: Path
  ) -> None:
    run_offline(10)
    first = read_csv(offline / "history.csv")[0]
    run_offline(2)
    rows = read_csv(offline / "history.csv")
    assert rows[0] == first

  def test_new_columns_in_a_later_run_leave_old_rows_empty(
    self, run_offline: Callable[..., Any], offline: Path
  ) -> None:
    run_offline(10)
    # A run with the metric `bleu` switched on adds `bleu_mean` columns.
    notebook = load()
    insert_after(
      notebook,
      "s2-code",
      'CONFIG["metrics"]["include"] = ["bleu"]\n',
      "t-extra-metric",
    )
    execute(notebook)
    rows = read_csv(offline / "history.csv")
    assert len(rows) == 2
    assert "bleu_mean" in rows[1]
    assert rows[0]["bleu_mean"] == ""
    assert rows[1]["bleu_mean"] != ""
    assert rows[0]["doc_hit_mean"] != ""
    assert len(read_jsonl(offline / "history.jsonl")) == 2

  def test_metrics_can_be_switched_off(self, offline: Path) -> None:
    notebook = load()
    insert_after(
      notebook, "s2-code", 'CONFIG["metrics"]["enabled"] = False\n', "t-no-metrics"
    )
    execute(notebook)
    rows = read_csv(offline / "history.csv")
    assert len(rows) == 1
    assert "doc_hit_mean" not in rows[0]
    assert rows[0]["n_questions"] == "7"

  def test_the_judge_can_be_switched_off(self, offline: Path) -> None:
    """Offline overrides the judge; this switches it off after the override."""
    notebook = load()
    insert_after(
      notebook, "s2-code", 'CONFIG["judge"] = {"provider": None}\n', "t-no-judge"
    )
    executed = execute(notebook)
    rows = read_csv(offline / "history.csv")
    assert rows[0]["judge_model"] == ""
    # Judged families are `skipped`, never a crash.
    assert "skipped" in "".join(
      "".join(o.get("text", "") + str(o.get("data", "")) for o in c.outputs)
      for c in executed.cells
      if c.get("id") == "s6-failures"
    )

  def test_a_dataset_as_a_bare_array_with_ground_truth_alias(
    self, offline: Path, tmp_path: Path
  ) -> None:
    dataset = tmp_path / "bare.json"
    dataset.write_text(
      json.dumps(
        [
          {"question": "¿Qué canal usa Faro?", "ground_truth": "faro-guardia"},
          {"question": "¿Y las copias?", "expected": "cada 6 horas"},
        ]
      ),
      encoding="utf-8",
    )
    notebook = load()
    insert_after(
      notebook, "s2-code", f'CONFIG["dataset"] = {str(dataset)!r}\n', "t-dataset"
    )
    insert_after(
      notebook,
      "s4-code",
      "print(golden[['id', 'expected', 'refusal']].to_json(orient='records'))\n",
      "t-show-golden",
    )
    executed = execute(notebook)
    shown = json.loads(stream_text(executed, "t-show-golden"))
    assert [r["id"] for r in shown] == ["q01", "q02"]
    assert shown[0]["expected"] == "faro-guardia"
    assert shown[1]["expected"] == "cada 6 horas"
    assert all(r["refusal"] is False for r in shown)
    assert read_csv(offline / "history.csv")[0]["n_questions"] == "2"

  def test_a_question_without_text_stops_the_notebook_with_a_clear_error(
    self, offline: Path, tmp_path: Path
  ) -> None:
    dataset = tmp_path / "bad.json"
    dataset.write_text(json.dumps([{"question": "  ", "expected": "x"}]), "utf-8")
    notebook = load()
    insert_after(
      notebook, "s2-code", f'CONFIG["dataset"] = {str(dataset)!r}\n', "t-dataset"
    )
    with pytest.raises(CellExecutionError, match="no tiene texto"):
      execute(notebook)
    assert not (offline / "history.csv").exists()

  @pytest.mark.parametrize(
    "content", ["[]", '{"questions": []}', '{"title": "x", "questions": []}']
  )
  def test_an_empty_dataset_writes_one_row_with_zero_questions(
    self, offline: Path, tmp_path: Path, content: str
  ) -> None:
    dataset = tmp_path / "empty.json"
    dataset.write_text(content, encoding="utf-8")
    notebook = load()
    insert_after(
      notebook, "s2-code", f'CONFIG["dataset"] = {str(dataset)!r}\n', "t-dataset"
    )
    executed = execute(notebook)
    for cell in executed.cells:
      if cell.cell_type == "code":
        assert not any(o.output_type == "error" for o in cell.outputs), cell.get("id")
    assert "No hay preguntas" in stream_text(executed, "s5-example")

    rows = read_csv(offline / "history.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["collection"] == "hermes_report_notebook_demo"
    assert row["n_questions"] == "0"
    assert row["seconds_mean"] == ""
    means = [k for k in row if k.endswith("_mean")]
    errors = [k for k in row if k.endswith("_errors")]
    assert means
    assert errors
    for key in means:
      assert row[key] == "", key
    for key in errors:
      assert row[key] == "0", key
    lines = read_jsonl(offline / "history.jsonl")
    assert len(lines) == 1
    assert lines[0]["n_questions"] == 0
    assert lines[0]["run_id"] == row["run_id"]
    detail = offline / f"{row['run_id']}-detail.csv"
    assert detail.exists()
    assert read_csv(detail) == []  # header only

  def test_an_empty_dataset_run_adds_a_row_after_a_normal_one(
    self, run_offline: Callable[..., Any], offline: Path, tmp_path: Path
  ) -> None:
    run_offline(10)
    first = read_csv(offline / "history.csv")[0]
    dataset = tmp_path / "empty.json"
    dataset.write_text("[]", encoding="utf-8")
    notebook = load()
    insert_after(
      notebook, "s2-code", f'CONFIG["dataset"] = {str(dataset)!r}\n', "t-dataset"
    )
    execute(notebook)
    rows = read_csv(offline / "history.csv")
    assert len(rows) == 2
    assert rows[0] == first  # the old row is intact
    assert rows[1]["n_questions"] == "0"
    assert rows[1]["doc_hit_mean"] == ""
    assert rows[1]["doc_hit_errors"] == "0"
    assert len(read_jsonl(offline / "history.jsonl")) == 2

  def test_empty_collections_write_nothing_and_do_not_fail(self, offline: Path) -> None:
    notebook = load()
    # Offline mode forces the demo collection right after s2-code, so set it after.
    insert_after(
      notebook, "s2-code", 'CONFIG["collections"] = []\n', "t-no-collections"
    )
    executed = execute(notebook)
    for cell in executed.cells:
      if cell.cell_type == "code":
        assert not any(o.output_type == "error" for o in cell.outputs), cell.get("id")
    assert "No hay colecciones" in stream_text(executed, "s5-example")
    assert "No hay colecciones" in stream_text(executed, "s8-code")
    assert not (offline / "history.csv").exists()
    assert not (offline / "history.jsonl").exists()
    assert list(offline.glob("*-detail.csv")) == []

  def test_empty_collections_leave_an_existing_history_untouched(
    self, run_offline: Callable[..., Any], offline: Path
  ) -> None:
    run_offline(10)
    before = {
      name: (offline / name).read_bytes() for name in ("history.csv", "history.jsonl")
    }
    details = sorted(p.name for p in offline.glob("*-detail.csv"))
    notebook = load()
    insert_after(
      notebook, "s2-code", 'CONFIG["collections"] = []\n', "t-no-collections"
    )
    execute(notebook)
    for name, content in before.items():
      assert (offline / name).read_bytes() == content, name
    assert sorted(p.name for p in offline.glob("*-detail.csv")) == details

  def test_empty_collections_and_empty_dataset_together_do_not_fail(
    self, offline: Path, tmp_path: Path
  ) -> None:
    dataset = tmp_path / "empty.json"
    dataset.write_text("[]", encoding="utf-8")
    notebook = load()
    insert_after(
      notebook,
      "s2-code",
      f'CONFIG["dataset"] = {str(dataset)!r}\nCONFIG["collections"] = []\n',
      "t-both-empty",
    )
    execute(notebook)
    assert not (offline / "history.csv").exists()

  def test_a_non_numeric_top_k_fails_loudly(
    self, offline: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setenv("HERMES_NOTEBOOK_TOP_K", "many")
    with pytest.raises(CellExecutionError, match="ValueError"):
      execute(load())
    assert not (offline / "history.csv").exists()

  def test_a_payload_filter_restricts_the_retrieved_sources(
    self, offline: Path
  ) -> None:
    notebook = load()
    insert_after(
      notebook,
      "s2-code",
      'CONFIG["filters"] = {"source": "manuales/faro.md"}\n',
      "t-filter",
    )
    insert_after(
      notebook,
      "s5-code",
      "print(sorted({s for c in items[CONFIG['collections'][0]]"
      " for s in {x.source for x in c.retrieved}}))\n",
      "t-sources",
    )
    executed = execute(notebook)
    sources = stream_text(executed, "t-sources").strip()
    assert sources == "['manuales/faro.md']"
    row = read_csv(offline / "history.csv")[0]
    assert json.loads(row["filters"]) == {"source": "manuales/faro.md"}

  def test_a_score_threshold_above_every_score_retrieves_nothing_without_crashing(
    self, offline: Path
  ) -> None:
    notebook = load()
    insert_after(
      notebook, "s2-code", 'CONFIG["score_threshold"] = 1.5\n', "t-threshold"
    )
    insert_after(
      notebook,
      "s5-code",
      "print(sum(len(i.retrieved) for c in items.values() for i in c))\n",
      "t-count",
    )
    executed = execute(notebook)
    assert stream_text(executed, "t-count").strip() == "0"
    rows = read_csv(offline / "history.csv")
    assert rows[0]["score_threshold"] == "1.5"
    assert rows[0]["n_questions"] == "7"

  def test_a_missing_collection_is_reported_per_question_not_fatal(
    self, offline: Path
  ) -> None:
    """Online-style failure inside the sweep: the row is still written."""
    notebook = load()
    # After the offline collection exists, ask the retrieval for a name that does not.
    insert_after(
      notebook, "s3-code", 'CONFIG["collections"] = ["no_such_collection"]\n', "t-gone"
    )
    executed = execute(notebook)
    out = "".join(
      str(o.get("data", {}).get("text/plain", "")) + o.get("text", "")
      for c in executed.cells
      if c.get("id") == "s5-code"
      for o in c.outputs
    )
    assert "no_such_collection" in out or "Not found" in out or "error" in out.lower()
    rows = read_csv(offline / "history.csv")
    assert [r["collection"] for r in rows] == ["no_such_collection"]


# --------------------------------------- collections the notebook may touch


RECORDER = '''
import json as _json
from qdrant_client import QdrantClient as _Q

class _Recorder:
  """Forward every call to a real in-memory client and remember its name."""

  def __init__(self, real):
    self._real = real
    self.calls = []

  def __getattr__(self, name):
    attr = getattr(self._real, name)
    if not callable(attr):
      return attr
    def wrapper(*args, **kwargs):
      self.calls.append(name)
      return attr(*args, **kwargs)
    return wrapper

_real = _Q(":memory:")
_real.create_collection(
  "other_data", vectors_config=models.VectorParams(size=256, distance="Cosine")
)
_vecs = embedder.get_text_embedding_batch(["Faro avisa al canal #faro-guardia."])
_real.upsert("other_data", points=[
  models.PointStruct(id=1, vector=_vecs[0], payload={"text": "Faro avisa al canal "
    "#faro-guardia.", "source": "otros/faro.md"})
])
_before = _real.count("other_data").count
RECORDER_SPY = _Recorder(_real)
make_client = lambda url: RECORDER_SPY
check_ollama = lambda *args, **kwargs: None
'''

ONLINE_STYLE_CONFIG = """
# Online path with local stand-ins: no Ollama, the Qdrant client is the spy.
CONFIG["collections"] = ["other_data"]
CONFIG["qdrant_url"] = "http://spy:6333"
CONFIG["embedding"] = {"provider": "hash", "model": "hash-256"}
CONFIG["generation"] = {**CONFIG["generation"], "provider": "echo", "model": "echo"}
CONFIG["judge"] = {"provider": "echo", "model": "echo"}
CONFIG["dataset"] = "examples/demo/preguntas.json"
"""


class TestCollectionsTouched:
  def test_offline_only_the_prefixed_collection_exists_in_the_client(
    self, offline: Path
  ) -> None:
    notebook = load()
    insert_after(
      notebook,
      "s8-code",
      "print(json.dumps(sorted(c.name for c in client.get_collections().collections)))\n",
      "t-names",
    )
    executed = execute(notebook)
    names = json.loads(stream_text(executed, "t-names"))
    assert names == ["hermes_report_notebook_demo"]

  def test_online_mode_reads_a_foreign_collection_and_never_writes(
    self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
  ) -> None:
    """Online (not offline) over ``other_data``: reads happen, no write call at all."""
    out = tmp_path / "out"
    monkeypatch.delenv("HERMES_NOTEBOOK_OFFLINE", raising=False)
    monkeypatch.setenv("HERMES_NOTEBOOK_OUT", str(out))
    notebook = load()
    # Order matters: CONFIG first, then the spy is built from the same embedder
    # settings (the hash embedder dimension is 256).
    insert_after(notebook, "s2-code", ONLINE_STYLE_CONFIG, "t-config")
    # The spy needs `embedder`, which exists only after make_embedder: build the
    # services cell first with the real make_client, then swap the client in.
    services = notebook.cells[cell_index(notebook, "s2-services")]
    services.source = services.source.replace(
      'client = make_client(CONFIG["qdrant_url"])\n', ""
    ).replace(
      "llm = make_llm(",
      RECORDER + "client = RECORDER_SPY\nllm = make_llm(",
      1,
    )
    # `embedder` is defined before `llm` in the cell, so the spy cell may use it.
    insert_after(
      notebook,
      "s8-code",
      "print(json.dumps({'calls': sorted(set(RECORDER_SPY.calls)),"
      " 'before': _before, 'after': _real.count('other_data').count,"
      " 'names': sorted(c.name for c in _real.get_collections().collections)}))\n",
      "t-calls",
    )
    executed = execute(notebook)
    report = json.loads(stream_text(executed, "t-calls"))
    assert "query_points" in report["calls"]
    assert report["names"] == ["other_data"]
    assert report["after"] == report["before"] == 1
    assert not set(report["calls"]) & set(WRITE_CALLS)
    rows = read_csv(out / "history.csv")
    assert [r["collection"] for r in rows] == ["other_data"]

  def test_ensure_collection_refuses_a_foreign_name_inside_the_kernel(
    self, offline: Path
  ) -> None:
    """The write path the notebook uses rejects names without the prefix."""
    notebook = load()
    insert_after(
      notebook,
      "s3-code",
      "try:\n"
      "  ensure_collection(client, 'precious_data', ingestion.points, ':memory:')\n"
      "  print('WROTE')\n"
      "except Exception as exc:\n"
      "  print(type(exc).__name__)\n",
      "t-refuse",
    )
    executed = execute(notebook)
    assert stream_text(executed, "t-refuse").strip() == "StoreError"


# ------------------------------------------------------------ with real services


def _http_json(url: str) -> Any:
  try:
    with urllib.request.urlopen(url, timeout=3) as reply:  # noqa: S310 - fixed http URL
      return json.load(reply)
  except (OSError, urllib.error.URLError, ValueError):
    return None


@pytest.mark.integration
def test_notebook_over_a_quijote_collection_with_real_ollama_and_qdrant(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  host = os.environ.get("OLLAMA_HOST", "").strip() or "http://localhost:11434"
  if not host.startswith("http"):
    host = f"http://{host}"
  qdrant_url = "http://localhost:6333"
  tags = _http_json(f"{host.rstrip('/')}/api/tags")
  have = {m["name"].removesuffix(":latest") for m in (tags or {}).get("models", [])}
  needed = {"nomic-embed-text", "qwen3:4b"}
  if not needed <= have:
    pytest.skip(f"Ollama at {host} is unreachable or lacks {sorted(needed - have)}")
  listing = _http_json(f"{qdrant_url}/collections")
  if listing is None:
    pytest.skip(f"Qdrant at {qdrant_url} is unreachable")
  before = sorted(c["name"] for c in listing["result"]["collections"])
  quijote = [n for n in before if n.startswith("hermes_report_don_quijote")]
  if not quijote:
    pytest.skip("run `hermes-report run examples/quijote/run.toml` first")

  # Two questions keep the real judge and LLM calls short.
  source = json.loads((ROOT / "examples/quijote/preguntas.json").read_text("utf-8"))
  dataset = tmp_path / "two.json"
  dataset.write_text(
    json.dumps({"title": "Dos", "questions": source["questions"][:2]}),
    encoding="utf-8",
  )
  out = tmp_path / "out"
  monkeypatch.delenv("HERMES_NOTEBOOK_OFFLINE", raising=False)
  monkeypatch.setenv("HERMES_NOTEBOOK_OUT", str(out))
  monkeypatch.setenv("HERMES_NOTEBOOK_COLLECTIONS", quijote[0])
  monkeypatch.setenv("HERMES_NOTEBOOK_DATASET", str(dataset))
  monkeypatch.setenv("HERMES_NOTEBOOK_QDRANT_URL", qdrant_url)
  monkeypatch.setenv("HERMES_NOTEBOOK_TOP_K", "5")

  execute(load(), timeout=900)

  rows = read_csv(out / "history.csv")
  assert len(rows) == 1
  assert rows[0]["collection"] == quijote[0]
  assert rows[0]["embedding_model"] == "nomic-embed-text"
  assert rows[0]["top_k"] == "5"
  assert rows[0]["n_questions"] == "2"
  assert float(rows[0]["doc_hit_mean"]) >= 0.0
  assert len(read_jsonl(out / "history.jsonl")) == 1
  assert len(list(out.glob("*-detail.csv"))) == 1
  # Reading only: the set of collections did not change.
  after = sorted(
    c["name"] for c in _http_json(f"{qdrant_url}/collections")["result"]["collections"]
  )
  assert after == before


@pytest.fixture(autouse=True)
def _no_stray_jupyter_output() -> Iterator[None]:
  """The stored notebook must be byte-identical after each test."""
  before = NOTEBOOK.read_bytes()
  yield
  assert NOTEBOOK.read_bytes() == before

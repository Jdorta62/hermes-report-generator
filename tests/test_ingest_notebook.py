"""The ingestion notebook ``notebooks/ingest_collections.ipynb`` (task 6.6.1).

The offline tests run the notebook with ``nbclient`` in a real kernel, with
``HERMES_INGEST_OFFLINE=1`` (hash embedder, Qdrant in memory),
``HERMES_INGEST_OUT`` and ``HERMES_INGEST_CONFIG`` inside ``tmp_path``. The
notebook object is executed in memory and never saved. The in-memory Qdrant
lives inside the kernel, so the scenarios that need a second pass (reuse,
force, changed settings) copy the cells ``s5`` to ``s9`` after the first pass
and run them in the same kernel. The integration test ingests two real
documents with Ollama and Qdrant and deletes what it creates.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "ingest_collections.ipynb"
README = ROOT / "README.md"
DEMO = "examples/demo/corpus"
MARK = "@@DUMP@@"

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

# Two collections that share manuales/faro.md and differ in max_chunk_size.
SHARED = {
  "corpus_dir": DEMO,
  "qdrant_url": "http://unused:6333",
  "collections": {
    "hermes_report_t_small": {
      "documents": ["manuales/faro.md", "manuales/brujula.md"],
      "ingest": {"max_chunk_size": 200},
    },
    "hermes_report_t_big": {
      "documents": ["manuales/faro.md"],
      "ingest": {"max_chunk_size": 2000},
    },
  },
}
SMALL_DOCS = {"manuales/faro.md", "manuales/brujula.md"}
BIG_DOCS = {"manuales/faro.md"}
ALL_DOCS = {
  "manuales/brujula.md",
  "manuales/faro.md",
  "politicas/teletrabajo.md",
  "politicas/vacaciones.md",
}

# Test-side cells: they read the kernel state and print it as one JSON line.
DUMP = f'''
import json as _json
def _dump():
  names = sorted(c.name for c in client.get_collections().collections)
  def rows(df):
    return json.loads(df.to_json(orient="records"))
  out = {{
    "names": names,
    "counts": {{n: client.count(n, exact=True).count for n in names}},
    "sources": {{
      n: sorted({{p.payload["source"] for p in client.scroll(n, limit=10000)[0]}})
      for n in names
    }},
    "before": BEFORE,
    "after": AFTER,
    "diff": rows(diff),
    "totals": rows(totals),
    "collection_stats": rows(collection_stats),
    "document_stats": rows(document_stats),
    "unused": rows(unused_documents),
    "ready": READY,
    "outcomes": {{n: r["outcome"] for n, r in RESULTS.items()}},
    "errors": {{n: r["error"] for n, r in RESULTS.items()}},
    "force": CONFIG["force"],
    "writes": getattr(client, "writes", None),
  }}
  print("{MARK}" + _json.dumps(out, default=str))
_dump()
'''

SEED = """
# Test-side: foreign collections that the notebook must never touch.
from qdrant_client import models as _m
for _name in ("other_data", "hermes_report_stray"):
  client.create_collection(
    _name, vectors_config=_m.VectorParams(size=256, distance="Cosine")
  )
  client.upsert(
    _name,
    points=[_m.PointStruct(id=1, vector=[0.1] * 256, payload={"source": "x.md"})],
  )
"""

SPY = f"""
# Test-side: record every write call and the collection it targets.
_WRITES = {WRITE_CALLS!r}

class _Spy:
  def __init__(self, real):
    self._real = real
    self.writes = []

  def __getattr__(self, name):
    attr = getattr(self._real, name)
    if not callable(attr) or name not in _WRITES:
      return attr
    def wrapper(*args, **kwargs):
      target = args[0] if args else kwargs.get("collection_name")
      self.writes.append([name, target])
      return attr(*args, **kwargs)
    return wrapper

client = _Spy(client)
"""


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


def code_ids(notebook: nbformat.NotebookNode) -> list[str]:
  return [c.id for c in notebook.cells if c.cell_type == "code"]


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


def cell_text(notebook: nbformat.NotebookNode, cell_id: str) -> str:
  """Everything a cell showed: stream text and the plain text of displays."""
  cell = notebook.cells[cell_index(notebook, cell_id)]
  parts = []
  for o in cell.outputs:
    if o.output_type == "stream":
      parts.append(o.get("text", ""))
    elif o.output_type in ("display_data", "execute_result"):
      parts.append(o.get("data", {}).get("text/plain", ""))
  return "\n".join(parts)


def dumps(notebook: nbformat.NotebookNode, cell_id: str) -> dict[str, Any]:
  line = next(
    ln for ln in cell_text(notebook, cell_id).splitlines() if ln.startswith(MARK)
  )
  result: dict[str, Any] = json.loads(line.removeprefix(MARK))
  return result


def no_cell_failed(notebook: nbformat.NotebookNode) -> None:
  for cell in notebook.cells:
    if cell.cell_type == "code":
      assert not any(o.output_type == "error" for o in cell.outputs), cell.get("id")


def with_second_pass(
  notebook: nbformat.NotebookNode,
  between: str = "",
  rerun: Sequence[str] = ("s5-code", "s6-code", "s7-code", "s7-documents", "s7-unused"),
  tail: Sequence[str] = ("s8-code", "s9-code"),
) -> nbformat.NotebookNode:
  """Dump the first pass, run ``between``, then copy ``rerun`` + ``tail`` cells.

  Cell ids of the copies are ``r2-<id>``; the dumps are ``t-dump-1`` / ``t-dump-2``.
  """
  insert_after(notebook, "s9-code", DUMP, "t-dump-1")
  anchor = "t-dump-1"
  if between:
    insert_after(notebook, anchor, between, "t-between")
    anchor = "t-between"
  for cid in (*rerun, *tail):
    copy_ = copy.deepcopy(notebook.cells[cell_index(notebook, cid)])
    copy_.id = f"r2-{cid}"
    copy_.outputs = []
    notebook.cells.insert(cell_index(notebook, anchor) + 1, copy_)
    anchor = copy_.id
  insert_after(notebook, anchor, DUMP, "t-dump-2")
  return notebook


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
  """Offline mode, output in ``tmp_path/out``; returns that folder."""
  out = tmp_path / "out"
  monkeypatch.setenv("HERMES_INGEST_OFFLINE", "1")
  monkeypatch.setenv("HERMES_INGEST_OUT", str(out))
  monkeypatch.delenv("HERMES_INGEST_CONFIG", raising=False)
  return out


@pytest.fixture
def use_config(
  env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[Any], Path]:
  def apply(config: Any) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_INGEST_CONFIG", str(path))
    return path

  return apply


@pytest.fixture
def shared(use_config: Callable[[Any], Path]) -> dict[str, Any]:
  config = copy.deepcopy(SHARED)
  use_config(config)
  return config


# ------------------------------------------------------------ the stored files


class TestStoredNotebook:
  def test_it_has_no_outputs_and_no_execution_counts(self) -> None:
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [c for c in raw["cells"] if c["cell_type"] == "code"]
    assert code
    for cell in code:
      assert cell["outputs"] == [], cell["id"]
      assert cell["execution_count"] is None, cell["id"]
    assert "image/png" not in NOTEBOOK.read_text(encoding="utf-8")

  def test_it_is_valid_nbformat_with_unique_cell_ids(self) -> None:
    notebook = load()
    nbformat.validate(notebook)
    ids = [c["id"] for c in notebook.cells]
    assert len(ids) == len(set(ids))
    # The ids the tests (and the task file) rely on.
    for needed in (
      "s1-code",
      "s2-code",
      "s3-code",
      "s4-code",
      "s5-code",
      "s6-code",
      "s7-code",
      "s7-documents",
      "s7-unused",
      "s8-code",
      "s9-code",
    ):
      assert needed in ids, needed

  def test_it_has_the_nine_sections_in_order(self) -> None:
    titles = [
      line
      for c in load().cells
      if c.cell_type == "markdown"
      for line in c.source.splitlines()
      if line.startswith("## ")
    ]
    assert [t.split(".")[0] for t in titles] == [f"## {n}" for n in range(1, 10)]

  def test_every_code_cell_starts_with_a_comment(self) -> None:
    for cell in load().cells:
      if cell.cell_type == "code":
        assert cell.source.lstrip().startswith("#"), cell.get("id")

  def test_markdown_is_spanish_and_explains_how_to_change_the_config(self) -> None:
    text = "\n".join(c.source for c in load().cells if c.cell_type == "markdown")
    assert "CONFIG" in text
    assert "HERMES_INGEST_OFFLINE" in text
    assert "rag_evaluation.ipynb" in text
    assert re.search(r"\b(Ejecuta|colección|documentos|Antes|Después)\b", text)

  def test_it_writes_only_through_ensure_collection(self) -> None:
    """Static guard: no direct write call and no chunking or embedding by itself."""
    source = "\n".join(c.source for c in load().cells if c.cell_type == "code")
    for call in WRITE_CALLS:
      assert f".{call}(" not in source, call
    assert source.count("ensure_collection(") == 1
    assert source.count("ingest_corpus(") == 1
    assert "hermes_ingest_md" not in source
    assert "MarkdownStructuralChunker" not in source
    assert "get_text_embedding" not in source

  def test_the_default_config_has_the_shape_of_the_task(self) -> None:
    """Run only the CONFIG cell, online and without a JSON override."""
    cell = next(c for c in load().cells if c.get("id") == "s2-code")
    scope: dict[str, Any] = {
      "os": os,
      "json": json,
      "Path": Path,
      "PREFIX": "hermes_report_",
      "ROOT": ROOT,
      "OFFLINE": False,
      "display": lambda value: None,
      "Any": Any,
    }
    saved = os.environ.pop("HERMES_INGEST_CONFIG", None)
    try:
      exec(cell.source, scope)  # noqa: S102 - the repository's own notebook cell
    finally:
      if saved is not None:
        os.environ["HERMES_INGEST_CONFIG"] = saved
    config = scope["CONFIG"]
    assert config["corpus_dir"] == "notebooks/data"
    assert config["qdrant_url"] == "http://localhost:6333"
    assert config["force"] is False
    assert config["defaults"]["ingest"] == {
      "max_chunk_size": 1500,
      "hard_boundary_levels": [1, 2],
      "strip_frontmatter": True,
    }
    assert config["defaults"]["embedding"] == {
      "provider": "ollama",
      "model": "nomic-embed-text",
    }
    agents = config["collections"]["hermes_report_golden_agents"]
    assert agents["documents"] == ["agentes-ia-arquitecturas.md"]
    assert agents["ingest"] == {"max_chunk_size": 800}
    python = config["collections"]["hermes_report_golden_python"]
    assert python["documents"] == [
      "fastapi-readme.md",
      "flask-readme.md",
      "pydantic-readme.md",
      "typer-readme.md",
    ]


class TestReadme:
  def section(self) -> str:
    text = README.read_text(encoding="utf-8")
    assert "## Cuaderno de ingesta" in text
    body = text.split("## Cuaderno de ingesta", 1)[1]
    return body.split("\n## ", 1)[0]

  def test_it_documents_the_notebook_and_its_variables(self) -> None:
    body = self.section()
    for needle in (
      "ingest_collections.ipynb",
      "HERMES_INGEST_CONFIG",
      "HERMES_INGEST_OUT",
      "HERMES_INGEST_OFFLINE",
      "rag_evaluation.ipynb",
      "collections.json",
      "hermes_report_",
      "force",
    ):
      assert needle in body, needle

  def test_its_config_example_is_valid_python_with_the_notebook_keys(self) -> None:
    body = self.section()
    block = re.search(r"```python\n(.*?)```", body, re.S)
    assert block
    tree = ast.parse(block.group(1))
    value = ast.literal_eval(tree.body[0].value)  # type: ignore[attr-defined]
    assert set(value) <= {
      "corpus_dir",
      "qdrant_url",
      "force",
      "defaults",
      "collections",
    }
    for name, spec in value["collections"].items():
      assert name.startswith("hermes_report_")
      assert spec["documents"]


# --------------------------------------------------- first offline pass, feedback


class TestFirstRun:
  def run(self) -> nbformat.NotebookNode:
    notebook = load()
    insert_after(notebook, "s9-code", DUMP, "t-dump")
    return execute(notebook)

  def test_collections_hold_the_expected_documents_and_chunk_sizes_differ(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    executed = self.run()
    no_cell_failed(executed)
    state = dumps(executed, "t-dump")
    assert state["names"] == ["hermes_report_t_big", "hermes_report_t_small"]
    assert set(state["sources"]["hermes_report_t_small"]) == SMALL_DOCS
    assert set(state["sources"]["hermes_report_t_big"]) == BIG_DOCS
    # The smaller max_chunk_size makes more points even for the shared document.
    assert (
      state["counts"]["hermes_report_t_small"] > state["counts"]["hermes_report_t_big"]
    )
    assert state["outcomes"] == {
      "hermes_report_t_small": "creada",
      "hermes_report_t_big": "creada",
    }

  def test_manifest_matches_what_was_loaded(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    executed = self.run()
    state = dumps(executed, "t-dump")
    manifest = json.loads((env / "collections.json").read_text(encoding="utf-8"))
    assert set(manifest["collections"]) == set(shared["collections"])
    assert manifest["qdrant_url"] == ":memory:"  # offline overrides the URL
    assert manifest["force"] is False
    assert manifest["timestamp"]
    for name, entry in manifest["collections"].items():
      assert entry["points"] == state["counts"][name]
      assert entry["outcome"] == "creada"
      assert entry["error"] is None
      assert entry["qdrant_url"] == ":memory:"
      assert entry["timestamp"]
      assert {d["source"] for d in entry["documents"]} == set(state["sources"][name])
      for doc in entry["documents"]:
        digest = hashlib.sha256((ROOT / DEMO / doc["source"]).read_bytes()).hexdigest()
        assert doc["sha256"] == digest, doc["source"]
        assert doc["chunks"] > 0
      assert sum(d["chunks"] for d in entry["documents"]) == entry["points"]
      assert entry["embedding"]["provider"] == "hash"
      assert entry["embedding"]["dimension"] == 256
    small = manifest["collections"]["hermes_report_t_small"]["ingest"]
    big = manifest["collections"]["hermes_report_t_big"]["ingest"]
    assert small["max_chunk_size"] == 200
    assert big["max_chunk_size"] == 2000
    assert small["hard_boundary_levels"] == [1, 2]  # merged over the defaults
    assert small["strip_frontmatter"] is True
    assert manifest["before"] == []
    assert [r["colección"] for r in manifest["after"]] == [
      "hermes_report_t_big",
      "hermes_report_t_small",
    ]
    assert sorted(state["ready"]) == sorted(manifest["collections"])
    assert "Para rag_evaluation.ipynb" in cell_text(executed, "s9-code")
    assert json.dumps(state["ready"]) in cell_text(executed, "s9-code")

  def test_antes_is_empty_and_announces_the_collections_to_create(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    executed = self.run()
    state = dumps(executed, "t-dump")
    assert state["before"] == []
    text = cell_text(executed, "s5-code")
    for column in ("colección", "puntos", "tamaño vector", "distancia", "acción"):
      assert column in text, column
    assert "Colecciones que se crearán" in text
    assert "hermes_report_t_small" in text
    assert "hermes_report_t_big" in text

  def test_despues_and_diff_say_new_with_the_points(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    executed = self.run()
    state = dumps(executed, "t-dump")
    after = {r["colección"]: r for r in state["after"]}
    assert set(after) == set(shared["collections"])
    for name, row in after.items():
      assert row["puntos"] == state["counts"][name]
      assert row["tamaño vector"] == 256
      assert row["distancia"] == "Cosine"
      assert row["configurada"] == "sí"
      assert row["acción"] == "creada"
    diff = {r["colección"]: r for r in state["diff"]}
    for name, row in diff.items():
      assert row["estado"] == "nueva"
      assert row["puntos antes"] == 0
      assert row["puntos después"] == state["counts"][name]
    text = cell_text(executed, "s8-code")
    assert "nueva" in text
    assert "reutilizada" not in text
    assert "reconstruida" not in text

  def test_totals(self, shared: dict[str, Any], env: Path) -> None:
    state = dumps(self.run(), "t-dump")
    totals = {r["total"]: r["valor"] for r in state["totals"]}
    points = sum(state["counts"].values())
    assert totals["colecciones antes → después"] == "0 → 2"
    assert totals["puntos antes → después"] == f"0 → {points}"
    assert totals["documentos ingeridos"] == 3  # faro is ingested once per collection
    assert totals["chunks calculados (embeddings)"] == points
    assert totals["chunks leídos de la caché"] == 0
    assert totals["segundos totales"] >= 0

  def test_stats_tables_list_the_expected_collections_and_counts(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    executed = self.run()
    state = dumps(executed, "t-dump")
    stats = {r["colección"]: r for r in state["collection_stats"]}
    assert set(stats) == set(shared["collections"])
    assert stats["hermes_report_t_small"]["documentos"] == 2
    assert stats["hermes_report_t_big"]["documentos"] == 1
    for name, row in stats.items():
      assert row["chunks"] == state["counts"][name]
      assert row["vector"] == 256
      assert row["embedder"] == "hash:hash-256"
      assert row["resultado"] == "creada"
      assert row["caché/ingeridos"].startswith("0/")
      for column in (
        "chunks/doc (mín/media/máx)",
        "caracteres (mín/media/mediana/máx)",
        "tokens (mín/media/mediana/máx)",
      ):
        numbers = [int(x) for x in row[column].split(" / ")]
        low, high = numbers[0], numbers[-1]
        assert 1 <= low <= high, (name, column)
        assert all(low <= n <= high for n in numbers), (name, column)
    small = stats["hermes_report_t_small"]["caracteres (mín/media/mediana/máx)"]
    assert int(small.split(" / ")[-1]) <= 200  # the chunker honours max_chunk_size
    assert "máx 200" in stats["hermes_report_t_small"]["troceado"]
    assert "máx 2000" in stats["hermes_report_t_big"]["troceado"]

    documents = state["document_stats"]
    assert {(r["colección"], r["documento"]) for r in documents} == {
      ("hermes_report_t_small", "manuales/faro.md"),
      ("hermes_report_t_small", "manuales/brujula.md"),
      ("hermes_report_t_big", "manuales/faro.md"),
    }
    for row in documents:
      assert row["chunks"] > 0
      assert row["caracteres"] == len(
        (ROOT / DEMO / row["documento"]).read_text("utf-8")
      )
      assert row["chunk medio"] > 0
      assert re.fullmatch(r"\d+-\d+", row["profundidad encabezados"])
      assert row["origen"] == "ingerido"
    per_collection = {
      name: sum(r["chunks"] for r in documents if r["colección"] == name)
      for name in stats
    }
    assert per_collection == {n: stats[n]["chunks"] for n in stats}

  def test_unused_documents_are_listed(self, shared: dict[str, Any], env: Path) -> None:
    executed = self.run()
    state = dumps(executed, "t-dump")
    assert {r["documento"] for r in state["unused"]} == (
      ALL_DOCS - SMALL_DOCS - BIG_DOCS
    )
    assert "politicas/teletrabajo.md" in cell_text(executed, "s3-code")
    assert "politicas/teletrabajo.md" in cell_text(executed, "s7-unused")

  def test_all_documents_used_says_so(
    self, shared: dict[str, Any], use_config: Callable[[Any], Path]
  ) -> None:
    config = copy.deepcopy(shared)
    config["collections"]["hermes_report_t_big"]["documents"] = sorted(ALL_DOCS)
    use_config(config)
    executed = self.run()
    state = dumps(executed, "t-dump")
    assert state["unused"] == []
    assert "Todos los documentos" in cell_text(executed, "s3-code")
    assert "Todos los documentos" in cell_text(executed, "s7-unused")

  def test_one_progress_line_per_document_under_a_header_per_collection(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    text = cell_text(self.run(), "s6-code")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    headers = [ln for ln in lines if re.match(r"^\[\d+/\d+\] ", ln)]
    progress = [ln for ln in lines if ln.startswith("  - ")]
    closing = [ln for ln in lines if ln.startswith("  -> ")]
    assert headers == [
      "[1/2] hermes_report_t_small: 2 documentos",
      "[2/2] hermes_report_t_big: 1 documento",
    ]
    assert len(progress) == 3
    pattern = re.compile(r"^  - (\S+): (\d+) chunks \((caché|ingerido)\), \d+\.\d\d s$")
    sources = []
    for line in progress:
      match = pattern.match(line)
      assert match, line
      assert int(match.group(2)) > 0
      assert match.group(3) == "ingerido"
      sources.append(match.group(1))
    assert sorted(sources) == sorted(
      ["manuales/faro.md", "manuales/brujula.md", "manuales/faro.md"]
    )
    assert len(closing) == 2
    for line in closing:
      assert re.match(r"^  -> creada: \d+ puntos, \d+\.\d\d s$", line), line
    # Order: header, its documents, its closing line.
    positions = [lines.index(h) for h in headers]
    assert lines[positions[0] + 1].startswith("  - ")
    assert lines[positions[1] - 1].startswith("  -> ")
    assert lines[-1].startswith("Total: ")

  def test_the_repository_stays_clean_and_the_cache_is_in_the_out_folder(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    existed = (ROOT / "notebooks" / "out").exists()
    executed = self.run()
    assert (ROOT / "notebooks" / "out").exists() == existed
    assert (env / "ingest-cache").is_dir()
    assert any((env / "ingest-cache").iterdir())
    for cell in executed.cells:
      if cell.cell_type == "code" and cell.get("id", "").startswith("s"):
        assert cell.outputs, f"{cell.get('id')} shows nothing"

  def test_default_offline_run_without_a_json_uses_two_demo_collections(
    self, env: Path
  ) -> None:
    executed = self.run()
    no_cell_failed(executed)
    state = dumps(executed, "t-dump")
    assert state["names"] == ["hermes_report_demo_manuales", "hermes_report_demo_todo"]
    assert state["counts"]["hermes_report_demo_manuales"] > 0
    assert "manuales/faro.md" in state["sources"]["hermes_report_demo_todo"]

  def test_a_json_that_asks_for_ollama_still_runs_offline_with_hash(
    self, use_config: Callable[[Any], Path], env: Path
  ) -> None:
    use_config(
      {
        "corpus_dir": DEMO,
        "defaults": {"embedding": {"provider": "ollama", "model": "nomic-embed-text"}},
        "collections": {
          "hermes_report_t_one": {
            "documents": ["manuales/faro.md"],
            "embedding": {"provider": "ollama", "model": "other-model"},
          },
          "hermes_report_t_two": {"documents": ["manuales/brujula.md"]},
        },
      }
    )
    executed = self.run()
    no_cell_failed(executed)
    state = dumps(executed, "t-dump")
    assert {r["embedder"] for r in state["collection_stats"]} == {"hash:hash-256"}

  def test_a_json_without_defaults_uses_the_package_defaults(
    self, use_config: Callable[[Any], Path], env: Path
  ) -> None:
    use_config(
      {
        "corpus_dir": DEMO,
        "collections": {"hermes_report_t_one": {"documents": ["manuales/faro.md"]}},
      }
    )
    executed = self.run()
    manifest = json.loads((env / "collections.json").read_text(encoding="utf-8"))
    ingest = manifest["collections"]["hermes_report_t_one"]["ingest"]
    assert ingest == {
      "max_chunk_size": 1500,
      "hard_boundary_levels": [1, 2],
      "strip_frontmatter": True,
    }
    no_cell_failed(executed)


# ----------------------------------------------- second pass in the same kernel


class TestSecondPass:
  def test_same_config_reuses_both_collections(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    executed = execute(with_second_pass(load()))
    no_cell_failed(executed)
    first = dumps(executed, "t-dump-1")
    second = dumps(executed, "t-dump-2")
    assert second["outcomes"] == {
      "hermes_report_t_small": "reutilizada",
      "hermes_report_t_big": "reutilizada",
    }
    assert second["counts"] == first["counts"]
    assert second["sources"] == first["sources"]
    # "Antes" now sees the collections, with the plan to check them for reuse.
    before = {r["colección"]: r for r in second["before"]}
    assert set(before) == set(shared["collections"])
    for name, row in before.items():
      assert row["puntos"] == first["counts"][name]
      assert row["configurada"] == "sí"
      assert row["acción"] == "comprobar si se reutiliza"
    for row in second["diff"]:
      assert row["estado"] == "reutilizada sin cambios"
      assert row["puntos antes"] == row["puntos después"]
    assert {r["resultado"] for r in second["collection_stats"]} == {"reutilizada"}
    # The per-document cache served the second pass: nothing was embedded again.
    totals = {r["total"]: r["valor"] for r in second["totals"]}
    assert totals["chunks calculados (embeddings)"] == 0
    assert totals["chunks leídos de la caché"] == sum(first["counts"].values())
    progress = cell_text(executed, "r2-s6-code")
    assert "(caché)" in progress
    assert "(ingerido)" not in progress
    assert progress.count("  -> reutilizada:") == 2
    assert "creada" not in progress
    assert second["ready"] == first["ready"]
    manifest = json.loads((env / "collections.json").read_text(encoding="utf-8"))
    assert {e["outcome"] for e in manifest["collections"].values()} == {"reutilizada"}

  def test_force_rebuilds_both_collections_with_the_same_points(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    notebook = with_second_pass(load(), between='CONFIG["force"] = True\n')
    executed = execute(notebook)
    no_cell_failed(executed)
    first = dumps(executed, "t-dump-1")
    second = dumps(executed, "t-dump-2")
    assert second["force"] is True
    assert second["outcomes"] == {
      "hermes_report_t_small": "recreada",
      "hermes_report_t_big": "recreada",
    }
    assert second["counts"] == first["counts"]
    before = {r["colección"]: r for r in second["before"]}
    assert {r["acción"] for r in before.values()} == {"reconstruir (force)"}
    for row in second["diff"]:
      assert row["estado"] == "reconstruida"
      assert row["detalle"] == (
        f"{first['counts'][row['colección']]} → {row['puntos después']} puntos"
      )
    assert {r["resultado"] for r in second["collection_stats"]} == {"recreada"}
    assert cell_text(executed, "r2-s6-code").count("  -> recreada:") == 2
    manifest = json.loads((env / "collections.json").read_text(encoding="utf-8"))
    assert manifest["force"] is True
    assert {e["outcome"] for e in manifest["collections"].values()} == {"recreada"}

  def test_force_on_the_first_pass_creates_instead_of_claiming_a_rebuild(
    self, shared: dict[str, Any], use_config: Callable[[Any], Path]
  ) -> None:
    """Implied: ``force`` on an absent collection is a creation, not a rebuild."""
    config = copy.deepcopy(shared)
    config["force"] = True
    use_config(config)
    notebook = load()
    insert_after(notebook, "s9-code", DUMP, "t-dump")
    state = dumps(execute(notebook), "t-dump")
    assert set(state["outcomes"].values()) == {"creada"}
    assert {r["estado"] for r in state["diff"]} == {"nueva"}

  def test_changed_chunk_size_rebuilds_instead_of_reusing(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    """Implied by "reuse only when it holds exactly the same points"."""
    between = (
      'CONFIG["collections"]["hermes_report_t_big"]["ingest"] = '
      '{"max_chunk_size": 400}\n'
    )
    notebook = with_second_pass(
      load(),
      between=between,
      rerun=("s3-code", "s5-code", "s6-code", "s7-code", "s7-documents", "s7-unused"),
    )
    executed = execute(notebook)
    no_cell_failed(executed)
    first = dumps(executed, "t-dump-1")
    second = dumps(executed, "t-dump-2")
    assert second["outcomes"] == {
      "hermes_report_t_small": "reutilizada",
      "hermes_report_t_big": "recreada",
    }
    big = "hermes_report_t_big"
    assert second["counts"][big] > first["counts"][big]  # 400 < 2000
    diff = {r["colección"]: r for r in second["diff"]}
    assert diff[big]["estado"] == "reconstruida"
    assert diff[big]["puntos antes"] == first["counts"][big]
    assert diff[big]["puntos después"] == second["counts"][big]
    assert (
      diff[big]["detalle"] == f"{first['counts'][big]} → {second['counts'][big]} puntos"
    )
    assert diff["hermes_report_t_small"]["estado"] == "reutilizada sin cambios"
    manifest = json.loads((env / "collections.json").read_text(encoding="utf-8"))
    assert manifest["collections"][big]["ingest"]["max_chunk_size"] == 400

  def test_a_changed_document_rebuilds_its_collections(
    self, tmp_path: Path, use_config: Callable[[Any], Path], env: Path
  ) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nPrimer texto.\n", encoding="utf-8")
    (corpus / "b.md").write_text("# B\n\nOtro texto.\n", encoding="utf-8")
    use_config(
      {
        "corpus_dir": str(corpus),
        "collections": {
          "hermes_report_t_a": {"documents": ["a.md"]},
          "hermes_report_t_b": {"documents": ["b.md"]},
        },
      }
    )
    between = (
      f"Path({str(corpus / 'a.md')!r}).write_text("
      "'# A\\n\\nTexto distinto y mas largo.\\n', encoding='utf-8')\n"
    )
    notebook = with_second_pass(
      load(),
      between=between,
      rerun=("s3-code", "s5-code", "s6-code", "s7-code", "s7-documents", "s7-unused"),
    )
    executed = execute(notebook)
    second = dumps(executed, "t-dump-2")
    assert second["outcomes"] == {
      "hermes_report_t_a": "recreada",
      "hermes_report_t_b": "reutilizada",
    }
    manifest = json.loads((env / "collections.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((corpus / "a.md").read_bytes()).hexdigest()
    assert (
      manifest["collections"]["hermes_report_t_a"]["documents"][0]["sha256"] == digest
    )

  def test_a_second_kernel_with_the_same_out_folder_reads_the_disk_cache(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    notebook = load()
    insert_after(notebook, "s9-code", DUMP, "t-dump")
    first = dumps(execute(notebook), "t-dump")
    notebook = load()
    insert_after(notebook, "s9-code", DUMP, "t-dump")
    executed = execute(notebook)
    second = dumps(executed, "t-dump")
    assert "(ingerido)" not in cell_text(executed, "s6-code")
    assert cell_text(executed, "s6-code").count("(caché)") == 3
    totals = {r["total"]: r["valor"] for r in second["totals"]}
    assert totals["chunks calculados (embeddings)"] == 0
    assert totals["chunks leídos de la caché"] == sum(first["counts"].values())
    stats = {r["colección"]: r for r in second["collection_stats"]}
    assert stats["hermes_report_t_small"]["caché/ingeridos"] == "2/0"
    assert {r["origen"] for r in second["document_stats"]} == {"caché"}
    # A fresh in-memory Qdrant: created again, same points.
    assert second["counts"] == first["counts"]
    assert set(second["outcomes"].values()) == {"creada"}


# ----------------------------------------------------------- validation errors

BASE_COLLECTIONS = {"hermes_report_t_ok": {"documents": ["manuales/faro.md"]}}


def cfg(**changes: Any) -> dict[str, Any]:
  config: dict[str, Any] = {
    "corpus_dir": DEMO,
    "defaults": {"ingest": {"max_chunk_size": 500}},
    "collections": copy.deepcopy(BASE_COLLECTIONS),
  }
  config.update(changes)
  return config


def col(**changes: Any) -> dict[str, Any]:
  spec: dict[str, Any] = {"documents": ["manuales/faro.md"]}
  spec.update(changes)
  return {"hermes_report_t_ok": spec}


VALIDATION_CASES: list[tuple[str, dict[str, Any], str]] = [
  (
    "unknown-document",
    cfg(collections=col(documents=["manuales/faro.md", "no-existe.md"])),
    "no-existe.md",
  ),
  (
    "no-prefix",
    cfg(collections={"golden": {"documents": ["manuales/faro.md"]}}),
    "hermes_report_",
  ),
  (
    "upper-case-prefix",
    cfg(collections={"Hermes_Report_x": {"documents": ["manuales/faro.md"]}}),
    "hermes_report_",
  ),
  ("empty-documents", cfg(collections=col(documents=[])), "documents"),
  ("missing-documents", cfg(collections={"hermes_report_t_ok": {}}), "documents"),
  (
    "documents-not-a-list",
    cfg(collections=col(documents="manuales/faro.md")),
    "documents",
  ),
  ("documents-with-numbers", cfg(collections=col(documents=[1, 2])), "documents"),
  (
    "repeated-document",
    cfg(collections=col(documents=["manuales/faro.md", "manuales/faro.md"])),
    "repetidos",
  ),
  ("spec-not-an-object", cfg(collections={"hermes_report_t_ok": ["a.md"]}), "objeto"),
  ("no-collections", cfg(collections={}), "al menos una colección"),
  ("missing-collections-key", {"corpus_dir": DEMO}, "al menos una colección"),
  (
    "unknown-chunker-key-in-collection",
    cfg(collections=col(ingest={"max_chunk_sise": 10})),
    "max_chunk_sise",
  ),
  (
    "unknown-chunker-key-in-defaults",
    cfg(defaults={"ingest": {"overlap": 10}}),
    "overlap",
  ),
  (
    "unknown-embedding-key",
    cfg(collections=col(embedding={"modelo": "x"})),
    "modelo",
  ),
  (
    "unknown-embedding-provider",
    cfg(defaults={"embedding": {"provider": "openai"}}),
    "provider",
  ),
  ("unknown-section-in-defaults", cfg(defaults={"retrieval": {}}), "retrieval"),
  ("unknown-top-key", cfg(colections={}), "colections"),
  (
    "zero-chunk-size",
    cfg(collections=col(ingest={"max_chunk_size": 0})),
    "max_chunk_size",
  ),
  (
    "negative-chunk-size",
    cfg(collections=col(ingest={"max_chunk_size": -5})),
    "max_chunk_size",
  ),
  (
    "boolean-chunk-size",
    cfg(collections=col(ingest={"max_chunk_size": True})),
    "max_chunk_size",
  ),
  (
    "string-chunk-size",
    cfg(collections=col(ingest={"max_chunk_size": "800"})),
    "max_chunk_size",
  ),
  (
    "float-chunk-size",
    cfg(collections=col(ingest={"max_chunk_size": 1.5})),
    "max_chunk_size",
  ),
  (
    "level-zero",
    cfg(collections=col(ingest={"hard_boundary_levels": [0]})),
    "hard_boundary_levels",
  ),
  (
    "level-seven",
    cfg(collections=col(ingest={"hard_boundary_levels": [7]})),
    "hard_boundary_levels",
  ),
  (
    "levels-not-a-list",
    cfg(collections=col(ingest={"hard_boundary_levels": "12"})),
    "hard_boundary_levels",
  ),
  (
    "levels-with-a-boolean",
    cfg(collections=col(ingest={"hard_boundary_levels": [True]})),
    "hard_boundary_levels",
  ),
  (
    "strip-frontmatter-text",
    cfg(collections=col(ingest={"strip_frontmatter": "yes"})),
    "strip_frontmatter",
  ),
  ("force-text", cfg(force="yes"), "force"),
  ("missing-corpus-dir", cfg(corpus_dir="no/such/folder"), "no/such/folder"),
  ("blank-corpus-dir", cfg(corpus_dir=" "), "corpus_dir"),
  ("corpus-dir-not-text", cfg(corpus_dir=3), "corpus_dir"),
  (
    "document-with-a-dot-prefix",
    cfg(collections=col(documents=["./manuales/faro.md"])),
    "./manuales/faro.md",
  ),
  (
    "document-with-other-case",
    cfg(collections=col(documents=["Manuales/Faro.md"])),
    "Manuales/Faro.md",
  ),
  (
    "document-outside-the-folder",
    cfg(collections=col(documents=["../corpus/manuales/faro.md"])),
    "../corpus/manuales/faro.md",
  ),
  ("blank-model", cfg(defaults={"embedding": {"model": "  "}}), "model"),
  ("defaults-not-an-object", cfg(defaults=["x"]), "defaults"),
]


# Offline mode replaces every embedding section by the hash embedder, so the
# embedding checks are only exercised online (validation still runs before any
# service is touched).
ONLINE_ONLY = {"unknown-embedding-key", "unknown-embedding-provider", "blank-model"}


class TestValidation:
  @pytest.mark.parametrize(
    ("config", "needle"),
    [pytest.param(c, n, id=i) for i, c, n in VALIDATION_CASES],
  )
  def test_invalid_config_fails_in_the_validation_cell_before_any_ingestion(
    self,
    use_config: Callable[[Any], Path],
    env: Path,
    config: dict[str, Any],
    needle: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    if request.node.callspec.id in ONLINE_ONLY:
      monkeypatch.delenv("HERMES_INGEST_OFFLINE")
    use_config(config)
    notebook = load()
    with pytest.raises(CellExecutionError) as caught:
      execute(notebook)
    assert caught.value.ename == "ValueError"
    assert needle in str(caught.value)
    assert "no se ha tocado Qdrant ni Ollama" in str(caught.value)
    failing = [
      c.id
      for c in notebook.cells
      if c.cell_type == "code" and any(o.output_type == "error" for o in c.outputs)
    ]
    assert failing == ["s3-code"]
    # No later cell ran: no service, no ingestion, no cache, no manifest.
    for later in ("s4-code", "s5-code", "s6-code", "s8-code", "s9-code"):
      assert notebook.cells[cell_index(notebook, later)].outputs == []
    assert not (env / "ingest-cache").exists()
    assert not (env / "collections.json").exists()

  def test_every_problem_is_reported_at_once_in_spanish(
    self, use_config: Callable[[Any], Path], env: Path
  ) -> None:
    use_config(
      {
        "corpus_dir": DEMO,
        "collections": {
          "golden": {"documents": ["x.md"], "ingest": {"zzz": 1}},
          "hermes_report_t_ok": {"documents": []},
        },
      }
    )
    with pytest.raises(CellExecutionError) as caught:
      execute(load())
    message = str(caught.value)
    for needle in ("golden", "x.md", "zzz", "documents", "disponibles"):
      assert needle in message, needle
    # The available names are listed so the user can fix the typo.
    for name in ALL_DOCS:
      assert name in message, name

  def test_validation_never_reaches_the_ingestion_even_if_it_is_sabotaged(
    self, use_config: Callable[[Any], Path], env: Path
  ) -> None:
    """A tripwire in place of ``ingest_corpus``/``make_client`` is never called."""
    use_config(cfg(collections=col(documents=["no-existe.md"])))
    notebook = load()
    insert_after(
      notebook,
      "s2-code",
      "def _boom(*a, **k):\n  raise SystemExit('REACHED')\n"
      "ingest_corpus = make_client = ensure_collection = _boom\n",
      "t-tripwire",
    )
    with pytest.raises(CellExecutionError) as caught:
      execute(notebook)
    assert "REACHED" not in str(caught.value)
    assert caught.value.ename == "ValueError"

  def test_a_valid_config_with_nulls_is_accepted(
    self, use_config: Callable[[Any], Path], env: Path
  ) -> None:
    use_config(
      {
        "corpus_dir": DEMO,
        "defaults": None,
        "collections": {
          "hermes_report_t_ok": {"documents": ["manuales/faro.md"], "ingest": None}
        },
      }
    )
    executed = execute(load())
    no_cell_failed(executed)

  def test_the_json_override_must_be_a_readable_json_object(
    self, env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
  ) -> None:
    missing = tmp_path / "nope.json"
    monkeypatch.setenv("HERMES_INGEST_CONFIG", str(missing))
    with pytest.raises(CellExecutionError, match="HERMES_INGEST_CONFIG"):
      execute(load())
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("HERMES_INGEST_CONFIG", str(broken))
    with pytest.raises(CellExecutionError, match="HERMES_INGEST_CONFIG"):
      execute(load())
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("HERMES_INGEST_CONFIG", str(array))
    with pytest.raises(CellExecutionError, match="objeto"):
      execute(load())
    assert not (env / "collections.json").exists()


# ----------------------------------------------------------- failures and safety


class TestFailureAndSafety:
  def test_a_failing_collection_is_reported_and_the_others_continue(
    self, tmp_path: Path, use_config: Callable[[Any], Path], env: Path
  ) -> None:
    """A real failure: a file without chunks makes ``ensure_collection`` refuse."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "good.md").write_text("# Bueno\n\nTexto util.\n", encoding="utf-8")
    (corpus / "empty.md").write_text("", encoding="utf-8")
    use_config(
      {
        "corpus_dir": str(corpus),
        "collections": {
          "hermes_report_t_bad": {"documents": ["empty.md"]},
          "hermes_report_t_good": {"documents": ["good.md"]},
        },
      }
    )
    notebook = load()
    insert_after(notebook, "s9-code", DUMP, "t-dump")
    executed = execute(notebook)
    no_cell_failed(executed)
    state = dumps(executed, "t-dump")
    assert state["outcomes"] == {
      "hermes_report_t_bad": "fallida",
      "hermes_report_t_good": "creada",
    }
    assert state["names"] == ["hermes_report_t_good"]
    assert state["ready"] == ["hermes_report_t_good"]
    progress = cell_text(executed, "s6-code")
    assert "ERROR" in progress
    assert "[2/2] hermes_report_t_good" in progress
    assert state["errors"]["hermes_report_t_bad"]
    diff = {r["colección"]: r for r in state["diff"]}
    assert diff["hermes_report_t_bad"]["estado"] == "fallida"
    assert (
      diff["hermes_report_t_bad"]["detalle"] == state["errors"]["hermes_report_t_bad"]
    )
    assert diff["hermes_report_t_good"]["estado"] == "nueva"
    manifest = json.loads((env / "collections.json").read_text(encoding="utf-8"))
    bad = manifest["collections"]["hermes_report_t_bad"]
    assert bad["outcome"] == "fallida"
    assert bad["error"]
    assert bad["points"] is None
    assert manifest["collections"]["hermes_report_t_good"]["outcome"] == "creada"
    stats = {r["colección"]: r for r in state["collection_stats"]}
    assert stats["hermes_report_t_bad"]["resultado"] == "fallida"
    assert {r["colección"] for r in state["document_stats"]} == {"hermes_report_t_good"}

  def test_every_collection_failing_still_closes_the_notebook(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    notebook = load()
    insert_after(
      notebook,
      "s4-code",
      "def _fail(*a, **k):\n  raise RuntimeError('sin servicio')\n"
      "ingest_corpus = _fail\n",
      "t-fail",
    )
    insert_after(notebook, "s9-code", DUMP, "t-dump")
    executed = execute(notebook)
    no_cell_failed(executed)
    state = dumps(executed, "t-dump")
    assert set(state["outcomes"].values()) == {"fallida"}
    assert state["ready"] == []
    assert state["names"] == []
    assert state["document_stats"] == []
    assert "sin servicio" in cell_text(executed, "s6-code")
    manifest = json.loads((env / "collections.json").read_text(encoding="utf-8"))
    assert {e["outcome"] for e in manifest["collections"].values()} == {"fallida"}

  def test_a_failed_rebuild_keeps_the_old_collection_and_says_so(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    """Second pass with a failing embed: the first pass's points stay in Qdrant."""
    between = (
      "def _fail(*a, **k):\n  raise RuntimeError('sin servicio')\n"
      "ingest_corpus = _fail\n"
    )
    executed = execute(with_second_pass(load(), between=between))
    no_cell_failed(executed)
    first = dumps(executed, "t-dump-1")
    second = dumps(executed, "t-dump-2")
    assert set(second["outcomes"].values()) == {"fallida"}
    assert second["counts"] == first["counts"]
    for row in second["diff"]:
      assert row["estado"] == "fallida"
      assert row["puntos antes"] == row["puntos después"]

  def test_only_prefixed_collections_are_ever_written_and_foreign_ones_untouched(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    notebook = load()
    insert_after(notebook, "s4-code", SEED, "t-seed")
    insert_after(notebook, "t-seed", SPY, "t-spy")
    notebook = with_second_pass(notebook, between='CONFIG["force"] = True\n')
    executed = execute(notebook)
    no_cell_failed(executed)
    first = dumps(executed, "t-dump-1")
    second = dumps(executed, "t-dump-2")
    for state in (first, second):
      assert state["counts"]["other_data"] == 1
      assert state["counts"]["hermes_report_stray"] == 1
      assert state["sources"]["other_data"] == ["x.md"]
    writes = second["writes"]
    assert writes
    # Pass one creates and fills; pass two (force) rebuilds: both only touch ours.
    assert {target for _, target in writes} == set(shared["collections"])
    assert all(str(t).startswith("hermes_report_") for _, t in writes)

  def test_antes_and_despues_show_foreign_and_unconfigured_collections(
    self, shared: dict[str, Any], env: Path
  ) -> None:
    notebook = load()
    insert_after(notebook, "s4-code", SEED, "t-seed")
    insert_after(notebook, "t-seed", SPY, "t-spy")
    insert_after(notebook, "s9-code", DUMP, "t-dump")
    executed = execute(notebook)
    no_cell_failed(executed)
    state = dumps(executed, "t-dump")
    before = {r["colección"]: r for r in state["before"]}
    assert set(before) == {"other_data", "hermes_report_stray"}
    assert before["other_data"]["configurada"] == "no"
    assert before["other_data"]["acción"] == "nunca se toca"
    assert before["other_data"]["puntos"] == 1
    assert before["other_data"]["tamaño vector"] == 256
    assert before["other_data"]["distancia"] == "Cosine"
    assert before["hermes_report_stray"]["configurada"] == "no"
    assert before["hermes_report_stray"]["acción"] == "sin tocar (no configurada)"
    after = {r["colección"]: r for r in state["after"]}
    assert set(after) == {"other_data", "hermes_report_stray", *shared["collections"]}
    assert after["other_data"]["acción"] == "nunca se toca"
    diff = {r["colección"]: r for r in state["diff"]}
    for name in ("other_data", "hermes_report_stray"):
      assert diff[name]["estado"] == "sin tocar"
      assert diff[name]["puntos antes"] == diff[name]["puntos después"] == 1
      assert "han cambiado" not in diff[name]["detalle"]
    for name in shared["collections"]:
      assert diff[name]["estado"] == "nueva"
    totals = {r["total"]: r["valor"] for r in state["totals"]}
    assert totals["colecciones antes → después"] == "2 → 4"
    points = sum(state["counts"][n] for n in shared["collections"])
    assert totals["puntos antes → después"] == f"2 → {points + 2}"
    # Writes, if any, only target configured collections.
    assert state["writes"]
    assert {t for _, t in state["writes"]} == set(shared["collections"])
    # "Antes" and "Después" are shown as tables.
    assert "other_data" in cell_text(executed, "s5-code")
    assert "nunca se toca" in cell_text(executed, "s8-code")

  def test_a_configured_name_that_is_a_foreign_looking_collection_is_refused(
    self, use_config: Callable[[Any], Path], env: Path
  ) -> None:
    """The prefix is enforced by validation, so ``other_data`` cannot be configured."""
    use_config(cfg(collections={"other_data": {"documents": ["manuales/faro.md"]}}))
    with pytest.raises(CellExecutionError, match="hermes_report_"):
      execute(load())


# ------------------------------------------------------- online services (stubs)


ONLINE_STUBS = """
# Test-side: Ollama and Qdrant stand-ins, so the online path runs without services.
from hermes_report.providers import HashEmbedder as _Hash
CALLED = {"check": [], "embedders": [], "clients": []}
check_ollama = lambda models, *a, **k: CALLED["check"].append(list(models))
def make_embedder(config):
  CALLED["embedders"].append([config.provider, config.model])
  return _Hash()
_mem = make_client(":memory:")
def make_client(url):
  CALLED["clients"].append(url)
  return _mem
"""


class TestServicesCell:
  def test_online_checks_each_distinct_ollama_model_once_and_builds_one_embedder_each(
    self,
    use_config: Callable[[Any], Path],
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    monkeypatch.delenv("HERMES_INGEST_OFFLINE")
    use_config(
      {
        "corpus_dir": DEMO,
        "qdrant_url": "http://stub:6333",
        "defaults": {"embedding": {"provider": "ollama", "model": "nomic-embed-text"}},
        "collections": {
          "hermes_report_t_a": {"documents": ["manuales/faro.md"]},
          "hermes_report_t_b": {"documents": ["manuales/brujula.md"]},
          "hermes_report_t_c": {
            "documents": ["politicas/vacaciones.md"],
            "embedding": {"model": "mxbai-embed-large"},
          },
          "hermes_report_t_d": {
            "documents": ["politicas/teletrabajo.md"],
            "embedding": {"provider": "hash", "model": "hash-256"},
          },
        },
      }
    )
    notebook = load()
    insert_after(notebook, "s3-code", ONLINE_STUBS, "t-stubs")
    insert_after(
      notebook,
      "s4-code",
      f"print('{MARK}' + json.dumps(CALLED))\n",
      "t-called",
    )
    executed = execute(notebook)
    no_cell_failed(executed)
    called = dumps(executed, "t-called")
    assert called["check"] == [["mxbai-embed-large", "nomic-embed-text"]]
    assert sorted(called["embedders"]) == [
      ["hash", "hash-256"],
      ["ollama", "mxbai-embed-large"],
      ["ollama", "nomic-embed-text"],
    ]
    assert called["clients"] == ["http://stub:6333"]
    manifest = json.loads((env / "collections.json").read_text(encoding="utf-8"))
    assert manifest["qdrant_url"] == "http://stub:6333"
    models = {n: e["embedding"]["model"] for n, e in manifest["collections"].items()}
    assert models == {
      "hermes_report_t_a": "nomic-embed-text",
      "hermes_report_t_b": "nomic-embed-text",
      "hermes_report_t_c": "mxbai-embed-large",
      "hermes_report_t_d": "hash-256",
    }

  def test_online_with_only_hash_embedders_does_not_call_ollama(
    self,
    use_config: Callable[[Any], Path],
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    monkeypatch.delenv("HERMES_INGEST_OFFLINE")
    use_config(
      {
        "corpus_dir": DEMO,
        "qdrant_url": "http://stub:6333",
        "defaults": {"embedding": {"provider": "hash", "model": "hash-256"}},
        "collections": {"hermes_report_t_a": {"documents": ["manuales/faro.md"]}},
      }
    )
    notebook = load()
    insert_after(notebook, "s3-code", ONLINE_STUBS, "t-stubs")
    insert_after(
      notebook, "s4-code", f"print('{MARK}' + json.dumps(CALLED))\n", "t-called"
    )
    called = dumps(execute(notebook), "t-called")
    assert called["check"] == []

  def test_a_failing_ollama_check_stops_before_qdrant_or_ingestion(
    self,
    use_config: Callable[[Any], Path],
    env: Path,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    monkeypatch.delenv("HERMES_INGEST_OFFLINE")
    use_config(
      {
        "corpus_dir": DEMO,
        "qdrant_url": "http://stub:6333",
        "collections": {"hermes_report_t_a": {"documents": ["manuales/faro.md"]}},
      }
    )
    notebook = load()
    insert_after(
      notebook,
      "s3-code",
      "def check_ollama(models, *a, **k):\n  raise RuntimeError('Ollama caido')\n"
      "def make_client(url):\n  raise SystemExit('REACHED QDRANT')\n",
      "t-stubs",
    )
    with pytest.raises(CellExecutionError, match="Ollama caido"):
      execute(notebook)
    assert not (env / "ingest-cache").exists()
    assert not (env / "collections.json").exists()


# ------------------------------------------------------------ with real services


def _http_json(url: str) -> Any:
  try:
    with urllib.request.urlopen(url, timeout=3) as reply:  # noqa: S310 - fixed http URL
      return json.load(reply)
  except (OSError, urllib.error.URLError, ValueError):
    return None


@pytest.mark.integration
def test_notebook_ingests_two_small_documents_with_real_ollama_and_qdrant(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  host = os.environ.get("OLLAMA_HOST", "").strip() or "http://localhost:11434"
  if not host.startswith("http"):
    host = f"http://{host}"
  qdrant_url = "http://localhost:6333"
  tags = _http_json(f"{host.rstrip('/')}/api/tags")
  have = {m["name"].removesuffix(":latest") for m in (tags or {}).get("models", [])}
  if "nomic-embed-text" not in have:
    pytest.skip(f"Ollama at {host} is unreachable or lacks nomic-embed-text")
  listing = _http_json(f"{qdrant_url}/collections")
  if listing is None:
    pytest.skip(f"Qdrant at {qdrant_url} is unreachable")
  data = ROOT / "notebooks" / "data"
  files = (
    sorted(data.glob("*.md"), key=lambda p: p.stat().st_size) if data.is_dir() else []
  )
  if len(files) < 2:
    pytest.skip("notebooks/data/ has fewer than two Markdown files")
  two = files[:2]
  before = sorted(c["name"] for c in listing["result"]["collections"])
  name = f"hermes_report_test_{uuid.uuid4().hex[:8]}"
  assert name not in before

  config_path = tmp_path / "config.json"
  config_path.write_text(
    json.dumps(
      {
        "corpus_dir": "notebooks/data",
        "qdrant_url": qdrant_url,
        "defaults": {"embedding": {"provider": "ollama", "model": "nomic-embed-text"}},
        "collections": {
          name: {
            "documents": [p.name for p in two],
            "ingest": {"max_chunk_size": 800},
          }
        },
      }
    ),
    encoding="utf-8",
  )
  out = tmp_path / "out"
  monkeypatch.delenv("HERMES_INGEST_OFFLINE", raising=False)
  monkeypatch.setenv("HERMES_INGEST_OUT", str(out))
  monkeypatch.setenv("HERMES_INGEST_CONFIG", str(config_path))
  notebook = load()
  insert_after(notebook, "s9-code", DUMP, "t-dump")
  try:
    executed = execute(notebook, timeout=900)
    state = dumps(executed, "t-dump")
    assert state["outcomes"] == {name: "creada"}
    assert set(state["sources"][name]) == {p.name for p in two}
    assert state["counts"][name] > 0
    assert state["ready"] == [name]
    manifest = json.loads((out / "collections.json").read_text(encoding="utf-8"))
    assert manifest["collections"][name]["embedding"]["model"] == "nomic-embed-text"
    assert manifest["collections"][name]["embedding"]["dimension"] == 768
    # Existing collections of Qdrant were only read.
    foreign_after = {
      r["colección"]: r["puntos"] for r in state["after"] if r["colección"] != name
    }
    foreign_before = {r["colección"]: r["puntos"] for r in state["before"]}
    assert foreign_after == foreign_before
  finally:
    from qdrant_client import QdrantClient

    assert name.startswith("hermes_report_test_")
    qdrant = QdrantClient(url=qdrant_url)
    if qdrant.collection_exists(name):
      qdrant.delete_collection(name)
    qdrant.close()
  after = sorted(
    c["name"] for c in _http_json(f"{qdrant_url}/collections")["result"]["collections"]
  )
  assert after == before


@pytest.fixture(autouse=True)
def _stored_notebook_is_never_modified() -> Any:
  """The stored notebook and README must be byte-identical after each test."""
  before = (NOTEBOOK.read_bytes(), README.read_bytes())
  yield
  assert (NOTEBOOK.read_bytes(), README.read_bytes()) == before

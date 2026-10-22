"""The shipped examples: the demo end to end, the Quijote files and the CI file.

The offline demo runs on copies in ``tmp_path`` (the ingestion cache lands next
to the run file, and a test must not write into the repository). The Quijote
integration test needs the real Ollama and Qdrant plus the prepared corpus; it
carries the ``integration`` marker, is deselected by default (see
``pyproject.toml``) and runs with ``uv run pytest -m integration``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tomllib
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from hermes_report.cli import main
from hermes_report.config import load_config
from hermes_report.corpus import load_corpus
from hermes_report.questions import load_questions
from hermes_report.record import (
  JudgeRecord,
  QuestionRecord,
  RuleRecord,
  RunRecord,
  read_record,
)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
DEMO = EXAMPLES / "demo"
QUIJOTE = EXAMPLES / "quijote"

# --------------------------------------------------------------------------
# The Quijote battery as it was in the playground (playground/quijote/
# questions.py, branch dev of hermes-generator), hard-coded on purpose: the
# file must match it field by field without importing the playground.
# ``chapters`` are (book, roman chapter); a book maps to a folder below.

_BOOK_I = "Parte I (1605)"
_BOOK_II = "Parte II (1615)"
_FOLDER = {_BOOK_I: "parte-1", _BOOK_II: "parte-2"}

# (id, question, expected, keywords, evidence, chapters)
_BATTERY: tuple[
  tuple[str, str, str, tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]],
  ...,
] = (
  (
    "q01",
    "¿Cuántos días tardó don Quijote en decidir qué nombre ponerle a su caballo?",
    "Cuatro días («Cuatro días se le pasaron en imaginar qué nombre le pondría»).",
    ("cuatro",),
    ("cuatro dias se le pasaron",),
    ((_BOOK_I, "I"),),
  ),
  (
    "q02",
    "¿Qué comía don Quijote los sábados?",
    "Duelos y quebrantos.",
    ("duelos", "quebrantos"),
    ("duelos y quebrantos",),
    ((_BOOK_I, "I"),),
  ),
  (
    "q03",
    "¿Cómo se llamaba la labradora a la que don Quijote dio el nombre de Dulcinea "
    "del Toboso?",
    "Aldonza Lorenzo.",
    ("aldonza",),
    ("aldonza lorenzo",),
    ((_BOOK_I, "I"), (_BOOK_I, "XXV")),
  ),
  (
    "q04",
    "¿Cuántos molinos de viento descubrieron don Quijote y Sancho en aquel campo?",
    "Treinta o cuarenta.",
    ("treinta", "cuarenta"),
    ("treinta o cuarenta molinos",),
    ((_BOOK_I, "VIII"),),
  ),
  (
    "q05",
    "¿Dónde encontró el narrador los cartapacios con la historia escrita por Cide "
    "Hamete Benengeli, y cuánto pagó por ellos?",
    "En el Alcaná de Toledo, por medio real.",
    ("alcana", "medio real"),
    ("alcana de toledo", "medio real"),
    ((_BOOK_I, "IX"),),
  ),
  (
    "q06",
    "¿Quién empezó a llamar a don Quijote «el Caballero de la Triste Figura»?",
    "Sancho Panza, tras la aventura del cuerpo muerto, por la mala figura que tenía "
    "su señor a la luz de las hachas.",
    ("sancho",),
    ("la mas mala figura",),
    ((_BOOK_I, "XIX"),),
  ),
  (
    "q07",
    "¿Qué era en realidad el yelmo de Mambrino que conquistó don Quijote?",
    "Una bacía de barbero.",
    ("bacia",),
    ("bacia de barbero",),
    ((_BOOK_I, "XXI"), (_BOOK_I, "XXV"), (_BOOK_I, "XLV")),
  ),
  (
    "q08",
    "¿Cómo se llamaba la ínsula que gobernó Sancho Panza?",
    "La ínsula Barataria.",
    ("barataria",),
    ("barataria",),
    ((_BOOK_II, "XLV"),),
  ),
  (
    "q09",
    "¿Quién se escondía tras el Caballero de la Blanca Luna que venció a don "
    "Quijote en la playa de Barcelona?",
    "El bachiller Sansón Carrasco.",
    ("carrasco",),
    ("me llaman el bachiller sanson carrasco",),
    ((_BOOK_II, "LXV"),),
  ),
  (
    "q10",
    "¿A quién deja don Quijote toda su hacienda en su testamento?",
    "A su sobrina, Antonia Quijana.",
    ("antonia",),
    ("antonia quijana",),
    ((_BOOK_II, "LXXIV"),),
  ),
  (
    "q11",
    "¿En qué año murió Sancho Panza?",
    "El libro no lo dice: el modelo debería negarse a responder.",
    (),
    (),
    (),
  ),
)


def _roman(text: str) -> int:
  """The value of a roman numeral (independent of the code under test)."""
  values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
  total = 0
  for char, following in zip(text, [*text[1:], ""], strict=True):
    value = values[char]
    total += -value if following and values[following] > value else value
  return total


def _chapter_file(book: str, roman: str) -> str:
  return f"{_FOLDER[book]}/capitulo-{_roman(roman):02d}.md"


def _expected_question(row: tuple[Any, ...]) -> dict[str, Any]:
  qid, text, expected, keywords, evidence, chapters = row
  out: dict[str, Any] = {"id": qid, "question": text, "expected": expected}
  if chapters:
    out["keywords"] = list(keywords)
    out["evidence"] = list(evidence)
    out["documents"] = [_chapter_file(b, c) for b, c in chapters]
  else:
    out["refusal"] = True
  return out


# ------------------------------------------------------------------ Quijote


def test_roman_helper_agrees_with_the_spec_example() -> None:
  # The task says (BOOK_I, "VIII") becomes parte-1/capitulo-08.md.
  assert _chapter_file(_BOOK_I, "VIII") == "parte-1/capitulo-08.md"
  assert _chapter_file(_BOOK_II, "LXXIV") == "parte-2/capitulo-74.md"
  assert _roman("XLV") == 45
  assert _roman("XIX") == 19
  assert _roman("LXV") == 65


def test_quijote_questions_match_the_playground_battery_field_by_field() -> None:
  data = json.loads((QUIJOTE / "preguntas.json").read_text(encoding="utf-8"))
  assert data["title"] == "Batería del Quijote"
  assert set(data) == {"title", "questions"}
  got = data["questions"]
  assert [q["id"] for q in got] == [row[0] for row in _BATTERY]
  for question, row in zip(got, _BATTERY, strict=True):
    # dict equality: same keys (no stray ones) and same values, in a list order
    assert question == _expected_question(row), row[0]


def test_quijote_q11_is_the_only_refusal_and_has_no_documents() -> None:
  data = json.loads((QUIJOTE / "preguntas.json").read_text(encoding="utf-8"))
  refusals = [q for q in data["questions"] if q.get("refusal")]
  assert [q["id"] for q in refusals] == ["q11"]
  assert not {"keywords", "evidence", "documents"} & set(refusals[0])


def test_quijote_questions_are_valid_for_the_loader_when_the_corpus_exists() -> None:
  corpus_dir = QUIJOTE / "corpus"
  if not (corpus_dir / "parte-2" / "capitulo-74.md").is_file():
    pytest.skip("run examples/quijote/prepare.py first")
  corpus = load_corpus(corpus_dir)
  assert len(corpus.documents) == 126  # 52 + 74 chapters
  questions = load_questions(QUIJOTE / "preguntas.json", corpus.sources)
  assert len(questions.questions) == 11


def test_quijote_run_file_has_the_documented_values() -> None:
  config = load_config(QUIJOTE / "run.toml")
  assert config.embedding.provider == "ollama"
  assert config.embedding.model == "nomic-embed-text"
  assert config.generation.provider == "ollama"
  assert config.generation.model == "qwen3:4b"
  assert config.retrieval.top_k == 10
  assert config.enrich.max_context_tokens == 3000
  assert config.corpus.dir == QUIJOTE / "corpus"
  assert config.questions.file == QUIJOTE / "preguntas.json"


# --------------------------------------------------------------------- demo


@pytest.fixture
def demo(tmp_path: Path) -> Path:
  """A throw-away copy of the demo (without any cache or earlier output)."""
  target = tmp_path / "demo"
  shutil.copytree(
    DEMO, target, ignore=shutil.ignore_patterns(".hermes-report-cache", "out")
  )
  return target


def _rules(question: QuestionRecord) -> RuleRecord:
  rules = question["rules"]
  assert rules is not None, question["query_id"]
  return rules


def _judge(question: QuestionRecord) -> JudgeRecord:
  judge = question["judge"]
  assert judge is not None, question["query_id"]
  return judge


def _run_offline(demo: Path, out: Path) -> RunRecord:
  assert main(["run", str(demo / "run-offline.toml"), "--out", str(out)]) == 0
  [folder] = list(out.iterdir())
  assert sorted(p.name for p in folder.iterdir()) == [
    "report.html",
    "run.json",
    "run.toml",
  ]
  return read_record(folder / "run.json")


def test_demo_offline_runs_end_to_end_into_a_folder_with_three_files(
  demo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  out = tmp_path / "out"
  record = _run_offline(demo, out)
  [folder] = list(out.iterdir())
  html = (folder / "report.html").read_text(encoding="utf-8")
  assert html.lower().startswith("<!doctype html>")
  assert "Demo de Nébula Logística" in html
  assert tomllib.loads((folder / "run.toml").read_text(encoding="utf-8"))["title"]
  assert capsys.readouterr().err == ""
  assert [q["query_id"] for q in record["questions"]] == [f"d0{n}" for n in range(1, 8)]
  assert all(q["error"] is None for q in record["questions"])
  assert all(_judge(q)["verdict"] == "partial" for q in record["questions"])
  assert record["config"]["embedding"]["provider"] == "hash"
  assert record["config"]["generation"]["provider"] == "echo"
  assert record["config"]["judge"]["provider"] == "echo"
  assert record["config"]["qdrant"]["url"] == ":memory:"
  assert len(record["ingest"]["documents"]) == 4


def test_demo_offline_verdicts_follow_the_question_kinds(
  demo: Path, tmp_path: Path
) -> None:
  record = _run_offline(demo, tmp_path / "out")
  verdicts = {q["query_id"]: _rules(q)["verdict"] for q in record["questions"]}
  # The echo model never answers right: every answerable question with keywords
  # that reached the context is a generation failure, never a retrieval one.
  assert verdicts["d05"] == "unrated"  # no keywords: judge only
  assert verdicts["d07"] == "invented_answer"  # refusal not honoured by echo
  for qid in ("d01", "d02", "d03", "d04", "d06"):
    assert verdicts[qid] == "generation_wrong", qid


def test_demo_offline_run_twice_is_stable_and_uses_only_hermes_report_names(
  demo: Path, tmp_path: Path
) -> None:
  first = _run_offline(demo, tmp_path / "out1")
  second = _run_offline(demo, tmp_path / "out2")  # now served from the cache
  assert first["collection"]["name"].startswith("hermes_report_")
  assert first["collection"]["name"] == second["collection"]["name"]
  assert [q["rules"] for q in first["questions"]] == [
    q["rules"] for q in second["questions"]
  ]
  assert all(d["cached"] for d in second["ingest"]["documents"])


def test_demo_questions_cover_the_required_kinds() -> None:
  data = json.loads((DEMO / "preguntas.json").read_text(encoding="utf-8"))
  questions = data["questions"]
  assert 6 <= len(questions) <= 8
  refusals = [q for q in questions if q.get("refusal")]
  assert len(refusals) == 1
  assert not refusals[0].get("documents")
  assert (
    len([q for q in questions if not q.get("keywords") and not q.get("refusal")]) >= 1
  )
  assert any(len(q.get("documents", [])) >= 2 for q in questions)  # across documents
  assert any(len(q.get("documents", [])) == 1 for q in questions)
  for q in questions:
    if not q.get("refusal"):
      assert q["expected"].strip(), q["id"]


def test_demo_corpus_has_four_to_six_documents_in_two_folders_with_headings() -> None:
  files = sorted((DEMO / "corpus").rglob("*.md"))
  assert 4 <= len(files) <= 6
  assert len({f.parent for f in files}) == 2
  for file in files:
    text = file.read_text(encoding="utf-8")
    assert re.search(r"^# \S", text, re.M), file
    assert re.search(r"^## \S", text, re.M), file


def test_demo_neighbour_question_has_its_evidence_in_adjacent_chunks(
  demo: Path, tmp_path: Path
) -> None:
  record = _run_offline(demo, tmp_path / "out")
  d02 = next(q for q in record["questions"] if q["query_id"] == "d02")
  chunks = {
    p["payload"]["chunk_index"]: p["payload"]["text"]
    for p in record["points"]
    if p["payload"].get("source") == "manuales/faro.md"
  }
  where = [
    [i for i, text in chunks.items() if phrase.lower() in text.lower()]
    for phrase in d02["spec"]["evidence"]
  ]
  # Each phrase is in exactly one chunk, and not the same one (so the answer
  # needs the neighbours of the retrieved chunk).
  assert all(len(w) == 1 for w in where), where
  assert where[0] != where[1]
  assert abs(where[0][0] - where[1][0]) == 1


def test_demo_evidence_phrases_are_in_their_documents() -> None:
  data = json.loads((DEMO / "preguntas.json").read_text(encoding="utf-8"))
  for q in data["questions"]:
    text = " ".join(
      " ".join((DEMO / "corpus" / d).read_text(encoding="utf-8").split())
      for d in q.get("documents", [])
    ).lower()
    for phrase in q.get("evidence", []):
      assert phrase.lower() in text, (q["id"], phrase)


def test_demo_online_run_file_uses_the_ollama_defaults() -> None:
  config = load_config(DEMO / "run.toml")
  assert config.embedding.provider == "ollama"
  assert config.embedding.model == "nomic-embed-text"
  assert config.generation.provider == "ollama"
  assert config.generation.model == "qwen3:4b"
  assert config.title.strip()
  assert (
    config.ingest.max_chunk_size
    == load_config(DEMO / "run-offline.toml").ingest.max_chunk_size
  )


def test_demo_offline_and_online_run_files_share_corpus_and_questions() -> None:
  online = load_config(DEMO / "run.toml")
  offline = load_config(DEMO / "run-offline.toml")
  assert online.corpus == offline.corpus
  assert online.questions == offline.questions


# ------------------------------------------------------- repo plumbing files


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_gitignore_keeps_generated_files_out_but_ships_the_examples() -> None:
  def ignored(path: str) -> bool:
    result = subprocess.run(
      ["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, check=False
    )
    return result.returncode == 0

  assert ignored("examples/quijote/corpus/parte-1/capitulo-01.md")
  assert ignored("examples/quijote/data/pg2000.txt")
  assert ignored("examples/demo/data/x.txt")
  assert ignored("examples/demo/out/20260101-x/run.json")
  assert ignored("out/run.json")
  assert ignored("examples/demo/.hermes-report-cache/x.json")
  assert not ignored("examples/demo/corpus/manuales/faro.md")
  assert not ignored("examples/demo/preguntas.json")
  assert not ignored("examples/quijote/preguntas.json")
  assert not ignored("examples/quijote/prepare.py")


def test_ci_reads_the_private_dependencies_before_uv_sync() -> None:
  text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
  assert "persist-credentials: false" in text
  token = "https://x-access-token:${{ secrets.HERMES_DEPS_TOKEN }}@github.com/"
  for source in ("ssh://git@github.com/", "https://github.com/"):
    assert f'url."{token}".insteadOf' in text
    assert f'"{source}"' in text
  assert text.index("insteadOf") < text.index("- run: uv sync")
  assert text.rindex("insteadOf") < text.index("- run: uv sync")
  assert "FU-1" in text


def test_pyproject_pins_uv_link_mode_copy_and_the_readme_explains_it() -> None:
  pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
  assert pyproject["tool"]["uv"]["link-mode"] == "copy"
  readme = (ROOT / "README.md").read_text(encoding="utf-8")
  assert "link-mode" in readme
  assert "nltk" in readme


# ------------------------------------------------- Quijote with real services


def _reachable(url: str) -> bool:
  try:
    with urllib.request.urlopen(url, timeout=3):  # noqa: S310 - fixed http URL
      return True
  except (OSError, urllib.error.URLError, ValueError):
    return False


def _ollama_models(host: str) -> set[str]:
  try:
    with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as reply:
      data = json.load(reply)
  except (OSError, urllib.error.URLError, ValueError):
    return set()
  return {m["name"] for m in data.get("models", [])}


@pytest.mark.integration
def test_quijote_end_to_end_with_real_ollama_and_qdrant(
  tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  host = os.environ.get("OLLAMA_HOST", "").strip() or "http://localhost:11434"
  if not host.startswith("http"):
    host = f"http://{host}"
  qdrant_url = "http://localhost:6333"
  models = _ollama_models(host)
  needed = {"nomic-embed-text", "qwen3:4b"}
  have = {m.removesuffix(":latest") for m in models}
  if not needed <= have:
    pytest.skip(f"Ollama at {host} is unreachable or lacks {sorted(needed - have)}")
  if not _reachable(f"{qdrant_url}/collections"):
    pytest.skip(f"Qdrant at {qdrant_url} is unreachable")
  if not (QUIJOTE / "corpus" / "parte-2" / "capitulo-74.md").is_file():
    pytest.skip("run examples/quijote/prepare.py first")

  client = QdrantClient(url=qdrant_url)
  before = {c.name for c in client.get_collections().collections}
  out = tmp_path / "out"
  code = main(["run", str(QUIJOTE / "run.toml"), "--out", str(out)])
  captured = capsys.readouterr()
  assert code == 0, captured.err
  after = {c.name for c in client.get_collections().collections}

  # Only hermes_report_* collections were created or removed.
  assert all(n.startswith("hermes_report_") for n in before ^ after)
  assert {n for n in before if not n.startswith("hermes_report_")} <= after

  [folder] = list(out.iterdir())
  assert (folder / "report.html").is_file()
  assert (folder / "run.toml").is_file()
  record = read_record(folder / "run.json")
  assert record["collection"]["name"].startswith("hermes_report_")
  assert record["collection"]["name"] in after
  assert [q["query_id"] for q in record["questions"]] == [
    f"q{n:02d}" for n in range(1, 12)
  ]
  for q in record["questions"]:
    assert q["error"] is None, q["query_id"]
    assert _rules(q)["verdict"] != "error", q["query_id"]
    assert _judge(q)["verdict"] in {"correct", "partial", "incorrect"}, q["query_id"]
  # The quality of the answers is reported, not asserted: it depends on the
  # models. Only check that the pipeline did something real (the first
  # run found the evidence for 2 of 10 answerable questions with the defaults).
  retrieved = [q for q in record["questions"] if _rules(q)["in_retrieval"]]
  assert len(retrieved) >= 1
  assert len(record["points"]) > 1000  # the whole book was ingested
  print(
    "VERDICTS "
    + json.dumps(
      {
        q["query_id"]: [_rules(q)["verdict"], _judge(q)["verdict"]]
        for q in record["questions"]
      },
      ensure_ascii=False,
    )
  )


# ------------------------------------------------- the Quijote prepare script


def _load_prepare() -> Any:
  spec = importlib.util.spec_from_file_location(
    "quijote_prepare", QUIJOTE / "prepare.py"
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


_FAKE_GUTENBERG = """\
Preface of the project
*** START OF THE PROJECT GUTENBERG EBOOK EL QUIJOTE ***

Dedicatoria que no debe aparecer.

Primera parte del ingenioso hidalgo don Quijote de la Mancha

Capítulo primero. Que trata de la condición
del famoso hidalgo

En un lugar de la Mancha, de cuyo nombre no quiero
acordarme, no ha mucho tiempo que vivía un hidalgo.

Capítulo II. Que trata de la primera salida

Salió una mañana.

Segunda parte del ingenioso hidalgo don Quijote de la Mancha

Capítulo III. Donde se cuenta la graciosa manera

Tercer capítulo.

Segunda parte del ingenioso caballero don Quijote de la Mancha

Capítulo I. Que trata de lo que el cura

Libro segundo.
*** END OF THE PROJECT GUTENBERG EBOOK EL QUIJOTE ***
license text
"""


def test_prepare_converts_and_splits_a_small_text() -> None:
  prepare = _load_prepare()
  markdown, chapters = prepare.to_markdown(_FAKE_GUTENBERG)
  assert chapters == 4
  assert "Dedicatoria" not in markdown  # the front matter is left out
  assert "license text" not in markdown
  files = prepare.split_chapters(markdown)
  assert list(files) == [
    "parte-1/capitulo-01.md",
    "parte-1/capitulo-02.md",
    "parte-1/capitulo-03.md",
    "parte-2/capitulo-01.md",
  ]
  first = "\n\n".join(files["parte-1/capitulo-01.md"])
  # H1 book, H2 chapter (wrapped heading joined, "primero" normalised), body.
  assert first.startswith(
    "# Parte I (1605)\n\n## Capítulo I. Que trata de la condición del famoso hidalgo\n\n"
  )
  assert "En un lugar de la Mancha, de cuyo nombre no quiero acordarme" in first
  # The inner-part bold line sits under the heading of the chapter it opens.
  third = files["parte-1/capitulo-03.md"]
  assert third[0] == "# Parte I (1605)"
  assert third[1].startswith("## Capítulo III.")
  assert third[2] == "**Segunda parte del ingenioso hidalgo don Quijote de la Mancha**"
  assert files["parte-2/capitulo-01.md"][0] == "# Parte II (1615)"


@pytest.mark.parametrize(
  "text",
  [
    "",
    "no markers at all",
    "*** START OF THE PROJECT GUTENBERG EBOOK X ***\nonly the start\n",
    "*** START OF THE PROJECT GUTENBERG EBOOK X ***\nno body start\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n",
  ],
)
def test_prepare_rejects_a_text_without_the_markers(text: str) -> None:
  with pytest.raises(ValueError, match="Gutenberg"):
    _load_prepare().to_markdown(text)


def test_prepare_exits_1_when_the_chapter_count_is_not_52_plus_74(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  prepare = _load_prepare()
  data, corpus = tmp_path / "data", tmp_path / "corpus"
  data.mkdir()
  (data / "pg2000.txt").write_text(_FAKE_GUTENBERG, encoding="utf-8")  # no download
  monkeypatch.setattr(prepare, "DATA_DIR", data)
  monkeypatch.setattr(prepare, "CORPUS_DIR", corpus)
  assert prepare.main() == 1
  assert "52 + 74" in capsys.readouterr().err


def test_prepare_exits_1_on_an_unexpected_text_and_keeps_the_old_corpus(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  prepare = _load_prepare()
  data, corpus = tmp_path / "data", tmp_path / "corpus"
  (corpus / "parte-1").mkdir(parents=True)
  old = corpus / "parte-1" / "capitulo-01.md"
  old.write_text("old", encoding="utf-8")
  data.mkdir()
  (data / "pg2000.txt").write_text("<html>not the book</html>", encoding="utf-8")
  monkeypatch.setattr(prepare, "DATA_DIR", data)
  monkeypatch.setattr(prepare, "CORPUS_DIR", corpus)
  assert prepare.main() == 1
  assert capsys.readouterr().err.startswith("error: ")
  assert old.read_text(encoding="utf-8") == "old"


def test_prepare_failed_download_exits_1_without_leaving_a_partial_file(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  prepare = _load_prepare()

  def refuse(*args: Any, **kwargs: Any) -> None:
    raise urllib.error.URLError("offline")

  monkeypatch.setattr(prepare, "DATA_DIR", tmp_path / "data")
  monkeypatch.setattr(prepare, "CORPUS_DIR", tmp_path / "corpus")
  monkeypatch.setattr(prepare.urllib.request, "urlopen", refuse)
  assert prepare.main() == 1
  assert "error:" in capsys.readouterr().err
  assert not (tmp_path / "data" / "pg2000.txt").exists()
  assert not (tmp_path / "corpus").exists()


def test_prepare_on_the_real_downloaded_text_gives_52_plus_74_clean_chapters(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  raw = QUIJOTE / "data" / "pg2000.txt"
  if not raw.is_file():
    pytest.skip("run examples/quijote/prepare.py first")
  prepare = _load_prepare()
  data, corpus = tmp_path / "data", tmp_path / "corpus"
  data.mkdir()
  shutil.copy(raw, data / "pg2000.txt")  # already there: nothing is downloaded
  monkeypatch.setattr(prepare, "DATA_DIR", data)
  monkeypatch.setattr(prepare, "CORPUS_DIR", corpus)
  assert prepare.main() == 0
  assert prepare.main() == 0  # idempotent
  names = sorted(p.relative_to(corpus).as_posix() for p in corpus.rglob("*.md"))
  assert names == sorted(
    [f"parte-1/capitulo-{n:02d}.md" for n in range(1, 53)]
    + [f"parte-2/capitulo-{n:02d}.md" for n in range(1, 75)]
  )
  for name in names:
    text = (corpus / name).read_text(encoding="utf-8")
    book = "Parte I (1605)" if name.startswith("parte-1") else "Parte II (1615)"
    assert text.startswith(f"# {book}\n\n## Capítulo "), name
    assert len(re.findall(r"^# ", text, re.M)) == 1, name
    assert len(re.findall(r"^## ", text, re.M)) == 1, name
    assert "GUTENBERG" not in text, name
  # The ten evidence phrases of the battery are inside their chapter files.
  data_q = json.loads((QUIJOTE / "preguntas.json").read_text(encoding="utf-8"))
  for q in data_q["questions"]:
    joined = " ".join(
      " ".join((corpus / d).read_text(encoding="utf-8").split())
      for d in q.get("documents", [])
    )
    folded = unicodedata.normalize("NFD", joined.lower())
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    for phrase in q.get("evidence", []):
      assert phrase in folded, (q["id"], phrase)

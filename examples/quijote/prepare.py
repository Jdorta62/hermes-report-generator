"""Download Don Quijote from Project Gutenberg and write one Markdown file per chapter.

Run it from the repository root, standard library only::

    uv run python examples/quijote/prepare.py

The plain-text edition (ebook 2000, about 2 MB) is downloaded once into
``examples/quijote/data/``. It is converted to Markdown (each book an H1, each
chapter an H2) and split into ``examples/quijote/corpus/parte-1/capitulo-01.md``
to ``parte-2/capitulo-74.md``. Every file starts with ``# <book>`` and
``## <chapter heading>``, so the ``heading_path`` of its chunks is
``[book, chapter]``. The four inner parts of the 1605 book are kept as bold
lines under the heading of the chapter that opens them; the front matter of
each book (dedications, prologues, verses) is left out.

Exit code: 0 when the 52 + 74 chapters were written, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
BOOK_I = "Parte I (1605)"
BOOK_II = "Parte II (1615)"

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
CORPUS_DIR = _HERE / "corpus"

_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END = "*** END OF THE PROJECT GUTENBERG EBOOK"
_BODY_START = "Primera parte del ingenioso hidalgo don Quijote de la Mancha"
_INNER_PART = re.compile(
  r"^(Primera|Segunda|Tercera|Cuarta) parte del ingenioso hidalgo don Quijote de la Mancha$"
)
_SECOND_BOOK = "Segunda parte del ingenioso caballero don Quijote de la Mancha"
_CHAPTER = re.compile(r"^Capítulo ([Pp]rimero|[IVXLC]+)\.")
_FIRST = re.compile(r"^Capítulo [Pp]rimero\.")
_EXPECTED = {"parte-1": 52, "parte-2": 74}
_VERSE_WIDTH = 50


def _download(target: Path) -> None:
  """Download the Gutenberg text to ``target``, through a temporary file."""
  request = urllib.request.Request(URL, headers={"User-Agent": "hermes-report-example"})
  with urllib.request.urlopen(request, timeout=120) as reply:  # noqa: S310
    data = reply.read()
  partial = target.with_name(target.name + ".part")
  partial.write_bytes(data)
  partial.replace(target)


def _paragraph(lines: list[str]) -> str:
  """Join the lines of a paragraph; verse keeps its line breaks."""
  stripped = [line.strip() for line in lines]
  if max(len(line) for line in stripped) < _VERSE_WIDTH and len(stripped) > 1:
    return "\n".join(stripped)
  return " ".join(stripped)


def to_markdown(text: str) -> tuple[str, int]:
  """Convert the Gutenberg text to Markdown; return it and its chapter count.

  Raises:
    ValueError: The Gutenberg markers or the start of the novel are missing.
  """
  lines = text.replace("\r\n", "\n").split("\n")
  try:
    start = next(i for i, line in enumerate(lines) if line.startswith(_START))
    end = next(i for i, line in enumerate(lines) if line.startswith(_END))
    body_start = next(i for i in range(start, end) if lines[i].strip() == _BODY_START)
  except StopIteration:
    raise ValueError("texto de Gutenberg inesperado: faltan las marcas") from None
  out: list[str] = [f"# {BOOK_I}"]
  chapters = 0
  block: list[str] = []

  def flush() -> None:
    nonlocal chapters
    if not block:
      return
    first = block[0].strip()
    if _INNER_PART.match(first):
      out.append(f"**{first}**")
    elif first == _SECOND_BOOK:
      out.append(f"# {BOOK_II}")
    elif _CHAPTER.match(first):
      heading = " ".join(line.strip() for line in block)
      out.append("## " + _FIRST.sub("Capítulo I.", heading, count=1))
      chapters += 1
    else:
      out.append(_paragraph(block))
    block.clear()

  for line in lines[body_start:end]:
    if line.strip():
      block.append(line)
    else:
      flush()
  flush()
  return "\n\n".join(out) + "\n", chapters


def split_chapters(markdown: str) -> dict[str, list[str]]:
  """Split the Markdown of :func:`to_markdown` into one text per chapter.

  Returns:
    ``{"parte-1/capitulo-01.md": text, ...}`` in book order. Each text starts
    with ``# <book>`` and ``## <chapter heading>``.
  """
  files: dict[str, list[str]] = {}
  folder = ""
  book = ""
  number = 0
  current: list[str] | None = None
  pending: list[str] = []
  for block in markdown.split("\n\n"):
    block = block.strip("\n")
    if not block:
      continue
    if block.startswith("# "):
      book = block
      folder = "parte-1" if block == f"# {BOOK_I}" else "parte-2"
      number = 0
      current = None
    elif block.startswith("## "):
      number += 1
      current = [book, block, *pending]
      pending = []
      files[f"{folder}/capitulo-{number:02d}.md"] = current
    elif block.startswith("**") and block.endswith("**"):
      pending.append(block)
    elif current is not None:
      current.append(block)
  return files


def write_corpus(files: dict[str, list[str]], corpus_dir: Path) -> dict[str, int]:
  """Write the chapter files, replacing the old ones; return the count per part."""
  for old in corpus_dir.glob("parte-*/capitulo-*.md"):
    old.unlink()
  counts: dict[str, int] = {}
  for name, blocks in files.items():
    target = corpus_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    part = name.split("/", 1)[0]
    counts[part] = counts.get(part, 0) + 1
  return counts


def main() -> int:
  """Download (once), convert and write the chapters; return the exit code."""
  raw = DATA_DIR / "pg2000.txt"
  try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if raw.exists():
      print(f"Texto ya descargado: {raw}")
    else:
      print(f"Descargando {URL} ...")
      _download(raw)
      print(f"Guardado en {raw}")
    markdown, chapters = to_markdown(raw.read_text(encoding="utf-8"))
    (DATA_DIR / "quijote.md").write_text(markdown, encoding="utf-8")
    counts = write_corpus(split_chapters(markdown), CORPUS_DIR)
  except (OSError, urllib.error.URLError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
  print(f"Capítulos en el texto: {chapters}")
  for part, expected in _EXPECTED.items():
    print(f"{part}: {counts.get(part, 0)} capítulos (esperados {expected})")
  if counts != _EXPECTED or chapters != sum(_EXPECTED.values()):
    print("error: el número de capítulos no es 52 + 74", file=sys.stderr)
    return 1
  print(f"Corpus escrito en {CORPUS_DIR}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

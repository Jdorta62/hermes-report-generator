"""Markdown corpus: the documents of a folder, hashed and sorted."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Corpus", "CorpusDocument", "CorpusError", "load_corpus"]

# Everything reading, resolving or globbing user-given input can raise.
_INPUT_ERRORS = (
  OSError,
  ValueError,
  RecursionError,
  RuntimeError,
  OverflowError,
  UnicodeError,
)


class CorpusError(Exception):
  """The corpus folder cannot be read as a corpus (message in Spanish)."""


@dataclass(frozen=True)
class CorpusDocument:
  """One Markdown file of the corpus.

  Attributes:
    source: Path relative to the corpus folder, with ``/`` separators.
    path: Absolute path of the file.
    sha256: Hex digest of the file bytes.
    chars: Length of the decoded text.
  """

  source: str
  path: Path
  sha256: str
  chars: int

  def read_bytes(self) -> bytes:
    """Read the file bytes.

    Raises:
      OSError: if the file cannot be read.
    """
    return self.path.read_bytes()


@dataclass(frozen=True)
class Corpus:
  """The documents of a corpus folder, sorted by ``source``.

  Attributes:
    dir: Absolute corpus folder.
    glob: Pattern that selected the documents, relative to ``dir``.
    documents: The documents, sorted by ``source``.
  """

  dir: Path
  glob: str
  documents: tuple[CorpusDocument, ...]

  @property
  def sources(self) -> tuple[str, ...]:
    """The ``source`` of every document, in order."""
    return tuple(doc.source for doc in self.documents)

  @property
  def fingerprint(self) -> str:
    """Hex sha256 over the ``source:sha256`` lines of the documents."""
    digest = hashlib.sha256()
    for doc in self.documents:
      digest.update(f"{doc.source}:{doc.sha256}\n".encode())
    return digest.hexdigest()


def load_corpus(dir: Path, glob: str = "**/*.md") -> Corpus:
  """Load the documents of ``dir`` that match ``glob``, UTF-8 only.

  Raises:
    CorpusError: if ``dir`` is not a folder, nothing matches or a file is not
      valid UTF-8.
  """
  try:
    root = dir.resolve()
    is_dir = root.is_dir()
  except _INPUT_ERRORS as exc:
    raise CorpusError(f"{dir}: no es una carpeta válida: {exc}") from None
  if not is_dir:
    raise CorpusError(f"{dir}: no es una carpeta")
  try:
    matches = [p for p in root.glob(glob) if p.is_file()]
  except _INPUT_ERRORS as exc:
    raise CorpusError(f"{dir}: patrón glob inválido {glob!r}: {exc}") from None
  if not matches:
    raise CorpusError(f"{dir}: ningún archivo coincide con {glob!r}")
  documents: list[CorpusDocument] = []
  for path in matches:
    try:
      relative = path.relative_to(root)
      source = relative.as_posix()
      source.encode("utf-8")
    except UnicodeEncodeError:
      raise CorpusError(f"{path}: el nombre no es UTF-8 válido") from None
    except _INPUT_ERRORS as exc:
      raise CorpusError(f"{path}: ruta no válida: {exc}") from None
    if ".." in relative.parts:
      raise CorpusError(f"{dir}: {glob!r} sale de la carpeta del corpus: {path}")
    try:
      data = path.read_bytes()
      text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
      raise CorpusError(f"{path}: no es UTF-8 válido: {exc}") from exc
    except _INPUT_ERRORS as exc:
      raise CorpusError(f"{path}: no se puede leer: {exc}") from None
    documents.append(
      CorpusDocument(
        source=source,
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        chars=len(text),
      )
    )
  documents.sort(key=lambda doc: doc.source)
  return Corpus(dir=root, glob=glob, documents=tuple(documents))

"""Tests for ``hermes_report.corpus``: listing, hashing, ordering, errors."""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import pytest

from hermes_report.corpus import CorpusError, load_corpus


def sha(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


@pytest.fixture
def folder(tmp_path: Path) -> Path:
  root = tmp_path / "corpus"
  root.mkdir()
  return root


def put(root: Path, relative: str, data: bytes | str = "# T\n") -> Path:
  path = root / relative
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
  return path


# --- the sample corpus ---------------------------------------------------------------


def test_sample_corpus(data_dir: Path) -> None:
  corpus = load_corpus(data_dir / "corpus")
  assert corpus.dir == (data_dir / "corpus").resolve()
  assert corpus.glob == "**/*.md"
  assert corpus.sources == (
    "notas.md",
    "parte-1/capitulo-01.md",
    "parte-1/capitulo-02.md",
  )
  assert len(corpus.documents) == 3


def test_sample_document_fields(data_dir: Path) -> None:
  doc = load_corpus(data_dir / "corpus").documents[1]
  raw = (data_dir / "corpus" / "parte-1" / "capitulo-01.md").read_bytes()
  assert doc.source == "parte-1/capitulo-01.md"
  assert doc.path == (data_dir / "corpus" / "parte-1" / "capitulo-01.md").resolve()
  assert doc.path.is_absolute()
  assert doc.sha256 == sha(raw)
  assert doc.chars == len(raw.decode("utf-8"))
  assert doc.read_bytes() == raw


# --- listing -------------------------------------------------------------------------


class TestListing:
  def test_sorted_by_source_not_by_creation(self, folder: Path) -> None:
    for name in ("b.md", "z/a.md", "a.md", "m/n/o.md", "A.md"):
      put(folder, name)
    assert load_corpus(folder).sources == ("A.md", "a.md", "b.md", "m/n/o.md", "z/a.md")

  def test_sort_is_by_source_string(self, folder: Path) -> None:
    # "a.md" < "a/b.md" because "." sorts before "/" in the source string.
    for name in ("a/b.md", "a.md", "a-b.md"):
      put(folder, name)
    sources = load_corpus(folder).sources
    assert list(sources) == sorted(sources)

  def test_order_is_independent_of_directory_order(self, tmp_path: Path) -> None:
    names = ["c.md", "a.md", "b/z.md", "b/a.md"]
    one, two = tmp_path / "one", tmp_path / "two"
    for name in names:
      put(one, name)
    for name in reversed(names):
      put(two, name)
    assert load_corpus(one).sources == load_corpus(two).sources

  def test_source_uses_forward_slashes_and_is_relative(self, folder: Path) -> None:
    put(folder, "a/b/c.md")
    (doc,) = load_corpus(folder).documents
    assert doc.source == "a/b/c.md"
    assert "\\" not in doc.source and not doc.source.startswith("/")

  def test_default_glob_ignores_other_extensions(self, folder: Path) -> None:
    put(folder, "a.md")
    put(folder, "b.txt")
    put(folder, "c.markdown")
    put(folder, "d.MD")
    assert load_corpus(folder).sources[0] == "a.md"
    assert "b.txt" not in load_corpus(folder).sources

  def test_custom_glob(self, folder: Path) -> None:
    put(folder, "a.md")
    put(folder, "sub/b.md")
    put(folder, "sub/c.txt")
    corpus = load_corpus(folder, "*.md")
    assert corpus.sources == ("a.md",)
    assert corpus.glob == "*.md"
    assert load_corpus(folder, "sub/*").sources == ("sub/b.md", "sub/c.txt")

  def test_directories_matching_the_glob_are_skipped(self, folder: Path) -> None:
    put(folder, "real.md")
    (folder / "folder.md").mkdir()
    put(folder, "folder.md/inner.md")
    assert load_corpus(folder).sources == ("folder.md/inner.md", "real.md")

  def test_only_directories_match(self, folder: Path) -> None:
    (folder / "x.md").mkdir()
    with pytest.raises(CorpusError):
      load_corpus(folder)

  def test_hidden_files_and_folders_are_listed(self, folder: Path) -> None:
    put(folder, ".hidden/x.md")
    assert load_corpus(folder).sources == (".hidden/x.md",)

  def test_symlinked_file_inside_is_listed(self, folder: Path) -> None:
    target = put(folder, "real.md")
    (folder / "link.md").symlink_to(target)
    assert load_corpus(folder).sources == ("link.md", "real.md")

  def test_broken_symlink_is_skipped(self, folder: Path) -> None:
    put(folder, "ok.md")
    (folder / "broken.md").symlink_to(folder / "missing.md")
    assert load_corpus(folder).sources == ("ok.md",)

  def test_relative_dir_is_resolved(
    self, folder: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    put(folder, "a.md")
    monkeypatch.chdir(folder.parent)
    corpus = load_corpus(Path("corpus"))
    assert corpus.dir == folder.resolve()
    assert corpus.documents[0].path.is_absolute()

  def test_unicode_file_names(self, folder: Path) -> None:
    put(folder, "capítulo ñ 😀.md")
    assert load_corpus(folder).sources == ("capítulo ñ 😀.md",)

  def test_dir_through_a_symlink(self, folder: Path, tmp_path: Path) -> None:
    put(folder, "a.md")
    link = tmp_path / "link"
    link.symlink_to(folder, target_is_directory=True)
    assert load_corpus(link).sources == ("a.md",)


# --- document fields -----------------------------------------------------------------


class TestDocument:
  def test_sha256_of_bytes(self, folder: Path) -> None:
    put(folder, "a.md", b"hola\n")
    assert load_corpus(folder).documents[0].sha256 == sha(b"hola\n")

  def test_chars_is_decoded_length_not_bytes(self, folder: Path) -> None:
    put(folder, "a.md", "ñandú ✓ 😀")
    doc = load_corpus(folder).documents[0]
    assert doc.chars == len("ñandú ✓ 😀") == 9
    assert len(doc.read_bytes()) > doc.chars

  def test_crlf_is_not_translated(self, folder: Path) -> None:
    data = b"a\r\nb\r\n"
    put(folder, "a.md", data)
    doc = load_corpus(folder).documents[0]
    assert doc.chars == 6
    assert doc.sha256 == sha(data)
    assert doc.read_bytes() == data

  def test_empty_file_is_a_document(self, folder: Path) -> None:
    put(folder, "empty.md", b"")
    doc = load_corpus(folder).documents[0]
    assert doc.chars == 0
    assert doc.sha256 == sha(b"")

  def test_bom_is_counted_and_hashed_as_is(self, folder: Path) -> None:
    data = b"\xef\xbb\xbfhola"
    put(folder, "a.md", data)
    doc = load_corpus(folder).documents[0]
    assert doc.sha256 == sha(data)
    assert doc.chars == 5

  def test_nul_bytes_are_valid_utf8(self, folder: Path) -> None:
    put(folder, "a.md", b"a\x00b")
    assert load_corpus(folder).documents[0].chars == 3

  def test_large_file(self, folder: Path) -> None:
    data = b"x" * 5_000_000
    put(folder, "big.md", data)
    doc = load_corpus(folder).documents[0]
    assert doc.chars == 5_000_000 and doc.sha256 == sha(data)

  def test_read_bytes_reads_the_current_file(self, folder: Path) -> None:
    path = put(folder, "a.md", b"one")
    doc = load_corpus(folder).documents[0]
    path.write_bytes(b"two")
    assert doc.read_bytes() == b"two"  # the hash is a snapshot, the read is live

  def test_read_bytes_raises_oserror_when_gone(self, folder: Path) -> None:
    path = put(folder, "a.md")
    doc = load_corpus(folder).documents[0]
    path.unlink()
    with pytest.raises(OSError):
      doc.read_bytes()

  def test_frozen(self, folder: Path) -> None:
    put(folder, "a.md")
    corpus = load_corpus(folder)
    with pytest.raises(dataclasses.FrozenInstanceError):
      corpus.documents[0].source = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
      corpus.glob = "x"  # type: ignore[misc]

  def test_documents_are_a_tuple(self, folder: Path) -> None:
    put(folder, "a.md")
    assert isinstance(load_corpus(folder).documents, tuple)


# --- fingerprint ---------------------------------------------------------------------


class TestFingerprint:
  def test_definition(self, folder: Path) -> None:
    put(folder, "b.md", b"BB")
    put(folder, "a.md", b"AA")
    expected = hashlib.sha256(
      f"a.md:{sha(b'AA')}\nb.md:{sha(b'BB')}\n".encode()
    ).hexdigest()
    assert load_corpus(folder).fingerprint == expected

  def test_single_document(self, folder: Path) -> None:
    put(folder, "a.md", b"AA")
    assert load_corpus(folder).fingerprint == sha(f"a.md:{sha(b'AA')}\n".encode())

  def test_stable_across_loads_and_locations(self, tmp_path: Path) -> None:
    for root in (tmp_path / "x", tmp_path / "y" / "deeper"):
      put(root, "a.md", b"AA")
      put(root, "d/b.md", b"BB")
    assert (
      load_corpus(tmp_path / "x").fingerprint
      == load_corpus(tmp_path / "y" / "deeper").fingerprint
    )
    assert (
      load_corpus(tmp_path / "x").fingerprint == load_corpus(tmp_path / "x").fingerprint
    )

  def test_content_change_changes_it(self, folder: Path) -> None:
    path = put(folder, "a.md", b"AA")
    before = load_corpus(folder).fingerprint
    path.write_bytes(b"AB")
    assert load_corpus(folder).fingerprint != before

  def test_one_byte_change_in_whitespace_changes_it(self, folder: Path) -> None:
    path = put(folder, "a.md", b"AA\n")
    before = load_corpus(folder).fingerprint
    path.write_bytes(b"AA\r\n")
    assert load_corpus(folder).fingerprint != before

  def test_rename_changes_it(self, folder: Path) -> None:
    path = put(folder, "a.md", b"AA")
    before = load_corpus(folder).fingerprint
    path.rename(folder / "b.md")
    assert load_corpus(folder).fingerprint != before

  def test_adding_a_file_changes_it(self, folder: Path) -> None:
    put(folder, "a.md")
    before = load_corpus(folder).fingerprint
    put(folder, "b.md")
    assert load_corpus(folder).fingerprint != before

  def test_swapping_contents_between_files_changes_it(self, folder: Path) -> None:
    a, b = put(folder, "a.md", b"AA"), put(folder, "b.md", b"BB")
    before = load_corpus(folder).fingerprint
    a.write_bytes(b"BB")
    b.write_bytes(b"AA")
    assert load_corpus(folder).fingerprint != before

  def test_glob_does_not_change_it_for_the_same_documents(self, folder: Path) -> None:
    put(folder, "a.md")
    assert load_corpus(folder, "*.md").fingerprint == load_corpus(folder).fingerprint

  def test_depends_only_on_selected_documents(self, folder: Path) -> None:
    put(folder, "a.md", b"AA")
    before = load_corpus(folder).fingerprint
    put(folder, "ignored.txt", b"zzz")
    assert load_corpus(folder).fingerprint == before

  def test_is_hex_sha256(self, folder: Path) -> None:
    put(folder, "a.md")
    value = load_corpus(folder).fingerprint
    assert len(value) == 64 and int(value, 16) >= 0 and value == value.lower()

  def test_filename_with_colon_does_not_collide(self, folder: Path) -> None:
    # "a:b.md" with hash H and "a.md" with a crafted hash cannot be confused
    # because the hash is a fixed-length hex digest.
    put(folder, "a:b.md", b"1")
    first = load_corpus(folder).fingerprint
    (folder / "a:b.md").unlink()
    put(folder, "a.md", b"1")
    assert load_corpus(folder).fingerprint != first


# --- errors --------------------------------------------------------------------------


class TestErrors:
  def test_missing_dir(self, tmp_path: Path) -> None:
    with pytest.raises(CorpusError) as info:
      load_corpus(tmp_path / "nope")
    assert "nope" in str(info.value)

  def test_nul_byte_in_dir(self) -> None:
    with pytest.raises(CorpusError):
      load_corpus(Path("a\0b"))

  def test_over_long_dir_name(self) -> None:
    with pytest.raises(CorpusError):
      load_corpus(Path("a" * 5000))

  @pytest.mark.parametrize("glob", ["a\0b", "[", "/abs/*.md", ""])
  def test_hostile_glob_is_a_corpus_error(self, folder: Path, glob: str) -> None:
    put(folder, "a.md")
    with pytest.raises(CorpusError):
      load_corpus(folder, glob)

  @pytest.mark.skipif(os.name != "posix", reason="needs raw bytes in file names")
  def test_non_utf8_file_name(self, folder: Path) -> None:
    put(folder, "ok.md")
    bad = os.path.join(os.fsencode(folder), b"caf\xe9.md")
    try:
      with open(bad, "wb") as handle:
        handle.write(b"# T\n")
    except OSError:
      pytest.skip("file system refuses non UTF-8 names")
    with pytest.raises(CorpusError) as info:
      load_corpus(folder)
    assert "UTF-8" in str(info.value)

  @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs chmod")
  def test_unreadable_file(self, folder: Path) -> None:
    path = put(folder, "a.md")
    path.chmod(0)
    try:
      with pytest.raises(CorpusError) as info:
        load_corpus(folder)
      assert str(path) in str(info.value)
    finally:
      path.chmod(0o644)

  def test_dir_is_a_file(self, folder: Path) -> None:
    path = put(folder, "a.md")
    with pytest.raises(CorpusError) as info:
      load_corpus(path)
    assert str(path) in str(info.value)

  def test_empty_dir(self, folder: Path) -> None:
    with pytest.raises(CorpusError) as info:
      load_corpus(folder)
    assert "**/*.md" in str(info.value)

  def test_no_match_names_the_glob(self, folder: Path) -> None:
    put(folder, "a.md")
    with pytest.raises(CorpusError) as info:
      load_corpus(folder, "*.rst")
    assert "*.rst" in str(info.value)

  def test_invalid_utf8_names_the_file(self, folder: Path) -> None:
    put(folder, "ok.md")
    bad = put(folder, "sub/bad.md", b"caf\xe9\n")
    with pytest.raises(CorpusError) as info:
      load_corpus(folder)
    assert str(bad.resolve()) in str(info.value) or "bad.md" in str(info.value)
    assert "UTF-8" in str(info.value)

  def test_latin1_and_utf16_are_rejected(self, folder: Path) -> None:
    put(folder, "a.md", "áé".encode("utf-16"))
    with pytest.raises(CorpusError):
      load_corpus(folder)

  def test_truncated_multibyte_sequence(self, folder: Path) -> None:
    put(folder, "a.md", "ñ".encode()[:1])
    with pytest.raises(CorpusError):
      load_corpus(folder)

  def test_overlong_encoding_and_surrogates_are_rejected(self, folder: Path) -> None:
    put(folder, "a.md", b"\xc0\xaf")
    put(folder, "b.md", b"\xed\xa0\x80")  # a lone surrogate encoded in UTF-8
    with pytest.raises(CorpusError):
      load_corpus(folder)

  @pytest.mark.parametrize("glob", ["", "/etc/*", "/abs/**/*.md"])
  def test_unusable_globs_are_corpus_errors(self, folder: Path, glob: str) -> None:
    put(folder, "a.md")
    with pytest.raises(CorpusError):
      load_corpus(folder, glob)

  @pytest.mark.parametrize(
    "glob", ["../*.md", "../**/*.md", "sub/../../*.md", "*/../../*.md"]
  )
  def test_globs_leaving_the_folder_are_rejected(self, folder: Path, glob: str) -> None:
    put(folder, "sub/a.md")
    put(folder.parent, "outside.md")
    with pytest.raises(CorpusError):
      load_corpus(folder, glob)

  def test_glob_dotdot_inside_the_folder_does_not_leak_outside_files(
    self, folder: Path
  ) -> None:
    put(folder, "sub/a.md")
    put(folder.parent, "secret.md", "SECRET")
    try:
      corpus = load_corpus(folder, "sub/../*.md")
    except CorpusError:
      return
    assert all(".." not in doc.source for doc in corpus.documents)

  def test_symlink_pointing_outside_is_not_read_as_outside_source(
    self, folder: Path, tmp_path: Path
  ) -> None:
    outside = put(tmp_path / "elsewhere", "o.md", "OUT")
    put(folder, "in.md")
    (folder / "escape.md").symlink_to(outside)
    try:
      corpus = load_corpus(folder)
    except CorpusError:
      return
    # Either way the source stays relative to the corpus folder.
    assert all(not doc.source.startswith(("/", "..")) for doc in corpus.documents)

  @pytest.mark.skipif(
    os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs permission bits that root ignores",
  )
  def test_unreadable_file_is_a_corpus_error(self, folder: Path) -> None:
    path = put(folder, "a.md")
    path.chmod(0)
    try:
      with pytest.raises(CorpusError) as info:
        load_corpus(folder)
      assert "a.md" in str(info.value)
    finally:
      path.chmod(0o644)

  def test_error_message_is_in_spanish(self, folder: Path) -> None:
    with pytest.raises(CorpusError) as info:
      load_corpus(folder)
    assert "ningún archivo" in str(info.value)

  def test_corpus_error_is_an_exception_with_plain_message(self) -> None:
    assert str(CorpusError("boom")) == "boom"

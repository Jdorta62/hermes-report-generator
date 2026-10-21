"""Stage template changes to protected live-data files for on-demand review.

Protected files are in Copier's `_skip_if_exists`, so `copier update` never
touches them. A Copier migration runs `collect` after every update: for each
protected file whose *template* version changed between the old and the new
template version, it saves both versions under
`.claude/template-updates/<from>-to-<to>/` (gitignored). The project's own
file is never modified until `apply` is run.

  collect --src DIR --from REF --to REF   (run by the Copier migration)
  status                                  list pending updates and their diffs
  apply [FILE ...]                        3-way merge the oldest pending update
  clean [--all]                           drop the oldest (or every) pending update
"""

import argparse
import difflib
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Must match the live-data entries of `_skip_if_exists` in copier.yml.
PROTECTED = (
  ".claude/CLAUDE.project.md",
  ".claude/api/INDEX.api",
  ".claude/api/STATUS.md",
)
CLAUDE_DIR = ".claude/"
UPDATES_DIR = Path(".claude") / "template-updates"
TEMPLATE_DIR = "template"
OLD_SUFFIX = ".old"
NEW_SUFFIX = ".new"


class UpdateError(Exception):
  """A pending update cannot be processed."""


def stage_path(folder: Path, rel: str) -> Path:
  """Where `rel` (a protected path) is staged inside `folder`, minus `.claude/`."""
  return folder / rel.removeprefix(CLAUDE_DIR)


def safe_name(ref: str) -> str:
  return re.sub(r"[^A-Za-z0-9._-]", "_", ref)


def version_key(name: str) -> list[int | str]:
  """Natural sort key so v0.0.10 sorts after v0.0.9."""
  return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def template_file(src: Path, ref: str, rel: str) -> str | None:
  """Content of `template/<rel>` at `ref` in the template repo, or None."""
  result = subprocess.run(
    ["git", "-C", str(src), "show", f"{ref}:{TEMPLATE_DIR}/{rel}"],
    capture_output=True,
    text=True,
  )
  return result.stdout if result.returncode == 0 else None


def collect(root: Path, src: Path, old_ref: str, new_ref: str) -> list[str]:
  """Stage protected files whose template version changed; return them."""
  folder = root / UPDATES_DIR / f"{safe_name(old_ref)}-to-{safe_name(new_ref)}"
  changed: list[str] = []
  for rel in PROTECTED:
    old = template_file(src, old_ref, rel)
    new = template_file(src, new_ref, rel)
    if new is None or old == new:
      continue
    target = stage_path(folder, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    if old is not None:
      Path(f"{target}{OLD_SUFFIX}").write_text(old, encoding="utf-8")
    Path(f"{target}{NEW_SUFFIX}").write_text(new, encoding="utf-8")
    changed.append(rel)
  return changed


def pending_folders(root: Path) -> list[Path]:
  updates = root / UPDATES_DIR
  if not updates.is_dir():
    return []
  folders = [d for d in updates.iterdir() if d.is_dir()]
  return sorted(folders, key=lambda d: version_key(d.name))


def pending_files(folder: Path) -> list[str]:
  """Protected paths (relative to the project root) staged in `folder`."""
  rels = {
    CLAUDE_DIR + path.relative_to(folder).as_posix().removesuffix(NEW_SUFFIX)
    for path in folder.rglob(f"*{NEW_SUFFIX}")
  }
  return sorted(rels)


def staged(folder: Path, rel: str) -> tuple[Path | None, Path]:
  target = stage_path(folder, rel)
  old = Path(f"{target}{OLD_SUFFIX}")
  return (old if old.is_file() else None), Path(f"{target}{NEW_SUFFIX}")


def status(root: Path) -> str:
  folders = pending_folders(root)
  if not folders:
    return "No pending template updates."
  lines: list[str] = []
  for folder in folders:
    lines.append(f"{folder.relative_to(root)}/")
    for rel in pending_files(folder):
      old, new = staged(folder, rel)
      old_text = old.read_text(encoding="utf-8") if old else ""
      diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile=f"template {rel} (before)",
        tofile=f"template {rel} (after)",
      )
      lines.append(f"  {rel}")
      lines.extend(f"    {line.rstrip()}" for line in diff)
  lines.append("")
  lines.append("Apply with `uv run just updates-apply` (oldest first) or by hand,")
  lines.append("then drop with `uv run just updates-clean`.")
  return "\n".join(lines)


def merge(root: Path, folder: Path, rel: str) -> bool:
  """3-way merge one staged file into the project file; True if clean."""
  old, new = staged(folder, rel)
  live = root / rel
  if not live.is_file():
    live.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(new, live)
    return True
  base = old if old is not None else Path(f"{stage_path(folder, rel)}.empty-base")
  if old is None:
    base.write_text("", encoding="utf-8")
  result = subprocess.run(
    [
      "git",
      "merge-file",
      "-L",
      "yours",
      "-L",
      "template before",
      "-L",
      "template after",
      str(live),
      str(base),
      str(new),
    ],
    capture_output=True,
    text=True,
  )
  if old is None:
    base.unlink()
  if result.returncode < 0:
    raise UpdateError(f"git merge-file failed on {rel}: {result.stderr.strip()}")
  return result.returncode == 0


def drop(folder: Path, rel: str) -> None:
  old, new = staged(folder, rel)
  for path in (old, new):
    if path is not None:
      path.unlink()
  remove_empty_dirs(folder)


def remove_empty_dirs(folder: Path) -> None:
  for directory in sorted((d for d in folder.rglob("*") if d.is_dir()), reverse=True):
    if not any(directory.iterdir()):
      directory.rmdir()
  if folder.is_dir() and not any(folder.iterdir()):
    folder.rmdir()
  updates = folder.parent
  if updates.is_dir() and not any(updates.iterdir()):
    updates.rmdir()


def apply(root: Path, only: list[str]) -> tuple[list[str], list[str]]:
  """Merge the oldest pending update; return (clean, conflicted) paths."""
  folders = pending_folders(root)
  if not folders:
    raise UpdateError("no pending template updates")
  folder = folders[0]
  rels = pending_files(folder)
  if only:
    unknown = sorted(set(only) - set(rels))
    if unknown:
      raise UpdateError(f"not pending in {folder.name}: {', '.join(unknown)}")
    rels = [rel for rel in rels if rel in only]
  clean: list[str] = []
  conflicted: list[str] = []
  for rel in rels:
    (clean if merge(root, folder, rel) else conflicted).append(rel)
    drop(folder, rel)
  return clean, conflicted


def clean_updates(root: Path, everything: bool) -> list[Path]:
  folders = pending_folders(root)
  targets = folders if everything else folders[:1]
  for folder in targets:
    shutil.rmtree(folder)
  updates = root / UPDATES_DIR
  if updates.is_dir() and not any(updates.iterdir()):
    updates.rmdir()
  return targets


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
  sub = parser.add_subparsers(dest="command", required=True)
  collect_cmd = sub.add_parser("collect")
  collect_cmd.add_argument("--src", type=Path, required=True)
  collect_cmd.add_argument("--from", dest="old_ref", required=True)
  collect_cmd.add_argument("--to", dest="new_ref", required=True)
  sub.add_parser("status")
  apply_cmd = sub.add_parser("apply")
  apply_cmd.add_argument("files", nargs="*")
  clean_cmd = sub.add_parser("clean")
  clean_cmd.add_argument("--all", action="store_true")
  args = parser.parse_args(argv)
  root: Path = args.root.resolve()

  try:
    if args.command == "collect":
      changed = collect(root, args.src, args.old_ref, args.new_ref)
      if changed:
        print(f"Template {args.old_ref} -> {args.new_ref} changed protected files:")
        print("\n".join(f"  - {rel}" for rel in changed))
        print("Your files were NOT modified. Review with `uv run just updates`.")
    elif args.command == "status":
      print(status(root))
    elif args.command == "apply":
      clean, conflicted = apply(root, args.files)
      for rel in clean:
        print(f"merged {rel}")
      for rel in conflicted:
        print(f"CONFLICT {rel}: resolve the <<<<<<< markers by hand")
      return 1 if conflicted else 0
    else:
      for folder in clean_updates(root, args.all):
        print(f"removed {folder.relative_to(root)}")
  except UpdateError as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())

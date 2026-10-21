"""Print the kanban board and flag inconsistencies.

Tasks are listed by sprint and folder with their `review:` value. The exit
code is 1 when any inconsistency is found, 0 otherwise.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from relocate_api import API_DIR, INDEX_NAME, STATUS_NAME, dotted_name, status_module

KANBAN_DIR = Path(".claude") / "kanban"
SPRINTS_DIR = KANBAN_DIR / "sprints"
BACKLOG_DIR = KANBAN_DIR / "backlog"
FOLDERS = ("todo", "in-progress", "done")
ALLOWED_REVIEWS = {
  "todo": {"", "pending"},
  "in-progress": {"", "pending", "FAIL"},
  "done": {"PASS"},
}
ID_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
SPRINT_PATTERN = re.compile(r"^sprint-(\d+)$")
EPIC_PATTERN = re.compile(r"^(\d+)-.+\.md$")


@dataclass(frozen=True)
class Task:
  """One task file on the board."""

  path: Path
  sprint: str
  folder: str
  meta: dict[str, str] | None

  @property
  def sprint_no(self) -> int:
    return int(self.sprint.removeprefix("sprint-"))

  @property
  def id_parts(self) -> tuple[int, int, int] | None:
    task_id = (self.meta or {}).get("id", "")
    if not ID_PATTERN.match(task_id):
      return None
    epic, sprint, number = (int(part) for part in task_id.split("."))
    return epic, sprint, number


@dataclass(frozen=True)
class Epic:
  """One epic file in the backlog."""

  path: Path
  number: int
  done: bool


@dataclass
class Board:
  """Tasks found plus every inconsistency detected."""

  tasks: list[Task] = field(default_factory=list)
  epics: list[Epic] = field(default_factory=list)
  sprint_numbers: list[int] = field(default_factory=list)
  issues: list[str] = field(default_factory=list)

  @property
  def current_sprint(self) -> int | None:
    """Highest sprint number found; every lower sprint is closed."""
    return self.sprint_numbers[-1] if self.sprint_numbers else None


def parse_frontmatter(text: str) -> dict[str, str] | None:
  """Flat `key: value` frontmatter between leading `---` lines."""
  lines = text.splitlines()
  if not lines or lines[0].strip() != "---":
    return None
  meta: dict[str, str] = {}
  for line in lines[1:]:
    if line.strip() == "---":
      return meta
    key, sep, value = line.partition(":")
    if sep:
      meta[key.strip()] = value.strip()
  return None


def sprint_number(directory: Path) -> int:
  match = SPRINT_PATTERN.match(directory.name)
  return int(match.group(1)) if match else -1


def load_tasks(root: Path, board: Board) -> None:
  sprints_root = root / SPRINTS_DIR
  if not sprints_root.is_dir():
    board.issues.append(f"{SPRINTS_DIR} does not exist")
    return
  sprints = [d for d in sprints_root.iterdir() if d.is_dir()]
  for sprint in sorted(sprints, key=lambda d: (sprint_number(d), d.name)):
    if sprint_number(sprint) < 0:
      board.issues.append(f"{sprint.relative_to(root)}: not named sprint-<N>")
      continue
    board.sprint_numbers.append(sprint_number(sprint))
    for child in sorted(sprint.iterdir()):
      if child.is_dir() and child.name not in FOLDERS:
        board.issues.append(f"{child.relative_to(root)}: unexpected folder")
      elif child.is_file() and child.name != ".gitkeep":
        board.issues.append(
          f"{child.relative_to(root)}: file outside todo/in-progress/done"
        )
    for folder in FOLDERS:
      directory = sprint / folder
      if not directory.is_dir():
        continue
      for path in sorted(directory.glob("*.md")):
        meta = parse_frontmatter(path.read_text(encoding="utf-8"))
        board.tasks.append(Task(path.relative_to(root), sprint.name, folder, meta))


def check_task(task: Task, board: Board) -> None:
  where = task.path.as_posix()
  if task.meta is None:
    board.issues.append(f"{where}: missing or unterminated frontmatter")
    return
  task_id = task.meta.get("id", "")
  review = task.meta.get("review", "")
  parts = task.id_parts
  if parts is None:
    board.issues.append(f"{where}: id {task_id!r} is not <epic>.<sprint>.<n>")
  else:
    if not task.path.name.startswith(f"{task_id}-"):
      board.issues.append(f"{where}: filename does not start with id {task_id}-")
    if parts[1] != task.sprint_no:
      board.issues.append(f"{where}: id sprint {parts[1]} but file is in {task.sprint}")
  sprint = task.meta.get("sprint")
  if sprint is not None and sprint != task.sprint:
    board.issues.append(f"{where}: sprint: {sprint} but file is in {task.sprint}")
  if review not in ALLOWED_REVIEWS[task.folder]:
    shown = review or "(empty)"
    board.issues.append(f"{where}: review: {shown} not allowed in {task.folder}/")


def load_epics(root: Path, board: Board) -> None:
  backlog = root / BACKLOG_DIR
  for directory, done in ((backlog, False), (backlog / "done", True)):
    if not directory.is_dir():
      continue
    for path in sorted(directory.iterdir()):
      if path.is_dir() or path.name == ".gitkeep":
        continue
      match = EPIC_PATTERN.match(path.name)
      if match is None:
        board.issues.append(f"{path.relative_to(root)}: not named <N>-<slug>.md")
        continue
      board.epics.append(Epic(path.relative_to(root), int(match.group(1)), done))
  seen: dict[int, Path] = {}
  for epic in board.epics:
    if epic.number in seen:
      board.issues.append(
        f"{epic.path.as_posix()}: duplicate epic {epic.number}"
        f" (also {seen[epic.number].as_posix()})"
      )
    else:
      seen[epic.number] = epic.path


def check_epics(board: Board) -> None:
  epics = {epic.number: epic for epic in board.epics}
  for task in board.tasks:
    parts = task.id_parts
    if parts is None:
      continue
    epic = epics.get(parts[0])
    where = task.path.as_posix()
    if epic is None:
      board.issues.append(f"{where}: epic {parts[0]} has no file in {BACKLOG_DIR}")
    elif epic.done and task.folder != "done":
      board.issues.append(
        f"{where}: epic {parts[0]} is closed but this task is not done"
      )


def check_closed_sprints(board: Board) -> None:
  current = board.current_sprint
  for task in board.tasks:
    if task.sprint_no != current and task.folder != "done":
      board.issues.append(
        f"{task.path.as_posix()}: {task.sprint} is closed but the task is in {task.folder}/"
      )


def check_duplicate_ids(board: Board) -> None:
  seen: dict[str, Path] = {}
  for task in board.tasks:
    task_id = (task.meta or {}).get("id", "")
    if not task_id:
      continue
    if task_id in seen:
      board.issues.append(
        f"{task.path.as_posix()}: duplicate id {task_id} (also {seen[task_id].as_posix()})"
      )
    else:
      seen[task_id] = task.path


def check_status(root: Path, board: Board) -> None:
  api_root = root / API_DIR
  status = api_root / STATUS_NAME
  if not status.is_file():
    return
  documented = {
    dotted_name(path.relative_to(api_root))
    for path in api_root.rglob("*.api")
    if path.name != INDEX_NAME
  }
  for number, line in enumerate(status.read_text(encoding="utf-8").splitlines(), 1):
    name = status_module(line)
    if name is not None and name not in documented:
      board.issues.append(f"{API_DIR / STATUS_NAME}:{number}: {name} has no .api file")


def build(root: Path) -> Board:
  board = Board()
  load_tasks(root, board)
  load_epics(root, board)
  for task in board.tasks:
    check_task(task, board)
  check_duplicate_ids(board)
  check_epics(board)
  check_closed_sprints(board)
  check_status(root, board)
  return board


def render(board: Board) -> str:
  current = board.current_sprint
  lines = [f"current sprint: sprint-{current}" if current is not None else "no sprints"]
  for sprint in dict.fromkeys(task.sprint for task in board.tasks):
    lines.append(sprint if sprint != f"sprint-{current}" else f"{sprint} (current)")
    for folder in FOLDERS:
      tasks = [t for t in board.tasks if t.sprint == sprint and t.folder == folder]
      if not tasks:
        continue
      lines.append(f"  {folder}/")
      for task in tasks:
        review = (task.meta or {}).get("review", "") or "-"
        lines.append(f"    {task.path.name:<50} review: {review}")
  if not board.tasks:
    lines.append("(no tasks)")
  if board.issues:
    lines.append("")
    lines.append(f"Inconsistencies ({len(board.issues)}):")
    lines.extend(f"  - {issue}" for issue in board.issues)
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
  args = parser.parse_args(argv)
  board = build(args.root.resolve())
  print(render(board))
  return 1 if board.issues else 0


if __name__ == "__main__":
  sys.exit(main())

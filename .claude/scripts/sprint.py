"""Open the next sprint, closing the current one.

Refuses while the current sprint still has tasks in todo/ or in-progress/:
once a sprint is closed no task may go back to it.
"""

import argparse
import sys
from pathlib import Path

from board import FOLDERS, SPRINTS_DIR, Board, load_tasks


class SprintError(Exception):
  """The next sprint cannot be opened."""


def open_next_sprint(root: Path) -> Path:
  """Create `sprint-<N+1>/{todo,in-progress,done}` and return its path."""
  board = Board()
  load_tasks(root, board)
  current = board.current_sprint
  pending = [
    task.path.as_posix()
    for task in board.tasks
    if task.sprint_no == current and task.folder != "done"
  ]
  if pending:
    listed = "\n".join(f"  - {path}" for path in pending)
    raise SprintError(f"sprint-{current} still has unfinished tasks:\n{listed}")
  sprint = root / SPRINTS_DIR / f"sprint-{(current or 0) + 1}"
  for folder in FOLDERS:
    (sprint / folder).mkdir(parents=True, exist_ok=True)
    (sprint / folder / ".gitkeep").touch()
  return sprint


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
  args = parser.parse_args(argv)
  root: Path = args.root.resolve()
  try:
    sprint = open_next_sprint(root)
  except SprintError as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  print(f"opened {sprint.relative_to(root)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

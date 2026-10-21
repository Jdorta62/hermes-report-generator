"""PostToolUse hook: run `ruff format` on the Python file just edited.

Reads the hook payload from stdin. Never blocks the tool call: always exits 0.
"""

import json
import os
import subprocess
import sys
from typing import Any


def edited_file(payload: Any) -> str | None:
  """Return `tool_input.file_path` from the payload if it is a .py file."""
  if not isinstance(payload, dict):
    return None
  tool_input = payload.get("tool_input")
  if not isinstance(tool_input, dict):
    return None
  file_path = tool_input.get("file_path")
  if isinstance(file_path, str) and file_path.endswith(".py"):
    return file_path
  return None


def main() -> int:
  try:
    payload = json.load(sys.stdin)
  except ValueError:
    return 0
  file_path = edited_file(payload)
  if file_path is None:
    return 0
  try:
    subprocess.run(
      ["uv", "run", "--no-sync", "ruff", "format", "--quiet", file_path],
      cwd=os.environ.get("CLAUDE_PROJECT_DIR") or None,
      capture_output=True,
      check=False,
      timeout=50,
    )
  except (OSError, subprocess.SubprocessError):
    pass
  return 0


if __name__ == "__main__":
  sys.exit(main())

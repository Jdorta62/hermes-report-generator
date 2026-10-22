# Command for the external .api generator (llm-apigen). Empty until wired.
APIGEN_CMD := ""

set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

# Paths checked when lint gets no arguments.
LINT_PATHS := "src tests examples .claude/scripts .claude/hooks"

default:
  @just --list

# Format and autofix the whole project.
fmt:
  uv run ruff format .
  uv run ruff check --fix .

# ruff check + ruff format --check + mypy on FILES (whole project if empty).
lint *FILES:
  #!/usr/bin/env bash
  set -euo pipefail
  if [ "$#" -eq 0 ]; then set -- {{ LINT_PATHS }}; fi
  uv run ruff check "$@"
  uv run ruff format --check "$@"
  uv run mypy "$@"

# pytest with ARGS (whole suite if empty).
test *ARGS:
  uv run pytest "$@"

# Coverage for comma-separated MODULES, running TESTS (whole suite if empty).
cov MODULES *TESTS:
  #!/usr/bin/env bash
  set -euo pipefail
  modules="$1"
  shift
  cov_args=()
  IFS=',' read -ra names <<< "$modules"
  for name in "${names[@]}"; do
    if [ -n "$name" ]; then cov_args+=("--cov=$name"); fi
  done
  if [ "${#cov_args[@]}" -eq 0 ]; then
    echo "error: cov needs at least one module, e.g. just cov \"pkg.mod\"" >&2
    exit 2
  fi
  uv run pytest "${cov_args[@]}" --cov-report=term-missing "$@"

# Full repo check: lint + test (the gate before every commit and in CI).
check: lint test

# Move hand-written .api files into .claude/api/ and rebuild INDEX.api.
relocate:
  uv run python .claude/scripts/relocate_api.py

# Show the kanban board and flag inconsistencies.
board:
  uv run python .claude/scripts/board.py

# Close the current sprint and open the next one (refuses with unfinished tasks).
sprint:
  uv run python .claude/scripts/sprint.py

# Pending template changes to protected files (CLAUDE.project.md, INDEX.api, STATUS.md).
updates:
  uv run python .claude/scripts/template_updates.py status

# 3-way merge the oldest pending template update (optionally only FILES).
updates-apply *FILES:
  uv run python .claude/scripts/template_updates.py apply "$@"

# Drop the oldest pending template update without applying it.
updates-clean:
  uv run python .claude/scripts/template_updates.py clean

# Run the .api generator (stub until APIGEN_CMD is set above).
api *ARGS:
  #!/usr/bin/env bash
  set -euo pipefail
  cmd={{ quote(APIGEN_CMD) }}
  if [ -z "$cmd" ]; then
    echo "error: APIGEN_CMD is not set; edit the variable at the top of the justfile" >&2
    exit 1
  fi
  eval "$cmd" '"$@"'

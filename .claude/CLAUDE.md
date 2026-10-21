# Hermes Report Generator

Only rules that cannot be deduced from the code or enforced by a tool.
Each rule carries its reason.

## Commands

Run every recipe as `uv run just <recipe>`.
Reason: `just` is a dev dependency installed in the venv, not a system tool.

- `just check`: lint + full test suite; run it before every commit.
  Reason: it is the same gate CI runs.
- `just lint <files>`, `just test <files>`, `just cov "<mod1,mod2>" <tests>`:
  scoped checks for the files a change touched.
  Reason: fast feedback without running the whole repo.
- `just fmt`: format and autofix.
  Reason: one entry point instead of remembering ruff flags.
- `just board`: kanban state and inconsistencies.
  Reason: the board is files; this is the only consistency check.
- `just relocate`: move hand-written `.api` files into `.claude/api/`.
  Reason: keeps `INDEX.api` and `STATUS.md` in sync deterministically.
- `just sprint`: close the current sprint and open the next one.
  Reason: a closed sprint never receives tasks again; it refuses while tasks are unfinished.

## Rules

- Use `uv` only (`uv add`, `uv remove`, `uv sync`, `uv run`); never bare `pip`.
  Reason: keeps `pyproject.toml` and `uv.lock` in sync.
- Type-annotate every public function and method signature.
  Reason: mypy strict and the `.api` interface files depend on them.
- Identifiers, comments and docstrings in English.
  Reason: one language across code and agent output, whatever the chat language.
- Do not add or upgrade dependencies without explicit approval.
  Reason: every dependency is long-term maintenance and supply-chain risk.
- Never read or write `.env` / `.env.*`.
  Reason: they hold secrets.
- Never write under `.claude/kanban/backlog/`.
  Reason: epic planning belongs to the human.

## Workflow

Implementation goes through `/run-task <task file>` (Manuel implements,
`just relocate`, Carlos reviews), then `just check` and `/commit`.
Create tasks with `/new-task`; load context with `/prime`.
Design or review architecture with `/architect` before creating the tasks.
Reason: keeps implementation and review separate and the verdict in the task file.

@CLAUDE.project.md

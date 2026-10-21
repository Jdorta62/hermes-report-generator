---
description: Load project context and report a short summary of the board state
---
Load the project context, then report. Do not start any work.

1. Read `.claude/CLAUDE.md` and `.claude/CLAUDE.project.md`.
2. Run `uv run just board` (a non-zero exit only means it found
   inconsistencies; keep its output).
3. Read `.claude/api/INDEX.api`.
4. If `.claude/template-updates/` exists, run `uv run just updates`.

Report in at most 10 lines:
- current sprint (highest `sprint-N`) and its tasks per folder
  (`todo/`, `in-progress/`, `done/`) with their `review:` value;
- every inconsistency `just board` flagged;
- number of modules indexed in `INDEX.api`;
- pending template updates, if any (tell me to review them);
- the obvious next step (e.g. "run `/run-task <path>`").

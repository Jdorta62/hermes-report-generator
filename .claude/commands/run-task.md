---
description: Run a task through Manuel, just relocate and Carlos (max 3 attempts)
argument-hint: <path to task file>
---
Run the task file `$ARGUMENTS` through the pipeline below. You orchestrate;
you never write production code, tests, or the `## Manuel` / `## Carlos`
sections yourself.

## Preconditions (stop and report if any fails)

1. The file exists and lives in `todo/` or `in-progress/` of the current
   sprint (the highest `.claude/kanban/sprints/sprint-N/`). Lower sprints
   are closed: never run a task there.
2. The git tree is clean except for that task file itself:
   `git status --porcelain` lists nothing else.

## Branch

3. Take `<id>` from the frontmatter and `<slug>` from the filename
   (`<id>-<slug>.md`). The branch is `task/<id>-<slug>`.
   - Already on it: stay.
   - It exists: `git switch task/<id>-<slug>`.
   - Otherwise: `git switch -c task/<id>-<slug>` from the current HEAD.
   Never switch to `main` or any other base branch first.

## Loop (attempt 1 to 3)

4. Invoke the `manuel` subagent (Agent tool) with the task file's current
   path and the attempt number. On a retry, tell him to read `## Carlos`
   first; the file is already in `in-progress/`.
5. Run `uv run just relocate`. Relay any warning it prints. If it exits
   non-zero, stop and report.
6. Invoke the `carlos` subagent (Agent tool) with the task file's path in
   `in-progress/`.
7. Read `review:` from the task file (it may now be in `done/`).
   - `PASS`: leave the loop.
   - `FAIL` and attempt < 3: start the next attempt at step 4.
   - `FAIL` on attempt 3: stop. Escalate to me with Carlos's verdict and
     the failures he listed. Do not try anything else.

## Finish

On PASS, report the files changed and Carlos's verdict, then tell me to run
`uv run just check` and `/commit`.

Never commit, never push, never touch `.claude/kanban/backlog/`.

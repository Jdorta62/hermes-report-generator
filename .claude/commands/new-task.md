---
description: Create a task file in the current sprint's todo/ from TASK-TEMPLATE
argument-hint: [epic number] <short task title>
---
Create a new task: $ARGUMENTS

1. **Current sprint**: the highest `N` among
   `.claude/kanban/sprints/sprint-N/`. Lower sprints are closed: never
   create, move or reopen a task there. If there is no sprint, stop and
   tell me to run `uv run just sprint`.
2. **Epic**: if the arguments start with a number, that is the epic and the
   rest is the title. Otherwise use the epic of the tasks already in
   `sprint-N` if they all share one; if the sprint is empty or mixes epics,
   ask me. The epic must have a file `.claude/kanban/backlog/<epic>-*.md`
   (not in `backlog/done/`, which holds closed epics); if not, stop and
   tell me.
3. **Next id**: `<epic>.<N>.<k>`. Collect the last segment `k` of every task
   file in `sprint-N/todo/`, `sprint-N/in-progress/` and `sprint-N/done/`
   (filenames `<id>-<slug>.md`, any epic) and use the highest `k` + 1
   (gaps are never filled), or `1` if the sprint has no tasks.
4. **Slug**: short kebab-case English summary of the title (ASCII only,
   at most 6 words).
5. Copy `.claude/kanban/TASK-TEMPLATE.md` to
   `.claude/kanban/sprints/sprint-N/todo/<id>-<slug>.md` and set the
   frontmatter: `id: <id>`, `sprint: sprint-N`, `review: pending`.
6. Fill `## Tarea` (scope and acceptance criteria) and `## Contexto`
   (decisions taken, relevant files, constraints) from this conversation and
   from the epic file: carry over its "Decisiones ya tomadas" and
   "Restricciones" that apply to this task. The task must be
   self-sufficient: it will be run from a cleared session with only this
   file, `.claude/CLAUDE.md` and the repository. If something needed is missing or
   ambiguous, ask me before writing it. Remove the template's HTML comments
   and placeholder bullets you did not fill.
7. Leave `## Manuel` and `## Carlos` empty.

Never write under `.claude/kanban/backlog/` (read only). Do not commit.
Finish by running `uv run just board` and telling me the path of the new
file.

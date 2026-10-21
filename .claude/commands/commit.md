---
description: Run just check, stage, propose a Conventional Commit message with agent trailers, and commit after my approval
argument-hint: [task file path]
---
1. Run `uv run just check`. If it fails, stop and report; never commit red.
2. Inspect `git status` and `git diff`, then stage the relevant paths with
   `git add <paths>`. Never stage `.env*` files.
3. Write a Conventional Commit message in English:
   `<type>(<optional scope>): <imperative summary>` (at most 72 chars),
   plus a short body explaining why when the change is not trivial.
4. Trailers: the task file is `$ARGUMENTS` if given, otherwise the task file
   under `.claude/kanban/sprints/` included in this commit (if none, add no
   trailer). For each of its `## Manuel` and `## Carlos` sections that has
   content, add that agent's `Co-Authored-By` trailer copied exactly from
   the "When committing" line of `.claude/agents/manuel.md` or
   `.claude/agents/carlos.md`. Add nothing else: no other trailer, no
   attribution line.
5. Show me the staged file list and the full commit message (trailers
   included), then stop and wait. Commit only after I explicitly approve;
   if I ask for changes, apply them and show the message again.
6. Commit with `git commit -F -` (heredoc). Never push. Report the hash.

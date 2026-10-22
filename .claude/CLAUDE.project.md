# Project rules (project-owned)

Rules specific to this project. Copier never overwrites this file on
`copier update`. Add each rule with a one-line reason.

## What this repo is

`hermes-report-generator` (package `hermes_report`) runs the whole Hermes
pipeline (Ingest → Qdrant → Retrieve → Enrich → Generator) over a folder of
Markdown files and a JSON of questions with expected answers. It grades each
answer with rules and an LLM judge, writes `run.json`, and renders
`report.html` from it. `compare` renders several runs side by side.
Design: `.claude/architect/designs/report-generator/design.md`.
Reason: it is a tool over the modules, not a module; it must not grow
pipeline logic of its own.

## Rules

- Use only the public API of the Hermes modules: names exported by the
  package root, plus `hermes_enrich.techniques`, `hermes_enrich.tokens`,
  `hermes_generator.answer.strategy.DefaultStrategy`,
  `hermes_generator.answer.llm_ollama.OllamaChatClient`,
  `hermes_ingest_md.chunking` and `hermes_retrieval.contracts`.
  Reason: the modules are pinned by commit and their internals change.
- Only Qdrant collections whose name starts with `hermes_report_` are ever
  created, deleted or written.
  Reason: José's Qdrant holds other data.
- Tests never touch José's resources: Qdrant `":memory:"`, `HashEmbedder`,
  `EchoLLM`. Tests that need real Ollama or Qdrant carry the `integration`
  marker and skip when the service is unreachable.
  Reason: the suite must run anywhere, without side effects.
- Text shown to people (report, CLI messages) is in Spanish; identifiers,
  comments and docstrings in English.
  Reason: José reads the reports; the code follows the ecosystem.
- The report is rendered only from the run record (`run.json`), never from
  live objects.
  Reason: `render` and `compare` must give the same HTML from the files.
- Sibling repositories, for reading only: the playground this tool ports
  is `/home/jose/CT/Hermes/Hermes-Generator/hermes-generator/playground/`
  (branch `dev`); the modules are under `/home/jose/CT/Hermes/`.
  Reason: the playground's tracing, evaluation and report are the starting
  point.

## Branches and commits

- `dev` is the integration branch. Task branches start from `dev` and are
  merged back with `--no-ff`; they stay local. Push `main` and `dev` only
  when an epic closes and the remote exists.
  Reason: José's preference; he creates `Jdorta62/hermes-report-generator`.

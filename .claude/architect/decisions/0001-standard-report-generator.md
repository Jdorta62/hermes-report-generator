# 0001. A standard report generator in its own repository
Status: accepted
Date: 2026-10-21

## Context
The playground of hermes-generator traces every phase of the pipeline into
an HTML report José likes, but its corpora and questions are written in
Python, ingest and retrieval run through bridges to sibling virtual
environments, and it cannot show the raw chunks stored in Qdrant.

## Decision
- A new repository, `hermes-report-generator` (package `hermes_report`),
  takes a TOML run file, a folder of Markdown files and a JSON of questions.
- The four modules are git dependencies in one venv, pinned by commit; the
  whole pipeline runs in process and is traced.
- Every run writes `run.json`; the HTML report is a pure rendering of it.
- Grading: the playground rules plus an LLM judge.
- The playground stays as it is.

## Alternatives
- Extending the playground: rejected; it lives in the generator, which must
  not depend on the other modules.
- Bridges to sibling venvs: rejected by José; one venv lets every phase be
  traced and needs no sibling checkouts.
- A JavaScript viewer over `run.json`: rejected for most of the report; only
  the chunk explorer is rendered in the browser, because the points are
  shared by every question.

## Consequences
- Bumping a module means bumping its pinned commit here.
- CI needs a token that reads the five private repositories (FU-1).

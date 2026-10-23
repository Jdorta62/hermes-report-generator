# 0003. The experiment notebook does its own retrieval and generation
Status: accepted
Date: 2026-10-22

## Context
José wants a Jupyter notebook to sweep collections, models, `top_k` and
filters against existing Qdrant collections, cell by cell. The project rule
says the tool has no pipeline logic of its own.

## Decision
`notebooks/rag_evaluation.ipynb` embeds, searches Qdrant and generates in
its own cells. This exception is confined to `notebooks/`: the
`hermes_report` package gains no retrieval or generation. The notebook
reuses `hermes_report.providers` and `hermes_report.metrics`, only reads
collections not named `hermes_report_*`, and its dependencies (pandas,
jupyterlab, ipykernel, nbclient) live in the `notebook` dependency group.

## Alternatives
- Notebook over `hermes_report.runner`: rejected by José; it needs a corpus
  and ingestion and cannot probe arbitrary existing collections.

## Consequences
The notebook can drift from the Hermes pipeline (prompt, enrichment). It is
tested end to end offline with `nbclient` so it at least keeps running.

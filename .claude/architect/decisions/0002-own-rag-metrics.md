# 0002. Own RAG metrics with the existing judge, no evaluation library
Status: accepted
Date: 2026-10-22

## Context
Epic 5 adds RAG metrics (faithfulness, answer relevance, context precision
and recall, correctness, citations and others). The judge already runs on
Ollama through `JudgeClient`; new dependencies need José's approval.

## Decision
`hermes_report.metrics` implements every metric itself on top of
`JudgeClient`, the run's `Embedder` and the `hermes_enrich` token counter.
BLEU and ROUGE-L are pure Python. No ragas, deepeval, langchain,
sacrebleu or rouge-score.

## Alternatives
- ragas: pulls langchain and provider wrappers, defaults to OpenAI, unstable API.
- deepeval: heavy, telemetry to disable, its own LLM abstraction.

## Consequences
Prompts, parsing and formulas are ours to test and maintain; results land
in `run.json` and the report like any other phase, and work offline with
the echo judge and the hash embedder.

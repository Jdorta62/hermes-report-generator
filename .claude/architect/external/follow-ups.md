# Follow-ups

Changes outside this repository, found while building it.

## FU-1 CI: one read token for the five private dependencies

- Found: 2026-10-21, design.
- Problem: `uv sync` fetches hermes-core over https and hermes-enrich,
  hermes-generator, hermes-retrieval and hermes-ingest-md
  (`hermes-ingestion` organisation) over ssh. The CI job token cannot read
  them.
- Expected fix: José adds a secret `HERMES_DEPS_TOKEN` (a token that can
  read the five repositories); the workflow rewrites both URL forms to https
  with that token.

## FU-2 Tags for the pinned modules

- Found: 2026-10-21, design.
- Problem: hermes-retrieval, hermes-ingest-md, hermes-enrich and
  hermes-generator are pinned to commits of their `dev` branches because no
  tag uses hermes-core v0.6.0 yet.
- Expected fix: tag releases in those repositories and pin the tags here.

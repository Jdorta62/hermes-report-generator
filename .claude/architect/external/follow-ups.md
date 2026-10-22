# Follow-ups

Changes outside this repository, found while building it.

## FU-1 CI: one read token for the five private dependencies

- Found: 2026-10-21, design.
- Problem: `uv sync` fetches hermes-core over https and hermes-enrich,
  hermes-generator, hermes-retrieval and hermes-ingest-md
  (`hermes-ingestion` organisation) over ssh. The CI job token cannot read
  them.
- Expected fix: José adds one read-only fine-grained token per owner, since
  such a token belongs to a single owner: `HERMES_CORE_TOKEN` (`Jdorta62`
  repositories) and `HERMES_INGESTION_TOKEN` (`hermes-ingestion`). The
  workflow rewrites both URL forms of each owner to https with its token.

## FU-2 Tags for the pinned modules

- Found: 2026-10-21, design.
- Problem: hermes-retrieval, hermes-ingest-md, hermes-enrich and
  hermes-generator are pinned to commits of their `dev` branches because no
  tag uses hermes-core v0.6.0 yet.
- Expected fix: tag releases in those repositories and pin the tags here.

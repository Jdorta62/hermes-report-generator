# Épica 2 — Pipeline trazado de punta a punta

Con las entradas de la épica 1, esta épica ejecuta el pipeline real en un
solo proceso: ingesta con hermes-ingest-md, colección en Qdrant, recuperación
con hermes-retrieval, enriquecimiento con hermes-enrich y generación con
hermes-generator. Traza cada fase, evalúa cada respuesta (reglas y juez LLM)
y escribe `run.json`. Diseño: §5 y §9.

## Alcance

- `hermes_report.providers`: `OllamaEmbedder`, `HashEmbedder`, `EchoLLM`,
  `make_embedder`, `make_llm`, `check_ollama`.
- `hermes_report.ingest`: ingesta por documento con caché.
- `hermes_report.store`: nombre de colección, `ensure_collection`,
  `read_points` (los puntos crudos de Qdrant).
- `hermes_report.trace` (portado del playground) y
  `hermes_report.pipeline.answer_question`.
- `hermes_report.evaluate`: `rule_verdict` y `judge`.
- `hermes_report.runner.run` y la orden `hermes-report run`, que escribe
  `run.toml` resuelto y `run.json` en su carpeta de salida.

## Fuera de alcance

- El HTML (épica 3). Hasta entonces `run` solo escribe el registro.

## Decisiones ya tomadas

All are decisions by the author; they are not reopened.

- Todo en proceso; los módulos solo por su API pública (reglas del
  proyecto).
- Solo se tocan colecciones `hermes_report_*`.
- El juez responde JSON `{"verdict", "reason"}` con Ollama `format="json"`.

## Restricciones

- Los tests usan Qdrant `":memory:"`, `HashEmbedder` y `EchoLLM`; los que
  usan servicios reales llevan el marcador `integration` y se saltan si no
  responden.

## Criterios de aceptación transversales

- `uv run just check` en verde.
- `hermes-report run` sobre el corpus de prueba, en memoria y con
  `EchoLLM`, escribe un `run.json` que `read_record` vuelve a leer.

## Depende de

Épica 1.

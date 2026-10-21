# Épica 1 — Entradas y registro de la ejecución

Hoy el playground del generador tiene el corpus y las preguntas escritos en
Python y su informe se pinta desde objetos vivos. Esta épica define las
entradas estándar (fichero TOML de ejecución, carpeta de Markdown, JSON de
preguntas) y el registro `run.json` del que saldrá todo informe. Diseño:
`.claude/architect/designs/report-generator/design.md` (§3 y §6).

## Alcance

### Entradas

- `hermes_report.config`: `RunConfig` y sus secciones, `load_config(path,
  overrides)`, validación estricta (claves desconocidas son error), valores
  por defecto y variables de entorno (`QDRANT_URL`, `OLLAMA_MODEL`),
  `ConfigError` con mensajes en español, `to_dict()` y `to_toml()` del
  fichero resuelto.
- `hermes_report.questions`: `Question`, `QuestionSet`,
  `load_questions(path, corpus_sources)`, `normalise`, `QuestionsError`.
- `hermes_report.corpus`: `CorpusDocument`, `Corpus`, `load_corpus(dir,
  glob)`, `CorpusError`.

### Registro

- `hermes_report.record`: el esquema 1 de `run.json` como `TypedDict`,
  `SCHEMA_VERSION`, `write_record` y `read_record`, y los conversores de los
  tipos de Hermes (`RetrievedChunk`, `SearchedQuery`, `EnrichedContext`,
  `GenerationResult`, `PromptMessage`, `ChunkRange`...) y de los puntos de
  Qdrant (resumen del vector).

## Fuera de alcance

- Ejecutar el pipeline (épica 2), pintar el informe (épica 3), comparar
  (épica 4).

## Decisiones ya tomadas

All are decisions by the author; they are not reopened.

- TOML con `tomllib` y JSON con `json`, de la biblioteca estándar.
- Las rutas relativas se resuelven contra la carpeta del fichero de
  ejecución.
- El registro guarda resúmenes de vectores (dimensión, norma, 8 primeros
  valores), nunca el vector entero.
- `read_record` rechaza otra versión de esquema.

## Restricciones

- Estilo de la plantilla: 2 espacios, mypy strict, `from __future__ import
  annotations`, docstrings Google con `Raises:`.
- Mensajes para personas en español; código en inglés.

## Criterios de aceptación transversales

- `uv run just check` en verde.
- Un `run.toml` y un `preguntas.json` de ejemplo en `tests/data/` se cargan
  sin errores, y cada error de entrada da un mensaje que dice el fichero y
  la clave.

## Depende de

Ninguna.

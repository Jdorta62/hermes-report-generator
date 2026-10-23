# Épica 6 — Cuaderno de ingesta por colecciones

Hoy la única forma de cargar una carpeta de Markdown en Qdrant es
`hermes-report run`, que crea una sola colección con todo el corpus y además
ejecuta las preguntas. El cuaderno de experimentos (`rag_evaluation.ipynb`)
solo lee colecciones que ya existen. Para preparar el golden dataset José
necesita repartir los documentos de `notebooks/data/` en varias colecciones,
cada una con su propio troceado, y luego evaluarlas con
`rag_evaluation.ipynb`. Esta épica añade un cuaderno que hace esa ingesta
real con `hermes-ingest-md`.

## Alcance

- Un cuaderno nuevo, `notebooks/ingest_collections.ipynb`, separado de
  `rag_evaluation.ipynb`.
- Entrada: una carpeta con archivos `.md` (por defecto `notebooks/data/`).
- Una configuración central donde se declaran:
  - cuántas colecciones crear y el nombre de cada una;
  - los documentos de cada colección, por nombre exacto (ruta relativa a la
    carpeta); un mismo documento puede ir en varias colecciones;
  - el troceado (`max_chunk_size`, `hard_boundary_levels`,
    `strip_frontmatter`) y el embedder de cada colección, con valores por
    defecto comunes que cada colección puede sustituir.
- Ingesta real: `MarkdownStructuralChunker` de `hermes-ingest-md` a través
  de `hermes_report.ingest.ingest_corpus`, con la caché por documento, y
  carga en Qdrant con `hermes_report.store.ensure_collection`.
- Una tabla final por colección y un manifiesto JSON con lo que se cargó,
  listo para copiar los nombres en `rag_evaluation.ipynb`.

## Fuera de alcance

- Patrones glob para elegir documentos (José eligió nombres exactos).
- Responder preguntas o calcular métricas: eso lo hace
  `rag_evaluation.ipynb`.
- Cambiar `hermes-report run` o `rag_evaluation.ipynb`.

## Decisiones ya tomadas

All are decisions by the author; they are not reopened.

- Cuaderno nuevo, no una ampliación de `rag_evaluation.ipynb` (José,
  2026-10-23).
- Troceado y embedder propios por colección, con valores por defecto
  comunes.
- Documentos por nombre exacto; un nombre que no existe en la carpeta es un
  error que lista los disponibles.
- Si la colección ya existe y contiene exactamente esos puntos, se
  reutiliza; si no, se recrea. Una opción fuerza la recreación.
- La ingesta va por el paquete (`hermes_report.corpus`, `.ingest`,
  `.store`); el cuaderno no trocea ni embebe por su cuenta.

## Restricciones

- Solo se crean, borran o escriben colecciones cuyo nombre empieza por
  `hermes_report_`; un nombre sin ese prefijo es un error antes de tocar
  Qdrant.
- Sin dependencias nuevas: el grupo `notebook` ya tiene lo necesario.
- Las pruebas no tocan los recursos de José: modo sin conexión con
  `HashEmbedder` y Qdrant `":memory:"`.
- Celdas markdown y mensajes en español; código y comentarios en inglés.
- El cuaderno se guarda sin salidas (`just notebook-clean`).

## Criterios de aceptación transversales

- `uv run just check` en verde.
- Con Ollama y Qdrant reales, el cuaderno carga `notebooks/data/` en las
  colecciones configuradas, y `rag_evaluation.ipynb` las lee sin cambios.

## Depende de

Épica 5.

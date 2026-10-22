# Épica 4 — Comparación, ejemplos y CI

Cierra la herramienta: compara varias ejecuciones, trae dos ejemplos listos
(un corpus inventado pequeño y el Quijote por capítulos), documenta el uso y
deja el CI preparado para los repositorios privados. Diseño: §8 y FU-1.

## Alcance

- `hermes_report.render.compare` y la orden `hermes-report compare`.
- `examples/demo/`: corpus inventado, preguntas y `run.toml` (en el
  repositorio; lo usan los tests).
- `examples/quijote/`: `prepare.py` descarga el ebook 2000 de Project
  Gutenberg y lo divide en un `.md` por capítulo (ignorado por git); las 11
  preguntas del playground en `preguntas.json` y su `run.toml`.
- README en español y CI con el secreto `HERMES_DEPS_TOKEN`.

## Fuera de alcance

- Crear el secreto en GitHub (José).

## Decisiones ya tomadas

All are decisions by the author; they are not reopened.

- Las preguntas del Quijote son las del playground, con `documents`
  apuntando a los capítulos.

## Restricciones

- El texto del Quijote nunca se guarda en el repositorio.

## Criterios de aceptación transversales

- `uv run just check` en verde.
- `examples/quijote` se ejecuta de punta a punta con Ollama y Qdrant reales.

## Depende de

Épica 3.

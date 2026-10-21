# Épica 3 — El informe HTML

Esta épica pinta `report.html` solo a partir de `run.json`: resumen, corpus
e ingesta, el explorador de chunks tal como están en Qdrant, y una pestaña
por pregunta con cada fase, el mapa de documentos y los documentos tocados
en crudo. Parte del informe del playground, que a José le gusta tal como
está. Diseño: §7.

## Alcance

- `hermes_report.render`: utilidades HTML compartidas, CSS y JS como datos
  del paquete, incrustados en el HTML.
- Pestañas **Resumen**, **Corpus e ingesta** y **Chunks en Qdrant**
  (explorador global con filtros y el selector "marcar pregunta").
- Pestaña por pregunta, con todas las fases y la vista **Documentos tocados
  (crudo)**.
- La orden `hermes-report render`, y `run` pinta el informe al terminar.

## Fuera de alcance

- La comparación (épica 4).

## Decisiones ya tomadas

All are decisions by the author; they are not reopened.

- Mismo aspecto que el informe del playground; se porta su CSS.
- Los puntos se incrustan una sola vez como JSON y `report.js` pinta el
  explorador y los documentos crudos; el resto se pinta en Python.
- Sin dependencias nuevas ni recursos externos: el HTML funciona sin red.

## Restricciones

- Textos en español.

## Criterios de aceptación transversales

- `uv run just check` en verde.
- `render` sobre el mismo `run.json` da exactamente el mismo HTML.

## Depende de

Épica 2.

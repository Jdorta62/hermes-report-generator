# hermes-report-generator

Informes de calidad del pipeline Hermes sobre un corpus Markdown.

Dada una carpeta de archivos Markdown y un JSON de preguntas con su respuesta
esperada, `hermes-report` ejecuta el pipeline completo, evalúa cada respuesta y
deja dos archivos: `run.json` (todo lo que la ejecución vio y produjo) y
`report.html` (un informe autocontenido, en español, generado a partir de
`run.json`). Con `compare` se ponen varias ejecuciones lado a lado.

Es la versión estándar del `playground/` de hermes-generator. El diseño está en
`.claude/architect/designs/report-generator/design.md`; las referencias
`§N` de este documento apuntan a sus secciones.

## Qué hace

Una ejecución (`hermes-report run`) recorre estas fases:

1. **Corpus e ingesta**: lista los `.md` del corpus y los trocea con
   `MarkdownStructuralChunker` (hermes-ingest-md). Los puntos de cada documento
   se guardan en una caché de disco, así que una segunda ejecución no vuelve a
   embeber lo que no cambió (§5.1).
2. **Colección de Qdrant**: carga los puntos en una colección propia
   (`hermes_report_<título>_<huella>`). Si ya contiene exactamente esos puntos,
   se reutiliza; si no, se recrea. Solo se tocan colecciones cuyo nombre empieza
   por `hermes_report_` (§5.2).
3. **Por cada pregunta**: recuperación (hermes-retrieval), enriquecimiento con la
   cadena `neighbors`, `merge`, `doc_order` y el ajuste al presupuesto
   (hermes-enrich), y generación de la respuesta con sus citas
   (hermes-generator). Cada fase queda registrada (§5.3).
4. **Evaluación**: reglas (palabras clave y frases de evidencia) y un juez LLM
   (§5.4).
5. **Salida**: `run.toml` (el archivo de ejecución ya resuelto), `run.json` y
   `report.html` en una carpeta nueva.

## Requisitos

- **Python 3.12** y [`uv`](https://docs.astral.sh/uv/).
- **Ollama** en `http://localhost:11434` (o en `$OLLAMA_HOST`) con los dos
  modelos por defecto:

  ```bash
  ollama pull nomic-embed-text   # embeddings
  ollama pull qwen3:4b           # generación y juez
  ```

- **Qdrant** en `http://localhost:6333` (o en `$QDRANT_URL`), por ejemplo:

  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

Para el modo sin conexión (`run-offline.toml`) no hace falta ninguno de los dos.

## Instalación

```bash
uv sync
```

Los cinco módulos de Hermes (`hermes-core`, `hermes-enrich`, `hermes-generator`,
`hermes-ingest-md` y `hermes-retrieval`) son repositorios privados de GitHub:
hace falta acceso por ssh (una clave añadida a tu cuenta) y que tu cuenta pueda
leerlos. `hermes-core` se descarga por https, el resto por ssh.

El proyecto fija `link-mode = "copy"` para `uv` en `pyproject.toml`: con enlaces
duros (lo normal en `uv`), nltk se niega a leer sus datos y la ingesta falla
al trocear párrafos largos.

## Inicio rápido: la demo

`examples/demo/` es un corpus pequeño con hechos inventados (Nébula Logística,
el sistema Faro, el portal Brújula), para que ningún modelo los conozca: lo que
conteste tiene que salir de las fuentes. Tiene cuatro documentos en dos
carpetas y siete preguntas: una que se responde con un solo chunk, una que
necesita chunks vecinos, una que cruza dos documentos, una que debe rechazarse
y una sin palabras clave (solo la evalúa el juez).

### Sin conexión

```bash
uv run hermes-report run examples/demo/run-offline.toml
```

Usa un embedder `hash`, un LLM `echo` y Qdrant en memoria (`":memory:"`). Sirve
para ver el informe completo y probar la herramienta sin Ollama ni Qdrant; las
respuestas no son reales (el LLM eco repite la primera frase de la fuente `[1]`).

### Con Ollama y Qdrant

```bash
uv run hermes-report run examples/demo/run.toml
```

Al terminar imprime la carpeta, `run.json` y `report.html`. Ábrelo en un
navegador.

## El Quijote

`examples/quijote/` ejecuta el mismo pipeline sobre el texto completo de *Don
Quijote*, un archivo por capítulo. El texto no está en el repositorio: se
descarga una vez de Project Gutenberg (ebook 2000, unos 2 MB) con un script que
solo usa la biblioteca estándar.

```bash
uv run python examples/quijote/prepare.py
uv run hermes-report run examples/quijote/run.toml
```

`prepare.py` guarda el texto en `examples/quijote/data/` y escribe
`examples/quijote/corpus/parte-1/capitulo-01.md` … `parte-2/capitulo-74.md`
(ambas carpetas están en `.gitignore`). Cada archivo empieza con `# <libro>` y
`## <capítulo>`, de modo que el `heading_path` de sus chunks es
`[libro, capítulo]`. Comprueba que hay 52 + 74 capítulos y, si no, termina con
código 1.

La primera ejecución ingiere unos 1700 chunks con `nomic-embed-text` y responde
las 11 preguntas con `qwen3:4b` y el juez: cuenta con unos 3 minutos en una GPU
y bastante más sin ella. Las siguientes reutilizan la caché de ingesta y la
colección de Qdrant. Con los valores por defecto la recuperación encuentra la
evidencia en pocas preguntas: el informe lo muestra como fallos de
recuperación, que es justo lo que sirve para ajustar `top_k`, el presupuesto o
el chunking y comparar las ejecuciones con `hermes-report compare`.

`preguntas.json` tiene 11 preguntas con la respuesta esperada, palabras clave y
frases de evidencia; la última (`q11`) es un rechazo. `run.toml` usa
`top_k = 10` y un presupuesto de 3000 tokens.

## El archivo de ejecución (TOML)

Las rutas relativas se resuelven respecto a la carpeta del archivo. Una clave
desconocida es un error. Claves principales (todas, con sus valores por
defecto, en §3.1):

| Sección | Claves | Notas |
|---|---|---|
| (raíz) | `title`, `description` | `title` es obligatorio |
| `[corpus]` | `dir`, `glob` | `dir` obligatorio; `glob` por defecto `**/*.md` |
| `[questions]` | `file` | obligatorio |
| `[ingest]` | `max_chunk_size`, `hard_boundary_levels`, `strip_frontmatter` | argumentos del chunker |
| `[embedding]` | `provider`, `model` | `ollama` (por defecto) o `hash` |
| `[qdrant]` | `url`, `collection` | `$QDRANT_URL`; `":memory:"` para pruebas; la colección debe empezar por `hermes_report_` |
| `[retrieval]` | `top_k`, `candidate_factor`, `score_threshold` | |
| `[enrich]` | `chain`, `max_context_tokens`, `max_chunks` | `0` tokens = sin límite |
| `[generation]` | `provider`, `model`, `temperature`, `seed`, `response_format`, `insufficient_message` | `ollama` o `echo`; modelo por defecto `$OLLAMA_MODEL` o `qwen3:4b` |
| `[judge]` | `enabled`, `provider`, `model` | por defecto, el mismo proveedor y modelo que la generación |
| `[metrics]` | `enabled`, `include`, `exclude` | familias de métricas RAG; ver «Métricas» |
| `[output]` | `dir`, `cache_dir` | `out` y `.hermes-report-cache` |

El diseño describe cada clave en §3.1.

Opciones de la línea de órdenes que sustituyen al archivo (§3.1 y §9):

```
hermes-report run RUN_TOML [--model M] [--judge-model M] [--no-judge]
                  [--no-metrics] [--metric NOMBRE ...]
                  [--top-k N] [--budget N] [--out DIR] [--only ID ...] [--reingest]
```

`--only` se puede repetir y ejecuta solo esas preguntas; `--reingest` ignora la
caché de ingesta y recarga la colección. El código de salida es 0 si todo fue
bien, 1 si alguna pregunta falló y 2 si la entrada no es válida o un servicio
no responde.

## El archivo de preguntas (JSON)

```json
{
  "title": "Batería del Quijote",
  "questions": [
    {
      "id": "q01",
      "question": "¿Cuántos días tardó don Quijote en decidir qué nombre ponerle a su caballo?",
      "expected": "Cuatro días.",
      "keywords": ["cuatro"],
      "evidence": ["cuatro dias se le pasaron"],
      "documents": ["parte-1/capitulo-01.md"]
    }
  ]
}
```

| Campo | Obligatorio | Significado |
|---|---|---|
| `id` | no | único; por defecto `q01`, `q02`… por posición |
| `question` | sí | la pregunta |
| `expected` | sí | la respuesta correcta, para quien lee y para el juez |
| `keywords` | no | palabras que deben aparecer todas en una respuesta correcta (sin distinguir mayúsculas ni acentos) |
| `evidence` | no | frases del corpus que deben estar todas en el contexto para que la respuesta sea alcanzable |
| `documents` | no | archivos del corpus (relativos a `corpus.dir`) que contienen la respuesta; deben existir |
| `refusal` | no | `true` cuando lo correcto es decir que las fuentes no bastan |

Una pregunta sin `keywords` queda como «sin valorar» para las reglas y solo la
califica el juez (§3.2).

## Qué sale

Cada ejecución crea su carpeta dentro de `output.dir` (§9):

```
<output.dir>/<AAAAMMDD-HHMMSS>-<título>/
├── run.toml      el archivo de ejecución ya resuelto (valores por defecto y opciones aplicados)
├── run.json      el registro completo de la ejecución
└── report.html   el informe
```

### Cómo leer el informe

`report.html` es un solo archivo, con CSS y JS dentro. Tiene estas pestañas (§7):

- **Resumen**: fichas de puntuación (reglas y juez), una fila por pregunta
  (veredicto, juez, marca de desacuerdo, posición de la primera evidencia,
  segundos), el archivo de ejecución resuelto y el entorno (versiones, modelos,
  URL).
- **Corpus e ingesta**: los documentos (caracteres, chunks, si vinieron de la
  caché, segundos), los argumentos del chunker, el modelo de embeddings, las
  estadísticas de tamaño de los chunks y la colección de Qdrant.
- **Chunks en Qdrant**: el explorador de todos los puntos de la colección tal
  como están guardados (texto, payload y resumen del vector), con filtros por
  documento y por texto. El selector «marcar pregunta» colorea cada chunk según
  su estado para esa pregunta: recuperado, añadido por el enriquecimiento, en
  el contexto final, citado o descartado.
- **Una pestaña por pregunta**, con cada fase como sección: la ficha de la
  pregunta y su evaluación, el mapa de documentos, la recuperación, la semilla,
  cada paso del enriquecimiento con sus lecturas en Qdrant, el ajuste al
  presupuesto y el `EnrichedContext`, el prompt, las llamadas al LLM, el
  `GenerationResult` con sus citas, el juez y los documentos tocados en crudo.

El veredicto de las reglas (§5.4) es uno de estos:

| Veredicto | Cuándo |
|---|---|
| `correct` | las palabras clave coinciden y la evidencia está en el contexto (o no se dio evidencia) |
| `correct_without_evidence` | las palabras clave coinciden, pero la evidencia no estaba en el contexto |
| `retrieval_failure` | respuesta errónea y la evidencia nunca se recuperó |
| `budget_failure` | respuesta errónea; la evidencia se recuperó pero el presupuesto la descartó |
| `generation_refused` | el modelo rechazó responder con la evidencia en el contexto |
| `generation_wrong` | respuesta errónea con la evidencia en el contexto, o sin evidencia dada |
| `correct_refusal` / `invented_answer` | pregunta de rechazo: se rechazó / no se rechazó |
| `unrated` | sin palabras clave: solo la califica el juez |
| `error` | el pipeline lanzó una excepción |

El juez responde `correct`, `partial` o `incorrect`; el informe marca las
preguntas en las que reglas y juez no coinciden.

## Métricas

Además del veredicto de las reglas y del juez, cada pregunta recibe métricas
RAG. Son señales aparte: nunca cambian el veredicto ni el código de salida.
Se calculan por **familia** y cada familia emite una o varias puntuaciones; el
informe muestra la media de cada una y `compare` las pone lado a lado. Una
familia a la que le faltan datos queda como «no aplicable» (p. ej. sin
`documents` en la pregunta) y una que necesita el juez, sin juez, como
«omitida»; si una falla, las demás siguen.

Por defecto están activas estas familias:

| Familia | Puntuaciones | Mide |
|---|---|---|
| `doc_retrieval` | `doc_hit`, `doc_recall`, `doc_mrr` | si lo recuperado viene de los `documents` esperados |
| `evidence_retrieval` | `evidence_hit`, `evidence_recall`, `evidence_mrr`, `evidence_recall_context`, `budget_loss` | si las frases de `evidence` están en lo recuperado y en el contexto final |
| `refusal` | `false_refusal`, `invented_answer` | rechazos indebidos y respuestas inventadas |
| `cost` | `seconds_*`, `context_tokens`, `prompt_tokens`, `reply_tokens` | tiempos por fase y tokens |
| `faithfulness` | `faithfulness` | parte de la respuesta que el contexto respalda (juez) |
| `answer_relevance` | `answer_relevance` | cuánto se parece a la pregunta lo que la respuesta contesta (juez y embeddings) |
| `context_precision` | `context_precision` | si los fragmentos relevantes van antes que los irrelevantes (juez) |
| `context_recall` | `context_recall` | parte de la respuesta esperada que el contexto cubre (juez) |
| `answer_correctness` | `correctness_precision`, `correctness_recall`, `correctness_f1` | respuesta frente a la esperada (juez) |
| `citations` | `citation_precision`, `citation_recall` | si las citas `[n]` apuntan a lo que respalda cada afirmación (juez) |

Seis familias más están apagadas por defecto: `bleu`, `rouge_l`,
`noise_sensitivity`, `context_entity_recall`, `toxicity` y `bias`. Se
activan y se desactivan en el archivo de ejecución:

```toml
[metrics]
enabled = true                  # false = ninguna métrica
include = ["bleu", "rouge_l"]   # familias apagadas por defecto que se añaden
exclude = ["citations"]         # familias que no se calculan
```

o en la línea de órdenes: `--metric NOMBRE` (se puede repetir) añade una
familia a `include` y `--no-metrics` apaga todas. Un nombre desconocido es un
error de configuración. `run.json` pasa a la versión 2 (las de la versión 1
siguen leyéndose; se muestran sin métricas).

## render y compare

`render` vuelve a generar el HTML a partir de un `run.json`, sin ejecutar nada
(el resultado es el mismo que el de la ejecución):

```bash
uv run hermes-report render out/<carpeta>/run.json [--out report.html]
```

`compare` junta dos o más ejecuciones (otro modelo, otro `top_k`, otro
presupuesto…) en un informe: las claves del archivo de ejecución que difieren,
fichas por ejecución y una matriz de preguntas con enlaces al informe de cada
una (§8):

```bash
uv run hermes-report compare out/a/run.json out/b/run.json [--out compare.html]
```

Sin `--out`, escribe `compare.html` en la carpeta actual. `--out` no puede estar
vacío ni ser solo espacios.

## Notebook de experimentos

`notebooks/rag_evaluation.ipynb` sirve para probar configuraciones sobre
colecciones de Qdrant que ya existen, celda a celda: otro `top_k`, otro
modelo, filtros por payload, otra selección de métricas. Hace su propia
recuperación y generación (no pasa por `hermes-report run`) y calcula las
mismas métricas con `hermes_report.metrics`. Solo lee las colecciones que le
indicas; la única que escribe es `hermes_report_notebook_demo`, y solo sin
conexión.

```bash
uv sync --group notebook    # pandas, jupyterlab, ipykernel y nbclient
uv run jupyter lab notebooks/rag_evaluation.ipynb
```

`uv sync` a secas ya incluye el grupo `notebook`; la orden anterior es la
forma explícita de pedirlo.

Para cambiar el experimento edita solo el diccionario `CONFIG` (sección 2 del
cuaderno): colecciones, modelos de embeddings, generación y juez, `top_k`,
`score_threshold`, `filters`, `metrics`, `dataset`, el mensaje de
información insuficiente y una `note` libre. Después ejecuta de esa celda en
adelante. Con `judge.provider = None` no hay juez y las familias que lo
necesitan quedan omitidas.

Sin Ollama ni Qdrant: `HERMES_NOTEBOOK_OFFLINE=1 uv run jupyter lab`. Usa el
embedder hash, el LLM y el juez de eco, Qdrant en memoria y la colección
`hermes_report_notebook_demo` construida con `examples/demo/corpus`. Sirve
para comprobar que funciona, no para medir calidad. Opcionales:
`HERMES_NOTEBOOK_TOP_K`, `HERMES_NOTEBOOK_COLLECTIONS`, `HERMES_NOTEBOOK_DATASET`
y `HERMES_NOTEBOOK_QDRANT_URL`.

**Dónde queda el histórico.** En `notebooks/out/` (ignorada por git), o en la
carpeta de `HERMES_NOTEBOOK_OUT`:

| Archivo | Contenido |
|---|---|
| `history.csv` | una fila por (ejecución × colección): configuración, medias `<puntuación>_mean`, errores `<puntuación>_errors`, `config_hash`, `note` |
| `history.jsonl` | las mismas filas, un objeto JSON por línea |
| `<run_id>-detail.csv` | las puntuaciones de cada pregunta de esa ejecución, con la respuesta |

Si una ejecución trae columnas nuevas, las filas antiguas quedan con el valor
vacío. La última celda recarga el histórico y lo ordena por la puntuación de
`SORT_BY`. El cuaderno se guarda sin salidas (`uv run just notebook-clean`
lo comprueba); límpialo antes de hacer commit.

## Cuaderno de ingesta

`notebooks/ingest_collections.ipynb` toma una carpeta de Markdown y la ingiere
de verdad con `hermes-ingest-md` en tantas colecciones de Qdrant como declare
su `CONFIG`, cada una con sus propios documentos, su troceado y su embedder.
Mientras trabaja muestra una línea por documento, estadísticas por colección y
por documento, y una foto de Qdrant antes y después con las diferencias (nuevas,
reconstruidas, reutilizadas, fallidas, sin tocar). Solo crea o escribe
colecciones `hermes_report_*`; las demás solo se leen. Deja un manifiesto
`collections.json` en `notebooks/out/ingest/` (o en `HERMES_INGEST_OUT`).

```python
CONFIG = {
  "corpus_dir": "notebooks/data",
  "qdrant_url": "http://localhost:6333",
  "force": False,                       # reconstruir aunque no haya cambios
  "defaults": {
    "ingest": {"max_chunk_size": 1500, "hard_boundary_levels": [1, 2], "strip_frontmatter": True},
    "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
  },
  "collections": {
    "hermes_report_golden_agents": {
      "documents": ["agentes-ia-arquitecturas.md"],   # nombres exactos, sin patrones
      "ingest": {"max_chunk_size": 800},              # opcional, sobre los valores por defecto
    },
  },
}
```

Una colección se reutiliza si ya tiene exactamente los mismos puntos y se
reconstruye en caso contrario (o siempre con `force`). La configuración se
valida antes de tocar Qdrant u Ollama y el error lista todos los problemas.

Para no editar el cuaderno, `HERMES_INGEST_CONFIG=ruta.json` sustituye `CONFIG`
por un JSON con la misma forma. `HERMES_INGEST_OUT` cambia la carpeta de salida y
`HERMES_INGEST_OFFLINE=1` ejecuta sin Ollama ni Qdrant (embedder hash, Qdrant en
memoria y `examples/demo/corpus`, salvo que el JSON indique otra carpeta).

Para evaluar después las colecciones, copia la lista final del cuaderno
(«Para rag_evaluation.ipynb») en la clave `collections` de `CONFIG` en
`notebooks/rag_evaluation.ipynb` y ejecútalo con el mismo `qdrant_url`.

## Actualizar los módulos fijados

Cada módulo de Hermes está fijado a un commit (o etiqueta) en
`[tool.uv.sources]` de `pyproject.toml`, y todos se actualizan a la vez. Para
subir uno:

1. Edita el `rev` (o el `tag`) de ese módulo en `pyproject.toml`. Si cambia
   `hermes-core`, comprueba que los demás módulos usan esa misma versión.
2. Actualiza el bloqueo y el entorno:

   ```bash
   uv lock --upgrade-package hermes-enrich   # el módulo que cambias
   uv sync
   ```

3. Pasa `uv run just check` y vuelve a ejecutar la demo.

## Desarrollo

```bash
uv run just check    # lint + cuaderno sin salidas + pruebas (la misma comprobación que la CI)
uv run just fmt      # formato y correcciones automáticas
uv run pytest -m integration   # pruebas con Ollama y Qdrant reales (se saltan si no responden)
```

`just check` deselecciona las pruebas marcadas `integration`.

La CI (`.github/workflows/ci.yml`) necesita dos secretos de solo lectura, uno
por propietario: `HERMES_CORE_TOKEN` para los repositorios de `Jdorta62` y
`HERMES_INGESTION_TOKEN` para la organización `hermes-ingestion`.

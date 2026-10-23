# Épica 5 — Métricas RAG y notebook de experimentos

Hoy cada respuesta se califica con reglas (palabras clave y frases de
evidencia) y un juez que dice `correct`, `partial` o `incorrect`. Eso indica
si la respuesta acierta, pero no mide por separado la calidad de la
recuperación y la de la generación, ni permite barrer hiperparámetros y
acumular los resultados. Esta épica añade las métricas de la tríada RAG,
calculadas con el juez LLM propio, al registro, al informe y a `compare`, y un
notebook de Jupyter, `rag_evaluation.ipynb`, para lanzar experimentos celda a
celda contra colecciones de Qdrant y guardar un histórico en CSV. Diseño:
`.claude/architect/designs/rag-metrics/design.md` (ADR 0002 y 0003).

## Alcance

### Métricas deterministas (sin LLM)

- Exactas y baratas; se calculan a partir de lo que el registro ya guarda.
  Si a una pregunta le falta el dato que necesitan (`documents`, `evidence`
  o el campo `source` en el payload), quedan como «no aplicable», que no es
  lo mismo que un fallo.
  - **Recuperación por documento**: Hit@k, Recall@k y MRR sobre
    `documents`.
  - **Recuperación por evidencia**: Hit@k, Recall@k y MRR sobre `evidence`,
    medidos tras la recuperación y en el contexto final. La diferencia entre
    los dos es la **pérdida por presupuesto**.
  - **Rechazos**, agregados por ejecución: tasa de rechazos falsos (rechaza
    con la evidencia en el contexto) y tasa de respuestas inventadas
    (responde cuando debía rechazar).
  - **Coste**: segundos por fase, tokens del contexto y tokens de las
    llamadas al LLM (los que informe Ollama o, si no, los de
    `hermes_enrich.tokens`).

### Métricas con el juez (`hermes_report.metrics`)

- Métricas con valores en `[0, 1]`, calculadas con un `JudgeClient`
  (`providers.py`) y sin dependencias nuevas:
  - **Faithfulness**: el juez extrae las afirmaciones de la respuesta y decide
    cuáles se apoyan en el contexto; el valor es apoyadas / total.
  - **Answer relevance**: hasta qué punto la respuesta contesta la pregunta
    (el diseño elige entre la nota del juez y el método de preguntas
    generadas + similitud de embeddings).
  - **Context precision**: el juez marca cada chunk recuperado como relevante
    o no frente a la respuesta esperada; el valor es la precisión media
    ponderada por posición (como en ragas).
  - **Context recall**: las afirmaciones de la respuesta esperada que están
    cubiertas por el contexto / total.
  - **Answer correctness**: precisión, recall y F1 entre las afirmaciones de
    la respuesta y las de la respuesta esperada.
  - **Citation precision / recall**: de las citas `[n]` del
    `GenerationResult`, cuántas apuntan a un chunk que apoya la afirmación
    que acompañan, y cuántas afirmaciones que lo necesitan llevan una cita
    correcta.

### Métricas opcionales (desactivadas por defecto)

- Se activan una a una, igual que las demás. Por defecto están desactivadas
  porque aportan poco a este corpus o porque no sirven con respuestas libres.
  Se implementan sin dependencias nuevas:
  - **BLEU** y **ROUGE-L** frente a `expected`, en Python puro, con la misma
    normalización que las reglas (minúsculas y sin acentos).
  - **Noise sensitivity** (juez): afirmaciones incorrectas de la respuesta
    que se apoyan en chunks recuperados pero irrelevantes.
  - **Context entity recall** (juez): entidades de `expected` presentes en
    el contexto / total.
  - **Toxicidad** y **sesgo** (juez): nota de la respuesta en `[0, 1]`.

### Selección de métricas

- Cada métrica tiene un nombre estable (`hit_at_k_docs`, `faithfulness`,
  `bleu`…). El registro de métricas dice si está activa por defecto, si
  necesita el juez y qué datos de la pregunta requiere.
- `[metrics]` en el archivo de ejecución y en la configuración del notebook:
  `enabled` (todas sí/no), `include` (activa nombres, también los
  opcionales) y `exclude` (desactiva nombres). Opciones `--no-metrics` y
  `--metric NOMBRE` (se puede repetir) en la línea de órdenes.

### Comportamiento común

- Cada métrica devuelve su valor, el detalle (afirmaciones, veredicto por
  chunk), las llamadas al juez en crudo y su error. Un fallo (excepción,
  JSON que no se puede leer, tiempo agotado) deja el valor en `None` con el
  error; nunca aborta la ejecución ni las demás métricas.
- Casos límite definidos: respuesta vacía, contexto vacío, preguntas de
  rechazo (`refusal`) y preguntas sin `expected` útil.

### Integración en `hermes-report run`

- Sección `[metrics]` (ver «Selección de métricas»). Las métricas con juez
  usan el mismo cliente que `[judge]`; con `--no-judge` se calculan solo las
  deterministas.
- `run.json` pasa a schema 2: `QuestionRecord.metrics` y medias por métrica
  en el registro. `read_record` sigue leyendo schema 1 (sin métricas) para que
  `render` y `compare` funcionen con ejecuciones antiguas.
- Informe: fichas con la media de cada métrica en **Resumen**, columnas por
  métrica en la tabla de preguntas y una sección **Métricas** en la pestaña de
  cada pregunta (detalle y llamadas en crudo).
- `compare`: medias por métrica en las fichas de cada ejecución y en la
  matriz de preguntas.

### Notebook `notebooks/rag_evaluation.ipynb`

- Independiente del runner: hace su propio retrieval contra Qdrant y su
  propia generación, para probar colecciones ya existentes sin pasar por la
  ingesta.
- Secciones, cada celda con comentarios que explican qué hace y por qué y con
  sus resultados intermedios a la vista:
  1. **Setup**: imports, clientes de Qdrant y Ollama, comprobación de que
     responden.
  2. **Configuración centralizada** (un solo diccionario): lista de
     colecciones, modelo de embeddings (proveedor y nombre), modelo de
     generación, modelo juez, `top_k`, `score_threshold`, filtros de payload,
     métricas activas, ruta del golden dataset, carpeta de salida y modo sin
     conexión.
  3. **Carga de datos**: el golden dataset en JSON, como array de objetos o
     con el formato de `preguntas.json` (`{"title", "questions": [...]}`),
     con `id`, `question`, `expected` (o `ground_truth` como alias) y
     `documents` opcional. Muestra la tabla de preguntas.
  4. **Retrieval y generación**: por cada colección y pregunta, embebe la
     pregunta, busca con `top_k`, umbral y filtros, arma el prompt y genera.
     Guarda contextos, respuesta y tiempos por pregunta.
  5. **Evaluación**: llama a `hermes_report.metrics` sobre cada respuesta.
     Los fallos quedan registrados por pregunta y métrica.
  6. **Registro de experimentos**: una fila por (ejecución × colección) en un
     `pandas.DataFrame` con timestamp, colección, modelo de embeddings, LLM,
     juez, `top_k`, umbral, filtros, número de preguntas, media de cada
     métrica, número de fallos por métrica, segundos medios, huella de la
     configuración y nota libre. Se añade a `history.csv` y `history.jsonl`
     en la carpeta de salida; el detalle por pregunta va a
     `<timestamp>-detail.csv`. Una última celda carga el histórico y ordena
     las combinaciones por métrica.
- Modo sin conexión: `HashEmbedder`, `EchoLLM`, `EchoJudgeClient` y Qdrant
  `":memory:"`, con una colección `hermes_report_notebook_demo` creada a
  partir de `examples/demo/`.

### Documentación

- Sección del README sobre las métricas y sobre el notebook (cómo abrirlo,
  cómo cambiar la configuración, dónde queda el histórico).

## Fuera de alcance

- ragas, deepeval, langchain, sacrebleu, rouge-score o cualquier otra
  librería de evaluación o de métricas.
- Gráficas en el notebook más allá de lo que da `DataFrame.plot`.
- Usar las métricas para cambiar el veredicto de las reglas o del juez: son
  señales aparte.

## Decisiones ya tomadas

All are decisions by the author; they are not reopened.

- Las métricas son propias y se calculan con el juez LLM de `providers.py`;
  no se usan ragas ni deepeval.
- El notebook es independiente del runner y hace su propio retrieval y
  generación. Es una excepción, limitada a `notebooks/`, a la regla de que la
  herramienta no tiene lógica de pipeline propia: el paquete `hermes_report`
  no gana retrieval ni generación.
- El notebook importa las métricas de `hermes_report.metrics`; no las
  duplica.
- Dependencias aprobadas por José (2026-10-22): pandas, jupyterlab,
  ipykernel y nbclient, en un grupo de dependencias aparte (`notebook`), no
  en las dependencias del paquete. Ninguna otra.
- Las métricas recomendadas están activas por defecto; BLEU, ROUGE-L, noise
  sensitivity, context entity recall, toxicidad y sesgo, desactivadas.
- Las métricas son señales aparte: no cambian el veredicto de las reglas ni
  el del juez.

## Restricciones

- Las reglas de `.claude/CLAUDE.project.md` siguen vigentes, también para el
  notebook: solo se crean, borran o escriben colecciones `hermes_report_*`
  (el notebook solo lee las demás), y las pruebas usan Qdrant `":memory:"`,
  `HashEmbedder` y `EchoLLM`.
- El informe se sigue generando solo a partir de `run.json`.
- Los prompts del juez y los textos del informe, en español; identificadores,
  comentarios y docstrings, en inglés (las celdas markdown del notebook, en
  español).
- Las salidas del notebook (histórico y detalle) no se guardan en git.

## Criterios de aceptación transversales

- `uv run just check` en verde.
- `examples/demo/run-offline.toml` genera un informe con métricas, y
  `compare` entre una ejecución de schema 1 y otra de schema 2 funciona.
- El notebook se ejecuta de punta a punta en modo sin conexión dentro de la
  suite de pruebas, y con Ollama y Qdrant reales sobre `examples/quijote`.
- Dos ejecuciones del notebook con configuraciones distintas dejan dos filas
  comparables en `history.csv`.

## Depende de

Épica 4.

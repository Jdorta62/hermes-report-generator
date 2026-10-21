"""Run file: the TOML description of a run, validated and with defaults applied."""

from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from typing import Any

__all__ = [
  "ConfigError",
  "CorpusConfig",
  "EmbeddingConfig",
  "EnrichConfig",
  "GenerationConfig",
  "IngestConfig",
  "JudgeConfig",
  "OutputConfig",
  "Overrides",
  "QdrantConfig",
  "QuestionsConfig",
  "RetrievalConfig",
  "RunConfig",
  "load_config",
]

_DEFAULT_QDRANT_URL = "http://localhost:6333"
_DEFAULT_OLLAMA_MODEL = "qwen3:4b"
_COLLECTION_PREFIX = "hermes_report_"
_EMBEDDING_PROVIDERS = ("ollama", "hash")
_LLM_PROVIDERS = ("ollama", "echo")
_ENRICHERS = ("neighbors", "merge", "doc_order")
_RESPONSE_FORMATS = ("plain", "markdown", "bullets")
# Everything reading, parsing or resolving user-given input can raise.
_INPUT_ERRORS = (
  OSError,
  ValueError,
  RecursionError,
  RuntimeError,
  OverflowError,
  UnicodeError,
)


class ConfigError(Exception):
  """The run file or an override is invalid (message in Spanish).

  Attributes:
    path: The run file.
    message: What is wrong.
    key: Where, dotted like ``retrieval.top_k``; ``None`` for the whole file.
  """

  def __init__(self, path: Path, message: str, key: str | None = None) -> None:
    super().__init__(path, message, key)
    self.path = path
    self.message = message
    self.key = key

  def __str__(self) -> str:
    if self.key is None:
      return f"{self.path}: {self.message}"
    return f"{self.path}: {self.key}: {self.message}"


@dataclass(frozen=True)
class CorpusConfig:
  """The ``[corpus]`` section."""

  dir: Path
  glob: str = "**/*.md"


@dataclass(frozen=True)
class QuestionsConfig:
  """The ``[questions]`` section."""

  file: Path


@dataclass(frozen=True)
class IngestConfig:
  """The ``[ingest]`` section: chunker arguments."""

  max_chunk_size: int = 1500
  hard_boundary_levels: tuple[int, ...] = (1, 2)
  strip_frontmatter: bool = True


@dataclass(frozen=True)
class EmbeddingConfig:
  """The ``[embedding]`` section."""

  provider: str = "ollama"
  model: str = "nomic-embed-text"


@dataclass(frozen=True)
class QdrantConfig:
  """The ``[qdrant]`` section; an empty collection means derived."""

  url: str
  collection: str = ""


@dataclass(frozen=True)
class RetrievalConfig:
  """The ``[retrieval]`` section."""

  top_k: int = 10
  candidate_factor: int = 1
  score_threshold: float | None = None


@dataclass(frozen=True)
class EnrichConfig:
  """The ``[enrich]`` section; ``max_context_tokens`` 0 means no limit."""

  chain: tuple[str, ...] = ("neighbors", "merge", "doc_order")
  max_context_tokens: int = 3000
  max_chunks: int | None = None


@dataclass(frozen=True)
class GenerationConfig:
  """The ``[generation]`` section."""

  provider: str
  model: str
  temperature: float = 0.0
  seed: int | None = 42
  response_format: str = "plain"
  insufficient_message: str | None = None


@dataclass(frozen=True)
class JudgeConfig:
  """The ``[judge]`` section, with provider and model already resolved."""

  enabled: bool
  provider: str
  model: str


@dataclass(frozen=True)
class OutputConfig:
  """The ``[output]`` section, with absolute paths."""

  dir: Path
  cache_dir: Path


@dataclass(frozen=True)
class Overrides:
  """Command-line values that replace run-file keys.

  ``out`` is resolved against the current directory (``Path(".")``, which is
  also ``Path("")``, means the current directory; the command line rejects an
  empty ``--out`` before building the path). ``only`` must be a tuple or list
  of non-empty strings, not a bare string; ``load_config`` raises
  ``ConfigError`` otherwise.
  """

  model: str | None = None
  judge_model: str | None = None
  no_judge: bool = False
  top_k: int | None = None
  budget: int | None = None
  out: Path | None = None
  only: tuple[str, ...] = ()
  reingest: bool = False


def _plain(value: object) -> object:
  """Plain-data form of a config value: absolute path strings, lists."""
  if isinstance(value, Path):
    return str(value.absolute())
  if isinstance(value, tuple):
    return list(value)
  return value


def _section_dict(section: Any) -> dict[str, Any]:
  """The fields of a section dataclass as plain data, ``None`` left out."""
  result: dict[str, Any] = {}
  for item in fields(section):
    value = _plain(getattr(section, item.name))
    if value is not None:
      result[item.name] = value
  return result


@dataclass(frozen=True)
class RunConfig:
  """A validated run: the run file with defaults and overrides applied.

  Attributes:
    path: Absolute path of the run file. It is the location of the run, not
      part of its content, so it does not take part in equality.
    title: Title of the run.
    description: Free text, may be empty.
    only: Question ids to run; empty means all.
    reingest: Whether to ignore the ingestion cache.
  """

  path: Path = field(compare=False)
  title: str
  corpus: CorpusConfig
  questions: QuestionsConfig
  ingest: IngestConfig
  embedding: EmbeddingConfig
  qdrant: QdrantConfig
  retrieval: RetrievalConfig
  enrich: EnrichConfig
  generation: GenerationConfig
  judge: JudgeConfig
  output: OutputConfig
  description: str = ""
  only: tuple[str, ...] = ()
  reingest: bool = False

  def to_dict(self) -> dict[str, Any]:
    """The resolved run file as plain data, in the TOML layout.

    Paths are absolute strings, tuples are lists and ``None`` values are left
    out. ``only`` and ``reingest`` are not included.
    """
    return {
      "title": self.title,
      "description": self.description,
      "corpus": _section_dict(self.corpus),
      "questions": _section_dict(self.questions),
      "ingest": _section_dict(self.ingest),
      "embedding": _section_dict(self.embedding),
      "qdrant": _section_dict(self.qdrant),
      "retrieval": _section_dict(self.retrieval),
      "enrich": _section_dict(self.enrich),
      "generation": _section_dict(self.generation),
      "judge": _section_dict(self.judge),
      "output": _section_dict(self.output),
    }

  def to_toml(self) -> str:
    """The resolved run file as TOML text that ``load_config`` reads back."""
    data = self.to_dict()
    lines: list[str] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    for key, value in data.items():
      if isinstance(value, dict):
        tables.append((key, value))
      else:
        lines.append(f"{key} = {_toml_value(value)}")
    for name, table in tables:
      lines.append("")
      lines.append(f"[{name}]")
      for key, value in table.items():
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _toml_string(value: str) -> str:
  escapes = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
  }
  out: list[str] = []
  for char in value:
    if char in escapes:
      out.append(escapes[char])
    elif ord(char) < 0x20 or ord(char) == 0x7F:
      out.append(f"\\u{ord(char):04x}")
    else:
      out.append(char)
  return '"' + "".join(out) + '"'


def _toml_value(value: object) -> str:
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, int):
    return str(value)
  if isinstance(value, float):
    if math.isnan(value):
      return "nan"
    if math.isinf(value):
      return "inf" if value > 0 else "-inf"
    return repr(value)
  if isinstance(value, str):
    return _toml_string(value)
  if isinstance(value, list):
    return "[" + ", ".join(_toml_value(item) for item in value) + "]"
  raise TypeError(f"cannot write {type(value).__name__} as TOML")


# --- validation helpers -------------------------------------------------------


def _as_str(path: Path, key: str, value: object, *, non_empty: bool = False) -> str:
  if not isinstance(value, str):
    raise ConfigError(path, "debe ser un texto", key)
  if non_empty and not value.strip():
    raise ConfigError(path, "no puede estar vacío", key)
  return value


def _as_int(
  path: Path,
  key: str,
  value: object,
  *,
  minimum: int | None = None,
  maximum: int | None = None,
) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ConfigError(path, "debe ser un número entero", key)
  if minimum is not None and value < minimum:
    raise ConfigError(path, f"debe ser mayor o igual que {minimum}", key)
  if maximum is not None and value > maximum:
    raise ConfigError(path, f"debe ser menor o igual que {maximum}", key)
  return value


def _as_float(
  path: Path, key: str, value: object, *, minimum: float | None = None
) -> float:
  if isinstance(value, bool) or not isinstance(value, int | float):
    raise ConfigError(path, "debe ser un número", key)
  try:
    number = float(value)
  except OverflowError:
    raise ConfigError(path, "es demasiado grande", key) from None
  if math.isnan(number):
    raise ConfigError(path, "debe ser un número", key)
  if minimum is not None and number < minimum:
    raise ConfigError(path, f"debe ser mayor o igual que {minimum:g}", key)
  return number


def _as_bool(path: Path, key: str, value: object) -> bool:
  if not isinstance(value, bool):
    raise ConfigError(path, "debe ser verdadero o falso", key)
  return value


def _as_choice(path: Path, key: str, value: object, *, choices: tuple[str, ...]) -> str:
  if not isinstance(value, str):
    raise ConfigError(path, "debe ser un texto", key)
  if value not in choices:
    raise ConfigError(
      path, f"valor {value!r} no válido; opciones: {', '.join(choices)}", key
    )
  return value


def _as_levels(path: Path, key: str, value: object) -> tuple[int, ...]:
  if not isinstance(value, list):
    raise ConfigError(path, "debe ser una lista de enteros", key)
  if not value:
    raise ConfigError(path, "no puede estar vacía", key)
  return tuple(
    _as_int(path, f"{key}[{i}]", item, minimum=1, maximum=6)
    for i, item in enumerate(value)
  )


def _as_chain(path: Path, key: str, value: object) -> tuple[str, ...]:
  if not isinstance(value, list):
    raise ConfigError(path, "debe ser una lista de textos", key)
  return tuple(
    _as_choice(path, f"{key}[{i}]", item, choices=_ENRICHERS)
    for i, item in enumerate(value)
  )


def _as_collection(path: Path, key: str, value: object) -> str:
  text = _as_str(path, key, value)
  if text and not text.startswith(_COLLECTION_PREFIX):
    raise ConfigError(
      path, f"debe estar vacío o empezar por {_COLLECTION_PREFIX!r}", key
    )
  return text


def _as_default_str(path: Path, key: str, value: object) -> str:
  """A text that is either empty (meaning the default) or non-blank."""
  if value == "":
    return ""
  return _as_str(path, key, value, non_empty=True)


def _env(name: str, default: str) -> str:
  """The environment variable ``name``, or ``default`` when unset or blank."""
  value = os.environ.get(name)
  if value is None or not value.strip():
    return default
  return value.strip()


def _resolve(path: Path, key: str, base: Path, value: str) -> Path:
  """``value`` as an absolute path against ``base``; ``ConfigError`` if unusable."""
  try:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()
  except _INPUT_ERRORS as exc:
    raise ConfigError(path, f"ruta no válida: {exc}", key) from None


def _check_kind(path: Path, key: str, target: Path, *, folder: bool) -> None:
  """Require ``target`` to be an existing folder (or file, if not ``folder``)."""
  try:
    exists = target.is_dir() if folder else target.is_file()
  except _INPUT_ERRORS as exc:
    raise ConfigError(path, f"ruta no válida: {exc}", key) from None
  if not exists:
    kind = "una carpeta" if folder else "un archivo"
    raise ConfigError(path, f"no es {kind} existente: {target}", key)


def _table(
  path: Path, data: dict[str, Any], name: str, allowed: tuple[str, ...]
) -> dict[str, Any]:
  """The section ``name`` of the run file ({} when absent), keys checked."""
  table = data.get(name, {})
  if not isinstance(table, dict):
    raise ConfigError(path, "debe ser una tabla", name)
  for key in table:
    if key not in allowed:
      raise ConfigError(path, "clave desconocida", f"{name}.{key}")
  return table


def _get(
  path: Path,
  table: dict[str, Any],
  section: str,
  key: str,
  convert: Callable[[Path, str, object], Any],
  default: Any,
) -> Any:
  if key not in table:
    return default
  return convert(path, f"{section}.{key}", table[key])


def _required(
  path: Path,
  table: dict[str, Any],
  section: str,
  key: str,
  convert: Callable[[Path, str, object], Any],
) -> Any:
  if key not in table:
    raise ConfigError(path, "falta la clave obligatoria", f"{section}.{key}")
  return convert(path, f"{section}.{key}", table[key])


def _override(
  path: Path,
  key: str,
  convert: Callable[[Path, str, object], Any],
  value: object,
) -> Any:
  """Validate a command-line value like a run-file one, marking its origin."""
  try:
    return convert(path, key, value)
  except ConfigError as exc:
    raise ConfigError(
      path, f"{exc.message} (opción de la línea de comandos)", key
    ) from None


def _read_toml(path: Path) -> dict[str, Any]:
  try:
    is_file = path.is_file()
  except _INPUT_ERRORS as exc:
    raise ConfigError(path, f"no se puede leer: {exc}") from None
  if not is_file:
    raise ConfigError(path, "el archivo no existe")
  try:
    text = path.read_bytes().decode("utf-8")
  except _INPUT_ERRORS as exc:
    raise ConfigError(path, f"no se puede leer: {exc}") from None
  try:
    return tomllib.loads(text)
  except tomllib.TOMLDecodeError as exc:
    raise ConfigError(path, f"TOML inválido: {exc}") from exc
  except RecursionError:
    raise ConfigError(path, "TOML inválido: anidamiento excesivo") from None
  except _INPUT_ERRORS as exc:
    raise ConfigError(path, f"TOML inválido: {exc}") from None


def _as_only(path: Path, key: str, value: object) -> tuple[str, ...]:
  """``Overrides.only``: a tuple or list of non-empty strings, never one string."""
  if not isinstance(value, tuple | list):
    raise ConfigError(path, "debe ser una lista de textos", key)
  return tuple(
    _as_str(path, f"{key}[{i}]", item, non_empty=True) for i, item in enumerate(value)
  )


_TOP_LEVEL = (
  "title",
  "description",
  "corpus",
  "questions",
  "ingest",
  "embedding",
  "qdrant",
  "retrieval",
  "enrich",
  "generation",
  "judge",
  "output",
)


def load_config(path: Path, overrides: Overrides | None = None) -> RunConfig:
  """Read, validate and resolve a run file, then apply ``overrides``.

  Relative paths are resolved against the folder of the run file. Defaults:
  ``$QDRANT_URL`` and ``$OLLAMA_MODEL`` for the Qdrant URL and the generation
  model; the judge takes the generation provider and model unless set.

  Raises:
    ConfigError: if the file is missing, is not valid TOML, breaks a rule or
      an override is invalid; the error names the file and the key.
  """
  try:
    path = path.resolve()
  except _INPUT_ERRORS as exc:
    raise ConfigError(path, f"ruta no válida: {exc}") from None
  base = path.parent
  data = _read_toml(path)
  for key, value in data.items():
    if key not in _TOP_LEVEL:
      kind = "sección desconocida" if isinstance(value, dict) else "clave desconocida"
      raise ConfigError(path, kind, key)

  if "title" not in data:
    raise ConfigError(path, "falta la clave obligatoria", "title")
  title = _as_str(path, "title", data["title"], non_empty=True)
  description = ""
  if "description" in data:
    description = _as_str(path, "description", data["description"])

  non_empty = partial(_as_str, non_empty=True)

  t = _table(path, data, "corpus", ("dir", "glob"))
  corpus_dir = _resolve(
    path, "corpus.dir", base, _required(path, t, "corpus", "dir", non_empty)
  )
  _check_kind(path, "corpus.dir", corpus_dir, folder=True)
  corpus = CorpusConfig(
    dir=corpus_dir, glob=_get(path, t, "corpus", "glob", non_empty, "**/*.md")
  )

  t = _table(path, data, "questions", ("file",))
  questions_file = _resolve(
    path, "questions.file", base, _required(path, t, "questions", "file", non_empty)
  )
  _check_kind(path, "questions.file", questions_file, folder=False)
  questions = QuestionsConfig(file=questions_file)

  t = _table(
    path,
    data,
    "ingest",
    ("max_chunk_size", "hard_boundary_levels", "strip_frontmatter"),
  )
  ingest = IngestConfig(
    max_chunk_size=_get(
      path, t, "ingest", "max_chunk_size", partial(_as_int, minimum=1), 1500
    ),
    hard_boundary_levels=_get(
      path, t, "ingest", "hard_boundary_levels", _as_levels, (1, 2)
    ),
    strip_frontmatter=_get(path, t, "ingest", "strip_frontmatter", _as_bool, True),
  )

  t = _table(path, data, "embedding", ("provider", "model"))
  embedding = EmbeddingConfig(
    provider=_get(
      path,
      t,
      "embedding",
      "provider",
      partial(_as_choice, choices=_EMBEDDING_PROVIDERS),
      "ollama",
    ),
    model=_get(path, t, "embedding", "model", non_empty, "nomic-embed-text"),
  )

  t = _table(path, data, "qdrant", ("url", "collection"))
  qdrant = QdrantConfig(
    url=_get(
      path,
      t,
      "qdrant",
      "url",
      non_empty,
      _env("QDRANT_URL", _DEFAULT_QDRANT_URL),
    ),
    collection=_get(path, t, "qdrant", "collection", _as_collection, ""),
  )

  t = _table(path, data, "retrieval", ("top_k", "candidate_factor", "score_threshold"))
  at_least_one = partial(_as_int, minimum=1)
  retrieval = RetrievalConfig(
    top_k=_get(path, t, "retrieval", "top_k", at_least_one, 10),
    candidate_factor=_get(path, t, "retrieval", "candidate_factor", at_least_one, 1),
    score_threshold=_get(path, t, "retrieval", "score_threshold", _as_float, None),
  )

  t = _table(path, data, "enrich", ("chain", "max_context_tokens", "max_chunks"))
  enrich = EnrichConfig(
    chain=_get(
      path, t, "enrich", "chain", _as_chain, ("neighbors", "merge", "doc_order")
    ),
    max_context_tokens=_get(
      path, t, "enrich", "max_context_tokens", partial(_as_int, minimum=0), 3000
    ),
    max_chunks=_get(path, t, "enrich", "max_chunks", at_least_one, None),
  )

  t = _table(
    path,
    data,
    "generation",
    (
      "provider",
      "model",
      "temperature",
      "seed",
      "response_format",
      "insufficient_message",
    ),
  )
  generation = GenerationConfig(
    provider=_get(
      path,
      t,
      "generation",
      "provider",
      partial(_as_choice, choices=_LLM_PROVIDERS),
      "ollama",
    ),
    model=_get(
      path,
      t,
      "generation",
      "model",
      non_empty,
      _env("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL),
    ),
    temperature=_get(
      path, t, "generation", "temperature", partial(_as_float, minimum=0.0), 0.0
    ),
    seed=_get(path, t, "generation", "seed", _as_int, 42),
    response_format=_get(
      path,
      t,
      "generation",
      "response_format",
      partial(_as_choice, choices=_RESPONSE_FORMATS),
      "plain",
    ),
    insufficient_message=_get(
      path, t, "generation", "insufficient_message", non_empty, None
    ),
  )

  t = _table(path, data, "judge", ("enabled", "provider", "model"))
  judge_provider = _get(path, t, "judge", "provider", _as_str, "")
  judge_model = _get(path, t, "judge", "model", _as_default_str, "")
  judge_model_is_default = not judge_model
  if not judge_provider:
    judge_provider = generation.provider
  _as_choice(path, "judge.provider", judge_provider, choices=_LLM_PROVIDERS)
  judge = JudgeConfig(
    enabled=_get(path, t, "judge", "enabled", _as_bool, True),
    provider=judge_provider,
    model=judge_model or generation.model,
  )

  t = _table(path, data, "output", ("dir", "cache_dir"))
  output = OutputConfig(
    dir=_resolve(
      path, "output.dir", base, _get(path, t, "output", "dir", non_empty, "out")
    ),
    cache_dir=_resolve(
      path,
      "output.cache_dir",
      base,
      _get(path, t, "output", "cache_dir", non_empty, ".hermes-report-cache"),
    ),
  )

  only: tuple[str, ...] = ()
  reingest = False
  if overrides is not None:
    non_empty_str = partial(_as_str, non_empty=True)
    if overrides.model is not None:
      model = _override(path, "generation.model", non_empty_str, overrides.model)
      generation = replace(generation, model=model)
      if judge_model_is_default:
        judge = replace(judge, model=model)
    if overrides.judge_model is not None:
      value = _override(path, "judge.model", non_empty_str, overrides.judge_model)
      judge = replace(judge, model=value)
    if overrides.no_judge:
      judge = replace(judge, enabled=False)
    if overrides.top_k is not None:
      top_k = _override(path, "retrieval.top_k", at_least_one, overrides.top_k)
      retrieval = replace(retrieval, top_k=top_k)
    if overrides.budget is not None:
      budget = _override(
        path,
        "enrich.max_context_tokens",
        partial(_as_int, minimum=0),
        overrides.budget,
      )
      enrich = replace(enrich, max_context_tokens=budget)
    if overrides.out is not None:
      text = _override(path, "output.dir", non_empty_str, str(overrides.out))
      output = replace(output, dir=_resolve(path, "output.dir", Path(), text))
    only = _override(path, "only", _as_only, overrides.only)
    reingest = overrides.reingest

  return RunConfig(
    path=path,
    title=title,
    description=description,
    corpus=corpus,
    questions=questions,
    ingest=ingest,
    embedding=embedding,
    qdrant=qdrant,
    retrieval=retrieval,
    enrich=enrich,
    generation=generation,
    judge=judge,
    output=output,
    only=only,
    reingest=reingest,
  )

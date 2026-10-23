"""Providers of a run: embedders, LLM clients and judge clients, real and offline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from typing import Any, Final, Protocol, runtime_checkable

import ollama
from hermes_core.generation import PromptMessage
from hermes_core.protocols import Embedder
from hermes_generator import GenerationParams, LLMClient, LLMReply
from hermes_generator.answer.codec import SENTINEL
from hermes_generator.answer.llm_ollama import OllamaChatClient

from hermes_report.config import EmbeddingConfig, GenerationConfig, JudgeConfig
from hermes_report.questions import normalise

__all__ = [
  "EchoJudgeClient",
  "EchoLLM",
  "HashEmbedder",
  "JudgeClient",
  "OllamaEmbedder",
  "OllamaJudgeClient",
  "ProviderError",
  "check_ollama",
  "make_embedder",
  "make_judge",
  "make_llm",
  "ollama_host",
]

_DEFAULT_HOST = "http://localhost:11434"
_TIMEOUT = 600
_CHECK_TIMEOUT = 10
_BATCH = 64
_PREFIXES: dict[str, tuple[str, str]] = {
  "nomic-embed-text": ("search_query: ", "search_document: "),
}
_TOKEN = re.compile(r"\w+")
_SOURCE_BLOCK = re.compile(
  r'<source n="(\d+)" doc="[^"]*">\n(.*?)\n</source>', re.DOTALL
)
_SENTENCE = re.compile(r"[^.!?]*[.!?]")
_ECHO_MAX_CHARS = 200
_ECHO_VERDICT = {
  "verdict": "partial",
  "reason": "juez simulado: no evalúa la respuesta",
}
_ECHO_NO_FLAG = "juez simulado: sin marcas"
_ECHO_NO_MARKERS = "El texto no lleva marcadores"
_ECHO_TEXT = re.compile(r"\nRespuesta( esperada)?:\n")
_ECHO_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_ECHO_MARKER = re.compile(r"\[(\d+)\]")
_ECHO_LEADING = re.compile(r"^(?:\[\d+\]\s*)+")
_ECHO_COUNT = re.compile(r"Número de preguntas: (\d+)")
_ECHO_ENTITY = re.compile(r"[A-ZÁÉÍÓÚÑ][\wáéíóúñ]{2,}|\d[\w.,:]*\d|\d")


class ProviderError(Exception):
  """A provider (Ollama or another client) failed or misbehaved (message in Spanish)."""


def ollama_host() -> str:
  """Return the Ollama URL: ``$OLLAMA_HOST`` or ``http://localhost:11434``.

  The value is stripped; unset or blank gives the default, and ``http://`` is
  prepended when there is no ``://``.
  """
  value = os.environ.get("OLLAMA_HOST", "").strip()
  if not value:
    return _DEFAULT_HOST
  return value if "://" in value else f"http://{value}"


# --- helpers ------------------------------------------------------------------


def _describe(exc: BaseException) -> str:
  """The exception as text, never empty."""
  text = str(exc)
  return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _check_model(model: object) -> str:
  if not isinstance(model, str):
    raise TypeError(f"model debe ser un str, no {type(model).__name__}")
  if not model.strip():
    raise ValueError("model no puede estar vacío")
  return model


def _resolve_host(host: object) -> str:
  """The host given, stripped, or ``ollama_host()`` when ``None`` or blank."""
  if host is None:
    return ollama_host()
  if not isinstance(host, str):
    raise TypeError(f"host debe ser un str o None, no {type(host).__name__}")
  return host.strip() or ollama_host()


def _build_client(host: str) -> Any:
  """An ``ollama.Client`` for ``host``; failures become ``ProviderError``."""
  try:
    return ollama.Client(host=host, timeout=_TIMEOUT)
  except Exception as exc:
    raise ProviderError(
      f"No se pudo crear el cliente de Ollama para {host}: {_describe(exc)}"
    ) from exc


def _read(obj: Any, name: str) -> Any:
  """``obj[name]``, else ``obj.name``; ``KeyError`` when neither exists."""
  try:
    return obj[name]
  except (KeyError, TypeError, IndexError, AttributeError):
    pass
  try:
    return getattr(obj, name)
  except AttributeError:
    raise KeyError(name) from None


def _check_text(value: object, name: str) -> str:
  if not isinstance(value, str):
    raise TypeError(f"{name} debe ser un str, no {type(value).__name__}")
  return value


def _check_texts(texts: object) -> list[str]:
  if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
    raise TypeError(f"texts debe ser una lista de str, no {type(texts).__name__}")
  for index, text in enumerate(texts):
    if not isinstance(text, str):
      raise TypeError(f"texts[{index}] debe ser un str, no {type(text).__name__}")
  return list(texts)


def _check_messages(messages: object) -> Sequence[PromptMessage]:
  if isinstance(messages, str | bytes) or not isinstance(messages, Sequence):
    raise TypeError(
      f"messages debe ser una secuencia de PromptMessage, no {type(messages).__name__}"
    )
  if not messages:
    raise ValueError("messages no puede estar vacío")
  for index, message in enumerate(messages):
    if not isinstance(getattr(message, "role", None), str) or not isinstance(
      getattr(message, "content", None), str
    ):
      raise TypeError(
        f"messages[{index}] debe tener role y content de tipo str, "
        f"no {type(message).__name__}"
      )
  return messages


def _as_vector(raw: object) -> list[float]:
  """One embedding as a non-empty list of finite floats."""
  if isinstance(raw, str | bytes) or not isinstance(raw, Iterable):
    raise TypeError(
      f"un embedding debe ser una lista de números, no {type(raw).__name__}"
    )
  vector: list[float] = []
  for item in raw:
    if isinstance(item, bool) or not isinstance(item, Real):
      raise TypeError(f"un embedding contiene {type(item).__name__}, no un número")
    number = float(item)
    if not math.isfinite(number):
      raise ValueError("un embedding contiene valores no finitos")
    vector.append(number)
  if not vector:
    raise ValueError("un embedding está vacío")
  return vector


# --- embedders ----------------------------------------------------------------


class OllamaEmbedder:
  """Embedder over an Ollama server; any client failure becomes ``ProviderError``.

  Satisfies ``hermes_core.protocols.Embedder`` structurally. Models named
  ``nomic-embed-text`` get the task prefixes ``search_query: `` for queries and
  ``search_document: `` for documents; other models get none.

  Registry: embedder[ollama]

  Attributes:
    model: The embedding model.
    host: The Ollama URL, used in error messages.
  """

  def __init__(
    self, model: str, host: str | None = None, client: Any | None = None
  ) -> None:
    """Create an embedder for ``model``.

    Args:
      model: Embedding model name, with or without ``:tag``.
      host: Ollama URL; ``None`` or blank means ``ollama_host()``.
      client: An ``ollama.Client`` or any object with the same ``embed``
        method; ``None`` builds one with a 600 s timeout.

    Raises:
      TypeError: ``model`` or ``host`` is not a ``str`` (``host`` may be ``None``).
      ValueError: ``model`` is blank.
      ProviderError: the Ollama client cannot be built.
    """
    self.model: str = _check_model(model)
    self.host: str = _resolve_host(host)
    self._client: Any = client if client is not None else _build_client(self.host)
    self._query_prefix, self._doc_prefix = _PREFIXES.get(
      self.model.split(":")[0], ("", "")
    )

  def get_query_embedding(self, query: str) -> list[float]:
    """Return the embedding of a retrieval query.

    Raises:
      TypeError: ``query`` is not a ``str``.
      ProviderError: the client fails or answers a wrong number of embeddings.

    Effects: net
    """
    _check_text(query, "query")
    return self._embed([self._query_prefix + query])[0]

  def get_text_embedding(self, text: str) -> list[float]:
    """Return the embedding of one chunk of text.

    Raises:
      TypeError: ``text`` is not a ``str``.
      ProviderError: the client fails or answers a wrong number of embeddings.

    Effects: net
    """
    _check_text(text, "text")
    return self._embed([self._doc_prefix + text])[0]

  def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
    """Return the embeddings of ``texts``, in requests of 64 at most.

    Raises:
      TypeError: ``texts`` is not a list of ``str``.
      ProviderError: the client fails or answers a wrong number of embeddings.

    Effects: net
    """
    items = _check_texts(texts)
    vectors: list[list[float]] = []
    for start in range(0, len(items), _BATCH):
      batch = [self._doc_prefix + text for text in items[start : start + _BATCH]]
      vectors.extend(self._embed(batch))
    return vectors

  def _embed(self, inputs: list[str]) -> list[list[float]]:
    try:
      response = self._client.embed(model=self.model, input=inputs)
      raw = _read(response, "embeddings")
      if isinstance(raw, str | bytes) or not isinstance(raw, Iterable):
        raise TypeError("la respuesta no trae una lista de embeddings")
      vectors = [_as_vector(vector) for vector in raw]
    except Exception as exc:
      raise ProviderError(
        f"Ollama falló al calcular embeddings con el modelo {self.model!r} "
        f"en {self.host}: {_describe(exc)}"
      ) from exc
    if len(vectors) != len(inputs):
      raise ProviderError(
        f"Ollama devolvió {len(vectors)} embeddings para {len(inputs)} textos "
        f"con el modelo {self.model!r} en {self.host}"
      )
    if len({len(vector) for vector in vectors}) > 1:
      raise ProviderError(
        f"Ollama devolvió embeddings de distinta dimensión con el modelo "
        f"{self.model!r} en {self.host}"
      )
    return vectors


class HashEmbedder:
  """Deterministic offline embedder: a bag of hashed words, L2-normalised.

  Satisfies ``hermes_core.protocols.Embedder`` structurally. Each token of the
  normalised text adds ``+1.0`` (even parity) or ``-1.0`` (odd parity) to one
  coordinate chosen by its SHA-256 digest. Queries and documents are embedded
  the same way. A text without tokens, or whose sum is all zeros, gives the unit
  vector ``[1.0, 0.0, ...]``.

  Registry: embedder[hash]

  Attributes:
    dimension: Length of every vector.
  """

  def __init__(self, dimension: int = 256) -> None:
    """Create an embedder of ``dimension`` coordinates.

    Raises:
      TypeError: ``dimension`` is not an ``int``.
      ValueError: ``dimension`` is less than 1.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, int):
      raise TypeError(f"dimension debe ser un int, no {type(dimension).__name__}")
    if dimension < 1:
      raise ValueError(f"dimension debe ser mayor o igual que 1, no {dimension}")
    self.dimension: int = dimension

  def get_query_embedding(self, query: str) -> list[float]:
    """Return the embedding of a retrieval query.

    Raises:
      TypeError: ``query`` is not a ``str``.
    """
    return self._embed(_check_text(query, "query"))

  def get_text_embedding(self, text: str) -> list[float]:
    """Return the embedding of one chunk of text.

    Raises:
      TypeError: ``text`` is not a ``str``.
    """
    return self._embed(_check_text(text, "text"))

  def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
    """Return the embeddings of ``texts``, in order.

    Raises:
      TypeError: ``texts`` is not a list of ``str``.
    """
    return [self._embed(text) for text in _check_texts(texts)]

  def _embed(self, text: str) -> list[float]:
    vector = [0.0] * self.dimension
    for token in _TOKEN.findall(normalise(text)):
      digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
      sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
      vector[int(digest[:8], 16) % self.dimension] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
      vector[0] = 1.0
      return vector
    return [value / norm for value in vector]


# --- LLM ----------------------------------------------------------------------


class EchoLLM:
  """Offline LLM: replies with the first sentence of source 1, cited as ``[1]``.

  Satisfies ``hermes_generator.LLMClient``. It reads the
  ``<source n="N" doc="...">`` blocks of the last message; without sources it
  replies with the ``INSUFFICIENT_CONTEXT`` sentinel. The sentence runs up to
  and including the first ``.``, ``!`` or ``?``, or is the first 200
  characters, with whitespace collapsed.

  Registry: llm_client[echo]

  Attributes:
    model: Always ``echo``.
  """

  model: str = "echo"

  def complete(
    self, messages: Sequence[PromptMessage], params: GenerationParams
  ) -> LLMReply:
    """Return the echo reply for the sources of the last message.

    Raises:
      TypeError: ``messages`` is not a sequence of messages, or ``params`` is not
        a ``GenerationParams``.
      ValueError: ``messages`` is empty.
    """
    items = _check_messages(messages)
    if not isinstance(params, GenerationParams):
      raise TypeError(f"params debe ser GenerationParams, no {type(params).__name__}")
    blocks = _SOURCE_BLOCK.findall(items[-1].content)
    if not blocks:
      return LLMReply(text=SENTINEL, model=self.model)
    first = next((text for number, text in blocks if number == "1"), blocks[0][1])
    sentence = _first_sentence(first)
    return LLMReply(text=f"{sentence} [1]".strip(), model=self.model)


def _first_sentence(text: str) -> str:
  collapsed = " ".join(text.split())
  match = _SENTENCE.match(collapsed)
  return match.group(0) if match else collapsed[:_ECHO_MAX_CHARS]


# --- judge --------------------------------------------------------------------


@runtime_checkable
class JudgeClient(Protocol):
  """Client of the LLM judge: chat messages in, the raw JSON reply text out.

  Registry: judge_client

  Attributes:
    model: The judge model.
  """

  model: str

  def complete_json(self, messages: Sequence[PromptMessage]) -> str:
    """Send ``messages`` and return the raw reply text, which should be JSON.

    Raises:
      ProviderError: the judge cannot be reached or answers badly.

    Effects: net
    """
    ...


class OllamaJudgeClient:
  """Judge over Ollama chat with ``format="json"`` at temperature 0.

  Registry: judge_client[ollama]

  Attributes:
    model: The judge model.
    host: The Ollama URL, used in error messages.
  """

  def __init__(
    self, model: str, host: str | None = None, client: Any | None = None
  ) -> None:
    """Create a judge client for ``model``.

    Args:
      model: Judge model name.
      host: Ollama URL; ``None`` or blank means ``ollama_host()``.
      client: An ``ollama.Client`` or any object with the same ``chat``
        method; ``None`` builds one with a 600 s timeout.

    Raises:
      TypeError: ``model`` or ``host`` is not a ``str`` (``host`` may be ``None``).
      ValueError: ``model`` is blank.
      ProviderError: the Ollama client cannot be built.
    """
    self.model: str = _check_model(model)
    self.host: str = _resolve_host(host)
    self._client: Any = client if client is not None else _build_client(self.host)

  def complete_json(self, messages: Sequence[PromptMessage]) -> str:
    """Send ``messages`` to Ollama and return the content of its reply.

    Raises:
      TypeError: ``messages`` is not a sequence of messages.
      ValueError: ``messages`` is empty.
      ProviderError: the client fails or the reply has no text content.

    Effects: net
    """
    items = _check_messages(messages)
    payload = [{"role": m.role, "content": m.content} for m in items]
    try:
      response = self._client.chat(
        model=self.model,
        messages=payload,
        format="json",
        options={"temperature": 0},
      )
      content = _read(_read(response, "message"), "content")
      if not isinstance(content, str):
        raise TypeError("la respuesta no trae texto")
    except Exception as exc:
      raise ProviderError(
        f"Ollama falló al juzgar con el modelo {self.model!r} en {self.host}: "
        f"{_describe(exc)}"
      ) from exc
    return content


def _numbered_count(block: str) -> int:
  """How many ``[1] ...``, ``[2] ...`` entries the numbered ``block`` holds."""
  count = 0
  position = 0
  while True:
    marker = f"[{count + 1}] "
    if count == 0:
      found = 0 if block.startswith(marker) else -1
    else:
      found = block.find(f"\n{marker}", position)
    if found < 0:
      return count
    count += 1
    position = found + 1


def _add_markers(claim: dict[str, Any], text: str, cites: bool) -> None:
  """Add the ``[n]`` markers of ``text`` to the citations of ``claim``."""
  if not cites:
    return
  for number in (int(n) for n in _ECHO_MARKER.findall(text)):
    if number not in claim["citations"]:
      claim["citations"].append(number)


def _echo_claims(system: str, user: str) -> dict[str, Any]:
  """One claim per sentence of the text; a marker after the full stop goes back."""
  found = _ECHO_TEXT.search(user)
  text = user[found.end() :] if found else user
  cites = _ECHO_NO_MARKERS not in system
  claims: list[dict[str, Any]] = []
  for raw in _ECHO_SPLIT.split(text.strip()):
    fragment = raw.strip()
    lead = _ECHO_LEADING.match(fragment)
    if lead is not None and claims:
      _add_markers(claims[-1], lead.group(0), cites)
      fragment = fragment[lead.end() :].strip()
    sentence = " ".join(_ECHO_MARKER.sub(" ", fragment).split())
    if sentence:
      claims.append({"text": sentence, "citations": []})
      _add_markers(claims[-1], fragment, cites)
  return {"claims": claims}


def _echo_verify(system: str, user: str) -> dict[str, Any]:
  """Every claim supported by source 1."""
  claims = user.rpartition("\n\nAfirmaciones:\n")[2]
  return {
    "verdicts": [
      {"claim": n, "supported": True, "sources": [1]}
      for n in range(1, _numbered_count(claims) + 1)
    ]
  }


def _echo_relevance(system: str, user: str) -> dict[str, Any]:
  """Every chunk relevant."""
  chunks = user.partition("\n\nFragmentos:\n")[2]
  return {
    "verdicts": [
      {"chunk": n, "relevant": True} for n in range(1, _numbered_count(chunks) + 1)
    ]
  }


def _echo_questions(system: str, user: str) -> dict[str, Any]:
  """The first sentence of the answer as a question, as many times as asked."""
  match = _ECHO_COUNT.match(user)
  count = int(match.group(1)) if match else 3
  answer = user.partition("\nRespuesta:\n")[2]
  sentence = " ".join(_ECHO_MARKER.sub(" ", _first_sentence(answer)).split())
  sentence = sentence.strip(" .!?¿#*-")
  question = f"¿{sentence}?" if sentence else "¿Qué dice la respuesta?"
  return {"questions": [question] * count, "noncommittal": False}


def _echo_entities(system: str, user: str) -> dict[str, Any]:
  """The capitalised words and the numbers of the text, once each."""
  text = user.partition("Texto:\n")[2]
  return {"entities": list(dict.fromkeys(_ECHO_ENTITY.findall(text)))}


def _echo_toxic(system: str, user: str) -> dict[str, Any]:
  return {"toxic": False, "reason": _ECHO_NO_FLAG}


def _echo_biased(system: str, user: str) -> dict[str, Any]:
  return {"biased": False, "reason": _ECHO_NO_FLAG}


# The shape the system message asks for -> the function that builds the reply.
# The first marker found in the system message wins.
_ECHO_OPERATIONS: Final[
  tuple[tuple[str, Callable[[str, str], dict[str, Any]]], ...]
] = (
  ('{"claims"', _echo_claims),
  ('{"verdicts": [{"claim"', _echo_verify),
  ('{"verdicts": [{"chunk"', _echo_relevance),
  ('{"questions"', _echo_questions),
  ('{"entities"', _echo_entities),
  ('{"toxic"', _echo_toxic),
  ('{"biased"', _echo_biased),
)


class EchoJudgeClient:
  """Offline judge: a valid, deterministic JSON reply for every judge operation.

  Registry: judge_client[echo]

  The operation is recognised from the JSON shape that the system message asks
  for, and the reply is derived from the numbered texts of the user message,
  so its counts and indexes always match. A system message of no metric
  operation (the rule judge) gets a fixed ``partial`` verdict.

  Attributes:
    model: Always ``echo``.
  """

  model: str = "echo"

  def complete_json(self, messages: Sequence[PromptMessage]) -> str:
    """Return the reply of the operation that ``messages`` ask for, as JSON text.

    Raises:
      TypeError: ``messages`` is not a sequence of messages.
      ValueError: ``messages`` is empty.
    """
    items = _check_messages(messages)
    system = "\n".join(m.content for m in items if m.role == "system")
    user = next((m.content for m in reversed(items) if m.role == "user"), "")
    for marker, build in _ECHO_OPERATIONS:
      if marker in system:
        return json.dumps(build(system, user), ensure_ascii=False)
    return json.dumps(_ECHO_VERDICT, ensure_ascii=False)


# --- factories ----------------------------------------------------------------


def make_embedder(config: EmbeddingConfig) -> Embedder:
  """Build the embedder of ``config``: ``ollama`` or ``hash``.

  Raises:
    TypeError: ``config`` is not an ``EmbeddingConfig``.
    ValueError: the provider is unknown or the model is blank.
    ProviderError: the Ollama client cannot be built.
  """
  if not isinstance(config, EmbeddingConfig):
    raise TypeError(f"config debe ser EmbeddingConfig, no {type(config).__name__}")
  if config.provider == "ollama":
    return OllamaEmbedder(config.model)
  if config.provider == "hash":
    return HashEmbedder()
  raise ValueError(f"proveedor de embeddings desconocido: {config.provider!r}")


def make_llm(config: GenerationConfig) -> LLMClient:
  """Build the LLM client of ``config``: ``ollama`` or ``echo``.

  Raises:
    TypeError: ``config`` is not a ``GenerationConfig``.
    ValueError: the provider is unknown or the model is blank.
    ProviderError: the Ollama client cannot be built.
  """
  if not isinstance(config, GenerationConfig):
    raise TypeError(f"config debe ser GenerationConfig, no {type(config).__name__}")
  if config.provider == "ollama":
    return OllamaChatClient(_build_client(ollama_host()), config.model)
  if config.provider == "echo":
    return EchoLLM()
  raise ValueError(f"proveedor de LLM desconocido: {config.provider!r}")


def make_judge(config: JudgeConfig) -> JudgeClient:
  """Build the judge client of ``config``: ``ollama`` or ``echo``.

  Raises:
    TypeError: ``config`` is not a ``JudgeConfig``.
    ValueError: the provider is unknown or the model is blank.
    ProviderError: the Ollama client cannot be built.
  """
  if not isinstance(config, JudgeConfig):
    raise TypeError(f"config debe ser JudgeConfig, no {type(config).__name__}")
  if config.provider == "ollama":
    return OllamaJudgeClient(config.model)
  if config.provider == "echo":
    return EchoJudgeClient()
  raise ValueError(f"proveedor de juez desconocido: {config.provider!r}")


# --- preflight ----------------------------------------------------------------


def _full_name(name: str) -> str:
  return name if ":" in name else f"{name}:latest"


def _installed(response: Any) -> set[str]:
  """The model names of an Ollama ``list`` response, with ``:latest`` filled in."""
  entries = _read(response, "models")
  if isinstance(entries, str | bytes | Mapping) or not isinstance(entries, Iterable):
    raise TypeError("la respuesta no trae una lista de modelos")
  names: set[str] = set()
  for entry in entries:
    name = None
    for key in ("model", "name"):
      try:
        name = _read(entry, key)
      except KeyError:
        continue
      if isinstance(name, str) and name:
        break
      name = None
    if name is not None:
      names.add(_full_name(name))
  return names


def check_ollama(
  models: Sequence[str], host: str | None = None, client: Any | None = None
) -> None:
  """Check that Ollama answers and has every one of ``models`` pulled.

  A name without a tag matches ``name:latest``. An empty ``models`` makes no
  call. The server is asked once.

  Args:
    models: Model names that must be available.
    host: Ollama URL; ``None`` or blank means ``ollama_host()``.
    client: An ``ollama.Client`` or any object with the same ``list`` method.

  Raises:
    TypeError: ``models`` is not a sequence of ``str``, or ``host`` is not a ``str``.
    ValueError: a model name is blank.
    ProviderError: Ollama does not answer, or a model is not pulled.

  Effects: net
  """
  if isinstance(models, str | bytes) or not isinstance(models, Sequence):
    raise TypeError(f"models debe ser una lista de str, no {type(models).__name__}")
  for index, model in enumerate(models):
    if not isinstance(model, str):
      raise TypeError(f"models[{index}] debe ser un str, no {type(model).__name__}")
    if not model.strip():
      raise ValueError(f"models[{index}] no puede estar vacío")
  resolved = _resolve_host(host)
  if not models:
    return
  if client is None:
    try:
      client = ollama.Client(host=resolved, timeout=_CHECK_TIMEOUT)
    except Exception as exc:
      raise ProviderError(
        f"Ollama no responde en {resolved} ({_describe(exc)})"
      ) from exc
  try:
    installed = _installed(client.list())
  except Exception as exc:
    raise ProviderError(f"Ollama no responde en {resolved} ({_describe(exc)})") from exc
  for model in models:
    if _full_name(model) not in installed:
      raise ProviderError(
        f"el modelo {model!r} no está descargado en {resolved}; "
        f"ejecuta `ollama pull {model}`"
      )

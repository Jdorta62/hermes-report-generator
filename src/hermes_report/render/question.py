"""One question tab of the report: its evaluation, document map and every phase."""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

from hermes_enrich.tokens import TokenCounter

from hermes_report.record import PointRecord, QuestionRecord, RuleRecord
from hermes_report.render.html import (
  JUDGE_LABELS,
  NO_DATA,
  RULE_TONES,
  answer_html,
  card,
  chip,
  deep_link,
  esc,
  grid,
  json_block,
  link,
  seconds,
  section,
  status_pill,
  table,
  yes_no,
)

__all__ = ["QuestionContext", "chunk_usage", "disagree", "question_panel"]

_SNIPPET: Final = 140
_PREVIEW: Final = 160
# A document with more chunks than this gets compact map cells.
_COMPACT_CELLS: Final = 400
_NO_PLACE: Final = "—"
_UNLIMITED: Final = "sin límite"
_LEGEND: Final = (
  '<div class="legend"><span class="cell orig">n</span> recuperado'
  '<span class="cell added">n</span> añadido por el enriquecimiento'
  '<span class="cell unused">n</span> no usado'
  '<span class="cell orig dropped">n</span> descartado por presupuesto'
  '<span class="cell orig cited">n</span> citado en la respuesta'
  '<b class="exp-mark"></b> documento esperado</div>'
)


# --- chunk names ---------------------------------------------------------------------


@dataclass(frozen=True)
class _ChunkInfo:
  """What the report calls a chunk: its label, where it sits and its index."""

  label: str
  place: str
  index: int | None
  source: str | None


def _index(value: object) -> int | None:
  return value if isinstance(value, int) and not isinstance(value, bool) else None


def _label(source: object, index: object, chunk_id: str) -> str:
  """``source#chunk_index``; the ``chunk_id`` when there is no source."""
  if not isinstance(source, str) or not source:
    return chunk_id
  position = _index(index)
  return source if position is None else f"{source}#{position}"


def _place(metadata: Mapping[str, Any]) -> str:
  """The heading path joined with `` › ``, or the character range."""
  path = metadata.get("heading_path")
  if isinstance(path, list) and path:
    return " › ".join(str(item) for item in path)
  start, end = _index(metadata.get("char_start")), _index(metadata.get("char_end"))
  if start is not None and end is not None:
    return f"caracteres {start}–{end}"
  return _NO_PLACE


def _info(chunk_id: str, metadata: Mapping[str, Any]) -> _ChunkInfo:
  source = metadata.get("source")
  return _ChunkInfo(
    label=_label(source, metadata.get("chunk_index"), chunk_id),
    place=_place(metadata),
    index=_index(metadata.get("chunk_index")),
    source=source if isinstance(source, str) and source else None,
  )


def _score(value: float) -> str:
  return f"{value:.2f}" if abs(value) >= 0.1 else f"{value:.4f}"


def _preview(text: object, limit: int) -> str:
  return " ".join(str(text).split())[:limit]


class QuestionContext:
  """What every question tab shares: the Qdrant points and the token counter.

  Built once per render. The token counts are cached by text.
  """

  def __init__(self, points: Sequence[PointRecord], counter: TokenCounter) -> None:
    self._counter = counter
    self._cache: dict[str, int] = {}
    self.infos: dict[str, _ChunkInfo] = {}
    self.by_source: dict[str, list[tuple[int, PointRecord]]] = {}
    for point in points:
      info = _info(point["id"], point["payload"])
      self.infos[point["id"]] = info
      if info.source is not None and info.index is not None:
        self.by_source.setdefault(info.source, []).append((info.index, point))
    for entries in self.by_source.values():
      entries.sort(key=lambda entry: entry[0])

  @property
  def counter_name(self) -> str:
    """The name of the token counter."""
    return self._counter.name

  def tokens(self, text: str) -> int:
    """The token count of ``text``."""
    count = self._cache.get(text)
    if count is None:
      count = self._cache[text] = self._counter.count(text)
    return count


# --- evaluation helpers shared with the summary -----------------------------------


def _tone(rules: RuleRecord | None) -> str | None:
  return None if rules is None else RULE_TONES.get(rules["verdict"], "info")


def disagree(item: QuestionRecord) -> bool:
  """The rules say right and the judge wrong, or the other way round."""
  tone = _tone(item["rules"])
  judged = item["judge"]["verdict"] if item["judge"] else None
  return (tone == "good" and judged == "incorrect") or (
    tone == "bad" and judged == "correct"
  )


def _count_text(count: object) -> str:
  """A token count as text; a dash only when it is missing, so ``0`` stays ``0``."""
  return "—" if count is None else esc(count)


def chunk_usage(item: QuestionRecord) -> dict[str, list[str]]:
  """The chunk ids a question touched, by state, each sorted.

  The states are the ones the document map colours:

  - ``retrieved``: the retriever returned the chunk;
  - ``added``: the enrichment added it, in any step or in the final context;
  - ``context``: it is in a piece of the final context;
  - ``cited``: a citation of the answer points to it;
  - ``dropped``: the budget discarded it (final context or generation result).

  A question without a stage that gives a state has an empty list for it.
  """
  retrieval = item["retrieval"]
  enrich = item["enrich"]
  final = enrich["context"] if enrich else None
  generation = item["generation"]
  result = generation["result"] if generation else None
  contexts = [step["after"] for step in enrich["steps"]] if enrich else []
  if final:
    contexts.append(final)
  added = {
    chunk["chunk_id"]
    for context in contexts
    for piece in context["pieces"]
    for chunk in piece["chunks"]
    if chunk["origin"]
  }
  dropped = set(final["dropped_chunk_ids"]) if final else set()
  cited: set[str] = set()
  if result:
    dropped |= set(result.get("dropped_chunk_ids") or ())
    cited = {cid for cite in result.get("citations") or () for cid in cite["chunk_ids"]}
  return {
    "retrieved": sorted({c["chunk_id"] for c in retrieval["chunks"]})
    if retrieval
    else [],
    "added": sorted(added),
    "context": sorted(set(_chunk_ids(final))) if final else [],
    "cited": sorted(cited),
    "dropped": sorted(dropped),
  }


# --- the tab ----------------------------------------------------------------------------


def question_panel(item: QuestionRecord, question_id: str, ctx: QuestionContext) -> str:
  """The whole tab of one question, from its record alone.

  The tab holds the header, the error box, the evaluation card, the flow of
  phase chips, the document map and one section per phase that has a record.
  """
  return _Panel(item, question_id, ctx).html()


class _Block(NamedTuple):
  """One phase section: its anchor, step mark, titles and body."""

  anchor: str
  step: str
  title: str
  chip_title: str
  summary: str
  body: str


class _Panel:
  def __init__(self, item: QuestionRecord, question_id: str, ctx: QuestionContext):
    self.item = item
    self.qid = question_id
    self.ctx = ctx
    generation = item["generation"]
    self.result = generation["result"] if generation else None
    self.final = item["enrich"]["context"] if item["enrich"] else None
    seen: dict[str, _ChunkInfo] = {}
    for ref in self._refs():
      seen[ref["chunk_id"]] = _info(ref["chunk_id"], ref)
    retrieval = item["retrieval"]
    for chunk in retrieval["chunks"] if retrieval else ():
      seen[chunk["chunk_id"]] = _info(chunk["chunk_id"], chunk["metadata"])
    self.infos: Mapping[str, _ChunkInfo] = ChainMap(ctx.infos, seen)

  # -- lookups --

  def _refs(self) -> Iterator[Mapping[str, Any]]:
    """Every chunk reference of the record, wherever it appears."""
    enrich = self.item["enrich"]
    contexts = []
    if enrich:
      for step in enrich["steps"]:
        contexts += [step["before"], step["after"]]
      if enrich["context"]:
        contexts.append(enrich["context"])
    for context in contexts:
      for piece in context["pieces"]:
        yield from piece["chunks"]
    if self.result:
      for source in self.result.get("sources") or ():
        yield from source["chunks"]

  def _name(self, chunk_id: str) -> str:
    info = self.infos.get(chunk_id)
    return info.label if info else chunk_id

  def _linked(self, chunk_id: str, inner: str, css: str = "chunk-link") -> str:
    """``inner`` as a link to the explorer card of ``chunk_id``.

    The point id is the retrieval ``chunk_id`` (ingest-md stores each chunk
    under it), so the id of the record is the id of the point. A chunk with no
    point in the record has nothing to show, so ``inner`` is left as it is.
    """
    info = self.ctx.infos.get(chunk_id)
    if info is None:
      return inner
    href = deep_link(doc=info.source, question=self.qid, chunk=chunk_id)
    return link(href, inner, css)

  def _code(self, chunk_id: str, label: str | None = None) -> str:
    name = self._name(chunk_id) if label is None else label
    return self._linked(chunk_id, f"<code>{esc(name)}</code>")

  def _where(self, chunk_id: str) -> str:
    info = self.infos.get(chunk_id)
    return info.place if info else _NO_PLACE

  def _anchor(self, suffix: str) -> str:
    return f"q-{self.qid}-{suffix}"

  # -- the page --

  def html(self) -> str:
    item = self.item
    spec = item["spec"]
    pill = status_pill(self.result["status"] if self.result else "error", "generation")
    error = f'<div class="error">{esc(item["error"])}</div>' if item["error"] else ""
    blocks = [
      block
      for block in (
        self._input(),
        self._seed(),
        *self._steps(),
        self._budget(),
        self._prompt(),
        self._llm(),
        self._outcome(),
        self._judge(),
      )
      if block is not None
    ]
    flow = "".join(
      chip(block.anchor, block.chip_title, block.summary) for block in blocks
    )
    return (
      f'<div class="sc-head"><h2>{esc(spec.get("question", ""))}</h2>{pill}'
      f'<span class="muted">{esc(self.qid)}</span>'
      f'<span class="muted">{esc(seconds(item["seconds"]))}</span></div>'
      + error
      + _evaluation_card(item)
      + (f'<div class="flow">{flow}</div>' if flow else "")
      + card("Mapa de documentos", self._doc_map())
      + self._raw_docs()
      + "".join(
        section(block.anchor, block.step, block.title, block.summary, block.body)
        for block in blocks
      )
    )

  # -- 0 Entrada --

  def _input(self) -> _Block | None:
    retrieval = self.item["retrieval"]
    if retrieval is None:
      return None
    chunks = retrieval["chunks"]
    queries = retrieval["queries"]
    if queries:
      query_rows = [
        f"<tr><td>{number}</td><td>{esc(query['text'])}</td><td>"
        + (
          "".join(f'<span class="tag">{esc(name)}</span> ' for name in query["chain"])
          or NO_DATA
        )
        + "</td></tr>"
        for number, query in enumerate(queries, 1)
      ]
      searched = table(
        [("#", False), ("consulta", False), ("técnicas", False)], query_rows
      )
    else:
      searched = '<p class="muted">Sin consultas.</p>'
    has_vector = any(chunk["vector_score"] is not None for chunk in chunks)
    chunk_rows = []
    for number, chunk in enumerate(chunks, 1):
      metadata = chunk["metadata"]
      text = chunk["text"].strip()
      snippet = text[:_SNIPPET] + ("…" if len(text) > _SNIPPET else "")
      vector = chunk["vector_score"]
      chunk_rows.append(
        f"<tr><td>{number}</td>"
        f"<td>{self._code(chunk['chunk_id'], _info(chunk['chunk_id'], metadata).label)}"
        "</td>"
        f"<td>{esc(_place(metadata))}</td><td>{_score(chunk['score'])}</td>"
        + (
          f"<td>{NO_DATA if vector is None else _score(vector)}</td>"
          if has_vector
          else ""
        )
        + f'<td class="snippet">{esc(snippet)}</td></tr>'
      )
    headers = [("#", False), ("chunk", False), ("ubicación", False), ("score", False)]
    if has_vector:
      headers.append(("vector", False))
    headers.append(("texto", False))
    found = (
      table(headers, chunk_rows) if chunk_rows else '<p class="muted">Lista vacía.</p>'
    )
    body = (
      f'<div class="question">{esc(self.item["spec"].get("question", ""))}</div>'
      f"<h4>Consultas buscadas</h4>{searched}"
      f"<h4>Salida del retriever (en su orden)</h4>{found}"
    )
    summary = f"{len(chunks)} chunks"
    return _Block(self._anchor("in"), "0", "Entrada", "Retrieval", summary, body)

  # -- S Semilla --

  def _seed(self) -> _Block | None:
    retrieval = self.item["retrieval"]
    enrich = self.item["enrich"]
    if retrieval is None:
      return None
    anchor = self._anchor("seed")
    if enrich and enrich["steps"]:
      seed = enrich["steps"][0]["before"]
      pieces = seed["pieces"]
      summary = f"{len(pieces)} piezas · {self._context_tokens(seed)} tok"
      body = (
        '<p class="muted">Una pieza por <code>chunk_id</code> distinto, en el orden '
        "del retriever.</p>" + self._pieces(pieces)
      )
      return _Block(anchor, "S", "Semilla del contexto", "Semilla", summary, body)
    if not retrieval["chunks"]:
      body = (
        '<p class="muted">Sin chunks no hay semilla: la cadena de enriquecimiento '
        "no se ejecuta.</p>"
      )
      return _Block(anchor, "S", "Semilla del contexto", "Semilla", "vacía", body)
    return None

  # -- E1…En --

  def _steps(self) -> list[_Block]:
    enrich = self.item["enrich"]
    if enrich is None:
      return []
    blocks = []
    for number, step in enumerate(enrich["steps"], 1):
      before_ids = _chunk_ids(step["before"])
      new_ids = frozenset(i for i in _chunk_ids(step["after"]) if i not in before_ids)
      before_n, after_n = len(step["before"]["pieces"]), len(step["after"]["pieces"])
      changes = []
      if new_ids:
        changes.append(f"+{len(new_ids)} chunks")
      if before_n != after_n:
        changes.append(f"{before_n}→{after_n} piezas")
      if not changes:
        reordered = [_piece_ids(p) for p in step["before"]["pieces"]] != [
          _piece_ids(p) for p in step["after"]["pieces"]
        ]
        changes.append("reordena" if reordered else "sin cambios")
      summary = ", ".join(changes)
      body = (
        f'<p class="muted">Fase <code>{esc(step["phase"])}</code> · '
        f"{step['seconds'] * 1000:.1f} ms · "
        f"{self._context_tokens(step['after'])} tok tras la fase</p>"
        f"<h4>Lecturas a Qdrant</h4>{self._reads(step['reads'])}"
        '<h4>Orden de piezas</h4><div class="order"><div><span class="k">Antes</span> '
        f"{self._order(step['before'])}</div>"
        f'<div><span class="k">Después</span> {self._order(step["after"])}</div></div>'
        f"<h4>Piezas tras {esc(step['name'])}</h4>"
        + self._pieces(step["after"]["pieces"], new_ids)
      )
      blocks.append(
        _Block(
          self._anchor(f"e{number}"),
          str(number),
          f"Enriquecedor: {step['name']}",
          step["name"],
          summary,
          body,
        )
      )
    return blocks

  def _order(self, context: Mapping[str, Any]) -> str:
    parts = []
    for number, piece in enumerate(context["pieces"], 1):
      chunks = piece["chunks"]
      if not chunks:
        parts.append(f"{number}. ?")
        continue
      first = chunks[0]["chunk_id"]
      text = f"{number}. {self._linked(first, esc(self._name(first)))}"
      if len(chunks) > 1:
        text += f"…{esc(self._name(chunks[-1]['chunk_id']).split('#')[-1])}"
      parts.append(text)
    return " · ".join(parts) or NO_DATA

  def _reads(self, reads: Sequence[Mapping[str, Any]]) -> str:
    if not reads:
      return '<p class="muted">No consulta Qdrant.</p>'
    out = []
    for read in reads:
      ranges = ", ".join(
        f"{esc(r['collection_name'])} / {esc(r['doc_id'][:16])} "
        f"[{r['first']}–{r['last']}]"
        for r in read["ranges"]
      )
      returned = sorted(
        read["returned"],
        key=lambda cid: (
          info.index if (info := self.infos.get(cid)) and info.index is not None else 0
        ),
      )
      got = " ".join(self._code(cid) for cid in returned)
      out.append(
        f'<div class="read"><div><span class="k">Pide</span> {ranges}</div>'
        f'<div><span class="k">Recibe</span> {len(returned)} chunk(s): {got}</div>'
        f'<div class="muted">{read["seconds"] * 1000:.1f} ms</div></div>'
      )
    return "".join(out)

  # -- pieces --

  def _context_tokens(self, context: Mapping[str, Any]) -> int:
    return sum(self.ctx.tokens(piece["text"]) for piece in context["pieces"])

  def _pill(self, ref: Mapping[str, Any], extra: str = "") -> str:
    origin = ref["origin"]
    classes = " ".join(
      part for part in ("pill", "added" if origin else "orig", extra) if part
    )
    detail = (
      f"score {_score(ref['score'])}"
      if ref["score"] is not None
      else "añadido por " + ", ".join(origin)
    )
    chunk_id = ref["chunk_id"]
    return self._linked(
      chunk_id,
      f'<span class="{classes}" '
      f'title="{esc(self._where(chunk_id))} · {esc(chunk_id)}">'
      f"<b>{esc(self._name(chunk_id))}</b><i>{esc(detail)}</i></span>",
    )

  def _pieces(
    self,
    pieces: Sequence[Mapping[str, Any]],
    new_ids: frozenset[str] = frozenset(),
    open_text: bool = False,
  ) -> str:
    if not pieces:
      return '<p class="muted">Sin piezas.</p>'
    cards = []
    for number, piece in enumerate(pieces, 1):
      chunks = piece["chunks"]
      transforms = "".join(
        f'<span class="tag">{esc(name)}</span>' for name in piece["transforms"]
      )
      pills = "".join(
        self._pill(chunk, "new" if chunk["chunk_id"] in new_ids else "")
        for chunk in chunks
      )
      scores = [
        c["score"] for c in chunks if not c["origin"] and c["score"] is not None
      ]
      score = _score(max(scores)) if scores else "—"
      title = self._where(chunks[0]["chunk_id"]) if chunks else _NO_PLACE
      collection = chunks[0]["collection_name"] if chunks else ""
      cards.append(
        f'<div class="piece"><div class="piece-head"><span class="num">{number}</span>'
        f'<span class="ptitle">{esc(title)}</span>'
        f'<span class="coll">{esc(collection)}</span>{transforms}'
        f'<span class="meta">{len(chunks)} chunk(s) · score {score} · '
        f"{self.ctx.tokens(piece['text'])} tok</span></div>"
        f'<div class="pills">{pills}</div>'
        f"<details{' open' if open_text else ''}><summary>Texto</summary>"
        f'<pre class="text">{esc(piece["text"])}</pre></details></div>'
      )
    return "".join(cards)

  # -- B Presupuesto --

  def _budget(self) -> _Block | None:
    enrich = self.item["enrich"]
    final = self.final
    if enrich is None or final is None or not enrich["steps"]:
      return None
    before = enrich["steps"][-1]["after"]
    budget = enrich["budget"]
    cap = budget.get("max_context_tokens")
    max_chunks = budget.get("max_chunks")
    dropped = final["dropped_chunk_ids"]
    by_id = {
      chunk["chunk_id"]: chunk
      for piece in before["pieces"]
      for chunk in piece["chunks"]
    }
    items = []
    for chunk_id in dropped:
      ref = by_id.get(chunk_id)
      items.append(
        f"<li>{self._pill(ref, 'gone')}</li>"
        if ref is not None
        else f'<li><span class="pill gone"><b>{esc(self._name(chunk_id))}</b></span></li>'
      )
    tok_before = self._context_tokens(before)
    tok_after = self._context_tokens(final)
    limit = str(cap) if cap else _UNLIMITED
    chunks_limit = str(max_chunks) if max_chunks else _UNLIMITED
    pct_after = min(100.0, 100 * tok_after / cap) if cap else 0.0
    pct_before = min(100.0, 100 * tok_before / cap) if cap else 0.0
    n_before, n_after = len(_chunk_ids(before)), len(_chunk_ids(final))
    body = (
      f'<div class="kv"><span class="k">Límite</span> {esc(limit)} tok · '
      f"{esc(chunks_limit)} chunks"
      f'<span class="k">Antes</span> {tok_before} tok · {n_before} chunks'
      f'<span class="k">Después</span> {tok_after} tok · {n_after} chunks</div>'
      f'<div class="bar"><div class="bar-before" style="width:{pct_before:.1f}%"></div>'
      f'<div class="bar-after" style="width:{pct_after:.1f}%"></div></div>'
      + (
        f'<h4>Descartados, en orden</h4><ol class="dropped">{"".join(items)}</ol>'
        if items
        else '<p class="muted">Todo cabe: no se descarta nada.</p>'
      )
      + "<h4>EnrichedContext (contrato de hermes-core)</h4>"
      '<p class="muted">Lo que devuelve <code>HermesEnrich.enrich</code> y '
      "recibe <code>HermesGenerator.generate</code>: lo que verá el LLM.</p>"
      + self._pieces(final["pieces"], open_text=len(final["pieces"]) <= 4)
    )
    summary = (f"{tok_after}/{cap} tok" if cap else f"{tok_after} tok") + (
      f" · −{len(dropped)} chunks" if dropped else ""
    )
    return _Block(
      self._anchor("budget"),
      "B",
      "Ajuste al presupuesto",
      "Presupuesto",
      summary,
      body,
    )

  # -- P Prompt --

  def _prompt(self) -> _Block | None:
    prompt = self.result.get("prompt") if self.result else None
    if not prompt:
      return None
    summary = f"{len(prompt)} mensajes"
    return _Block(
      self._anchor("prompt"),
      "P",
      "Prompt enviado",
      "Prompt",
      summary,
      _messages(prompt),
    )

  # -- L LLM --

  def _llm(self) -> _Block | None:
    generation = self.item["generation"]
    if generation is None:
      return None
    anchor = self._anchor("llm")
    calls = generation["llm_calls"]
    if not calls:
      if self.result is not None:
        body = (
          '<p class="muted">El contexto está vacío: el generador contesta sin '
          "llamar al LLM.</p>"
        )
        return _Block(anchor, "L", "LLM", "LLM", "no se llama", body)
      body = f'<div class="error">{esc(self.item["error"] or "")}</div>'
      return _Block(anchor, "L", "LLM", "LLM", "error", body)
    parts = []
    for number, call in enumerate(calls, 1):
      parts.append(
        (f"<h4>Llamada {number} de {len(calls)}</h4>" if len(calls) > 1 else "")
        + self._call(call)
      )
    total = sum(call["seconds"] for call in calls)
    summary = (
      f"{total:.2f} s" if len(calls) == 1 else f"{len(calls)} llamadas · {total:.2f} s"
    )
    if any(call["reply"] is None for call in calls):
      summary = "error" if len(calls) == 1 else f"{summary} · con error"
    return _Block(anchor, "L", "LLM", "LLM", summary, "".join(parts))

  def _call(self, call: Mapping[str, Any]) -> str:
    reply = call["reply"]
    if reply is None:
      return f'<div class="error">{esc(call["error"] or "")}</div>'
    params = ", ".join(
      f"{key}={value}" for key, value in call["params"].items() if value is not None
    )
    return (
      f'<div class="kv"><span class="k">Modelo</span> {esc(reply.get("model") or "?")}'
      f'<span class="k">Parámetros</span> <code>{esc(params)}</code>'
      f'<span class="k">Tokens</span> prompt {_count_text(reply.get("prompt_tokens"))}'
      f" · respuesta {_count_text(reply.get('completion_tokens'))}"
      f'<span class="k">Tiempo</span> {call["seconds"]:.2f} s</div>'
      f'<h4>Respuesta cruda</h4><pre class="text raw">{esc(reply["text"])}</pre>'
    )

  # -- R Resultado --

  def _outcome(self) -> _Block | None:
    result = self.result
    if result is None:
      return None
    names = {
      source["number"]: sorted(
        {chunk.get("source") or "?" for chunk in source["chunks"]}
      )
      for source in result.get("sources") or ()
    }
    cite_rows = [
      f'<tr><td><span class="cite">{esc(cite["number"])}</span></td><td>'
      + " ".join(self._code(cid) for cid in cite["chunk_ids"])
      + f"</td><td>{esc(', '.join(names.get(cite['number'], [])))}</td></tr>"
      for cite in result.get("citations") or ()
    ]
    cites = (
      table([("cita", False), ("chunks", False), ("fichero", False)], cite_rows)
      if cite_rows
      else '<p class="muted">Sin citas.</p>'
    )
    body = (
      f'<div class="answer">{answer_html(str(result["answer"]))}</div>'
      f'<div class="kv"><span class="k">Estado</span> '
      f"{status_pill(result['status'], 'generation')}"
      f'<span class="k">Estrategia</span> {esc(result.get("strategy", "?"))}'
      f'<span class="k">Formato</span> {esc(result.get("response_format", "?"))}'
      f'<span class="k">Enriquecimientos</span> '
      f"{esc(' → '.join(result.get('enrichments') or ()) or '—')}"
      f'<span class="k">Fuentes</span> {len(result.get("sources") or ())} piezas</div>'
      f"<h4>Citas</h4>{cites}"
      "<details><summary>GenerationResult completo (JSON)</summary>"
      f"{json_block(result)}</details>"
    )
    return _Block(
      self._anchor("out"), "R", "GenerationResult", "Resultado", result["status"], body
    )

  # -- J Juez --

  def _judge(self) -> _Block | None:
    judge = self.item["judge"]
    if judge is None:
      return None
    body = (
      grid(
        [
          ("Modelo", esc(judge["model"])),
          ("Veredicto", status_pill(judge["verdict"], "judge")),
          ("Motivo", esc(judge["reason"] or "—")),
          ("Tiempo", esc(seconds(judge["seconds"]))),
        ]
      )
      + "<h4>Mensajes enviados</h4>"
      + _messages(judge["messages"])
      + "<h4>Respuesta cruda</h4>"
      + (
        f'<pre class="text raw">{esc(judge["raw"])}</pre>'
        if judge["raw"] is not None
        else '<p class="muted">Sin respuesta.</p>'
      )
    )
    if judge["error"]:
      body = f'<div class="error">{esc(judge["error"])}</div>' + body
    verdict = judge["verdict"]
    summary = (
      "error" if judge["error"] else JUDGE_LABELS.get(verdict or "", verdict or "—")
    )
    return _Block(self._anchor("judge"), "J", "Juez", "Juez", summary, body)

  # -- document map --

  def _sources(self, usage: Mapping[str, Sequence[str]]) -> list[str]:
    """The documents the question touched: the expected ones and those of any
    retrieved or added chunk, sorted. The map and the raw view list the same."""
    sources = {str(name) for name in self.item["spec"].get("documents", ())}
    for chunk_id in (*usage["retrieved"], *usage["added"]):
      info = self.infos.get(chunk_id)
      if info is not None and info.source is not None:
        sources.add(info.source)
    return sorted(sources)

  def _doc_map(self) -> str:
    expected = {str(name) for name in self.item["spec"].get("documents", ())}
    usage = chunk_usage(self.item)
    retrieved, added = set(usage["retrieved"]), set(usage["added"])
    dropped, cited = set(usage["dropped"]), set(usage["cited"])
    sources = self._sources(usage)
    if not sources:
      return '<p class="muted">No hay documentos que mostrar.</p>'
    out = []
    for source in sources:
      entries = self.ctx.by_source.get(source, [])
      compact = " sm" if len(entries) > _COMPACT_CELLS else ""
      cells = []
      for index, point in entries:
        chunk_id = point["id"]
        state = (
          "orig"
          if chunk_id in retrieved
          else "added"
          if chunk_id in added
          else "unused"
        )
        classes = f"cell{compact} {state}"
        if chunk_id in dropped:
          classes += " dropped"
        if chunk_id in cited:
          classes += " cited"
        payload = point["payload"]
        text = payload.get("text")
        tip = " · ".join(
          part
          for part in (
            f"#{index}",
            " › ".join(str(h) for h in payload.get("heading_path") or ())
            if isinstance(payload.get("heading_path"), list)
            else "",
            _preview("" if text is None else text, _PREVIEW),
          )
          if part
        )
        cells.append(
          self._linked(
            chunk_id,
            f'<span class="{classes}" title="{esc(tip)}">{index}</span>',
            "cl",
          )
        )
      is_expected = source in expected
      count = f"{len(entries)} chunks" + (" · esperado" if is_expected else "")
      out.append(
        f'<div class="maprow{" expected" if is_expected else ""}">'
        f'<div class="mapdoc"><b>{esc(source)}</b><span>{esc(count)}</span></div>'
        '<div class="cells">'
        + ("".join(cells) or '<span class="muted">sin chunks en la colección</span>')
        + "</div></div>"
      )
    return "".join(out) + _LEGEND

  # -- raw documents --

  def _raw_docs(self) -> str:
    """The section of the documents the question touched, as stored in Qdrant.

    Each document is a collapsed ``<details>`` that ``report.js`` fills the first
    time it opens, from the embedded points and with this question's states.
    """
    sources = self._sources(chunk_usage(self.item))
    if not sources:
      return ""
    qid = esc(self.qid)
    items = "".join(
      f'<details class="rawdoc" data-doc="{esc(source)}"><summary>'
      f"<code>{esc(source)}</code> "
      f'<span class="muted">{_chunk_count(len(self.ctx.by_source.get(source, [])))}'
      "</span>"
      '</summary><div class="rawbody"></div></details>'
      for source in sources
    )
    explorer = link(
      deep_link(question=self.qid, used=True),
      "Abrir en el explorador solo los chunks que usó esta pregunta",
      "raw-link",
    )
    return (
      f'<div class="card rawdocs" id="raw-{qid}" data-q="{qid}">'
      "<h3>Documentos tocados (crudo)</h3>"
      '<p class="muted">Los chunks tal como están en Qdrant, en el orden del '
      "documento, con el estado que tienen para esta pregunta.</p>"
      f"<p>{explorer}</p>{items}"
      '<noscript><p class="muted">Se necesita JavaScript para ver los chunks.</p>'
      "</noscript></div>"
    )


def _chunk_count(count: int) -> str:
  return "1 chunk" if count == 1 else f"{count} chunks"


def _chunk_ids(context: Mapping[str, Any]) -> list[str]:
  return [chunk["chunk_id"] for piece in context["pieces"] for chunk in piece["chunks"]]


def _piece_ids(piece: Mapping[str, Any]) -> list[str]:
  return [chunk["chunk_id"] for chunk in piece["chunks"]]


def _messages(messages: Sequence[Mapping[str, Any]]) -> str:
  if not messages:
    return '<p class="muted">Sin mensajes.</p>'
  return "".join(
    f'<div class="msg {esc(m["role"])}"><div class="role">{esc(m["role"])}</div>'
    f'<pre class="text">{esc(m["content"])}</pre></div>'
    for m in messages
  )


# --- evaluation card ---------------------------------------------------------------------


def _evaluation_card(item: QuestionRecord) -> str:
  spec = item["spec"]
  rules = item["rules"]
  judge = item["judge"]
  generation = item["generation"]
  result = generation["result"] if generation else None
  retrieval = item["retrieval"]
  items: list[tuple[str, str]] = [
    ("Veredicto", status_pill(rules["verdict"] if rules else None))
  ]
  if result is not None:
    items.append(
      ("Respuesta", f'<div class="answer">{answer_html(str(result["answer"]))}</div>')
    )
  items.append(("Esperada", esc(spec.get("expected", ""))))
  if spec.get("refusal"):
    items.append(("Se espera rechazo", "sí"))
  items.append(("Palabras clave", esc(", ".join(spec.get("keywords", ())) or "—")))
  evidence = spec.get("evidence", ())
  if evidence:
    items.append(("Evidencia", f"«{esc(' … '.join(evidence))}»"))
  documents = spec.get("documents", ())
  if documents:
    items.append(
      (
        "Documentos esperados",
        f"{esc(', '.join(documents))} "
        f"{yes_no(rules['expected_docs_retrieved'] if rules else None)}",
      )
    )
  if rules is not None:
    rank = rules["retrieved_rank"]
    total = len(retrieval["chunks"]) if retrieval else 0
    items += [
      (
        "¿Recuperada?",
        yes_no(rules["in_retrieval"])
        + (
          f' <span class="muted">primer chunk con la evidencia: #{rank} de {total}'
          "</span>"
          if rank
          else ""
        ),
      ),
      ("¿En el contexto final?", yes_no(rules["in_context"])),
      ("¿En una pieza citada?", yes_no(rules["in_cited"])),
      ("¿Respuesta correcta?", yes_no(rules["answer_ok"])),
    ]
  if retrieval is not None:
    items.append(
      (
        "Recuperación",
        f"{len(retrieval['chunks'])} chunks · {esc(seconds(retrieval['seconds']))}",
      )
    )
  items.append(
    (
      "Juez",
      status_pill(judge["verdict"], "judge")
      if judge
      else "sin juez para esta pregunta",
    )
  )
  if judge is not None:
    items += [
      ("Motivo del juez", esc(judge["reason"] or "—")),
      ("Modelo del juez", esc(judge["model"])),
      ("Tiempo del juez", esc(seconds(judge["seconds"]))),
    ]
    if disagree(item):
      items.append(
        ("Discrepancia", '<span class="status mark">reglas y juez discrepan</span>')
      )
  body = grid(items)
  if judge is not None and judge["error"]:
    body += f'<div class="error">{esc(judge["error"])}</div>'
  if judge is not None and judge["raw"] is not None:
    body += (
      '<details class="raw-reply"><summary>Respuesta cruda del juez</summary>'
      f'<pre class="text raw">{esc(judge["raw"])}</pre></details>'
    )
  return card("Evaluación", body)

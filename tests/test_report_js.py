"""Behaviour of ``report.js``: the explorer, deep links and the raw documents.

Choice (task 3.3.3): the script is run with ``node`` over a small DOM written for
these tests (``DOM_STUB`` below: elements, class lists, events, a few selectors).
No library and no new dependency; the whole file is skipped when ``node`` is not on
``PATH``. The data the script reads is the real thing: the two JSON blocks are cut
out of a page rendered by ``render_run``, so the escaping (``\\u003c``) and the
parsing are tested together. The stub does not model layout, so the scrolling is
checked through ``scrollIntoView`` calls.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hermes_report.record import RunRecord
from hermes_report.render import html as h
from hermes_report.render import render_run

NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

DOM_STUB = r"""
'use strict';
class Node {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.attrs = {};
    this.className = '';
    this._text = '';
    this.listeners = {};
    this.style = {};
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.open = false;
    this.scrolled = 0;
    const self = this;
    this.dataset = new Proxy({}, {
      get: (_, name) => self.attrs['data-' + String(name)],
    });
  }
  set innerHTML(v) { throw new Error('innerHTML must never be used'); }
  get id() { return this.attrs.id; }
  set id(v) { this.attrs.id = v; }
  get classList() {
    const self = this;
    const names = () => self.className.split(/\s+/).filter(Boolean);
    return {
      contains: (c) => names().includes(c),
      add: (c) => { if (!names().includes(c)) self.className = names().concat([c]).join(' '); },
      remove: (c) => { self.className = names().filter((n) => n !== c).join(' '); },
      toggle: (c, on) => {
        const has = names().includes(c);
        const want = on === undefined ? !has : !!on;
        if (want && !has) self.className = names().concat([c]).join(' ');
        if (!want && has) self.className = names().filter((n) => n !== c).join(' ');
      },
    };
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  set href(v) { this.attrs.href = v; }
  get href() { return this.attrs.href; }
  appendChild(child) {
    if (child.parentNode) child.parentNode.children = child.parentNode.children.filter((c) => c !== child);
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  set textContent(v) {
    this.children.forEach((c) => { c.parentNode = null; });
    this.children = [];
    this._text = String(v);
  }
  addEventListener(type, fn, capture) {
    (this.listeners[type] = this.listeners[type] || []).push({ fn, capture: !!capture });
  }
  matches(sel) {
    const m = /^([a-zA-Z0-9]*)((?:\.[\w-]+|\[[^\]]+\])*)$/.exec(sel);
    if (!m) throw new Error('selector not supported by the stub: ' + sel);
    if (m[1] && this.tagName !== m[1].toUpperCase()) return false;
    const rest = m[2].match(/\.[\w-]+|\[[^\]]+\]/g) || [];
    return rest.every((part) => {
      if (part[0] === '.') return this.classList.contains(part.slice(1));
      const a = /^\[([\w-]+)(?:(\^?=)['"]?(.*?)['"]?)?\]$/.exec(part);
      const v = this.getAttribute(a[1]);
      if (v === null) return false;
      if (!a[2]) return true;
      return a[2] === '=' ? v === a[3] : v.startsWith(a[3]);
    });
  }
  closest(sel) {
    for (let n = this; n && n.tagName; n = n.parentNode) if (n.matches(sel)) return n;
    return null;
  }
  all(sel) {
    const out = [];
    const walk = (n) => n.children.forEach((c) => { if (c.matches(sel)) out.push(c); walk(c); });
    walk(this);
    return out;
  }
  querySelectorAll(sel) { return this.all(sel); }
  querySelector(sel) { return this.all(sel)[0] || null; }
  scrollIntoView() { this.scrolled += 1; }
}

function makeEnv(scenario) {
  const body = new Node('body');
  const documentListeners = {};
  const document = {
    title: scenario.title || 'T',
    body,
    createElement: (tag) => new Node(tag),
    getElementById: (id) => {
      if (scenario.noGetElementById) throw new Error('no getElementById');
      return findId(body, id);
    },
    querySelectorAll: (sel) => body.all(sel),
    addEventListener: (type, fn, capture) => {
      (documentListeners[type] = documentListeners[type] || []).push({ fn, capture: !!capture });
    },
  };
  function findId(node, id) {
    for (const c of node.children) {
      if (c.attrs.id === id) return c;
      const hit = findId(c, id);
      if (hit) return hit;
    }
    return null;
  }
  const store = Object.assign({}, scenario.storage || {});
  const windowListeners = {};
  const scrollTo = [];
  const location = { hash: scenario.hash || '' };
  const history = { replaceState(a, b, url) { location.hash = url; } };
  const window = {
    addEventListener: (type, fn) => { (windowListeners[type] = windowListeners[type] || []).push(fn); },
    scrollTo: (x, y) => { scrollTo.push([x, y]); },
  };
  const localStorage = {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = v; },
  };

  function el(tag, attrs, cls) {
    const n = new Node(tag);
    Object.keys(attrs || {}).forEach((k) => n.setAttribute(k, attrs[k]));
    if (cls) n.className = cls;
    return n;
  }
  // The skeleton of the page that report.js expects (see render/run.py).
  const nav = el('nav');
  const main = el('main');
  body.appendChild(nav);
  body.appendChild(main);
  const panels = {};
  scenario.keys.forEach((key) => {
    const tab = el('button', { 'data-key': key }, 'tab');
    nav.appendChild(tab);
    const panel = el('section', { 'data-key': key, id: 'tab-' + key }, 'panel');
    main.appendChild(panel);
    panels[key] = panel;
  });
  if (scenario.explorer !== false && panels.chunks) {
    panels.chunks.appendChild(el('div', { id: 'explorer' }));
  }
  const raw = {};
  Object.keys(scenario.raw || {}).forEach((qid) => {
    const holder = el('div', { id: 'raw-' + qid, 'data-q': qid }, 'card rawdocs');
    raw[qid] = {};
    scenario.raw[qid].forEach((source) => {
      const d = el('details', { 'data-doc': source }, 'rawdoc');
      d.appendChild(el('summary'));
      d.appendChild(el('div', {}, 'rawbody'));
      holder.appendChild(d);
      raw[qid][source] = d;
    });
    (panels['q-' + qid] || main).appendChild(holder);
  });
  if (scenario.pointsText !== null && scenario.pointsText !== undefined) {
    const s = el('script', { type: 'application/json', id: 'hr-points' });
    s.textContent = scenario.pointsText;
    body.appendChild(s);
  }
  if (scenario.usageText !== null && scenario.usageText !== undefined) {
    const s = el('script', { type: 'application/json', id: 'hr-usage' });
    s.textContent = scenario.usageText;
    body.appendChild(s);
  }

  function fire(node, type, extra) {
    const ev = Object.assign({ type, target: node, defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; } }, extra || {});
    if (type === 'toggle') {
      // `toggle` does not bubble: the script listens in the capture phase on document.
      (documentListeners.toggle || []).filter((l) => l.capture).forEach((l) => l.fn(ev));
      ((node.listeners || {}).toggle || []).forEach((l) => l.fn(ev));
      return ev;
    }
    for (let n = node; n; n = n.parentNode) {
      ((n.listeners || {})[type] || []).forEach((l) => l.fn(ev));
    }
    (documentListeners[type] || []).forEach((l) => l.fn(ev));
    return ev;
  }

  const api = {
    Node, document, location, history, window, localStorage, store, scrollTo, body, panels, raw,
    tabs: () => body.all('.tab'),
    explorer: () => document.getElementById('explorer'),
    fire,
    click: (node) => fire(node, 'click'),
    open: (details) => { details.open = true; return fire(details, 'toggle'); },
    hashchange: (hash) => {
      location.hash = hash;
      (windowListeners.hashchange || []).forEach((fn) => fn({}));
    },
    setValue: (node, value, type) => { node.value = value; fire(node, type || 'change'); },
    setChecked: (node, on) => { node.checked = on; fire(node, 'change'); },
    type: (node, text) => { node.value = text; fire(node, 'input'); },
    // what a test usually wants to read
    cards: (root) => root.all('article'),
    cardIds: (root) => root.all('article').map((c) => c.getAttribute('data-chunk')),
    byClass: (root, cls) => root.all('.' + cls),
    texts: (nodes) => nodes.map((n) => n.textContent),
    controls: () => {
      const ex = document.getElementById('explorer');
      const selects = ex.all('select');
      return {
        root: ex,
        docSelect: selects[0],
        questionSelect: selects[1],
        search: ex.all('input').find((i) => i.type === 'search'),
        used: ex.all('input').find((i) => i.type === 'checkbox'),
        usedLabel: ex.all('label')[2],
        count: ex.all('.ex-count')[0],
        pagers: ex.all('.pager'),
        list: ex.all('.ex-list')[0],
      };
    },
    pager: (pagerNode) => {
      const [previous, label, next] = [pagerNode.children[0], pagerNode.children[1], pagerNode.children[2]];
      return { previous, next, label: label.textContent };
    },
    optionValues: (select) => select.all('option').map((o) => o.value),
    optionLabels: (select) => select.all('option').map((o) => o.textContent),
  };
  return { api, window, document, location, history, localStorage };
}

const scenario = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const source = require('fs').readFileSync(process.argv[2], 'utf8');
const env = makeEnv(scenario);
let result;
try {
  new Function('document', 'location', 'window', 'localStorage', 'history', source)(
    env.document, env.location, env.window, env.localStorage, env.history);
  const body = new Function('t', scenario.steps);
  result = { ok: true, value: body(env.api) };
} catch (error) {
  result = { ok: false, error: String(error && error.stack || error) };
}
console.log(JSON.stringify(result));
"""


# ------------------------------------------------------------------ plumbing


@pytest.fixture(scope="module")
def stub(tmp_path_factory: pytest.TempPathFactory) -> Path:
  path = tmp_path_factory.mktemp("js") / "stub.js"
  path.write_text(DOM_STUB, encoding="utf-8")
  return path


@pytest.fixture(scope="module")
def script(tmp_path_factory: pytest.TempPathFactory) -> Path:
  path = tmp_path_factory.mktemp("jsrc") / "report.js"
  path.write_text(
    (h._ASSETS / "report.js").read_text(encoding="utf-8"), encoding="utf-8"
  )
  return path


Run = Callable[..., Any]


@pytest.fixture
def run_js(stub: Path, script: Path, tmp_path: Path) -> Run:
  """``run_js(steps, points=..., usage=..., ...)``: runs ``report.js`` and ``steps``.

  ``steps`` is JavaScript with the test API ``t`` in scope and a ``return``; its
  JSON-able value comes back. Points and usage are written the way the page does
  (``json_script``) and handed over as the text of the two JSON blocks.
  """
  assert NODE

  def run(
    steps: str,
    *,
    points: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
    keys: list[str] | None = None,
    raw: dict[str, list[str]] | None = None,
    hash_: str = "",
    storage: dict[str, str] | None = None,
    raw_texts: tuple[str | None, str | None] | None = None,
    explorer: bool = True,
  ) -> Any:
    pointsText: str | None
    usageText: str | None
    if raw_texts is None:
      pointsText = _block_text(
        h.json_script("hr-points", points if points is not None else [])
      )
      usageText = _block_text(
        h.json_script("hr-usage", usage if usage is not None else {})
      )
    else:
      pointsText, usageText = raw_texts
    ids = list(usage or {})
    scenario = {
      "keys": keys or ["resumen", "corpus", "chunks", *[f"q-{q}" for q in ids]],
      "steps": steps,
      "pointsText": pointsText,
      "usageText": usageText,
      "hash": hash_,
      "storage": storage or {},
      "raw": raw or {},
      "explorer": explorer,
    }
    file = tmp_path / "scenario.json"
    file.write_text(json.dumps(scenario), encoding="utf-8")
    done = subprocess.run(
      [NODE, str(stub), str(script), str(file)],
      capture_output=True,
      text=True,
      timeout=60,
      check=False,
    )
    assert done.returncode == 0, done.stderr
    outcome = json.loads(done.stdout)
    assert outcome["ok"], outcome["error"]
    return outcome["value"]

  return run


def _block_text(block: str) -> str:
  match = re.fullmatch(
    r'<script type="application/json" id="[^"]+">(.*)</script>\n', block, re.S
  )
  assert match
  return match.group(1)


STATE_KEYS = ["retrieved", "added", "context", "cited", "dropped"]


def usage_entry(**kwargs: list[str]) -> dict[str, list[str]]:
  return {key: kwargs.get(key, []) for key in STATE_KEYS}


def point(
  chunk_id: str,
  source: str = "a.md",
  index: int | None = 0,
  text: str = "texto",
  heading: list[str] | None = None,
  **extra: Any,
) -> dict[str, Any]:
  payload: dict[str, Any] = {"source": source, "text": text, **extra}
  if index is not None:
    payload["chunk_index"] = index
  if heading is not None:
    payload["heading_path"] = heading
  return {
    "id": chunk_id,
    "payload": payload,
    "vector": {"dimension": 3, "norm": 1.0, "head": [0.1, 0.2, 0.3]},
  }


def synthetic(count: int, per_source: int = 40) -> list[dict[str, Any]]:
  """``count`` points over several sources, ids ``p000``..., record order kept."""
  return [
    point(
      f"p{n:03d}",
      source=f"doc{n // per_source}.md",
      index=n % per_source,
      text=f"chunk number {n}",
      heading=["Parte", f"Sección {n}"],
    )
    for n in range(count)
  ]


# ------------------------------------------------------------------ the script as a whole


def test_the_script_parses_with_node(script: Path) -> None:
  assert NODE
  done = subprocess.run([NODE, "--check", str(script)], capture_output=True, text=True)
  assert done.returncode == 0, done.stderr


def test_the_script_never_assigns_html_nor_calls_the_network() -> None:
  js = (h._ASSETS / "report.js").read_text(encoding="utf-8")
  assert "innerHTML" not in js
  assert "outerHTML" not in js
  assert "insertAdjacentHTML" not in js
  assert "document.write" not in js
  assert "eval(" not in js
  assert (
    re.search(r"\b(?:fetch|XMLHttpRequest|WebSocket|sendBeacon|import)\b", js) is None
  )


def test_the_css_only_uses_variables_it_defines_for_light_and_dark() -> None:
  css = (h._ASSETS / "report.css").read_text(encoding="utf-8")
  defined = set(re.findall(r"(--[\w-]+)\s*:", css))
  used = set(re.findall(r"var\((--[\w-]+)\)", css))
  assert used <= defined, used - defined
  root = css.split("@media (prefers-color-scheme: dark)")
  dark = root[1].split("}")[0]
  light = root[0]
  for name in re.findall(
    r"var\((--[\w-]+)\)", css[css.index("chunk links, explorer") :]
  ):
    assert name in light
    if name != "--mono":
      assert name in dark, f"{name} has no dark value"
  for selector in (".chunk-card", ".badge", ".pager", ".explorer-bar", ".rawdoc"):
    assert selector in css
  assert ".cell.expected" not in css


def test_the_stub_rejects_html_assignment(run_js: Run, stub: Path) -> None:
  """A guard on the harness itself: if the stub did not throw, the check is empty."""
  assert NODE
  with pytest.raises(AssertionError, match="innerHTML must never be used"):
    run_js("const n = new t.Node('div'); n.innerHTML = '<b>'; return 1;")


# ------------------------------------------------------------------ the explorer


def test_explorer_lists_all_chunks_with_the_count_and_one_page(run_js: Run) -> None:
  out = run_js(
    """
    const c = t.controls();
    return { count: c.count.textContent, ids: t.cardIds(c.list),
      docs: t.optionValues(c.docSelect), docLabels: t.optionLabels(c.docSelect),
      qs: t.optionValues(c.questionSelect), qLabels: t.optionLabels(c.questionSelect),
      pager: t.pager(c.pagers[0]).label, usedShown: c.usedLabel.style.display };
    """,
    points=[point("c1", "b.md", 0), point("c2", "a.md", 1), point("c3", "a.md", 0)],
    usage={"q01": usage_entry(), "q02": usage_entry()},
    hash_="#chunks",
  )
  assert out["count"] == "3 de 3 chunks"
  assert out["ids"] == ["c1", "c2", "c3"]  # record order
  assert out["docs"] == ["", "a.md", "b.md"]
  assert out["docLabels"][0] == "todos"
  assert out["qs"] == ["", "q01", "q02"]
  assert out["qLabels"][0] == "ninguna"
  assert out["pager"] == "página 1 de 1"
  assert out["usedShown"] == "none"  # no question marked: no "solo usados" filter


def test_card_shows_the_source_index_range_heading_size_and_full_text(
  run_js: Run,
) -> None:
  long_text = "línea uno\nlínea dos " + "x" * 3000
  out = run_js(
    """
    const card = t.cards(t.controls().list)[0];
    const pre = card.all('pre')[0];
    return { head: card.all('.cc-head')[0].textContent, text: pre.textContent,
      preClass: pre.className, path: card.all('.cc-path')[0].textContent,
      id: card.all('.cc-id')[0].textContent,
      link: card.all('a')[0].getAttribute('href'), cls: card.className };
    """,
    points=[
      point(
        "c-1",
        "dir/é.md",
        7,
        long_text,
        ["Parte 1", "Capítulo 2"],
        char_start=100,
        char_end=250,
        chunk_size=150,
      )
    ],
    usage={"q01": usage_entry()},
    hash_="#chunks",
  )
  assert "dir/é.md" in out["head"]
  assert "#7" in out["head"]
  assert "100–250" in out["head"]
  assert "150 car." in out["head"]
  assert out["text"] == long_text  # whole text, no cut
  assert out["preClass"] == "text"
  assert out["path"] == "Parte 1 › Capítulo 2"
  assert out["id"] == "c-1"
  assert out["link"] == "#chunks?doc=dir%2F%C3%A9.md&chunk=c-1"
  assert out["cls"] == "chunk-card"


def test_card_payload_and_vector_are_filled_when_the_details_open(run_js: Run) -> None:
  out = run_js(
    """
    const card = t.cards(t.controls().list)[0];
    const details = card.all('details')[0];
    const before = details.children.length;
    t.open(details);
    const json = details.all('pre')[0].textContent;
    t.open(details);  // a second opening must not add a second copy
    return { before, after: details.children.length, json,
      kv: details.all('.kv')[0].textContent };
    """,
    points=[
      {
        "id": "v1",
        "payload": {"source": "a.md", "text": "t", "nested": {"k": [1, 2]}},
        "vector": {
          "dimension": 12,
          "norm": 1.23456789,
          "head": [0.123456789, -2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
      }
    ],
    usage={"q01": usage_entry()},
    hash_="#chunks",
  )
  assert out["before"] == 1  # only the <summary> until it is opened
  assert out["after"] == 3  # summary, payload, vector - and still 3 after reopening
  assert json.loads(out["json"]) == {
    "source": "a.md",
    "text": "t",
    "nested": {"k": [1, 2]},
  }
  assert out["json"] == json.dumps(
    {"source": "a.md", "text": "t", "nested": {"k": [1, 2]}},
    indent=2,
    ensure_ascii=False,
  )
  kv = out["kv"]
  assert "Dimensión" in kv and "12" in kv
  assert "Norma" in kv and "1.23457" in kv
  assert "0.12346" in kv and "-2" in kv
  assert kv.endswith(", …]")  # only the first eight values, then an ellipsis
  assert "9" not in kv.split("Primeros valores")[1]


def test_card_of_a_point_without_vector_or_odd_payload_does_not_break(
  run_js: Run,
) -> None:
  out = run_js(
    """
    const c = t.controls();
    const cards = t.cards(c.list);
    cards.forEach((card) => t.open(card.all('details')[0]));
    return { n: cards.length, count: c.count.textContent,
      heads: cards.map((card) => card.all('.cc-head')[0].textContent),
      kv: cards[0].all('details')[0].all('.kv')[0].textContent,
      texts: cards.map((card) => card.all('pre')[0].textContent) };
    """,
    points=[
      {"id": "n1", "payload": {"source": "a.md", "text": "x"}, "vector": None},
      {"id": "n2", "payload": None, "vector": None},
      {
        "id": "n3",
        "payload": {"source": 5, "chunk_index": "3", "text": 7, "heading_path": "no"},
        "vector": None,
      },
      {
        "id": "n4",
        "payload": {"text": "sin fuente", "chunk_index": 2},
        "vector": {"dimension": 0, "norm": 0, "head": []},
      },
    ],
    usage={"q01": usage_entry()},
    hash_="#chunks",
  )
  assert out["n"] == 4
  assert out["count"] == "4 de 4 chunks"
  assert "sin vector" in out["kv"]
  assert "(sin fuente)" in out["heads"][1]
  assert out["texts"] == ["x", "", "", "sin fuente"]
  assert "1 car." in out["heads"][0]  # the size falls back to the text length


# ---- paging


@pytest.mark.parametrize(
  ("total", "pages", "last"),
  [
    (0, 1, 0),
    (1, 1, 1),
    (49, 1, 49),
    (50, 1, 50),
    (51, 2, 1),
    (100, 2, 50),
    (101, 3, 1),
    (120, 3, 20),
  ],
)
def test_paging_at_the_boundaries_of_fifty(
  run_js: Run, total: int, pages: int, last: int
) -> None:
  out = run_js(
    """
    const c = t.controls();
    const seen = [];
    let guard = 0;
    for (;;) {
      const top = t.pager(c.pagers[0]);
      const bottom = t.pager(c.pagers[1]);
      seen.push({ n: t.cards(c.list).length, label: top.label, bottom: bottom.label,
        prevOff: top.previous.disabled, nextOff: top.next.disabled,
        bprevOff: bottom.previous.disabled, bnextOff: bottom.next.disabled,
        first: t.cardIds(c.list)[0] || null, count: c.count.textContent });
      if (top.next.disabled || guard++ > 10) break;
      t.click(top.next);
    }
    return seen;
    """,
    points=synthetic(total),
    usage={"q01": usage_entry()},
    hash_="#chunks",
  )
  assert len(out) == pages
  assert [page["n"] for page in out[:-1]] == [50] * (pages - 1)
  assert out[-1]["n"] == last
  for number, page in enumerate(out, 1):
    assert page["label"] == page["bottom"] == f"página {number} de {pages}"
    assert page["prevOff"] is (number == 1)
    assert page["bprevOff"] is (number == 1)
    assert page["nextOff"] is (number == pages)
    assert page["bnextOff"] is (number == pages)
    assert page["count"] == f"{total} de {total} chunks"
  if total:
    assert [page["first"] for page in out] == [f"p{50 * i:03d}" for i in range(pages)]


def test_paging_walks_forward_and_back_over_a_record_of_120_points(run_js: Run) -> None:
  out = run_js(
    """
    const c = t.controls();
    const log = [];
    const snap = () => { const ids = t.cardIds(c.list); log.push([ids[0], ids[ids.length - 1], ids.length]); };
    snap();
    t.click(t.pager(c.pagers[1]).next); snap();      // bottom pager works too
    t.click(t.pager(c.pagers[0]).next); snap();
    t.click(t.pager(c.pagers[0]).previous); snap();
    t.click(t.pager(c.pagers[1]).previous); snap();
    return log;
    """,
    points=synthetic(120),
    usage={"q01": usage_entry()},
    hash_="#chunks",
  )
  assert out == [
    ["p000", "p049", 50],
    ["p050", "p099", 50],
    ["p100", "p119", 20],
    ["p050", "p099", 50],
    ["p000", "p049", 50],
  ]


def test_changing_a_filter_goes_back_to_the_first_page_and_clamps_the_page(
  run_js: Run,
) -> None:
  out = run_js(
    """
    const c = t.controls();
    t.click(t.pager(c.pagers[0]).next);
    t.click(t.pager(c.pagers[0]).next);
    const third = t.pager(c.pagers[0]).label;
    t.type(c.search, 'chunk number 7');   // p007 and p070..p079: 11 chunks
    const filtered = { label: t.pager(c.pagers[0]).label, count: c.count.textContent,
      n: t.cards(c.list).length };
    t.type(c.search, '');
    const cleared = t.pager(c.pagers[0]).label;
    t.type(c.search, 'no such text anywhere');
    const none = { count: c.count.textContent, n: t.cards(c.list).length,
      note: c.list.textContent, label: t.pager(c.pagers[0]).label,
      off: [t.pager(c.pagers[0]).previous.disabled, t.pager(c.pagers[0]).next.disabled] };
    return { third, filtered, cleared, none };
    """,
    points=synthetic(120),
    usage={"q01": usage_entry()},
    hash_="#chunks",
  )
  assert out["third"] == "página 3 de 3"
  assert out["filtered"] == {
    "label": "página 1 de 1",
    "count": "11 de 120 chunks",
    "n": 11,
  }
  assert out["cleared"] == "página 1 de 3"
  assert out["none"]["count"] == "0 de 120 chunks"
  assert out["none"]["n"] == 0
  assert out["none"]["note"] == "Ningún chunk coincide."
  assert out["none"]["label"] == "página 1 de 1"
  assert out["none"]["off"] == [True, True]


# ---- filters


FILTER_POINTS = [
  point(
    "11111111-aaaa",
    "parte-1/capitulo-01.md",
    0,
    "El Hidalgo vivía en La Mancha",
    ["Parte 1", "Capítulo 1"],
  ),
  point(
    "22222222-bbbb",
    "parte-1/capitulo-02.md",
    0,
    "Sancho Panza",
    ["Parte 1", "Capítulo 2"],
  ),
  point("33333333-cccc", "otro.md", 0, "ÁRBOL de ñandú", ["Épica"]),
  point("44444444-dddd", "otro.md", 1, "sin nada", None),
]


def filtered_ids(run_js: Run, steps: str, **kw: Any) -> Any:
  return run_js(
    "const c = t.controls(); " + steps,
    points=FILTER_POINTS,
    usage={"q01": usage_entry(), "q02": usage_entry()},
    hash_="#chunks",
    **kw,
  )


@pytest.mark.parametrize(
  ("needle", "expected"),
  [
    ("sancho", ["22222222-bbbb"]),  # case ignored, text
    ("SANCHO PANZA", ["22222222-bbbb"]),
    (
      "capitulo",
      ["11111111-aaaa", "22222222-bbbb"],
    ),  # accents ignored: heading "Capítulo"
    ("capítulo", ["11111111-aaaa", "22222222-bbbb"]),
    ("CAPÍTULO 2", ["22222222-bbbb"]),
    ("arbol", ["33333333-cccc"]),  # text "ÁRBOL"
    ("ÁRBOL", ["33333333-cccc"]),
    ("nandu", ["33333333-cccc"]),
    ("epica", ["33333333-cccc"]),  # heading "Épica"
    ("33333333-cccc", ["33333333-cccc"]),  # chunk_id
    ("44444444", ["44444444-dddd"]),
    ("otro.md", ["33333333-cccc", "44444444-dddd"]),  # source
    ("parte-1/", ["11111111-aaaa", "22222222-bbbb"]),
    ("parte 1 › capítulo 1", ["11111111-aaaa"]),  # the heading path joined by ›
    ("   sancho   ", ["22222222-bbbb"]),  # surrounding blanks are trimmed
    ("", ["11111111-aaaa", "22222222-bbbb", "33333333-cccc", "44444444-dddd"]),
    ("   ", ["11111111-aaaa", "22222222-bbbb", "33333333-cccc", "44444444-dddd"]),
    ("zzz", []),
    ("[", []),  # regex-looking input is plain text
    (".*", []),
    ("(", []),
  ],
)
def test_text_filter(run_js: Run, needle: str, expected: list[str]) -> None:
  out = filtered_ids(
    run_js,
    "t.type(c.search, "
    + json.dumps(needle)
    + "); return { ids: t.cardIds(c.list), count: c.count.textContent };",
  )
  assert out["ids"] == expected
  assert out["count"] == f"{len(expected)} de 4 chunks"


def test_text_filter_ignores_decomposed_accents_in_data_and_in_the_box(
  run_js: Run,
) -> None:
  out = run_js(
    """
    const c = t.controls();
    t.type(c.search, 'cafe\\u0301');          // typed decomposed
    const a = t.cardIds(c.list);
    t.type(c.search, 'CAFÉ');                  // typed composed, uppercase
    const b = t.cardIds(c.list);
    t.type(c.search, 'resume');
    return [a, b, t.cardIds(c.list)];
    """,
    points=[point("x1", "a.md", 0, "un café"), point("x2", "a.md", 1, "résumé")],
    usage={"q01": usage_entry()},
    hash_="#chunks",
  )
  assert out == [["x1"], ["x1"], ["x2"]]


def test_document_select_and_text_filter_combine(run_js: Run) -> None:
  out = filtered_ids(
    run_js,
    """
    t.setValue(c.docSelect, 'otro.md');
    const doc = t.cardIds(c.list);
    t.type(c.search, 'sin');
    const both = t.cardIds(c.list);
    t.setValue(c.docSelect, 'parte-1/capitulo-01.md');
    const mismatch = t.cardIds(c.list);
    t.setValue(c.docSelect, '');
    const all = t.cardIds(c.list);
    return { doc, both, mismatch, all, count: c.count.textContent };
    """,
  )
  assert out["doc"] == ["33333333-cccc", "44444444-dddd"]
  assert out["both"] == ["44444444-dddd"]
  assert out["mismatch"] == []
  assert out["all"] == ["44444444-dddd"]  # the text box "sin" is still applied
  assert out["count"] == "1 de 4 chunks"  # the text filter is still on


# ---- marking a question


STATE_POINTS = [point(f"s{n}", "a.md", n, f"texto {n}") for n in range(7)]
STATE_USAGE = {
  "q01": usage_entry(
    retrieved=["s0", "s1", "s2"],
    added=["s3", "s2"],
    context=["s0", "s3"],
    cited=["s0"],
    dropped=["s1", "ghost"],
  ),
  "q02": usage_entry(retrieved=["s5"], cited=["s5"]),
}


def test_marking_a_question_gives_each_card_its_classes_and_badges(run_js: Run) -> None:
  out = run_js(
    """
    const c = t.controls();
    const snap = () => Object.fromEntries(t.cards(c.list).map((card) =>
      [card.getAttribute('data-chunk'), { cls: card.className.split(' ').filter((x) => x !== 'chunk-card'),
        badges: t.texts(card.all('.badge')), badgeCls: card.all('.badge').map((b) => b.className) }]));
    const none = snap();
    t.setValue(c.questionSelect, 'q01');
    const q1 = snap();
    const shown = c.usedLabel.style.display;
    t.setValue(c.questionSelect, 'q02');
    const q2 = snap();
    t.setValue(c.questionSelect, '');
    return { none, q1, q2, shown, back: snap(), hidden: c.usedLabel.style.display };
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
    hash_="#chunks",
  )
  for card in out["none"].values():
    assert card == {"cls": [], "badges": [], "badgeCls": []}
  q1 = out["q1"]
  assert q1["s0"]["badges"] == ["recuperado", "en contexto", "citado"]
  assert q1["s0"]["cls"] == ["st-retrieved", "st-context", "st-cited"]
  assert q1["s1"]["badges"] == ["recuperado", "descartado"]
  assert q1["s2"]["badges"] == ["recuperado", "añadido"]
  assert q1["s3"]["badges"] == ["añadido", "en contexto"]
  assert q1["s4"]["badges"] == []
  assert q1["s0"]["badgeCls"] == ["badge retrieved", "badge context", "badge cited"]
  assert out["shown"] == ""
  # another question gives other states; the first one leaves nothing behind
  assert out["q2"]["s5"]["badges"] == ["recuperado", "citado"]
  assert out["q2"]["s0"]["badges"] == []
  assert out["back"] == out["none"]
  assert out["hidden"] == "none"


def test_solo_usados_follows_the_marked_question_and_turns_off_with_ninguna(
  run_js: Run,
) -> None:
  out = run_js(
    """
    const c = t.controls();
    t.setValue(c.questionSelect, 'q01');
    t.setChecked(c.used, true);
    const q1 = t.cardIds(c.list);
    t.setValue(c.questionSelect, 'q02');         // the box stays ticked
    const q2 = { ids: t.cardIds(c.list), checked: c.used.checked };
    t.setValue(c.questionSelect, '');            // "ninguna" turns the filter off
    return { q1, q2, afterNone: t.cardIds(c.list).length, checked: c.used.checked,
      hidden: c.usedLabel.style.display };
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
    hash_="#chunks",
  )
  assert out["q1"] == ["s0", "s1", "s2", "s3"]
  assert out["q2"] == {"ids": ["s5"], "checked": True}
  assert out["afterNone"] == 7
  assert out["checked"] is False
  assert out["hidden"] == "none"


def test_solo_usados_after_marking_and_ticking(run_js: Run) -> None:
  out = run_js(
    """
    const c = t.controls();
    t.setValue(c.questionSelect, 'q01');
    const all = t.cardIds(c.list).length;
    t.setChecked(c.used, true);
    const used = t.cardIds(c.list);
    const count = c.count.textContent;
    t.type(c.search, 'texto 3');
    const narrowed = t.cardIds(c.list);
    t.setValue(c.docSelect, 'zzz.md');
    return { all, used, count, narrowed, none: t.cardIds(c.list) };
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
    hash_="#chunks",
  )
  assert out["all"] == 7
  # "ghost" is in the usage but has no point: it simply has no card
  assert out["used"] == ["s0", "s1", "s2", "s3"]
  assert out["count"] == "4 de 7 chunks"
  assert out["narrowed"] == ["s3"]
  assert out["none"] == []


# ------------------------------------------------------------------ deep links


def test_a_deep_link_at_load_opens_the_tab_sets_the_filters_and_scrolls(
  run_js: Run,
) -> None:
  out = run_js(
    """
    const c = t.controls();
    const target = t.cards(c.list).find((card) => card.classList.contains('target'));
    return { active: t.tabs().filter((b) => b.classList.contains('active')).map((b) => b.dataset.key),
      panels: Object.keys(t.panels).filter((k) => t.panels[k].classList.contains('active')),
      doc: c.docSelect.value, q: c.questionSelect.value, used: c.used.checked,
      ids: t.cardIds(c.list), count: c.count.textContent,
      target: target ? target.getAttribute('data-chunk') : null, scrolled: target ? target.scrolled : 0,
      targets: t.cards(c.list).filter((card) => card.classList.contains('target')).length,
      hash: t.location.hash, saved: t.store['hermes-report-tab:T'],
      badges: t.texts(target.all('.badge')) };
    """,
    points=STATE_POINTS + [point("other", "b.md", 0)],
    usage=STATE_USAGE,
    hash_="#chunks?doc=a.md&q=q01&chunk=s2",
  )
  assert out["active"] == ["chunks"]
  assert out["panels"] == ["chunks"]
  assert out["doc"] == "a.md"
  assert out["q"] == "q01"
  assert out["used"] is False  # no used=1: the question only marks
  assert "other" not in out["ids"]
  assert out["count"] == "7 de 8 chunks"
  assert out["target"] == "s2"
  assert out["targets"] == 1
  assert out["scrolled"] == 1
  assert out["badges"] == ["recuperado", "añadido"]
  assert out["hash"] == "#chunks?doc=a.md&q=q01&chunk=s2"
  assert out["saved"] == "chunks"


def test_deep_link_with_used_ticks_the_solo_usados_box(run_js: Run) -> None:
  out = run_js(
    "const c = t.controls(); return { used: c.used.checked, ids: t.cardIds(c.list), shown: c.usedLabel.style.display };",
    points=STATE_POINTS,
    usage=STATE_USAGE,
    hash_="#chunks?q=q01&used=1",
  )
  assert out == {"used": True, "ids": ["s0", "s1", "s2", "s3"], "shown": ""}


def test_deep_link_used_without_a_known_question_does_not_filter(run_js: Run) -> None:
  out = run_js(
    "const c = t.controls(); return { used: c.used.checked, n: t.cards(c.list).length, q: c.questionSelect.value };",
    points=STATE_POINTS,
    usage=STATE_USAGE,
    hash_="#chunks?q=nope&used=1",
  )
  assert out == {"used": False, "n": 7, "q": ""}


def test_deep_link_to_a_chunk_on_a_later_page_opens_that_page(run_js: Run) -> None:
  out = run_js(
    """
    const c = t.controls();
    const target = t.cards(c.list).find((card) => card.classList.contains('target'));
    return { label: t.pager(c.pagers[0]).label, ids: t.cardIds(c.list).length,
      target: target.getAttribute('data-chunk'), scrolled: target.scrolled,
      first: t.cardIds(c.list)[0] };
    """,
    points=synthetic(120, per_source=200),
    usage={"q01": usage_entry()},
    hash_="#chunks?chunk=p117",
  )
  assert out == {
    "label": "página 3 de 3",
    "ids": 20,
    "target": "p117",
    "scrolled": 1,
    "first": "p100",
  }


def test_deep_link_target_on_the_exact_page_boundaries(run_js: Run) -> None:
  for chunk, page in (("p049", 1), ("p050", 2), ("p099", 2), ("p100", 3)):
    out = run_js(
      "const c = t.controls(); return t.pager(c.pagers[0]).label;",
      points=synthetic(120, per_source=200),
      usage={"q01": usage_entry()},
      hash_=f"#chunks?chunk={chunk}",
    )
    assert out == f"página {page} de 3", chunk


def test_deep_link_page_accounts_for_the_other_filters(run_js: Run) -> None:
  # doc1 holds p040..p079 (40 points); the chunk is its 31st: still the first page
  out = run_js(
    "const c = t.controls(); return { label: t.pager(c.pagers[0]).label, n: t.cards(c.list).length, count: c.count.textContent };",
    points=synthetic(120),
    usage={"q01": usage_entry()},
    hash_="#chunks?doc=doc1.md&chunk=p070",
  )
  assert out == {"label": "página 1 de 1", "n": 40, "count": "40 de 120 chunks"}


@pytest.mark.parametrize(
  "hash_",
  [
    "#chunks?",
    "#chunks?&&&",
    "#chunks?doc",
    "#chunks?=x",
    "#chunks?doc=%E0%A4%A",  # malformed percent-encoding
    "#chunks?doc=%",
    "#chunks?chunk=nope",
    "#chunks?doc=zzz.md&q=zzz&chunk=zzz",
    "#chunks?doc=a.md&doc=b.md",
    "#chunks?q=q01&q=",
    "#chunks??q=q01",
    "#chunks?__proto__=1&constructor=2",
  ],
)
def test_malformed_deep_links_do_not_throw_and_show_everything(
  run_js: Run, hash_: str
) -> None:
  out = run_js(
    """
    const c = t.controls();
    return { active: t.tabs().filter((b) => b.classList.contains('active')).map((b) => b.dataset.key),
      n: t.cards(c.list).length, count: c.count.textContent };
    """,
    points=STATE_POINTS + [point("other", "b.md", 0)],
    usage=STATE_USAGE,
    hash_=hash_,
  )
  assert out["active"] == ["chunks"]
  assert out["count"].endswith("de 8 chunks")


def test_deep_link_values_are_percent_decoded(run_js: Run) -> None:
  out = run_js(
    "const c = t.controls(); return { doc: c.docSelect.value, ids: t.cardIds(c.list), target: t.cards(c.list)[0].classList.contains('target') };",
    points=[point("id with space&amp", "dir/é & x.md", 0), point("z", "z.md", 0)],
    usage={"q 1": usage_entry()},
    hash_="#chunks?doc=dir%2F%C3%A9%20%26%20x.md&q=q%201&chunk=id%20with%20space%26amp",
  )
  assert out == {"doc": "dir/é & x.md", "ids": ["id with space&amp"], "target": True}


def test_a_deep_link_round_trips_through_the_python_helper(run_js: Run) -> None:
  source = 'sub/ñ ü "q" & #1?.md'
  chunk = "ch#1 &=? ü"
  out = run_js(
    "const c = t.controls(); return { doc: c.docSelect.value, q: c.questionSelect.value, target: t.cards(c.list).filter((x) => x.classList.contains('target')).map((x) => x.getAttribute('data-chunk')) };",
    points=[point(chunk, source, 0), point("z", "z.md", 0)],
    usage={"q&1=2": usage_entry()},
    hash_=h.deep_link(doc=source, question="q&1=2", chunk=chunk),
  )
  assert out == {"doc": source, "q": "q&1=2", "target": [chunk]}


def test_clicking_a_deep_link_anchor_routes_without_following_it(run_js: Run) -> None:
  out = run_js(
    """
    const anchor = new t.Node('a');
    anchor.setAttribute('href', '#chunks?doc=a.md&q=q02&chunk=s5&used=1');
    const inner = new t.Node('span');
    anchor.appendChild(inner);
    t.panels['q-q01'].appendChild(anchor);
    t.click(t.tabs()[0]);                 // start on the first tab
    const ev = t.click(inner);            // click on the span inside the anchor
    const c = t.controls();
    return { prevented: ev.defaultPrevented, active: t.tabs().filter((b) => b.classList.contains('active')).map((b) => b.dataset.key),
      q: c.questionSelect.value, used: c.used.checked, ids: t.cardIds(c.list), hash: t.location.hash };
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
    keys=["resumen", "corpus", "chunks", "q-q01", "q-q02"],
  )
  assert out["prevented"] is True
  assert out["active"] == ["chunks"]
  assert out["q"] == "q02" and out["used"] is True
  assert out["ids"] == ["s5"]
  assert out["hash"] == "#chunks?doc=a.md&q=q02&chunk=s5&used=1"


def test_clicking_an_ordinary_anchor_or_text_is_left_alone(run_js: Run) -> None:
  out = run_js(
    """
    const other = new t.Node('a');
    other.setAttribute('href', '#elsewhere');
    t.body.appendChild(other);
    const textNode = { };                 // a click target without .closest
    const a = t.click(other);
    const b = t.fire(textNode, 'click', { target: textNode });
    const c = t.fire(t.body, 'click', { target: null });
    return [a.defaultPrevented, b.defaultPrevented, c.defaultPrevented];
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
  )
  assert out == [False, False, False]


def test_hashchange_opens_the_explorer_with_the_new_filters(run_js: Run) -> None:
  out = run_js(
    """
    t.hashchange('#chunks?q=q02&chunk=s5');
    const c = t.controls();
    const first = { q: c.questionSelect.value, target: t.cards(c.list).filter((x) => x.classList.contains('target')).length };
    t.hashchange('#q-q01');
    const tab = t.tabs().filter((b) => b.classList.contains('active')).map((b) => b.dataset.key);
    t.hashchange('#nonsense');            // names nothing: stays where it is
    return { first, tab, still: t.tabs().filter((b) => b.classList.contains('active')).map((b) => b.dataset.key) };
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
    keys=["resumen", "corpus", "chunks", "q-q01", "q-q02"],
  )
  assert out["first"] == {"q": "q02", "target": 1}
  assert out["tab"] == ["q-q01"]
  assert out["still"] == ["q-q01"]


def test_a_plain_click_on_the_chunks_tab_keeps_the_filters(run_js: Run) -> None:
  out = run_js(
    """
    const c = t.controls();
    t.type(c.search, 'texto 4');
    t.click(t.tabs()[0]);
    t.click(t.tabs()[2]);
    return { ids: t.cardIds(t.controls().list), search: t.controls().search.value,
      active: t.tabs().filter((b) => b.classList.contains('active')).map((b) => b.dataset.key) };
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
    hash_="#chunks",
  )
  assert out == {"ids": ["s4"], "search": "texto 4", "active": ["chunks"]}


def test_the_explorer_is_built_once(run_js: Run) -> None:
  out = run_js(
    """
    const root = t.explorer();
    const before = root.children.length;
    t.hashchange('#chunks?q=q01');
    t.hashchange('#chunks');
    t.click(t.tabs()[2]);
    return { before, after: t.explorer().children.length, selects: root.all('select').length };
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
    hash_="#chunks",
  )
  assert out == {"before": 5, "after": 5, "selects": 2}


# ------------------------------------------------------------------ raw documents


RAW_POINTS = [
  point("a2", "a.md", 2, "tercero"),
  point("a0", "a.md", 0, "primero"),
  point("b0", "b.md", 0, "otro documento"),
  point("a1", "a.md", 1, "segundo"),
  point("an", "a.md", None, "sin índice"),
]
RAW_USAGE = {
  "q01": usage_entry(retrieved=["a0", "a1"], cited=["a1"], dropped=["a0"]),
  "q02": usage_entry(retrieved=["a2"], added=["b0"], context=["a2"]),
}


def test_a_raw_document_is_empty_until_it_opens_then_shows_its_chunks_in_order(
  run_js: Run,
) -> None:
  out = run_js(
    """
    const doc = t.raw.q01['a.md'];
    const body = doc.all('.rawbody')[0];
    const before = body.children.length;
    t.open(doc);
    const cards = t.cards(body);
    const result = { before, ids: t.cardIds(body),
      states: cards.map((card) => t.texts(card.all('.badge'))),
      texts: cards.map((card) => card.all('pre')[0].textContent),
      otherClosed: t.raw.q01['b.md'].all('.rawbody')[0].children.length,
      otherQuestionClosed: t.raw.q02['a.md'].all('.rawbody')[0].children.length };
    t.open(doc);              // opening again must not draw the cards twice
    t.open(doc);
    result.again = t.cards(body).length;
    return result;
    """,
    points=RAW_POINTS,
    usage=RAW_USAGE,
    raw={"q01": ["a.md", "b.md"], "q02": ["a.md"]},
    keys=["resumen", "chunks", "q-q01", "q-q02"],
  )
  assert out["before"] == 0
  # chunk_index order; a chunk with no index goes last
  assert out["ids"] == ["a0", "a1", "a2", "an"]
  assert out["texts"] == ["primero", "segundo", "tercero", "sin índice"]
  # the states are those of THIS question
  assert out["states"] == [
    ["recuperado", "descartado"],
    ["recuperado", "citado"],
    [],
    [],
  ]
  assert out["otherClosed"] == 0
  assert out["otherQuestionClosed"] == 0
  assert out["again"] == 4


def test_raw_documents_of_two_questions_show_each_their_own_states(run_js: Run) -> None:
  out = run_js(
    """
    t.open(t.raw.q01['a.md']); t.open(t.raw.q02['a.md']);
    const states = (q) => Object.fromEntries(t.cards(t.raw[q]['a.md']).map((card) =>
      [card.getAttribute('data-chunk'), t.texts(card.all('.badge'))]));
    return { q1: states('q01'), q2: states('q02') };
    """,
    points=RAW_POINTS,
    usage=RAW_USAGE,
    raw={"q01": ["a.md"], "q02": ["a.md"]},
    keys=["resumen", "chunks", "q-q01", "q-q02"],
  )
  assert out["q1"]["a2"] == []
  assert out["q2"]["a2"] == ["recuperado", "en contexto"]
  assert out["q2"]["a0"] == []


def test_a_raw_document_the_collection_does_not_have_says_so(run_js: Run) -> None:
  out = run_js(
    """
    const doc = t.raw.q01['ghost.md'];
    t.open(doc);
    const body = doc.all('.rawbody')[0];
    return { text: body.textContent, cards: t.cards(body).length };
    """,
    points=RAW_POINTS,
    usage=RAW_USAGE,
    raw={"q01": ["ghost.md"]},
    keys=["resumen", "chunks", "q-q01"],
  )
  assert out == {"text": "Sin chunks en la colección.", "cards": 0}


def test_a_closing_toggle_or_an_unrelated_details_does_nothing(run_js: Run) -> None:
  out = run_js(
    """
    const doc = t.raw.q01['a.md'];
    const body = doc.all('.rawbody')[0];
    doc.open = false;
    t.fire(doc, 'toggle');                // fired when it closes: nothing to draw
    const closed = body.children.length;
    const stray = new t.Node('details');
    t.body.appendChild(stray);
    stray.open = true;
    t.fire(stray, 'toggle');
    t.fire({ }, 'toggle', { target: null });
    return { closed };
    """,
    points=RAW_POINTS,
    usage=RAW_USAGE,
    raw={"q01": ["a.md"]},
    keys=["resumen", "chunks", "q-q01"],
  )
  assert out == {"closed": 0}


def test_raw_documents_work_without_ever_visiting_the_chunks_tab(run_js: Run) -> None:
  out = run_js(
    """
    t.open(t.raw.q01['a.md']);
    return { ids: t.cardIds(t.raw.q01['a.md']), explorerChildren: t.explorer().children.length };
    """,
    points=RAW_POINTS,
    usage=RAW_USAGE,
    raw={"q01": ["a.md"]},
    keys=["resumen", "chunks", "q-q01"],
    hash_="#q-q01",
  )
  assert out["ids"] == ["a0", "a1", "a2", "an"]
  assert out["explorerChildren"] == 0  # the explorer is built only when it is asked for


def test_raw_chunk_cards_have_a_permalink_into_the_explorer(run_js: Run) -> None:
  out = run_js(
    """
    t.open(t.raw.q02['a.md']);
    const card = t.cards(t.raw.q02['a.md'])[0];
    return card.all('a')[0].getAttribute('href');
    """,
    points=RAW_POINTS,
    usage=RAW_USAGE,
    raw={"q02": ["a.md"]},
    keys=["resumen", "chunks", "q-q02"],
  )
  assert out == "#chunks?doc=a.md&q=q02&chunk=a0"


# ------------------------------------------------------------------ robustness of the data


def test_the_real_json_blocks_with_hostile_text_survive_the_round_trip(
  run_js: Run,
) -> None:
  nasty = '</script><!-- <script>alert(1)</script> -->   "q" \\ ]]> <b>x</b>'
  out = run_js(
    """
    const c = t.controls();
    t.open(t.cards(c.list)[0].all('details')[0]);
    const card = t.cards(c.list)[0];
    t.type(c.search, 'alert(1)');
    return { text: card.all('pre')[0].textContent, source: card.all('.cc-source')[0].textContent,
      n: t.cards(c.list).length, payload: card.all('details')[0].all('pre')[0].textContent,
      q: t.optionValues(c.questionSelect) };
    """,
    points=[point(nasty, nasty, 0, nasty, [nasty])],
    usage={nasty: usage_entry(retrieved=[nasty])},
    hash_="#chunks",
  )
  assert out["text"] == nasty
  assert out["source"] == nasty
  assert out["n"] == 1
  assert json.loads(out["payload"])["text"] == nasty
  assert out["q"] == ["", nasty]


def test_usage_and_points_from_a_rendered_page_drive_the_explorer(
  real_run: RunRecord, run_js: Run
) -> None:
  """End to end with the real page: the blocks cut from ``render_run`` output."""
  text = render_run(real_run)
  blocks = re.findall(
    r'<script type="application/json" id="([^"]+)">(.*?)</script>', text, re.S
  )
  assert [name for name, _ in blocks] == ["hr-points", "hr-usage"]
  points = json.loads(blocks[0][1])
  usage = json.loads(blocks[1][1])
  first = next(q for q, entry in usage.items() if entry["cited"])
  out = run_js(
    f"""
    const c = t.controls();
    t.setValue(c.questionSelect, {json.dumps(first)});
    const marked = t.cards(c.list).filter((card) => card.classList.contains('st-cited'));
    t.setChecked(c.used, true);
    return {{ count: c.count.textContent, cited: marked.map((x) => x.getAttribute('data-chunk')),
      used: t.cards(c.list).length }};
    """,
    raw_texts=(blocks[0][1], blocks[1][1]),
    keys=["resumen", "chunks", *[f"q-{q}" for q in usage]],
    hash_="#chunks",
  )
  assert out["count"].endswith(f"de {len(points)} chunks")
  assert sorted(out["cited"]) == sorted(usage[first]["cited"])
  touched = set().union(*(set(v) for v in usage[first].values()))
  assert out["used"] == len(touched & {p["id"] for p in points})


@pytest.fixture(scope="module")
def real_run(tmp_path_factory: pytest.TempPathFactory) -> RunRecord:
  import shutil as sh
  import warnings

  from conftest import DATA

  from hermes_report import runner
  from hermes_report.config import Overrides, load_config

  project = tmp_path_factory.mktemp("jsproj") / "proj"
  sh.copytree(DATA, project)
  config = load_config(project / "run.toml", Overrides(out=project / "out"))
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    return runner.run(config, progress=lambda line: None).record


def test_missing_or_broken_data_gives_an_empty_explorer_not_an_error(
  run_js: Run,
) -> None:
  for texts in ((None, None), ("{not json", "[1,"), ("null", "null"), ("", "")):
    out = run_js(
      "const c = t.controls(); return { count: c.count.textContent, n: t.cards(c.list).length, q: t.optionValues(c.questionSelect) };",
      raw_texts=texts,
      keys=["resumen", "chunks"],
      hash_="#chunks",
    )
    assert out["n"] == 0, texts
    assert out["count"] == "0 de 0 chunks", texts


def test_a_page_without_the_explorer_element_still_routes_tabs(run_js: Run) -> None:
  out = run_js(
    "return t.tabs().filter((b) => b.classList.contains('active')).map((b) => b.dataset.key);",
    points=STATE_POINTS,
    usage=STATE_USAGE,
    explorer=False,
    hash_="#chunks?doc=a.md",
  )
  assert out == ["chunks"]


def test_points_with_duplicate_sources_and_ids_do_not_crash(run_js: Run) -> None:
  out = run_js(
    "const c = t.controls(); t.open(t.raw.q01['a.md']); return { n: t.cards(c.list).length, raw: t.cards(t.raw.q01['a.md']).length };",
    points=[point("d", "a.md", 0), point("d", "a.md", 0)],
    usage={"q01": usage_entry(retrieved=["d"])},
    raw={"q01": ["a.md"]},
    keys=["resumen", "chunks", "q-q01"],
    hash_="#chunks",
  )
  assert out == {"n": 2, "raw": 2}


NAMES = ["constructor", "__proto__", "toString", "hasOwnProperty", "valueOf"]


@pytest.mark.xfail(
  strict=True,
  reason="usageSets/usage are plain objects: `usageSets['constructor']` is Object's "
  "own function, so setsOf() reads undefined and throws (needs hasOwnProperty or Object.create(null))",
)
@pytest.mark.parametrize("name", NAMES)
def test_a_question_id_named_like_an_object_member_is_an_ordinary_id(
  run_js: Run, name: str
) -> None:
  quoted = json.dumps(name)
  out = run_js(
    f"""
    const c = t.controls();
    t.setValue(c.questionSelect, {quoted});
    return {{ q: c.questionSelect.value,
      badges: t.cards(c.list).map((card) => t.texts(card.all('.badge'))) }};
    """,
    points=[point("x1", "a.md", 0), point("x2", "a.md", 1)],
    usage={name: usage_entry(retrieved=["x1"])},
    keys=["resumen", "chunks", f"q-{name}"],
    hash_="#chunks",
  )
  assert out == {"q": name, "badges": [["recuperado"], []]}


@pytest.mark.xfail(
  strict=True,
  reason="bySource is a plain object: a source named like an Object member hits the "
  "inherited property and `.push` is not a function",
)
@pytest.mark.parametrize("name", NAMES)
def test_a_document_named_like_an_object_member_is_an_ordinary_document(
  run_js: Run, name: str
) -> None:
  quoted = json.dumps(name)
  out = run_js(
    f"""
    const c = t.controls();
    t.setValue(c.docSelect, {quoted});
    t.open(t.raw.q01[{quoted}]);
    return {{ n: t.cards(c.list).length, raw: t.cards(t.raw.q01[{quoted}]).length }};
    """,
    points=[point("x1", name, 0), point("x2", "plain.md", 0)],
    usage={"q01": usage_entry(retrieved=["x1"])},
    raw={"q01": [name]},
    keys=["resumen", "chunks", "q-q01"],
    hash_="#chunks",
  )
  assert out == {"n": 1, "raw": 1}


def test_chunk_ids_named_like_object_members_do_not_get_states(run_js: Run) -> None:
  out = run_js(
    """
    const c = t.controls();
    t.setValue(c.questionSelect, 'q01');
    return t.cards(c.list).map((card) => t.texts(card.all('.badge')));
    """,
    points=[
      point("constructor", "a.md", 0),
      point("__proto__", "a.md", 1),
      point("real", "a.md", 2),
    ],
    usage={"q01": usage_entry(retrieved=["real"])},
    hash_="#chunks",
  )
  assert out == [[], [], ["recuperado"]]


# ------------------------------------------------------------------ the earlier tab behaviour, with the data blocks in the page


def test_tab_routing_still_works_next_to_the_explorer(run_js: Run) -> None:
  out = run_js(
    """
    const active = () => t.tabs().filter((b) => b.classList.contains('active')).map((b) => b.dataset.key);
    const first = active();
    t.click(t.tabs()[3]);
    return { first, afterClick: active(), saved: t.store['hermes-report-tab:T'], hash: t.location.hash };
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
    keys=["resumen", "corpus", "chunks", "q-q01"],
  )
  assert out == {
    "first": ["resumen"],
    "afterClick": ["q-q01"],
    "saved": "q-q01",
    "hash": "#q-q01",
  }


def test_data_goto_anchor_still_switches_tab_and_scrolls_to_the_top(
  run_js: Run,
) -> None:
  out = run_js(
    """
    const a = new t.Node('a');
    a.setAttribute('href', '#q-q01'); a.setAttribute('data-goto', 'q-q01');
    t.body.appendChild(a);
    const ev = t.click(a);
    return { prevented: ev.defaultPrevented, scroll: t.scrollTo,
      active: t.tabs().filter((b) => b.classList.contains('active')).map((b) => b.dataset.key) };
    """,
    points=STATE_POINTS,
    usage=STATE_USAGE,
    keys=["resumen", "chunks", "q-q01"],
  )
  assert out == {"prevented": True, "scroll": [[0, 0]], "active": ["q-q01"]}


def test_a_copy_of_the_record_with_no_questions_still_builds_the_explorer(
  run_js: Run,
) -> None:
  out = run_js(
    "const c = t.controls(); return { count: c.count.textContent, q: t.optionLabels(c.questionSelect) };",
    points=STATE_POINTS,
    usage={},
    keys=["resumen", "chunks"],
    hash_="#chunks",
  )
  assert out == {"count": "7 de 7 chunks", "q": ["ninguna"]}

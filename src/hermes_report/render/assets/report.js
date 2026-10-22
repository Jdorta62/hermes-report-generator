(function () {
  "use strict";
  var PAGE_SIZE = 50;
  var VECTOR_VALUES = 8;
  // The states a question gives a chunk: key in `hr-usage`, label of the badge.
  var STATES = [
    ["retrieved", "recuperado"],
    ["added", "añadido"],
    ["context", "en contexto"],
    ["cited", "citado"],
    ["dropped", "descartado"]
  ];

  var tabs = document.querySelectorAll(".tab");
  var panels = document.querySelectorAll(".panel");
  var storageKey = "hermes-report-tab:" + document.title;
  var keys = Array.prototype.map.call(tabs, function (t) { return t.dataset.key; });

  // ---- tabs and routing ---------------------------------------------------

  function show(key, hash) {
    tabs.forEach(function (t) { t.classList.toggle("active", t.dataset.key === key); });
    panels.forEach(function (p) { p.classList.toggle("active", p.dataset.key === key); });
    try { localStorage.setItem(storageKey, key); } catch (e) {}
    try { history.replaceState(null, "", hash || "#" + key); } catch (e) {}
  }

  function parseHash(hash) {
    var params = {};
    var mark = hash.indexOf("?");
    if (mark < 0) { return params; }
    hash.slice(mark + 1).split("&").forEach(function (pair) {
      var eq = pair.indexOf("=");
      var name = eq < 0 ? pair : pair.slice(0, eq);
      if (!name) { return; }
      var value = eq < 0 ? "" : pair.slice(eq + 1);
      try { value = decodeURIComponent(value); } catch (e) {}
      params[name] = value;
    });
    return params;
  }

  // Opens what `hash` (without the leading "#") names; false when it names nothing.
  // `chunks?doc=..&q=..&chunk=..` also sets the filters of the explorer.
  function route(hash) {
    var name = hash.split("?")[0];
    if (name === "chunks") {
      show("chunks", "#" + hash);
      var explorer = ensureExplorer();
      if (explorer && hash.indexOf("?") >= 0) { explorer.open(parseHash(hash)); }
      return true;
    }
    if (keys.indexOf(name) >= 0) { show(name); return true; }
    return false;
  }

  tabs.forEach(function (t) {
    t.addEventListener("click", function () { route(t.dataset.key); });
  });
  document.addEventListener("click", function (ev) {
    var target = ev.target;
    if (!target || !target.closest) { return; }
    var goto = target.closest("a[data-goto]");
    if (goto) { ev.preventDefault(); show(goto.dataset.goto); window.scrollTo(0, 0); return; }
    var deep = target.closest("a[href^='#chunks']");
    if (deep) {
      ev.preventDefault();
      var href = deep.getAttribute("href");
      route(href.slice(1));
      // A link to a chunk scrolls to its card; one without a chunk starts at the top.
      if (!parseHash(href).chunk) { window.scrollTo(0, 0); }
    }
  });
  window.addEventListener("hashchange", function () { route(location.hash.slice(1)); });

  // ---- data ---------------------------------------------------------------

  function readJson(id) {
    try {
      var node = document.getElementById(id);
      return node ? JSON.parse(node.textContent) : null;
    } catch (e) { return null; }
  }

  var points = readJson("hr-points") || [];
  var usage = readJson("hr-usage") || {};
  var questionIds = Object.keys(usage);
  // Maps keyed by ids and sources of the record use no prototype, so a key such as
  // "constructor" or "__proto__" is an ordinary key.
  var usageSets = Object.create(null);
  var prepared = null;

  function text(value) { return typeof value === "string" ? value : ""; }
  function integer(value) { return typeof value === "number" && isFinite(value) ? value : null; }
  // Lower case without accents, so "Capitulo" finds "Capítulo".
  function fold(value) {
    return value.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  }

  // One entry per point, in record order, with what the filters need.
  function prepare() {
    if (prepared) { return prepared; }
    var entries = points.map(function (point, order) {
      var payload = point.payload || {};
      var path = Array.isArray(payload.heading_path) ? payload.heading_path.join(" › ") : "";
      return {
        point: point,
        order: order,
        source: text(payload.source),
        index: integer(payload.chunk_index),
        path: path,
        hay: fold([point.id, text(payload.source), path, text(payload.text)].join("\n"))
      };
    });
    var bySource = Object.create(null);
    entries.forEach(function (entry) {
      if (entry.source) { (bySource[entry.source] = bySource[entry.source] || []).push(entry); }
    });
    Object.keys(bySource).forEach(function (source) {
      bySource[source].sort(function (a, b) {
        return (a.index === null ? Infinity : a.index) - (b.index === null ? Infinity : b.index) ||
          a.order - b.order;
      });
    });
    prepared = { entries: entries, bySource: bySource, sources: Object.keys(bySource).sort() };
    return prepared;
  }

  function setsOf(question) {
    if (!usageSets[question]) {
      var sets = Object.create(null);
      var own = Object.prototype.hasOwnProperty.call(usage, question) ? usage[question] : null;
      STATES.forEach(function (state) {
        var ids = (own && own[state[0]]) || [];
        sets[state[0]] = Object.create(null);
        ids.forEach(function (id) { sets[state[0]][id] = true; });
      });
      usageSets[question] = sets;
    }
    return usageSets[question];
  }

  // The state keys that `question` gives to the chunk `id`.
  function statesOf(question, id) {
    if (!question) { return []; }
    var sets = setsOf(question);
    return STATES.map(function (s) { return s[0]; }).filter(function (key) {
      return sets[key][id] === true;
    });
  }

  // The point id is the retrieval chunk id, so the chunk id of the record is the key.
  function deepLink(doc, question, chunk) {
    var parts = [];
    if (doc) { parts.push("doc=" + encodeURIComponent(doc)); }
    if (question) { parts.push("q=" + encodeURIComponent(question)); }
    if (chunk) { parts.push("chunk=" + encodeURIComponent(chunk)); }
    return "#chunks" + (parts.length ? "?" + parts.join("&") : "");
  }

  // ---- DOM helpers (text only: nothing of the record is ever parsed as HTML) --

  function el(tag, css, content) {
    var node = document.createElement(tag);
    if (css) { node.className = css; }
    if (content !== undefined && content !== null) { node.textContent = content; }
    return node;
  }

  // Runs `fill` the first time `details` is opened.
  function lazy(details, fill) { details._fill = fill; }

  document.addEventListener("toggle", function (ev) {
    var node = ev.target;
    if (!node || !node.open) { return; }
    if (node._fill) { var fill = node._fill; node._fill = null; fill(); return; }
    if (node.classList && node.classList.contains("rawdoc") && !node.getAttribute("data-done")) {
      node.setAttribute("data-done", "1");
      fillRawDoc(node);
    }
  }, true);

  // ---- one chunk card -----------------------------------------------------

  function vectorSummary(vector) {
    var box = el("div", "kv");
    if (!vector) {
      box.appendChild(el("span", "k", "Vector"));
      box.appendChild(el("span", "muted", "sin vector"));
      return box;
    }
    var all = vector.head || [];
    var head = all.slice(0, VECTOR_VALUES).map(function (v) {
      return String(Number(Number(v).toPrecision(5)));
    });
    var more = all.length > head.length ? ", …" : "";
    [["Dimensión", String(vector.dimension)],
     ["Norma", String(Number(Number(vector.norm).toPrecision(6)))],
     ["Primeros valores", "[" + head.join(", ") + more + "]"]
    ].forEach(function (row) {
      box.appendChild(el("span", "k", row[0]));
      box.appendChild(el("code", "", row[1]));
    });
    return box;
  }

  // The card of one chunk: where it is, its full text, and payload and vector.
  function buildCard(entry, question) {
    var point = entry.point;
    var payload = point.payload || {};
    var states = statesOf(question, point.id);
    var card = el("article", "chunk-card" + states.map(function (s) { return " st-" + s; }).join(""));
    card.setAttribute("data-chunk", point.id);

    var head = el("div", "cc-head");
    head.appendChild(el("b", "cc-source", entry.source || "(sin fuente)"));
    if (entry.index !== null) { head.appendChild(el("span", "cc-index", "#" + entry.index)); }
    var start = integer(payload.char_start);
    var end = integer(payload.char_end);
    if (start !== null && end !== null) {
      head.appendChild(el("span", "muted", "caracteres " + start + "–" + end));
    }
    var size = integer(payload.chunk_size);
    if (size === null) { size = text(payload.text).length; }
    head.appendChild(el("span", "muted", size + " car."));
    STATES.forEach(function (state) {
      if (states.indexOf(state[0]) >= 0) {
        head.appendChild(el("span", "badge " + state[0], state[1]));
      }
    });
    var permalink = el("a", "chunk-link cc-link", "enlace");
    permalink.setAttribute("href", deepLink(entry.source, question, point.id));
    head.appendChild(permalink);
    card.appendChild(head);

    if (entry.path) { card.appendChild(el("div", "cc-path", entry.path)); }
    card.appendChild(el("code", "cc-id", point.id));
    card.appendChild(el("pre", "text", text(payload.text)));

    var details = el("details", "cc-raw");
    details.appendChild(el("summary", "", "Payload y vector"));
    lazy(details, function () {
      details.appendChild(el("pre", "text json", JSON.stringify(payload, null, 2)));
      details.appendChild(vectorSummary(point.vector));
    });
    card.appendChild(details);
    return card;
  }

  // ---- raw documents of a question ----------------------------------------

  function fillRawDoc(details) {
    var holder = details.closest("[data-q]");
    var question = holder ? holder.getAttribute("data-q") : "";
    var body = details.querySelector(".rawbody");
    var entries = prepare().bySource[details.getAttribute("data-doc")] || [];
    if (!body) { return; }
    if (!entries.length) {
      body.appendChild(el("p", "muted", "Sin chunks en la colección."));
      return;
    }
    entries.forEach(function (entry) { body.appendChild(buildCard(entry, question)); });
  }

  // ---- the explorer -------------------------------------------------------

  var explorerApi = null;

  function ensureExplorer() {
    if (explorerApi) { return explorerApi; }
    var root = document.getElementById("explorer");
    if (!root) { return null; }
    explorerApi = makeExplorer(root);
    return explorerApi;
  }

  function option(value, label) {
    var node = el("option", "", label);
    node.value = value;
    return node;
  }

  function labelled(label, control) {
    var wrap = el("label", "");
    wrap.appendChild(el("span", "", label));
    wrap.appendChild(control);
    return wrap;
  }

  function makeExplorer(root) {
    var data = prepare();
    var state = { doc: "", text: "", question: "", used: false, page: 0, target: "" };

    var docSelect = el("select", "");
    docSelect.appendChild(option("", "todos"));
    data.sources.forEach(function (source) { docSelect.appendChild(option(source, source)); });
    var search = el("input", "");
    search.type = "search";
    search.placeholder = "Buscar en texto, chunk_id, fuente o encabezados";
    var questionSelect = el("select", "");
    questionSelect.appendChild(option("", "ninguna"));
    questionIds.forEach(function (id) { questionSelect.appendChild(option(id, id)); });
    var usedBox = el("input", "");
    usedBox.type = "checkbox";
    var usedLabel = labelled("solo usados por la pregunta", usedBox);

    var bar = el("div", "explorer-bar");
    bar.appendChild(labelled("documento", docSelect));
    bar.appendChild(search);
    bar.appendChild(labelled("marcar pregunta", questionSelect));
    bar.appendChild(usedLabel);
    var count = el("div", "ex-count");
    var pagerTop = el("div", "pager");
    var list = el("div", "ex-list");
    var pagerBottom = el("div", "pager");
    root.textContent = "";
    [bar, count, pagerTop, list, pagerBottom].forEach(function (node) { root.appendChild(node); });

    function matching() {
      var needle = fold(state.text.trim());
      return data.entries.filter(function (entry) {
        if (state.doc && entry.source !== state.doc) { return false; }
        if (needle && entry.hay.indexOf(needle) < 0) { return false; }
        if (state.used && state.question && !statesOf(state.question, entry.point.id).length) {
          return false;
        }
        return true;
      });
    }

    function go(step) { state.page += step; state.target = ""; render(); }

    function fillPager(pager, pages) {
      pager.textContent = "";
      var previous = el("button", "", "anterior");
      previous.type = "button";
      previous.disabled = state.page <= 0;
      previous.addEventListener("click", function () { go(-1); });
      var next = el("button", "", "siguiente");
      next.type = "button";
      next.disabled = state.page >= pages - 1;
      next.addEventListener("click", function () { go(1); });
      pager.appendChild(previous);
      pager.appendChild(el("span", "", "página " + (state.page + 1) + " de " + pages));
      pager.appendChild(next);
    }

    function render() {
      var found = matching();
      var pages = Math.max(1, Math.ceil(found.length / PAGE_SIZE));
      if (state.target) {
        for (var i = 0; i < found.length; i++) {
          if (found[i].point.id === state.target) { state.page = Math.floor(i / PAGE_SIZE); break; }
        }
      }
      state.page = Math.min(Math.max(state.page, 0), pages - 1);
      usedLabel.style.display = state.question ? "" : "none";
      count.textContent = found.length + " de " + data.entries.length + " chunks";
      list.textContent = "";
      var shown = found.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
      var targetCard = null;
      shown.forEach(function (entry) {
        var card = buildCard(entry, state.question);
        if (state.target && entry.point.id === state.target) {
          card.classList.add("target");
          targetCard = card;
        }
        list.appendChild(card);
      });
      if (!shown.length) { list.appendChild(el("p", "muted", "Ningún chunk coincide.")); }
      fillPager(pagerTop, pages);
      fillPager(pagerBottom, pages);
      if (targetCard && targetCard.scrollIntoView) { targetCard.scrollIntoView({ block: "start" }); }
    }

    function sync() {
      docSelect.value = state.doc;
      search.value = state.text;
      questionSelect.value = state.question;
      usedBox.checked = state.used;
    }

    function changed() { state.page = 0; state.target = ""; render(); }
    docSelect.addEventListener("change", function () { state.doc = docSelect.value; changed(); });
    search.addEventListener("input", function () { state.text = search.value; changed(); });
    questionSelect.addEventListener("change", function () {
      state.question = questionSelect.value;
      if (!state.question) { state.used = false; usedBox.checked = false; }
      changed();
    });
    usedBox.addEventListener("change", function () { state.used = usedBox.checked; changed(); });

    sync();
    render();
    return {
      // Sets the filters of a deep link (`doc`, `q`, `chunk`, `used`) and scrolls to the card.
      open: function (params) {
        state.doc = data.sources.indexOf(params.doc) >= 0 ? params.doc : "";
        state.question = questionIds.indexOf(params.q) >= 0 ? params.q : "";
        state.used = !!state.question && params.used === "1";
        state.text = "";
        state.target = params.chunk || "";
        state.page = 0;
        sync();
        render();
      }
    };
  }

  // ---- comparison matrix (compare.html only) -------------------------------

  var onlyDiff = null;
  try { onlyDiff = document.getElementById("only-diff"); } catch (e) {}
  if (onlyDiff) {
    onlyDiff.parentNode.hidden = false;
    onlyDiff.addEventListener("change", function () {
      document.querySelectorAll("tr[data-q]:not(.diff)").forEach(function (row) {
        row.hidden = onlyDiff.checked;
      });
    });
  }

  // ---- start --------------------------------------------------------------

  var saved = null;
  try { saved = localStorage.getItem(storageKey); } catch (e) {}
  var hashed = location.hash.slice(1);
  if (hashed && route(hashed)) { return; }
  route(keys.indexOf(saved) >= 0 ? saved : keys[0]);
})();

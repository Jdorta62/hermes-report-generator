(function () {
  var tabs = document.querySelectorAll(".tab");
  var panels = document.querySelectorAll(".panel");
  var storageKey = "hermes-report-tab:" + document.title;
  function show(key) {
    tabs.forEach(function (t) { t.classList.toggle("active", t.dataset.key === key); });
    panels.forEach(function (p) { p.classList.toggle("active", p.dataset.key === key); });
    try { localStorage.setItem(storageKey, key); } catch (e) {}
    try { history.replaceState(null, "", "#" + key); } catch (e) {}
  }
  tabs.forEach(function (t) { t.addEventListener("click", function () { show(t.dataset.key); }); });
  document.addEventListener("click", function (ev) {
    var a = ev.target.closest && ev.target.closest("a[data-goto]");
    if (a) { ev.preventDefault(); show(a.dataset.goto); window.scrollTo(0, 0); }
  });
  var keys = Array.prototype.map.call(tabs, function (t) { return t.dataset.key; });
  window.addEventListener("hashchange", function () {
    var named = location.hash.slice(1);
    if (keys.indexOf(named) >= 0) { show(named); }
  });
  var saved = null;
  try { saved = localStorage.getItem(storageKey); } catch (e) {}
  var hashed = location.hash.slice(1);
  if (keys.indexOf(hashed) >= 0) { saved = hashed; }
  show(keys.indexOf(saved) >= 0 ? saved : keys[0]);
})();

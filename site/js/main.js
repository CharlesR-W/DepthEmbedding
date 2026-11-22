// Page shell: hash router (#embedding?<state>, #experiments/<entry>, #method/<id>), maths, heading anchors, and the
// Experiments tab's lazy figures (embedding figures: js/embfig.js; live charts: js/live.js).
const TABS = ["embedding", "experiments", "method"];
let embedding = null, current = null;

function resolveHash() {
  const raw = decodeURIComponent(location.hash.slice(1));
  const qi = raw.indexOf("?"), path = qi < 0 ? raw : raw.slice(0, qi), qs = qi < 0 ? "" : raw.slice(qi + 1);
  let [tab, id] = path.split("/");
  if (!TABS.includes(tab)) {       // a bare #id: the tab that holds it, else the viewer
    const e = tab && document.getElementById(tab), p = e && e.closest(".tab-panel");
    id = p ? tab : null; tab = p ? p.dataset.tab : "embedding";
  }
  const panel = document.getElementById(`tab-${tab}`);
  // #experiments/<id> also finds the entry section x-<id>
  const anchor = (id && (panel.querySelector(`#${CSS.escape(id)}`) || (tab === "experiments" && panel.querySelector(`#x-${CSS.escape(id)}`)))) || null;
  return { tab, id, anchor, query: new URLSearchParams(qs) };
}

async function route() {
  const { tab, id, anchor, query } = resolveHash();
  const wanted = location.hash;
  current = tab;
  document.body.dataset.tab = tab;
  document.querySelectorAll(".tab-panel").forEach(p => p.hidden = p.dataset.tab !== tab);
  document.querySelectorAll(".tablist a").forEach(a => a.setAttribute("aria-selected", String(a.dataset.tab === tab)));
  if (tab === "embedding") {
    if (!embedding) {
      const { DepthEmbedding } = await import("./embedding.js");
      embedding = window.__emb = new DepthEmbedding(document.getElementById("emb"));
    }
    if ([...query.keys()].length && wanted !== embedding.lastHash) await embedding.applyParams(query);
    else await embedding.layout();
  }
  if (tab === "experiments") { xpView(id, anchor); figures(); return; }
  xp.cur = undefined;
  if (anchor) scrollToAnchor(anchor);
  else if (tab !== "embedding") window.scrollTo(0, 0);
}

// ---------------------------------------------------- Experiments: the overview, or one entry at a time
// #experiments = lede and at-a-glance cards; #experiments/x-<id> (or /<id>, or an anchor inside the entry) = that
// entry alone under a sticky sub-navigation; #experiments/all = every entry on one page (printing, searching).
const xp = { list: null, cur: undefined, nav: null };
const escH = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
function xpEntries() {
  if (xp.list) return xp.list;
  const panel = document.getElementById("tab-experiments");
  const secs = [...panel.querySelectorAll(".xp > section.experiment[id]")], byId = new Map(secs.map(s => [s.id, s]));
  const order = [];                // the cards' reading order (groups), then any entry without a card
  panel.querySelectorAll(".x-card h4 a[href]").forEach(a => { const s = byId.get(a.getAttribute("href").split("/")[1]); if (s && !order.includes(s)) order.push(s); });
  secs.forEach(s => { if (!order.includes(s)) order.push(s); });
  const text = el => { if (!el) return ""; const c = el.cloneNode(true); c.querySelectorAll(".anchor").forEach(a => a.remove()); return c.textContent.trim(); };
  xp.list = order.map(s => {
    const card = panel.querySelector(`.x-card h4 a[href="#experiments/${s.id}"]`)?.closest(".x-card");
    return { id: s.id, el: s, card, title: text(s.querySelector(":scope > h2")), group: text(card?.closest(".x-card-group")?.querySelector(":scope > h3")) };
  });
  return xp.list;
}
function xpNav() {
  if (xp.nav) return xp.nav;
  const nav = document.createElement("nav");
  nav.className = "x-subnav"; nav.setAttribute("aria-label", "Experiment entries"); nav.hidden = true;
  nav.innerHTML = `<div class="x-subnav-inner"><a class="x-back" href="#experiments"><span aria-hidden="true">←</span> All experiments</a>`
    + `<span class="x-where"></span><span class="x-pn"><a class="x-prev"></a><a class="x-next"></a></span></div>`;
  const panel = document.getElementById("tab-experiments");
  panel.prepend(nav);
  // a quiet way to see every entry on one page
  const glance = panel.querySelector(".x-glance");
  if (glance) {
    const p = document.createElement("p");
    p.className = "x-showall";
    p.innerHTML = `<a href="#experiments/all">Show all ${xpEntries().length} entries on one page</a> <span>(long; for printing or searching)</span>`;
    glance.after(p);
  }
  return (xp.nav = nav);
}
function xpView(id, anchor) {
  const panel = document.getElementById("tab-experiments"), art = panel.querySelector(".xp"), list = xpEntries(), nav = xpNav();
  const sec = anchor?.closest(".xp > section.experiment") || null;
  const view = id === "all" && !anchor ? "all" : sec ? "entry" : "landing";
  const prev = xp.cur, entry = list.find(e => e.el === sec);
  art.dataset.view = panel.dataset.view = document.body.dataset.xview = view;
  list.forEach(e => e.el.classList.toggle("x-current", e.el === sec));
  nav.hidden = view !== "entry";
  if (entry) {
    const i = list.indexOf(entry), pv = list[i - 1], nx = list[i + 1];
    nav.querySelector(".x-where").innerHTML = `<b>${escH(entry.group || "Experiments")}</b> <span>${i + 1} of ${list.length}</span>`;
    for (const [a, e, dir] of [[nav.querySelector(".x-prev"), pv, "Previous"], [nav.querySelector(".x-next"), nx, "Next"]]) {
      a.hidden = !e;
      if (!e) continue;
      a.href = `#experiments/${e.id}`; a.title = `${dir}: ${e.title}`;
      const t = escH(e.title);
      a.innerHTML = dir === "Previous" ? `<span aria-hidden="true">‹</span> <span class="x-pn-t">${t}</span><span class="x-pn-s">Previous</span>` : `<span class="x-pn-t">${t}</span><span class="x-pn-s">Next</span> <span aria-hidden="true">›</span>`;
    }
    document.title = `${entry.title} · Experiments · Depth embedding`;
  } else document.title = "Experiments · Depth embedding";
  xp.cur = sec;
  // scroll: an anchor inside the entry goes there; a new entry starts at its top; the overview returns to the card of
  // the entry just left (back button or "All experiments"), else to the top
  if (anchor && anchor !== sec) scrollToAnchor(anchor);
  else if (view === "entry") { if (prev !== sec) window.scrollTo({ top: 0, behavior: "instant" }); }
  else if (view === "landing" && prev) {
    const card = list.find(e => e.el === prev)?.card;
    window.scrollTo({ top: 0, behavior: "instant" });
    if (card) requestAnimationFrame(() => { const r = card.getBoundingClientRect(); if (r.bottom > innerHeight - 20) window.scrollTo({ top: scrollY + r.top - innerHeight / 3, behavior: "instant" }); card.classList.add("x-was"); setTimeout(() => card.classList.remove("x-was"), 1600); });
  } else window.scrollTo({ top: 0, behavior: "instant" });
}
window.addEventListener("hashchange", route);

// A deep link must land on its entry even though figures above it are still loading: figures reserve their height
// in CSS, and the scroll is repeated once maths and fonts have settled.
function scrollToAnchor(anchor) {
  for (let d = anchor.closest("details"); d; d = d.parentElement.closest("details")) d.open = true;      // an anchor inside a closed block
  const go = () => anchor.scrollIntoView({ block: "start" });
  requestAnimationFrame(go);
  const want = () => parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 64;
  const again = () => { if (location.hash && resolveHash().anchor === anchor && Math.abs(anchor.getBoundingClientRect().top - want()) > 40 && scrollY > 0) go(); };
  document.fonts?.ready.then(() => setTimeout(again, 50));
  setTimeout(again, 600);
}

// ---------------------------------------------------- Experiments: figures render when they come near the viewport
let io = null;
function figures() {
  if (io) return;
  io = new IntersectionObserver(es => {
    for (const e of es) if (e.isIntersecting) { io.unobserve(e.target); renderFigure(e.target); }
  }, { rootMargin: "600px 0px" });
  document.querySelectorAll("#tab-experiments figure.emb-fig, #tab-experiments figure.live").forEach(f => {
    const d = f.closest("details");
    if (d && !d.open) d.addEventListener("toggle", () => { if (d.open) renderFigure(f); }, { once: true });
    else io.observe(f);
  });
}
async function renderFigure(f) {
  if (f.dataset.rendered) return; f.dataset.rendered = "1";
  try {
    if (f.matches(".emb-fig")) await (await import("./embfig.js")).mount(f);
    else await (await import("./live.js")).render(f);
  } catch (e) { console.error("figure:", e); }
}

function maths() {
  window.renderMathInElement(document.body, {
    delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }],
    ignoredClasses: ["no-math", "tooltip", "ef"], throwOnError: false,
  });
}
function anchors() {
  document.querySelectorAll(".prose h2[id], .prose h3[id], .xp h2[id], .xp h3[id], .xp section.experiment[id] > h2").forEach(hd => {
    const id = hd.id || hd.parentElement.id;
    const a = document.createElement("a"); a.className = "anchor"; a.href = `#${hd.closest(".tab-panel").dataset.tab}/${id}`;
    a.textContent = "#"; a.setAttribute("aria-label", "Link to this section");
    hd.prepend(a);
  });
}
// long code references in the definition tables: allow line breaks after / and : rather than mid-identifier
function codeBreaks() {
  document.querySelectorAll("table.x-defs code").forEach(c => {
    if (c.children.length) return;
    const parts = c.textContent.split(/(?<=[/:])/);
    if (parts.length < 2) return;
    c.replaceChildren(...parts.flatMap((t, i) => i ? [document.createElement("wbr"), t] : [t]));
  });
}
function init() {
  anchors(); codeBreaks();
  if (window.renderMathInElement) maths(); else window.addEventListener("load", maths);
  route();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();

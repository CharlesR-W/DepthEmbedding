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
  return { tab, anchor, query: new URLSearchParams(qs) };
}

async function route() {
  const { tab, anchor, query } = resolveHash();
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
  if (tab === "experiments") figures();
  if (anchor) scrollToAnchor(anchor);
  else if (tab !== "embedding") window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

// A deep link must land on its entry even though figures above it are still loading: figures reserve their height
// in CSS, and the scroll is repeated once maths and fonts have settled.
function scrollToAnchor(anchor) {
  const go = () => anchor.scrollIntoView({ block: "start" });
  requestAnimationFrame(go);
  const again = () => { if (location.hash && resolveHash().anchor === anchor && Math.abs(anchor.getBoundingClientRect().top - 64) > 40 && scrollY > 0) go(); };
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
function init() {
  anchors();
  if (window.renderMathInElement) maths(); else window.addEventListener("load", maths);
  route();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();

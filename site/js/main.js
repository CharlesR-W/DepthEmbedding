// Page shell: hash router (#embedding?<state>, #method/<id>), maths, heading anchors.
const TABS = ["embedding", "method"];
let embedding = null, current = null;

function resolveHash() {
  const raw = decodeURIComponent(location.hash.slice(1));
  const qi = raw.indexOf("?"), path = qi < 0 ? raw : raw.slice(0, qi), qs = qi < 0 ? "" : raw.slice(qi + 1);
  let [tab, id] = path.split("/");
  if (!TABS.includes(tab)) {       // a bare #id: the tab that holds it, else the viewer
    const e = tab && document.getElementById(tab), p = e && e.closest(".tab-panel");
    id = p ? tab : null; tab = p ? p.dataset.tab : "embedding";
  }
  const anchor = (id && document.getElementById(`tab-${tab}`).querySelector(`#${CSS.escape(id)}`)) || null;
  return { tab, anchor, query: new URLSearchParams(qs) };
}

async function route() {
  const { tab, anchor, query } = resolveHash();
  const wanted = location.hash, changed = tab !== current;
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
  if (anchor) requestAnimationFrame(() => anchor.scrollIntoView({ block: "start" }));
  else if (tab !== "embedding") window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

function maths() {
  window.renderMathInElement(document.body, {
    delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }],
    ignoredClasses: ["no-math", "tooltip"], throwOnError: false,
  });
}
function anchors() {
  document.querySelectorAll(".prose h2[id], .prose h3[id]").forEach(hd => {
    const a = document.createElement("a"); a.className = "anchor"; a.href = `#${hd.closest(".tab-panel").dataset.tab}/${hd.id}`;
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

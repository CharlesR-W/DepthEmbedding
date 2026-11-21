// Lazy, cached data access. All URLs are relative to the page; build_site.py writes them under d/.
const cache = new Map();
function once(key, fn) {
  if (!cache.has(key)) { const p = fn().catch(e => { cache.delete(key); throw e; }); cache.set(key, p); }
  return cache.get(key);
}
export function bytes(url) {
  return once("b:" + url, async () => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.arrayBuffer(); });
}
export function json(url) {
  return once("j:" + url, async () => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); });
}
export function image(url) {
  return once("i:" + url, () => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error(url)); im.src = url; }));
}
export const meta = () => json("d/meta.json");
export const sprite = () => image("d/sprite.webp");
export const labels = () => bytes("d/labels.bin").then(b => new Int8Array(b));

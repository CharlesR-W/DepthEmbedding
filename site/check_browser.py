"""Smoke-test a served build with headless Chrome (requires playwright-cli)."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

CHECK_JS = r"""
async page => {
  const {base, dir, capture, reports} = CONFIG;
  const origin = new URL(base).origin, issues = new Set();
  const screenshots = [];
  page.on('console', msg => { if (msg.type() === 'error') issues.add(`console: ${msg.text()}`); });
  page.on('pageerror', err => issues.add(`page error: ${err.message}`));
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin && url.protocol !== 'data:') {
      issues.add(`external request: ${url.href}`);
      await route.abort();
    } else await route.continue();
  });
  page.on('response', r => { if (r.status() >= 400) issues.add(`HTTP ${r.status()}: ${r.url()}`); });
  const shot = async (name, opts = {}) => {
    const path = `${dir}/${name}.png`;
    if (opts.locator) await opts.locator.screenshot({path, timeout: 30000});
    else await page.screenshot({path, fullPage: opts.full ?? true, timeout: 30000});
    screenshots.push(path);
  };
  const test = async (label, fn) => {
    try { await fn(); }
    catch (err) { issues.add(`${label}: ${err.message}`); }
  };
  const route = async tab => {
    const panel = page.locator(`#tab-${tab}`);
    await page.evaluate(t => { location.hash = t; }, tab);
    await panel.waitFor({state:'visible', timeout: 15000});
    return panel;
  };
  const exercise = async fig => {
    const settle = () => page.waitForLoadState('networkidle');
    const slider = fig.getByRole('slider', {name:'Site', exact:true});
    if (await slider.count()) {
      const old = await slider.inputValue();
      await slider.press('End');
      await slider.evaluate((el, value) => {
        el.value = value;
        el.dispatchEvent(new Event('input', {bubbles:true}));
        el.dispatchEvent(new Event('change', {bubbles:true}));
      }, old);
    }
    const play = fig.getByRole('button', {name:'Play through the sites', exact:true});
    if (await play.count()) {
      await play.click();
      await fig.getByRole('button', {name:'Pause', exact:true}).click({timeout:3000});
    }
    const groupLabels = await fig.getByRole('radiogroup').evaluateAll(gs => gs.map(g => g.getAttribute('aria-label')));
    for (const name of groupLabels) {
      const group = fig.getByRole('radiogroup', {name, exact:true});
      const on = group.locator('[role="radio"][aria-checked="true"]');
      const off = group.locator('[role="radio"][aria-checked="false"]:not([disabled])').first();
      if (!await on.count() || !await off.count()) continue;
      const original = await on.innerText();
      await off.click();
      await settle();
      await group.getByRole('radio', {name:original, exact:true}).click();
      await settle();
    }
    const labels = await fig.getByRole('combobox').evaluateAll(ss => ss.map(s => s.getAttribute('aria-label')));
    for (const name of labels) {
      const sel = fig.getByRole('combobox', {name, exact:true});
      if (!await sel.count() || !await sel.isVisible()) continue;
      const old = await sel.inputValue();
      const other = await sel.evaluate(s => [...s.options].find(o => !o.disabled && o.value !== s.value)?.value ?? null);
      if (other === null) continue;
      await sel.selectOption(other);
      await settle();
      await sel.selectOption(old);
      await settle();
    }
  };
  const inspect = async (scope, exerciseEmb = false) => {
    const savedDetails = await scope.locator('details').evaluateAll(ds => ds.map(d => { const old = d.open; d.open = true; return old; }));
    try {
      const images = scope.locator('img');
      const broken = await images.evaluateAll(async imgs => {
        await Promise.all(imgs.map(im => { im.loading = 'eager'; return im.decode().catch(() => {}); }));
        return imgs.filter(im => !im.complete || !im.naturalWidth).map(im => im.currentSrc || im.src);
      });
      for (const url of broken) issues.add(`broken image: ${url}`);
      const figures = scope.locator('figure.emb-fig, figure.live');
      for (let i = 0, n = await figures.count(); i < n; i++) {
        const fig = figures.nth(i);
        const handle = await fig.elementHandle();
        const label = await fig.evaluate(f => f.dataset.emb || f.dataset.src || f.className);
        await test(label, async () => {
          await fig.scrollIntoViewIfNeeded();
          const rest = () => page.waitForFunction(f => {
            const visible = el => el.getBoundingClientRect().width && el.getBoundingClientRect().height;
            if (f.classList.contains('emb-fig')) {
              const canvases = [...f.querySelectorAll('canvas')].filter(visible);
              return canvases.length && canvases.every(c => {
                const ctx = c.getContext('2d');
                if (!ctx || !c.width || !c.height) return false;
                const {data} = ctx.getImageData(0, 0, c.width, c.height);
                for (let i=4; i<data.length; i+=4) {
                  if (data[i] !== data[0] || data[i+1] !== data[1] || data[i+2] !== data[2] || data[i+3] !== data[3]) return true;
                }
                return false;
              });
            }
            const charts = [...f.querySelectorAll('.live-body svg')].filter(visible);
            return charts.length && charts.every(s => !!s.querySelector('path[d], rect[width], line, circle[r], polyline[points], text'));
          }, handle, {timeout: 20000});
          await rest();
          if (exerciseEmb && await fig.evaluate(el => el.classList.contains('emb-fig'))) {
            await exercise(fig);
            await rest();
          }
        });
        await handle.dispose();
      }
    } finally {
      await scope.locator('details').evaluateAll((ds, saved) => ds.forEach((d, i) => { d.open = saved[i]; }), savedDetails);
    }
  };
  const captured = new Set();
  // thumbnail capture: the plot area of the entry's lead embedding figure (labels, rings and notes hidden), else its
  // first live chart, else its number tiles stacked in one column
  const captureLead = async (id, entry) => {
    const emb = entry.locator('figure.emb-fig').first();
    if (await emb.count()) {
      await emb.evaluate(async f => { await f.__emb.ready; f.__emb.setThumb(true); });
      await page.waitForTimeout(150);
      try { await shot(`capture-${id}`, {locator: emb.locator('.ef-panels')}); }
      finally { await emb.evaluate(f => f.__emb.setThumb(false)); }
      return;
    }
    const live = entry.locator('figure.live .live-body').first();
    if (await live.count()) { await shot(`capture-${id}`, {locator: live}); return; }
    const tiles = entry.locator('.x-numbers').first();
    const old = await tiles.evaluate(t => { const o = t.getAttribute('style'); t.style.gridTemplateColumns = '1fr'; t.style.width = '440px'; return o; });
    try { await shot(`capture-${id}`, {locator: tiles}); }
    finally { await tiles.evaluate((t, o) => { if (o == null) t.removeAttribute('style'); else t.setAttribute('style', o); }, old); }
  };
  for (const [size, width, height] of [['desktop', 1440, 900], ['mobile', 390, 844]]) {
    await page.setViewportSize({width, height});
    await page.goto(base, {waitUntil:'load'});
    const tabs = await page.locator('.tab-panel[data-tab]').evaluateAll(els => els.map(e => e.dataset.tab));
    if (!tabs.includes('embedding') || !tabs.includes('method')) issues.add('main page is missing Viewer or Method');
    for (const tab of tabs) await test(`${size} ${tab}`, async () => {
      const panel = await route(tab);
      if (tab === 'embedding') {
        await page.waitForLoadState('networkidle');
      }
      // the Experiments tab opens on its overview (lede and cards); each entry is shown alone at #experiments/x-<id>
      await inspect(tab === 'experiments' ? panel.locator('.xp > :not(section.experiment)') : panel, false);
      if (tab === 'experiments') {
        const shown = await panel.locator('section.experiment').evaluateAll(els => els.filter(e => e.getBoundingClientRect().height > 0).length);
        if (shown) issues.add(`${size} experiments overview: ${shown} entries visible, expected none`);
      }
      await page.evaluate(() => window.scrollTo(0, 0));
      await shot(`${tab === 'embedding' ? 'viewer' : tab}-${size}`);
      if (tab === 'experiments') {
        const entries = await panel.locator('section.experiment').evaluateAll(els => els.map(e => e.id));
        for (const id of entries) await test(`${size} ${id}`, async () => {
          await page.evaluate(id => { location.hash = `experiments/${id}`; }, id);
          const entry = page.locator(`#${id}`);
          await entry.waitFor({state:'visible'});
          const shown = await panel.locator('section.experiment').evaluateAll(els => els.filter(e => e.getBoundingClientRect().height > 0).map(e => e.id));
          if (shown.length !== 1 || shown[0] !== id) issues.add(`${size} ${id}: visible entries ${JSON.stringify(shown)}`);
          await inspect(entry, size === 'desktop');
          await page.evaluate(() => window.scrollTo(0, 0));
          await shot(`${id}-${size}`);
          const xid = id.replace(/^x-/, '');
          if (size === 'desktop' && (capture.includes(xid) || capture.includes('all'))) {
            await captureLead(xid, entry);
            captured.add(xid);
          }
        });
      }
    });
    for (const [name, detailTab] of reports) await test(`${size} report ${name}`, async () => {
      const url = new URL(`reports/${name}.html#tab=overview`, base).href;
      await page.goto(url, {waitUntil:'load'});
      await page.waitForFunction(() => !!window.MathJax?.typesetPromise && !!window.Plotly, {timeout:15000});
      const reportTabs = page.locator('nav.tabs button[data-tab]');
      if (size === 'desktop') {
        // The report screenshots show one representative chart tab, but exercise the other lazy tabs too.
        for (let i = 0; i < await reportTabs.count(); i++) {
          const button = reportTabs.nth(i);
          const key = await button.getAttribute('data-tab');
          if (key === detailTab) continue;
          await button.click();
          await inspect(page.locator(`section.tab[data-tab="${key}"]`));
        }
      }
      await page.locator(`nav.tabs button[data-tab="${detailTab}"]`).click();
      const detail = page.locator(`section.tab[data-tab="${detailTab}"]`);
      await detail.locator('mjx-container').first().waitFor({state:'visible', timeout:15000});
      await detail.locator('.js-plotly-plot .main-svg').first().waitFor({state:'visible', timeout:15000});
      const font = await page.evaluate(async () => {
        await document.fonts.ready;
        return [...document.fonts].some(f => f.family.startsWith('MJX') && f.status === 'loaded');
      });
      if (!font) issues.add(`report ${name}: no MathJax font loaded`);
      await inspect(detail);
      await page.evaluate(() => window.scrollTo(0, 0));
      await shot(`report-${name}-${size}`);
    });
  }
  for (const id of capture) if (id !== 'all' && !captured.has(id)) issues.add(`cannot capture ${id}: entry absent`);
  if (capture.includes('all')) capture.splice(0, capture.length, ...captured);
  return '@@JSON@@' + JSON.stringify({issues:[...issues], screenshots, captured:[...captured]}) + '@@END@@';
}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base_url", help="e.g. http://127.0.0.1:8812/ (include any deploy path)")
    ap.add_argument("out_dir", type=Path, help="directory for screenshots")
    ap.add_argument("--session", help="unique playwright-cli session name (default includes PID)")
    ap.add_argument("--capture", metavar="ID[,ID…]|all",
                    help="also write site/assets/fig/x_ID_thumb.webp for these entries, or all (requires Pillow)")
    a = ap.parse_args()
    base = a.base_url.rstrip("/") + "/"
    url = urlsplit(base)
    if url.scheme not in ("http", "https") or not url.hostname or url.fragment or url.query:
        ap.error("base_url must be an HTTP(S) directory URL without fragment or query")
    capture = [c for c in (a.capture or "").split(",") if c]
    if any(not re.fullmatch(r"[a-z][a-z0-9-]*", c) for c in capture):
        ap.error("--capture expects lower-case experiment ids (comma-separated) or all")
    if a.capture:
        try:
            from PIL import Image
        except ImportError:
            ap.error("--capture requires Pillow; try uv run --with pillow python ...")
    dest = a.out_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    session = a.session or f"depth-check-{os.getpid()}"
    cmd = ["playwright-cli", f"-s={session}"]

    def run(*args, timeout=45, check=True):
        r = subprocess.run(cmd + list(args), cwd=dest, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode:
            raise RuntimeError(f"playwright-cli {args[0]} failed:\n{r.stdout}\n{r.stderr}")
        return r.stdout

    js = CHECK_JS.replace("CONFIG", json.dumps(dict(base=base, dir=str(dest), capture=capture,
                                                 reports=[("inversion", "sources"), ("xp", "x1")])), 1)
    # Use an argument rather than a temporary JS file when invoking the pre-built CLI.
    try:
        try:
            run("open", "about:blank", "--browser=chrome")
            output = run("run-code", js, "--raw", timeout=900)
        finally:
            run("close", check=False)
        result = json.loads(output.strip())
        if not isinstance(result, str) or not result.startswith("@@JSON@@") or not result.endswith("@@END@@"):
            raise RuntimeError(f"unexpected browser output: {output[-2000:]}")
        result = json.loads(result[len("@@JSON@@"):-len("@@END@@")])
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
        ap.exit(1, f"browser check failed: {exc}\n")
    for issue in result["issues"]:
        print(f"error: {issue}")
    print(f"{len(result['screenshots'])} screenshots and {len(result['issues'])} browser errors: {dest}")
    if capture and not result["issues"]:
        for xid in result.get("captured", []):
            thumb = Path(__file__).resolve().parent / "assets" / "fig" / f"x_{xid}_thumb.webp"
            thumb.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(dest / f"capture-{xid}.png") as im:
                make_thumb(im).save(thumb, quality=80, method=6)
            print(f"wrote {thumb} ({thumb.stat().st_size // 1024} KB)")
    return int(bool(result["issues"]))


THUMB = (800, 400)          # 2:1, twice the widest card image (CSS aspect-ratio 2 / 1)


def make_thumb(im):
    """Trim the white margin around the drawn content, then centre it on white at 2:1 and scale to THUMB."""
    from PIL import Image, ImageOps
    rgb = im.convert("RGB")
    ink = ImageOps.invert(rgb.convert("L")).point(lambda v: 255 if v > 10 else 0)    # anything darker than ~245
    box = ink.getbbox() or (0, 0, *rgb.size)
    rgb = rgb.crop(box)
    w, h = rgb.size
    m = round(0.05 * max(w, h / 2))
    W, H = w + 2 * m, h + 2 * m
    W, H = max(W, 2 * H), max(H, -(-W // 2))
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(rgb, ((W - w) // 2, (H - h) // 2))
    return canvas.resize(THUMB, Image.LANCZOS)


if __name__ == "__main__":
    raise SystemExit(main())

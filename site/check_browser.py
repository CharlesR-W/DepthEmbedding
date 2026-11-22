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
  let captured = false;
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
      await inspect(panel, size === 'desktop' && tab === 'experiments');
      await page.evaluate(() => window.scrollTo(0, 0));
      await shot(`${tab === 'embedding' ? 'viewer' : tab}-${size}`);
      if (tab === 'experiments') {
        const entries = await panel.locator('section.experiment').evaluateAll(els => els.map(e => e.id));
        for (const id of entries) await test(`${size} ${id}`, async () => {
          await page.evaluate(id => { location.hash = `experiments/${id}`; }, id);
          const entry = page.locator(`#${id}`);
          await entry.waitFor({state:'visible'});
          await shot(`${id}-${size}`, {locator:entry});
          if (capture && id === `x-${capture}` && size === 'desktop') {
            const lead = entry.locator('figure.emb-fig, figure.live, figure').first();
            await shot(`capture-${capture}`, {locator: await lead.count() ? lead : entry});
            captured = true;
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
  if (capture && !captured) issues.add(`cannot capture ${capture}: entry absent`);
  return '@@JSON@@' + JSON.stringify({issues:[...issues], screenshots}) + '@@END@@';
}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base_url", help="e.g. http://127.0.0.1:8812/ (include any deploy path)")
    ap.add_argument("out_dir", type=Path, help="directory for screenshots")
    ap.add_argument("--session", help="unique playwright-cli session name (default includes PID)")
    ap.add_argument("--capture", metavar="ID", help="also write site/assets/fig/x_ID_thumb.webp (requires Pillow)")
    a = ap.parse_args()
    base = a.base_url.rstrip("/") + "/"
    url = urlsplit(base)
    if url.scheme not in ("http", "https") or not url.hostname or url.fragment or url.query:
        ap.error("base_url must be an HTTP(S) directory URL without fragment or query")
    if a.capture and not re.fullmatch(r"[a-z][a-z0-9-]*", a.capture):
        ap.error("--capture expects a lower-case experiment id")
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

    js = CHECK_JS.replace("CONFIG", json.dumps(dict(base=base, dir=str(dest), capture=a.capture,
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
    if a.capture and not result["issues"]:
        thumb = Path(__file__).resolve().parent / "assets" / "fig" / f"x_{a.capture}_thumb.webp"
        thumb.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(dest / f"capture-{a.capture}.png") as im:
            im.thumbnail((720, 480))
            im.convert("RGB").save(thumb, quality=82)
        print(f"wrote {thumb}")
    return int(bool(result["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())

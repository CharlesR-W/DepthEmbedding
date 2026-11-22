"""Check xp/pages/notebook.html: embedded numbers equal the summaries; every control value draws without error;
one full-page screenshot per tab and one screenshot per figure; no console errors.

    python3 xp/verify_notebook.py [screenshot dir, default /tmp/xp_shots] [--width 1440]

Needs `playwright-cli` (headless Chrome).  Serves the page on a loopback port (file:// is blocked).  Exit status 1 on
any mismatch, draw error or console error.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

XP = Path(__file__).resolve().parent
PAGE = XP / 'pages' / 'notebook.html'
args = [a for a in sys.argv[1:] if not a.startswith('--') and not a.isdigit()]
SHOTS = Path(args[0] if args else '/tmp/xp_shots')
WIDTH = int(sys.argv[sys.argv.index('--width') + 1]) if '--width' in sys.argv else 1440
PORT = 8791
fail = []

# ---- 1. numbers: the page's embedded data is the summaries' data, and printed numbers come from the intended fields
html = PAGE.read_text()
data = json.loads(re.search(r'<script type="application/json" id="nb-data">(.*?)</script>', html, re.S).group(1).replace('<\\/', '</'))
x1 = json.loads((XP / 'out/x1_summary.json').read_text())
x3 = json.loads((XP / 'out/x3_summary.json').read_text())
for p in x1['probes']:
    a, b = data['x1']['measures'][p], x1['measures'][p]
    for k in ('nets', 'group', 'cells', 'null', 'readout', 'matrix', 'modes'):
        if a[k] != b[k]:
            fail.append(f'x1 {p}.{k} differs from the summary')
for k in ('attacks', 'class_mass', 'arrival', 'path', 'logits_only', 'extension'):
    if data['x3'][k] != x3[k]:
        fail.append(f'x3 {k} differs from the summary')
c = x1['measures']['c10']['cells']['joint_knn15_trunk']
if 'c10|c100' in c and 'CIFAR-10 against CIFAR-100 ' + ', '.join(f"{c['c10|c100']['mean'][i]:.2f}" for i in (0, 3, 6, 9)) not in html:
    fail.append('X1 takeaway: joint agreement c10|c100 at stem, b3, b6, b9 not printed as in the summary')
u, nz = x3['attacks']['u8']['c10-s0']['disp_site_y'], x3['attacks']['n8']['c10-s0']['disp_site_y']
if f'{u[3]:.2f} / {nz[3]:.2f} (b3)' not in html:
    fail.append('X3 takeaway: b3 site-local displacement not printed as in the summary')
x4p = XP / 'out/x4_summary.json'
if x4p.exists():
    x4 = json.loads(x4p.read_text())
    for l, P in x4['per_start'].items():
        for k in ('stats', 'embedding', 'aligned', 'gain', 'switches', 'speed_rel', 'speed_unit_rel'):
            if json.dumps(data['x4']['per_start'][l][k], sort_keys=True) != json.dumps(P[k], sort_keys=True):
                fail.append(f'x4 {l}.{k} differs from the summary')
    if data['x4']['probe_gain']['rho'] != x4['probe_gain']['rho']:
        fail.append('x4 probe_gain.rho differs from the summary')
    rr = dict(zip(x4['per_start']['b3']['sites'], x4['per_start']['b3']['aligned']['site_peak_rho_all']))
    if f"{rr['b4']:.2f} at b4, {rr['b7']:.2f} at b7" not in html:
        fail.append('X4 takeaway: site_peak_rho of the b3 start not printed as in the summary')
    if f"{x4['probe_gain']['rho']['pixels>logits_ang']:.2f} from pixels" not in html:
        fail.append('X4 takeaway: probe Spearman not printed as in the summary')
for n, d in x1['training'].items():
    if f"{100 * d['test_acc']:.1f}%" not in html:
        fail.append(f'training table: test accuracy of {n} missing')
print('numbers:', 'OK' if not fail else fail)

# ---- 2. browser
tabs = re.findall(r'<button data-tab="([\w-]+)">', html)
figs = {t: re.findall(r'FIGS\["([\w-]+)"\] = \{tab: "' + t + '"', html) for t in tabs}
SHOTS.mkdir(parents=True, exist_ok=True)
srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1', '-d', str(PAGE.parent)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
URL = f'http://127.0.0.1:{PORT}/{PAGE.name}'
run = lambda *a: subprocess.run(['playwright-cli', '-s=xpverify', *a], capture_output=True, text=True, cwd='/tmp')  # noqa: E731


def settle(f, timeout=30):
    """Wait until a screenshot file exists and has stopped growing (the CLI can return before the write finishes)."""
    last, t0 = -1, time.time()
    while time.time() - t0 < timeout:
        n = f.stat().st_size if f.exists() else -1
        if n > 0 and n == last:
            return
        last = n
        time.sleep(0.7)
    fail.append(f'screenshot not written: {f.name}')
SWEEP = r"""() => { const bad = [];
  for (const w of document.querySelectorAll('section.tab.active .ctl[data-ctl]')) {
    const tab = w.dataset.tab, name = w.dataset.ctl, vals = JSON.parse(w.dataset.values), keep = STATE[tab][name];
    for (const v of vals) { setControl(tab, name, v);
      document.querySelectorAll('section.tab.active .plot').forEach(p => { if (p.textContent.includes('draw error')) bad.push(name + '=' + v + ': ' + p.textContent); }); }
    setControl(tab, name, keep); }
  return JSON.stringify(bad); }"""
try:
    run('open', URL, '--browser=chrome')
    run('resize', str(WIDTH), '900')
    for t in tabs:
        run('goto', f'{URL}#tab={t}')
        run('reload')
        time.sleep(4)
        r = run('eval', SWEEP).stdout
        m = re.search(r'"(\[.*?\])"', r.replace('\\"', '"'))
        if m and m.group(1) != '[]':
            fail.append(f'{t}: draw errors {m.group(1)[:300]}')
        time.sleep(1)
        out = SHOTS / f'{t}_{WIDTH}.png'
        out.unlink(missing_ok=True)
        run('screenshot', f'--filename={out}', '--full-page')
        settle(out)
        run('eval', "() => { document.querySelector('nav.tabs').style.position = 'static'; }")
        for f in figs[t]:
            fo = SHOTS / f"{t}__{f}_{WIDTH}.png"
            fo.unlink(missing_ok=True)
            run('screenshot', f'#fig-{f}', f'--filename={fo}')
            settle(fo)
        print('tab', t, 'figures', len(figs[t]), '->', SHOTS)
    con = run('console').stdout
finally:
    run('close')
    srv.terminate()
errs = [l for l in con.splitlines() if l.startswith('[ERROR]') and 'favicon' not in l]
print('console errors:', errs or 'none')
fail += errs
print('FAIL' if fail else 'PASS', fail if fail else '')
sys.exit(1 if fail else 0)

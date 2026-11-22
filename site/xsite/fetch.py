"""Preflight or fetch all base-plan inputs. Run with `uv run --with modal python site/xsite/fetch.py`."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FETCH = ROOT / 'data/fetch'
XP = 'crw-depth-xp-results'
INV = 'crw-depth-invert-results'
SITES = ('stem', *(f'b{i}' for i in range(1, 10)), 'pool', 'logits')
NETS = ('c10-s0', 'c10-s1', 'init-s0', 'init-s1', 'c10x-s0', 'c100-s0', 'c100-s1', 'svhn-s0', 'svhn-s1',
        'anim-s0', 'anim-s1', 'veh-s0', 'veh-s1', 'rand-s0', 'c100c-s0', 'c100c-s1', 'in32-s0', 'in32-s1')
X3_SETS = ('clean', 'u1', 'u2', 'u4', 'u8', 't8', 'n1', 'n2', 'n4', 'n8', 'natA', 'natB')
X4_STARTS = ('pixels', 'b3', 'b6', 'b9')
INV_SETS = ('H', 'c100', 'svhn', 'noise20', 'pgd2', 'pgd8')

REMOTE = {
    XP: ['probes.npz', 'x3/DONE', 'x3/sets.npz', 'x3/check_logits_only.npz',
         *(f'x3/op_c10-s0_{k}.npz' for k in ('ea100', 'knn15')),
         *(f'x3/c10-s0_{s}.npz' for s in X3_SETS),
         *(f'embed/{n}/c10/op_knn15_{c}.npz' for n in NETS for c in ('trunk', 'full')),
         *(f'embed/{n}/c10/{f}' for n in NETS for f in ('meta.npz', 'DONE')),
         *(f'nets/{n}/info.json' for n in NETS if n not in ('c10-s0', 'c10-s1', 'init-s0', 'init-s1')),
         'compare/c10.npz', 'x4/pairs.npz', 'x4/probe_gain.npz', 'x4/checks.json',
         *(f'x4/paths_{s}.npz' for s in X4_STARTS)],
    INV: ['splits.npz', *(f'joint/P2__{k}__L12/modes.npz' for k in ('pr300', 'ea300')),
          *(f'joint/P2__pr300__L12/coords_{s}.npz' for s in INV_SETS),
          *(f'decode/P2__pr300__L12/{s}__{s}.npz' for s in SITES),
          *(f'acts/{s}/meta.npz' for s in ('c10', *INV_SETS[1:]))],
}


def locals_needed():
    return ['data/invert/results.json', 'data/invert/splits.npz', 'data/r20s0b/probe.npz',
            'data/r20s0b/spacetime/st_64000_0.npz',
            *(f'xp/out/x{i}_summary.json' for i in (1, 3, 4)),
            *(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'data/invert/figs').glob('*.png')))]


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value, *, indent=2):
    part = path.with_name(path.name + '.part')
    opts = {'indent': indent} if indent is not None else {'separators': (',', ':')}
    part.write_bytes((json.dumps(value, ensure_ascii=False, allow_nan=False, **opts) + '\n').encode('utf-8'))
    os.replace(part, path)


async def retry(call, what, attempts=8):
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as exc:
            if attempt == attempts - 1 or isinstance(exc, (FileNotFoundError, PermissionError)):
                raise
            delay = min(60, 2 ** (attempt + 1))
            print(f'{what}: {type(exc).__name__}; retrying in {delay}s', file=sys.stderr, flush=True)
            await asyncio.sleep(delay)


async def run(inventory_only):
    import modal

    pins = {}
    manifest = Path(__file__).with_name('inputs.json')
    if manifest.exists():
        for op in json.loads(manifest.read_text())['ops'].values():
            for key, digest in op['bundle'].items():
                if key in pins and pins[key] != digest:
                    raise ValueError(f'Conflicting pinned digests: {key}')
                pins[key] = digest
    vols = {name: modal.Volume.from_name(name) for name in REMOTE}
    entries = {}
    missing = []
    # One call per directory, paced even on success, instead of one per file.
    for name, paths in REMOTE.items():
        groups = {}
        for path in paths:
            groups.setdefault(str(Path(path).parent), []).append(path)
        for group, wanted in groups.items():
            vol = vols[name]
            try:
                rows = await retry(lambda: vol.listdir.aio('/' if group == '.' else group), f'{name}:{group}')
            except (modal.exception.NotFoundError, FileNotFoundError):
                rows = []
            files = {r.path.lstrip('/'): r.size for r in rows if r.type.name == 'FILE'}
            for path in wanted:
                key = f'{name}/{path}'
                if path not in files:
                    missing.append(f'{name}:{path}')
                else:
                    entries[key] = (name, path, files[path])
                    print(f'{name}:{path}\t{files[path]}', flush=True)
            await asyncio.sleep(0.4)
    for path in locals_needed():
        src = ROOT / path
        if not src.is_file():
            missing.append(f'local:{path}')
        else:
            entries[f'local/{path}'] = (None, path, src.stat().st_size)
            print(f'local:{path}\t{src.stat().st_size}', flush=True)
    if not any((ROOT/'data/invert/figs').glob('*.png')):
        missing.append('local:data/invert/figs/*.png')
    if missing:
        raise RuntimeError('Missing inputs:\n' + '\n'.join(missing))
    print(f'Inventory OK: {len(entries)} files, {sum(e[2] for e in entries.values())} bytes', flush=True)
    if inventory_only:
        return

    FETCH.mkdir(parents=True, exist_ok=True)
    ready_path = FETCH / 'READY.json'
    old = json.loads(ready_path.read_text()) if ready_path.exists() else {}
    ready = {}
    limit = asyncio.Semaphore(3)

    async def transfer(key, name, path, size):
        async with limit:
            target = FETCH / key
            source = f'{name or "local"}:{path}'
            old_entry = old.get(key, {})
            if (old_entry.get('bytes') == size and old_entry.get('volume_path') == source and target.is_file()
                    and target.stat().st_size == size and sha256(target) == old_entry.get('sha256')
                    and (name or sha256(ROOT/path) == old_entry.get('sha256'))):
                if key in pins and old_entry['sha256'] != pins[key]:
                    raise ValueError(f'{key}: existing input does not match the pinned raw bundle')
                ready[key] = old_entry
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + '.part')

            async def download():
                import shutil
                with part.open('wb') as f:
                    if name is None:
                        with (ROOT/path).open('rb') as src:
                            shutil.copyfileobj(src, f)
                    else:
                        async for chunk in vols[name].read_file.aio(path):
                            f.write(chunk)
                if part.stat().st_size != size:
                    raise IOError(f'{key}: expected {size}, got {part.stat().st_size}')

            try:
                await retry(download, source)
                digest = sha256(part)
                if key in pins and digest != pins[key]:
                    raise ValueError(f'{key}: downloaded input does not match the pinned raw bundle')
                os.replace(part, target)
            finally:
                part.unlink(missing_ok=True)
            ready[key] = {'sha256': digest, 'bytes': size, 'volume_path': source,
                          'fetched_at': datetime.now(timezone.utc).isoformat()}
            print(f'Fetched {key}', flush=True)

    async with asyncio.TaskGroup() as tasks:
        for key, entry in entries.items():
            tasks.create_task(transfer(key, *entry))
    write_json(ready_path, dict(sorted(ready.items())))
    print(f'Published {ready_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', action='store_true', help='list required remote and local files and stop')
    args = parser.parse_args()
    # `.part` and READY are single-writer, including if the command is run by hand twice.
    if args.inventory:
        asyncio.run(run(True))
    else:
        import fcntl
        FETCH.mkdir(parents=True, exist_ok=True)
        with (FETCH / '.fetch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(run(False))


if __name__ == '__main__':
    main()

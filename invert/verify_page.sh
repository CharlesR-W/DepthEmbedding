#!/usr/bin/env bash
# One full-page screenshot per tab of invert/pages/invert.html, plus console errors and the page size.
#   bash invert/verify_page.sh [shot-dir] [width]
# Runs headless Chrome through playwright-cli from the shot directory (keeps its logs out of the repo).
# playwright-cli blocks file: URLs, so the page is served on a loopback port for the duration of the check.
# Exit status 1 if the page logged console errors or draw errors, or is over 15 MB.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAGE="$HERE/pages/invert.html"
SHOTS="${1:-/tmp/invert-shots}"
WIDTH="${2:-1440}"
mkdir -p "$SHOTS"
SHOTS="$(cd "$SHOTS" && pwd)"
# Chrome's profile and playwright's temp files go beside the shots (a full /tmp otherwise breaks `open`)
export TMPDIR="$SHOTS/.tmp"; mkdir -p "$TMPDIR"
S="invert-verify-$$"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$HERE/pages" >/dev/null 2>&1 &
SRV=$!
URL="http://127.0.0.1:$PORT/invert.html"
TABS=$(grep -o 'nav class="tabs">.*</nav>' "$PAGE" | grep -o 'data-tab="[^"]*"' | sed 's/data-tab="//;s/"//')
cd "$SHOTS"
pw() { playwright-cli -s="$S" "$@"; }
trap 'pw close >/dev/null 2>&1 || true; kill $SRV 2>/dev/null || true' EXIT
sleep 1
pw open "$URL" --browser=chrome >/dev/null || { echo "could not open $URL"; exit 1; }
pw resize "$WIDTH" 1000 >/dev/null
fail=0
for t in $TABS; do
  pw eval "() => { setTab('$t'); return new Promise(r => setTimeout(r, 2500)); }" >/dev/null
  pw screenshot --filename="$SHOTS/$t.png" --full-page >/dev/null || { echo "screenshot of $t failed"; fail=1; }
  n=$(pw eval "() => document.querySelectorAll('section.tab[data-tab=\"$t\"] .plot p').length + ':' + Array.from(document.querySelectorAll('section.tab[data-tab=\"$t\"] .plot p')).map(p => p.textContent.slice(0, 60)).join(' | ')" | grep -o '"[^"]*"' | head -1 || true)
  echo "tab $t -> $SHOTS/$t.png  notes in plots: $n"
  case "$n" in *"draw error"*) fail=1;; esac
done
# favicon.ico 404s come from the loopback server only (a page opened from disk makes no such request)
errs=$(pw console error 2>&1 | grep -v 'favicon.ico' || true)
echo "--- console (errors) ---"; echo "$errs"
if echo "$errs" | grep -q '^\[ERROR\]'; then fail=1; fi
size=$(stat -c %s "$PAGE")
echo "page size: $((size / 1000)) kB"
[ "$size" -gt 15000000 ] && { echo "over 15 MB"; fail=1; }
exit $fail

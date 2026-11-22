#!/usr/bin/env bash
# One full-page screenshot per tab of invert/pages_ddm/ddm.html, plus console errors, draw errors, unrendered TeX and size.
#   bash invert/verify_ddm_page.sh [shot-dir] [width]
# Runs headless Chrome through playwright-cli from the shot directory (keeps its logs out of the repo); the page is
# served on a loopback port only for the duration of the check.  Exit status 1 on any console or draw error.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAGE="$HERE/pages_ddm/ddm.html"
SHOTS="${1:-/tmp/ddm-shots}"
WIDTH="${2:-1440}"
mkdir -p "$SHOTS"
SHOTS="$(cd "$SHOTS" && pwd)"
export TMPDIR="$SHOTS/.tmp"; mkdir -p "$TMPDIR"
S="ddm-verify-$$"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$HERE/pages_ddm" >/dev/null 2>&1 &
SRV=$!
URL="http://127.0.0.1:$PORT/ddm.html"
TABS=$(grep -o 'nav class="tabs">.*</nav>' "$PAGE" | grep -o 'data-tab="[^"]*"' | sed 's/data-tab="//;s/"//')
cd "$SHOTS"
pw() { playwright-cli -s="$S" "$@"; }
trap 'pw close >/dev/null 2>&1 || true; kill $SRV 2>/dev/null || true' EXIT
sleep 1
pw open "$URL" --browser=chrome >/dev/null || { echo "could not open $URL"; exit 1; }
pw resize "$WIDTH" 1000 >/dev/null
pw eval "() => new Promise(r => setTimeout(r, 3000))" >/dev/null
fail=0
for t in $TABS; do
  pw eval "() => { setTab('$t'); return new Promise(r => setTimeout(r, 2500)); }" >/dev/null
  pw screenshot --filename="$SHOTS/$t.png" --full-page >/dev/null || { echo "screenshot of $t failed"; fail=1; }
  probe='() => { const sec = document.querySelector("section.tab[data-tab=TAB]");
    const errs = Array.from(sec.querySelectorAll(".plot p")).map(p => p.textContent.slice(0, 80));
    const plots = sec.querySelectorAll(".plot").length, drawn = sec.querySelectorAll(".plot .main-svg").length;
    const txt = sec.innerText + document.querySelector(".banner").innerText;
    const tex = (txt.match(/[$][^$]{1,40}[$]|\\(frac|sum|sigma|lambda|phi|boxed)/g) || []).length;
    return "plots " + plots + ", drawn " + drawn + ", literal TeX " + tex + (errs.length ? ", notes: " + errs.join(" | ") : ""); }'
  info=$(pw eval "${probe//TAB/$t}" | grep -o '"[^"]*"' | head -1 || true)
  echo "tab $t -> $SHOTS/$t.png  $info"
  case "$info" in *"draw error"*) fail=1;; esac
done
errs=$(pw console error 2>&1 | grep -v 'favicon.ico' || true)
msgs=$(pw console 2>&1 | grep -iv 'favicon' | head -40 || true); echo "--- console (all) ---"; echo "$msgs"
echo "--- console (errors) ---"; echo "$errs"
if echo "$errs" | grep -q '^\[ERROR\]'; then fail=1; fi
size=$(stat -c %s "$PAGE")
echo "page size: $((size / 1000)) kB"
[ "$size" -gt 15000000 ] && { echo "over 15 MB"; fail=1; }
exit $fail

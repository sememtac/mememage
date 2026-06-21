#!/usr/bin/env bash
#
# Mememage CORE — end to end, from the shell.
#
# Proves the core (encode / decode / verify) works with no networking and no
# record schema — just the bar in the pixels, the record fields YOU choose, and
# the content hash. This is the workflow a programmer adopts Mememage with:
#
#     encode an image  →  store the record  →  decode/verify (even after a JPEG share)
#
# Run it:   bash examples/core_quickstart.sh
#   keep:   KEEP=1 bash examples/core_quickstart.sh   # leaves the files in ./core_demo_out
# Needs:    pip install mememage   (Pillow included). Add [encrypt] for the bonus step.
#
set -euo pipefail

# Use the installed `mememage` console script if present, else the module.
if command -v mememage >/dev/null 2>&1; then MM=(mememage); else MM=(python3 -m mememage); fi

# Default: a temp dir, wiped on exit (no mess). KEEP=1: a real ./core_demo_out
# you can open afterward.
if [ "${KEEP:-}" = "1" ]; then
  WORK="$(pwd)/core_demo_out"; rm -rf "$WORK"; mkdir -p "$WORK"
else
  WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
fi
IMG="$WORK/photo.png"; REC="$WORK/photo.json"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()  { printf '   \033[1;32m✓ %s\033[0m\n' "$*"; }
die() { printf '   \033[1;31m✗ %s\033[0m\n' "$*"; exit 1; }

say "0. A programmer has an image (any PNG — here we make one)"
python3 - "$IMG" <<'PY'
import sys
from PIL import Image
Image.new("RGB", (1280, 720), (40, 90, 120)).save(sys.argv[1])
PY
ok "image: $IMG"

say "1. ENCODE — write the bar + the record, with the fields I choose"
"${MM[@]}" encode "$IMG" \
  --field prompt="a quiet river at dawn" \
  --field author="catmemes" \
  --field license="CC-BY-4.0" \
  -o "$REC"
echo "   --- $REC ---"; cat "$REC"; echo
[ -s "$REC" ] || die "no record written"
ok "encoded"

say "2. DECODE (bare) — pull the bar straight out of the pixels (no record, no network)"
"${MM[@]}" decode "$IMG"

say "3. DECODE — verify the image against its record (exit 0 == match)"
"${MM[@]}" decode "$IMG" --record "$REC"
ok "VERIFIED"

say "4. TAMPER — change one field in the record; the hash MUST break"
python3 - "$REC" "$WORK/tampered.json" <<'PY'
import sys, json
d = json.load(open(sys.argv[1])); d["license"] = "MIT"
json.dump(d, open(sys.argv[2], "w"))
PY
if "${MM[@]}" decode "$IMG" --record "$WORK/tampered.json"; then
  die "tamper NOT detected"
else
  ok "tamper detected (ALTERED, non-zero exit)"
fi

say "5. IN THE WILD — the bar survives a JPEG re-encode (a share / screenshot)"
python3 - "$IMG" "$WORK/photo.jpg" <<'PY'
import sys
from PIL import Image
Image.open(sys.argv[1]).save(sys.argv[2], "JPEG", quality=70)
PY
"${MM[@]}" decode "$WORK/photo.jpg" --record "$REC"
ok "verified from a JPEG copy"

say "6. SCRIPTING — machine-readable JSON for a CI gate / pipeline"
"${MM[@]}" decode "$IMG" --record "$REC" --json
ok "JSON emitted"

say "7. (bonus) ENCRYPT — lock a private field behind a password (verifies without it)"
if python3 -c "from mememage import crypto; raise SystemExit(0 if crypto.is_encryption_available() else 1)" 2>/dev/null; then
  SECRET="$WORK/secret.png"; cp "$IMG" "$SECRET"
  MM_PW=hunter2 "${MM[@]}" encode "$SECRET" --field title="public" --field gps="45.5,-122.6" \
    --private gps --password-env MM_PW -o "$WORK/secret.json" >/dev/null
  "${MM[@]}" decode "$SECRET" --record "$WORK/secret.json" | grep -q ENCRYPTED \
    && ok "field encrypted — record still VERIFIED without the password" || die "encryption sample failed"
  MM_PW=hunter2 "${MM[@]}" decode "$SECRET" --record "$WORK/secret.json" --unlock --password-env MM_PW \
    | grep -q "45.5" && ok "unlocked with the password" || die "unlock failed"
else
  echo "   (no cryptography — pip install \"mememage[encrypt]\" to try field encryption)"
fi

printf '\n\033[1;32m================  CORE VALIDATED  ================\033[0m\n'
echo "encode → store → decode → tamper-detect → survive-JPEG → (encrypt): all green."

if [ "${KEEP:-}" = "1" ]; then
  printf '\nFiles kept in \033[1m%s\033[0m:\n' "$WORK"
  ls -1 "$WORK"
  echo "  open the encoded image:  $IMG"
  echo "  read the record:         cat \"$REC\""
fi

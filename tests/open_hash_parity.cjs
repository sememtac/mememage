// Parity harness for the "open" hash version. Loads the real verify.js
// hashing engine under Node and prints computeContentHash() for each record in
// the JSON array at argv[2], so the pytest wrapper can assert the browser hash
// equals Python's core.compute_content_hash for the same records.
//
// verify.js is a browser script (no module.exports); the hash path only needs
// crypto.subtle + TextEncoder, both Node globals. We load it into an isolated
// vm context and append an export hook.
//
// Run: node tests/open_hash_parity.cjs <records.json>   (prints a JSON array)
const fs = require('fs');
const vm = require('vm');
const path = require('path');

// The core repo ships the verifier at the root (../verify.js); the development
// tree keeps it under docs/js/. Try the root first, fall back.
const _candidates = [
  path.join(__dirname, '..', 'verify.js'),
  path.join(__dirname, '..', 'docs', 'js', 'verify.js'),
];
const VERIFY_JS = _candidates.find((p) => fs.existsSync(p)) || _candidates[1];
const src = fs.readFileSync(VERIFY_JS, 'utf8');
const sandbox = { crypto, TextEncoder, console, atob: globalThis.atob, btoa: globalThis.btoa };
vm.createContext(sandbox);
vm.runInContext(
  src + '\n;globalThis.__api = { computeContentHash, _hashableFields };',
  sandbox
);
const { computeContentHash } = sandbox.__api;

(async () => {
  const records = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const out = [];
  for (const rec of records) out.push(await computeContentHash(rec));
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });

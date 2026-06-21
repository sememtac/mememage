/* Mememage core — browser verifier.
 *
 * Recompute a record's content hash (the "open" model) and check it against the
 * hash carried in an image's bar. Pure integrity: no keys, no network, no
 * signatures. Matches mememage/hashing.py byte-for-byte — enforced by
 * tests/open_hash_parity.cjs (Python hash === this hash for the same record).
 *
 *   const ok = await verify(record, barContentHash);   // true iff data intact
 *
 * Drop it in any page (no build step) or load it under Node. SHA-256 uses
 * crypto.subtle when available and falls back to a pure-JS implementation
 * (iOS Safari on self-signed HTTPS, file://, etc.).
 */

var OPEN_HASH_VERSION = 'open';
var HASH_EXCLUDED_OPEN = { content_hash: 1, signature: 1 };

// The fields the content hash covers under the open model: every field except
// the structurally-circular pair (content_hash, signature) and `_`-prefixed keys
// (reserved for decoder internals, never hashed). Mirrors hashing.open_hashable_fields.
function _hashableFields(record) {
  var out = {};
  Object.keys(record).filter(function (k) {
    return !HASH_EXCLUDED_OPEN[k] && k.charAt(0) !== '_';
  }).sort().forEach(function (k) { out[k] = record[k]; });
  return out;
}

function sortKeysDeep(obj) {
  if (Array.isArray(obj)) return obj.map(sortKeysDeep);
  if (obj !== null && typeof obj === 'object') {
    var s = {};
    Object.keys(obj).sort().forEach(function (k) { s[k] = sortKeysDeep(obj[k]); });
    return s;
  }
  return obj;
}

var _SHA256_K = new Uint32Array([
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
]);

function _sha256_js(bytes) {
  var H = new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);
  var bitLen = bytes.length * 8;
  var padLen = (bytes.length + 9 + 63) & ~63;
  var padded = new Uint8Array(padLen);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  padded[padLen - 4] = (bitLen >>> 24) & 0xff;
  padded[padLen - 3] = (bitLen >>> 16) & 0xff;
  padded[padLen - 2] = (bitLen >>> 8)  & 0xff;
  padded[padLen - 1] = bitLen & 0xff;
  var W = new Uint32Array(64);
  for (var block = 0; block < padLen; block += 64) {
    for (var i = 0; i < 16; i++) {
      W[i] = (padded[block + i*4] << 24) | (padded[block + i*4 + 1] << 16) | (padded[block + i*4 + 2] << 8) | padded[block + i*4 + 3];
    }
    for (var i = 16; i < 64; i++) {
      var s0 = ((W[i-15] >>> 7) | (W[i-15] << 25)) ^ ((W[i-15] >>> 18) | (W[i-15] << 14)) ^ (W[i-15] >>> 3);
      var s1 = ((W[i-2] >>> 17) | (W[i-2] << 15)) ^ ((W[i-2] >>> 19) | (W[i-2] << 13)) ^ (W[i-2] >>> 10);
      W[i] = (W[i-16] + s0 + W[i-7] + s1) >>> 0;
    }
    var a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (var i = 0; i < 64; i++) {
      var S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      var ch = (e & f) ^ (~e & g);
      var t1 = (h + S1 + ch + _SHA256_K[i] + W[i]) >>> 0;
      var S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      var mj = (a & b) ^ (a & c) ^ (b & c);
      var t2 = (S0 + mj) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    H[0] = (H[0]+a)>>>0; H[1] = (H[1]+b)>>>0; H[2] = (H[2]+c)>>>0; H[3] = (H[3]+d)>>>0;
    H[4] = (H[4]+e)>>>0; H[5] = (H[5]+f)>>>0; H[6] = (H[6]+g)>>>0; H[7] = (H[7]+h)>>>0;
  }
  var out = new Uint8Array(32);
  for (var i = 0; i < 8; i++) {
    out[i*4]     = (H[i] >>> 24) & 0xff;
    out[i*4 + 1] = (H[i] >>> 16) & 0xff;
    out[i*4 + 2] = (H[i] >>> 8)  & 0xff;
    out[i*4 + 3] = H[i] & 0xff;
  }
  return out;
}

async function _sha256_bytes(input) {
  if (typeof crypto !== 'undefined' && crypto.subtle && typeof crypto.subtle.digest === 'function') {
    try { return new Uint8Array(await crypto.subtle.digest('SHA-256', input)); }
    catch (e) { /* fall through */ }
  }
  return _sha256_js(input);
}

async function sha256_16(obj) {
  var sorted = sortKeysDeep(obj);
  // Canonical JSON: sorted keys, no whitespace, ASCII-escaped (matches Python's
  // json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=True)).
  var canonical = JSON.stringify(sorted).replace(/[\u0080-\uffff]/g, function (c) {
    return '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0');
  });
  var hash = await _sha256_bytes(new TextEncoder().encode(canonical));
  return Array.from(hash).map(function (b) { return b.toString(16).padStart(2, '0'); }).join('').slice(0, 16);
}

// Recompute a record's content hash under the open model (16 hex chars).
async function computeContentHash(record) {
  try { return await sha256_16(_hashableFields(record)); }
  catch (e) { return null; }
}

// Verify a record against the content hash read from an image's bar.
// Returns true iff the data is intact (the re-hash matches).
async function verify(record, barContentHash) {
  var h = await computeContentHash(record);
  return h !== null && h === barContentHash;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { computeContentHash: computeContentHash, verify: verify, _hashableFields: _hashableFields };
}

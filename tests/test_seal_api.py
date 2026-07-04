"""The core encode / decode / verify API — the raw Mememage surface.

encode stamps a bar + builds an open-hash record from arbitrary fields; decode
reads the bar back out (identifier + content hash); verify checks a record against
an image. This pins the contract plus the cross-engine promise: a record encoded
in Python matches in the real browser verify.js engine (so a verifier anywhere
validates it).
"""

import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import pytest

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import mememage

HERE = os.path.dirname(os.path.abspath(__file__))
CJS = os.path.join(HERE, "open_hash_parity.cjs")
_NODE = shutil.which("node")


def _png(w=1024, h=576):
    p = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    Image.new("RGB", (w, h), (40, 90, 120)).save(p)
    return p


@unittest.skipUnless(HAS_PIL, "Pillow required")
class TestCoreApi(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "a quiet river", "by": "andy"})
        self.assertEqual(rec.record["hash_version"], "open")
        self.assertTrue(rec.identifier.startswith("mememage-"))
        bar = mememage.decode(img)
        self.assertEqual(bar.identifier, rec.identifier)
        self.assertEqual(bar.content_hash, rec.content_hash)

    def test_verify_match(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "p", "license": "CC0"})
        v = mememage.verify(img, rec)
        self.assertTrue(v)                      # truthy == match
        self.assertTrue(v.match)
        self.assertEqual(v.reason, "")          # no reason on success

    def test_verify_detects_field_tamper(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "p", "license": "CC0"})
        bad = copy.deepcopy(rec.record)
        bad["license"] = "MIT"
        v = mememage.verify(img, bad)
        self.assertFalse(v)
        self.assertFalse(v.match)
        self.assertIn("hash mismatch", v.reason)

    def test_verify_no_bar(self):
        clean = _png()  # never encoded
        v = mememage.verify(clean, {"identifier": "x", "content_hash": "y",
                                    "hash_version": "open"})
        self.assertFalse(v)
        self.assertIn("no Mememage bar", v.reason)

    def test_record_save_load_roundtrip(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "store me anywhere"})
        path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        rec.save(path)
        loaded = json.load(open(path, encoding="utf-8"))
        self.assertTrue(mememage.verify(img, loaded))

    def test_content_addressing_is_deterministic(self):
        a = mememage.encode(_png(), {"prompt": "x", "by": "andy"})
        b = mememage.encode(_png(), {"prompt": "x", "by": "andy"})
        self.assertEqual(a.identifier, b.identifier)
        c = mememage.encode(_png(), {"prompt": "x", "by": "ANDY"})
        self.assertNotEqual(a.identifier, c.identifier)

    def test_custom_identifier_and_prefix(self):
        img = _png()
        rec = mememage.encode(img, {"a": 1}, prefix="phoenix")
        self.assertTrue(rec.identifier.startswith("phoenix-"))
        img2 = _png()
        rec2 = mememage.encode(img2, {"a": 1}, identifier="custom-deadbeefcafe0001")
        self.assertEqual(rec2.identifier, "custom-deadbeefcafe0001")
        self.assertTrue(mememage.verify(img2, rec2))

    def test_reserved_keys_rejected(self):
        for k in ("identifier", "content_hash", "hash_version", "signature",
                  "encrypted_fields"):
            with self.assertRaises(ValueError):
                mememage.encode(_png(), {k: "x"})

    def test_underscore_keys_rejected(self):
        # `_`-prefixed keys are reserved for decoder internals (not hashed) —
        # encode refuses them so nothing silently goes unprotected.
        with self.assertRaises(ValueError):
            mememage.encode(_png(), {"_source": "x"})

    def test_non_png_input_writes_png(self):
        # Any image Pillow opens is accepted; a non-PNG input yields a lossless
        # PNG output (the original is left untouched).
        jp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        Image.new("RGB", (800, 600), (40, 90, 120)).save(jp, "JPEG")
        r = mememage.encode(jp, {"a": 1})
        self.assertTrue(r.image_path.endswith(".png"))
        self.assertTrue(os.path.exists(r.image_path))
        self.assertTrue(os.path.exists(jp))                 # original untouched
        self.assertTrue(mememage.verify(r.image_path, r.record))

    def test_lossy_out_rejected(self):
        with self.assertRaises(ValueError):
            mememage.encode(_png(), {"a": 1}, out="result.jpg")

    def test_encode_in_memory(self):
        # PIL in -> barred PIL out, no disk; out=BytesIO -> bytes (Pillow idiom).
        import io
        src = Image.new("RGB", (1024, 576), (40, 90, 120))
        r = mememage.encode(src, {"t": "x"})
        self.assertIsNone(r.image_path)                  # no disk write
        self.assertIsInstance(r.image, Image.Image)      # barred image in memory
        self.assertTrue(mememage.verify(r.image, r.record))
        self.assertIsNone(mememage.decode(src))            # caller's image untouched
        buf = io.BytesIO()
        r2 = mememage.encode(src, {"x": 1}, out=buf)
        data = buf.getvalue()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(mememage.decode(data).content_hash, r2.content_hash)

    def test_bad_prefix_rejected(self):
        for p in ("ab", "x" * 11, "1abc", "-abc", "ab/cd"):
            with self.assertRaises(ValueError):
                mememage.encode(_png(), {"a": 1}, prefix=p)

    def test_empty_fields_ok(self):
        img = _png()
        rec = mememage.encode(img, None)
        self.assertTrue(mememage.verify(img, rec))

    def test_reading_accepts_in_memory_images(self):
        # decode/verify take pixels, not just a path — no disk round-trip.
        import io
        img = _png()
        r = mememage.encode(img, {"t": "x"})
        cid = r.content_hash
        pim = Image.open(img)                                # PIL Image
        self.assertEqual(mememage.decode(pim).content_hash, cid)
        self.assertTrue(mememage.verify(pim, r.record))
        raw = open(img, "rb").read()                         # raw bytes
        self.assertEqual(mememage.decode(raw).content_hash, cid)
        self.assertTrue(mememage.verify(raw, r.record))
        self.assertEqual(mememage.decode(io.BytesIO(raw)).content_hash, cid)  # file-like
        try:                                                 # numpy array of pixels
            import numpy as np
            arr = np.array(Image.open(img))
            self.assertEqual(mememage.decode(arr).content_hash, cid)
            self.assertTrue(mememage.verify(arr, r.record))
        except ImportError:
            pass


@unittest.skipUnless(HAS_PIL, "Pillow required")
class TestDecode(unittest.TestCase):
    """decode() — read the bar's payload (identifier + content hash). The inverse
    of encode. Resolving + verifying the record is the caller's, via verify()."""

    def test_decode_returns_bar(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "p", "by": "andy"})
        bar = mememage.decode(img)
        self.assertEqual(bar.identifier, rec.identifier)
        self.assertEqual(bar.content_hash, rec.content_hash)

    def test_decode_no_bar(self):
        self.assertIsNone(mememage.decode(_png()))   # never encoded → None

    def test_resolve_and_verify_compose(self):
        # The old all-in-one decomposes into: decode -> resolve (your store) -> verify.
        img = _png()
        rec = mememage.encode(img, {"prompt": "river"})
        store = {rec.identifier: rec.record}         # the caller's storage
        bar = mememage.decode(img)
        record = store.get(bar.identifier)
        self.assertIsNotNone(record)
        self.assertTrue(mememage.verify(img, record))

    def test_decode_all_returns_every_bar(self):
        # An image can carry more than one bar in any placement. decode(all_bars=True)
        # returns every one; plain decode() returns the first (bottom-most).
        from PIL import Image
        from mememage import bar as _bar
        canvas = Image.new("RGB", (900, 560), (40, 90, 120))
        canvas.paste(_bar.embed_into(Image.new("RGB", (900, 60), (60, 110, 90)),
                     "mememage-aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa"), (0, 500))    # bottom, full width
        canvas.paste(_bar.embed_into(Image.new("RGB", (600, 60), (110, 70, 130)),
                     "mememage-bbbbbbbbbbbbbbbb", "bbbbbbbbbbbbbbbb"), (170, 180))  # offset + higher
        bars = mememage.decode(canvas, all_bars=True)
        self.assertEqual(sorted(b.identifier for b in bars),
                         ["mememage-aaaaaaaaaaaaaaaa", "mememage-bbbbbbbbbbbbbbbb"])
        self.assertEqual(mememage.decode(canvas).identifier, "mememage-aaaaaaaaaaaaaaaa")

    def test_decode_all_empty_when_no_bar(self):
        self.assertEqual(mememage.decode(_png(), all_bars=True), [])


try:
    from mememage import crypto
    HAS_CRYPTO = crypto.is_encryption_available()
except Exception:
    HAS_CRYPTO = False


@unittest.skipUnless(HAS_PIL, "Pillow required")
@unittest.skipUnless(HAS_CRYPTO, "cryptography required")
class TestEncryption(unittest.TestCase):
    PW = "correct horse battery staple"

    def test_password_encrypts_all_fields(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "secret", "gps": "1,2"}, password=self.PW)
        # private fields leave the cleartext; only the envelope remains
        self.assertIn("encrypted_fields", rec.record)
        self.assertNotIn("prompt", rec.record)
        self.assertNotIn("gps", rec.record)
        self.assertTrue(mememage.is_encrypted(rec))

    def test_encrypted_matches_without_password(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "secret"}, password=self.PW)
        self.assertTrue(mememage.verify(img, rec.record))   # hash over the shell

    def test_unlock_reveals(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "secret", "n": 7}, password=self.PW)
        view = mememage.unlock(rec, self.PW)
        self.assertEqual(view["prompt"], "secret")
        self.assertEqual(view["n"], 7)
        self.assertNotIn("encrypted_fields", view)

    def test_unlock_wrong_password(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "secret"}, password=self.PW)
        with self.assertRaises(ValueError):
            mememage.unlock(rec, "nope")

    def test_partial_private(self):
        img = _png()
        rec = mememage.encode(img, {"title": "pub", "diary": "priv"},
                              password=self.PW, private=["diary"])
        self.assertIn("title", rec.record)            # public
        self.assertNotIn("diary", rec.record)         # encrypted
        self.assertEqual(mememage.unlock(rec, self.PW)["diary"], "priv")

    def test_verify_then_unlock(self):
        img = _png()
        rec = mememage.encode(img, {"prompt": "secret"}, password=self.PW)
        # the encrypted record verifies (the hash covers the ciphertext)
        self.assertTrue(mememage.verify(img, rec.record))
        self.assertIn("encrypted_fields", rec.record)
        self.assertNotIn("prompt", rec.record)
        # unlock reveals the private fields with the password
        self.assertEqual(mememage.unlock(rec.record, self.PW)["prompt"], "secret")

    def test_ciphertext_tamper_breaks_match(self):
        import copy
        img = _png()
        rec = mememage.encode(img, {"prompt": "secret"}, password=self.PW)
        bad = copy.deepcopy(rec.record)
        bad["encrypted_fields"]["ct"] = "00" + bad["encrypted_fields"]["ct"][2:]
        self.assertFalse(mememage.verify(img, bad))

    def test_private_without_password_errors(self):
        with self.assertRaises(ValueError):
            mememage.encode(_png(), {"a": 1}, private=["a"])


@pytest.mark.skipif(_NODE is None, reason="node not installed")
@unittest.skipUnless(HAS_PIL, "Pillow required")
class TestVerifiesInBrowser(unittest.TestCase):
    """A Python-encoded record must match in the REAL verify.js engine — the
    'verify anywhere' promise. We hash the record with the browser engine and
    assert it equals the content_hash stamped in the bar."""

    def test_browser_hash_matches_bar(self):
        records = [
            mememage.encode(_png(), {"prompt": "a cat", "by": "andy"}).record,
            mememage.encode(_png(), {"tags": ["x", "y"], "n": 3, "f": 0.5,
                                     "nested": {"k": "v"}}).record,
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
            path = f.name
        try:
            res = subprocess.run([_NODE, CJS, path], capture_output=True,
                                 text=True, timeout=30)
            self.assertEqual(res.returncode, 0, res.stderr)
            js_hashes = json.loads(res.stdout)
        finally:
            os.unlink(path)
        for rec, jsh in zip(records, js_hashes):
            self.assertEqual(
                jsh, rec["content_hash"],
                "browser would NOT match a Python-encoded record")


if __name__ == "__main__":
    unittest.main()

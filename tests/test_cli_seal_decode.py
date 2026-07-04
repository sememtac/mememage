"""The `mememage encode` / `mememage decode` CLI — the core API from a shell.

Drives the real CLI via subprocess (``python -m mememage ...``) and checks the
human + JSON output and, crucially, the exit codes — `decode` exits 0 iff the
data matches the image, so it drops straight into a shell pipeline or CI gate.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _run(*args, **kw):
    return subprocess.run([sys.executable, "-m", "mememage", *args],
                          capture_output=True, text=True, **kw)


def _png():
    p = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    Image.new("RGB", (1024, 576), (40, 90, 120)).save(p)
    return p


def _record_path(stdout):
    """The soul path `mememage encode` wrote — named {identifier}.soul, so we
    read it from the output rather than guessing from the image name."""
    for line in stdout.splitlines():
        if "record:" in line:
            return line.split("record:", 1)[1].strip()
    return None


@unittest.skipUnless(HAS_PIL, "Pillow required")
class TestEncodeDecodeCli(unittest.TestCase):
    def test_encode_then_decode_verified(self):
        img = _png()
        r = _run("encode", img, "--field", "prompt=a river", "--field", "by=andy")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("identifier:", r.stdout)
        rec = _record_path(r.stdout)
        # Record is named by the IDENTIFIER, not the image name (.json for the
        # core; .soul is reserved for the provenance chain).
        self.assertTrue(os.path.basename(rec).startswith("mememage-"))
        self.assertTrue(rec.endswith(".json"))
        self.assertTrue(os.path.exists(rec))

        d = _run("decode", img, "--record", rec)
        self.assertEqual(d.returncode, 0, d.stderr)        # match → exit 0
        self.assertIn("VERIFIED", d.stdout)

    def test_encode_fields_json_and_typed_values(self):
        img = _png()
        out = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        fields = json.dumps({"tags": ["a", "b"], "n": 7, "flag": True})
        r = _run("encode", img, "--fields", "-", "-o", out, input=fields)
        self.assertEqual(r.returncode, 0, r.stderr)
        rec = json.load(open(out, encoding="utf-8"))
        self.assertEqual(rec["tags"], ["a", "b"])
        self.assertEqual(rec["n"], 7)           # typed (int), not "7"
        self.assertIs(rec["flag"], True)
        self.assertEqual(rec["hash_version"], "open")

    def test_decode_readonly_prints_bar(self):
        img = _png()
        _run("encode", img, "--field", "a=1")
        d = _run("decode", img)                  # no --record → just the bar
        self.assertEqual(d.returncode, 0, d.stderr)
        self.assertIn("Bar:", d.stdout)
        self.assertIn("Hash:", d.stdout)

    def test_decode_json_output(self):
        img = _png()
        r = _run("encode", img, "--field", "prompt=p")
        rec = _record_path(r.stdout)
        d = _run("decode", img, "--record", rec, "--json")
        self.assertEqual(d.returncode, 0, d.stderr)
        obj = json.loads(d.stdout)
        self.assertTrue(obj["match"])
        self.assertTrue(obj["identifier"].startswith("mememage-"))

    def test_decode_tampered_exits_nonzero(self):
        img = _png()
        r = _run("encode", img, "--field", "prompt=p")
        rec_path = _record_path(r.stdout)
        rec = json.load(open(rec_path, encoding="utf-8"))
        rec["prompt"] = "TAMPERED"
        bad = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(rec, bad)
        bad.close()
        d = _run("decode", img, "--record", bad.name)
        self.assertEqual(d.returncode, 1)        # ALTERED → exit 1
        self.assertIn("ALTERED", d.stdout)

    def test_encode_non_png_writes_png(self):
        jp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        Image.new("RGB", (800, 600), (40, 90, 120)).save(jp, "JPEG")
        r = _run("encode", jp, "--field", "a=1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(os.path.splitext(jp)[0] + ".png"))

    def test_encode_reserved_key_errors(self):
        r = _run("encode", _png(), "--field", "content_hash=x")
        self.assertNotEqual(r.returncode, 0)

    def test_encode_encrypt_and_decode_unlock(self):
        try:
            from mememage import crypto
            if not crypto.is_encryption_available():
                self.skipTest("cryptography not available")
        except Exception:
            self.skipTest("crypto unavailable")
        img = _png()
        env = dict(os.environ, MM_PW="swordfish")
        r = _run("encode", img, "--field", "title=pub", "--field", "gps=1,2",
                 "--private", "gps", "--password-env", "MM_PW", env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        rec = _record_path(r.stdout)
        obj = json.load(open(rec, encoding="utf-8"))
        self.assertIn("title", obj)
        self.assertNotIn("gps", obj)
        self.assertIn("encrypted_fields", obj)

        # decode without password → VERIFIED + ENCRYPTED, exit 0
        d = _run("decode", img, "--record", rec)
        self.assertEqual(d.returncode, 0, d.stderr)
        self.assertIn("VERIFIED", d.stdout)
        self.assertIn("ENCRYPTED", d.stdout)

        # decode --unlock --password-env → reveals
        u = _run("decode", img, "--record", rec, "--unlock",
                 "--password-env", "MM_PW", env=env)
        self.assertEqual(u.returncode, 0, u.stderr)
        self.assertIn("UNLOCKED", u.stdout)
        self.assertIn("1,2", u.stdout)            # gps revealed


if __name__ == "__main__":
    unittest.main()

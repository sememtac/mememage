"""Centered-brightness bar (dom +/- delta/2) + Otsu/absolute dual-threshold decode.

The bar tints its two bit levels around the image's own dominant brightness
instead of pinning them to absolute 64/192, so it sinks toward the image on dark
backgrounds. The decoder recovers the per-image threshold by Otsu, and still
reads legacy absolute bars via the always-present 128 candidate.
"""
import struct
import unittest

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from mememage import bar

ID, HASH = "mememage-0123456789abcdef", "fedcba9876543210"


def _dark(w=1024, h=1024, level=24):
    """A near-uniform dark image (worst case for bar glare)."""
    img = Image.new("RGB", (w, h), (level - 4, level, level + 6))
    return img


def _bar_brightness_range(img):
    a, b = bar._HEADER_PIXELS, img.size[0] - bar._FOOTER_PIXELS
    px = img.load()
    vals = [sum(px[x, img.size[1] - 1][:3]) / 3 for x in range(a, b)]
    return min(vals), max(vals)


def _embed_absolute(img, identifier, content_hash):
    """Mint a LEGACY absolute-64/192 bar (the pre-centered-scheme encoding),
    to prove the decoder's backward-compat path."""
    img = img.convert("RGB")
    w, h = img.size
    payload = f"{identifier}\x00{content_hash}".encode("utf-8")
    codeword = bar.rs_encode(payload, bar._RS_NSYM)
    crc = bar._crc16(codeword)
    frame = (bar._FRAME_MAGIC + struct.pack("B", bar._FRAME_GEN)
             + struct.pack("B", bar._RS_NSYM) + struct.pack(">H", len(payload))
             + struct.pack(">H", crc) + codeword)
    bits = [(byte >> p) & 1 for byte in frame for p in range(7, -1, -1)]
    dr, dg, db = bar._dominant_color(img)
    dom = (dr + dg + db) / 3

    def bit_rgb(bit, x=None):
        off = (192 if bit else 64) - dom
        return tuple(max(0, min(255, round(c + off))) for c in (dr, dg, db))

    data_width = w - bar._HEADER_PIXELS - bar._FOOTER_PIXELS
    if data_width >= bar._PIXELS_PER_BIT_WIDE * len(bits):
        bar._write_even_fill(img, w, h, bits, bit_rgb)
    else:
        bar._write_sequential(img, w, h, data_width, bits, bit_rgb, dr, dg, db, payload)
    return img


@unittest.skipUnless(HAS_PIL, "Pillow required")
class TestCenteredBar(unittest.TestCase):
    def test_centered_is_quiet_on_dark(self):
        b = bar.embed_into(_dark(level=24), ID, HASH)
        lo, hi = _bar_brightness_range(b)
        # The bright bit-1 pixels must land far below the old absolute 192 —
        # bounded by the dominant brightness plus delta/2 (~24 + 48).
        self.assertLess(hi, 130, f"bar too bright on dark image (hi={hi})")

    def test_centered_roundtrips(self):
        b = bar.embed_into(_dark(), ID, HASH)
        self.assertEqual(bar.extract_bar(b), (ID, HASH))

    def test_centered_survives_jpeg_q50(self):
        import io
        b = bar.embed_into(_dark(level=20), ID, HASH).convert("RGB")
        buf = io.BytesIO()
        b.save(buf, "JPEG", quality=50)
        buf.seek(0)
        self.assertEqual(bar.extract_bar(Image.open(buf)), (ID, HASH))

    def test_centered_works_across_brightness(self):
        for level in (10, 40, 90, 160, 230):
            b = bar.embed_into(_dark(level=level), ID, HASH)
            self.assertEqual(bar.extract_bar(b), (ID, HASH), f"failed at level {level}")

    def test_legacy_absolute_bar_still_decodes(self):
        # A bar minted by the old absolute scheme must still decode (the 128
        # threshold candidate), including after JPEG.
        import io
        b = _embed_absolute(_dark(level=24), ID, HASH)
        self.assertEqual(bar.extract_bar(b), (ID, HASH))
        buf = io.BytesIO()
        b.convert("RGB").save(buf, "JPEG", quality=50)
        buf.seek(0)
        self.assertEqual(bar.extract_bar(Image.open(buf)), (ID, HASH))


if __name__ == "__main__":
    unittest.main()

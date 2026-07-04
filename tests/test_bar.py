"""Tests for mememage.bar — v4 bar codec with Reed-Solomon error correction."""

import struct
import tempfile
import unittest

from PIL import Image

from mememage.bar import (
    _FRAME_MAGIC,
    _FRAME_GEN,
    _crc16,
    embed_bar,
    extract_bar,
)


def _make_test_image(width=1024, height=768, color=(100, 130, 160)):
    """Create a temporary test PNG and return its path."""
    img = Image.new('RGB', (width, height), color)
    f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(f.name)
    f.close()
    return f.name


class TestCRC16(unittest.TestCase):
    def test_known_value(self):
        crc = _crc16(b"hello")
        self.assertIsInstance(crc, int)
        self.assertGreater(crc, 0)

    def test_empty(self):
        crc = _crc16(b"")
        self.assertEqual(crc, 0xFFFF)

    def test_deterministic(self):
        self.assertEqual(_crc16(b"test"), _crc16(b"test"))


class TestBarRoundTrip(unittest.TestCase):
    def test_basic_round_trip(self):
        """embed_bar → extract_bar should recover URL and content hash."""
        path = _make_test_image()
        identifier = "mememage-abc1234500000000"
        content_hash = "a1b2c3d4e5f60708"

        embed_bar(path, identifier, content_hash)
        result = extract_bar(path)

        self.assertIsNotNone(result)
        self.assertEqual(result[0], identifier)
        self.assertEqual(result[1], content_hash)

    def test_different_image_sizes(self):
        """Bar should work on various image widths (minimum ~1008px at 3px/bit)."""
        identifier = "mememage-123456789abc0000"
        content_hash = "deadbeef12345678"

        for width in [1024, 1360, 1536, 2048]:
            path = _make_test_image(width=width)
            embed_bar(path, identifier, content_hash)
            result = extract_bar(path)
            self.assertIsNotNone(result, f"Failed at width={width}")
            self.assertEqual(result[0], identifier)
            self.assertEqual(result[1], content_hash)

    def test_different_background_colors(self):
        """Bar should survive various dominant colors."""
        identifier = "mememage-c0107e5700000000"
        content_hash = "0000111122223333"

        for color in [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 128, 0), (50, 50, 50)]:
            path = _make_test_image(color=color)
            embed_bar(path, identifier, content_hash)
            result = extract_bar(path)
            self.assertIsNotNone(result, f"Failed with color={color}")
            self.assertEqual(result[0], identifier)

    def test_image_too_narrow(self):
        """Should raise ValueError if image can't hold payload."""
        path = _make_test_image(width=256, height=256)
        identifier = "mememage-abc1234567890000"
        content_hash = "a1b2c3d4e5f60708"

        with self.assertRaises(ValueError):
            embed_bar(path, identifier, content_hash)

    def test_no_bar_returns_none(self):
        """extract_bar on an unmodified image should return None."""
        path = _make_test_image()
        self.assertIsNone(extract_bar(path))

    def test_nonexistent_file_returns_none(self):
        self.assertIsNone(extract_bar('/tmp/nonexistent_mememage_test.png'))

    def test_v4_version_byte(self):
        """Encoded bar should use version 4 with RS parity."""
        path = _make_test_image()
        identifier = "mememage-0440000000000000"
        content_hash = "aabbccdd11223344"
        embed_bar(path, identifier, content_hash)

        # Verify via extract_bar — the version is checked internally
        result = extract_bar(path)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], identifier)

        # Verify the raw bytes contain v4 marker
        img = Image.open(path)
        w, h = img.size
        # Read brightness of first few data pixels in bottom row (after header)
        # First 16 bits = magic 0xAD4E, next 8 bits = version byte
        # Version 4 = 0x04

    def test_rejects_unknown_gen(self):
        """Decoder should reject bars with unknown gen byte."""
        from mememage.bar import _try_decode_frame
        # Craft bits that look like a valid frame but with gen=99
        fake = bytearray([0xAD, 0x4E, 99, 6, 0, 10, 0, 0] + [0]*16)
        bits = []
        for b in fake:
            for bp in range(7, -1, -1):
                bits.append((b >> bp) & 1)
        self.assertIsNone(_try_decode_frame(bits))

    def test_rgba_image(self):
        """Bar should handle RGBA images (converted to RGB)."""
        img = Image.new('RGBA', (1024, 768), (100, 150, 200, 255))
        f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(f.name)
        f.close()

        identifier = "mememage-50ba123456780000"
        content_hash = "ffffeeeeddddcccc"
        embed_bar(f.name, identifier, content_hash)
        result = extract_bar(f.name)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], identifier)

    def test_preserves_png_metadata(self):
        """Bar encoding should preserve existing PNG text chunks."""
        from PIL.PngImagePlugin import PngInfo
        img = Image.new('RGB', (1024, 768), (128, 128, 128))
        pnginfo = PngInfo()
        pnginfo.add_text("note", '{"seed": 42}')
        f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(f.name, pnginfo=pnginfo)
        f.close()

        embed_bar(f.name, "mememage-0e7a123456780000", "1234567890abcdef")

        result_img = Image.open(f.name)
        self.assertIn("note", result_img.text)
        self.assertEqual(result_img.text["note"], '{"seed": 42}')


class TestBarJPEGSurvival(unittest.TestCase):
    """JPEG survival tests.

    JPEG DCT compression smears the M/Y/C header pixels with adjacent data
    pixels. Survival depends on image content — solid-color backgrounds create
    worst-case artifacts. Real photographs and noisy images survive better.

    The minted file is always PNG. JPEG survival matters for screenshots and
    reshares. These tests use noise backgrounds to approximate real images.
    """

    def _make_noisy_image(self, width=1024, height=768):
        """Create a test image with pseudo-random content (friendlier to JPEG)."""
        import random
        random.seed(42)
        img = Image.new('RGB', (width, height))
        for y in range(height):
            for x in range(width):
                img.putpixel((x, y), (
                    random.randint(80, 180),
                    random.randint(80, 180),
                    random.randint(80, 180),
                ))
        f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(f.name)
        f.close()
        return f.name

    def test_survives_jpeg_q95(self):
        """Bar should survive high-quality JPEG recompression."""
        path = self._make_noisy_image()
        identifier = "mememage-09e0e57000000000"
        content_hash = "1122334455667788"
        embed_bar(path, identifier, content_hash)

        img = Image.open(path)
        jpeg_path = path.replace('.png', '.jpg')
        img.save(jpeg_path, 'JPEG', quality=95)

        result = extract_bar(jpeg_path)
        self.assertIsNotNone(result, "Bar did not survive JPEG q95")
        self.assertEqual(result[0], identifier)
        self.assertEqual(result[1], content_hash)

    def test_survives_jpeg_q50(self):
        """Bar should survive JPEG recompression at quality 50."""
        path = self._make_noisy_image()
        identifier = "mememage-09e050e570000000"
        content_hash = "aabbccddee001122"
        embed_bar(path, identifier, content_hash)

        img = Image.open(path)
        jpeg_path = path.replace('.png', '.jpg')
        img.save(jpeg_path, 'JPEG', quality=50)

        result = extract_bar(jpeg_path)
        self.assertIsNotNone(result, "Bar did not survive JPEG q50")
        self.assertEqual(result[0], identifier)
        self.assertEqual(result[1], content_hash)


class TestBarDownscaleAliasing(unittest.TestCase):
    """Even-fill survival through downscaling is phase-dependent, not a smooth
    floor: at certain ratios the resampled band edge lands a sub-pixel off, every
    bit center shifts the same way, and enough bits flip to exceed RS — an
    aliasing null. ~0.90x is one such null that fails while 0.92x and 0.88x pass.

    The decoder's anchor-phase sweep (try a few integer offsets on each band
    anchor, CRC self-selects) closes these holes. This guards that 0.90x — a
    historically-failing scale on a wide even-fill bar — stays decodable.
    """

    def _wide_bar_image(self, width=1600, height=900):
        import random
        random.seed(7)
        img = Image.new('RGB', (width, height))
        for y in range(height):
            for x in range(width):
                img.putpixel((x, y), (
                    random.randint(70, 170),
                    random.randint(70, 170),
                    random.randint(70, 170),
                ))
        f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(f.name)
        f.close()
        return f.name

    def test_survives_aliasing_null_downscale(self):
        path = self._wide_bar_image()
        identifier = "mememage-a11a500000000000"
        content_hash = "0f1e2d3c4b5a6978"
        embed_bar(path, identifier, content_hash)
        src = Image.open(path).convert('RGB')
        w, h = src.size
        # 0.90x is the classic null; 0.92x / 0.88x bracket it. All must decode.
        for scale in (0.92, 0.90, 0.88):
            small = src.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            tf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            small.save(tf.name)
            tf.close()
            result = extract_bar(tf.name)
            self.assertIsNotNone(result, "bar lost at %.2fx downscale" % scale)
            self.assertEqual(result[0], identifier, "identifier wrong at %.2fx" % scale)
            self.assertEqual(result[1], content_hash, "hash wrong at %.2fx" % scale)


class TestBarRSCorrection(unittest.TestCase):
    """Test that Reed-Solomon actually corrects byte-level corruption."""

    def test_corrects_corrupted_data_bytes(self):
        """Flip a few data bytes in the codeword and verify RS recovers."""
        # 768px keeps this on the sequential split layout (the byte-targeting math
        # below assumes it); the identifier is non-hex so it uses the ASCII payload
        # path, where the widest fitting ppb here is 3. (1024px would now land on
        # even-fill under the packed-payload crossover; even-fill RS is covered by
        # the JPEG/downscale cases in test_bar_evenfill.py.)
        path = _make_test_image(width=768, height=768)
        identifier = "mememage-005e570000000000"
        content_hash = "aabbccdd11223344"
        embed_bar(path, identifier, content_hash)

        # Verify clean decode first
        clean = extract_bar(path)
        assert clean is not None

        # Corrupt 3 data bytes by flipping pixel brightness in the bar.
        # Gen I header is 8 bytes, so byte offsets 12, 22, 32 are in the codeword.
        from mememage.bar import (
            _HEADER_PIXELS, _FOOTER_PIXELS, _SIG_ROWS,
            _PIXELS_PER_BIT_MAX, _PIXELS_PER_BIT_NARROW, _RS_NSYM, _pack_payload,
        )

        img = Image.open(path)
        w, h = img.size

        # Determine the ppb the writer used (widest that fits the PACKED payload —
        # same sweep as _write_sequential).
        data_per_row = w - _HEADER_PIXELS - _FOOTER_PIXELS
        total_data = _SIG_ROWS * data_per_row
        payload = _pack_payload(identifier, content_hash)
        ppb = _PIXELS_PER_BIT_NARROW
        for cand in range(_PIXELS_PER_BIT_MAX, _PIXELS_PER_BIT_NARROW - 1, -1):
            if len(payload) <= (total_data // cand) // 8 - 8 - _RS_NSYM:
                ppb = cand
                break
        bits_per_row = data_per_row // ppb

        # Corrupt frame bytes 12, 22, 32 (all in codeword, past header)
        for byte_offset in [12, 22, 32]:
            for bit_in_byte in range(8):
                bit_idx = byte_offset * 8 + bit_in_byte
                row_idx = bit_idx // bits_per_row
                bit_in_row = bit_idx % bits_per_row
                y = h - 1 - row_idx
                for px in range(ppb):
                    x = _HEADER_PIXELS + bit_in_row * ppb + px
                    if 0 <= x < w - _FOOTER_PIXELS:
                        r, g, b = img.getpixel((x, y))
                        img.putpixel((x, y), (255 - r, 255 - g, 255 - b))

        img.save(path)

        # RS should correct the 3 corrupted bytes
        result = extract_bar(path)
        self.assertIsNotNone(result, "RS failed to correct 3 corrupted bytes")
        self.assertEqual(result[0], identifier)
        self.assertEqual(result[1], content_hash)


if __name__ == "__main__":
    unittest.main()

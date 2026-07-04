"""Decoder vertical-scan + multi-bar + the embed height guard.

The encoder always writes the bar into the bottom 2 rows; the decoder reads it
there (fast path) and falls back to a vertical scan so an image still decodes if
the bar was relocated or content was appended below it after minting. The asym
camo needs one reference row above the 2 data rows, so a 3px image is the floor
(now enforced, symmetric to the width check).
"""
import io
import unittest

from PIL import Image

from mememage.bar import embed_into, extract_bar, extract_bars

ID, H16 = "mememage-aa8194d91f1da238", "47f11bad5dcc9ad2"


def _content(w, h, s=0):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 3 + s) % 256, (y * 5 + s) % 256, ((x + y) * 2 + s) % 256)
    return img


def _moved(barred, pad=40):
    """Append `pad` content rows below a barred image — bottom-addition pushes
    the bar (and its preserved asym reference row) up off the bottom."""
    w, h = barred.size
    out = Image.new("RGB", (w, h + pad))
    out.paste(barred, (0, 0))
    out.paste(_content(w, pad, s=99), (0, h))
    return out


class TestVerticalScan(unittest.TestCase):
    def test_bottom_fast_path(self):
        self.assertEqual(extract_bar(embed_into(_content(480, 300), ID, H16)), (ID, H16))

    def test_scan_off_does_not_find_moved_bar(self):
        moved = _moved(embed_into(_content(480, 300), ID, H16))
        self.assertIsNone(extract_bar(moved, scan=False))

    def test_scan_finds_moved_bar_sequential(self):
        moved = _moved(embed_into(_content(480, 300), ID, H16))
        self.assertEqual(extract_bar(moved), (ID, H16))

    def test_scan_finds_moved_bar_even_fill(self):
        moved = _moved(embed_into(_content(1216, 300), ID, H16))  # >984px crossover
        self.assertEqual(extract_bar(moved), (ID, H16))

    def test_scan_survives_jpeg(self):
        moved = _moved(embed_into(_content(480, 300), ID, H16))
        buf = io.BytesIO()
        moved.save(buf, format="JPEG", quality=80)
        self.assertEqual(extract_bar(Image.open(io.BytesIO(buf.getvalue()))), (ID, H16))

    def test_multi_bar(self):
        id2, ch2 = "mememage-deadbeefcafe1234", "0011223344556677"
        moved = _moved(embed_into(_content(480, 300), ID, H16))
        two = embed_into(moved, id2, ch2)  # second bar at the new bottom
        self.assertEqual(sorted(extract_bars(two)), sorted([(ID, H16), (id2, ch2)]))

    def test_no_bar_returns_none(self):
        self.assertIsNone(extract_bar(_content(480, 300)))
        self.assertEqual(extract_bars(_content(480, 300)), [])


def _grad(w, h):
    """A smooth gradient background — realistic content, few spurious bands."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 200) // w + 30, (y * 200) // h + 30, 150)
    return img


def _inset(barred, left, top, right, bottom, bg=_grad):
    """Composite a barred image into a larger canvas with content margins — the
    bar is now neither bottom-anchored nor full-width."""
    w, h = barred.size
    out = bg(w + left + right, h + top + bottom)
    out.paste(barred, (left, top))
    return out


class TestFullCanvasScan(unittest.TestCase):
    """The M/Y/C↔C/Y/M bands are a "data begins/ends here" fiducial: the
    decoder finds the bar ANYWHERE on the canvas — any row AND any horizontal
    offset/width — so a bar whose canvas was extended (margins) or that was
    pasted into a larger image still decodes. CRC + RS reject false matches."""

    def test_horizontal_margins(self):
        b = embed_into(_content(480, 300), ID, H16)
        self.assertEqual(extract_bar(_inset(b, 120, 0, 90, 0)), (ID, H16))

    def test_inset_all_sides(self):
        b = embed_into(_content(480, 300), ID, H16)
        self.assertEqual(extract_bar(_inset(b, 120, 80, 90, 60)), (ID, H16))

    def test_even_fill_inset(self):
        b = embed_into(_content(1216, 300), ID, H16)   # >984px crossover → even-fill
        self.assertEqual(extract_bar(_inset(b, 100, 50, 100, 50)), (ID, H16))

    def test_pasted_off_center(self):
        b = embed_into(_content(480, 300), ID, H16)
        self.assertEqual(extract_bar(_inset(b, 137, 211, 200, 90)), (ID, H16))

    def test_inset_survives_jpeg_q80(self):
        b = embed_into(_content(480, 300), ID, H16)
        buf = io.BytesIO()
        _inset(b, 120, 80, 90, 60).save(buf, format="JPEG", quality=80)
        self.assertEqual(extract_bar(Image.open(io.BytesIO(buf.getvalue()))), (ID, H16))

    def test_noisy_content_margins(self):
        # Harsher: pasted into linear-ramp content whose margins carry wide
        # spurious colour runs — the multi-candidate span search + CRC/RS still
        # locks onto the real bar.
        b = embed_into(_content(480, 300), ID, H16)
        self.assertEqual(extract_bar(_inset(b, 120, 60, 90, 40, bg=_content)), (ID, H16))

    def test_no_false_positive(self):
        self.assertIsNone(extract_bar(_grad(1000, 700)))
        self.assertIsNone(extract_bar(_content(1000, 700, s=7)))

    def test_multi_bar_mixed_placement(self):
        # One image, three DISTINCT bars in the three placements at once:
        #   A — correct position: bottom, full width
        #   B — different height:  mid-canvas, full width (vertical scan)
        #   C — elsewhere:         horizontally offset, narrower (full-canvas scan)
        # extract_bars unions the edge-anchored + full-canvas scans, so it
        # returns all three; extract_bar returns the bottom-most (A).
        A = ("mememage-aaaa111122223333", "aaaa111122223333")
        B = ("mememage-bbbb444455556666", "bbbb444455556666")
        C = ("mememage-cccc777788889999", "cccc777788889999")
        canvas = _grad(900, 600)
        canvas.paste(embed_into(_content(900, 60, s=1), *A), (0, 540))    # bar row 599
        canvas.paste(embed_into(_content(900, 60, s=2), *B), (0, 200))    # bar row 259
        canvas.paste(embed_into(_content(560, 60, s=3), *C), (180, 360))  # bar row 419, x180..739
        self.assertEqual(set(extract_bars(canvas)), {A, B, C})
        self.assertEqual(extract_bar(canvas), A)

    def test_multi_bar_survives_jpeg(self):
        # Two full-width bars (bottom + a different height) through JPEG q85.
        # Full-width bars keep their bands on the clean image edges, so both
        # survive re-encoding. (An OFF-position bar bordered by content is the
        # weaker case under heavy JPEG — see the module notes; not asserted.)
        A = ("mememage-1234123412341234", "1234123412341234")
        B = ("mememage-9876987698769876", "9876987698769876")
        canvas = _grad(900, 520)
        canvas.paste(embed_into(_content(900, 60, s=4), *A), (0, 460))   # bottom, full width
        canvas.paste(embed_into(_content(900, 60, s=5), *B), (0, 160))   # different height, full width
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=85)
        found = set(extract_bars(Image.open(io.BytesIO(buf.getvalue()))))
        self.assertIn(A, found)
        self.assertIn(B, found)


class TestHeightGuard(unittest.TestCase):
    def test_below_minimum_raises(self):
        for h in (1, 2):
            with self.assertRaises(ValueError) as cm:
                embed_into(_content(480, h), ID, H16)
            self.assertIn("3px", str(cm.exception))

    def test_minimum_height_round_trips(self):
        self.assertEqual(extract_bar(embed_into(_content(480, 3), ID, H16)), (ID, H16))


if __name__ == "__main__":
    unittest.main()

"""Bar codec: high-res even-fill layout + legacy sequential fallback.

Covers the width-adaptive redundancy redesign:
  - even-fill (2px-tall, both-ends-anchored) kicks in above the crossover,
  - sequential split stays byte-identical below it,
  - round-trips survive downscale, JPEG, combined, and a 1px bottom crop,
  - legacy bars and small images still decode,
  - the full identifier-prefix length range works at 512px.
"""
import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image
import numpy as np

from mememage import bar

IDENT = "mememage-a3f8c2d1e5b6f0a4"
HASH = "0123456789abcdef"
EXPECT = (IDENT, HASH)


def _img(w, h, seed=3):
    yy, xx = np.mgrid[0:h, 0:w]
    r = (120 + 80 * np.sin(xx / 600.0) + yy * 40 // h).clip(0, 255)
    g = (140 + 60 * np.sin(xx / 900.0 + 1) + yy * 60 // h).clip(0, 255)
    b = (180 + 50 * np.cos(xx / 500.0)).clip(0, 255)
    arr = np.stack([r, g, b], -1).astype(np.uint8)
    rng = np.random.default_rng(seed)
    arr = np.clip(arr.astype(np.int16) + rng.integers(-12, 12, arr.shape, dtype=np.int16),
                  0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _embed(tmp_path, w, h, ident=IDENT, name="x.png"):
    p = tmp_path / name
    _img(w, h).save(p)
    bar.embed_bar(str(p), ident, HASH)
    return p


def _save_jpeg(img, path, q):
    """Write a single clean JPEG at quality q (no double-compression)."""
    img.convert("RGB").save(str(path), "JPEG", quality=q)
    return path


# --- mode selection -------------------------------------------------------

def test_highres_uses_even_fill(tmp_path):
    p = _embed(tmp_path, 4096, 2304)
    img = Image.open(p).convert("RGB")
    # both rows identical in the even-fill layout
    w, h = img.size
    row0 = [img.getpixel((x, h - 1)) for x in range(0, w, 37)]
    row1 = [img.getpixel((x, h - 2)) for x in range(0, w, 37)]
    assert row0 == row1
    assert bar.extract_bar(str(p)) == EXPECT


def test_smallimage_uses_sequential(tmp_path):
    # 512px is below the crossover -> sequential split (rows differ)
    p = _embed(tmp_path, 512, 512)
    img = Image.open(p).convert("RGB")
    w, h = img.size
    row0 = [img.getpixel((x, h - 1)) for x in range(0, w, 7)]
    row1 = [img.getpixel((x, h - 2)) for x in range(0, w, 7)]
    assert row0 != row1  # sequential halves differ
    assert bar.extract_bar(str(p)) == EXPECT


# --- clean round-trips across widths -------------------------------------

@pytest.mark.parametrize("w,h", [
    (512, 512), (640, 640), (768, 1024), (1024, 1024), (1280, 720),
    (1392, 1392), (1600, 900), (2048, 2048), (2560, 1440), (3072, 1728),
    (4096, 2304), (8192, 4608),
])
def test_clean_roundtrip(tmp_path, w, h):
    p = _embed(tmp_path, w, h)
    assert bar.extract_bar(str(p)) == EXPECT


# --- prefix length range at the 512 floor --------------------------------

@pytest.mark.parametrize("prefix", ["mmm", "abc", "mememage", "phoenix", "tencharsxx"])
def test_512_prefix_lengths(tmp_path, prefix):
    ident = f"{prefix}-a3f8c2d1e5b6f0a4"
    p = _embed(tmp_path, 512, 512, ident=ident, name=f"p_{prefix}.png")
    assert bar.extract_bar(str(p)) == (ident, HASH)


def test_512_overlong_prefix_raises(tmp_path):
    # 16-char prefix (50B payload) exceeds 512px capacity -> must raise
    ident = "sixteencharsxxxx-a3f8c2d1e5b6f0a4"
    p = tmp_path / "big.png"
    _img(512, 512).save(p)
    with pytest.raises(ValueError):
        bar.embed_bar(str(p), ident, HASH)


# --- downscale resilience (even-fill, high res) --------------------------

# Asym camo (default since 2026-06-23) trades aggressive CLEAN-downscale
# resilience for a near-invisible data strip: its per-column "1" level is
# re-predicted from the row above the bar, and that prediction aliases under
# heavy lossless resampling. Reliable clean-downscale floor is ~0.9x (was ~0.4x
# for the centered scheme). The REALISTIC path — downscale + JPEG, which is how
# every platform actually shrinks — is unaffected (test_downscale_plus_jpeg
# survives 0.66x + q50). Lifting the clean-downscale floor is a known future bar
# improvement; see memory project_bar_data_camouflage.
@pytest.mark.parametrize("scale", [0.95, 0.9])
def test_downscale_clean(tmp_path, scale):
    p = _embed(tmp_path, 4096, 2304)
    img = Image.open(p).convert("RGB")
    nw, nh = round(4096 * scale), round(2304 * scale)
    small = img.resize((nw, nh), Image.LANCZOS)
    sp = tmp_path / "s.png"
    small.save(sp)
    assert bar.extract_bar(str(sp)) == EXPECT


# --- JPEG resilience ------------------------------------------------------

@pytest.mark.parametrize("q", [90, 70, 50])
def test_jpeg(tmp_path, q):
    p = _embed(tmp_path, 4096, 2304)
    img = Image.open(p).convert("RGB")
    jp = tmp_path / "j.jpg"
    _save_jpeg(img, jp, q)
    assert bar.extract_bar(str(jp)) == EXPECT


def test_downscale_plus_jpeg(tmp_path):
    # combined 0.66x + q50 is inside the envelope; assert it survives
    p = _embed(tmp_path, 4096, 2304)
    img = Image.open(p).convert("RGB")
    small = img.resize((round(4096 * 0.66), round(2304 * 0.66)), Image.LANCZOS)
    jp = tmp_path / "sj.jpg"
    _save_jpeg(small, jp, 50)
    assert bar.extract_bar(str(jp)) == EXPECT


# --- 0.5x downscale + JPEG (soft-anchor regression pins) ------------------

# Regression pins for the soft ordering predicates, NOT a universal 0.5x
# promise. At 0.5x the 8px bands fall to 4px and JPEG 4:2:0 chroma
# subsampling defeats the strict absolute cutoffs, so _find_header_end
# returned None and 0.5x was unconditionally dead — these cases decode only
# because of the soft fallback. Whether any *given* image survives 0.5x is
# content-dependent (bit margin, a separate limit): a real-image corpus put
# even-fill survival at 59/60 by 0.8x but far lower at 0.5x. The bar module
# docstring states the honest envelope; these tests guard the anchor half.

@pytest.mark.parametrize("q", [80, 70])
def test_half_scale_plus_jpeg(tmp_path, q):
    p = _embed(tmp_path, 2048, 1434)
    img = Image.open(p).convert("RGB")
    small = img.resize((1024, 717), Image.LANCZOS)
    jp = tmp_path / "half.jpg"
    _save_jpeg(small, jp, q)
    assert bar.extract_bar(str(jp)) == EXPECT


@pytest.mark.parametrize("resampler", [Image.BILINEAR, Image.BOX])
def test_half_scale_plus_jpeg_resamplers(tmp_path, resampler):
    # platforms differ in resampling kernel; the promise can't be
    # LANCZOS-specific
    p = _embed(tmp_path, 2048, 1434)
    img = Image.open(p).convert("RGB")
    small = img.resize((1024, 717), resampler)
    jp = tmp_path / "half_r.jpg"
    _save_jpeg(small, jp, 80)
    assert bar.extract_bar(str(jp)) == EXPECT


def test_soft_predicates_survive_chroma_dilution():
    """The mechanism pin: chroma-smeared band pixels that fail the strict
    absolute cutoffs (measured from a real 0.5x + q80 round-trip) still
    classify correctly by channel ordering."""
    # yellow band pixel after 0.5x + q80 4:2:0 — b crept to 122 (> 120)
    assert not bar._is_yellow(241, 233, 122)
    assert bar._is_yellow_soft(241, 233, 122)
    # cyan band pixel after the same round-trip — r crept to 127 (> 120)
    assert not bar._is_cyan(127, 202, 171)
    assert bar._is_cyan_soft(127, 202, 171)
    # soft stays selective: gray/white/skin-ish pixels match nothing
    for px in [(128, 128, 128), (250, 250, 250), (220, 180, 150)]:
        assert not (bar._is_magenta_soft(*px) or bar._is_yellow_soft(*px)
                    or bar._is_cyan_soft(*px))


# --- bottom 1px crop (vertical-redundancy payoff) ------------------------

def test_bottom_1px_crop(tmp_path):
    p = _embed(tmp_path, 4096, 2304)
    img = Image.open(p).convert("RGB")
    w, h = img.size
    cropped = img.crop((0, 0, w, h - 1))  # drop the bottom row
    cp = tmp_path / "c.png"
    cropped.save(cp)
    assert bar.extract_bar(str(cp)) == EXPECT


# --- no false positive on a bar-less image -------------------------------

def test_no_bar_returns_none(tmp_path):
    p = tmp_path / "plain.png"
    _img(2048, 2048).save(p)
    assert bar.extract_bar(str(p)) is None

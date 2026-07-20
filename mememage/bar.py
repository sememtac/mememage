"""Steganographic bar codec for minted images.

Encodes an identifier and content hash into a 2-pixel-tall bar at the
bottom of an image. Survives JPEG recompression, social media
re-encoding, and platform pipeline conversion (tested to q50 at typical
sizes; small images — where bits are only 2-3px wide — have a tighter
quality floor). Downscale is bounded by two mechanisms, neither of which
gives a clean cutoff — resist the urge to state one:

  * ANCHORS: the 8px M/Y/C bands shrink with the image. Once they approach
    ~4px, JPEG 4:2:0 chroma subsampling smears them past the strict colour
    cutoffs (that's what the soft ordering predicates rescue — see
    _BAND_PREDICATE_PASSES). On a LOSSLESS resize there is no chroma
    subsampling, so the bands keep classifying well below that.
  * BIT MARGIN (content-dependent): each bit needs roughly 3px after
    scaling — necessary but NOT sufficient. Saturated or high-frequency
    content near the bar loses the delta/2 margin under recompression and
    fails even with fat bits.

Because both depend on content, resampler, and whether the copy was
recompressed, downscale survival is NOT monotone in the scale factor
(measured: a 0.45x + JPEG copy failed where the same image at 0.4x
decoded). Do not write "nothing decodes below Nx" — it is false; a
lossless 0.35x copy of a 3072px render decodes.

What IS measured, on 16 real images x 3 resamplers x JPEG q70-q80:
even-fill images (>=~1000px wide) decode 60/60 at 0.9x and 59/60 at 0.8x.
That is the promise. At 0.7x it's 50/60 and at 0.5x 19/60 — the
soft-anchor fallback made half-size *reachable* (it was unconditionally
dead before), never guaranteed. Heavy double-reshares of a downscaled
copy are outside the envelope.

Frame format (Gen I):
    [2B magic 0xAD4E][1B gen=1][1B nsym][2B payload_len BE][2B CRC-16][RS(payload, nsym)]
    nsym = number of Reed-Solomon parity bytes (6 = corrects up to 3 byte errors).
    CRC-16 is computed over the full RS codeword (payload + parity).

Payload (packed binary):
    [1B prefix_len][prefix ASCII][8B identifier][8B content_hash]
    The identifier is canonical <prefix>-<16 hex> (prefix 3-10 chars); the
    first byte is the prefix length, so the decoder reads the prefix back
    directly. Source-agnostic: the identifier is a locator the decoder resolves
    through search, and the content hash verifies whatever is found.
    (_pack_payload / _parse_payload enforce this shape.)

Bar layout per row:
    [M×8][Y×8][C×8][data pixels...][C×8][Y×8][M×8]

Each data bit rides a PER-COLUMN level that copies the smoothed content one
row above the bar (asym row-3-copy camo): "1" IS that level (invisible), "0"
is _ASYM_DELTA below it. On dark content the level is lifted toward _ASYM_FLOOR
by a saturation-capped floor so the mark stays detectable without a colour pop.
The decoder re-predicts the same per-column threshold from the row above. The
8-pixel-wide M/Y/C color bands survive JPEG DCT blocks and bracket the data.

Because each bit is encoded RELATIVE to the content row above the bar, the bar
carries 3 rows of context, not 2: the 2 data rows PLUS that 1 reference row.
``embed_into`` requires an image at least 3px tall for this reason, and the
vertical-scan decoder crops top-down so the reference always rides along.

A 2-row crop therefore usually does NOT decode — its per-column reference is
gone. Usually, not always: the decoder falls back to Otsu and a fixed
threshold, which recover the bits without any reference whenever the
surrounding content is roughly uniform. Measured on the real decoder, a bare
2-row bar fails on textured content (gradients, photos) but decodes on flat,
dark, bright, or noisy content. So this is a property of the encoding, NOT a
security boundary: do not describe it as "a bare bar cannot be transplanted".
Detecting a bar moved onto a different image is the job of a portrait /
localized-tamper check (EMBODIED in the reference application), not of the
codec.

Two width-adaptive layouts share that frame format (the choice is
capacity-emergent — no flag, no version bump):

  * Even-fill (high res, when the whole frame fits in one row at >=3px/bit):
    the frame's bits are spread to EVENLY FILL the full width between the
    flush bilateral bands, painted identically in BOTH rows (2px tall). Fat
    bits => downscale resilience (a bit survives downscale fraction s while
    ppb*s >~ 2.5 dest px); the even fill => zero idle pixels; the 2px height
    => vertical redundancy (survives a 1px bottom crop, stronger under JPEG).
    Decode anchors to BOTH band edges and evenly divides — no scale factor,
    so positional drift cannot accumulate across the width.

  * Sequential (small images, below the crossover): the frame is split across
    the two rows at the WIDEST integer px/bit that fits, swept 6 down to 2
    (fatter is quieter and tougher under JPEG). The decoder estimates scale
    from band width and sweeps px/bit widest-first.

The decoder tries even-fill first, then the scale-swept sequential read; both
self-validate via CRC + Reed-Solomon, so the data selects the correct one. (The
public `extract_bar` adds a vertical-scan fallback that finds the bar at any
height — see its docstring; this module note describes the in-place read.)
"""

import struct

from mememage.rs import rs_encode, rs_decode


def _get_Image():
    """Lazy import Pillow so the package can be imported without it."""
    from PIL import Image
    # Register HEIC/HEIF support if available (Apple iMessage, iPhone photos)
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    return Image


def _resolve_image(image):
    """Resolve any in-memory or on-disk image to a PIL Image (read side).

    Accepts a path (str / os.PathLike), raw ``bytes``, a file-like object, a PIL
    ``Image`` (returned as-is), or a numpy array of pixels. Lets ``decode`` /
    ``verify`` work without a round-trip through disk.
    """
    Image = _get_Image()
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(image))
    if hasattr(image, "shape") and hasattr(image, "dtype"):   # numpy array (no numpy import)
        return Image.fromarray(image)
    return Image.open(image)        # path (str / os.PathLike) or file-like object

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRAME_MAGIC = b'\xAD\x4E'
_FRAME_GEN = 1                 # Gen I: 8px bands, RS error correction, adaptive ppb
_RS_NSYM = 6                   # RS parity bytes — corrects up to 3 byte errors

_SIG_ROWS = 2
# 8px per band == one JPEG/WebP DCT block. A band that fills a whole 8x8 block
# survives lossy compression as (near) pure M/Y/C; a narrower band shares its
# block with data pixels, so the codec averages the color away. The decoder also
# bails below 3px (see _detect_bar) and measures band width to recover the
# scale factor after a resize — both need the width. Shrinking this trades the
# bar's JPEG/downscale survival for a few bytes of capacity; don't.
_HEADER_BAND = 8              # pixels per color band in header/footer
_HEADER_PIXELS = 3 * _HEADER_BAND  # total header width (24px)
_FOOTER_PIXELS = 3 * _HEADER_BAND  # total footer width (24px)
_HEADER_COLORS = [(255, 0, 255), (255, 255, 0), (0, 255, 255)]
_FOOTER_COLORS = [(0, 255, 255), (255, 255, 0), (255, 0, 255)]

_PIXELS_PER_BIT_WIDE = 3       # crossover ppb: even-fill triggers at data_width >= this * n_bits
_PIXELS_PER_BIT_NARROW = 2    # sweep floor — the narrowest px/bit tried; only
                              # very small images (~≤515px wide) land here
_PIXELS_PER_BIT_MAX = 6       # sequential picks the WIDEST ppb that fits (fatter = quieter +
                              # JPEG-tougher); the packed payload frees the room to widen.
_RGB_THRESHOLD = 128         # benign scalar default for the decode helpers' `threshold=` param
                             # (the asym decode always passes its per-column curve explicitly)
# --- Asym row-3-copy camouflage --------------------
# The data bits ride a PER-COLUMN center that copies the smoothed content one row
# ABOVE the bar (floored on dark): a "1" bit IS that level
# (so it reads as a continuation of the image — invisible), a "0" bit is darker
# by _ASYM_DELTA, and filler past the payload is "1" (invisible). The decoder
# never compares to the bar's own (asymmetric, biased) distribution — it
# RE-PREDICTS the per-column "1" level from the preserved row above and
# thresholds delta/2 below it. Robust (Discord-tier shares and multi-pass
# re-saves validated on real images; the quality floor is content- and
# size-dependent — roughly q45-q50 at typical sizes, tighter on small
# images) AND near-invisible.
_ASYM_DELTA = 40             # "0" sits this far below the per-column "1" level. The
                             # quieter<->tougher knob: 40 is Discord-safe; lower is quieter but
                             # loses margin under a heavy re-share.
_ASYM_FLOOR = 50             # min luma for a detectable "1" on dark content. The
                             # saturation-capped _hue_floor keeps the dark mark hue-neutral and
                             # JPEG-stable, so the floor can stay this low (dimmer bar); 45 is
                             # the cliff.
_FLOOR_SCALE_CAP = 2.0       # max multiplicative scale in _hue_floor before the
                             # lift goes additive (near-black -> hue-neutral, not saturated).
_ASYM_BOX_RADIUS = 34        # px; box-blur radius for the per-column center (encode + decode).
                             # A box filter (integer-sum / one division), NOT a Gaussian:
                             # math.exp diverges by 1 ULP between glibc and V8, which would
                             # break byte-exact writer parity (tests/bar_encode_parity.cjs).
                             # A box filter is IEEE-deterministic across runtimes. Radius 34
                             # (window 69) approximates the prior sigma-20 Gaussian support.



# ---------------------------------------------------------------------------
# CRC-16/CCITT-FALSE
# ---------------------------------------------------------------------------

def _crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# Asym camouflage helpers (per-column center from the row above the bar)
# ---------------------------------------------------------------------------

def _smooth1d(values, radius):
    """Pure-Python 1-D box blur (moving average) with edge padding.

    A box filter — integer/float sum over a fixed window divided once — is
    IEEE-deterministic across runtimes (Python and V8 produce bit-identical
    doubles given identical inputs), which a Gaussian is NOT (math.exp differs
    by 1 ULP glibc↔V8). That bit-identity is what lets the JS bar writer stay
    byte-for-byte equal to the Python writer (the parity test), so the asym
    camo center is re-derived identically on both sides.
    """
    n = len(values)
    if n == 0 or radius <= 0:
        return list(values)
    width = 2 * radius + 1
    out = [0.0] * n
    for i in range(n):
        acc = 0.0
        for k in range(-radius, radius + 1):
            idx = i + k
            idx = 0 if idx < 0 else (n - 1 if idx >= n else idx)
            acc += values[idx]
        out[i] = acc / width
    return out


def _hue_floor(r, g, b, floor):
    """Lift a dark colour toward luma ``floor`` for a detectable "1" bit, capping
    saturation so near-black tinted content doesn't pop.

    Plain multiplicative scaling (r*floor/L) preserves hue, but on near-black
    content with a faint tint the scale factor explodes (floor/L → large) and
    amplifies the tint into a SATURATED mark that (a) visibly pops and (b) FAILS a
    q80 share — its (R+G+B)/3 rides a chroma channel JPEG subsamples away. So the
    scale is CAPPED at _FLOOR_SCALE_CAP and the rest of the lift is additive
    (hue-neutral): moderately-dark colour keeps its hue (gentle multiplier, so it
    still camouflages by colour), near-black goes neutral (cap + additive, no pop).
    Pure arithmetic (×, +, min) — byte-exact across Python/V8 for JS parity.
    """
    L = 0.299 * r + 0.587 * g + 0.114 * b
    if L >= floor:
        return r, g, b
    s = min(floor / L, _FLOOR_SCALE_CAP) if L >= 2 else _FLOOR_SCALE_CAP
    r2, g2, b2 = r * s, g * s, b * s
    L2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
    if L2 < floor:
        k = floor - L2
        r2, g2, b2 = r2 + k, g2 + k, b2 + k
    return min(255.0, r2), min(255.0, g2), min(255.0, b2)


def _asym_levels_columns(img, w, h):
    """Per-column ("1" rgb, "0" rgb, decode threshold) for the asym scheme.

    One level = the smoothed, saturation-capped floored content (the "1"
    camouflage), "0" = that minus _ASYM_DELTA, threshold = the midpoint. The
    decoder re-derives the identical per-column curve from the preserved row above,
    so it never disagrees with the writer."""
    y = h - _SIG_ROWS - 1                  # row immediately above the 2 bar rows
    if y < 0:
        y = max(0, h - 1)
    rr = [0.0] * w; gg = [0.0] * w; bb = [0.0] * w
    pix = img.load()   # one accessor, not a getpixel (with its per-call load) per column
    for x in range(w):
        px = pix[x, y][:3]
        rr[x], gg[x], bb[x] = px[0], px[1], px[2]
    rr = _smooth1d(rr, _ASYM_BOX_RADIUS); gg = _smooth1d(gg, _ASYM_BOX_RADIUS); bb = _smooth1d(bb, _ASYM_BOX_RADIUS)
    d = _ASYM_DELTA
    one_rgb = []; zero_rgb = []; thr = []
    for x in range(w):
        r, g, b = rr[x], gg[x], bb[x]
        cr, cg, cb = _hue_floor(r, g, b, _ASYM_FLOOR)
        one_rgb.append((cr, cg, cb))
        zero_rgb.append((cr - d, cg - d, cb - d))
        thr.append((cr + cg + cb) / 3.0 - d / 2.0)
    return one_rgb, zero_rgb, thr


def _asym_threshold_curve(img):
    """Per-column decode threshold (midpoint between the "1" and "0" levels),
    re-derived from preserved content (the row above), not the bar."""
    w, h = img.size
    _, _, thr = _asym_levels_columns(img, w, h)
    return thr


def _thr(threshold, px):
    """Threshold at column ``px``. A scalar candidate returns as-is; the asym
    candidate is a per-column list (the row-3-predicted threshold curve)."""
    if isinstance(threshold, list):
        if 0 <= px < len(threshold):
            return threshold[px]
        return threshold[-1] if threshold else _RGB_THRESHOLD
    return threshold


# ---------------------------------------------------------------------------
# M/Y/C band color predicates (shared by detect + even-fill anchoring)
# ---------------------------------------------------------------------------

def _is_magenta(r, g, b):
    return r > 130 and g < 120 and b > 130

def _is_yellow(r, g, b):
    return r > 130 and g > 130 and b < 120

def _is_cyan(r, g, b):
    return r < 120 and g > 130 and b > 130


# SOFT variants — relative channel dominance instead of absolute cutoffs.
# JPEG 4:2:0 chroma subsampling smears a downscaled band's colour toward its
# neighbours: at 0.5x the 8px bands are 4px, only ~2px of chroma survive, and
# a pixel like (241, 233, 122) fails the strict yellow test (b < 120) by 2
# while still being unmistakably yellow BY ORDERING (r-b=119, g-b=111). The
# ordering margin survives dilution because subsampling shifts all of a
# pixel's chroma together — it can't reorder the channels until the band is
# blended nearly away. Used as a second-pass fallback by the band-edge
# finders (strict first, so pristine images never take this path); any false
# anchor it admits on band-coloured content is rejected downstream by the
# magic + CRC-16 + RS frame checks, so the fallback is a strict decode-side
# superset — same stance as the anchor-phase sweep.
_SOFT_DOMINANCE = 40

def _is_magenta_soft(r, g, b):
    return r - g > _SOFT_DOMINANCE and b - g > _SOFT_DOMINANCE

def _is_yellow_soft(r, g, b):
    return r - b > _SOFT_DOMINANCE and g - b > _SOFT_DOMINANCE

def _is_cyan_soft(r, g, b):
    return g - r > _SOFT_DOMINANCE and b - r > _SOFT_DOMINANCE


# ---------------------------------------------------------------------------
# Pixel writers (shared by both bar layouts)
# ---------------------------------------------------------------------------

def _paint_bands(img, w, y):
    """Header (M,Y,C) flush left, footer (C,Y,M) flush right, on row y."""
    for ci, color in enumerate(_HEADER_COLORS):
        for px in range(_HEADER_BAND):
            img.putpixel((ci * _HEADER_BAND + px, y), color)
    for ci, color in enumerate(_FOOTER_COLORS):
        for px in range(_HEADER_BAND):
            img.putpixel((w - _FOOTER_PIXELS + ci * _HEADER_BAND + px, y), color)


def _write_even_fill(img, w, h, bits, bit_rgb):
    """High-res layout: spread bits to EVENLY FILL the full width between the
    flush bilateral bands, painted identically in both rows (2px tall).

    The even fill leaves no idle pixels and makes each bit as fat as the width
    allows (downscale resilience). Anchoring decode to both band edges means no
    scale factor is needed, so positional drift cannot accumulate.
    """
    a = _HEADER_PIXELS
    b = w - _FOOTER_PIXELS
    span = b - a
    n = len(bits)
    for y in (h - 1, h - 2):
        _paint_bands(img, w, y)
        for i in range(n):
            x0 = a + round(i * span / n)
            x1 = a + round((i + 1) * span / n)
            for x in range(x0, x1):
                img.putpixel((x, y), bit_rgb(bits[i], x))


def _write_sequential(img, w, h, data_width, bits, bit_rgb, payload):
    """Small-image layout: split the frame sequentially across the two rows at the
    widest integer px/bit that fits (6 down to 2 for narrow images)."""
    total_data_pixels = _SIG_ROWS * data_width
    header_overhead = 8
    rs_overhead = _RS_NSYM

    # Pick the WIDEST px/bit that fits (fatter bits = quieter + JPEG-tougher). The
    # packed payload is what frees the room to widen past the old fixed 3. The
    # decoder sweeps the same candidates widest-first and CRC/RS self-selects.
    ppb = None
    for cand in range(_PIXELS_PER_BIT_MAX, _PIXELS_PER_BIT_NARROW - 1, -1):
        cap = (total_data_pixels // cand) // 8 - header_overhead - rs_overhead
        if len(payload) <= cap:
            ppb = cand
            break
    if ppb is None:
        cap_narrow = (total_data_pixels // _PIXELS_PER_BIT_NARROW) // 8 - header_overhead - rs_overhead
        raise ValueError(
            f"Bar payload too large ({len(payload)}B) for image width "
            f"({w}px, {cap_narrow}B capacity at {_PIXELS_PER_BIT_NARROW}px/bit)"
        )

    bits_per_row = data_width // ppb
    for row_offset in range(_SIG_ROWS):
        y = h - 1 - row_offset
        _paint_bands(img, w, y)
        row_bit_start = row_offset * bits_per_row
        for bit_idx_local in range(bits_per_row):
            bit_idx = row_bit_start + bit_idx_local
            base_x = _HEADER_PIXELS + bit_idx_local * ppb
            if bit_idx < len(bits):
                for px in range(ppb):
                    img.putpixel((base_x + px, y), bit_rgb(bits[bit_idx], base_x + px))
            else:
                # Filler past the payload = "1" (asym copies the row above =
                # invisible).
                for px in range(ppb):
                    if base_x + px < w - _FOOTER_PIXELS:
                        img.putpixel((base_x + px, y), bit_rgb(1, base_x + px))


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

_HEXSET = frozenset('0123456789abcdef')


def _pack_payload(identifier, content_hash):
    """Pack the bar payload to BINARY: ``[prefix_len][prefix][8 id-bytes][8 hash-bytes]``.

    Identifiers are canonical ``<prefix>-<16 hex>`` (prefix 3-10 chars) and the
    content hash is 16 hex — the only form Mememage stamps. The first byte is the
    prefix length, which the decoder reads back directly. A non-canonical
    identifier or hash is a programming error (``api.encode`` — and the
    reference application's mint pipeline — only produce canonical ones)
    and raises.
    """
    pre, sep, idhex = identifier.rpartition('-')
    if not (sep and 3 <= len(pre) <= 10 and len(idhex) == 16
            and _HEXSET.issuperset(idhex)
            and len(content_hash) == 16 and _HEXSET.issuperset(content_hash)):
        raise ValueError(
            f"bar identifier must be canonical <prefix>-<16 hex> + 16-hex hash; "
            f"got {identifier!r} / {content_hash!r}")
    # Length byte = UTF-8 BYTE count, not the char count. It must match
    # _parse_payload's byte-slice read AND codec.js's TextEncoder pack (which
    # already uses the byte length). Identical to len(pre) for ASCII (the only
    # charset the prefix validator allows today), so this is a no-op for every
    # current identifier. (If the charset ever widens, the byte count here
    # stays correct but _parse_payload's 3-10 length bound must widen too —
    # a multi-byte prefix can exceed 10 bytes at ≤10 chars.)
    pre_bytes = pre.encode('utf-8')
    return bytes([len(pre_bytes)]) + pre_bytes + bytes.fromhex(idhex) + bytes.fromhex(content_hash)


def embed_into(image, identifier, content_hash):
    """Write a Mememage bar into the bottom 2 rows of an image, IN MEMORY.

    Reads any image :func:`_resolve_image` accepts (path, bytes, file-like, PIL
    Image, numpy array) and returns a NEW barred RGB ``PIL.Image`` — no disk, and
    the caller's image is never mutated. The disk wrapper :func:`embed_bar` and
    the core ``encode`` both build on this.

    Args:
        image: the source image (any in-memory or on-disk form).
        identifier: canonical Mememage identifier, ``<prefix>-<16 hex>``
            (e.g. "mememage-a3f8c2d1e5b60718").
        content_hash: 16 hex char content hash.

    Raises:
        ValueError: If the image is too narrow for the payload, shorter
            than 3px, or the identifier/hash isn't canonical.
    """
    payload = _pack_payload(identifier, content_hash)

    # Fresh RGB copy — convert() always returns a new image, so the caller's
    # image is left untouched even when it's already RGB.
    img = _resolve_image(image).convert('RGB')
    w, h = img.size

    # The asym camo reads a reference row immediately above the 2 bar rows
    # (`h - _SIG_ROWS - 1`), so the bar needs at least one clean content row
    # above it — a 3px-tall image (1 reference + 2 data) is the floor. Below
    # that the reference clamps onto a bar row and the camo/decode degrade
    # silently; fail loud instead, symmetric to the width check in the writers.
    if h < _SIG_ROWS + 1:
        raise ValueError(
            f"Bar needs an image at least {_SIG_ROWS + 1}px tall "
            f"({_SIG_ROWS} data rows + 1 reference row); got {h}px"
        )

    # Build Gen I frame with Reed-Solomon error correction (frame format is
    # identical in both layouts below — only the pixel layout differs).
    codeword = rs_encode(payload, _RS_NSYM)  # payload + 6 parity bytes
    crc = _crc16(codeword)
    frame = (
        _FRAME_MAGIC
        + struct.pack('B', _FRAME_GEN)
        + struct.pack('B', _RS_NSYM)
        + struct.pack('>H', len(payload))
        + struct.pack('>H', crc)
        + codeword
    )

    # Convert to bits
    bits = []
    for byte in frame:
        for bit_pos in range(7, -1, -1):
            bits.append((byte >> bit_pos) & 1)

    data_width = w - _HEADER_PIXELS - _FOOTER_PIXELS

    # Asym camo applies to BOTH layouts. Its data pixels copy image content,
    # which can be M/Y/C-hued and would masquerade as the flush bands — but the
    # band-edge finders no longer measure the data-adjacent edge by running into
    # the data (they COMPUTE it from the data-free magenta/cyan span; see
    # _find_header_end), so content-coloured data can't fool the anchoring that
    # even-fill decode depends on. Sequential at 1:1 reads from fixed positions.
    is_even_fill = data_width >= _PIXELS_PER_BIT_WIDE * len(bits)

    # Asym camouflage: each bit rides PER-COLUMN levels derived from the smoothed
    # content one row above (see _asym_levels_columns). Filler past the payload =
    # "1". The decoder re-derives the same levels from the row above, so it never
    # disagrees with the writer. The data pixels copy image content, which can be
    # M/Y/C-hued and would masquerade as the flush bands — but the band-edge
    # finders COMPUTE the data-adjacent edge from the data-free magenta/cyan span
    # (see _find_header_end), so content-coloured data can't fool the anchoring.
    _one_rgb, _zero_rgb, _ = _asym_levels_columns(img, w, h)

    def _bit_rgb(bit, x):
        cr, cg, cb = _one_rgb[x] if bit else _zero_rgb[x]
        return (max(0, min(255, round(cr))),
                max(0, min(255, round(cg))),
                max(0, min(255, round(cb))))

    # Layout choice is capacity-emergent, no flag or version bump:
    #   - Above the crossover (the whole frame fits in ONE row at >=3px/bit),
    #     spread the bits to EVENLY FILL the full width between the flush bands,
    #     painted 2px tall (both rows identical): fat bits => downscale resilience,
    #     even fill => zero idle pixels, 2px => 1px-crop survival, both-end
    #     anchoring => drift-free decode.
    #   - Below the crossover (small images), the sequential split across the two
    #     rows at the widest px/bit that fits.
    if is_even_fill:
        _write_even_fill(img, w, h, bits, _bit_rgb)
    else:
        _write_sequential(img, w, h, data_width, bits, _bit_rgb, payload)

    return img


def embed_bar(image_path, identifier, content_hash):
    """Encode a bar into an image file (overwritten in place, PNG, chunks kept).

    Disk wrapper over :func:`embed_into` — preserves the source PNG's text
    metadata chunks. ``encode`` calls ``embed_into`` directly for its in-memory
    path.
    """
    img = embed_into(image_path, identifier, content_hash)

    # Preserve PNG metadata from the original on disk.
    from PIL.PngImagePlugin import PngInfo
    original = _get_Image().open(image_path)
    pnginfo = PngInfo()
    if hasattr(original, 'text'):
        for key, value in original.text.items():
            if key.startswith('XML:'):
                pnginfo.add_itxt(key, value)
            else:
                pnginfo.add_text(key, value)
    original.close()

    if not str(image_path).lower().endswith('.png'):
        raise ValueError(f"Bar encoding requires PNG format, got: {image_path}")
    img.save(image_path, pnginfo=pnginfo)


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------

def _detect_bar(img, y=None):
    """Check if row ``y`` has the M/Y/C header pattern.

    Scans for the magenta→yellow→cyan transition to detect presence
    and measure the band width (reveals scale factor if resized).
    ``y`` defaults to the bottom row (``h - 1``) — the embed position;
    the vertical scan passes explicit rows to find a relocated bar.

    Returns (magenta_width, yellow_width, cyan_width) or None.
    """
    w, h = img.size
    if h < _SIG_ROWS or w < 20:
        return None

    if y is None:
        y = h - 1

    pix = img.load()   # one accessor for the whole row walk

    def at(x):
        return pix[x, y][:3]

    # Scan for magenta run from left edge
    magenta_w = 0
    for x in range(min(20, w)):
        if _is_magenta(*at(x)):
            magenta_w += 1
        else:
            break
    if magenta_w < 3:
        return None

    # Skip transition zone (1-2 pixels of JPEG smear between bands)
    # then scan for yellow
    yellow_start = magenta_w
    for x in range(magenta_w, min(magenta_w + 3, w)):
        if _is_yellow(*at(x)):
            yellow_start = x
            break
    yellow_w = 0
    for x in range(yellow_start, min(yellow_start + 20, w)):
        if _is_yellow(*at(x)):
            yellow_w += 1
        else:
            break
    if yellow_w < 3:
        return None

    # Skip transition, then scan for cyan
    cyan_start = yellow_start + yellow_w
    for x in range(cyan_start, min(cyan_start + 3, w)):
        if _is_cyan(*at(x)):
            cyan_start = x
            break
    cyan_w = 0
    for x in range(cyan_start, min(cyan_start + 20, w)):
        if _is_cyan(*at(x)):
            cyan_w += 1
        else:
            break
    if cyan_w < 3:
        return None

    return magenta_w, yellow_w, cyan_w


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

# Even-fill frame byte-length sweep. Frame = 8B header + payload + 6B parity.
# Packed payload = prefix_len(1) + prefix(3-10) + 8B id + 8B hash = 20..27B, so a
# packed frame is 34..41B. The window keeps generous margin (33..64B) so CRC/RS
# self-select the true frame length; it is parity-locked to docs/js/data.js.
_EVENFILL_MIN_BYTES = 33
_EVENFILL_MAX_BYTES = 64


# The strict M/Y/C predicates first, then the soft (channel-ordering)
# fallback. Strict-first means a pristine or lightly-compressed image
# resolves exactly as before at zero added cost; the soft pass only runs
# when strict finds nothing — the chroma-subsampled-downscale regime
# (e.g. 0.5x + JPEG, where the 4px bands fail the absolute cutoffs but
# keep their channel ordering). A soft false-anchor on band-coloured
# content is harmless: the frame's magic + CRC-16 + RS checks reject it.
# NOTE: docs/js/codec.js mirrors these finders with strict predicates only —
# port the soft pass there at the next decoder unseal (decode-capability
# parity; the writer is untouched, so byte-exact writer parity is unaffected).
_BAND_PREDICATE_PASSES = (
    (_is_magenta, _is_yellow, _is_cyan),
    (_is_magenta_soft, _is_yellow_soft, _is_cyan_soft),
)


def _find_header_end(img, y, w):
    """Return x just past the header's M->Y->C run (where data starts), or None.

    The data-adjacent edge is COMPUTED, not measured by running the cyan count
    into the data: asym camo data pixels can be cyan-hued and would extend the
    run past the true edge. ``mag_start`` (the image's left edge) and
    ``cyan_start`` (bounded by yellow) never touch data, and the span between
    them is exactly two band widths — so band_width = (cyan_start-mag_start)/2
    and data_start = cyan_start + band_width. Small transition-skip tolerance
    absorbs JPEG smear between bands; the caller's phase sweep absorbs ±1-2px.
    Tries the strict predicates first, then the soft ordering fallback (see
    ``_BAND_PREDICATE_PASSES``).
    """
    for is_m, is_y, is_c in _BAND_PREDICATE_PASSES:
        r = _find_header_end_with(img, y, w, is_m, is_y, is_c)
        if r is not None:
            return r
    return None


def _find_header_end_with(img, y, w, is_m, is_y, is_c):
    pix = img.load()   # one accessor for the whole row walk

    def run(pred, x):
        n = 0
        while x < w and pred(*pix[x, y][:3]):
            x += 1
            n += 1
        return x, n

    x = 0
    while x < w and x < 40 and not is_m(*pix[x, y][:3]):
        x += 1
    mag_start = x
    x, nm = run(is_m, x)
    while x < w and x < 60 and not is_y(*pix[x, y][:3]):
        x += 1
    x, ny = run(is_y, x)
    while x < w and x < 80 and not is_c(*pix[x, y][:3]):
        x += 1
    cyan_start = x
    x, nc = run(is_c, x)
    if nm < 2 or ny < 2 or nc < 2:
        return None
    band_width = (cyan_start - mag_start) / 2.0
    return int(round(cyan_start + band_width))


def _find_footer_start(img, y, w):
    """Return x at the left edge of the footer's C/Y/M run (where data ends),
    or None. Scans inward from the right edge (footer order is C,Y,M, so from
    the right it reads M->Y->C).

    Data-adjacent edge is COMPUTED (see :func:`_find_header_end`): ``mag_start``
    (image right edge) and ``cyan_start`` (bounded by yellow) are data-free, so
    band_width = (mag_start-cyan_start)/2 and the footer's data-side edge is
    cyan_start - band_width + 1 (cyan_start is the rightmost cyan pixel).
    Tries the strict predicates first, then the soft ordering fallback (see
    ``_BAND_PREDICATE_PASSES``)."""
    for is_m, is_y, is_c in _BAND_PREDICATE_PASSES:
        r = _find_footer_start_with(img, y, w, is_m, is_y, is_c)
        if r is not None:
            return r
    return None


def _find_footer_start_with(img, y, w, is_m, is_y, is_c):
    pix = img.load()   # one accessor for the whole row walk

    def run(pred, x):
        n = 0
        while x >= 0 and pred(*pix[x, y][:3]):
            x -= 1
            n += 1
        return x, n

    x = w - 1
    while x >= 0 and x > w - 40 and not is_m(*pix[x, y][:3]):
        x -= 1
    mag_start = x
    x, nm = run(is_m, x)
    while x >= 0 and x > w - 60 and not is_y(*pix[x, y][:3]):
        x -= 1
    x, ny = run(is_y, x)
    while x >= 0 and x > w - 80 and not is_c(*pix[x, y][:3]):
        x -= 1
    cyan_start = x
    x, nc = run(is_c, x)
    if nm < 2 or ny < 2 or nc < 2:
        return None
    band_width = (mag_start - cyan_start) / 2.0
    return int(round(cyan_start - band_width)) + 1


def _otsu_threshold(img):
    """Per-image bimodal bit threshold — a scalar fallback for the asym curve.

    Otsu over the middle 60% of the bottom rows (avoids the M/Y/C bands and is
    scale-robust — no fixed band-pixel offsets), returned as the MIDPOINT of the
    two class means rather than the boundary index (robust on an exact bar, where
    the levels are delta-spikes). Rescues hard content (e.g. pure-saturated
    backgrounds) where the asym per-column threshold's per-channel clamp shrinks
    the (R+G+B)/3 margin. Returns None on a degenerate (flat) region.
    """
    try:
        w, h = img.size
        if w < 5 or h < 1:
            return None
        x0, x1 = int(w * 0.20), int(w * 0.80)
        if x1 <= x0:
            return None
        hist = [0] * 256
        total = 0
        pix = img.load()   # one accessor, not a getpixel per histogram sample
        for y in range(max(0, h - _SIG_ROWS), h):
            for x in range(x0, x1):
                r, g, b = pix[x, y][:3]
                hist[int((r + g + b) / 3.0) & 255] += 1
                total += 1
        if total == 0:
            return None
        sum_all = sum(i * hist[i] for i in range(256))
        sumB = wB = 0.0
        best = -1.0
        thr = None
        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += i * hist[i]
            mB, mF = sumB / wB, (sum_all - sumB) / wF
            var = wB * wF * (mB - mF) ** 2
            if var > best:
                best, thr = var, (mB + mF) / 2.0
        return thr
    except Exception:
        return None


def _decode_even_fill(img, threshold=_RGB_THRESHOLD):
    """Decode the high-res even-fill layout by anchoring to BOTH band edges.

    Finds where the header band ends (a) and the footer band starts (b), then
    reads bits by evenly dividing [a, b] — no scale factor, so no drift. Reads
    the two rows averaged (noise immunity) and, if that fails, the bottom row
    alone (survives a 1px bottom crop, where the row above is now image). The
    frame byte-length is swept; CRC self-selects.
    """
    w, h = img.size
    if h < 1 or w < 3 * _HEADER_PIXELS:
        return None
    y = h - 1
    a0 = _find_header_end(img, y, w)
    b0 = _find_footer_start(img, y, w)
    if a0 is None or b0 is None or b0 - a0 < 8:
        return None

    read_modes = [(h - 1, h - 2)] if h >= 2 else [(h - 1,)]
    if h >= 2:
        read_modes.append((h - 1,))  # bottom row only (bottom-crop survivor)

    # The anchor-offset x byte-length sweep below re-samples the SAME one or two
    # rows on every attempt — precompute each row's per-column luma once so the
    # sweep is pure arithmetic (same values, ~1000x fewer pixel reads).
    pix = img.load()
    luma = {}
    for rows in read_modes:
        for ry in rows:
            if ry not in luma:
                row = [0.0] * w
                for x in range(w):
                    r, g, bl = pix[x, ry][:3]
                    row[x] = (r + g + bl) / 3.0
                luma[ry] = row

    # Band-edge detection lands on an integer pixel, but after a downscale the
    # true sub-pixel edge can sit a pixel or two away. That shift moves every
    # bit center the same way, flipping enough bits to exceed RS at particular
    # scales — aliasing nulls (e.g. ~0.9x can fail while 0.92x and 0.88x pass).
    # Sweep a few integer phase offsets on each anchor and let CRC self-select.
    # (0, 0) is tried first, so a clean image returns on the first pass at zero
    # added cost and every previously-decodable image still decodes — this is a
    # strict superset of the single-anchor read.
    for da in (0, -1, 1, -2, 2):
        for db in (0, -1, 1, -2, 2):
            a, b = a0 + da, b0 + db
            span = b - a
            if span < 8:
                continue
            for n_bytes in range(_EVENFILL_MIN_BYTES, _EVENFILL_MAX_BYTES + 1):
                n = n_bytes * 8
                for rows in read_modes:
                    bits = []
                    ok = True
                    for i in range(n):
                        px = int(round(a + (i + 0.5) * span / n))
                        if px < 0 or px >= w:
                            ok = False
                            break
                        acc = 0.0
                        for ry in rows:
                            acc += luma[ry][px]
                        bits.append(1 if acc / len(rows) >= _thr(threshold, px) else 0)
                    if not ok:
                        continue
                    result = _try_decode_frame(bits)
                    if result is not None:
                        return result
    return None


def extract_bar(image, scan=True):
    """Extract identifier and content hash from a barred image.

    The encoder always writes the bar into the bottom 2 rows, so this reads
    that position first (the fast path). If ``scan`` is True and the bottom
    read fails, it falls back to a vertical scan that finds the bar wherever
    its M/Y/C band signature appears — so an image still decodes if the bar
    was relocated, or content was appended below it, AFTER minting. (The
    encoder never places the bar anywhere but the bottom; the scan only READS
    one that something else moved.) CRC + Reed-Solomon self-select, so a
    band-ish content row can't produce a false decode.

    Args:
        image: a path, raw bytes, a file-like object, a PIL Image, or a numpy
            array — anything :func:`_resolve_image` accepts (no disk required).
        scan: fall back to the vertical scan when the bottom read fails
            (default True). Pass False to read strictly the bottom 2 rows.

    Returns:
        (identifier, content_hash) tuple, or None if no valid bar found.
    """
    try:
        img = _resolve_image(image)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
    except Exception:
        return None

    result = _extract_at_bottom(img)
    if result is not None:
        return result
    if scan:
        for _, res in _scan_for_bars(img, first_only=True):
            return res
        # Last resort: the bar's canvas was extended (margins) or it was pasted
        # into a larger image, so it's neither bottom-anchored nor full-width —
        # find it by its band signature anywhere on the canvas.
        for _, res in _scan_anywhere(img, first_only=True):
            return res
    return None


def extract_bars(image):
    """Find ALL valid bars in an image via vertical scan, de-duplicated.

    The encoder emits a single bottom-anchored bar; this supports images that
    carry several (e.g. stamped by different parties). Returns a list of
    (identifier, content_hash), bottom-most first; empty list if none.
    """
    try:
        img = _resolve_image(image)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
    except Exception:
        return []
    # Union of both scans so a single image can carry bars in MIXED placements
    # — a bottom/full-width one (edge-anchored scan) alongside a relocated or
    # pasted-in one (full-canvas scan). De-dup by (id, hash); edge-anchored
    # first (cheaper, bottom-most first), then the anywhere fallback adds the
    # off-position ones it didn't already surface.
    results, seen = [], set()
    for _, res in _scan_for_bars(img, first_only=False):
        if res not in seen:
            seen.add(res)
            results.append(res)
    for _, res in _scan_anywhere(img, first_only=False):
        if res not in seen:
            seen.add(res)
            results.append(res)
    return results


# --- Full-canvas band search (locate a relocated/pasted bar anywhere) ---
# The M/Y/C↔C/Y/M bands are a distinctive "data begins/ends here" fiducial.
# _detect_bar (the fast gate) only reads them from the row's left edge; these
# helpers scan a row's colour runs to find the header AND footer ANYWHERE, so a
# bar whose canvas was extended (side/top/bottom margins) or that was pasted
# into a larger image still decodes. The frame's magic + CRC + Reed-Solomon
# reject a false band match downstream, so this only ever promotes a real bar.
_HSPAN_MIN_RUN = 3   # min consecutive px per band segment (JPEG erodes 8 → ~4)


def _row_color_runs(px, w, y):
    """Maximal M/Y/C runs on row ``y`` as [(colour, start, length)]; any
    non-band pixel breaks a run and is skipped."""
    runs = []
    x = 0
    while x < w:
        r, g, b = px[x, y][:3]
        if _is_magenta(r, g, b):
            c, pred = 'M', _is_magenta
        elif _is_yellow(r, g, b):
            c, pred = 'Y', _is_yellow
        elif _is_cyan(r, g, b):
            c, pred = 'C', _is_cyan
        else:
            x += 1
            continue
        j = x + 1
        while j < w:
            rr, gg, bb = px[j, y][:3]
            if not pred(rr, gg, bb):
                break
            j += 1
        runs.append((c, x, j - x))
        x = j
    return runs


def _bar_span_candidates(img, y):
    """Candidate (x0, x1) bar spans on row ``y`` from its bands, best-first.

    Header is M→Y→C (flush left of the bar); footer is C→Y→M (flush right).
    Adjacency (small inter-band gaps) rejects scattered content colours. A noisy
    margin can still produce several plausible bands, so we return every
    header×footer pairing, widest (most band-like) triples first, and let
    CRC + RS pick the real bar. The caller caps total decode attempts.
    """
    w, h = img.size
    if y < _SIG_ROWS or y >= h or w < 2 * _HSPAN_MIN_RUN * 3:
        return []
    px = img.load()
    runs = [r for r in _row_color_runs(px, w, y) if r[2] >= _HSPAN_MIN_RUN]
    if len(runs) < 6:   # need header (M,Y,C) + footer (C,Y,M)
        return []

    def adjacent(i, seq, maxgap=4):
        for k in range(3):
            if runs[i + k][0] != seq[k]:
                return False
        for k in range(2):
            gap = runs[i + k + 1][1] - (runs[i + k][1] + runs[i + k][2])
            if gap < 0 or gap > maxgap:
                return False
        return True

    def width(i):
        return runs[i][2] + runs[i + 1][2] + runs[i + 2][2]

    headers = sorted((i for i in range(len(runs) - 2) if adjacent(i, ('M', 'Y', 'C'))),
                     key=width, reverse=True)
    footers = sorted((i for i in range(len(runs) - 2) if adjacent(i, ('C', 'Y', 'M'))),
                     key=width, reverse=True)
    out = []
    for hi in headers:
        x0 = runs[hi][1]
        for fi in footers:
            x1 = runs[fi + 2][1] + runs[fi + 2][2] - 1   # rightmost M pixel of footer
            if x1 - x0 + 1 >= 2 * _HSPAN_MIN_RUN * 3 and (x0, x1) not in out:
                out.append((x0, x1))
    return out


def _scan_anywhere(img, first_only):
    """Fallback: find the bar by its band signature ANYWHERE on the canvas —
    any row AND any horizontal offset/width — then crop to that span so the
    existing bottom-path decode reads it flush. Handles a bar whose canvas was
    extended (margins) or that was pasted into a larger image. Heavier than the
    edge-anchored scan (a full-row band search + a few decode attempts per
    candidate row), so callers run it only after the fast paths fail. CRC + RS
    reject false band matches; a total-attempt cap bounds pathological inputs.
    """
    seen = set()
    attempts = 0
    y = img.size[1] - 1
    while y >= _SIG_ROWS:
        hit = False
        for (x0, x1) in _bar_span_candidates(img, y):
            attempts += 1
            if attempts > 500:
                return
            res = _extract_at_bottom(img.crop((x0, 0, x1 + 1, y + 1)))
            if res is not None:
                hit = True
                if res not in seen:
                    seen.add(res)
                    yield (y, res)
                    if first_only:
                        return
                break
        y -= _SIG_ROWS if hit else 1


def _scan_for_bars(img, first_only):
    """Yield (bottom_row, (identifier, content_hash)) for each row carrying the
    M/Y/C band signature whose crop decodes; de-dups by (id, hash).

    A cheap per-row band gate (no crop) rejects most rows; only on a match does
    it crop so the candidate row pair sits at the bottom and run the REAL
    bottom-path decode on that crop — so the asym reference lands on the row
    above the bar, exactly as at the true bottom. Scans bottom-up.
    """
    w, h = img.size
    seen = set()
    y = h - 1
    while y >= _SIG_ROWS:
        if _detect_bar(img, y) is not None:
            res = _extract_at_bottom(img.crop((0, 0, w, y + 1)))
            if res is not None and res not in seen:
                seen.add(res)
                yield (y, res)
                if first_only:
                    return
                y -= _SIG_ROWS   # skip this bar's partner (top) row
                continue
        y -= 1


def _extract_at_bottom(img):
    """Decode a bar at the bottom 2 rows of an already-resolved RGB ``img``.

    Tries the high-res even-fill layout first (both-ends-anchored, drift-free),
    then the small-image scale-swept sequential layout. Both self-
    validate via CRC + Reed-Solomon, so the correct one is selected by the data.
    Returns (identifier, content_hash) or None. The non-scanning core of
    :func:`extract_bar`.
    """
    # Threshold candidates. The asym per-column curve (predicted "1" level minus
    # delta/2, re-derived from the row above) is the PRIMARY — computed at the
    # image's current resolution so the scale-swept read indexes it correctly.
    # Otsu (the per-image bimodal midpoint) + the absolute 128 follow as scalar
    # FALLBACKS that rescue hard content where the asym curve's per-channel clamp
    # eats the delta margin (e.g. pure-saturated backgrounds, where "0"=center-Δ
    # bottoms out on a 0/255 channel). CRC + Reed-Solomon self-select across the
    # scale/ppb/offset sweep; the post-RS CRC re-check guards miscorrections.
    thresholds = []
    try:
        thresholds.append(_asym_threshold_curve(img))
    except Exception:
        pass
    otsu = _otsu_threshold(img)
    if otsu is not None:
        thresholds.append(otsu)
    thresholds.append(_RGB_THRESHOLD)

    # Band detection is threshold-independent — do it once, reuse per candidate.
    # Scale 1:1 is ALWAYS tried (band detection isn't needed at native scale, and
    # it can fail on a heavily-recompressed bar even when the sequential read at
    # 1:1 still decodes cleanly — CRC+RS guards false positives). Band detection
    # only adds the resized-scale sweep on top.
    bands = _detect_bar(img)
    scale_candidates = [1.0]
    if bands:
        raw_scale = (sum(bands) / 3) / _HEADER_BAND
        if abs(raw_scale - 1.0) >= 0.05:
            # Image appears resized. Band width detection can be off by ±2px
            # per band due to JPEG/interpolation, so the scale estimate has
            # ~±5% error. Sweep around the estimate in 1% steps.
            for offset_pct in range(-8, 9):
                s = round(raw_scale + offset_pct * 0.01, 3)
                if 0.3 < s < 3.0 and s != 1.0 and s not in scale_candidates:
                    scale_candidates.append(s)

    for threshold in thresholds:
        # High-res even-fill layout (full-width, both-ends anchored).
        result = _decode_even_fill(img, threshold)
        if result is not None:
            return result

        # Small-image sequential layout (scale-swept) — the current writer
        # output for every image below the even-fill crossover.
        if not scale_candidates:
            continue
        for scale in scale_candidates:
            # Sweep px/bit widest-first (the encoder picks the widest that fits);
            # CRC + RS self-select, so a wrong ppb just fails frame validation.
            for ppb in range(_PIXELS_PER_BIT_MAX, _PIXELS_PER_BIT_NARROW - 1, -1):
                bits = _decode_bits_at_scale(img, scale, ppb, threshold)
                result = _try_decode_frame(bits)
                if result is not None:
                    return result

    return None


def _decode_bits_at_scale(img, scale, ppb, threshold=_RGB_THRESHOLD):
    """Read data bits from the bar at a given scale factor and pixels-per-bit."""
    w, h = img.size
    pix = img.load()   # one accessor, not a getpixel (with its per-call load) per sample

    if abs(scale - 1.0) < 0.01:
        # Exact pixel positions — no rounding drift
        data_start = _HEADER_PIXELS
        data_end = w - _FOOTER_PIXELS
        bits_per_row = (data_end - data_start) // ppb

        bits = []
        for row_offset in range(_SIG_ROWS):
            y = h - 1 - row_offset
            for bit_idx in range(bits_per_row):
                x0 = data_start + bit_idx * ppb
                # Average ALL ppb columns of the bit (and its threshold) — far
                # more noise-immune than a single center pixel under JPEG.
                acc = tacc = 0.0; cnt = 0
                for dx in range(ppb):
                    cx = x0 + dx
                    if cx >= data_end:
                        break
                    r, g, b = pix[cx, y][:3]
                    acc += (r + g + b) / 3.0; tacc += _thr(threshold, cx); cnt += 1
                bits.append(1 if cnt and acc / cnt >= tacc / cnt else 0)
        return bits

    # Scaled decode — infer original layout
    orig_w = round(w / scale)
    orig_data_per_row = orig_w - _HEADER_PIXELS - _FOOTER_PIXELS
    orig_bits_per_row = orig_data_per_row // ppb

    bits = []
    for row_offset in range(_SIG_ROWS):
        y = h - 1 - row_offset
        for bit_idx in range(orig_bits_per_row):
            # Average the bit's full scaled span (both sides) for noise immunity.
            sx0 = round((_HEADER_PIXELS + bit_idx * ppb) * scale)
            sx1 = round((_HEADER_PIXELS + (bit_idx + 1) * ppb) * scale)
            acc = tacc = 0.0; cnt = 0
            for sx in range(sx0, max(sx0 + 1, sx1)):
                if sx < 0 or sx >= w:
                    break
                r, g, b = pix[sx, y][:3]
                acc += (r + g + b) / 3.0; tacc += _thr(threshold, sx); cnt += 1
            if cnt == 0:
                break
            bits.append(1 if acc / cnt >= tacc / cnt else 0)
    return bits


def _bits_to_bytes(bits):
    """Convert a bit list to a bytearray (MSB first, 8 bits per byte)."""
    raw = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | bits[i + j]
        raw.append(byte_val)
    return raw


def _parse_payload(payload_bytes):
    """Parse a packed payload into (identifier, content_hash) or None.

    Packed binary (the only form, see :func:`_pack_payload`):
      [prefix_len 3-10][prefix][8 id-bytes][8 hash-bytes]
    """
    if not payload_bytes:
        return None
    n = payload_bytes[0]
    if not (3 <= n <= 10 and len(payload_bytes) >= 1 + n + 16):
        return None
    try:
        prefix = payload_bytes[1:1 + n].decode('utf-8')
    except UnicodeDecodeError:
        return None
    idhex = payload_bytes[1 + n:1 + n + 8].hex()
    hashhex = payload_bytes[1 + n + 8:1 + n + 16].hex()
    return f"{prefix}-{idhex}", hashhex


def _try_decode_frame(bits):
    """Try to decode a Gen I frame (RS error correction). Returns (identifier, hash) or None."""
    raw = _bits_to_bytes(bits)

    if len(raw) < 8:
        return None
    if raw[0:2] != bytearray(_FRAME_MAGIC):
        return None
    if raw[2] != _FRAME_GEN:
        return None

    nsym = raw[3]
    payload_len = struct.unpack('>H', bytes(raw[4:6]))[0]
    stored_crc = struct.unpack('>H', bytes(raw[6:8]))[0]

    codeword_len = payload_len + nsym
    if len(raw) < 8 + codeword_len:
        return None

    codeword = bytes(raw[8:8 + codeword_len])

    # Try RS decode (corrects up to nsym//2 byte errors)
    try:
        payload = rs_decode(codeword, nsym)
        # Verify CRC after RS to catch rare miscorrections (>nsym//2 errors
        # that land near a different valid codeword).
        if _crc16(rs_encode(payload, nsym)) != stored_crc:
            return None
    except ValueError:
        # RS failed — try raw payload with CRC as last resort
        if _crc16(codeword) != stored_crc:
            return None
        payload = codeword[:payload_len]

    return _parse_payload(payload)



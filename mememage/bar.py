"""Steganographic bar codec for minted images.

Encodes an identifier and content hash into a 2-pixel-tall bar at the
bottom of an image. Survives JPEG recompression (tested to q50),
social media re-encoding, and platform pipeline conversion.

Frame format (Gen I):
    [2B magic 0xAD4E][1B gen=1][1B nsym][2B payload_len BE][2B CRC-16][RS(payload, nsym)]
    nsym = number of Reed-Solomon parity bytes (6 = corrects up to 3 byte errors).
    CRC-16 is computed over the full RS codeword (payload + parity).

Payload (UTF-8):
    <identifier>\x00<content_hash 16 hex chars>
    Source-agnostic. The decoder resolves the identifier through search.
    The content hash verifies whatever is found.

Bar layout per row:
    [M×8][Y×8][C×8][data pixels...][C×8][Y×8][M×8]

Each pixel keeps the dominant HUE; the two bit levels are placed
symmetrically around the local dominant BRIGHTNESS (dom ± _BAR_DELTA/2),
so the bar sinks toward the image's own tone instead of glowing on dark
images. The decoder recovers the per-image threshold by Otsu; legacy
absolute bars (bit 0 → 64, bit 1 → 192) still decode via the fixed-128
candidate. The 8-pixel-wide M/Y/C color bands survive JPEG DCT blocks
and bracket the data.

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
    the two rows at an integer px/bit (3, falling back to 2). Byte-identical
    to the original layout, so small images and all pre-existing bars are
    unchanged. The decoder estimates scale from band width and sweeps.

The decoder tries even-fill first, then the legacy scale-swept sequential
read; both self-validate via CRC + Reed-Solomon, so the data selects the
correct one.
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
_LOCAL_CONTEXT_ROWS = 6

_PIXELS_PER_BIT_WIDE = 3       # crossover ppb: even-fill triggers at data_width >= this * n_bits
_PIXELS_PER_BIT_NARROW = 2    # 2px/bit for narrower images (768px portraits)
_PIXELS_PER_BIT_MAX = 6       # sequential picks the WIDEST ppb that fits (fatter = quieter +
                              # JPEG-tougher); the packed payload frees the room to widen.
_RGB_TARGET_0 = 64           # legacy absolute levels — still read via the 128 candidate
_RGB_TARGET_1 = 192
_RGB_THRESHOLD = 128         # absolute decode candidate (legacy bars + always-present fallback)
# --- Asym row-3-copy camouflage (Gen I, decode-compatible) --------------------
# The data bits ride a PER-COLUMN center that copies the smoothed content one row
# ABOVE the bar (floored on dark, hue-preserving): a "1" bit IS that level
# (so it reads as a continuation of the image — invisible), a "0" bit is darker
# by _ASYM_DELTA, and filler past the payload is "1" (invisible). The decoder
# never compares to the bar's own (asymmetric, biased) distribution — it
# RE-PREDICTS the per-column "1" level from the preserved row above and
# thresholds delta/2 below it. That makes it robust (q35+, multi-pass, Discord,
# worst-case noise validated) AND much quieter than the centered scheme.
# Backward-compatible: a new decode CANDIDATE; legacy/centered bars still decode
# on the Otsu/128 candidates. New mints get the quiet bar; existing are untouched.
_ASYM_ENCODE = True
_ASYM_DELTA = 40             # "0" sits this far below the per-column "1" level. Lowered
                             # 48->40 (~17% quieter) — the packed payload's fatter bits give
                             # the room. Δ40 validated discord-safe (8/8 single q80 + Instagram
                             # resize; 7/8 on a harsh double-reshare, amber the hard case).
                             # Δ32 is quieter (~33%) but trades heavy-reshare robustness.
_ASYM_FLOOR = 70             # min luma for a detectable "1" on near-black content
_ASYM_BOX_RADIUS = 34        # px; box-blur radius for the per-column center (encode + decode).
                             # A box filter (integer-sum / one division), NOT a Gaussian:
                             # math.exp diverges by 1 ULP between glibc and V8, which would
                             # break byte-exact writer parity (tests/bar_encode_parity.cjs).
                             # A box filter is IEEE-deterministic across runtimes. Radius 34
                             # (window 69) approximates the prior sigma-20 Gaussian support.

_BAR_DELTA = 64              # centered scheme (legacy/fallback): bit-0/bit-1 separation (dom ± 32).
                             # Lowered from 96 to camouflage the data strip (~33% quieter, and
                             # the "1" bits on dark content drop from ~96 to ~64). 64 is the
                             # measured floor that still survives JPEG q50 on a WORST-CASE
                             # full-width-noise image (56 fails q50). Otsu decode is adaptive +
                             # RS covers the rest, so no decoder change and legacy 96-delta
                             # bars still decode. New mints get quieter bars; existing
                             # bars are unaffected.



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
    """Lift a colour to a minimum luma WITHOUT amplifying its tint.

    Adds the luma deficit equally to each channel (additive lift), so the absolute
    channel differences are preserved and the result is a *muted* lift of the
    content's colour. A near-black blue (2,5,20) becomes (66,69,84) at luma 70 —
    NOT a saturated (24,60,241). The old multiplicative scaling (r*floor/L) blew up
    tiny tints in near-black content into bright saturated colours, which showed as
    blue pixels in the bar over dark-blue regions (e.g. cave shadows). Adding the
    same delta to each channel raises luma by exactly that delta, so luma still
    reaches the floor; clamping only nudges luma a hair below on rare saturated
    darks, and the decoder re-derives the identical value so it stays consistent.
    """
    L = 0.299 * r + 0.587 * g + 0.114 * b
    if L >= floor:
        return r, g, b
    d = floor - L
    return min(255.0, r + d), min(255.0, g + d), min(255.0, b + d)


def _asym_center_columns(img, w, h):
    """Per-column (center_rgb, center_luma) for the asym scheme: the smoothed,
    floored colour of the row immediately above the bar. The "1" bits copy this
    (camouflage); "0" = center - delta. The decoder re-derives the same curve."""
    y = h - _SIG_ROWS - 1                  # row immediately above the 2 bar rows
    if y < 0:
        y = max(0, h - 1)
    rr = [0.0] * w; gg = [0.0] * w; bb = [0.0] * w
    for x in range(w):
        px = img.getpixel((x, y))[:3]
        rr[x], gg[x], bb[x] = px[0], px[1], px[2]
    rr = _smooth1d(rr, _ASYM_BOX_RADIUS); gg = _smooth1d(gg, _ASYM_BOX_RADIUS); bb = _smooth1d(bb, _ASYM_BOX_RADIUS)
    center_rgb = []; center_val = []
    for x in range(w):
        r, g, b = _hue_floor(rr[x], gg[x], bb[x], _ASYM_FLOOR)
        center_rgb.append((r, g, b))
        # (R+G+B)/3, NOT luma — the decoder measures each bar pixel as (r+g+b)/3
        # (see _decode_bits_at_scale), so the predicted "1" level must use the same
        # metric. On coloured content luma and (r+g+b)/3 differ by ~10, which ate
        # the margin at low delta; matching them restores the full delta/2 margin.
        center_val.append((r + g + b) / 3.0)
    return center_rgb, center_val


def _asym_threshold_curve(img):
    """Per-column decode threshold: the predicted "1" level (from the row above)
    minus delta/2 — the midpoint between the "1" (=center) and "0" (=center-delta)
    levels. Robust because it's re-derived from preserved content, not the bar."""
    w, h = img.size
    _, center_lum = _asym_center_columns(img, w, h)
    half = _ASYM_DELTA / 2.0
    return [center_lum[x] - half for x in range(w)]


def _thr(threshold, px):
    """Threshold at column ``px``. A scalar candidate returns as-is; the asym
    candidate is a per-column list (the row-3-predicted threshold curve)."""
    if isinstance(threshold, list):
        if 0 <= px < len(threshold):
            return threshold[px]
        return threshold[-1] if threshold else _RGB_THRESHOLD
    return threshold


# ---------------------------------------------------------------------------
# Dominant color from local context
# ---------------------------------------------------------------------------

def _dominant_color(img, sig_rows=_SIG_ROWS):
    """Mean color from the rows just above the signature bar."""
    w, h = img.size
    context_end = h - sig_rows
    context_start = max(0, context_end - _LOCAL_CONTEXT_ROWS)
    if context_end <= context_start:
        # Very short image — whole image mean
        total_r, total_g, total_b, count = 0, 0, 0, 0
        for y in range(h):
            for x in range(w):
                r, g, b = img.getpixel((x, y))[:3]
                total_r += r
                total_g += g
                total_b += b
                count += 1
        if count == 0:
            return 128, 128, 128
        return round(total_r / count), round(total_g / count), round(total_b / count)

    total_r, total_g, total_b, count = 0, 0, 0, 0
    for y in range(context_start, context_end):
        for x in range(w):
            r, g, b = img.getpixel((x, y))[:3]
            total_r += r
            total_g += g
            total_b += b
            count += 1
    return round(total_r / count), round(total_g / count), round(total_b / count)


# ---------------------------------------------------------------------------
# M/Y/C band color predicates (shared by detect + even-fill anchoring)
# ---------------------------------------------------------------------------

def _is_magenta(r, g, b):
    return r > 130 and g < 120 and b > 130

def _is_yellow(r, g, b):
    return r > 130 and g > 130 and b < 120

def _is_cyan(r, g, b):
    return r < 120 and g > 130 and b > 130


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


def _write_sequential(img, w, h, data_width, bits, bit_rgb, dom_r, dom_g, dom_b, payload, filler_bit=0):
    """Legacy/small-image layout: split the frame sequentially across the two
    rows at an integer px/bit (3, falling back to 2 for narrow images). Kept
    byte-identical to the original so small images and pre-existing bars are
    unchanged.
    """
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
                # Filler past the payload. Legacy: bit-0 level (clean Otsu
                # bimodality). Asym: filler_bit=1 (copies row above = invisible).
                for px in range(ppb):
                    if base_x + px < w - _FOOTER_PIXELS:
                        img.putpixel((base_x + px, y), bit_rgb(filler_bit, base_x + px))


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

_HEXSET = frozenset('0123456789abcdef')


def _pack_payload(identifier, content_hash):
    """Build the bar payload bytes.

    Canonical records (identifier ``<prefix>-<16 hex>`` + 16-hex content hash)
    pack to BINARY: ``[prefix_len][prefix][8 id-bytes][8 hash-bytes]`` — ~16 bytes
    smaller than the old ASCII-hex form, which is what buys fatter (quieter,
    JPEG-tougher) bar bits. Non-canonical identifiers (e.g. raw-API content
    addresses) fall back to the ASCII ``identifier\\x00hash`` form. The two are
    self-distinguishing with no tag byte: the packed form's first byte is the
    prefix length (3-10), the ASCII form's first byte is the identifier's leading
    letter (>=65) — disjoint ranges.
    """
    pre, sep, idhex = identifier.rpartition('-')
    if (sep and 3 <= len(pre) <= 10 and len(idhex) == 16
            and _HEXSET.issuperset(idhex)
            and len(content_hash) == 16 and _HEXSET.issuperset(content_hash)):
        return bytes([len(pre)]) + pre.encode('utf-8') + bytes.fromhex(idhex) + bytes.fromhex(content_hash)
    return f"{identifier}\x00{content_hash}".encode('utf-8')


def embed_into(image, identifier, content_hash):
    """Write a Mememage bar into the bottom 2 rows of an image, IN MEMORY.

    Reads any image :func:`_resolve_image` accepts (path, bytes, file-like, PIL
    Image, numpy array) and returns a NEW barred RGB ``PIL.Image`` — no disk, and
    the caller's image is never mutated. The disk wrapper :func:`embed_bar` and
    the core ``encode`` both build on this.

    Args:
        image: the source image (any in-memory or on-disk form).
        identifier: Mememage identifier (e.g. "mememage-a3f8c2d1e5b6").
        content_hash: 16 hex char content hash.

    Raises:
        ValueError: If the image is too narrow for the payload.
    """
    payload = _pack_payload(identifier, content_hash)

    # Fresh RGB copy — convert() always returns a new image, so the caller's
    # image is left untouched even when it's already RGB.
    img = _resolve_image(image).convert('RGB')
    w, h = img.size

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
    use_asym = _ASYM_ENCODE

    # Dominant color from local context (legacy/centered scheme + sequential sig).
    dom_r, dom_g, dom_b = _dominant_color(img)
    dom_avg = (dom_r + dom_g + dom_b) / 3

    if use_asym:
        # Asym camouflage: each bit rides a PER-COLUMN center that copies the
        # smoothed, floored content one row above. "1" = center (invisible),
        # "0" = center - delta. Filler past the payload = "1" (invisible). The
        # decoder re-derives the center from the row above (see _asym_*).
        _center_rgb, _center_lum = _asym_center_columns(img, w, h)
        _filler_bit = 1

        def _bit_rgb(bit, x):
            cr, cg, cb = _center_rgb[x]
            if bit:
                return (round(cr), round(cg), round(cb))
            return (max(0, round(cr - _ASYM_DELTA)),
                    max(0, round(cg - _ASYM_DELTA)),
                    max(0, round(cb - _ASYM_DELTA)))
    else:
        # Centered brightness (legacy): keep the dominant hue, place the two bit
        # levels symmetrically around ONE clamped global dominant brightness.
        _half = _BAR_DELTA / 2.0
        _center = max(_half, min(255 - _half, dom_avg))
        _lo, _hi = _center - _half, _center + _half
        _filler_bit = 0

        def _bit_rgb(bit, x=None):
            target_avg = _hi if bit else _lo
            offset = target_avg - dom_avg
            return (max(0, min(255, round(dom_r + offset))),
                    max(0, min(255, round(dom_g + offset))),
                    max(0, min(255, round(dom_b + offset))))

    # Layout choice is capacity-emergent, no flag or version bump:
    #   - Above the crossover (the whole frame fits in ONE row at >=3px/bit),
    #     spread the bits to EVENLY FILL the full width between the flush
    #     bilateral bands, and paint them 2px tall (both rows identical).
    #     Fat bits => downscale resilience; the even fill => zero idle pixels;
    #     the 2px height => vertical redundancy (survives a 1px bottom crop,
    #     stronger under JPEG); both-end band anchoring => drift-free decode.
    #   - Below the crossover (small images), fall back to the original
    #     sequential split across the two rows (byte-identical to legacy).
    if is_even_fill:
        _write_even_fill(img, w, h, bits, _bit_rgb)
    else:
        _write_sequential(img, w, h, data_width, bits, _bit_rgb, dom_r, dom_g, dom_b,
                          payload, filler_bit=_filler_bit)

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

def _detect_bar(img):
    """Check if the bottom row has the M/Y/C header pattern.

    Scans for the magenta→yellow→cyan transition to detect presence
    and measure the band width (reveals scale factor if resized).

    Returns (magenta_width, yellow_width, cyan_width) or None.
    """
    w, h = img.size
    if h < _SIG_ROWS or w < 20:
        return None

    y = h - 1

    def at(x):
        return img.getpixel((x, y))[:3]

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
# packed frame is 34..41B. The ASCII fallback (non-canonical ids) can be larger,
# so the window spans 33..64B with margin on both sides; CRC self-selects.
_EVENFILL_MIN_BYTES = 33
_EVENFILL_MAX_BYTES = 64


def _find_header_end(img, y, w):
    """Return x just past the header's M->Y->C run (where data starts), or None.

    The data-adjacent edge is COMPUTED, not measured by running the cyan count
    into the data: asym camo data pixels can be cyan-hued and would extend the
    run past the true edge. ``mag_start`` (the image's left edge) and
    ``cyan_start`` (bounded by yellow) never touch data, and the span between
    them is exactly two band widths — so band_width = (cyan_start-mag_start)/2
    and data_start = cyan_start + band_width. Small transition-skip tolerance
    absorbs JPEG smear between bands; the caller's phase sweep absorbs ±1-2px.
    """
    def run(pred, x):
        n = 0
        while x < w and pred(*img.getpixel((x, y))[:3]):
            x += 1
            n += 1
        return x, n

    x = 0
    while x < w and x < 40 and not _is_magenta(*img.getpixel((x, y))[:3]):
        x += 1
    mag_start = x
    x, nm = run(_is_magenta, x)
    while x < w and x < 60 and not _is_yellow(*img.getpixel((x, y))[:3]):
        x += 1
    x, ny = run(_is_yellow, x)
    while x < w and x < 80 and not _is_cyan(*img.getpixel((x, y))[:3]):
        x += 1
    cyan_start = x
    x, nc = run(_is_cyan, x)
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
    cyan_start - band_width + 1 (cyan_start is the rightmost cyan pixel)."""
    def run(pred, x):
        n = 0
        while x >= 0 and pred(*img.getpixel((x, y))[:3]):
            x -= 1
            n += 1
        return x, n

    x = w - 1
    while x >= 0 and x > w - 40 and not _is_magenta(*img.getpixel((x, y))[:3]):
        x -= 1
    mag_start = x
    x, nm = run(_is_magenta, x)
    while x >= 0 and x > w - 60 and not _is_yellow(*img.getpixel((x, y))[:3]):
        x -= 1
    x, ny = run(_is_yellow, x)
    while x >= 0 and x > w - 80 and not _is_cyan(*img.getpixel((x, y))[:3]):
        x -= 1
    cyan_start = x
    x, nc = run(_is_cyan, x)
    if nm < 2 or ny < 2 or nc < 2:
        return None
    band_width = (mag_start - cyan_start) / 2.0
    return int(round(cyan_start - band_width)) + 1


def _otsu_threshold(img):
    """Per-image bit threshold for the centered bar.

    Otsu over the middle 60% of the bottom rows (avoids the M/Y/C bands and is
    scale-robust — no fixed band-pixel offsets), returned as the MIDPOINT of the
    two class means rather than the boundary index. The midpoint is what makes
    this robust on an exact (lossless) bar, where the two levels are delta-spikes
    and a boundary index would land *on* the bit-0 level and misread it. Returns
    None on a degenerate (flat) region so the caller falls back to the absolute
    128 candidate.
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
        for y in range(max(0, h - _SIG_ROWS), h):
            for x in range(x0, x1):
                r, g, b = img.getpixel((x, y))[:3]
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
                            r, g, bl = img.getpixel((px, ry))[:3]
                            acc += (r + g + bl) / 3.0
                        bits.append(1 if acc / len(rows) >= _thr(threshold, px) else 0)
                    if not ok:
                        continue
                    result = _try_decode_frame(bits)
                    if result is not None:
                        return result
    return None


def extract_bar(image):
    """Extract identifier and content hash from a barred image.

    Tries the high-res even-fill layout first (both-ends-anchored, drift-free),
    then falls back to the legacy scale-swept sequential layout. Both self-
    validate via CRC + Reed-Solomon, so the correct one is selected by the data.

    Args:
        image: a path, raw bytes, a file-like object, a PIL Image, or a numpy
            array — anything :func:`_resolve_image` accepts (no disk required).

    Returns:
        (identifier, content_hash) tuple, or None if no valid bar found.
    """
    try:
        img = _resolve_image(image)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
    except Exception:
        return None

    # Brightness threshold candidates. The per-image Otsu midpoint reads the
    # centered bar (levels hug the dominant brightness); the absolute 128 reads
    # legacy absolute bars and is always present as a fallback. Otsu is tried
    # first (it's the current scheme); CRC + Reed-Solomon self-select, so a wrong
    # threshold just fails frame validation and the next candidate is tried.
    thresholds = []
    otsu = _otsu_threshold(img)
    if otsu is not None:
        thresholds.append(otsu)
    thresholds.append(_RGB_THRESHOLD)
    # Asym camouflage bar — a PER-COLUMN threshold re-derived from the row above
    # the bar (predicted "1" level minus delta/2). Tried after the scalar
    # candidates so legacy/centered bars resolve first; asym bars fail the scalar
    # candidates (their center varies per column) and resolve here. CRC+RS
    # self-selects. The curve is computed at the image's current resolution, so
    # the scale-swept sequential read indexes it correctly on downscaled images.
    try:
        thresholds.append(_asym_threshold_curve(img))
    except Exception:
        pass

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

        # Legacy / small-image sequential layout (scale-swept).
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
                    r, g, b = img.getpixel((cx, y))[:3]
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
                r, g, b = img.getpixel((sx, y))[:3]
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
    """Parse a decoded payload into (identifier, content_hash) or None.

    Handles three forms (self-distinguishing, see :func:`_pack_payload`):
      Packed:  [prefix_len 3-10][prefix][8 id-bytes][8 hash-bytes]  (canonical)
      ASCII:   identifier\\x00hash                                   (non-canonical / raw API)
      Legacy:  <url>/<id>/record.json\\x00hash                       (old URL payload)
    """
    if not payload_bytes:
        return None
    n = payload_bytes[0]
    if 3 <= n <= 10 and len(payload_bytes) >= 1 + n + 16:
        # Packed binary form — first byte is the prefix length.
        try:
            prefix = payload_bytes[1:1 + n].decode('utf-8')
        except UnicodeDecodeError:
            prefix = None
        if prefix is not None:
            idhex = payload_bytes[1 + n:1 + n + 8].hex()
            hashhex = payload_bytes[1 + n + 8:1 + n + 16].hex()
            return f"{prefix}-{idhex}", hashhex
    try:
        text = payload_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return None
    if '\x00' not in text:
        return None
    first, content_hash = text.split('\x00', 1)
    if '/' in first:
        # Old format: URL — extract identifier
        import re
        match = re.search(r'mememage-[a-f0-9]+', first)
        identifier = match.group(0) if match else first
    else:
        # New format: bare identifier
        identifier = first
    return identifier, content_hash


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



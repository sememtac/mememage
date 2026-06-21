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

_PIXELS_PER_BIT_WIDE = 3       # 3px/bit for images >= 1024px wide (resize resilient)
_PIXELS_PER_BIT_NARROW = 2    # 2px/bit for narrower images (768px portraits)
_RGB_TARGET_0 = 64           # legacy absolute levels — still read via the 128 candidate
_RGB_TARGET_1 = 192
_RGB_THRESHOLD = 128         # absolute decode candidate (legacy bars + always-present fallback)
_BAR_DELTA = 96              # centered scheme: bit-0/bit-1 brightness separation (dom ± 48)



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
            rgb = bit_rgb(bits[i])
            for x in range(x0, x1):
                img.putpixel((x, y), rgb)


def _write_sequential(img, w, h, data_width, bits, bit_rgb, dom_r, dom_g, dom_b, payload):
    """Legacy/small-image layout: split the frame sequentially across the two
    rows at an integer px/bit (3, falling back to 2 for narrow images). Kept
    byte-identical to the original so small images and pre-existing bars are
    unchanged.
    """
    total_data_pixels = _SIG_ROWS * data_width
    header_overhead = 8
    rs_overhead = _RS_NSYM

    ppb = _PIXELS_PER_BIT_WIDE
    capacity_wide = (total_data_pixels // _PIXELS_PER_BIT_WIDE) // 8 - header_overhead - rs_overhead
    if len(payload) > capacity_wide:
        ppb = _PIXELS_PER_BIT_NARROW
        capacity_narrow = (total_data_pixels // _PIXELS_PER_BIT_NARROW) // 8 - header_overhead - rs_overhead
        if len(payload) > capacity_narrow:
            raise ValueError(
                f"Bar payload too large ({len(payload)}B) for image width "
                f"({w}px, {capacity_narrow}B capacity at {ppb}px/bit)"
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
                rgb = bit_rgb(bits[bit_idx])
                for px in range(ppb):
                    img.putpixel((base_x + px, y), rgb)
            else:
                # Filler past the payload = the bit-0 level (not the dom color),
                # so the decode brightness histogram stays cleanly bimodal for
                # Otsu. dom_r/g/b kept in the signature for backward-compat callers.
                fill = bit_rgb(0)
                for px in range(ppb):
                    if base_x + px < w - _FOOTER_PIXELS:
                        img.putpixel((base_x + px, y), fill)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

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
    payload = f"{identifier}\x00{content_hash}".encode('utf-8')

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

    # Dominant color from local context
    dom_r, dom_g, dom_b = _dominant_color(img)
    dom_avg = (dom_r + dom_g + dom_b) / 3

    # Centered brightness: keep the dominant hue, but place the two bit levels
    # symmetrically around the local dominant brightness instead of pinning them
    # to absolute 64/192. The center is clamped so the full _BAR_DELTA still fits
    # in [0,255]; the bar then sinks toward the image's own tone on dark images
    # (the old absolute 192 always glowed) while keeping the same separation, so
    # the JPEG envelope is unchanged. The decoder finds the threshold by Otsu.
    _half = _BAR_DELTA / 2.0
    _center = max(_half, min(255 - _half, dom_avg))
    _lo, _hi = _center - _half, _center + _half

    def _bit_rgb(bit):
        target_avg = _hi if bit else _lo
        offset = target_avg - dom_avg
        return (max(0, min(255, round(dom_r + offset))),
                max(0, min(255, round(dom_g + offset))),
                max(0, min(255, round(dom_b + offset))))

    data_width = w - _HEADER_PIXELS - _FOOTER_PIXELS

    # Layout choice is capacity-emergent, no flag or version bump:
    #   - Above the crossover (the whole frame fits in ONE row at >=3px/bit),
    #     spread the bits to EVENLY FILL the full width between the flush
    #     bilateral bands, and paint them 2px tall (both rows identical).
    #     Fat bits => downscale resilience; the even fill => zero idle pixels;
    #     the 2px height => vertical redundancy (survives a 1px bottom crop,
    #     stronger under JPEG); both-end band anchoring => drift-free decode.
    #   - Below the crossover (small images), fall back to the original
    #     sequential split across the two rows (byte-identical to legacy).
    if data_width >= _PIXELS_PER_BIT_WIDE * len(bits):
        _write_even_fill(img, w, h, bits, _bit_rgb)
    else:
        _write_sequential(img, w, h, data_width, bits, _bit_rgb, dom_r, dom_g, dom_b, payload)

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
# Payload = prefix(3-10) + '-' + 16hex + NUL + 16hex = 37..44B, so the frame is
# 51..58B. The window has a little margin on both sides; CRC self-selects.
_EVENFILL_MIN_BYTES = 49
_EVENFILL_MAX_BYTES = 64


def _find_header_end(img, y, w):
    """Return x just past the header's M->Y->C run (where data starts), or None.

    Scans from the left edge with small transition-skip tolerance for JPEG
    smear between bands.
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
    x, nm = run(_is_magenta, x)
    while x < w and x < 60 and not _is_yellow(*img.getpixel((x, y))[:3]):
        x += 1
    x, ny = run(_is_yellow, x)
    while x < w and x < 80 and not _is_cyan(*img.getpixel((x, y))[:3]):
        x += 1
    x, nc = run(_is_cyan, x)
    if nm < 2 or ny < 2 or nc < 2:
        return None
    return x


def _find_footer_start(img, y, w):
    """Return x at the left edge of the footer's C/Y/M run (where data ends),
    or None. Scans inward from the right edge (footer order is C,Y,M, so from
    the right it reads M->Y->C)."""
    def run(pred, x):
        n = 0
        while x >= 0 and pred(*img.getpixel((x, y))[:3]):
            x -= 1
            n += 1
        return x, n

    x = w - 1
    while x >= 0 and x > w - 40 and not _is_magenta(*img.getpixel((x, y))[:3]):
        x -= 1
    x, nm = run(_is_magenta, x)
    while x >= 0 and x > w - 60 and not _is_yellow(*img.getpixel((x, y))[:3]):
        x -= 1
    x, ny = run(_is_yellow, x)
    while x >= 0 and x > w - 80 and not _is_cyan(*img.getpixel((x, y))[:3]):
        x -= 1
    x, nc = run(_is_cyan, x)
    if nm < 2 or ny < 2 or nc < 2:
        return None
    return x + 1


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
                        bits.append(1 if acc / len(rows) >= threshold else 0)
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

    # Band detection is threshold-independent — do it once, reuse per candidate.
    bands = _detect_bar(img)
    scale_candidates = None
    if bands:
        raw_scale = (sum(bands) / 3) / _HEADER_BAND
        scale_candidates = [1.0]  # always try 1:1 first
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
            for ppb in [_PIXELS_PER_BIT_WIDE, _PIXELS_PER_BIT_NARROW]:
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
                cx = data_start + bit_idx * ppb + ppb // 2
                r, g, b = img.getpixel((cx, y))[:3]
                bits.append(1 if (r + g + b) / 3 >= threshold else 0)
        return bits

    # Scaled decode — infer original layout
    orig_w = round(w / scale)
    orig_data_per_row = orig_w - _HEADER_PIXELS - _FOOTER_PIXELS
    orig_bits_per_row = orig_data_per_row // ppb

    bits = []
    for row_offset in range(_SIG_ROWS):
        y = h - 1 - row_offset
        for bit_idx in range(orig_bits_per_row):
            orig_cx = _HEADER_PIXELS + bit_idx * ppb + ppb / 2
            sx = round(orig_cx * scale)
            if sx < 0 or sx >= w:
                break
            r, g, b = img.getpixel((sx, y))[:3]
            bits.append(1 if (r + g + b) / 3 >= threshold else 0)
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

    Handles both formats:
      New: identifier\\x00hash (source-agnostic)
      Old: <url>/<id>/record.json\\x00hash (legacy URL payload, backward compat)
    """
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



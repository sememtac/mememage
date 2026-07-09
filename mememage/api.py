"""encode / decode / verify — the Mememage core API.

The whole of Mememage core: write a bar into an image, get back a record whose
fields you chose, and verify the data against the image by math alone. Just the
bar, the record, and the hash — no networking, no record schema, no field
semantics; anything more belongs to the application built on top. This is the
small surface a tool can ``pip install`` and use::

    import mememage
    result = mememage.encode("shot.png", {"title": "a cat"})  # write the bar + record
    result.save("shot.json")                  # store the record separately
    ...
    bar = mememage.decode("shot.jpg")         # read it back: identifier + content hash
    if mememage.verify("shot.jpg", result.record):   # the record matches the image
        ...

The bar carries exactly two things: the IDENTIFIER (a key to a record stored
separately — core does not fetch it) and the CONTENT HASH (a 64-bit digest —
the first 16 hex of SHA-256 — over the record). ``decode`` reads them back out; ``verify`` re-hashes a record and checks
it against the hash in the pixels. The ``open`` hash makes every field of the
record tamper-evident. Core stops at integrity (hash + optional field encryption);
identity and authorship (signing) are out of scope.

Pillow is required (the bar codec, shipped in the base install); ``cryptography``
only if you encrypt fields.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import namedtuple
from dataclasses import dataclass

from mememage import bar, hashing

OPEN = hashing.OPEN_HASH_VERSION  # "open"

# Keys encode computes / reserves — passing them in `fields` is a mistake, not a
# silent override. `signature` is reserved (not just computed): the open hash
# leaves it OUT (the structurally-circular slot), so a detached signature can be
# attached later by an external tool without re-hashing — a user field
# named `signature` would therefore go unprotected, so it's refused.
_RESERVED = {"identifier", "content_hash", "hash_version", "signature",
             "encrypted_fields"}

# Identifier prefix rule: starts with a letter, ends alphanumeric, 3-10 chars,
# URL/path-safe; capped at 10 so the bar still fits a 512px image.
_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,8}[A-Za-z0-9]$")


Bar = namedtuple("Bar", ["identifier", "content_hash"])


@dataclass
class Record:
    """What :func:`encode` returns: the ``record`` (your fields + identifier +
    content hash), the now-barred ``image`` (a PIL Image, always in memory), and
    ``image_path`` (where it was written, or None for an in-memory-only encode).
    Store the record wherever you like; a verifier finds it by identifier and
    trusts it by hash."""
    record: dict
    image_path: "str | None" = None
    image: "object | None" = None        # the barred PIL Image (in-memory output)

    @property
    def identifier(self) -> str:
        return self.record["identifier"]

    @property
    def content_hash(self) -> str:
        return self.record["content_hash"]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.record, indent=indent, ensure_ascii=False)

    def save(self, path: str) -> str:
        """Write the record to ``path`` as JSON and return the path. Put it on
        your own server, a CDN, IPFS, a file — the verifier is source-agnostic
        and trusts it by hash alone."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path


@dataclass
class Verification:
    """What :func:`verify` returns. Truthy iff the record matches the image — the
    re-hashed record equals the content hash baked in the pixels. ``reason``
    explains a failure (empty on success). (Core verifies integrity only;
    authorship/signatures are out of scope.) For the bar's identifier or hash,
    call :func:`decode`."""
    match: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.match


def _validate_prefix(prefix: str) -> None:
    if not isinstance(prefix, str) or not _PREFIX_RE.match(prefix):
        raise ValueError(
            f"invalid prefix {prefix!r}: 3-10 chars, [A-Za-z][A-Za-z0-9_-]*"
            "[A-Za-z0-9] — URL/path/filename-safe (the bar must fit a 512px image)")


def _canonical(obj) -> str:
    """The canonical JSON the hash + identifier derive from — sorted keys,
    normalized floats (1.0 → 1, matching JS), no whitespace."""
    return json.dumps(hashing._normalize_for_hash(obj), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)


def _swap_to_png(path: str) -> str:
    """``/a/photo.jpg`` -> ``/a/photo.png`` — the lossless output for a non-PNG input."""
    import os
    return os.path.splitext(path)[0] + ".png"


def _content_identifier(fields: dict, prefix: str) -> str:
    """Content-address the record: ``<prefix>-<16 hex of canonical(fields)>``.
    Same fields → same identifier (natural dedup); include a unique field
    (timestamp, id) for a fresh one. The data determines its own name."""
    digest = hashlib.sha256(_canonical(fields).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def encode(image, fields=None, *, prefix="mememage", identifier=None,
           password=None, private=None, out=None) -> Record:
    """Write a Mememage bar into an image and return its :class:`Record`.

    Adds ``identifier``, ``content_hash`` and ``hash_version="open"`` to your
    fields, hashes everything (the ``open`` model — every field tamper-evident),
    and embeds the identifier + content hash into the bottom two pixel rows
    (a packed binary payload — prefix + 8 raw identifier bytes + 8 raw hash
    bytes — framed with CRC-16 + Reed-Solomon; see ``bar.py``).

    Reads **any image** — a path, raw ``bytes``, a file-like object, a PIL Image,
    or a numpy array (HEIC paths need the ``[heic]`` extra). The barred image is
    always returned in memory as ``Record.image`` (a PIL Image). It is also
    written to disk as a **lossless PNG** when there's a destination — the bar is
    exact pixel data, so a lossy save would scramble it:

    - a **path** input with no ``out`` → in place if PNG, else a ``.png`` sibling.
    - ``out=<path.png>`` → written there. ``out=<file-like>`` (e.g. ``BytesIO``) →
      written to the stream. ``out`` must be PNG.
    - an **in-memory** input (PIL/bytes/ndarray) with no ``out`` → no disk; the
      barred image is ``Record.image`` only.

    Field visibility (``password``): your own encryption. With a password, the
    private fields leave the cleartext record and become an ``encrypted_fields``
    envelope (AES-256-GCM via PBKDF2). The content hash then covers the
    CIPHERTEXT — so the record still verifies *without* the password (the proof
    is over the encrypted shell), and tampering with the ciphertext breaks it.
    Reveal the fields with :func:`unlock` (the reference decoder web app — a
    separate application, not part of this package — offers the same unlock in
    the browser). The password is not stored; only the ciphertext is kept.

    Args:
        image: the source image — a path, bytes, a file-like, a PIL Image, or a
            numpy array. Never mutated (encode works on a copy).
        fields: your data as a JSON-serializable dict (None → identifier + hash
            only). Reserved keys (identifier/content_hash/hash_version/
            signature/encrypted_fields) can't be passed.
        prefix: identifier namespace, 3-10 chars, IA-safe. Default ``mememage``.
        identifier: override the auto content-addressed identifier with your own
            canonical ``<prefix>-<16 lower-hex>`` string (anything else raises
            ``ValueError`` — the bar packs exactly that shape).
        password: encrypt private fields under this passphrase. Needs the
            [encrypt] extra (cryptography). None → fully public record.
        private: which field names to encrypt (list). With a password, ``None``
            encrypts ALL your fields; a list encrypts just those, leaving the
            rest public. Without a password it's an error.
        out: a destination for the barred PNG — a ``.png`` path or a file-like
            object. None → see the path/in-memory rules above.

    Returns: :class:`Record` (``record`` + ``image`` (the barred PIL Image) +
        ``image_path`` (the PNG written, or None)). On an encrypted record the
        ``record`` holds the public fields + ``encrypted_fields``; the private
        fields live only in the envelope.

    Raises: ``ValueError`` if ``out`` isn't a ``.png``, on a reserved/`_`-prefixed
        key, a bad prefix, ``private`` without ``password``, or an unknown
        ``private`` name; ``RuntimeError`` if ``password`` is set but cryptography
        is unavailable.
    """
    fields = dict(fields or {})
    clash = _RESERVED & set(fields)
    if clash:
        raise ValueError(f"encode computes these — don't pass them: {sorted(clash)}")
    underscored = sorted(k for k in fields if isinstance(k, str) and k.startswith("_"))
    if underscored:
        raise ValueError(
            f"`_`-prefixed keys are reserved for decoder internals and are NOT "
            f"hashed (they'd be unprotected): {underscored}")

    record = dict(fields)
    record["hash_version"] = OPEN

    # Identity. Content-addressed from YOUR fields (stable whether or not you
    # sign), unless you supplied one. The bar carries a CANONICAL
    # ``<prefix>-<16 hex>`` identifier, so an override must take that shape.
    ident = identifier
    if ident is None:
        _validate_prefix(prefix)
        ident = _content_identifier(fields, prefix)
    else:
        pre, sep, idhex = ident.rpartition("-")
        if not (sep and len(idhex) == 16
                and all(c in "0123456789abcdef" for c in idhex)):
            raise ValueError(
                f"identifier must be canonical <prefix>-<16 lower-hex>, got {ident!r}")
        _validate_prefix(pre)   # a supplied identifier's prefix obeys the same
                                # contract as the prefix= auto-path — one rule.
    record["identifier"] = ident

    # Field visibility — encrypt the private fields behind a password BEFORE the
    # hash, so the proof covers the ciphertext (tamper-evident shell; verifies
    # without the password). The encrypted_fields envelope is AES-256-GCM built
    # from WebCrypto-compatible primitives (PBKDF2-HMAC-SHA256 + AES-GCM), so a
    # browser can implement unlock via SubtleCrypto — the reference decoder web
    # app does; the bundled verify.js is hash-only by design.
    if password is not None:
        from mememage import crypto
        if not crypto.is_encryption_available():
            raise RuntimeError("field encryption needs the cryptography library "
                               "(`pip install mememage[encrypt]`).")
        if private is not None:
            unknown = [k for k in private if k not in fields]
            if unknown:
                raise ValueError(f"private names fields you didn't pass: {sorted(unknown)}")
        priv_keys = list(fields) if private is None else [k for k in private if k in fields]
        priv = {}
        for k in priv_keys:
            priv[k] = record.pop(k)          # leaves the cleartext shell
        if priv:
            record["encrypted_fields"] = crypto.encrypt_field(
                json.dumps(priv, sort_keys=True, separators=(",", ":")), password)
    elif private:
        raise ValueError("private=… needs a password=…")

    # Proof. The content hash covers identity + the public shell + the ciphertext.
    content_hash = hashing.compute_content_hash(record)
    record["content_hash"] = content_hash

    # Bar the pixels in memory (any input form -> a new barred RGB PIL Image).
    barred = bar.embed_into(image, ident, content_hash)

    # Write a lossless PNG, or keep it in memory. The bar is exact pixel data, so
    # any written file must be PNG (a lossy save would scramble it).
    import os
    written = None
    if out is not None:
        if isinstance(out, (str, os.PathLike)):
            if not str(out).lower().endswith(".png"):
                raise ValueError("encode writes a lossless PNG (the bar can't survive "
                                 f"lossy formats); out must end in .png, got {out}")
            barred.save(out, "PNG")
            written = str(out)
        else:                                   # file-like (e.g. BytesIO)
            barred.save(out, "PNG")
    elif isinstance(image, (str, os.PathLike)):
        # Path input, no out: write in place (PNG) or to a `.png` sibling.
        src = str(image)
        written = src if src.lower().endswith(".png") else _swap_to_png(src)
        barred.save(written, "PNG")
    # else: in-memory input, no out -> Record.image only (no disk).

    return Record(record=record, image_path=written, image=barred)


def decode(image, all_bars=False) -> "Bar | None | list[Bar]":
    """Read the bar's payload out of an image. The inverse of :func:`encode`.

    By default returns the FIRST (bottom-most) bar as ``Bar(identifier,
    content_hash)``, or None — the common case: one image, one record.

    With ``all_bars=True`` returns a LIST of *every* bar in the image (empty if
    none) — for images stamped with more than one. Each bar is located wherever
    it sits: the bottom (where :func:`encode` writes it), a different height,
    horizontally offset, or pasted in from another image. A spurious entry
    would have to defeat the magic bytes, CRC-16, and Reed-Solomon checks at
    once, so bar-ish content is rejected rather than misread.

    ``image`` is anything in memory or on disk — a path, raw ``bytes``, a
    file-like object, a PIL ``Image``, or a numpy array. No network, no disk
    round-trip: just the values stamped in the pixels.

    The identifier points to a record you store separately; resolving it is yours
    (a dict, a file, a DB, a URL — core doesn't fetch). The content hash lets
    :func:`verify` confirm that record against the image."""
    if all_bars:
        return [Bar(identifier=i, content_hash=h) for (i, h) in bar.extract_bars(image)]
    result = bar.extract_bar(image)
    if not result:
        return None
    identifier, content_hash = result
    return Bar(identifier=identifier, content_hash=content_hash)


def is_encrypted(record) -> bool:
    """True if the record carries an ``encrypted_fields`` envelope (private
    fields behind a password)."""
    rec = record.record if isinstance(record, Record) else (record or {})
    return bool(rec.get("encrypted_fields"))


def unlock(record, password) -> dict:
    """Decrypt an encrypted record's private fields and return the full readable
    view (public fields + the decrypted private fields; ``encrypted_fields``
    dropped).

    The ENCRYPTED record is what you :func:`verify` — its hash is over the
    ciphertext. This is the *readable* view for display; don't re-hash it. A
    record with no ``encrypted_fields`` is returned unchanged. Raises ``ValueError``
    on the wrong password.
    """
    rec = record.record if isinstance(record, Record) else dict(record)
    env = rec.get("encrypted_fields")
    if not env:
        return dict(rec)
    from mememage import crypto
    private = json.loads(crypto.decrypt_field(env, password))
    view = {k: v for k, v in rec.items() if k != "encrypted_fields"}
    view.update(private)
    return view


def verify(image, record) -> Verification:
    """Verify a record against an image. Returns a :class:`Verification` (truthy iff
    the data matches — the re-hashed record equals the content hash in the bar).

    Reads the bar, recomputes the content hash over the record, and compares. A
    match means the data is intact and belongs to this image. (Core verifies
    integrity only; authorship/signatures are out of scope.)

    Args:
        image: the barred image — a path, bytes, a file-like, a PIL Image, or a
            numpy array.
        record: the record (a dict) or a :class:`Record`.
    """
    rec = record.record if isinstance(record, Record) else record

    bar_data = decode(image)
    if bar_data is None:
        return Verification(False, "no Mememage bar in the image")

    recomputed = hashing.compute_content_hash(rec)
    if recomputed != bar_data.content_hash:
        return Verification(False, f"hash mismatch: image bar says {bar_data.content_hash}, "
                              f"data recomputes to {recomputed}")
    return Verification(True)



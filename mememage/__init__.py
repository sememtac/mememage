"""Mememage — a bar in an image's pixels points to a JSON record.

Stamp a 2-pixel bar into an image; the bar carries an **identifier** (a key to a
JSON record stored separately) and a **content hash** (a 64-bit digest — the first
16 hex of SHA-256 — over the record).
The core API is three functions, all pure image operations:

- ``encode(image, fields)`` — write the bar, build the record from your fields.
- ``decode(image)``         — read the bar back: identifier + content hash.
- ``verify(image, record)`` — does a record match the image, by hash?

Resolving the record (a dict, a file, a DB, a URL) is the caller's — core does not
fetch. Optional field encryption: ``encode(password=…, private=[…])``
(AES-256-GCM), revealed with ``unlock``.

``pip install mememage`` (Pillow included); add ``[encrypt]`` for field encryption.
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("mememage")
except Exception:  # running from a source tree that isn't installed —
    # a sentinel, not a real version (pyproject.toml is the source of truth)
    __version__ = "0.0.0"

from mememage.api import (
    Bar, Record, Verification, decode, encode, is_encrypted, unlock, verify,
)

__all__ = [
    "encode",
    "decode",
    "verify",
    "unlock",
    "is_encrypted",
    "Record",
    "Bar",
    "Verification",
]

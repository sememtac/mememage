"""The content-hash kernel — canonical JSON + SHA-256, the ``open`` model.

A record's content hash is SHA-256 of its canonical JSON (sorted keys, normalized
floats, no whitespace), first 16 hex chars. This module is the generic, single
source of that computation — the same bytes the browser verifier hashes, so Python
and JS can't drift.

``open`` hash model: hash **every** field except the structurally-circular pair
(``content_hash``, ``signature``) and ``_``-prefixed keys (caller-private fields).
Whatever you put in the record is protected. An application can define its own
``hash_version`` with a curated inclusion set; ``open`` is the generic default.
"""
import hashlib
import json

OPEN_HASH_VERSION = "open"
_HASH_EXCLUDED_OPEN = {"content_hash", "signature"}


def _normalize_for_hash(obj):
    """Recursively normalize values to match JS JSON.stringify behavior."""
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_for_hash(v) for v in obj]
    if isinstance(obj, float):
        # Guard against NaN/Infinity which would break canonical JSON
        if obj != obj or obj == float('inf') or obj == float('-inf'):
            return 0
        # Whole floats to ints (1.0 → 1) to match JS JSON.stringify
        if obj == int(obj):
            return int(obj)
        # Non-integer floats pass through unchanged — Python 3.1+ and modern JS
        # both use shortest-representation, so common values (0.75, 0.8, 3.5) are
        # identical. Do NOT round here — that would break existing records.
    return obj


def hash_fields(hashable: dict, stored: "str | None" = None) -> str:
    """Hash an already-filtered field dict → 16 hex chars.

    Canonical JSON (sorted keys, normalized floats, no whitespace) → SHA-256 →
    first 16 hex. If ``stored`` is given and the normalized hash doesn't match it,
    retry with un-normalized JSON (records minted before float normalization) and
    return that if it matches — so historical records still verify.
    """
    normalized = _normalize_for_hash(hashable)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True)
    result = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    if stored and result != stored:
        raw_canonical = json.dumps(hashable, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True)
        raw_result = hashlib.sha256(raw_canonical.encode("utf-8")).hexdigest()[:16]
        if raw_result == stored:
            return raw_result
    return result


def open_hashable_fields(record: dict) -> dict:
    """The ``open`` model's hashable subset: every field except the circular pair
    and ``_``-prefixed keys."""
    return {k: v for k, v in record.items()
            if k not in _HASH_EXCLUDED_OPEN and not k.startswith("_")}


def compute_content_hash(record: dict) -> str:
    """Content hash of a record under the ``open`` model (the generic core path).

    SHA-256 of the canonical JSON over every field except the circular pair and
    ``_``-keys; first 16 hex. This is what ``encode`` bakes into the bar and what
    ``verify`` recomputes.
    """
    return hash_fields(open_hashable_fields(record), record.get("content_hash"))

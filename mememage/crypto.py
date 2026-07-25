"""Field encryption primitives — AES-256-GCM via PBKDF2-HMAC-SHA256.

The core crypto used by ``encode(password=…)`` / ``unlock``: turn a string into a
``{salt, iv, ct, tag}`` envelope under a passphrase, and back. Pure and generic —
no record schema, no field semantics. ``cryptography`` is imported lazily, so
importing this module costs nothing until you actually encrypt.

The KDF cost (600k iterations, OWASP 2024) is the only knob: it sets how expensive
each guess is. No password policy — any passphrase is accepted.
"""
import os
import unicodedata

_PBKDF2_ITERATIONS = 600_000  # OWASP 2024 recommendation for SHA-256

# An envelope written before the count was recorded was sealed at 600k. Never
# change this: it is what opens those records.
_LEGACY_ITERATIONS = 600_000

# A hostile envelope could ask for a billion iterations and hang whoever opens
# it. The cap is generous enough for any plausible future raise of the OWASP
# figure, and a wall against that denial of service.
_MAX_ITERATIONS = 10_000_000


def is_encryption_available() -> bool:
    """True if the cryptography library is installed (the ``[encrypt]`` extra)."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
        return True
    except ImportError:
        return False


def normalize_password(password: str) -> str:
    """NFC-normalize a passphrase before it becomes key material.

    A passphrase is bytes to PBKDF2, and the same visible text can arrive as
    different code points: "café" is 4 characters composed (NFC) and 5 decomposed
    (NFD), and macOS input paths and filesystems produce either. Without this, a
    password that looks identical fails to unlock on another machine or browser.

    NFC is what RFC 8265's OpaqueString profile (the IETF profile for passwords)
    prescribes: accept all printable Unicode, preserve case, normalize to NFC. It
    is a no-op for ASCII, so no existing record is affected — every envelope
    sealed so far used an ASCII passphrase.

    The JS SDK normalizes identically (String.prototype.normalize("NFC")), and
    chains.py's password verifier uses this same helper, so the gate and the lock
    can never disagree.
    """
    return unicodedata.normalize("NFC", password)


def _derive_key(password: str, salt: bytes, iterations: int = _PBKDF2_ITERATIONS) -> bytes:
    """PBKDF2-HMAC-SHA256, returns 32-byte AES key."""
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(normalize_password(password).encode("utf-8"))


def _iterations_of(envelope: dict) -> int:
    """The KDF cost this envelope was sealed at.

    The count is stored per envelope so the constant can be raised later without
    stranding a single existing record — the same reason ``hash_version`` travels
    with a record. An envelope from before this field was added was sealed at
    600k, so that is the default; it must never change.
    """
    if "iterations" not in envelope:
        return _LEGACY_ITERATIONS
    n = envelope["iterations"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"encrypted envelope has a bad iteration count: {n!r}")
    if n > _MAX_ITERATIONS:
        raise ValueError(
            f"encrypted envelope asks for {n} PBKDF2 iterations, above the "
            f"{_MAX_ITERATIONS} ceiling — refusing to open it")
    return n


def encrypt_field(plaintext: str, password: str) -> dict:
    """AES-256-GCM encrypt a string.

    Returns {"salt": hex, "iv": hex, "ct": hex, "tag": hex, "iterations": int}.
    Salt is 16 bytes (for PBKDF2), IV is 12 bytes (AES-GCM standard). The KDF cost
    travels WITH the envelope so it can be raised later without stranding records.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = os.urandom(16)
    iv = os.urandom(12)
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    ct_with_tag = aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
    # AES-GCM appends 16-byte tag to ciphertext
    ct = ct_with_tag[:-16]
    tag = ct_with_tag[-16:]
    return {
        "salt": salt.hex(),
        "iv": iv.hex(),
        "ct": ct.hex(),
        "tag": tag.hex(),
        "iterations": _PBKDF2_ITERATIONS,
    }


def decrypt_field(envelope: dict, password: str) -> str:
    """AES-256-GCM decrypt. Returns plaintext string.

    Raises ValueError on wrong password (auth tag failure).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag

    # Check the shape before touching it. A malformed envelope used to escape as
    # whatever Python happened to raise — TypeError("'NoneType' object is not
    # subscriptable"), KeyError('salt'), a fromhex complaint — which tells the
    # caller nothing and contradicts the documented ValueError contract.
    if not isinstance(envelope, dict):
        raise ValueError(
            f"encrypted envelope must be a dict of hex strings "
            f"{{salt, iv, ct, tag}}; got {type(envelope).__name__}")
    missing = [k for k in ("salt", "iv", "ct", "tag") if k not in envelope]
    if missing:
        raise ValueError(f"encrypted envelope is missing {', '.join(missing)}")
    try:
        salt = bytes.fromhex(envelope["salt"])
        iv = bytes.fromhex(envelope["iv"])
        ct = bytes.fromhex(envelope["ct"])
        tag = bytes.fromhex(envelope["tag"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"encrypted envelope is not valid hex: {exc}") from None
    key = _derive_key(password, salt, _iterations_of(envelope))
    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(iv, ct + tag, None)
    except InvalidTag:
        raise ValueError("Wrong password — decryption failed")
    return plaintext.decode("utf-8")

"""Field encryption primitives — AES-256-GCM via PBKDF2-HMAC-SHA256.

The core crypto used by ``encode(password=…)`` / ``unlock``: turn a string into a
``{salt, iv, ct, tag}`` envelope under a passphrase, and back. Pure and generic —
no record schema, no field semantics. ``cryptography`` is imported lazily, so
importing this module costs nothing until you actually encrypt.

The KDF cost (600k iterations, OWASP 2024) is the only knob: it sets how expensive
each guess is. No password policy — any passphrase is accepted.
"""
import os

_PBKDF2_ITERATIONS = 600_000  # OWASP 2024 recommendation for SHA-256


def is_encryption_available() -> bool:
    """True if the cryptography library is installed (the ``[encrypt]`` extra)."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
        return True
    except ImportError:
        return False


def _derive_key(password: str, salt: bytes) -> bytes:
    """PBKDF2-HMAC-SHA256, returns 32-byte AES key."""
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


def encrypt_field(plaintext: str, password: str) -> dict:
    """AES-256-GCM encrypt a string.

    Returns {"salt": hex, "iv": hex, "ct": hex, "tag": hex}.
    Salt is 16 bytes (for PBKDF2), IV is 12 bytes (AES-GCM standard).
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
    }


def decrypt_field(envelope: dict, password: str) -> str:
    """AES-256-GCM decrypt. Returns plaintext string.

    Raises ValueError on wrong password (auth tag failure).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag

    salt = bytes.fromhex(envelope["salt"])
    iv = bytes.fromhex(envelope["iv"])
    ct = bytes.fromhex(envelope["ct"])
    tag = bytes.fromhex(envelope["tag"])
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(iv, ct + tag, None)
    except InvalidTag:
        raise ValueError("Wrong password — decryption failed")
    return plaintext.decode("utf-8")

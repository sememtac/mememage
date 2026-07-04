# Mememage

Encode an identifier into an image's pixels; verify a JSON record against any copy.

Mememage writes a 2-pixel-tall bar into the bottom rows of an image. The bar holds two values:

- **identifier** — a short string that points to a JSON record, stored separately (a server, a CDN, IPFS, a file).
- **content hash** — a SHA-256 over the record. `verify` recomputes it and compares against the bar.

The bar survives JPEG, resaves, screenshots, and re-uploads, so the identifier reads back from any copy. `encode` reads any image Pillow can open and writes a lossless PNG; `decode` and `verify` work on any format the bar survives. Record fields are arbitrary — captions, credits, generation parameters, links.

```bash
pip install mememage                 # encode / decode / verify — Pillow included
# pip install "mememage[encrypt]"    # adds AES-256 field encryption
```

## Quickstart

Three functions, all pure image operations:

```python
import mememage

# encode — write the bar into the image, build the record from your fields
result = mememage.encode("photo.png", {"title": "Morning fog", "by": "catmemes"})
result.identifier            # 'mememage-3dc5f03a747bb38e' (derived from the fields)
result.save("photo.json")    # the record — store or serve it separately

# decode — read the bar back out of the pixels (the inverse of encode)
bar = mememage.decode("photo.jpg")      # any format the bar survived: PNG, JPEG, a screenshot
bar.identifier, bar.content_hash

# verify — does a record match an image? (recomputed hash == the bar's)
mememage.verify("photo.jpg", result.record)        # True if the record is intact
```

You resolve the record from its identifier — look it up wherever you keep it, then verify:

```python
bar    = mememage.decode("photo.jpg")    # identifier + content hash from the pixels
record = my_store[bar.identifier]        # your storage: a dict, a file, a DB, a URL
mememage.verify("photo.jpg", record)     # True if the record matches the image
```

- **`encode` accepts any image** — a path, `bytes`, a PIL `Image`, or a numpy array (HEIC needs the `[heic]` extra) — and returns the barred image as `Record.image`. Given a destination — a path (in place, or a `.png` sibling for non-PNG), `out=<path.png>`, or `out=<stream>` (e.g. `BytesIO`) — it writes the file. Output is always PNG: the bar is lossless, and a lossy re-encode would corrupt it. An in-memory input with no destination never touches disk.
- **`decode` / `verify` accept the same in-memory forms** — a path, `bytes`, a file-like, a PIL `Image`, or a numpy array. No disk round-trip.
- **No network I/O** — `decode` returns the identifier; you resolve the record. Core is pure pixel + hash operations.

## Encrypt private fields

- Mark fields `private` to encrypt them (AES-256-GCM via PBKDF2) under a password.
- The record still **verifies without the password** — the hash covers the ciphertext.
- `unlock` returns the decrypted fields. The password is not stored; only the ciphertext is kept in the record.

```python
result = mememage.encode("photo.png", {"title": "Public", "gps": "45.5,-122.6"},
                         password="hunter2", private=["gps"])
mememage.verify("photo.png", result.record)              # matches — no password
mememage.unlock(result, "hunter2")["gps"]                # '45.5,-122.6'
```

## Command line

```bash
mememage encode photo.png --field title="Morning fog" -o photo.json   # write the record
mememage decode photo.jpg --record photo.json                         # VERIFIED (exit 0) / ALTERED (exit 1)
mememage decode photo.jpg                                             # read the identifier only (no record)
```

Without `-o`, the record is written next to the image as `<identifier>.json`.
`decode` exits 0 on a match.

## License

MIT.

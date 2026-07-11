<p align="center">
  <img src="https://mememage.art/img/mememage-icon.png" width="112" alt="Mememage">
</p>

# Mememage

Encode an identifier into an image's pixels; verify a JSON record against any copy.

Mememage writes a 2-pixel-tall bar into the bottom of an image. It carries an **identifier** (a short pointer to a JSON record you store anywhere) and a **content hash** (first 16 hex of SHA-256 over that record). `verify` recomputes the hash and compares it to the bar. Change any field, verification fails.

Core proves the record-to-image *binding*, by hash alone. It does **not** police the pixels (edit the image but leave the bar and it still verifies) and does **not** prove authorship (a signature's job, out of scope). Two keys stay outside the hash by design, `signature` and `_`-prefixed keys, and `encode` refuses both as your own field names.

The bar survives JPEG, resaves, screenshots, and re-uploads. **Downscaling is the limit:** images ≥ ~1000px wide survive a shrink to ~0.8× plus one recompression (59/60 real-image round-trips across three resamplers and JPEG q70 to q80). Past that, no promise.

```bash
pip install mememage                 # encode / decode / verify (Pillow included)
# pip install "mememage[encrypt]"    # adds AES-256 field encryption
```

## Quickstart

```python
import mememage

# encode: write the bar, build a record from your fields
result = mememage.encode("photo.png", {"title": "Morning fog", "by": "catmemes"})
result.identifier            # 'mememage-3dc5f03a747bb38e'  (derived from your fields)
result.save("photo.json")    # a record, stored or served separately

# decode: read the bar back out of any copy (PNG, JPEG, a screenshot)
bar = mememage.decode("photo.jpg")
bar.identifier, bar.content_hash

# verify: does a record match an image?
mememage.verify("photo.jpg", result.record)        # truthy if intact
```

Core does no networking. `decode` hands you an identifier; resolve the record wherever you kept it (a dict, a file, a DB, a URL), then `verify`.

**Inputs / outputs.** `encode`, `decode`, and `verify` accept a path, `bytes`, a file-like, a PIL `Image`, or a numpy array (HEIC needs the `[heic]` extra). `encode` returns a barred `Record.image` and, given a destination, writes a lossless **PNG** (in place for a PNG path, a `.png` sibling otherwise, or `out=<path/stream>`); an in-memory input with no destination never touches disk. Record fields are yours (captions, credits, generation params, links) except a few reserved names: `identifier`, `content_hash`, `hash_version`, `signature`, `encrypted_fields`.

## Encrypt private fields

Mark fields `private` to encrypt them (AES-256-GCM via PBKDF2) under a password. It still **verifies without the password** (the hash covers the ciphertext), and `unlock` reveals the fields. Passwords are never stored.

```python
result = mememage.encode("photo.png", {"title": "Public", "gps": "45.5,-122.6"},
                         password="hunter2", private=["gps"])
mememage.verify("photo.png", result.record)              # matches, no password
mememage.unlock(result, "hunter2")["gps"]                # '45.5,-122.6'
```

## Command line

```bash
mememage encode photo.png --field title="Morning fog" -o photo.json   # write the record
mememage decode photo.jpg --record photo.json                         # VERIFIED (0) / RECORD ALTERED (1)
mememage decode photo.jpg                                             # read the identifier only
```

Without `-o`, the record lands beside the image as `<identifier>.json`. With `--record`, `decode` exits 0 on a match, 1 on a mismatch; without it, exit 0 just means a bar was read.

## License

MIT.

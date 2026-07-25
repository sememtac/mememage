<p align="center">
  <img src="https://mememage.art/img/mememage-icon.png" width="112" alt="Mememage">
</p>

# Mememage

Encode an identifier into the pixels of an image. Verify a JSON record against any copy.

Mememage writes a bar into the bottom of an image. The bar is 2 pixels tall. It holds an **identifier** and a **content hash**. The identifier is a short pointer to a JSON record that you store anywhere. The content hash is the first 16 hex characters of the SHA-256 of that record. `verify` recomputes the hash and compares it to the bar. If you change any field, verification fails.

Core proves the link between a record and an image, by hash alone. It does **not** check the pixels. You can edit the image, and if the bar stays, the image still verifies. Core does **not** prove authorship. That is the job of a signature, which is out of scope. Two keys stay outside the hash by design: `signature` and `_`-prefixed keys. `encode` does not let you use either name for your own field.

The bar survives JPEG, resaves, screenshots, and re-uploads. **Downscaling is the limit.** An image that is about 1000 px wide or more survives a shrink to about 0.8x and one re-compression (59 of 60 real-image round-trips, across three resamplers and JPEG q70 to q80). Past that limit, Mememage makes no promise.

```bash
pip install mememage                 # encode / decode / verify (Pillow included)
# pip install "mememage[encrypt]"    # adds AES-256 field encryption
```

## Quickstart

```python
import mememage

# encode: write the bar and build a record from your fields
result = mememage.encode("photo.png", {"title": "Morning fog", "by": "catmemes"})
result.identifier            # 'mememage-3dc5f03a747bb38e'  (from your fields)
result.save("photo.json")    # store or serve the record separately

# decode: read the bar from any copy (PNG, JPEG, a screenshot)
bar = mememage.decode("photo.jpg")
bar.identifier, bar.content_hash

# verify: does a record match an image?
mememage.verify("photo.jpg", result.record)        # truthy if the record is intact
```

Core does no networking. `decode` gives you an identifier. Resolve the record where you kept it (a dict, a file, a database, a URL). Then call `verify`.

**Hash models (`hash_version`).** Core implements the **`open`** model. It hashes every field except `content_hash` and `signature`. So the record is tamper-evident, whatever you put in it. `encode` stamps `hash_version: "open"`. An application can define its own `hash_version` with a fixed set of fields. Core does not implement those models. For such a record, `verify` reports **unsupported** (`Verification.supported == False`), not a hash mismatch. It fails closed (`bool()` is `False`), but this is *not* evidence of tampering. Verify those records with the application that defines the version, for example its own decoder. The CLI prints `UNSUPPORTED` and exits `3`.

**Inputs and outputs.** `encode`, `decode`, and `verify` accept a path, `bytes`, a file-like object, a PIL `Image`, or a numpy array. (HEIC needs the `[heic]` extra.) `encode` returns a barred `Record.image`. Given a destination, it writes a lossless **PNG**: in place for a PNG path, a `.png` sibling for any other path, or to `out=<path or stream>`. An in-memory input with no destination never touches the disk. The record fields are yours (captions, credits, generation parameters, links). A few names are reserved: `identifier`, `content_hash`, `hash_version`, `signature`, and `encrypted_fields`.

## Encrypt private fields

Mark fields as `private` to encrypt them under a password (AES-256-GCM via PBKDF2). The record still **verifies without the password**, because the hash covers the ciphertext. `unlock` reveals the fields. Mememage never stores the password.

```python
result = mememage.encode("photo.png", {"title": "Public", "gps": "45.5,-122.6"},
                         password="hunter2", private=["gps"])
mememage.verify("photo.png", result.record)              # matches, no password
mememage.unlock(result, "hunter2")["gps"]                # '45.5,-122.6'
```

## Command line

```bash
mememage encode photo.png --field title="Morning fog" -o photo.json   # write the record
mememage decode photo.jpg --record photo.json                         # VERIFIED (0) / RECORD ALTERED (1) / UNSUPPORTED (3)
mememage decode photo.jpg                                             # read the identifier only
```

Without `-o`, the record lands beside the image as `<identifier>.json`. With `--record`, `decode` exits 0 on a match and 1 on a mismatch. Without `--record`, exit 0 means only that a bar was read.

## License

MIT.

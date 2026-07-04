"""Mememage core CLI — `encode` / `decode` from a shell.

    mememage encode photo.png --field title="Morning fog" -o photo.json  # write the record
    mememage decode photo.jpg --record photo.json                        # VERIFIED / ALTERED

Without -o, the record is written next to the image as <identifier>.json.
`decode` exits 0 only on a match, so it drops straight into a CI gate.
"""
import argparse
import sys


def _resolve_password(env_name, prompt):
    """A password WITHOUT putting it in argv (visible in `ps` / shell history).
    From ``--password-env VAR`` if given, else an interactive getpass prompt on a
    TTY, else an error (non-interactive needs the env var)."""
    import getpass
    import os
    if env_name:
        val = os.environ.get(env_name)
        if not val:
            print(f"Error: env var {env_name} is unset/empty", file=sys.stderr)
            sys.exit(1)
        return val
    if sys.stdin.isatty():
        return getpass.getpass(prompt)
    print("Error: no password — pass --password-env VAR (non-interactive)",
          file=sys.stderr)
    sys.exit(1)


def cmd_encode(args):
    """Write a bar + build an open-hash record from arbitrary fields."""
    import json as _json
    import os
    import mememage

    fields = {}
    if args.fields:
        try:
            src = sys.stdin.read() if args.fields == "-" else \
                open(args.fields, encoding="utf-8").read()
            loaded = _json.loads(src)
        except Exception as e:
            print(f"Error reading --fields: {e}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(loaded, dict):
            print("Error: --fields must be a JSON object", file=sys.stderr)
            sys.exit(1)
        fields.update(loaded)
    for kv in (args.field or []):
        if "=" not in kv:
            print(f"Error: --field expects KEY=VALUE, got {kv!r}", file=sys.stderr)
            sys.exit(1)
        k, v = kv.split("=", 1)
        fields[k] = v

    # Field visibility — encrypt private fields behind a password.
    password = None
    private = None
    if args.encrypt or args.private:
        password = _resolve_password(args.password_env, "Encrypt password: ")
        if args.private:
            private = [k.strip() for k in args.private.split(",") if k.strip()]

    try:
        result = mememage.encode(args.image, fields, prefix=args.prefix,
                                 identifier=args.identifier,
                                 password=password, private=private)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Name the record by its IDENTIFIER (not the image), so a record is found
    # by the code the bar carries. Plain .json for the core; .soul is the
    # provenance chain's. Matches the main CLI.
    out = args.out or os.path.join(
        os.path.dirname(args.image), result.identifier + ".json")
    result.save(out)
    print(f"Encoded {args.image}")
    print(f"  image:        {result.image_path}")
    print(f"  identifier:   {result.identifier}")
    print(f"  content hash: {result.content_hash}")
    if result.record.get("encrypted_fields"):
        print(f"  encrypted:    {len(private) if private else 'all'} field(s)")
    print(f"  record:       {out}")


def cmd_decode(args):
    """Read the bar (identifier + content hash). With --record, also verify it."""
    import json as _json
    import mememage

    bar = mememage.decode(args.image)
    if bar is None:
        print("No Mememage bar in the image.", file=sys.stderr)
        sys.exit(1)

    # Pure read.
    if not args.record:
        if args.json:
            print(_json.dumps({"identifier": bar.identifier,
                               "content_hash": bar.content_hash}))
        else:
            print(f"Bar:  {bar.identifier}")
            print(f"Hash: {bar.content_hash}")
        return

    # --record: verify against a LOCAL record file (no network — resolving is yours).
    try:
        with open(args.record, encoding="utf-8") as f:
            record = _json.load(f)
    except Exception as e:
        print(f"Error reading --record: {e}", file=sys.stderr)
        sys.exit(2)
    v = mememage.verify(args.image, record)

    if args.json:
        print(_json.dumps({"identifier": bar.identifier, "content_hash": bar.content_hash,
                           "match": bool(v), "reason": v.reason}))
        sys.exit(0 if v else 1)

    print(f"Bar:  {bar.identifier}")
    print(f"Hash: {bar.content_hash}")
    if v:
        print("VERIFIED — record matches the image")
        if record.get("encrypted_fields"):
            if args.unlock or args.password_env:
                password = _resolve_password(args.password_env, "Unlock password: ")
                try:
                    view = mememage.unlock(record, password)
                    print("UNLOCKED — private fields decrypted:")
                    _core = ("identifier", "content_hash", "hash_version",
                             "signature", "encrypted_fields")
                    for k, val in sorted(view.items()):
                        if not (k.startswith("_") or k in _core):
                            print(f"  {k}: {val}")
                except Exception:
                    print("(wrong password — could not decrypt)")
            else:
                print("ENCRYPTED — private fields (pass --unlock / --password-env to reveal)")
    else:
        print(f"ALTERED — {v.reason}")
    sys.exit(0 if v else 1)


def main():
    p = argparse.ArgumentParser(prog="mememage",
                                description="Encode an identifier into an image's pixels; "
                                            "verify a JSON record against any copy.")
    sub = p.add_subparsers(dest="command")

    pe = sub.add_parser("encode", help="Encode a bar + build a record from your fields")
    pe.add_argument("image", help="PNG image to encode (modified in place)")
    pe.add_argument("--field", action="append", metavar="KEY=VALUE",
                    help="A record field (repeatable). String values; use --fields for typed/nested.")
    pe.add_argument("--fields", metavar="JSON_FILE",
                    help="Read record fields from a JSON object file ('-' for stdin)")
    pe.add_argument("--prefix", default="mememage",
                    help="Identifier prefix (default: mememage)")
    pe.add_argument("--identifier", help="Override the content-addressed identifier")
    pe.add_argument("--encrypt", action="store_true",
                    help="Encrypt ALL fields behind a password (private record)")
    pe.add_argument("--private", metavar="F1,F2",
                    help="Encrypt only these comma-separated fields (rest public)")
    pe.add_argument("--password-env", metavar="VAR",
                    help="Read the encrypt password from this env var (else prompt)")
    pe.add_argument("-o", "--out",
                    help="Record output path (default: <identifier>.json next to the image)")

    pd = sub.add_parser("decode", help="Read the bar (identifier + content hash); with --record, verify")
    pd.add_argument("image", help="Image to decode (PNG, JPEG, screenshot)")
    pd.add_argument("--record", dest="record", metavar="FILE",
                    help="A local record file (JSON) to verify the image against")
    pd.add_argument("--unlock", action="store_true",
                    help="With --record: decrypt the record's private fields (prompts for password)")
    pd.add_argument("--password-env", metavar="VAR",
                    help="Read the unlock password from this env var (else prompt)")
    pd.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    args = p.parse_args()
    try:
        if args.command == "encode":
            cmd_encode(args)
        elif args.command == "decode":
            cmd_decode(args)
        else:
            p.print_help()
            sys.exit(1)
    except BrokenPipeError:
        # A downstream reader (head / less / grep -q) closed the pipe early.
        # Redirect stdout to devnull so the interpreter's final flush doesn't
        # re-raise, then exit cleanly. The standard CLI idiom (Python docs).
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)


if __name__ == "__main__":
    main()

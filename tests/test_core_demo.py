"""Run the shipped core-use demo (examples/core_quickstart.sh) as a test.

The demo is what a programmer runs to validate the raw core from the shell —
encode an image with a record, decode/verify it, prove tamper detection and
JPEG survival. Running it here keeps it from silently rotting: if the CLI or
the core breaks, the demo's `set -e` makes this fail.
"""

import os
import shutil
import subprocess
import unittest

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "examples", "core_quickstart.sh")
_BASH = shutil.which("bash")

try:
    from PIL import Image  # noqa: F401
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@pytest.mark.skipif(_BASH is None, reason="bash not available")
@pytest.mark.skipif(not HAS_PIL, reason="Pillow required (the bar codec)")
@pytest.mark.skipif(not os.path.exists(DEMO), reason="demo script missing")
class TestCoreDemo(unittest.TestCase):
    def test_demo_runs_green(self):
        # Inherit the env (editable install → `mememage` / `python3` on PATH are
        # this checkout); just ensure the repo is importable for `python3 -m`.
        env = dict(os.environ)
        repo = os.path.dirname(HERE)
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([_BASH, DEMO], capture_output=True, text=True,
                                timeout=120, env=env)
        self.assertEqual(result.returncode, 0,
                         f"demo failed:\nSTDOUT:\n{result.stdout}\n"
                         f"STDERR:\n{result.stderr}")
        self.assertIn("CORE VALIDATED", result.stdout)
        # The load-bearing steps each announce themselves.
        for marker in ("VERIFIED", "tamper detected", "verified from a JPEG"):
            self.assertIn(marker, result.stdout, f"missing step: {marker}")


if __name__ == "__main__":
    unittest.main()

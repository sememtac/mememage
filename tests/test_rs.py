"""Tests for the pure Python Reed-Solomon codec."""

import os
import unittest

from mememage.rs import rs_encode, rs_decode


class TestRoundTrip(unittest.TestCase):
    def test_encode_decode_no_errors(self):
        data = b"Hello, Reed-Solomon!"
        nsym = 6
        encoded = rs_encode(data, nsym)
        assert len(encoded) == len(data) + nsym
        decoded = rs_decode(encoded, nsym)
        assert decoded == data

    def test_empty_data(self):
        data = b""
        encoded = rs_encode(data, 6)
        decoded = rs_decode(encoded, 6)
        assert decoded == data

    def test_single_byte(self):
        data = b"\x42"
        encoded = rs_encode(data, 6)
        decoded = rs_decode(encoded, 6)
        assert decoded == data

    def test_various_nsym(self):
        data = b"test payload for RS"
        for nsym in (2, 4, 6, 8, 10):
            encoded = rs_encode(data, nsym)
            decoded = rs_decode(encoded, nsym)
            assert decoded == data, f"Failed for nsym={nsym}"

class TestErrorCorrection(unittest.TestCase):
    def _corrupt(self, data, positions):
        """Flip a byte at each position."""
        data = bytearray(data)
        for pos in positions:
            data[pos] ^= 0xFF
        return bytes(data)

    def test_corrects_1_error(self):
        data = b"https://example.com/records/mememage-abc12345/record.json"
        nsym = 6
        encoded = rs_encode(data, nsym)
        corrupted = self._corrupt(encoded, [10])
        decoded = rs_decode(corrupted, nsym)
        assert decoded == data

    def test_corrects_2_errors(self):
        data = b"https://example.com/records/mememage-abc12345/record.json"
        nsym = 6
        encoded = rs_encode(data, nsym)
        corrupted = self._corrupt(encoded, [5, 30])
        decoded = rs_decode(corrupted, nsym)
        assert decoded == data

    def test_corrects_3_errors(self):
        data = b"https://example.com/records/mememage-abc12345/record.json"
        nsym = 6
        encoded = rs_encode(data, nsym)
        corrupted = self._corrupt(encoded, [0, 25, 49])
        decoded = rs_decode(corrupted, nsym)
        assert decoded == data

    def test_fails_on_4_errors(self):
        data = b"https://example.com/records/mememage-abc12345/record.json"
        nsym = 6
        encoded = rs_encode(data, nsym)
        corrupted = self._corrupt(encoded, [0, 10, 20, 30])
        with self.assertRaises(ValueError):
            rs_decode(corrupted, nsym)

    def test_error_in_parity_bytes(self):
        """Errors in parity region should also be correctable."""
        data = b"test data"
        nsym = 6
        encoded = rs_encode(data, nsym)
        # Corrupt last byte (in parity region)
        corrupted = self._corrupt(encoded, [len(encoded) - 1])
        decoded = rs_decode(corrupted, nsym)
        assert decoded == data

    def test_corrects_random_positions(self):
        """Test with random data and random error positions."""
        rng = os.urandom
        data = rng(60)  # typical mememage payload size
        nsym = 6
        encoded = rs_encode(data, nsym)
        # Corrupt 3 random positions
        import random
        random.seed(42)
        positions = random.sample(range(len(encoded)), 3)
        corrupted = self._corrupt(encoded, positions)
        decoded = rs_decode(corrupted, nsym)
        assert decoded == data


class TestEdgeCases(unittest.TestCase):
    def test_all_zeros(self):
        data = bytes(20)
        encoded = rs_encode(data, 6)
        decoded = rs_decode(encoded, 6)
        assert decoded == data

    def test_all_ones(self):
        data = bytes([0xFF] * 20)
        encoded = rs_encode(data, 6)
        decoded = rs_decode(encoded, 6)
        assert decoded == data

    def test_max_length_payload(self):
        """RS can handle up to 255 total bytes (data + parity)."""
        data = os.urandom(249)  # 249 + 6 = 255
        encoded = rs_encode(data, 6)
        decoded = rs_decode(encoded, 6)
        assert decoded == data

    def test_parity_bytes_are_deterministic(self):
        data = b"deterministic test"
        e1 = rs_encode(data, 6)
        e2 = rs_encode(data, 6)
        assert e1 == e2


if __name__ == "__main__":
    unittest.main()

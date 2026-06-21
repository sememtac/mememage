"""Pure Python Reed-Solomon codec over GF(2^8).

Zero external dependencies. Provides forward error correction for the
steganographic bar — corrects up to nsym//2 byte errors in the payload.

Primitive polynomial: 0x11D (x^8 + x^4 + x^3 + x^2 + 1), same as QR codes.
Generator root: alpha = 0x02.
"""

# ---------------------------------------------------------------------------
# GF(2^8) arithmetic — log/antilog tables for fast multiply/divide
# ---------------------------------------------------------------------------

_PRIM = 0x11D  # primitive polynomial
_GF_EXP = [0] * 512  # antilog table (doubled for wraparound)
_GF_LOG = [0] * 256  # log table

def _init_tables():
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= _PRIM
    # Double the exp table for easy modular access
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]

_init_tables()


def _gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 255]


def _gf_pow(x, power):
    return _GF_EXP[(_GF_LOG[x] * power) % 255] if x != 0 else 0


def _gf_poly_mul(p, q):
    """Multiply two polynomials in GF(2^8)."""
    r = [0] * (len(p) + len(q) - 1)
    for j, qj in enumerate(q):
        for i, pi in enumerate(p):
            r[i + j] ^= _gf_mul(pi, qj)
    return r


def _gf_poly_eval(poly, x):
    """Evaluate a polynomial at x in GF(2^8)."""
    y = poly[0]
    for coeff in poly[1:]:
        y = _gf_mul(y, x) ^ coeff
    return y


# ---------------------------------------------------------------------------
# Generator polynomial — computed once per nsym value and cached
# ---------------------------------------------------------------------------

_gen_cache = {}

def _generator_poly(nsym):
    """Build the RS generator polynomial for nsym parity symbols."""
    if nsym in _gen_cache:
        return _gen_cache[nsym]
    g = [1]
    for i in range(nsym):
        g = _gf_poly_mul(g, [1, _GF_EXP[i]])
    _gen_cache[nsym] = g
    return g


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def rs_encode(data: bytes, nsym: int) -> bytes:
    """Append nsym parity bytes to data. Returns data + parity."""
    if nsym <= 0:
        return data
    if len(data) + nsym > 255:
        raise ValueError(
            f"RS codeword too large: {len(data)} + {nsym} = {len(data) + nsym} > 255 (GF(256) limit)"
        )
    gen = _generator_poly(nsym)
    # Polynomial long division: data * x^nsym mod gen
    feedback = list(data) + [0] * nsym
    for i in range(len(data)):
        coeff = feedback[i]
        if coeff != 0:
            for j in range(1, len(gen)):
                feedback[i + j] ^= _gf_mul(gen[j], coeff)
    # Parity is the remainder (last nsym bytes)
    parity = feedback[len(data):]
    return bytes(data) + bytes(parity)


# ---------------------------------------------------------------------------
# Decoder — syndrome, Berlekamp-Massey, Chien search, Forney
# ---------------------------------------------------------------------------

def _syndromes(msg, nsym):
    """Compute nsym syndromes. S_i = msg(alpha^i) for i in 0..nsym-1."""
    return [_gf_poly_eval(msg, _GF_EXP[i]) for i in range(nsym)]


def _berlekamp_massey(synd, nsym):
    """Find the error locator polynomial via Berlekamp-Massey.

    Uses the standard iterative form: C is the current error locator,
    B is the previous best, L tracks the current number of errors.
    """
    C = [1]   # error locator polynomial (coefficients, constant term first)
    B = [1]   # copy of C at last length change
    L = 0     # current number of assumed errors
    m = 1     # steps since last length change
    b = 1     # discrepancy at last length change

    for n in range(nsym):
        # Compute discrepancy d
        d = synd[n]
        for i in range(1, L + 1):
            if i < len(C):
                d ^= _gf_mul(C[i], synd[n - i])

        if d == 0:
            m += 1
        elif 2 * L <= n:
            # Need to increase L
            T = list(C)
            coeff = _gf_div(d, b)
            # C(x) = C(x) - (d/b) * x^m * B(x)
            shift = [0] * m
            scaled = shift + [_gf_mul(coeff, bi) for bi in B]
            while len(C) < len(scaled):
                C.append(0)
            for i in range(len(scaled)):
                C[i] ^= scaled[i]
            L = n + 1 - L
            B = T
            b = d
            m = 1
        else:
            # Just update C, don't change L
            coeff = _gf_div(d, b)
            shift = [0] * m
            scaled = shift + [_gf_mul(coeff, bi) for bi in B]
            while len(C) < len(scaled):
                C.append(0)
            for i in range(len(scaled)):
                C[i] ^= scaled[i]
            m += 1

    errs = len(C) - 1
    if errs * 2 > nsym:
        raise ValueError(f"Too many errors to correct ({errs} > {nsym // 2})")
    return C


def _find_errors(err_loc, nmsg):
    """Find error positions by evaluating err_loc at all possible positions.

    err_loc is stored with constant term at index 0:
        sigma(x) = C[0] + C[1]*x + C[2]*x^2 + ...
    An error at position j means sigma(alpha^(-j)) = 0.
    """
    errs = len(err_loc) - 1
    positions = []
    for i in range(nmsg):
        # Evaluate sigma at alpha^(-i) = alpha^(255-i)
        xi = _GF_EXP[255 - i]
        val = err_loc[0]
        for k in range(1, len(err_loc)):
            val ^= _gf_mul(err_loc[k], _gf_pow(xi, k))
        if val == 0:
            positions.append(i)
    if len(positions) != errs:
        raise ValueError("Could not locate all errors")
    return positions


def _solve_error_values(synd, positions):
    """Compute error values by solving the linear system from syndromes.

    Given error positions p_0..p_{k-1}, syndromes satisfy:
        S_i = sum_j(e_j * alpha^(i * p_j))

    This is a k×k linear system in e_j. Solve via Gaussian elimination
    in GF(2^8). Simpler and more robust than Forney for small k.
    """
    k = len(positions)
    if k == 0:
        return {}

    # Build the k×(k+1) augmented matrix [A | S]
    # A[i][j] = alpha^(i * positions[j])
    matrix = []
    for i in range(k):
        row = []
        for j in range(k):
            power = i * positions[j]
            row.append(_GF_EXP[power % 255] if power > 0 else 1)
        row.append(synd[i])
        matrix.append(row)

    # Gaussian elimination with partial pivoting
    for col in range(k):
        # Find pivot
        pivot = None
        for row in range(col, k):
            if matrix[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Singular matrix in error value computation")
        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]

        # Scale pivot row
        inv = _gf_div(1, matrix[col][col])
        for j in range(col, k + 1):
            matrix[col][j] = _gf_mul(matrix[col][j], inv)

        # Eliminate column
        for row in range(k):
            if row != col and matrix[row][col] != 0:
                factor = matrix[row][col]
                for j in range(col, k + 1):
                    matrix[row][j] ^= _gf_mul(factor, matrix[col][j])

    return {positions[j]: matrix[j][k] for j in range(k)}


def rs_decode(data: bytes, nsym: int) -> bytes:
    """Decode an RS codeword. Returns corrected data (parity stripped).

    Raises ValueError if errors are uncorrectable.
    """
    if nsym <= 0:
        return data
    msg = list(data)
    synd = _syndromes(msg, nsym)

    # No errors if all syndromes are zero
    if max(synd) == 0:
        return bytes(msg[:len(msg) - nsym])

    err_loc = _berlekamp_massey(synd, nsym)
    # _find_errors returns polynomial positions (x^k), but array index is (n-1-k)
    poly_positions = _find_errors(err_loc, len(msg))
    magnitudes = _solve_error_values(synd, poly_positions)

    # Convert polynomial positions to array indices and apply corrections
    nmsg = len(msg)
    for poly_pos, mag in magnitudes.items():
        array_pos = nmsg - 1 - poly_pos
        msg[array_pos] ^= mag

    # Verify correction worked
    synd_check = _syndromes(msg, nsym)
    if max(synd_check) != 0:
        raise ValueError("RS decode failed: residual syndromes after correction")

    return bytes(msg[:len(msg) - nsym])

"""Common operators for the end-to-end Hankel tensor experiments.

Conventions (match the manuscript):
  signal X in C^{s x n};  n1 + n2 = n + 1;  0-indexed anti-diagonal a = i + j.
  w[a] = #{(i,j): i+j=a} = min(a+1, n-a, n1, n2).
  Hankel lift      H(X)[i,j,k] = X[k, i+j]
  normalized lift  G(X)[i,j,k] = X[k, i+j] / sqrt(w[i+j]),   G* G = I
  Tucker synthesis Phi(S,U1,U2,U3) = S x1 U1 x2 U2 x3 U3
  mode-1 unfold: rows i, cols i3*n2 + i2  (mode-2/3 by cyclic analogy),
  so  M1(Phi) = U1 M1(S) kron(U3,U2)^T  with numpy.kron (last factor fast).

All routines are dense and dimension-checked.
"""
import numpy as np

# ---------------- weights and lifts ----------------

def weights(n, n1, n2):
    a = np.arange(n)
    return np.minimum.reduce([a + 1, n - a, np.full(n, n1), np.full(n, n2)])

def lift_G(X, n1, n2, w):
    s, n = X.shape
    idx = np.arange(n1)[:, None] + np.arange(n2)[None, :]        # (n1,n2)
    Z = X[:, idx] / np.sqrt(w)[idx]                              # (s,n1,n2)
    return np.ascontiguousarray(np.transpose(Z, (1, 2, 0)))      # (n1,n2,s)

_adj_cache = {}

def adj_G(Z, n, w):
    n1, n2, s = Z.shape
    key = (n1, n2, s)
    if key not in _adj_cache:
        a = (np.arange(n1)[:, None] + np.arange(n2)[None, :]).ravel()
        _adj_cache[key] = (a[:, None] * s + np.arange(s)[None, :]).ravel()
    lin = _adj_cache[key]
    flat = Z.reshape(n1 * n2 * s)
    acc = np.bincount(lin, weights=flat.real, minlength=n * s) \
        + 1j * np.bincount(lin, weights=flat.imag, minlength=n * s)
    return acc.reshape(n, s).T / np.sqrt(w)

def proj_hankel(Z, n, w):
    """P_H = G G*."""
    return lift_G(adj_G(Z, n, w), Z.shape[0], Z.shape[1], w)

# ---------------- Tucker utilities ----------------

def unfold(T, mode):
    """Mode-m unfolding with col index = (slow other) * (fast dim) + fast other,
    fast axis = the next mode cyclically (mode1 cols: i3*n2+i2)."""
    if mode == 0:
        return np.transpose(T, (0, 2, 1)).reshape(T.shape[0], -1)
    if mode == 1:
        return np.transpose(T, (1, 2, 0)).reshape(T.shape[1], -1)
    return np.transpose(T, (2, 1, 0)).reshape(T.shape[2], -1)

def fold(M, mode, shape):
    n1, n2, s = shape
    if mode == 0:
        return np.transpose(M.reshape(n1, s, n2), (0, 2, 1))
    if mode == 1:
        return np.transpose(M.reshape(n2, s, n1), (2, 0, 1))
    return np.transpose(M.reshape(s, n2, n1), (2, 1, 0))

def mode_mul(T, A, mode):
    """T x_mode A  (A acts on that mode's dimension)."""
    Tm = np.moveaxis(T, mode, 0)
    out = np.tensordot(A, Tm, axes=(1, 0))
    return np.moveaxis(out, 0, mode)

def tucker(S, U1, U2, U3):
    return mode_mul(mode_mul(mode_mul(S, U1, 0), U2, 1), U3, 2)

def hosvd_r(Z, r):
    """Truncated HOSVD to multilinear rank (r,r,r)."""
    Us = []
    for m in range(3):
        M = unfold(Z, m)
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        Us.append(U[:, :r])
    core = mode_mul(mode_mul(mode_mul(Z, Us[0].conj().T, 0),
                             Us[1].conj().T, 1), Us[2].conj().T, 2)
    return core, Us[0], Us[1], Us[2]

# ---------------- model and sampling ----------------

def make_signal(n, s, r, h, rng):
    """Equispaced cluster with normalized separation h; unit-norm Gaussian b_l
    (the coefficient convention of Li et al.)."""
    f0 = rng.random()
    f = (f0 + np.arange(r) * (h / n)) % 1.0
    A = np.exp(2j * np.pi * np.outer(np.arange(n), f))            # (n,r)
    B = (rng.standard_normal((s, r)) + 1j * rng.standard_normal((s, r)))
    B /= np.linalg.norm(B, axis=0, keepdims=True)
    return B @ A.T, f, B                                          # X (s,n)

def make_mask(n, s, p, rng):
    m = int(round(p * s * n))
    pick = rng.choice(s * n, size=m, replace=False)
    mask = np.zeros(s * n, bool); mask[pick] = True
    return mask.reshape(s, n)

# ---------------- metrics ----------------

def rel_err(Xh, X):
    return np.linalg.norm(Xh - X) / np.linalg.norm(X)

def circ_dist(a, b):
    d = np.abs(a - b) % 1.0
    return np.minimum(d, 1.0 - d)

def esprit_freqs(Xh, n1, n2, r, w, n):
    """ESPRIT on the PLAIN Hankel lift H(X) (mode-1 range = span V1);
    the normalized lift G would break shift invariance."""
    idx = np.arange(n1)[:, None] + np.arange(n2)[None, :]
    Z = np.ascontiguousarray(np.transpose(Xh[:, idx], (1, 2, 0)))
    _, U1, _, _ = hosvd_r(Z, r)
    A_, B_ = U1[:-1, :], U1[1:, :]
    Psi = np.linalg.lstsq(A_, B_, rcond=None)[0]
    lam = np.linalg.eigvals(Psi)
    return (np.angle(lam) / (2 * np.pi)) % 1.0

def freq_errors(fh, f):
    """Best matching over permutations (r small)."""
    from itertools import permutations
    r = len(f)
    best = None
    for perm in permutations(range(r)):
        d = circ_dist(fh[list(perm)], f)
        key = (d.max(), d.mean())
        if best is None or key < best[0]:
            best = (key, d)
    d = best[1]
    return d.mean(), d.max()

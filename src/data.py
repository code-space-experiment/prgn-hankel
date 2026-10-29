"""Synthetic data of the experiments.

Signal: r = 4 frequencies f_l = f0 + l h / n (mod 1), f0 uniform, with normalized
spacing h = n Delta; s = 32 channels, X = B A^T with A_{t,l} = exp(2 pi i t f_l),
t = 0, ..., n-1.  Coefficient regimes (columns of B normalized to unit norm):
  gauss : independent complex Gaussian columns;
  well  : orthonormal columns;
  ill   : b_l = sqrt(rho) b0 + sqrt(1-rho) g_l, rho = 0.99;
  rank1 : same with rho = 0.9999.
Observation set: m = round(p s n) entries drawn uniformly without replacement.
Noise: complex Gaussian W scaled so that ||W||_F = delta ||X||_F.
"""
import numpy as np
from hankel_common import weights, make_mask

N, N1, N2, S, R = 127, 64, 64, 32, 4


def signal(h, rng, regime):
    f0 = rng.random()
    f = (f0 + np.arange(R) * (h / N)) % 1.0
    A = np.exp(2j * np.pi * np.outer(np.arange(N), f))
    G = rng.standard_normal((S, R)) + 1j * rng.standard_normal((S, R))
    if regime == "gauss":
        B = G
    elif regime == "well":
        B, _ = np.linalg.qr(G)
    else:
        rho = {"ill": 0.99, "rank1": 0.9999}[regime]
        b0 = rng.standard_normal(S) + 1j * rng.standard_normal(S)
        b0 /= np.linalg.norm(b0)
        Gn = G / np.linalg.norm(G, axis=0, keepdims=True)
        B = np.sqrt(rho) * b0[:, None] + np.sqrt(1 - rho) * Gn
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    return B @ A.T, f, B


def instance(h, p, seed, delta=0.0, regime="gauss"):
    """Returns the clean signal X, the observed signal Y, frequencies f, coefficients B,
    the observation set and the anti-diagonal weights."""
    rng = np.random.default_rng(seed); w = weights(N, N1, N2)
    X, f, B = signal(h, rng, regime)
    mask = make_mask(N, S, p, rng)
    Y = X
    if delta > 0:
        nrng = np.random.default_rng(10_000 + seed)
        E = nrng.standard_normal(X.shape) + 1j * nrng.standard_normal(X.shape)
        Y = X + E * (delta * np.linalg.norm(X) / np.linalg.norm(E))
    return X, Y, f, B, mask, w

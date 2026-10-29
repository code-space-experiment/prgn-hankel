"""Shared pieces of all methods.

Problem: data and objective
    f(Z) = (1/2p) || mask o (G* Z - D X) ||_F^2 + (1/2) || (I - GG*) Z ||_F^2
over multilinear-rank-(r,r,r) tensors Z; spectral_init: the sequential spectral
initialization of Li et al. (Algorithm 2), shared by all methods; run_metrics:
relative signal error; Tangent and TangentOps: tangent vectors (S', U1', U2', U3')
of the Tucker parametrization, the Jacobian J of (S,U1,U2,U3) -> S x1 U1 x2 U2 x3 U3,
its adjoint and A^*A.
"""
import numpy as np
from hankel_common import (lift_G, adj_G, proj_hankel, unfold, fold,
                           mode_mul, tucker, hosvd_r, rel_err)


class Problem:
    def __init__(self, X, mask, p, n1, n2, w, r):
        self.X, self.mask, self.p, self.r = X, mask, p, r
        self.n1, self.n2, self.w = n1, n2, w
        self.s, self.n = X.shape
        self.DX_obs = mask * (np.sqrt(w) * X)          # mask o D(X)

    def data_residual_sig(self, Z):
        return self.mask * (adj_G(Z, self.n, self.w)) - self.DX_obs

    def grad_Z(self, Z):
        M = self.data_residual_sig(Z)
        return lift_G(M, self.n1, self.n2, self.w) / self.p \
             + Z - proj_hankel(Z, self.n, self.w)

    def loss(self, Z):
        M = self.data_residual_sig(Z)
        pen = Z - proj_hankel(Z, self.n, self.w)
        return (np.vdot(M, M).real / (2 * self.p)
                + 0.5 * np.vdot(pen, pen).real)

    def signal(self, Z):
        return adj_G(Z, self.n, self.w) / np.sqrt(self.w)

def spectral_init(pb):
    """Sequential spectral initialization (Li et al., Algorithm 2)."""
    Y0 = pb.mask * pb.X
    Z0 = lift_G(np.sqrt(pb.w) * Y0, pb.n1, pb.n2, pb.w) / pb.p  # G(D Y)/p
    r = pb.r
    U1 = np.linalg.svd(unfold(Z0, 0), full_matrices=False)[0][:, :r]
    U2 = np.linalg.svd(unfold(Z0, 1), full_matrices=False)[0][:, :r]
    Zc = mode_mul(Z0, U1.conj().T, 0)
    U3 = np.linalg.svd(unfold(Zc, 2), full_matrices=False)[0][:, :r]
    S = mode_mul(mode_mul(Zc, U2.conj().T, 1), U3.conj().T, 2)
    return S, U1, U2, U3

def run_metrics(pb, Z, X_true):
    return rel_err(pb.signal(Z), X_true)


class Tangent:
    """Tuple (xS, x1, x2, x3) with real inner product."""
    def __init__(self, xS, x1, x2, x3):
        self.xS, self.x1, self.x2, self.x3 = xS, x1, x2, x3
    def __add__(self, o):
        return Tangent(self.xS + o.xS, self.x1 + o.x1,
                       self.x2 + o.x2, self.x3 + o.x3)
    def __sub__(self, o):
        return Tangent(self.xS - o.xS, self.x1 - o.x1,
                       self.x2 - o.x2, self.x3 - o.x3)
    def scale(self, a):
        return Tangent(a * self.xS, a * self.x1, a * self.x2, a * self.x3)
    def inner(self, o):
        return (np.vdot(self.xS, o.xS) + np.vdot(self.x1, o.x1)
                + np.vdot(self.x2, o.x2) + np.vdot(self.x3, o.x3)).real
    def norm(self):
        return np.sqrt(self.inner(self))

class TangentOps:
    """Tangent-space operators at a point F = (S, U1, U2, U3)."""
    def __init__(self, pb):
        self.pb = pb
        self.hvp_count = 0

    def J(self, F, T):
        S, U1, U2, U3 = F
        return (tucker(T.xS, U1, U2, U3) + tucker(S, T.x1, U2, U3)
                + tucker(S, U1, T.x2, U3) + tucker(S, U1, U2, T.x3))

    def Jt(self, F, W):
        S, U1, U2, U3 = F
        xS = mode_mul(mode_mul(mode_mul(W, U1.conj().T, 0),
                               U2.conj().T, 1), U3.conj().T, 2)
        x1 = unfold(mode_mul(mode_mul(W, U2.conj().T, 1), U3.conj().T, 2), 0) \
             @ unfold(S, 0).conj().T
        x2 = unfold(mode_mul(mode_mul(W, U1.conj().T, 0), U3.conj().T, 2), 1) \
             @ unfold(S, 1).conj().T
        x3 = unfold(mode_mul(mode_mul(W, U1.conj().T, 0), U2.conj().T, 1), 2) \
             @ unfold(S, 2).conj().T
        return Tangent(xS, x1, x2, x3)

    def AtA(self, Z):
        pb = self.pb
        return lift_G(pb.mask * adj_G(Z, pb.n, pb.w),
                      pb.n1, pb.n2, pb.w) / pb.p \
             + Z - proj_hankel(Z, pb.n, pb.w)

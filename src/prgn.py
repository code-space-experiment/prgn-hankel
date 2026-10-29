"""PRGN (Algorithm 1 of the paper).

PRGN: reference form.  Normal-gauge coordinates (U_j^* Udot_j = 0), zero-start
PCG with the block preconditioner D^{-1}, stop when
||g - H xi||_{D^{-1}} <= ||g||_{D^{-1}}/16, unit step, retraction
(S - Sdot) x_j polar(U_j - Udot_j); no line search.
PRGNFast: the same mathematics with faster kernels.
"""
import numpy as np
from hankel_common import unfold, tucker, rel_err, lift_G, adj_G
import methods
from methods import Tangent, TangentOps


class PRGN(TangentOps):
    def __init__(self, pb, tol=1/16, maxcg=200):
        super().__init__(pb); self.tol, self.maxcg = tol, maxcg
        self.cg_steps = []
    def PN(self, F, T):
        _, U1, U2, U3 = F
        return Tangent(T.xS, *(x - U @ (U.conj().T @ x) for U, x in zip((U1, U2, U3), (T.x1, T.x2, T.x3))))
    def Dinv(self, F, T):
        S_ = F[0]; xs = []
        for j, x in enumerate((T.x1, T.x2, T.x3)):
            M = unfold(S_, j); C = M @ M.conj().T
            xs.append(np.linalg.solve(C.T, x.T).T)
        return Tangent(T.xS, *xs)
    def H(self, F, v):
        self.hvp_count += 1
        return self.PN(F, self.Jt(F, self.AtA(self.J(F, v))))
    def solve(self, F, g):
        x = g.scale(0.0); r = g; z = self.Dinv(F, r); p = z; rz = r.inner(z); g0 = np.sqrt(rz)
        k = 0
        while np.sqrt(rz) > self.tol * g0 and k < self.maxcg:
            Hp = self.H(F, p); a = rz / p.inner(Hp)
            x = x + p.scale(a); r = r - Hp.scale(a)
            z = self.Dinv(F, r); rzn = r.inner(z); p = z + p.scale(rzn / rz); rz = rzn; k += 1
        self.cg_steps.append(k)
        return x
    def run(self, X_true, maxit=300, tol=1e-9):
        pb = self.pb
        F = methods.spectral_init(pb); errs = []
        for it in range(maxit):
            g = self.PN(F, self.Jt(F, pb.grad_Z(tucker(*F))))
            xi = self.solve(F, g)
            S_, U1, U2, U3 = F
            def pol(U, x):
                Uq, _, Vt = np.linalg.svd(U - x, full_matrices=False); return Uq @ Vt
            Fn = (S_ - xi.xS, pol(U1, xi.x1), pol(U2, xi.x2), pol(U3, xi.x3))
            ratio = max(rel_err(a, b) for a, b in zip(Fn, F))
            F = Fn
            errs.append(methods.run_metrics(pb, tucker(*F), X_true))
            if ratio <= tol or not np.isfinite(errs[-1]) or errs[-1] > 1e3:
                break
        return np.array(errs), pb.signal(tucker(*F))


class PRGNFast(PRGN):
    """Same iteration with implementation-level savings: one Hankel adjoint per
    Hessian-vector product, A^*A Z = Z + G(mask/p - 1)G^* Z, and the Gram
    factorizations of the preconditioner cached per outer iterate."""
    def AtA(self, Z):
        pb = self.pb
        a = adj_G(Z, pb.n, pb.w)
        return Z + lift_G((pb.mask / pb.p - 1.0) * a, pb.n1, pb.n2, pb.w)
    def Dinv(self, F, T):
        if getattr(self, "_dkey", None) is not F:
            S_ = F[0]; self._dkey = F; self._Cinv = []
            for j in range(3):
                M = unfold(S_, j); self._Cinv.append(np.linalg.inv(M @ M.conj().T))
        return Tangent(T.xS, *(x @ Ci for x, Ci in zip((T.x1, T.x2, T.x3), self._Cinv)))

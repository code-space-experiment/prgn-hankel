"""GPU (PyTorch) implementation of the specified block-preconditioned
Gauss--Newton algorithm: normal-gauge coordinates, zero-start PCG with the
Gram-block preconditioner D^{-1}, stop at ||g - H xi||_{D^{-1}} <=
||g||_{D^{-1}}/16, unit step, polar retraction.  Same mathematics as the
NumPy version (prgn.PRGN / prgn.PRGNFast)."""
import math, time
import numpy as np
import torch


def weights(n, n1, n2):
    a = np.arange(n)
    return np.minimum.reduce([a + 1, n - a, np.full(n, n1), np.full(n, n2)])


def make_problem(n, n1, n2, s, r, h, p, seed):
    """Identical random draws to hankel_common.make_signal / make_mask."""
    rng = np.random.default_rng(seed)
    f0 = rng.random()
    f = (f0 + np.arange(r) * (h / n)) % 1.0
    A = np.exp(2j * np.pi * np.outer(np.arange(n), f))
    B = rng.standard_normal((s, r)) + 1j * rng.standard_normal((s, r))
    B /= np.linalg.norm(B, axis=0, keepdims=True)
    X = B @ A.T
    m = int(round(p * s * n))
    pick = rng.choice(s * n, size=m, replace=False)
    mask = np.zeros(s * n, bool); mask[pick] = True
    return X, mask.reshape(s, n), f


class PRGNTorch:
    def __init__(self, X, mask, p, n1, n2, r, device="cuda", dtype=torch.complex128, tol=1 / 16, maxcg=200):
        self.dev, self.dt, self.tol, self.maxcg = device, dtype, tol, maxcg
        s, n = X.shape
        self.s, self.n, self.n1, self.n2, self.r, self.p = s, n, n1, n2, r, p
        rdt = torch.float64 if dtype == torch.complex128 else torch.float32
        w = weights(n, n1, n2)
        self.sqw = torch.tensor(np.sqrt(w), dtype=rdt, device=device)
        idx = (np.arange(n1)[:, None] + np.arange(n2)[None, :])
        self.idx = torch.tensor(idx, device=device)
        self.idxf = self.idx.reshape(-1)
        self.mask = torch.tensor(mask, device=device).to(rdt)
        self.X = torch.tensor(X, dtype=dtype, device=device)
        self.DX = self.mask * (self.sqw * self.X)
        self.cg_steps = []

    # ---- Hankel lift and adjoint ----
    def lift(self, M):                      # (s,n) -> (n1,n2,s)
        return (M / self.sqw)[:, self.idx].permute(1, 2, 0)

    def adj(self, Z):                       # (n1,n2,s) -> (s,n)
        Zp = Z.permute(2, 0, 1).reshape(self.s, -1)
        out = torch.zeros(self.s, self.n, dtype=self.dt, device=self.dev)
        out.index_add_(1, self.idxf, Zp)
        return out / self.sqw

    # ---- tensor algebra ----
    @staticmethod
    def unfold(T, j):
        return T.movedim(j, 0).reshape(T.shape[j], -1)

    @staticmethod
    def mm(T, A, j):                        # T x_j A
        return torch.tensordot(A, T, dims=([1], [j])).movedim(0, j)

    def tucker(self, S, U1, U2, U3):
        return self.mm(self.mm(self.mm(S, U1, 0), U2, 1), U3, 2)

    def J(self, F, x):
        S, U1, U2, U3 = F
        xS, x1, x2, x3 = x
        return (self.tucker(xS, U1, U2, U3) + self.tucker(S, x1, U2, U3)
                + self.tucker(S, U1, x2, U3) + self.tucker(S, U1, U2, x3))

    def Jt(self, F, W):
        S, U1, U2, U3 = F
        H = lambda U: U.conj().T
        W1 = self.mm(W, H(U1), 0); W2 = self.mm(W, H(U2), 1)
        xS = self.mm(self.mm(W1, H(U2), 1), H(U3), 2)
        x1 = self.unfold(self.mm(W2, H(U3), 2), 0) @ self.unfold(S, 0).conj().T
        x2 = self.unfold(self.mm(W1, H(U3), 2), 1) @ self.unfold(S, 1).conj().T
        x3 = self.unfold(self.mm(W1, H(U2), 1), 2) @ self.unfold(S, 2).conj().T
        return [xS, x1, x2, x3]

    def AtA(self, Z):
        a = self.adj(Z)
        return Z + self.lift((self.mask / self.p - 1.0) * a)

    def grad(self, Z):
        a = self.adj(Z)
        return Z + self.lift((self.mask * a - self.DX) / self.p - a)

    # ---- normal-gauge tangent operations ----
    def PN(self, F, x):
        U = F[1:]
        return [x[0]] + [xj - Uj @ (Uj.conj().T @ xj) for xj, Uj in zip(x[1:], U)]

    def setup_D(self, F):
        S = F[0]
        self.Cinv = []
        for j in range(3):
            M = self.unfold(S, j)
            self.Cinv.append(torch.linalg.inv(M @ M.conj().T))

    def Dinv(self, x):
        return [x[0]] + [xj @ Ci for xj, Ci in zip(x[1:], self.Cinv)]

    @staticmethod
    def inner(a, b):
        return sum(torch.vdot(u.reshape(-1), v.reshape(-1)).real for u, v in zip(a, b))

    def H(self, F, v):
        return self.PN(F, self.Jt(F, self.AtA(self.J(F, v))))

    def solve(self, F, g):
        x = [torch.zeros_like(t) for t in g]; r = g
        z = self.Dinv(r); pv = z; rz = self.inner(r, z); g0 = torch.sqrt(rz)
        k = 0
        while torch.sqrt(rz) > self.tol * g0 and k < self.maxcg:
            Hp = self.H(F, pv); a = rz / self.inner(pv, Hp)
            x = [xi + a * pi for xi, pi in zip(x, pv)]
            r = [ri - a * hi for ri, hi in zip(r, Hp)]
            z = self.Dinv(r); rzn = self.inner(r, z)
            pv = [zi + (rzn / rz) * pi for zi, pi in zip(z, pv)]
            rz = rzn; k += 1
        self.cg_steps.append(k)
        return x

    def spectral_init(self):
        Z0 = self.lift(self.mask * self.sqw * self.X) / self.p      # noiseless observations
        def top(M):
            e, V = torch.linalg.eigh(M @ M.conj().T)
            return V[:, -self.r:]
        U1 = top(self.unfold(Z0, 0)); U2 = top(self.unfold(Z0, 1))
        Zc = self.mm(Z0, U1.conj().T, 0)
        U3 = top(self.unfold(Zc, 2))
        S = self.mm(self.mm(Zc, U2.conj().T, 1), U3.conj().T, 2)
        return (S, U1, U2, U3)

    def rel_err(self, F):
        Xh = self.adj(self.tucker(*F)) / self.sqw
        return float(torch.linalg.norm(Xh - self.X) / torch.linalg.norm(self.X))

    def run(self, target=1e-6, maxit=300, tol_ratio=1e-9):
        sync = (lambda: torch.cuda.synchronize()) if self.dev != "cpu" else (lambda: None)
        sync(); t0 = time.perf_counter()
        F = self.spectral_init(); sync(); t_init = time.perf_counter() - t0
        t_solver, errs, t_hit = 0.0, [], None
        for it in range(maxit):
            sync(); t0 = time.perf_counter()
            self.setup_D(F)
            g = self.PN(F, self.Jt(F, self.grad(self.tucker(*F))))
            xi = self.solve(F, g)
            S, U1, U2, U3 = F
            def pol(U, x):
                Q, _, Vh = torch.linalg.svd(U - x, full_matrices=False); return Q @ Vh
            Fn = (S - xi[0], pol(U1, xi[1]), pol(U2, xi[2]), pol(U3, xi[3]))
            ratio = max(float(torch.linalg.norm(a - b) / torch.linalg.norm(b)) for a, b in zip(Fn, F))
            F = Fn
            sync(); t_solver += time.perf_counter() - t0
            e = self.rel_err(F); errs.append(e)          # not timed
            if t_hit is None and e <= target:
                t_hit = t_solver
            if ratio <= tol_ratio or not math.isfinite(e) or e > 1e3:
                break
        return dict(errs=errs, t_init=t_init, t_solver=t_solver, t_hit=t_hit,
                    cg_steps=self.cg_steps, hvp=int(sum(self.cg_steps)))


def run_mixed(X, mask, p, n1, n2, r, device="cuda", switch=1e-4, target=1e-6, max_lo=50):
    """Complex64 iterations until rel. error <= switch (or stagnation), then
    complex128 refinement from the same factors."""
    lo = PRGNTorch(X, mask, p, n1, n2, r, device=device, dtype=torch.complex64)
    sync = torch.cuda.synchronize
    sync(); t0 = time.perf_counter(); F = lo.spectral_init(); sync(); t_init = time.perf_counter() - t0
    t_solver, errs, t_hit = 0.0, [], None
    for it in range(max_lo):
        sync(); t0 = time.perf_counter()
        lo.setup_D(F)
        g = lo.PN(F, lo.Jt(F, lo.grad(lo.tucker(*F))))
        xi = lo.solve(F, g)
        S, U1, U2, U3 = F
        def pol(U, x):
            Q, _, Vh = torch.linalg.svd(U - x, full_matrices=False); return Q @ Vh
        F = (S - xi[0], pol(U1, xi[1]), pol(U2, xi[2]), pol(U3, xi[3]))
        sync(); t_solver += time.perf_counter() - t0
        e = lo.rel_err(F); errs.append(e)
        if e <= switch or (len(errs) > 2 and e > 0.5 * errs[-2]):
            break
    hi = PRGNTorch(X, mask, p, n1, n2, r, device=device, dtype=torch.complex128)
    F = tuple(t.to(torch.complex128) for t in F)
    for it in range(60):
        sync(); t0 = time.perf_counter()
        hi.setup_D(F)
        g = hi.PN(F, hi.Jt(F, hi.grad(hi.tucker(*F))))
        xi = hi.solve(F, g)
        S, U1, U2, U3 = F
        def pol(U, x):
            Q, _, Vh = torch.linalg.svd(U - x, full_matrices=False); return Q @ Vh
        Fn = (S - xi[0], pol(U1, xi[1]), pol(U2, xi[2]), pol(U3, xi[3]))
        ratio = max(float(torch.linalg.norm(a - b) / torch.linalg.norm(b)) for a, b in zip(Fn, F))
        F = Fn
        sync(); t_solver += time.perf_counter() - t0
        e = hi.rel_err(F); errs.append(e)
        if t_hit is None and e <= target:
            t_hit = t_solver
            break
        if ratio <= 1e-9:
            break
    return dict(errs=errs, t_init=t_init, t_solver=t_solver, t_hit=t_hit,
                hvp=int(sum(lo.cg_steps) + sum(hi.cg_steps)), n_lo=len(lo.cg_steps), n_hi=len(hi.cg_steps))

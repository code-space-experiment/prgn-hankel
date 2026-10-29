"""Runs PRGN on one synthetic instance.

  python example.py [h] [p] [seed] [delta] [regime]
h: normalized spacing n*Delta (default 1.5); p: sampling ratio (0.2); seed (0);
delta: relative noise level (0); regime: coefficients gauss | well | ill | rank1 (gauss).
(n, n1, n2, s, r) = (127, 64, 64, 32, 4).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from methods import Problem
from hankel_common import esprit_freqs, freq_errors
from data import N, N1, N2, R, instance
from prgn import PRGNFast

a = sys.argv[1:] + [None] * 5
h = float(a[0] or 1.5); p = float(a[1] or 0.2); seed = int(a[2] or 0)
delta = float(a[3] or 0.0); regime = a[4] or "gauss"

X, Y, f, B, mask, w = instance(h, p, seed, delta, regime)
pb = Problem(Y, mask, p, N1, N2, w, R)
errs, Xh = PRGNFast(pb).run(X)
_, err_f = freq_errors(esprit_freqs(Xh, N1, N2, R, w, N), f)
for k, e in enumerate(errs, 1):
    print(f"iteration {k:3d}   relative error {e:.3e}")
print(f"true frequencies      {np.sort(f)}")
print(f"frequency error (max) {err_f:.3e}")

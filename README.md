# PRGN: preconditioned Riemannian Gauss–Newton for off-the-grid Hankel tensor recovery

Implementation of PRGN (Algorithm 1) of

> *Harmonic Geometry and Preconditioned Riemannian Optimization for Off-the-Grid Hankel Tensor Recovery*,
> submitted to Applied and Computational Harmonic Analysis.

## Requirements

Python 3.8 or later with NumPy. PyTorch with CUDA is needed only for the GPU version.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python example.py                 # h = 1.5, p = 0.2, seed 0, noiseless
python example.py 1.0 0.1 3       # spacing h = n*Delta, sampling ratio p, seed
python example.py 1.0 0.2 0 0.01  # relative noise level 0.01
```

The script draws a synthetic instance, runs PRGN from the spectral initialization, prints the relative error per iteration and the frequency error of ESPRIT on the recovered signal.

## Files

| File | Content |
|---|---|
| `src/prgn.py` | PRGN: reference implementation `PRGN` and the same iteration with faster kernels `PRGNFast` |
| `src/prgn_torch.py` | PRGN on the GPU (PyTorch), in FP64 (`PRGNTorch`) and in mixed precision (`run_mixed`) |
| `src/methods.py` | Objective, spectral initialization, tangent vectors and the Jacobian of the Tucker parametrization |
| `src/hankel_common.py` | Hankel lifting and its adjoint, Tucker operations, ESPRIT and frequency error |
| `src/data.py` | Synthetic signals, observation sets and noise |

## License

To be added.

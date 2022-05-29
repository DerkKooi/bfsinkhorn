## Bosonic and Fermionic Sinkhorn
A python package to calculate (non-interacting) orbital energies for 1-body Reduced Density Matrix Functional Theory in the canonical ensemble [1]

## Requirements:
- Python 3.7 or newer
- jax 0.3.0 or preferably newer (and corresponding jaxlib)
- Optional: numpy (for running the notebooks in figures)
- Optional: pyscf (for running the notebooks in figures pertaining to electronic systems)

## Setup:
- Clone repository
- cd bfsinkhorn
- pip install .

## Usage
See the figures folder of the repository for examples of how to use the Bosonic and Fermionic Sinkhorn algorithms.

Submodules:
- utils contains some simple log and exp related functions plus their parallelization using vmap.
- boson contains the Bosonic Sinkhorn algorithm and the required functions to compute (auxiliary) free energies and correlations
- fermion contains the Fermionic Sinkhorn algorithm and the required functions to compute (auxiliary) partition function ratios and correlations

The boson and fermion submodules contain functions to compute (auxiliary) free energies, (auxiliary) partition function ratios and correlations that may be useful for other purposes. Their usage is illustrated in the sinkhorn functions in the same submodule.

## Removal
- pip uninstall bfsinkhorn

## References
1. D.P. Kooi. Efficient Bosonic and Fermionic Sinkhorn Algorithms for Non-Interacting Ensembles in One-body Reduced Density Matrix Functional Theory in the Canonical Ensemble. arXiv:2205.15058 (2022). URL: https://arxiv.org/abs/2205.15058

## License
MIT License

Copyright (c) 2022 Derk P. Kooi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
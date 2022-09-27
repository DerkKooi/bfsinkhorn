try:
    import bfsinkhorn
except ImportError:
    import sys

    sys.path.append("..")
    import bfsinkhorn

import bfsinkhorn.boson

import jax.numpy as jnp


def test_import():
    assert bfsinkhorn


def test_bosonic_sinkhorn():

    # Fake orbital energies -> fake occupations
    eps = jnp.array([0.0, 0.2, 1.0, 5.0])
    N = 3
    n = bfsinkhorn.boson.compute_occupations(eps, N)
    assert n.shape == eps.shape

    eps = eps - jnp.sum(eps * n) / N

    # Check if inversion works
    result = bfsinkhorn.boson.sinkhorn(n, N, threshold=1e-12, max_iters=1000)
    assert jnp.allclose(result["eps"], eps)

    return


if __name__ == "__main__":
    test_import()
    test_bosonic_sinkhorn()

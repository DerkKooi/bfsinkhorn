# Import bfsinkhorn
import bfsinkhorn

# Import bosonic package
import bfsinkhorn.boson

# Import jax config and numpy and set floats to 64-bit
from jax.config import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)


def test_bosonic_sinkhorn():

    # Fake orbital energies -> fake occupations
    eps = jnp.array([0.0, 0.2, 0.2, 1.0, 5.0, 10.0, 20.0])
    N = 3
    n = bfsinkhorn.boson.compute_occupations(eps, N)
    assert n.shape == eps.shape

    # Shift orbital energies to match sinkhorn default
    eps = eps - jnp.sum(eps * n) / N

    # Check if inversion works
    result = bfsinkhorn.boson.sinkhorn(n, N, threshold=1e-12, max_iters=1000)
    assert jnp.allclose(result["eps"], eps)

    return


def test_bosonic_sinkhorn_class():

    # Fake orbital energies -> fake occupations
    eps = jnp.array([0.0, 0.2, 0.2, 1.0, 5.0, 10.0, 20.0])
    N = 3
    n = bfsinkhorn.boson.compute_occupations(eps, N)
    assert n.shape == eps.shape

    # Shift orbital energies to match sinkhorn default
    eps = eps - jnp.sum(eps * n) / N

    # Check if inversion works
    sinkhorn = bfsinkhorn.boson.Sinkhorn(N, tol=1e-12, maxiter=1000, anderson=True)
    params, _ = sinkhorn.run(n)
    assert jnp.allclose(params, eps)

    return


if __name__ == "__main__":
    test_bosonic_sinkhorn()
    test_bosonic_sinkhorn_class()

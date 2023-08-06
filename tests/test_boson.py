# Import bfsinkhorn
import jax.numpy as jnp

# Import jax config and numpy and set floats to 64-bit
from jax.config import config

import bfsinkhorn

# Import bosonic package
import bfsinkhorn.boson

config.update("jax_enable_x64", True)


def test_bosonic_sinkhorn():
    # Fake orbital energies -> fake occupations
    eps = jnp.array([0.0, 0.2, 0.2, 1.0, 5.0, 10.0, 20.0])
    N = 3
    n = bfsinkhorn.boson.compute_occupations(N, eps, 1.0)
    assert n.shape == eps.shape

    # Shift orbital energies to match sinkhorn default
    eps = eps - jnp.sum(eps * n) / N

    # Check if inversion works
    solver = bfsinkhorn.boson.Sinkhorn(N)
    result = solver(n)
    assert jnp.allclose(result["eps"], eps)


if __name__ == "__main__":
    test_bosonic_sinkhorn()

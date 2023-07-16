# Import bfsinkhorn
import jax.numpy as jnp

# Import jax config and numpy and set floats to 64-bit
from jax.config import config

import bfsinkhorn

# Import fermionic package
import bfsinkhorn.fermion

config.update("jax_enable_x64", True)


def test_fermionic_sinkhorn():
    # Fake orbital energies -> fake occupations
    eps = jnp.array([0.0, 0.2, 0.2, 1.0, 5.0, 10.0, 20.0])
    N = 3
    n = bfsinkhorn.fermion.compute_occupations(eps, N, 1.0)
    assert n.shape == eps.shape

    # Shift orbital energies to match sinkhorn default
    eps = eps - jnp.sum(eps * n) / N

    # Check if inversion works
    result = bfsinkhorn.fermion.sinkhorn(n, N, threshold=1e-12, max_iters=1000)
    assert jnp.allclose(result["eps"], eps)

    return


if __name__ == "__main__":
    test_fermionic_sinkhorn()

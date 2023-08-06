# Import bfsinkhorn

# Import jax config and numpy and set floats to 64-bit
from jax.config import config

import bfsinkhorn

# Import bosonic package
import bfsinkhorn.boson

# Import fermionic package
import bfsinkhorn.fermion

config.update("jax_enable_x64", True)


def test_import():
    assert bfsinkhorn
    assert bfsinkhorn.boson
    assert bfsinkhorn.fermion


if __name__ == "__main__":
    test_import()

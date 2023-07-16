import jax.numpy as jnp
from jax import jit, vmap


@jit
def log1mexp(a: float) -> float:
    r"""Computes log(1 - exp(-a)) for a > 0

    This should be stable whether or not a is above or below log(2)

    Parameters
    ----------
    a : float
        The argument to the exponential

    Returns
    -------
    log1mexp : float
        The result of log(1 - exp(-a))
    """
    return jnp.heaviside(a - jnp.log(2), 0.5) * jnp.log1p(-jnp.exp(-a)) + (
        1 - jnp.heaviside(a - jnp.log(2), 0.5)
    ) * jnp.log(-jnp.expm1(-a))


@jit
def minlogsumminexp(exponents: jnp.ndarray) -> float:
    r"""Calculate minus the log of the sum of exponents

    This is done in a stable way by shifting the exponents (log-sum-exp trick)

    Parameters
    ----------
    exponents : 1-dimensional ndarray
        The exponents to sum

    Returns
    -------
    minlogsumminexp : float
        The result of log(sum(exp(exponents)))
    """
    min = jnp.min(exponents)
    return -jnp.log(jnp.sum(jnp.exp(-exponents + min))) + min


@jit
def minlogsumminexp_array(exponents: jnp.ndarray) -> jnp.ndarray:
    r"""Calculate minus the log of the multiple sums of exponents for an array

    This is done in a stable way by shifting the exponents (log-sum-exp trick)

    Parameters
    ----------
    exponents : 2-dimensional ndarray
        The exponents to sum (on axis one)

    Returns
    -------
    minlogsumminexp : 1-dimensional ndarray
        The result of log(sum(exp(exponents)))
    """
    min = jnp.min(exponents, axis=1)
    return -jnp.log(jnp.sum(jnp.exp(-exponents + min), axis=1)) + min


# Parallelize minlogsumminexp with vmap,
# where the exponents are all multiplied by a factor k
minlogsumminexp_vmap = jit(
    vmap(
        lambda k, exponents: minlogsumminexp(k * exponents),
        in_axes=(0, None),
        out_axes=0,
    )
)

# vmap the sum of exponents for different k multiplying the exponents
summinexp_vmap = jit(
    vmap(
        lambda k, exponents: jnp.sum(jnp.exp(-k * exponents)),
        in_axes=(0, None),
        out_axes=0,
    )
)

# vmap the power function, unused
power_vmap = jit(vmap(jnp.power, (None, 0), 0))

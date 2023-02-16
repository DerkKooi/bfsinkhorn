from jax import jit, vmap, lax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import xlogy
from .utils import summinexp_vmap


@partial(jit, static_argnums=(1))
def compute_partition_function(eps, N, beta):
    r"""Compute the fermionic partition function ratios

    Parameters
    ----------
    eps : 1-dimensional ndarray
      The orbital energies
    N : int, static
      The number of particles
    beta : float
      Inverse temperature

    Returns
    --------
    Q : 1-dimensional ndarray of length N
      The partition function ratios Q_M for M=1 to M=N
    """
    # Compute quantity C_k for k=1 to k=N
    C = jnp.empty(N + 1)
    # C[0] = 1. is just a filler
    C = C.at[0].set(1.0)
    k = jnp.arange(1, N + 1)
    C = C.at[1:].set(summinexp_vmap(k, beta * eps))

    # Calculate E_k = C_k/(C_{k-1}), k=1 is still filler, it's just C_1
    E = C[1:] / C[:-1]

    # Calculate Q_M for M=1 to M=N recursively, Q_1 is simply C_1
    Q = jnp.empty(N)
    Q = Q.at[0].set(C[1])

    def outer_loop(M, Q):
        def inner_recursion(k, previous):
            return 1 - E[M - k] / Q[k - 1] * previous

        Q = Q.at[M - 1].set(C[1] * lax.fori_loop(1, M, inner_recursion, 1.0) / M)
        return Q

    return lax.fori_loop(2, N + 1, outer_loop, Q)


@jit
def compute_aux_partition_function(eps, Q, beta):
    r"""Compute the auxiliary fermionic partition function ratios

    This computes $Q_M^p = Z_M^p/Z_N$ for M=N-1 and M=N
    Note that this corresponds to REMOVING an orbital

    Parameters
    ----------
    eps : float
      The orbital energy of the orbital at hand
    Q : 1-dimensional ndarray of length N
      The partition function ratios for M=1 to M=N
    beta : float
      Inverse temperature

    Returns
    --------
    Q : 1-dimensional ndarray of length N
      The partition function ratios for M=0 to M=N
    """

    N = Q.shape[0]

    # Compute exponential of the orbital energy
    expeps = jnp.exp(-beta * eps)

    # Calculate Qp for M=N-1 and M=N
    Qp = jnp.ones(2)

    def inner_recursion(k, previous):
        return 1 - expeps / Q[k] * previous

    Qp = Qp.at[0].set(lax.fori_loop(0, N - 1, inner_recursion, Qp[0]) / Q[N - 1])
    Qp = Qp.at[1].set(lax.fori_loop(0, N, inner_recursion, Qp[1]))
    return Qp


# vmap of compute_aux_partition_function over orbital energies
compute_aux_partition_function_vmap = jit(
    vmap(compute_aux_partition_function, in_axes=(0, None, None), out_axes=(1))
)


@jit
def compute_correlations(n, eps, beta):
    r"""Compute the fermionic correlations <\hat{n}_p \hat{n}_q>

    We use the expression for non-degenerate levels,
    so we have to correct afterwards for degenerate levels

    Parameters
    ----------
    n : 1-dimensional ndarray
      The occupation numbers
    eps : 1-dimensional ndarray
      The orbital energies
    beta : float
      Inverse temperature

    Returns
    --------
    correlations: ndarray with dimension (size(n), size(n))
      The fermionic correlations <\hat{n}_p \hat{n}_q>
    """
    # Compute off-diagonal terms
    correlations = (
        jnp.outer(n * jnp.exp(beta * eps), 1) - jnp.outer(1, n * jnp.exp(beta * eps))
    ) / (jnp.outer(jnp.exp(beta * eps), 1) - jnp.outer(1, jnp.exp(beta * eps)))

    # Set diagonal terms
    i, j = jnp.diag_indices(eps.shape[0])
    correlations = correlations.at[i, j].set(n)
    return correlations


@jit
def compute_correlations_degen(eps, Q, beta):
    r"""Compute the fermionic correlations <\hat{n}_p \hat{n}_q> for degenerate levels

    Parameters
    ----------
    eps : float
      The orbital energy of the degenerate levels
    Q : 1-dimensional ndarray
      The partition function ratios for M=1 to M=N
    beta : float
      Inverse temperature

    Returns
    --------
    correl_degen: float
      Correlations for the degenerate level"""

    N = Q.shape[0]

    # Compute exponentials of the orbital energy
    expeps = jnp.exp(-beta * eps)
    expeps2 = jnp.exp(-beta * eps * 2)

    # Compute the correlations by recursion
    def inner_recursion(kp, previous):
        k = N - kp
        return 1 - (k - 1) / (k - 2) * expeps / Q[N - k] * previous

    correlations_degen = (
        expeps2 * lax.fori_loop(0, N - 2, inner_recursion, 1.0) / (Q[N - 1] * Q[N - 2])
    )
    return correlations_degen


# vmap of compute_aux_partition_function over degenerate orbital energies, they are not exactly equal so weigh them equally
compute_correlations_degen_vmap = jit(
    vmap(
        lambda eps, Q, beta: compute_correlations_degen((eps[0] + eps[1]) / 2, Q, beta),
        in_axes=(1, None, None),
        out_axes=0,
    )
)


@partial(jit, static_argnums=(1))
def compute_occupations(eps, N, beta=1.0):
    r"""Compute the occupation numbers for a given set of orbital energies

    Parameters
    ----------
    eps : 1-dimensional ndarray
      The orbital energies
    N : int
      The number of electrons
    beta : float
      Inverse temperature

    Returns
    --------
    n : 1-dimensional ndarray
      The occupation numbers
    """

    # Compute partition function ratios
    Q = compute_partition_function(eps, N, beta)

    # Compute auxiliary partition function missing one orbital
    Qp = compute_aux_partition_function_vmap(eps, Q, beta)

    # Compute the occupations
    n = jnp.exp(-beta * eps) * Qp[0]

    return n


def compute_correlations_full(n, eps, Q, beta=1.0, degen_cutoff=1e-7):
    r"""
    Compute the correlations <\hat{n}_p \hat{n}_q> as well
    """
    norb = n.shape[0]
    correlations = compute_correlations(n, eps, beta)

    # Compute the degenerate level correlations if they are present
    degens = jnp.where(
        jnp.abs(
            jnp.ones((norb, norb)) * jnp.exp(beta * eps)
            - (jnp.ones((norb, norb)) * jnp.exp(beta * eps)).T
            + jnp.eye(norb)
        )
        < degen_cutoff
    )
    if degens[0].size > 0:
        correlations = correlations.at[degens[0], degens[1]].set(
            compute_correlations_degen_vmap(
                jnp.vstack((eps[degens[0]], eps[degens[1]])), Q, beta
            )
        )
    return correlations


def sinkhorn(
    n,
    N,
    beta=1.0,
    eps=None,
    max_iters: int = 100,
    threshold=10**-10,
    old=False,
    verbose=True,
):
    r"""(Fermionic) Sinkhorn algorithm

    Parameters
    ----------
    n : 1-dimensional ndarray
      The occupation numbers
    N : int
      The number of particles
    beta : float, optional
      The inverse temperature, default is 1.
    eps : 1-dimensional ndarray, optional
      The starting orbital energies, default is grand canonical guess
    max_iters : int, optional
      The maximum number of iterations, default is 100
    threshold : float, optional
      The convergence threshold, default is 10**-10
    old : bool, optional
      Whether to use the ``naive`` Sinkhorn, default is using Fermionic Sinkhorn
    comp_correlations : bool, optional
      Whether to
    degen_cutoff : float, optional
      The cutoff for degenerate levels, default is 10**-7
    verbose : bool, optional
      Whether to print the progress, default is True

    Returns
    --------
    result : dictionary
      The result of the algorithm with:
      - 'converged' : bool
        Whether the algorithm converged
      - 'eps' : 1-dimensional ndarray
        The orbital energies
      - 'Q' : 1-dimensional ndarray
        The partition function ratios
      - 'errors' : list of floats
        The errors of the algorithm at different iterations
      - 'S' : float
        The entropy of the system
      - 'eps_GC' : 1-dimensional ndarray
        The grand canonical orbital energies
      - 'S_GC' : float
        The entropy of the grand canonical system
      - 'n_approx' : 1-dimensional ndarray
        The approximate occupation numbers
      - 'correlations' : ndarray with dimension (size(n), size(n))
        The correlations <\hat{n}_p \hat{n}_q>
    """

    norb = n.shape[0]
    result = {}
    result["converged"] = False

    # Compute the orbital energies and entropy for the grand canonical ensemble
    eps_GC = -jnp.log(n / (1 - n)) / beta
    S_GC = -jnp.sum(xlogy(n, n) + xlogy(1 - n, 1 - n)) / beta

    # Shift orbital energies such that \sum_p n_p \epsilon_p = 0.
    if eps is None:
        # Grand canonical guess for the orbital energies
        eps = eps_GC - jnp.sum(n * eps_GC) / N
    else:
        eps = eps - jnp.sum(n * eps) / N

    errors = []
    for i in range(max_iters + 1):
        # Compute partition function ratios
        Q = compute_partition_function(eps, N, beta).block_until_ready()
        # Compute corresponding free energy differences
        dF = -jnp.log(Q) / beta
        # Compute free energy of the N-electron system.
        F = jnp.sum(dF)

        # Compute auxiliary partition function missing one orbital
        Qp = compute_aux_partition_function_vmap(eps, Q, beta).block_until_ready()

        # Compute the occupation numbers n_approx with current orbital energies
        n_approx = jnp.exp(-beta * eps) * Qp[0]
        errors.append(jnp.sum(jnp.abs(n_approx - n)))

        if verbose:
            print(f"iter {i}, error {errors[i]}")

        # If the error in the occupation numbers is below the threshold, stop
        if errors[i] < threshold:
            result["converged"] = True
            break

        # If we are not at the last iterate, update the orbital energies
        if i != max_iters:
            if not old:
                # Fermionic Sinkhorn iteration
                eps = -jnp.log(n / (1 - n) * Qp[1] / Qp[0]) / beta
            else:
                # Regular Sinkhorn iteration
                eps = -jnp.log(n / Qp[0]) / beta

            # Shift orbital energies such that \sum_p n_p \epsilon_p = 0
            eps = eps - jnp.sum(n * eps) / N

        if jnp.any(jnp.isnan(eps)) or jnp.any(jnp.isinf(eps)):
            print("Encountered inf or nan in orbital energies")
            print(eps)
            break

    # Compute entropy, note that this assumes that the orbital energies are shifted
    S = -beta * F

    # Pack the results into a dictionary
    result["eps"] = eps
    result["errors"] = errors
    result["Q"] = Q
    result["S"] = S
    result["eps_GC"] = eps_GC
    result["S_GC"] = S_GC
    result["n_approx"] = n_approx
    return result

from jax import jit, vmap, lax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import xlogy
from .utils import minlogsumminexp, minlogsumminexp_vmap


@partial(jit, static_argnums=(1))
def compute_free_energy(eps, N, beta):
    """Compute bosonic free energies

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
    F : 1-dimensional ndarray of length N+1
      The free energy F_M for M=0 to M=N
    """

    # First compute D_k = -1/\beta \log(\sum_p e^{-\beta k \eps_p})
    # in ascending order with inf pad on the left
    D = jnp.ones(2 * N - 1) * jnp.inf
    k = jnp.arange(1, N + 1)
    D = D.at[N - 1 :].set(minlogsumminexp_vmap(k, beta * eps) / beta)

    # Build array for free energies in descending order (!),
    # initialize at inf, except for M=0: free energy is zero
    F = jnp.ones(N + 1) * jnp.inf
    F = F.at[N].set(0)

    # Compute free energies iteratively for M=1 to M=N
    # We use some padding tricks such that
    # the slices are the same size for every iteration.
    def inner_loop(i, F):
        exponents = lax.dynamic_slice(D, [i - 1], [N]) + F[1 : N + 1]
        F = F.at[N - i].set((minlogsumminexp(beta * exponents) + jnp.log(i)) / beta)
        return F

    F = lax.fori_loop(1, N + 1, inner_loop, F)

    return jnp.flip(F)


@jit
def compute_aux_free_energy(eps, F, beta):
    """Compute bosonic auxiliary free energy,

    Note that this corresponds to ADDING an extra orbital
    with the same orbital energy
    This computes $F_M^p$ for M=N-1 and M=N,

    Parameters
    ----------
    eps : float
      The orbital energy of the orbital at hand
    F : 1-dimensional ndarray of length N+1
      The partition function ratios for M=1 to M=N
    beta : float
      Inverse temperature

    Returns
    --------
    Fp : 1-dimensional ndarray of length 2
      The auxiliary free energies for M=N and M=N-1
    """

    N = F.shape[0] - 1
    Fp = jnp.empty(2)

    # Compute auxiliary free energy of M=N-1 system
    k = jnp.arange(N)
    exponents = jnp.flip(F)[1:] + k * eps
    Fp = Fp.at[0].set(minlogsumminexp(beta * exponents) / beta)

    # Compute auxiliary free energy of M=N system
    k = jnp.arange(N + 1)
    exponents = jnp.flip(F) + k * eps
    Fp = Fp.at[1].set(minlogsumminexp(beta * exponents) / beta)

    return Fp


# vmap of compute_aux_free_energy over orbital energies
compute_aux_free_energy_vmap = jit(
    vmap(compute_aux_free_energy, in_axes=(0, None, None), out_axes=0)
)


@jit
def compute_correlations(n, eps, F, beta):
    """Compute the bosonic correlations <\hat{n}_p \hat{n}_q>

    We use the expression for non-degenerate levels,
    so we have to correct afterwards for degenerate levels

    Parameters
    ----------
    n : 1-dimensional ndarray
      The occupation numbers
    eps : 1-dimensional ndarray
      The orbital energies
    F : 1-dimensional ndarray of length N+1
      The free energy F_M for M=0 to M=N
    beta : float
      Inverse temperature

    Returns
    --------
    correlations: ndarray with dimension (size(n), size(n))
      The bosonic correlations <\hat{n}_p \hat{n}_q>
    """
    N = F.shape[0] - 1

    # Compute off-diagonal terms
    correlations = -(
        jnp.outer(n * jnp.exp(beta * eps), 1) - jnp.outer(1, n * jnp.exp(beta * eps))
    ) / (jnp.outer(jnp.exp(beta * eps), 1) - jnp.outer(1, jnp.exp(beta * eps)))

    # Compute diagonal terms
    k = jnp.arange(1, N + 1)
    correlations_diag = vmap(
        lambda eps: jnp.sum(
            (2 * k - 1) * jnp.exp(-beta * (jnp.flip(F[:N]) + eps * k - F[N]))
        ),
        in_axes=(0),
        out_axes=0,
    )(eps)

    # Set diagonal terms
    i, j = jnp.diag_indices(eps.shape[0])
    correlations = correlations.at[i, j].set(correlations_diag)
    return correlations


@jit
def compute_correlations_degen(eps, F, beta):
    """Compute the bosonic correlations <\hat{n}_p \hat{n}_q> for degenerate levels

    Parameters
    ----------
    eps : float
      The orbital energy of the degenerate levels
    F : 1-dimensional ndarray of length N+1
      The free energy F_M for M=0 to M=N
    beta : float
      Inverse temperature

    Returns
    --------
    correl_degen: float
      Correlations for the degenerate level"""

    N = F.shape[0] - 1

    k = jnp.arange(2, N + 1)
    correlations_degen = jnp.sum(
        (k - 1) * jnp.exp(-beta * (jnp.flip(F[: N - 1]) + eps * k - F[N]))
    )
    return correlations_degen


# vmap of compute_aux_partition_function over degenerate orbital energies, they are not exactly equal so weigh them equally
compute_correlations_degen_vmap = jit(
    vmap(
        lambda eps, F, beta: compute_correlations_degen((eps[0] + eps[1]) / 2, F, beta),
        in_axes=(1, None, None),
        out_axes=0,
    )
)


def sinkhorn(
    n,
    N,
    beta=1.0,
    eps=None,
    max_iters=100,
    threshold=10**-10,
    old=False,
    comp_correlations=False,
    degen_cutoff=10**-7,
    verbose=True,
):
    """(Bosonic) Sinkhorn algorithm

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
      Whether to compute the correlations <\hat{n}_p \hat{n}_q> as well, default is not
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
    eps_GC = -jnp.log(n / (1 + n)) / beta
    S_GC = -jnp.sum(xlogy(n, n) + xlogy(1 - n, 1 - n)) / beta

    # Shift orbital energies such that \sum_p n_p \epsilon_p = 0.
    if eps is None:
        # Grand canonical guess for the orbital energies
        eps = eps_GC - jnp.sum(n * eps_GC) / N
    else:
        eps = eps - jnp.sum(n * eps) / N

    # Perform Sinkhorn iterations and store error at every iteration
    errors = []
    for i in range(max_iters + 1):
        # Compute free energies
        F = compute_free_energy(eps, N, beta).block_until_ready()
        # Compute auxiliary free energies
        Fp = compute_aux_free_energy_vmap(eps, F, beta).block_until_ready()

        # Compute the occupation numbers n_approx with current orbital energies
        n_approx = jnp.exp(-beta * (eps + Fp[:, 0] - F[N]))
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
                # Bosonic Sinkhorn iteration
                eps = eps_GC + Fp[:, 1] - Fp[:, 0]
            else:
                # Regular Sinkhorn iteration
                eps = -jnp.log(n) / beta + F[-1] - Fp[:, 0]

            # Shift orbital energies such that \sum_p n_p \epsilon_p = 0
            eps = eps - jnp.sum(n * eps) / N

        if jnp.any(jnp.isnan(eps)) or jnp.any(jnp.isinf(eps)):
            print("Encountered inf or nan in orbital energies")
            print(eps)
            break

    # Compute entropy, note that this assumes that the orbital energies are shifted
    S = -beta * F[-1]

    if comp_correlations:
        correlations = compute_correlations(n, eps, F, beta).block_until_ready()

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
                    jnp.vstack((eps[degens[0]], eps[degens[1]])), F, beta
                ).block_until_ready()
            )

    # Pack the results in a dictionary
    result["eps"] = eps
    result["errors"] = errors
    result["F"] = F
    result["S"] = S
    result["eps_GC"] = eps_GC
    result["S_GC"] = S_GC
    result["n_approx"] = n_approx
    if comp_correlations:
        result["correlations"] = correlations
    return result

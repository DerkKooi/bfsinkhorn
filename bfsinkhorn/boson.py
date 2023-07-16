from functools import partial
from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import jacfwd, jacrev, jit, lax, vmap
from jax.scipy.special import xlogy
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt._src.base import OptStep

from .utils import minlogsumminexp, minlogsumminexp_array, minlogsumminexp_vmap


@partial(jit, static_argnums=(1,))
def compute_free_energy(eps: jnp.ndarray, N: int, beta: float) -> jnp.ndarray:
    r"""
    Compute bosonic free energies

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
def compute_aux_free_energy(eps: jnp.ndarray, F: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    r"""
    Compute bosonic auxiliary free energy,

    Note that this corresponds to ADDING an extra orbital
    with the same orbital energy
    This computes $F_M^p$ for M=N-1 and M=N,

    Parameters
    ----------
    eps : float
      The orbital energy of the orbital at hand
    F : 1-dimensional ndarray of length N+1
      The free energies for M=0 to M=N
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
def compute_aux_free_energy_all(eps: float, F: jnp.ndarray, beta: float):
    r"""
    Compute all bosonic auxiliary free energies,

    Note that this corresponds to ADDING an extra orbital
    with the same orbital energy
    This computes $F_M^p$ for M=0 to M=N

    Parameters
    ----------
    eps : float
      The orbital energy of the orbital at hand
    F : 1-dimensional ndarray of length N+1
      The free energies for M=0 to M=N
    beta : float
      Inverse temperature

    Returns
    --------
    Fp : 1-dimensional ndarray of length N+1
      The auxiliary free energies for M=0 to M=N
    """

    N = F.shape[0] - 1

    # Initialize exponents with infs for the off-diagonal terms
    exponents = jnp.ones((N + 1, N + 1)) * jnp.inf

    # Compute auxiliary free energy of M=N-1 system
    M, k = jnp.tril_indices_from(exponents)
    exponents = exponents.at[M, k].set(F[M - k] + k * eps)
    Fp = minlogsumminexp_array(beta * exponents) / beta

    return Fp


@jit
def compute_correlations(n: jnp.ndarray, eps: jnp.ndarray, F: jnp.ndarray, beta: float):
    r"""
    Compute the bosonic correlations <\hat{n}_p \hat{n}_q>

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
        lambda eps: jnp.sum((2 * k - 1) * jnp.exp(-beta * (jnp.flip(F[:N]) + eps * k - F[N]))),
        in_axes=(0),
        out_axes=0,
    )(eps)

    # Set diagonal terms
    i, j = jnp.diag_indices(eps.shape[0])
    correlations = correlations.at[i, j].set(correlations_diag)
    return correlations


@jit
def compute_correlations_degen(eps: float, F: jnp.ndarray, beta: float) -> float:
    r"""
    Compute the bosonic correlations <\hat{n}_p \hat{n}_q> for degenerate levels

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
    correlations_degen = jnp.sum((k - 1) * jnp.exp(-beta * (jnp.flip(F[: N - 1]) + eps * k - F[N])))
    return correlations_degen


# vmap of compute_aux_partition_function over degenerate orbital energies,
# they are not exactly equal so weigh them equally
compute_correlations_degen_vmap = jit(
    vmap(
        lambda eps, F, beta: compute_correlations_degen((eps[0] + eps[1]) / 2, F, beta),
        in_axes=(1, None, None),
        out_axes=0,
    )
)


def compute_correlations_full(
    n: jnp.ndarray, eps: jnp.ndarray, F: jnp.ndarray, beta: float, degen_cutoff: float = 1e-7
) -> jnp.ndarray:
    r"""
    Compute the bosonic correlations <\hat{n}_p \hat{n}_q>

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
    degen_cutoff : float
      Cutoff for degeneracy, default is 1e-7

    Returns
    --------
    correlations: ndarray with dimension (size(n), size(n))
      The bosonic correlations <\hat{n}_p \hat{n}_q>
    """
    norb = eps.shape[0]
    correlations = compute_correlations(n, eps, F, beta)

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
    return correlations


@partial(jit, static_argnums=(1,))
def compute_occupations(eps: jnp.ndarray, N: int, beta: float) -> jnp.ndarray:
    r"""
    Compute the occupation numbers for a given set of orbital energies

    Parameters
    ----------
    eps : 1-dimensional ndarray
      The orbital energies
    N : int
      The number of particles
    beta : float
      Inverse temperature

    Returns
    --------
    n : 1-dimensional ndarray
      The occupation numbers
    """

    # Compute free energy
    F = compute_free_energy(eps, N, beta)

    # Compute auxiliary free energy
    Fp = compute_aux_free_energy_vmap(eps, F, beta)

    # Compute occupations
    n = jnp.exp(-beta * (eps + Fp[:, 0] - F[N]))

    return n


@partial(jit, static_argnums=(3, 4))
def fixed_point(
    eps: jnp.ndarray,
    n: jnp.ndarray,
    beta: float = 1.0,
    N: int = 2,
    old: bool = False,
) -> jnp.ndarray:
    r"""
    Compute the orbital energies given the (Bosonic) Sinkhorn algorithm

    Note that the orbital energies are normalized in the last step

    Parameters
    ----------
    eps : 1-dimensional ndarray
      The orbital energies
    n : 1-dimensional ndarray
      The occupation numbers
    N : int
      The number of particles, default is 2
    beta : float
      Inverse temperature
    old : bool
      Whether to use the old version of the Sinkhorn algorithm

    Returns
    --------
    eps : 1-dimensional ndarray
      The updated orbital energies
    """

    # Fix norm in n
    n = n / jnp.sum(n) * N

    # Compute free energy
    F = compute_free_energy(eps, N, beta)

    # Compute auxiliary free energy
    Fp = compute_aux_free_energy_vmap(eps, F, beta)

    # Compute new orbital energies
    if old:
        eps_new = -jnp.log(n) / beta + F[-1] - Fp[:, 0]
    else:
        eps_new = -jnp.log(n / (1 + n)) / beta + Fp[:, 1] - Fp[:, 0]

    return eps_new - jnp.sum(n * eps_new) / N


@jit
def eps_GC_guess(n: jnp.ndarray, beta: float) -> jnp.ndarray:
    """
    Compute the grand canonical guess for the orbital energies

    Parameters
    ----------
    n : 1-dimensional ndarray
      The occupation numbers

    Returns
    --------
    eps_GC : 1-dimensional ndarray
      The orbital energies
    """
    return -jnp.log(n / (1 + n)) / beta


@jit
def compute_S_GC(n: jnp.ndarray, beta: float) -> float:
    """
    Compute the grand canonical entropy

    Parameters
    ----------
    n : 1-dimensional ndarray
      The occupation numbers

    Returns
    --------
    S_GC : float
      The grand canonical entropy
    """
    return -jnp.sum(xlogy(n, n) + xlogy(1 + n, 1 + n)) / beta


@partial(jit, static_argnums=(2,))
def normalize_eps(n: jnp.ndarray, eps: jnp.ndarray, N: int) -> jnp.ndarray:
    """
    Normalize the orbital energies

    Parameters
    ----------
    n : 1-dimensional ndarray
      The occupation numbers
    eps : 1-dimensional ndarray
      The orbital energies
    N : int

    Returns
    --------
    new_eps : 1-dimensional ndarray
      The normalized orbital energies
    """

    return eps - jnp.sum(n * eps) / N


class Sinkhorn:
    """(Bosonic) Sinkhorn algorithm"""

    def __init__(
        self,
        N: int,
        old: bool = False,
        anderson: bool = False,
        implicit_diff: bool = True,
        maxiter: int = 100,
        tol: float = 1e-10,
        verbose: bool = False,
        history_size: int = 5,
        mixing_frequency: int = 1,
        anderson_beta: float = 1,
        ridge: float = 1e-5,
    ):
        self.N = N
        self.method = "anderson" if anderson else "fixed_point"

        self._fixed_point = jit(partial(fixed_point, N=N, old=old))

        if anderson:
            # Anderson acceleration does not seem to offer any advantage unless old is true
            self._optimizer = AndersonAcceleration(
                self._fixed_point,
                history_size=history_size,
                mixing_frequency=mixing_frequency,
                beta=anderson_beta,
                maxiter=maxiter,
                tol=tol,
                ridge=ridge,
                verbose=verbose,
                implicit_diff=implicit_diff,
                jit=not verbose,
            )
        else:
            self._optimizer = FixedPointIteration(
                self._fixed_point,
                maxiter=maxiter,
                tol=tol,
                verbose=verbose,
                implicit_diff=implicit_diff,
                jit=not verbose,
            )

        def _run_GC_guess(n: jnp.ndarray, beta: float = 1.0) -> OptStep:
            """
            Run the Sinkhorn algorithm with the grand canonical guess

            Parameters
            ----------
            n : 1-dimensional ndarray
              The occupation numbers
            beta : float
              Inverse temperature

            Returns
            --------
            OptStep
              The result of the optimization
            """
            eps = eps_GC_guess(n, beta)
            eps = normalize_eps(n, eps, N)
            return self._optimizer.run(eps, n, beta)

        def _run_eps_only(n: jnp.ndarray, eps0: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
            """
            Run the Sinkhorn algorithm with eps0 as the guess

            Return only the orbital energies for differentiation.

            Parameters
            ----------
            n : 1-dimensional ndarray
              The occupation numbers
            eps0 : 1-dimensional ndarray
              The orbital energies
            beta : float
              Inverse temperature, default is 1

            Returns
            --------
            eps : 1-dimensional ndarray
              The orbital energies
            """
            return self._optimizer.run(eps0, n, beta).params

        self._run_GC_guess = jit(_run_GC_guess)

        # Parallelize over multiple systems
        self.run_parallel = jit(vmap(_run_GC_guess, in_axes=(0, None)))

        # Calculate the orbital energies
        self._run_eps_only = jit(_run_eps_only)

        # Calculate the derivatives of the orbital energies with respect to the occupation numbers
        # This is dodgy because only certain variations of n are allowed
        self.deps_dn = jit(jacrev(_run_eps_only, argnums=0))
        if implicit_diff is False:
            self.deps_dn_fwd = jit(jacfwd(_run_eps_only, argnums=0))

        # Calculate the occupation numbers and derivatives w.r.t the orbital energies
        self.compute_occupations = jit(lambda eps, beta=1.0: compute_occupations(eps, self.N, beta))

        # Note that there is a zero singular value for eps all shifting the same amount
        self.dn_deps = jit(jacfwd(self.compute_occupations, argnums=0))
        self.dn_deps_rev = jit(jacrev(self.compute_occupations, argnums=0))

        self.d2n_deps2 = jit(jacfwd(jacfwd(self.compute_occupations, argnums=0), argnums=0))
        self.d2n_deps2_rev = jit(jacfwd(jacrev(self.compute_occupations, argnums=0), argnums=0))

        # Calculate the free energy and entropy
        self.compute_free_energy = jit(lambda eps, beta=1.0: compute_free_energy(eps, self.N, beta))
        self.compute_entropy = jit(lambda eps, beta=1.0: -beta * self.compute_free_energy(eps))

    def run(
        self, n: jnp.ndarray, eps: Optional[jnp.ndarray] = None, beta: float = 1.0
    ) -> Dict[str, Union[jnp.ndarray, int, float]]:
        """
        Run the Sinkhorn algorithm

        Parameters
        ----------
        n : 1-dimensional ndarray
          The occupation numbers
        eps : 1-dimensional ndarray, optional
          The starting orbital energies, default is grand canonical guess
        beta : float
          Inverse temperature

        Returns
        --------
        results : dict
          The results of the optimization
            eps : 1-dimensional ndarray
              The resulting orbital energies
            n_approx : 1-dimensional ndarray
              The computed occupation numbers from the orbital energies
            eps_error : float
              The error in the orbital energies
            n_error : float
              The error in the occupation numbers
            iter_num : int
              The number of iterations
            F : np.ndarray
              The free energies for M=0 to M=N
            S : float
              The entropy
            eps_gc : 1-dimensional ndarray
              The orbital energies from the grand canonical guess
            S_GC : float
              The entropy from the grand canonical guess
        """

        if eps is None:
            new_eps, state = self._run_GC_guess(n, beta)
        else:
            new_eps, state = self._optimizer.run(eps, n, beta)
        n_approx = self.compute_occupations(new_eps, beta)
        n_error = jnp.sum(jnp.abs(n - n_approx))
        F = self.compute_free_energy(new_eps, beta)
        results = {
            "eps": new_eps,
            "n_approx": n_approx,
            "eps_error": state.error,
            "n_error": n_error,
            "iter_num": state.iter_num,
            "F": F,
            "S": -beta * F[-1],
            "eps_gc": eps_GC_guess(n, beta),
            "S_GC": compute_S_GC(n, beta),
        }
        if self.method == " anderson":
            results.update(
                {
                    "params_history": state.params_history,
                    "residual_gram": state.residual_gram,
                    "residuals_history": state.residuals_history,
                }
            )
        return results

    def compute_correlations(
        self, n: jnp.ndarray, eps: jnp.ndarray, beta: float = 1.0
    ) -> jnp.ndarray:
        r"""
        Compute the two-point correlations < n_i n_j >

        This is equal to :math:`n_i n_j - \frac{\partial n_i}{\partial \epsilon_j}`

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
        correlations : 1-dimensional ndarray
          The correlations
        """
        return jnp.outer(n, n) - self.dn_deps(eps, beta)

    def compute_correlations(
        self, n: jnp.ndarray, eps: jnp.ndarray, beta: float = 1.0
    ) -> jnp.ndarray:
        r"""
        Compute the two-point correlations < n_i n_j >

        This is equal to :math:`n_i n_j - \frac{\partial n_i}{\partial \epsilon_j}`

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
        correlations : 1-dimensional ndarray
          The correlations
        """
        return jnp.outer(n, n) - self.dn_deps(eps, beta)

    def compute_rdm2(self, n: jnp.ndarray, eps: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        r"""
        Compute the 2-RDM element < a^\dagger_p a^\dagger_q a_q a_p >

        This is equal to
        :math:`n_i n_j - \frac{\partial n_i}{\partial \epsilon_j} - \delta_{ij} n_i`

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
        rdm2 : 1-dimensional ndarray
            The diagonal elements of the 2-RDM
        """
        return jnp.outer(n, n) - self.dn_deps(eps, beta) - jnp.diag(n)

    def compute_rdm3(
        self,
        n: jnp.ndarray,
        eps: jnp.ndarray,
        rdm2: Optional[jnp.ndarray] = None,
        beta: float = 1.0,
    ) -> jnp.ndarray:
        r"""
        Compute the 3-RDM element :math`< a^\dagger_p a^\dagger_q a^\dagger_r a_r a_q a_p >`

        This is equal to a nice and long expression.

        Parameters
        ----------
        n : 1-dimensional ndarray
          The occupation numbers
        eps : 1-dimensional ndarray
          The orbital energies
        rdm2 : 2-dimensional ndarray, optional
            The 2-RDM, default is computed from eps, n and beta
        beta : float
          Inverse temperature

        Returns
        --------
        rdm3 : 1-dimensional ndarray
            The diagonal elements of the 3-RDM
        """
        norb = n.shape[-1]
        if rdm2 is None:
            rdm2 = self.compute_rdm2(n, eps, beta)
        nxn = jnp.outer(n, n)
        nxnxn = jnp.tensordot(n, nxn, axes=0)
        dxnxn = jnp.tensordot(jnp.diag(n), n, axes=0)
        dxdxn = jnp.eye(norb)[:, :, None] * jnp.diag(n)[None, :, :]
        dxn2 = jnp.eye(norb)[:, :, None] * rdm2[None, :, :]
        nxn2 = jnp.tensordot(n, rdm2, axes=0)
        d2n_deps2 = self.d2n_deps2(eps, beta)
        return (
            -dxdxn
            + dxnxn
            + dxnxn.transpose(2, 1, 0)
            + dxnxn.transpose(0, 2, 1)
            - dxn2
            - dxn2.transpose(2, 1, 0)
            - dxn2.transpose(0, 2, 1)
            + nxn2
            + nxn2.transpose(2, 1, 0)
            + nxn2.transpose(1, 0, 2)
            - 2 * nxnxn
            + d2n_deps2
        )


def sinkhorn(
    n,
    N,
    beta=1.0,
    eps=None,
    max_iters=100,
    threshold=10**-10,
    old=False,
    verbose=True,
):
    r"""(Bosonic) Sinkhorn algorithm

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

    n.shape[0]
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

    # Pack the results in a dictionary
    result["eps"] = eps
    result["errors"] = errors
    result["F"] = F
    result["S"] = S
    result["eps_GC"] = eps_GC
    result["S_GC"] = S_GC
    result["n_approx"] = n_approx
    return result

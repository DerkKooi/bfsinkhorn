from functools import partial
from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import jacfwd, jacrev, jit, lax, vmap
from jax.scipy.special import xlogy
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt._src.base import OptStep

from .utils import summinexp_vmap


@partial(jit, static_argnums=(1))
def compute_partition_function(eps: jnp.ndarray, N: int, beta: float) -> jnp.ndarray:
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
def compute_aux_partition_function(eps: jnp.ndarray, Q: jnp.ndarray, beta: float) -> jnp.ndarray:
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
      The partition function ratios for M=N-1 and M=N
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


def compute_aux_partition_function_direct(
    n: jnp.ndarray, eps: jnp.ndarray, Q: jnp.ndarray, N: int, beta: float
) -> jnp.ndarray:
    r"""Compute the auxiliary fermionic partition function ratios

    This computes $Q_M^p = Z_M^p/Z_N$ for M=N-1 and M=N
    Note that this corresponds to REMOVING an orbital

    Parameters
    ----------
    n : 1-dimensional ndarray
      The occupation numbers
    eps : 1- dimensional array
      The orbital energy of the orbital at hand
    Q : 1-dimensional ndarray of length N
      The partition function ratios for M=1 to M=N
    N : int
      The number of particles
    beta : float
      Inverse temperature

    Returns
    --------
    Qp : 2-dimensional ndarray of length (n_orb, 2)
      The auxiliary partition function ratios for M=N-1 and M=N
    """
    norb = eps.shape[0]

    # All of this is just to remove the diagonal
    # There must be a better solution
    eps_repeat = jnp.repeat(eps, norb).reshape(norb, norb)
    n_repeat = jnp.repeat(n, norb).reshape(norb, norb)
    eps_slash = jnp.empty((norb, norb - 1))
    n_slash = jnp.empty((norb, norb - 1))
    triu_indices = jnp.triu_indices(norb, k=1)
    tril_indices = jnp.tril_indices(norb, k=-1)
    eps_slash = eps_slash.at[(triu_indices[0], triu_indices[1] - 1)].set(eps_repeat[triu_indices])
    eps_slash = eps_slash.at[(tril_indices[0], tril_indices[1])].set(eps_repeat[tril_indices])
    n_slash = n_slash.at[(triu_indices[0], triu_indices[1] - 1)].set(n_repeat[triu_indices])
    n_slash = n_slash.at[(tril_indices[0], tril_indices[1])].set(n_repeat[tril_indices])

    # Compute the scaling factor for the orbital energies
    scale = jnp.sum(eps_slash * n_slash, axis=1) / N
    eps_slash = eps_slash - scale[:, None]
    Qp = vmap(compute_partition_function, in_axes=(0, None, None), out_axes=(1))(
        eps_slash, N, beta
    )[N - 2 :]
    Qp = Qp.at[0].set(Qp[0] / Q[N - 2])
    Qp = Qp.at[1].set(Qp[1] / Q[N - 1])
    return Qp


# vmap of compute_aux_partition_function over orbital energies
compute_aux_partition_function_vmap = jit(
    vmap(compute_aux_partition_function, in_axes=(0, None, None), out_axes=(1))
)


@jit
def compute_correlations(n: jnp.ndarray, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
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
def compute_correlations_degen(eps: float, Q: jnp.ndarray, beta: float) -> float:
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


# vmap of compute_aux_partition_function over degenerate orbital energies,
# they are not exactly equal so weigh them equally
compute_correlations_degen_vmap = jit(
    vmap(
        lambda eps, Q, beta: compute_correlations_degen((eps[0] + eps[1]) / 2, Q, beta),
        in_axes=(1, None, None),
        out_axes=0,
    )
)


@partial(jit, static_argnums=(1))
def compute_occupations(eps: jnp.ndarray, N: int, beta: float) -> jnp.ndarray:
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


def compute_correlations_full(
    n: jnp.ndarray, eps: jnp.ndarray, Q: jnp.ndarray, beta: float, degen_cutoff=1e-7
) -> jnp.ndarray:
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
            compute_correlations_degen_vmap(jnp.vstack((eps[degens[0]], eps[degens[1]])), Q, beta)
        )
    return correlations


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
    eps_new : 1-dimensional ndarray
      The updated orbital energies
    """

    # Fix norm in n
    n = n / jnp.sum(n) * N

    # Compute partition function ratios
    Q = compute_partition_function(eps, N, beta)

    # Compute auxiliary partition function missing one orbital
    Qp = compute_aux_partition_function_vmap(eps, Q, beta)

    # Compute new orbital energies
    if not old:
        # Fermionic Sinkhorn iteration
        eps_new = -jnp.log(n / (1 - n) * Qp[1] / Qp[0]) / beta
    else:
        # Regular Sinkhorn iteration
        eps_new = -jnp.log(n / Qp[0]) / beta

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
    return -jnp.log(n / (1 - n)) / beta


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
    return -jnp.sum(xlogy(n, n) + xlogy(1 - n, 1 - n)) / beta


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


@partial(jit, static_argnums=(1,))
def compute_free_energy(eps: jnp.ndarray, N: int, beta: float) -> float:
    # Compute partition function ratios
    Q = compute_partition_function(eps, N, beta)
    # Compute corresponding free energy differences
    dF = -jnp.log(Q) / beta
    # Compute free energy of the N-electron system.
    F = jnp.sum(dF)
    return F


class Sinkhorn:
    """(Bosonic) Sinkhorn algorithm"""

    def __init__(
        self,
        N: int,
        hotstart: int = 10,
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
        if hotstart > 0 and old:
            raise ValueError("hotstart is only for old = False")

        self._fixed_point = jit(partial(fixed_point, N=N, old=old))
        if hotstart > 0:
            self._fixed_point_hotstart = jit(partial(fixed_point, N=N, old=True))

        jaxopt_kwargs = {
            "maxiter": maxiter,
            "tol": tol,
            "verbose": verbose,
            "implicit_diff": implicit_diff,
            "jit": not verbose,
        }

        if anderson:
            # Anderson acceleration does not seem to offer any advantage unless old is true
            jaxopt_kwargs.update(
                {
                    "history_size": history_size,
                    "mixing_frequency": mixing_frequency,
                    "beta": anderson_beta,
                    "ridge": ridge,
                }
            )
            self._optimizer = AndersonAcceleration(self._fixed_point, **jaxopt_kwargs)
            if hotstart > 0:
                jaxopt_kwargs.update({"maxiter": hotstart})
                self._optimizer_hotstart = AndersonAcceleration(
                    self._fixed_point_hotstart, **jaxopt_kwargs
                )
        else:
            self._optimizer = FixedPointIteration(self._fixed_point, **jaxopt_kwargs)
            if hotstart > 0:
                jaxopt_kwargs.update({"maxiter": hotstart})
                self._optimizer_hotstart = FixedPointIteration(
                    self._fixed_point_hotstart, **jaxopt_kwargs
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
            eps0 = eps_GC_guess(n, beta)
            eps0 = normalize_eps(n, eps0, N)
            return self._optimizer.run(eps0, n, beta)

        def _run_GC_guess_hotstart(n: jnp.ndarray, beta: float = 1.0) -> OptStep:
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
            eps0 = eps_GC_guess(n, beta)
            eps0 = normalize_eps(n, eps0, N)
            eps_hotstart, _ = self._optimizer_hotstart.run(eps0, n, beta)
            return self._optimizer.run(eps_hotstart, n, beta)

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

        if hotstart > 0:
            self._run_GC_guess = jit(_run_GC_guess_hotstart)
            self.run_parallel = jit(vmap(_run_GC_guess_hotstart, in_axes=(0, None)))
        else:
            self._run_GC_guess = jit(_run_GC_guess)
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
        self.compute_partition_function = jit(
            lambda eps, beta=1.0: compute_partition_function(eps, self.N, beta)
        )
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
        Q = self.compute_partition_function(new_eps, beta)
        F = self.compute_free_energy(new_eps, beta)
        results = {
            "eps": new_eps,
            "n_approx": n_approx,
            "eps_error": state.error,
            "n_error": n_error,
            "iter_num": state.iter_num,
            "Q": Q,
            "F": F,
            "S": -beta * F,
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

    def compute_rdm2(self, n: jnp.ndarray, eps: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        r"""
        Compute the 2-RDM element < a^\dagger_p a^\dagger_q a_q a_p >

        This is equal to
        :math:`n_i n_j - \frac{\partial n_i}{\partial \epsilon_j} - \delta_{ij} n_i`

        Parameters
        ----------
        n : 1-dimensional ndarray (norb)
          The occupation numbers
        eps : 1-dimensional ndarray (norb)
          The orbital energies
        beta : float
          Inverse temperature

        Returns
        --------
        rdm2 : 2-dimensional ndarray (norb, norb, norb)
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
    n: jnp.ndarray,
    N: int,
    beta: float = 1.0,
    eps: Optional[jnp.ndarray] = None,
    max_iters: int = 100,
    threshold: float = 10**-10,
    old: bool = False,
    verbose: bool = True,
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

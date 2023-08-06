from functools import partial
from typing import Dict, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit, lax, vmap
from jax.scipy.special import xlogy
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt._src.base import IterativeSolver, OptStep

from .utils import summinexp_vmap


@partial(jit, static_argnums=(0))
def compute_partition_function(N: int, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
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
        # Q = Q.at[M - 1].set(C[1] * lax.scan(inner_recursion, 1.0, jnp.arange(1, M))[0] / M)
        return Q

    # return lax.scan(outer_loop, Q, jnp.arange(2, N + 1))[0]
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

    # def inner_recursion(k, previous):
    #     return 1 - expeps / Q[k] * previous

    def inner_recursion(previous, k):
        return 1 - expeps / Q[k] * previous, None

    # Qp = Qp.at[0].set(lax.fori_loop(0, N - 1, inner_recursion, Qp[0]) / Q[N - 1])
    # Qp = Qp.at[1].set(lax.fori_loop(0, N, inner_recursion, Qp[1]))
    Qp = Qp.at[0].set(lax.scan(inner_recursion, Qp[0], jnp.arange(0, N - 1))[0] / Q[N - 1])
    Qp = Qp.at[1].set(lax.scan(inner_recursion, Qp[1], jnp.arange(0, N))[0])
    return Qp


# vmap of compute_aux_partition_function over orbital energies
compute_aux_partition_function_vmap = jit(
    vmap(compute_aux_partition_function, in_axes=(0, None, None), out_axes=(1))
)


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


@partial(jit, static_argnums=(0))
def compute_occupations(N: int, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
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
    Q = compute_partition_function(N, eps, beta)

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


@jit
def normalize_eps(n: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
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

    return eps - jnp.sum(n * eps) / jnp.sum(n)


@partial(jit, static_argnums=(0,))
def compute_free_energy(N: int, eps: jnp.ndarray, beta: float) -> float:
    # Compute partition function ratios
    Q = compute_partition_function(N, eps, beta)
    # Compute corresponding free energy differences
    dF = -jnp.log(Q) / beta
    # Compute free energy of the N-electron system.
    F = jnp.sum(dF)
    return F


class Sinkhorn(eqx.Module):
    """(Bosonic) Sinkhorn algorithm"""

    N: int
    old: bool
    hotstart: int
    anderson: bool
    use_jacrev_deps_dn: bool
    use_jacrev_d2eps_dn2: bool
    use_jacrev_dn_deps: bool
    use_jacrev_d2n_deps2: bool
    _optimizer: Optional[IterativeSolver]
    _optimizer_old: Optional[IterativeSolver]

    def __init__(
        self,
        N: int,
        old: bool = False,
        hotstart: int = 0,
        anderson: bool = False,
        implicit_diff: bool = True,
        maxiter: int = 100,
        tol: float = 1e-10,
        verbose: bool = False,
        history_size: int = 5,
        mixing_frequency: int = 1,
        anderson_beta: float = 1,
        ridge: float = 1e-5,
        use_jacrev_deps_dn: bool = True,
        use_jacrev_d2eps_dn2: bool = True,
        use_jacrev_dn_deps: bool = False,
        use_jacrev_d2n_deps2: bool = False,
    ):
        self.N = N
        self.hotstart = hotstart
        self.old = old
        self.anderson = anderson
        self.use_jacrev_deps_dn = use_jacrev_deps_dn
        self.use_jacrev_d2eps_dn2 = use_jacrev_d2eps_dn2
        self.use_jacrev_dn_deps = use_jacrev_dn_deps
        self.use_jacrev_d2n_deps2 = use_jacrev_d2n_deps2

        if implicit_diff and (not use_jacrev_deps_dn or not use_jacrev_d2eps_dn2):
            raise ValueError("Implicit diff requires using jacrev for both deps_dn and d2eps_dn2")

        optimizer_kwargs = {"tol": tol, "verbose": verbose, "jit": not verbose}

        if anderson:
            optimizer_kwargs.update(
                {
                    "history_size": history_size,
                    "mixing_frequency": mixing_frequency,
                    "beta": anderson_beta,
                    "ridge": ridge,
                }
            )
            # Anderson acceleration does not seem to offer any advantage unless old is true
            if not old:
                self._optimizer = AndersonAcceleration(
                    self._fixed_point,
                    maxiter=maxiter - self.hotstart,
                    **optimizer_kwargs,
                )
            else:
                self._optimizer = None
            if old or hotstart > 0:
                self._optimizer_old = AndersonAcceleration(
                    self._fixed_point_old,
                    maxiter=maxiter if old else self.hotstart,
                    **optimizer_kwargs,
                )
            else:
                self._optimizer_old = None
        else:
            if not old:
                self._optimizer = FixedPointIteration(
                    self._fixed_point,
                    maxiter=maxiter - self.hotstart,
                    **optimizer_kwargs,
                )
            else:
                self._optimizer = None
            if old or hotstart > 0:
                self._optimizer_old = FixedPointIteration(
                    self._fixed_point_old,
                    maxiter=maxiter if old else self.hotstart,
                    **optimizer_kwargs,
                )
            else:
                self._optimizer_old = None

    def _fixed_point(
        self,
        eps: jnp.ndarray,
        n: jnp.ndarray,
        beta: float,
    ) -> jnp.ndarray:
        r"""
        Compute the orbital energies from a (Bosonic) Sinkhorn step

        Note that the orbital energies are normalized such that
        \sum_p n_p eps_p = 0

        Parameters
        ----------
        eps : 1-dimensional ndarray
        The orbital energies
        n : 1-dimensional ndarray
        The occupation numbers
        beta : float
        Inverse temperature

        Returns
        --------
        eps : 1-dimensional ndarray
        The updated orbital energies
        """

        # Fix norm in n
        n = n / jnp.sum(n) * self.N

        # Compute partition function ratios
        Q = compute_partition_function(self.N, eps, beta)

        # Compute auxiliary partition function missing one orbital
        Qp = compute_aux_partition_function_vmap(eps, Q, beta)

        # Compute new orbital energies
        # Fermionic Sinkhorn iteration
        eps = -jnp.log(n / (1 - n) * Qp[1] / Qp[0]) / beta

        return normalize_eps(n, eps)

    def _fixed_point_old(
        self,
        eps: jnp.ndarray,
        n: jnp.ndarray,
        beta: float,
    ) -> jnp.ndarray:
        r"""
        Compute the orbital energies from a (Bosonic) Sinkhorn step

        Note that the orbital energies are normalized such that
        \sum_p n_p eps_p = 0

        Parameters
        ----------
        eps : 1-dimensional ndarray
        The orbital energies
        n : 1-dimensional ndarray
        The occupation numbers
        beta : float
        Inverse temperature

        Returns
        --------
        eps : 1-dimensional ndarray
        The updated orbital energies
        """

        # Fix norm in n
        n = n / jnp.sum(n) * self.N

        # Compute partition function ratios
        Q = compute_partition_function(self.N, eps, beta)

        # Compute auxiliary partition function missing one orbital
        Qp = compute_aux_partition_function_vmap(eps, Q, beta)

        # Compute new orbital energies
        # Regular Sinkhorn iteration
        eps = -jnp.log(n / Qp[0]) / beta

        return normalize_eps(n, eps)

    def run_sinkhorn_fixed_iters(
        self,
        n: jnp.ndarray,
        eps: Optional[jnp.ndarray] = None,
        beta: float = 1.0,
        n_iters: int = 100,
    ) -> Dict[str, Union[jnp.ndarray, int, float]]:
        """
        Run the Sinkhorn algorithm with fixed number of iterations with GC guess

        Parameters
        ----------
        n : 1-dimensional ndarray
          The occupation numbers
        beta : float
            Inverse temperature
        eps : 1-dimensional ndarray
          The orbital energies
        niter : int
            Number of iterations to run

        Returns
        -------
        results : dict
            Dictionary containing the results
                eps : 1-dimensional ndarray
                The resulting orbital energies
                n_approx : 1-dimensional ndarray
                The computed occupation numbers from the orbital energies
                eps_error : float
                The error in the orbital energies
                n_error : 1-dimensional ndarray
                The error in the occupation numbers
                iter_num : int
                The number of iterations
                F : np.ndarray
                The free energies for M=0 to M=N
                S : float
                The entropy
        """
        # Make sure n is normalized
        n = n / jnp.sum(n) * self.N

        # Initialize eps if necessary

        eps_GC = eps_GC_guess(n, beta)
        if eps is None:
            eps = eps_GC

        n_error = jnp.array([jnp.sum(jnp.abs(n - compute_occupations(self.N, eps, beta)))])

        if self.old or self.hotstart > 0:
            # Initialize the state
            state = self._optimizer_old.init_state(eps, n, beta)
            step = self._optimizer_old.update(eps, state, n, beta)

            # Function to take an optimizer step
            def take_step_old(step: OptStep, _) -> Tuple[OptStep, float]:
                step = self._optimizer_old.update(*step, n, beta)
                current_n = compute_occupations(self.N, step.params, beta)
                return step, jnp.sum(jnp.abs(current_n - n))

            # Run iterations
            step, n_error_old = lax.scan(
                take_step_old, step, jnp.arange(n_iters if self.old else self.hotstart)
            )
            eps = step.params
            n_error = jnp.concatenate([n_error, n_error_old])
        if not self.old:
            # Initialize the state
            state = self._optimizer.init_state(eps, n, beta)
            step = self._optimizer.update(eps, state, n, beta)

            # Function to take an optimizer step
            def take_step(step: OptStep, _) -> Tuple[OptStep, float]:
                step = self._optimizer.update(*step, n, beta)
                current_n = compute_occupations(self.N, step.params, beta)
                return step, jnp.sum(jnp.abs(current_n - n))

            # Run iterations
            step, n_error_new = lax.scan(take_step, step, jnp.arange(n_iters - self.hotstart))
            n_error = jnp.concatenate([n_error, n_error_new])

        # Unpack step and return results in dictionary
        eps, state = step.params, step.state
        n_approx = compute_occupations(self.N, eps, beta)
        Q = compute_partition_function(self.N, eps, beta)
        F = compute_free_energy(self.N, eps, beta)
        results = {
            "eps": eps,
            "n_approx": n_approx,
            "eps_error": state.error,
            "n_error": n_error,
            "iter_num": state.iter_num,
            "Q": Q,
            "F": F,
            "S": -beta * F,
        }
        if self.anderson:
            results.update(
                {
                    "params_history": state.params_history,
                    "residual_gram": state.residual_gram,
                    "residuals_history": state.residuals_history,
                }
            )
        results["eps_GC"] = eps_GC
        results["S_GC"] = compute_S_GC(eps_GC, beta)

        return results

    def _run_eps_only(self, n: jnp.ndarray, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
        """
        Run the Sinkhorn algorithm

        Return only the orbital energies for use in automatic differentiation.

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
        eps : 1-dimensional ndarray
            The converged orbital energies
        """
        return self._optimizer.run(eps, n, beta).params

    def run_sinkhorn(
        self, n: jnp.ndarray, eps: Optional[jnp.ndarray] = None, beta: float = 1.0
    ) -> Dict[str, Union[jnp.ndarray, int, float]]:
        """
        Run the Sinkhorn algorithm

        Parameters
        ----------
        n : 1-dimensional ndarray
          The occupation numbers
        eps : 1-dimensional ndarray
          The starting orbital energies
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
        """
        # Make sure n is normalized
        n = n / jnp.sum(n) * self.N

        # Initialize eps if necessary
        eps_GC = eps_GC_guess(n, beta)
        if eps is None:
            eps = eps_GC

        # Run the Sinkhorn algorithm
        if self.old or self.hotstart > 0:
            eps, state = self._optimizer_old.run(eps, n, beta)
        if not self.old:
            eps, state = self._optimizer.run(eps, n, beta)

        # Compute all the results
        n_approx = compute_occupations(self.N, eps, beta)
        n_error = jnp.sum(jnp.abs(n - n_approx))
        Q = compute_partition_function(self.N, eps, beta)
        F = compute_free_energy(self.N, eps, beta)
        results = {
            "eps": eps,
            "n_approx": n_approx,
            "eps_error": state.error,
            "n_error": n_error,
            "iter_num": state.iter_num,
            "Q": Q,
            "F": F,
            "S": -beta * F,
            "eps_GC": eps_GC,
            "S_GC": compute_S_GC(eps_GC, beta),
        }
        if self.anderson:
            results.update(
                {
                    "params_history": state.params_history,
                    "residual_gram": state.residual_gram,
                    "residuals_history": state.residuals_history,
                }
            )
        return results

    __call__ = run_sinkhorn

    def deps_dn(self, n: jnp.ndarray, eps: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """
        Compute the derivative of the orbital energies with respect to the occupation numbers

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
        deps_dn : 2-dimensional ndarray
          The derivative of the orbital energies with respect to the occupation numbers
        """
        if self.use_jacrev_deps_dn:
            deriv = jacrev(self._run_eps_only, argnums=0)(n, eps, beta)
        else:
            deriv = jacfwd(self._run_eps_only, argnums=0)(n, eps, beta)

        # The derivative does not have the desired properties that
        # \sum_p \partial \epsilon_p / \partial n_q = 0
        # \sum_q \partial \epsilon_p / \partial n_q = 0
        # so we restore this by subtracting the projections

        ones = jnp.ones_like(n) / n.shape[0]
        return (
            deriv
            - jnp.outer(jnp.sum(deriv, axis=1), ones)
            - jnp.outer(ones, jnp.sum(deriv, axis=0))
            + jnp.outer(ones, ones) * jnp.sum(deriv)
        )

    def d2eps_dn2(self, n: jnp.ndarray, eps: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """
        Compute the second derivative of the orbital energies with respect to the occupation numbers

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
        d2eps_dn2 : 3-dimensional ndarray
          The second derivative of the orbital energies with respect to the occupation numbers
        """
        if self.use_jacrev_d2eps_dn2:
            deriv2 = jacrev(self.deps_dn, argnums=0)(n, eps, beta=beta)
        else:
            deriv2 = jacfwd(self.deps_dn, argnums=0)(n, eps, beta=beta)

        # The derivative does not have the desired properties that
        # \sum_r \partial^2 \epsilon_p / (\partial n_q \partial n_r) = 0
        # so we restore this by subtracting the projection
        ones = jnp.ones_like(n) / n.shape[0]
        return deriv2 - jnp.sum(deriv2, axis=2)[:, :, None] * ones[None, None, :]

    def dn_deps(self, eps: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """
        Compute the derivative of the occupation numbers with respect to the orbital energies

        Parameters
        ----------
        eps : 1-dimensional ndarray
          The orbital energies
        beta : float
          Inverse temperature

        Returns
        --------
        dn_deps : 2-dimensional ndarray
          The derivative of the occupation numbers with respect to the orbital energies
        """
        if self.use_jacrev_dn_deps:
            return jacrev(partial(compute_occupations, self.N), argnums=0)(eps, beta)
        else:
            return jacfwd(partial(compute_occupations, self.N), argnums=0)(eps, beta)

    def d2n_deps2(self, eps: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
        """
        Compute the second derivative of the occupation numbers with respect to the orbital energies

        Parameters
        ----------
        eps : 1-dimensional ndarray
          The orbital energies
        beta : float
          Inverse temperature

        Returns
        --------
        d2n_deps2 : 3-dimensional ndarray
          The second derivative of the occupation numbers with respect to the orbital energies
        """
        if self.use_jacrev_d2n_deps2:
            return jacrev(self.dn_deps, argnums=0)(eps, beta)
        else:
            return jacfwd(self.dn_deps, argnums=0)(eps, beta)

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
        rdm2 = self.compute_rdm2(n, eps, beta=beta)
        nxn = jnp.outer(n, n)
        nxnxn = jnp.tensordot(n, nxn, axes=0)
        dxnxn = jnp.tensordot(jnp.diag(n), n, axes=0)
        dxdxn = jnp.eye(norb)[:, :, None] * jnp.diag(n)[None, :, :]
        dxn2 = jnp.eye(norb)[:, :, None] * rdm2[None, :, :]
        nxn2 = jnp.tensordot(n, rdm2, axes=0)
        d2n_deps2 = self.d2n_deps2(eps, beta=beta)
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

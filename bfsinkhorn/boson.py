from typing import Dict, Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit, lax, vmap
from jax.scipy.special import xlogy
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt.base import IterativeSolver, OptStep

from .utils import minlogsumminexp, minlogsumminexp_array, minlogsumminexp_vmap


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

    Returns
    --------
    new_eps : 1-dimensional ndarray
    The normalized orbital energies
    """
    return eps - jnp.sum(n * eps) / jnp.sum(n)


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


@jit
def eps_update(n: jnp.ndarray, _: float, Fp: jnp.ndarray, beta: float) -> jnp.ndarray:
    return -jnp.log(n / (1 + n)) / beta + Fp[:, 1] - Fp[:, 0]


@jit
def eps_update_old(n: jnp.ndarray, FN: float, Fp: jnp.ndarray, beta: float) -> jnp.ndarray:
    return -jnp.log(n) / beta + FN - Fp[:, 0]


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


class Sinkhorn(eqx.Module):
    """(Bosonic) Sinkhorn algorithm"""

    N: int
    old: bool
    anderson: bool
    use_jacrev_deps_dn: bool
    use_jacrev_d2eps_dn2: bool
    use_jacrev_dn_deps: bool
    use_jacrev_d2n_deps2: bool
    _optimizer: IterativeSolver

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
        use_jacrev_deps_dn: bool = True,
        use_jacrev_d2eps_dn2: bool = True,
        use_jacrev_dn_deps: bool = False,
        use_jacrev_d2n_deps2: bool = False,
    ):
        self.N = N
        self.old = old
        self.anderson = anderson
        self.use_jacrev_deps_dn = use_jacrev_deps_dn
        self.use_jacrev_d2eps_dn2 = use_jacrev_d2eps_dn2
        self.use_jacrev_dn_deps = use_jacrev_dn_deps
        self.use_jacrev_d2n_deps2 = use_jacrev_d2n_deps2

        if implicit_diff and (not use_jacrev_deps_dn or not use_jacrev_d2eps_dn2):
            raise ValueError("Implicit diff requires using jacrev for both deps_dn and d2eps_dn2")

        if anderson:
            # Anderson acceleration does not seem to offer any advantage unless old is true
            self._optimizer = AndersonAcceleration(
                self.fixed_point,
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
                self.fixed_point,
                maxiter=maxiter,
                tol=tol,
                verbose=verbose,
                implicit_diff=implicit_diff,
                jit=not verbose,
            )

    def compute_free_energy(self, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
        r"""
        Compute bosonic free energies

        Parameters
        ----------
        eps : 1-dimensional ndarray
        The orbital energies
        beta : float
        Inverse temperature

        Returns
        --------
        F : 1-dimensional ndarray of length N+1
        The free energy F_M for M=0 to M=N
        """

        # First compute D_k = -1/\beta \log(\sum_p e^{-\beta k \eps_p})
        # in ascending order with inf pad on the left
        D = jnp.ones(2 * self.N - 1) * jnp.inf
        k = jnp.arange(1, self.N + 1)
        D = D.at[self.N - 1 :].set(minlogsumminexp_vmap(k, beta * eps) / beta)

        # Build array for free energies in descending order (!),
        # initialize at inf, except for M=0: free energy is zero
        F = jnp.ones(self.N + 1) * jnp.inf
        F = F.at[self.N].set(0)

        # Compute free energies iteratively for M=1 to M=N
        # We use some padding tricks such that
        # the slices are the same size for every iteration.
        def inner_loop(i, F):
            exponents = lax.dynamic_slice(D, [i - 1], [self.N]) + F[1 : self.N + 1]
            F = F.at[self.N - i].set((minlogsumminexp(beta * exponents) + jnp.log(i)) / beta)
            return F

        F = lax.fori_loop(1, self.N + 1, inner_loop, F)

        return jnp.flip(F)

    def compute_occupations(self, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
        r"""
        Compute the occupation numbers for a given set of orbital energies

        Parameters
        ----------
        eps : 1-dimensional ndarray
        The orbital energies
        beta : float
        Inverse temperature

        Returns
        --------
        n : 1-dimensional ndarray
        The occupation numbers corresponding to eps and beta
        """

        # Compute free energy
        F = self.compute_free_energy(eps, beta)

        # Compute auxiliary free energy
        Fp = compute_aux_free_energy_vmap(eps, F, beta)

        # Compute occupations
        n = jnp.exp(-beta * (eps + Fp[:, 0] - F[self.N]))

        return n

    def fixed_point(
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

        # Compute free energy
        F = self.compute_free_energy(eps, beta)

        # Compute auxiliary free energy
        Fp = compute_aux_free_energy_vmap(eps, F, beta)

        # Compute new orbital energies
        eps = lax.cond(self.old, eps_update_old, eps_update, *(n, F[-1], Fp, beta))

        return normalize_eps(n, eps)

    def run_sinkhorn_fixed_iters(
        self, n: jnp.ndarray, beta: float, eps: Optional[jnp.ndarray] = None, n_iters: int = 100
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

        init_n_error = jnp.sum(jnp.abs(n - self.compute_occupations(eps, beta)))

        # Initialize the state
        state = self._optimizer.init_state(eps, n, beta)
        step = self._optimizer.update(eps, state, n, beta)

        # Function to take an optimizer step
        def take_step(step: OptStep, _) -> Tuple[OptStep, float]:
            step = self._optimizer.update(*step, n, beta)
            current_n = self.compute_occupations(step.params, beta)
            return step, jnp.sum(jnp.abs(current_n - n))

        # Run iterations
        step, n_error = lax.scan(take_step, step, jnp.arange(n_iters))

        # Unpack step and return results in dictionary
        eps, state = step.params, step.state
        n_approx = self.compute_occupations(eps, beta)
        F = self.compute_free_energy(eps, beta)
        results = {
            "eps": eps,
            "n_approx": n_approx,
            "eps_error": state.error,
            "n_error": jnp.concatenate([jnp.array([init_n_error]), n_error]),
            "iter_num": n_iters,
            "F": F,
            "S": -beta * F[-1],
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
        self, n: jnp.ndarray, beta: float, eps: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
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
        eps, state = self._optimizer.run(eps, n, beta)

        # Compute all the results
        n_approx = self.compute_occupations(eps, beta)
        n_error = jnp.sum(jnp.abs(n - n_approx))
        F = self.compute_free_energy(eps, beta)
        results = {
            "eps": eps,
            "n_approx": n_approx,
            "eps_error": state.error,
            "n_error": n_error,
            "iter_num": state.iter_num,
            "F": F,
            "S": -beta * F[-1],
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

    def deps_dn(self, n: jnp.ndarray, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
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

    def d2eps_dn2(self, n: jnp.ndarray, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
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
            deriv2 = jacrev(self.deps_dn, argnums=0)(n, eps, beta)
        else:
            deriv2 = jacfwd(self.deps_dn, argnums=0)(n, eps, beta)

        # The derivative does not have the desired properties that
        # \sum_r \partial^2 \epsilon_p / (\partial n_q \partial n_r) = 0
        # so we restore this by subtracting the projection
        ones = jnp.ones_like(n) / n.shape[0]
        return deriv2 - jnp.sum(deriv2, axis=2)[:, :, None] * ones[None, None, :]

    def dn_deps(self, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
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
        return lax.cond(
            self.use_jacrev_dn_deps,
            jacrev(self.compute_occupations, argnums=0),
            jacfwd(self.compute_occupations, argnums=0),
            *(eps, beta),
        )

    def d2n_deps2(self, eps: jnp.ndarray, beta: float) -> jnp.ndarray:
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
        return lax.cond(
            self.use_jacrev_d2n_deps2,
            jacrev(self.dn_deps, argnums=0),
            jacfwd(self.dn_deps, argnums=0),
            *(eps, beta),
        )

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
        beta: float,
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

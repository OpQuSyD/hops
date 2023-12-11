"""Funcionality for dealing with a finite bath in HOPS."""
import numpy as np
import stocproc
from typing import Optional


class EtaFiniteBath(stocproc.StocProc):
    r"""A stochastic process for a finite environment implementing the stocproc api.

    This situation is modelled by the hamiltonian

    .. math::

       H(t) = H_\mathrm{sys} + L ∑_l γ_l^\ast \exp(ω_l t) a_l^† + L^† ∑_l γ_l \exp(-ω_l t) a_l

    with pure imaginary :math:`ω`.

    We define the stochastic process as

    .. math::

       η(t) = \mathrm{i} ∑_l γ_l \exp(-ω_l t) z_l

    with the :math:`z_l` being gaussian complex random variables with :math:`\langle z z^\ast\rangle = 1`.

    The process has the following properties:

    .. math::

       \begin{align}
       \langle η(t)\rangle &= 0 \\
       \langle η(t) η(s)\rangle &= 0 \\
       \langle η(t) η^\ast(s)\rangle &= ∑_l |γ_l|^2 \exp(-\mathrm{i} |ω_l| (t-s)) = ∑_l g_l \exp(-\mathrm{i} |ω_l| (t-s))
       \end{align}

    :param gamma: the coupling constants
    :param omega: the purely imaginary oscillator frequencies (imag. unit times frequency)
    :param T: temperature
    """
    # TODO: (Valentin) move to stocproc...

    def __post_init__(
        self, gamma: np.ndarray, omega: np.ndarray, T: Optional[float] = None
    ):
        self.z_ast = None
        self._gamma = np.asarray(gamma)
        self._omega = np.asarray(omega)
        self._T = T

        if not (np.ndim(self._gamma) == 1):
            raise RuntimeError("g must be 1D")

        if not (self._omega.shape == self._gamma.shape):
            raise RuntimeError("omega and gamma must have the same length")

        if not np.all(np.real(self._omega) == 0):
            raise RuntimeError("for finite bath, omega must be purely imaginary")

        if not np.all(np.imag(self._gamma) == 0):
            raise RuntimeError("for finite bath, gamma should be real")
        if self._T is not None:
            if self._T == 0:
                bar_n = 0
            else:
                bar_n = np.exp(-np.abs(self._omega) / self._T) / (
                    1 - np.exp(-np.abs(self._omega) / self._T)
                )
            self._gamma = np.sqrt(bar_n) * self._gamma

    def __repr__(self):
        return f"EtaFinitebathForNumerics({self._gamma, self._omega, self._T})"

    def new_process(self, z):
        """Compute a new realization of the stochastic process."""
        self.z = z

    def __call__(self, t):
        return 1j * np.sum(self._gamma * np.exp(-self._omega * t) * self.z)

    def __getstate__(self):
        return (self._gamma, self._omega)

    def __setstate__(self, state):
        self.z_ast = None
        self._gamma, self._omega = state

    def get_num_y(self):
        """Get the number of random variables needed to generate the processs."""
        return len(self._omega)

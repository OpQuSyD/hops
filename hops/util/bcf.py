import numpy as np
from scipy.special import gamma as gamma_func
import mpmath
import warnings
from typing import Union
from collections.abc import Callable
from numpy.typing import NDArray, ArrayLike
from . import bcf_fits
import logging
from functools import lru_cache
from .zeta_function import lerch_phi

log = logging.getLogger(__name__)


@lru_cache(maxsize=10000)
def zeta_func(s, a, shift=1):
    return lerch_phi(shift, s, a)


def coth(x):
    return -(1 + np.exp(-2 * x)) / np.expm1(-2 * x)


#############################################################
#
# class definition
#
#############################################################


class OhmicSD_zeroTemp:
    r"""
    Represents an super / sub Ohmic spectral density function

    Ohmic spectral density

    ..  math::

        J(ω) = η ω^s \exp(-ω / ω_c)

    This class implements set_state and get_state and is suitable for
    binfootprint.

    :param s: The :math:`s` parameter.
    :param eta: The :math:`η` parameter.
    :param w_c: The :math:`ω_c` parameter.
    :param normed: If set :any:`True`, ``eta`` is scaled such that
        :math:`∫_0^∞ J(ω) = 1`.
    :param unit_strength_normalization: If set :any:`True`,
        ``eta`` is scaled such that :math:`\mathrm{Im} ∫_0^∞ α(ω) = -1`.

    The options ``normed`` and ``unit_strength_normalization`` are
    mutually exclusive.
    """

    def __init__(
        self,
        s: float,
        eta: float,
        w_c: float,
        normed: bool = False,
        unit_strength_normalization: bool = False,
    ):
        self.s = s
        self.w_c = w_c
        self.normed = normed

        self.eta = eta

        if normed:
            self.eta = eta / self.integral()

        elif unit_strength_normalization:
            self.eta = s * np.pi / (gamma_func(s + 1) * w_c**s)

        self.c = self.eta * gamma_func(self.s)

    def __call__(self, w: Union[np.ndarray, float]):
        if isinstance(w, np.ndarray):
            res = np.empty_like(w)
            idx_l0 = np.where(w < 0)
            res[idx_l0] = 0
            idx_ge0 = np.where(w >= 0)
            res[idx_ge0] = (
                self.eta * w[idx_ge0] ** self.s * np.exp(-w[idx_ge0] / self.w_c)
            )
            return res
        else:
            if w < 0:
                return 0
            else:
                return self.eta * w**self.s * np.exp(-w / self.w_c)

    def maximum_at(self):
        return self.s * self.w_c

    def maximum_val(self):
        return self(self.maximum_at())

    def integral(self):
        """
        :math:`∫_0^∞ J(ω)`
        """

        return self.eta * self.w_c ** (self.s + 1) * gamma_func(self.s + 1)

    def reorganization_energy(self):
        """
        :math:`∫_0^∞ J(ω)/ω`
        """

        return self.eta * self.w_c**self.s * gamma_func(self.s)

    def shifted_bath_influence(self, t):
        r"""
        :math:`∫_0^∞ J(ω)/ω \exp(-i ω t)`
        """

        return self.c * (self.w_c / (1 + 1j * self.w_c * t)) ** self.s

    def __str__(self):
        return (
            "J(w) = eta w^s exp(-w / w_c)\n"
            + "eta  = {}\n".format(self.eta)
            + "s    = {}\n".format(self.s)
            + "w_c  = {}\n".format(self.w_c)
        )

    def __getstate__(self):
        return self.s, self.eta, self.w_c

    def __setstate__(self, state):
        self.__init__(*state)


class ShiftedSD:
    r"""Shifts the spectral density ``spectral_density`` by ``ω_s ≥ 0``, so that

    .. math::

      J(ω) \rightarrow J(ω - ω_s).
    """

    def __init__(
        self,
        spectral_density,
        ω_s: float,
    ):
        self.original_spectral_density = spectral_density
        self.ω_s = ω_s

        if ω_s < 0:
            raise ValueError(f"The value of ``ω_s`` has to be positive!")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.original_spectral_density}, {self.ω_s})"
        )

    def __call__(self, ω: Union[np.ndarray, float]):
        return self.original_spectral_density(ω - self.ω_s)

    def __getstate__(self):
        return self.original_spectral_density, self.ω_s

    def __setstate__(self, state):
        return self.__init__(*state)


class ShiftedBCF:
    r"""Shifts the BCF ``bcf`` by ``ω_s ≥ 0``, so that

    .. math::

      α(τ) \rightarrow α(τ) e^{-i ω τ}.
    """

    def __init__(
        self,
        bcf,
        ω_s: float,
    ):
        self.original_bcf = bcf
        self.ω_s = ω_s
        self._exp = -1j * self.ω_s

        if ω_s < 0:
            raise ValueError(f"The value of ``ω_s`` has to be positive!")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.original_bcf}, {self.ω_s})"

    def __call__(self, t: Union[np.ndarray, float]):
        return self.original_bcf(t) * np.exp(self._exp * t)

    def exponential_coefficients(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        The normalized zero temperature BCF fit coefficients
        :math:`G_i,W_i` with ``n`` terms.
        """
        if not hasattr(self.original_bcf, "exponential_coefficients"):
            raise NotImplemented(
                f"The bcf {self.original_bcf.__name__} does not implement the ``exponential_coefficients`` method."
            )

        g, w = self.original_bcf.exponential_coefficients(n)
        w += 1j * self.ω_s

        return g, w

    def __getstate__(self):
        return self.original_bcf, self.ω_s

    def __setstate__(self, state):
        return self.__init__(*state)


class OhmicBCF_zeroTemp(object):
    r"""Ohmic bath correlation functions (BCF)

    consider ohmic spectral density
    J(w) = eta w^s exp(-w / w_c)

    general BCF
    alpha(tau) = 1/pi int_0^infty d w J(w) exp(-i w tau)

    -> ohmic BCF
    alpha(tau) = eta/pi w_c^(s+1) int_0^infty d x x^s exp(-x) exp(-i x w_c t)
               = eta/pi w_c^(s+1) (1 + i w_c t)^(-s-1) gamma_func(s+1)

    :param normed: If set :any:`True`, ``eta`` is scaled such that
        :math:`∫_0^∞ J(ω) = 1`.
    :param unit_strength_normalization: If set :any:`True`,
        ``eta`` is scaled such that :math:`\mathrm{Im} ∫_0^∞ α(ω) = -1`.

    The options ``normed`` and ``unit_strength_normalization`` are
    mutually exclusive.
    """

    def __init__(self, s, eta, w_c, normed=False, unit_strength_normalization=False):
        self.s = float(s)
        self.w_c = float(w_c)
        self.normed = normed
        self.unit_strength_normalization = unit_strength_normalization
        if normed and unit_strength_normalization:
            raise RuntimeError(
                "Only one of ``normed`` and ``unit_strength_normalization`` can be true."
            )

        if normed:
            self.eta = None
            self._c1 = 1 / np.pi

        elif unit_strength_normalization:
            self.eta = None
            self._c1 = w_c * s

        else:
            self.eta = eta
            self._c1 = self.eta * gamma_func(s + 1) * w_c ** (s + 1) / np.pi

    def __call__(self, tau):
        return self._c1 * (1 + 1j * self.w_c * tau) ** (-(self.s + 1))

    def div1(self, tau):
        return (
            -self._c1
            * (self.s + 1)
            * (1 + 1j * self.w_c * tau) ** (-(self.s + 2))
            * 1j
            * self.w_c
        )

    def exponential_coefficients(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        The normalized zero temperature BCF fit coefficients
        :math:`G_i,W_i` with ``n`` terms.
        """

        g, w = bcf_fits.get_ohm_g_w(n, self.s, self.w_c, scaled=False)
        g *= self._c1

        return g, w

    def __getstate__(self):
        return self.s, self.eta, self.w_c, self.normed, self.unit_strength_normalization

    def __setstate__(self, state):
        self.__init__(*state)

    def __eq__(self, other):
        return (
            (self.s == other.s) and (self.eta == other.eta) and (self.w_c == other.w_c)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.s}, {self.eta}, {self.w_c}, {self.normed}, {self.unit_strength_normalization})"


class OhmicBCF_nonZeroTemp(object):
    def __init__(self, s, eta, w_c, beta):
        self.s_p1 = s + 1
        self.eta = eta
        self.w_c = w_c
        self.beta = beta
        self._c = self.eta / self.beta ** (self.s_p1) * gamma_func(self.s_p1) / np.pi
        self._beta_w_c = self.beta * self.w_c

    def __repr__(self):
        return f"{self.__class__.__name__}({self.s_p1 - 1}, {self.eta}, {self.w_c}, {self.beta})"

    def __call__(self, tau):
        if isinstance(tau, np.ndarray):
            res = np.empty(shape=tau.shape, dtype=np.complex128)
            res_flat = res.flat
            tau_flat = tau.flat
            for i, ti in enumerate(tau_flat):
                zf = zeta_func(
                    self.s_p1,
                    (1 + self._beta_w_c + 1j * self.w_c * ti) / self._beta_w_c,
                )
                res_flat[i] = self._c * (  # type: ignore
                    (self._beta_w_c / (1 + 1j * self.w_c * ti)) ** self.s_p1
                    + zf
                    + np.conj(zf)
                )
            return res
        else:
            zf = zeta_func(
                self.s_p1, (1 + self._beta_w_c + 1j * self.w_c * tau) / self._beta_w_c
            )
            return self._c * (
                (self._beta_w_c / (1 + 1j * self.w_c * tau)) ** self.s_p1
                + zf
                + np.conj(zf)
            )

    def __bfkey__(self):
        return self.s_p1, self.eta, self.w_c, self.beta


class Ohmic_StochasticPotentialCorrelations(object):
    def __init__(self, s, eta, w_c, beta, normed=False, shift=0):
        self.s_p1 = s + 1
        self.eta = eta
        self.w_c = w_c
        self.beta = beta

        if normed:
            self.eta = 1 / (gamma_func(self.s_p1) * self.w_c**self.s_p1)

        self._c = self.eta / self.beta ** (self.s_p1) * gamma_func(self.s_p1) / np.pi
        self._beta_w_c = self.beta * self.w_c

        self.shift = shift
        self._expshift = np.exp(-self.shift * self.beta)

    def __call__(self, tau):
        shift_fac = (
            np.exp(-self.shift * (tau * 1j)) * self._expshift if self.shift > 0 else 1
        )
        if isinstance(tau, np.ndarray):
            res = np.empty(shape=tau.shape, dtype=np.complex128)
            res_flat = res.flat
            tau_flat = tau.flat
            for i, ti in enumerate(tau_flat):
                res_flat[i] = self._c * zeta_func(  # type: ignore
                    self.s_p1,
                    (1 + self._beta_w_c + 1j * self.w_c * ti) / self._beta_w_c,
                    self._expshift,
                )
            return res * shift_fac
        else:
            return (
                self._c
                * zeta_func(
                    self.s_p1,
                    (1 + self._beta_w_c + 1j * self.w_c * tau) / self._beta_w_c,
                    self._expshift,
                )
                * shift_fac
            )

    def __bfkey__(self):
        return self.s_p1, self.eta, self.w_c, self.beta, self.shift


def BOSE_distribution(x_in: Union[ArrayLike, float]) -> np.ndarray:
    """Calculate :math:`(exp(x)-1)^-1` and check if ``x_in`` is positive."""

    x = np.asarray(x_in)
    result = np.empty_like(x_in)

    if (x <= 0).any():
        raise ValueError("x < 0")

    overthresh = x > 40

    np.exp(-x, out=result, where=overthresh)
    np.expm1(x, out=result, where=~overthresh)
    np.reciprocal(result, out=result, where=~overthresh)

    return result


# def BOSE_distribution_single(x):
#     """calc (exp(x)-1)^-1"""
#     if x < 0:
#         raise ValueError("x < 0")
#     elif x > 40:
#         return np.exp(-x)
#     else:
#         return 1 / np.expm1(x)


# def BOSE_distribution(x):
#     try:
#         return np.asarray([BOSE_distribution_single(xi) for xi in x])
#     except:
#         return BOSE_distribution_single(x)


class Ohmic_StochasticPotentialDensity(object):
    def __init__(self, s, eta, w_c, beta, normed=False, shift=0):
        self.osd = OhmicSD_zeroTemp(s, eta, w_c, normed=normed)
        self.beta = beta
        self.shift = shift

    def __call__(self, w):
        return BOSE_distribution(self.beta * w) * self.osd(w - self.shift)

    def __bfkey__(self):
        return self.osd, self.beta, self.shift


class LorentzianSD(object):
    r"""Lorentzian spectral density, given by

    .. math::

        J(\omega) = \sum_i \frac{\eta_i}{1 + (\omega-\omega_{c,i})^2 / \gamma_i^2}

    """

    def __init__(self, eta, gamma, omega_c):
        try:
            l = len(eta)
        except TypeError:
            eta = [eta]
            l = 1

        try:
            l_ = len(gamma)
        except TypeError:
            gamma = [gamma]
            l_ = 1

        if l != l_:
            raise RuntimeError("'eta' and 'gamma' must have the same length")

        try:
            l_ = len(omega_c)
        except TypeError:
            omega_c = [omega_c]
            l_ = 1

        if l != l_:
            raise RuntimeError("'eta' and 'omega_c' must have the same length")

        self.eta = eta
        self.gamma = gamma
        self.omega_c = omega_c
        self._l = l

    def __call__(self, w):
        J = 0
        for i in range(self._l):
            J += self.eta[i] / (1 + ((w - self.omega_c[i]) / self.gamma[i]) ** 2)
        return J

    def __bfkey__(self):
        return self.eta, self.gamma, self.omega_c

    def maximum_at(self):
        return max(self.omega_c)

    def maximum_val(self):
        return self(self.maximum_at())


class LorentzianBCF(object):
    r"""lorentzian bath correlation functions (BCF)

    Bath correlation function resulting from a lorentzian spectral density

    .. math::

        J(\omega) = \sum_i \frac{\eta_i}{1 + (\omega-\omega_{c,i})^2 / \gamma_i^2}

    From the general formula for BCF

    .. math::

        \alpha(\tau) = 1/\pi \int_0^\infty J(\omega) \exp(-i \omega \tau) d\omega

    we obtain the Lorentzian BCF

    ..math::

        \alpha(\tau) = \sum_i \eta_i * \gamma_i * \exp(-\gamma_i |\tau| - 1j * omega_{c,i}*\tau )

    Requires ``negative_frequencies=True``.
    """

    def __init__(self, eta, gamma, omega_c):
        try:
            l = len(eta)
        except TypeError:
            eta = [eta]
            l = 1

        try:
            l_ = len(gamma)
        except TypeError:
            gamma = [gamma]
            l_ = 1

        if l != l_:
            raise RuntimeError("'eta' and 'gamma' must have the same length")

        try:
            l_ = len(omega_c)
        except TypeError:
            omega_c = [omega_c]
            l_ = 1

        if l != l_:
            raise RuntimeError("'eta' and 'omega_c' must have the same length")

        self.eta = eta
        self.gamma = gamma
        self.omega_c = omega_c
        self._l = l

    def __call__(self, tau):
        alpha = 0
        for i in range(self._l):
            alpha += (
                self.eta[i]
                * self.gamma[i]
                * np.exp(-self.gamma[i] * np.abs(tau) - 1j * self.omega_c[i] * tau)
            )
        return alpha

    def __getstate__(self):
        return self.eta, self.gamma, self.omega_c

    def __setstate__(self, state):
        self.__init__(*state)

    def __eq__(self, other):
        return (
            np.all(self.omega_c == other.omega_c)
            and np.all(self.eta == other.eta)
            and np.all(self.gamma == other.gamma)
        )

    def __repr__(self):
        s = "\n---------------------------------------"
        s += "\nbcf    : eta_i gamma_i exp(- gamma_i |t| - 1j omega_c_i t)"
        s += "\neta    :{}".format(self.eta)
        s += "\ngamma  :{}".format(self.gamma)
        s += "\nomega_c:{}".format(self.omega_c)
        s += "\n---------------------------------------\n"
        return s

    def get_g_w_for_hierarchy(self):
        return (
            [self.eta[i] * self.gamma[i] for i in range(self._l)],
            [self.gamma[i] + 1j * self.omega_c[i] for i in range(self._l)],
        )


class PseudoSD(object):
    """
    represents the pseudo spectral density of a BCF of a non-zero temperature environment
    """

    def __init__(self, sd_at_t_zero: Callable[[float], float], T: float):
        self.sd = sd_at_t_zero
        self.T = T

    def __bfkey__(self):
        return self.T, self.sd.__bfkey__() if hasattr(self.sd, "bfkey") else self.sd

    def __str__(self):
        return "{} at T={}".format(self.sd, self.T)

    def __call__(self, omega: float, T: float = None):
        if T is None:
            T = self.T

        if isinstance(omega, np.ndarray):
            if T > 0:
                res = self.sd(np.abs(omega)) / (1 - np.exp(-np.abs(omega) / T))
                flat_view_on_res = res
                omega_flat = omega.flatten()
                idx_neg = np.where(omega_flat < 0)
                flat_view_on_res[idx_neg] = flat_view_on_res[idx_neg] * np.exp(
                    -np.abs(omega_flat[idx_neg]) / T
                )
            else:
                res = np.zeros(shape=omega.shape, dtype=np.float64)
                flat_view_on_res = res
                omega_flat = omega.flatten()
                idx_non_neg = np.where(omega_flat >= 0)
                flat_view_on_res[idx_non_neg] = self.sd(omega_flat[idx_non_neg])
        else:
            if T > 0:
                if omega >= 0:
                    res = self.sd(omega) / (1 - np.exp(-omega / T))
                else:
                    res = self.sd(-omega) * np.exp(omega / T) / (1 - np.exp(omega / T))
            else:
                if omega >= 0:
                    res = self.sd(omega)
                else:
                    res = 0

        return res


class BCF_aprx(object):
    r"""approximation of the bath correlation function using a multi-exponential representation

    alpha(tau) = sum_i=1^n g_i exp(-omega_i tau)
    """

    def __init__(self, g, omega):
        # g = np.asarray(g)
        # omega = np.asarray(omega)
        self._n = g.shape
        assert self._n == omega.shape
        self.g: np.ndarray = g
        self.omega: np.ndarray = omega

    def __bfkey__(self):
        return self.g, self.omega

    def __call__(self, tau):
        r"""return alpha(tau) = sum_i=1^n g_i exp(-w_i tau) for arbitrary shape of tau"""
        try:
            s_tau = tau.shape
            dims_tau = len(s_tau)

            tau_ = tau.reshape((1,) + s_tau)
            g_ = self.g.reshape(self._n + dims_tau * (1,))
            omega_ = self.omega.reshape(self._n + dims_tau * (1,))

            res = np.empty(shape=s_tau, dtype=np.complex128)
            idx_pos = np.where(tau >= 0)
            res[idx_pos] = np.sum(g_ * np.exp(-omega_ * tau_[(0,) + idx_pos]), axis=0)
            idx_neg = np.where(tau < 0)
            res[idx_neg] = np.conj(
                np.sum(g_ * np.exp(-omega_ * np.abs(tau_[(0,) + idx_neg])), axis=0)
            )
        except Exception as e:
            if tau >= 0:
                res = np.sum(self.g * np.exp(-self.omega * tau))
            else:
                res = np.sum(self.g * np.exp(self.omega * tau)).conj()

        return res

    def get_SD(self, t_max, n):
        """
        t0, t1, t2 ... ,tn=t_max

        al(t0), al(t1), al(t2), ... al(t n-1)    +    al(tn), al(t n+1) = al(t n-1)^ast, ... al(t 2n-1) = al(-1)^ast

        idx: 0 ... n-1   n .. -1

        -> dt = t_max / n-1

        """
        t, dt = np.linspace(0, t_max, n, retstep=True)
        alpha_t = self.__call__(t)
        alpha_t = np.hstack((alpha_t[:-1], np.conj(alpha_t[:0:-1])))

        N = 2 * (n - 1)
        t_, dt_ = np.linspace(0, 2 * t_max, N, endpoint=False, retstep=True)
        assert dt == dt_
        assert len(t_) == len(alpha_t)

        fft_alpha = np.fft.ifft(alpha_t) * dt * N

        dw = 2 * np.pi / N / dt

        freq = np.hstack((np.arange(N // 2), -np.arange(1, N // 2 + 1)[::-1])) * dw

        return 0.5 * np.fft.ifftshift(np.real(fft_alpha)), np.fft.ifftshift(freq)

    def get_SD_analyt(self, w: NDArray):
        sd = np.asarray(
            [
                np.sum(
                    (
                        self.g.real * self.omega.real
                        - self.g.imag * (wi - self.omega.imag)
                    )
                    / (self.omega.real**2 + (wi - self.omega.imag) ** 2)
                )
                for wi in w
            ]
        )
        return sd

    def ft(self, om):
        if isinstance(om, np.ndarray):
            res = np.empty(shape=om.shape, dtype=np.complex128)
            res_flat_view = res.flat
            om_flat = om.flatten().reshape(-1, 1)
            res_flat_view[:] = np.sum(
                self.g
                * 2
                * self.omega
                / ((om_flat - 1j * self.omega) * (om_flat + 1j * self.omega)),
                axis=1,
            )
            return res

        else:
            return (
                self.g
                * 2
                * self.omega
                / ((om - 1j * self.omega) * (om + 1j * self.omega))
            )

    def n(self):
        return self._n[0]


class WeightFunction_one(object):
    def __call__(self, tau, bcf):
        return 1

    def __bfkey__(self):
        return "weight_one"


class WeightFunction_relative(object):
    def __call__(self, tau, bcf):
        return 1 / np.abs(bcf(tau))

    def __bfkey__(self):
        return "weight_one_over_abs_bcf"

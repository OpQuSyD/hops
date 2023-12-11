"""
Implementation of
:any:`hops.util.abstract_truncation_scheme.TruncationScheme`.
"""

from typing import SupportsFloat, Union
import numpy as np

from hops.core.hierarchy_parameters import SysP
from .dynamic_matrix import DynamicMatrix, MatrixType
from collections.abc import Sequence
from .abstract_truncation_scheme import TruncationScheme
from hops.core.hi_idx import HiIdx
from hops.core.hierarchy_math import operator_norm


class TruncationScheme_Simplex_multi(TruncationScheme):
    """A simplex truncation scheme for multiple baths.

    The value of each element of the multi index must not exceed the
    corresponding value of ``kmax_list``. Addition the sum of all elements
    of the multi index may not exceed ``sum_kmax`` (see the parameter
    documentation for addititional information).

    :param kmax_list: The upper bounds for each element of the multi-index.
    :param sum_kmax: The simplex parameter for each bath.

                     If set to ``"simplex"`` the simplex parameter for each bath will be
                     maximum of the respective element of ``kmax_list``.

                     If set to ``"cuboid"`` the simplex parameter will be
                     the sum of the sum of each element of ``kmax_list``.
    """

    def __init__(self, kmax_list: list[list[int]], sum_kmax: Union[str, list[int]]):
        self.kmax_list: list[list[int]] = kmax_list
        """The upper bound of the multi index."""

        self.sum_kmax: list[int]
        """The simplex parameter."""

        if sum_kmax == "simplex":
            self.sum_kmax = [max(kmlist_i) for kmlist_i in self.kmax_list]
        elif sum_kmax == "cuboid":
            self.sum_kmax = [sum(kmlist_i) for kmlist_i in self.kmax_list]
        else:
            assert isinstance(sum_kmax, list)
            self.sum_kmax = sum_kmax

        if len(kmax_list) != len(self.sum_kmax):
            raise ValueError(
                "The lists `kmax_list` and `sum_kmax` must be of equal length."
            )

    def __call__(self, hi_idx):
        for i in range(hi_idx.N):
            k = 0
            for j in range(hi_idx.n_list[i]):
                if hi_idx[i, j] > self.kmax_list[i][j]:
                    return False
                k += hi_idx[i, j]
            if k > self.sum_kmax[i]:
                return False

        return True

    def __bfkey__(self):
        return (self.kmax_list, self.sum_kmax)


class TruncationScheme_Power_multi(TruncationScheme):
    r"""A generalized simplex condition where the condition

    ..  math::

        ∑_{μ=1}^{N^{(n)}}
        \left(\frac{k^{(n)}_{μ}}{k^{(n)}_{\max μ}}\right)^{p^{(n)}}
        \leq 1

    for each bath.

    :param kmax_list: A list of the :math:`\vec{k}_{\mathrm{max}}`
        for each bath.
    :param p_list: A list of the :math:`p` parameter for each bath.
    """

    @classmethod
    def from_g_w(
        cls,
        g: list[np.ndarray],
        w: list[np.ndarray],
        p: list[float],
        q: list[float],
        kfac: list[float],
        sqrt: bool = True,
    ):
        r"""Generate the ``kmax_list`` from the BCF expansion
        parameters :math:`G^{(n)}` and :math:`W^{(n)}`.

        The :math:`k^{(n)}_{\mathrm{max},μ}` are being chosen
        proportional to
        :math:`(\sqrt{|G^{(n)}_{μ}|}/|W^{(n)}_{μ}|)^{q^{(n)}}` (or
        :math:`(|G^{(n)}_{μ}|/|W^{(n)}_{μ}|)^{q^{(n)}}` if ``sqrt`` is
        :any:`False`.), normalized by the mininimum of that
        expression and multiplied by ``kfac``.  If any
        :math:`k^{(n)}_{\mathrm{max},μ}` is smaller than one, it will be
        set to one.

        :param g: The coefficients of the BCF expansion :math:`G^{(n)}_μ`.
        :param w: The exponents of the BCF expansion :math:`W^{(n)}_μ`.
        :param p: The ``p`` parameters.  See the class docstring.
        :param q: The ``q`` parameters.  See the method docstring.
        :param kfac: The scaling factors.  See the method docstring.
        :param sqrt: Whether to take the square root of the
            :math:`G_μ`.
        """

        g_treated = [np.sqrt(el) for el in g] if sqrt else g

        kmax_list_pre = [
            np.array([(abs(g_el[i]) / abs(w_el[i])) ** q_el for i in range(len(g_el))])
            for g_el, w_el, q_el in zip(g_treated, w, q)
        ]

        kmax_list = [
            np.clip(fac * lst / min(lst), 1, None)
            for lst, fac in zip(kmax_list_pre, kfac)
        ]

        return cls(kmax_list, p)

    def __init__(self, kmax_list: list[list[float]], p_list: list[float]):
        self.kmax_list: list[list[float]] = kmax_list
        self.p_list: list[float] = p_list

    def __call__(self, hi_idx: HiIdx) -> bool:
        for i in range(hi_idx.N):
            k: int = 0
            for j in range(hi_idx.n_list[i]):
                k += (hi_idx[i, j] / self.kmax_list[i][j]) ** self.p_list[i]
            if k > 1:
                return False

        return True

    def __bfkey__(self):
        return (self.kmax_list, self.p_list)


class TruncationScheme_mMode(TruncationScheme):
    """A truncation scheme where the total number of nonzero indices
    may not exceed ``m``.

    :param m: The maximum number of nonzero indices.
    """

    def __init__(self, m: int):
        self.m: int = m

    def __call__(self, hi_idx: HiIdx) -> bool:
        val = 0

        for i in range(hi_idx.N):
            for j in range(hi_idx.n_list[i]):
                if hi_idx[i, j]:
                    val += 1

        return val <= self.m

    def __bfkey__(self):
        return self.m


class BathMemory(TruncationScheme):
    """
    A truncation scheme that is based on estimating the influence of a
    hierarchy state on the lower hierarchy states through its norm.

    :param g: The factors of the bcf expansion.
    :param w: The exponents of the bcf expansion.
    :param bcf_scale: The bcf scale(s).
    :param L: The coupling operators.
    :param multipliers: A tuple of multipliers.  The first one is
        being multiplied to the nonlinear pump rate (only effecive for
        the nonlinear method).  The second one is multiplied to the
        norm estimate.

        Leaving them at ``1`` is usually a good idea.

    :param influence_tolerance: The maximum "influence" that the last
        hierarchy states are allowed to have on the next higher ones.
    :param nonlinear: Whether the nonlinear method is being used.
    """

    @classmethod
    def from_system(
        cls,
        system: SysP,
        multipliers: tuple[float, float] = (1, 1),
        influence_tolerance: float = 1e-2,
        nonlinear: bool = False,
    ):
        """
        An alternative constructor that uses the
        :any:`hops.core.hierarchy_parameters.SysP` to configure the
        truncation scheme.

        For the rest of the parameters, see the class docstring.

        :param system: The system configuration.
        """

        return cls(
            system.g,
            system.w,
            [np.ones_like(g) for g in system.g],  # system.g is already scaled
            system.L,
            multipliers,
            influence_tolerance,
            nonlinear,
        )

    def __init__(
        self,
        g: Sequence[np.ndarray],
        w: Sequence[np.ndarray],
        bcf_scale: Sequence[SupportsFloat],
        L: Sequence[DynamicMatrix],
        multipliers: tuple[float, float] = (1, 1),
        influence_tolerance: float = 1e-2,
        nonlinear: bool = False,
    ):
        self._l_norms = [L_i.max_operator_norm() for L_i in L]
        self._w_real = [w_i.real for w_i in w]
        self._w = w
        self._g = [g_i * scale for g_i, scale in zip(g, bcf_scale)]
        self._L = L

        self._multipliers = multipliers

        self._influence_tolerance = influence_tolerance
        self._norms: dict[HiIdx, float] = {}

    def _get_norm(self, hi_idx: HiIdx) -> float:
        """
        Get the norm estimate of the auxilliary state with index
        ``hi_idx``.

        The norms are being cached in a lookup table.
        """

        if hi_idx.depth == 0:
            return 1

        exp_norm = self._norms.get(hi_idx, None)

        if exp_norm is None:
            memory = 0
            val = 0
            for i in range(hi_idx.N):
                for j in range(hi_idx.n_list[i]):
                    val += hi_idx[i, j] * self._w_real[i][j]
                    if hi_idx[i, j] == 0:
                        continue

                    new_hi_idx = HiIdx.from_other(hi_idx)
                    new_hi_idx[i, j] -= 1

                    memory += np.abs(
                        (
                            self._l_norms[i]
                            * np.sqrt(np.abs(self._g[i][j] * hi_idx[i, j]))
                            * self._multipliers[1]
                            * self._get_norm(new_hi_idx)
                        )
                    )

            exp_norm = min([memory / abs(val), 1])

            self._norms[hi_idx] = exp_norm

        return exp_norm

    def __call__(self, hi_idx: HiIdx) -> bool:
        if hi_idx.depth == 1:
            return True

        expected_influence = 0
        this_norm = self._get_norm(hi_idx)

        for bath in range(len(hi_idx.n_list)):
            for i, ind_val in enumerate(hi_idx[bath]):
                k_ref = HiIdx.from_other(hi_idx)
                if hi_idx[bath, i] == 0:
                    continue

                if self._get_norm(k_ref) == 0:
                    continue

                val = self._l_norms[bath] * np.sqrt(np.abs(self._g[bath][i] * ind_val))

                if val > expected_influence:
                    expected_influence = val

        if expected_influence * (this_norm) > self._influence_tolerance:
            return True

        return False

    def __bfkey__(self):
        return (
            self._w,
            self._g,
            self._L,
            self._multipliers,
            self._influence_tolerance,
        )

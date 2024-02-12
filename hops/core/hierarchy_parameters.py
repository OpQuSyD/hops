"""Coniguration Parameter Objects for HOPS."""
from typing import Optional, Union, Any, SupportsFloat
from collections.abc import Sequence
from enum import Enum

from binfootprint import ABCParameter

from stocproc import StocProc
from ..util.abstract_truncation_scheme import TruncationScheme, TruncationScheme_Simplex
import binfootprint as bf
from dataclasses import dataclass
import numpy as np
from beartype import beartype
from .idx_dict import IdxDict
from ..util.dynamic_matrix import DynamicMatrix, ConstantMatrix


class ResultType(str, Enum):
    """Describes what information will be kept and saved after the HOPS
    integration."""

    #: save only the trajectory
    ZEROTH_ORDER_ONLY = "ZEROTH_ORDER_ONLY"

    #: save the trajectory and the first hierarchy states
    ZEROTH_AND_FIRST_ORDER = "ZEROTH_AND_FIRST"

    #: save the trajectory and the stochastic process
    ZEROTH_ORDER_AND_ETA_LAMBDA = "ZEROTH_ORDER_AND_ETA_LAMBDA"

    #: save everything
    ALL = "ALL"


class HiP(bf.ABCParameter):
    r"""
    Parameters to configure the HOPS hierarchy and output.

    :param terminator: Whether to use a terminator.

    :param k_max: The maximal hierarchy depth for a simplex cutoff.

        See
        :any:`hops.util.abstract_truncation_scheme.TruncationScheme_Simplex`\.
        Can be omitted if a ``truncation_scheme`` is provided.

    :param seed: The initial seed of the RNG.
    :param nonlinear: Whether to use the nonlinear HOPS.

    :param terminator: Whether to use a terminator.

        .. note::

            Currently not implemented.

    :param result_type: What result type to use.
    :param stream_result_type: The result type to use for streaming
        results.  If :any:`None`, the results won't be streamed.

        If not :any:`None`, results of this type will be written to a
        FIFO file specified when instantiating :any:`HIData`.

    :param accum_only: Whether to only accumulate results and not keep
        the trajectories.
    :param rand_skip: How many random samples to skip before
        beginning.

        Must be :any:`None` if ``accum_only`` is not set.

    :param truncation_scheme: A truncation scheme to use.

        If this is not specified, then ``k_max`` must be set!

    :param save_therm_rng_seed: Whether to save the random seed for
        the thermal process for each trajectory.

        See also :any:`HIData`.  This is used to reconstruct the
        thermal stochastic process.

    :param auto_normalize: Keep :math:`|\norm{ψ^0}-1|=1` by adding an
        extra term to the HOPS differential equation.

        This is usually a very good idea.

        .. note::

            This only affects the nonlinear method.
    """

    __slots__ = [
        "k_max",
        "seed",
        "nonlinear",
        "terminator",
        "result_type",
        "accum_only",
        "rand_skip",
        "truncation_scheme",
        "save_therm_rng_seed",
        "auto_normalize",
        "__non_key__",
    ]

    @beartype
    def __init__(
        self,
        k_max: Optional[int] = None,
        seed: int = 0,
        nonlinear: bool = True,
        terminator: bool = False,
        result_type: ResultType = ResultType.ZEROTH_ORDER_ONLY,
        accum_only: Optional[bool] = None,
        rand_skip: Optional[int] = None,
        truncation_scheme: Optional[TruncationScheme] = None,
        save_therm_rng_seed: bool = False,
        auto_normalize: bool = True,
        stream_result_type: Optional[ResultType] = None,
    ):
        self.k_max = k_max
        self.seed = seed
        self.nonlinear = nonlinear

        self.terminator = terminator

        if self.terminator:
            raise NotImplementedError("A terminator is not yet implemented.")

        self.result_type = result_type
        self.accum_only = accum_only
        self.rand_skip = rand_skip or 0
        self.auto_normalize = auto_normalize

        self.save_therm_rng_seed = save_therm_rng_seed

        if accum_only is None:
            if self.rand_skip > 0:
                raise ValueError(
                    "If accum_only is 'None' (not set) rand_skip must also be '0'!"
                )

        self.truncation_scheme: TruncationScheme

        if k_max is None:
            if truncation_scheme is None:
                raise ValueError("Specify 'k_max' or provide a 'truncation scheme'!")

            self.truncation_scheme = truncation_scheme

        if k_max is not None:
            if truncation_scheme is not None:
                raise ValueError("Specify EITHER 'k_max' or 'truncation_scheme'!")

            self.truncation_scheme = TruncationScheme_Simplex(k_max)

        self.__non_key__: dict[str, Any] = dict(stream_result_type=stream_result_type)


class IntP(bf.ABCParameter):
    """Parameters to configure the integration.

    :param t_steps: A tuple containing the maximal simulation time and
        how many time steps to store.
    :param t: The time points which are to be stored.

        Is only effective if ``t_steps`` is :any:`None`.

    :param solver_args: See :any:`solver_args`.
    """

    __slots__ = ["t", "solver_args"]

    @beartype
    def __init__(
        self,
        t_steps: Optional[tuple[SupportsFloat, int]] = None,
        t: Optional[np.ndarray] = None,
        **solver_args
    ):
        assert (
            t is not None or t_steps is not None
        ), "You must specify either t or t_steps"

        self.t: np.ndarray
        """The time points which are to be stored."""

        if t_steps is not None:
            t_max, steps = t_steps
            self.t = np.linspace(0, float(t_max), steps)
        else:
            assert t is not None
            self.t = t

        assert (
            self.t[0] >= 0
        ).all(), """The first time point must be greater than zero."""

        self.solver_args: dict[str, Any] = solver_args
        """A dictionary of arguments passed directly into
        :any:`scipy.integrate.solve_ivp`."""

    @property
    def t_max(self):
        """The maximal simulation time."""
        return np.max(self.t)

    @property
    def t_steps(self):
        """The number of time steps that is being saved."""
        return self.t.size


class SysP(bf.ABCParameter):
    r"""Parameters to describe the physical system.

    :param H_sys: The time independent part of the system
        hamiltonian.

    :param L: The coupling operators :math:`L^{(n)}`.
    :param psi0: The initial state.
    :param g: The coefficients in the bcf expansions.

        ..  math::

            α^{(n)}(t)\approx ∑_{μ=1}^{N_n} G^{(n)}_μ\exp(-W^{(n)}_μ t).

        Every element of the list corresponds to one bath.


    :param w: The exponents :math:`W^{(n)}_i` in the bcf expansions.

    :param bcf_scale: The scaling factors :math:`η^{(n)}` multiplied
        to the BCF.  Controls the coupling strength.

    :param T: The temperatures of the baths.

        ..  note::

            This parameter only is descriptive and doesn't influence HOPS.
    :param descripton: A free form description of the configuration.
    """

    __slots__ = [
        "H_sys",
        "L",
        "psi0",
        "g",
        "w",
        "bcf_scale",
        "__non_key__",
    ]

    @beartype
    def __init__(
        self,
        H_sys: Union[list[list], np.ndarray, DynamicMatrix],
        L: Sequence[Union[list[list], np.ndarray, DynamicMatrix]],
        psi0: np.ndarray,
        g: list[np.ndarray],
        w: list[np.ndarray],
        bcf_scale: Sequence[SupportsFloat],
        T: Optional[Sequence[SupportsFloat]] = None,
        description: Optional[str] = None,
    ):
        self.H_sys = (
            H_sys if isinstance(H_sys, DynamicMatrix) else ConstantMatrix(H_sys)
        )
        self.L: list[DynamicMatrix] = [
            L_i if isinstance(L_i, DynamicMatrix) else ConstantMatrix(L_i) for L_i in L
        ]
        self.psi0 = psi0 / np.linalg.norm(psi0)

        self.w = w
        self.bcf_scale = bcf_scale
        self.g = [g_el * scale for (g_el, scale) in zip(g, self.bcf_scale)]
        """The BCF expansion coefficients.

        .. warning::

           They have been multiplied with ``bcf_scale``.
        """

        self.__non_key__: dict[str, Any] = {}
        self.__non_key__["T"] = T
        self.__non_key__["desc"] = description

        # TODO: more checks!!

    @property
    def dim_sys(self) -> int:
        """The system dimension."""

        return self.H_sys.shape[0]

    @property
    def num_baths(self) -> int:
        """The number of baths."""

        return len(self.g)

    @property
    def number_bcf_terms(self) -> list[int]:
        """
        A list of the number of terms :math:`N^{(n)}` in the BCF expansions.
        """

        return [len(g) for g in self.g]

    @property
    def total_number_bcf_terms(self) -> int:
        """
        The number of terms in the BCF expansions added together.
        """

        return sum(self.number_bcf_terms)

    @property
    def unscaled_g(self) -> list[np.ndarray]:
        """
        The :any:`g` unscaled with the respective BCF scales
        ``bcf_scale``.
        """

        return [g / scale for (g, scale) in zip(self.g, self.bcf_scale)]


@beartype
@dataclass
class HIParams(ABCParameter):
    """A simple container to hold all of the HOPS configuration."""

    __slots__ = ["HiP", "IntP", "SysP", "Eta", "EtaTherm", "__non_key__"]

    HiP: HiP
    """Hierarchy Parameters."""

    IntP: IntP
    """Integration Parameters."""

    SysP: SysP
    """System Parameters."""

    Eta: Sequence[StocProc]
    """The driving stochastic processes for each bath (NMQSD).

    Use :any:`hops.core.utility.ZeroProcess` to set the
    process to zero.

    The scaling will be set automatically.
    """

    EtaTherm: Sequence[Optional[StocProc]]
    """The thermal stochastic processes (for T > 0) for each bath.

    The scaling will be set automatically.
    """

    def __post_init__(self):
        if not (len(self.Eta) == len(self.EtaTherm) == len(self.SysP.bcf_scale)):
            raise ValueError(
                "`Eta` and `EtaTherm` must have a length equal to the bath size."
            )
        for scale, eta, eta_therm in zip(self.SysP.bcf_scale, self.Eta, self.EtaTherm):
            eta.set_scale(scale)
            if eta_therm:
                eta_therm.set_scale(scale)

        if (
            self.HiP.nonlinear
            and self.HiP.auto_normalize
            and (np.vdot(self.SysP.psi0, self.SysP.psi0) - 1) > 1e-4
        ):
            raise ValueError(
                "The initial state should be normalized if `SysP.auto_normalize` is `True`."
            )

        self.__non_key__: dict[str, Any] = {}
        self.__non_key__["indices"] = IdxDict(n_list=self.SysP.number_bcf_terms)
        self.__non_key__["indices"].make(self.HiP.truncation_scheme)

    @property
    def indices(self):
        """The hierachy index lookup table for the HOPS computation."""
        return self.__non_key__["indices"]

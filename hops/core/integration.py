r"""
 The actual implementation of the HOPS method for the linear and
 nonlinear case.

 The quantities used here sometimes have a matrix index
 :math:`(\vec{k})_{i,n}=k_i^{(n)}` which is being flattened out by the
 use of a lookup table :any:`IdxDict`.  The index :math:`n` labels the
 bath (there are a total of :math:`N_b` baths) and the index :math:`i`
 labels the respective BCF expansion term.  This is implicit in most
 docstrings.

 For the notation used here see also :ref:`notation`.

 The HOPS method uses an exponential expansion
 :math:`\sum_{μ=1}^{N^{(n)}} G^{(n)}_μ \exp(-W^{(n)}_μ t)` of the
 environment bath correlation functions (BCFs).  The number of
 expansion terms :math:`N^{(n)}` is herein referred to as "size" of
 the environment.  For each term in the in the BCF expansion an
 associated hierarchy depth :math:`l_n` can be defined.

 The general hops equations are

 ..  math::

     \dot{ψ}^\underline{k} = O ψ^\underline{k} + K(t) ψ^\underline{k}
     +  ∑_{n=1}^{N_b}B^{(n)}(t)∑_{j=1}^{N^{(n)}}
     \frac{G^{(n)}_j}{\bar{g}^{(n)}_j} ψ^{\underline{k} + \underline{e}_{n,j}}
     +  ∑_{n=1}^{N_b}C^{(n)}(t) ∑_{l=1}^{N^{(n)}}
     k^{(n)}_l \bar{g}^{(n)}_l ψ^{\underline{k} - \underline{e}_{n,j}},

 with :math:`O = -∑_{n=1}^{N_b}∑_{μ=0}^{N^{(n)}} k^{(n)}_μ W^{(n)}_μ`
 and :math:`(\underline{e}_{n,j})_{k,l}=δ_{nk}δ_{jl}`.  The details
 may be found in ``Hartmann2021``.

 As discussed above :math:`\underline{k}` is a multi-index.  The range
 of this multi-index may be somewhat nontrivial, so that it is easiest
 to flatten it out into a scalar index.

 This is being achieved with the help of the functionality in
 :any:`hops.core.idx_dict` module.

 The object that is dealt with to compute the time derive now has the
 shape ``(number of indices, system dimension)`` and shall be called
 :math:`Ψ`.  The above hops equation can now be transformed into a
 matrix equation that acts in two stages.

 ..  note::

     On the lowest level, this object is again flattened out to conform
     with the needs of the ``scipy`` integration routines. This however,
     is of no concern for the current presentation.

 The first stage is dealing with the connections between different
 hierarchy state (the parts :math:`O` and the sums in the above
 equation).  Those are realized by the action sparse matrices acting
 on the first dimension of :math:`Ψ`.

 The second stage acts on the system dimensions via the
 (time-dependent) matrices :math:`K`, :math:`B` and :math:`C`.

 This leads to the matrix equation

 ..  math::

     \dot{Ψ} = OΨ + [K(t) Ψ^⊺]^⊺
             + ∑_{n=1}^{N_b}[B^{(n)}(t) (M^{(n)}_\mathrm{up} Ψ)^⊺]^⊺
             + ∑_{n=1}^{N_b}[C^{(n)}(t) (M^{(n)}_\mathrm{down} Ψ)^⊺]^⊺

 which can be handled rather nicely by the integration routines.

 For concrete definitions of those matrices see the implementations of
 :any:`hops.core.integration.HOPSActor`:

 * :any:`hops.core.integration.LinearHOPSActor`

 * :any:`hops.core.integration.NonLinearHOPSActor`.

 Implementation Details
 ~~~~~~~~~~~~~~~~~~~~~~

 The following ``get_*`` functions are creating the sparse matrix that
 convey the hierarchy structure.  After those come the `HOPSActor`
 class and its implementation.  They contain everything to actually
 integrate a hops trajectory.  They can be subclassed to implement
 different methods.

 If a new HOPS method is implemented: think about where to fit it and
 if you have to generalize the abstract :any:`HOPSActor`.  The
 ``get_*`` functions may have to become members of the actor classes
 as the structure differs between the methods.

 The :any:`HOPSSupervisor` is just a dispatcher that instantiates the
 :any:`HOPSActor`\ s and saves the results using :any:`HIData`.

 It should ideally be independent of the concrete implementatins of
 the :any:`HOPSActor`.
 """

# python environment libs
from abc import ABC, abstractmethod
from beartype import beartype
import logging
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as spe
import scipy.integrate
from typing import Optional, Type
import stocproc as sp

from . import hierarchy_data as hid
from . import hierarchy_parameters as hip

from .utility import uni_to_gauss
from ..util.dynamic_matrix import DynamicMatrix, MatrixType, DynamicMatrixList

from .hi_idx import HiIdx
from .idx_dict import IdxDict

import ray
from ray.exceptions import TaskCancelledError
from tqdm import tqdm

import signal
from . import signal_delay

log = logging.getLogger(__name__)

# Note to Kai: you may need to modify the following three functions
# (or create alternative versionsion)
#
# Talk to Valentin first!


def get_O_vector(
    w: list[np.ndarray],
    indices: IdxDict,
) -> np.ndarray:
    r"""
    Returns a collumn vector of prefactors :math:`O_\underline{k}` as
    in :math:`O_\underline{k}\psi^{\underline{k}} =
    -\left(\sum_{n=1}^{N_b}\sum_{l=1}^{N^{(n)}} k^{(n)}_l
    W^{(n)}_l\right) \psi^{\underline{k}}` of the shape
    ``(len(psi),1)`` where ``len(psi)`` is the length of the
    stochastic vector inclusive auxilliary states.

    :param w: the exponential factors :math:`W^{(n)}_i` in the BCF
              expansion.
    :param idx_dict: the lookup table for the indices
    """

    size = len(indices)
    o_vec = np.empty(shape=(size, 1), dtype=np.complex128)  # collumn vector
    all_w = np.concatenate(w)

    for k, scalar in indices.items():
        w_sum = 0
        w_sum = sum(k.get_all_k_np_array() * all_w)
        o_vec[scalar, 0] = -w_sum

    return o_vec


@beartype
def get_M_up(
    bath_index: int,
    g: np.ndarray,
    indices: IdxDict,
) -> spe.spmatrix:
    r"""
    :param bath_index: The index (zero based) of the bath.
    :param g: The prefactors :math:`G^{(n)}_i` in the BCF expansion
              (see module docstring).

        For finite baths, this is the coupling constant.

    :param indices: The lookup table for the hierarchy indices
        (already initialized).

    :returns: A sparse matrix for the terms in the HOPS equations
              that couple to the higher hierarchy depths.

        The term that is being addressed is (in the usual notation):

        ..  math::

            ∑_{j=1}^{N^{(n)}}
               \frac{G^{(n)}_j}{\bar{g}^{(n)}_j}
                    ψ^{\underline{k} + \underline{e}_{n,j}}

        The above only concerns a single hierarchy state.  The code
        here deals with all of them at once by creating the sparse
        matrix :math:`M` so that

        ..  math::

            \frac{d}{dt}\psi = B^{(n)} M_{\mathrm{up}}^{(n)} \psi.
    """

    data = []
    row = []
    col = []

    for hi_idx_bin, idx_ref in indices.binitems():
        hi_idx = indices.bin_to_idx(hi_idx_bin)

        # coupling from level k=hi_idx[bath_index][i] to k+1
        for i, gi in enumerate(g):
            # increase each entry in kappa by one
            hi_idx_to = HiIdx.from_other(hi_idx)
            hi_idx_to[bath_index, i] += 1

            # the corresponding bytes as key
            hi_idx_to_bin = hi_idx_to.to_bin()

            # if the key matches an equation considered
            # in the truncated hierarchy
            if hi_idx_to_bin in indices:
                idx_to = indices[hi_idx_to_bin]
                data.append(1j * np.sqrt(hi_idx_to[bath_index, i] * gi))
                row.append(idx_ref)
                col.append(idx_to)

    size = len(indices)
    return spe.coo_matrix((data, (row, col)), shape=(size, size)).tocsr()


@beartype
def get_M_down(
    bath_index: int,
    g: np.ndarray,
    indices: IdxDict,
) -> spe.spmatrix:
    r"""
    :param bath_index: The index (zero based) of the bath.
    :param g: The prefactors :math:`G^{(n)}_i` in the BCF expansion
              (see module docstring).

        For finite baths, this is the coupling constant.

    :param indices: The lookup table for the hierarchy indices
        (already initialized).

    :returns: A sparse matrix for the terms in the HOPS equations
              that couple to the lower hierarchy depths.

        The term that is being addressed is (in the usual notation)

        ..  math::

            C^{(n)} ∑_{l=1}^{N^{(n)}}
             k^{(n)}_l \bar{g}^{(n)}_l ψ^{\underline{k} - \underline{e}_{n,j}}.

        The above only concerns a single hierarchy state.  The code
        here deals with all of them at once by creating the sparse
        matrix :math:`M` so that

        ..  math::

            \frac{d}{dt}\psi = C^{(n)} M_{\mathrm{down}}^{(n)} \psi.

        See also :any:`get_M_up`.
    """

    row = []
    col = []
    data = []

    for hi_idx_bin, idx_ref in indices.binitems():
        hi_idx = indices.bin_to_idx(hi_idx_bin)

        # coupling from k=hi_idx[bath_index][i] to k-1
        for i in range(len(g)):
            # decrease each entry in k by one
            if hi_idx[bath_index, i] == 0:
                continue

            hi_idx_to = HiIdx.from_other(hi_idx)
            hi_idx_to[bath_index, i] -= 1

            idx_to = indices[hi_idx_to]
            data.append(-1j * np.sqrt(hi_idx[bath_index, i] * g[i]))
            row.append(idx_ref)
            col.append(idx_to)

    size = len(indices)
    return spe.coo_matrix((data, (row, col)), shape=(size, size)).tocsr()


class ThermNoiseHamiltonian(DynamicMatrix):
    """Defines a time-dependent (callable) Hamiltonian which accounts for a
    thermal initial environmental state.

    Instances can be called with a time argument.


    .. note::

       __getstate__ is used to uniquely identify a instance
       (see binaryfootprint library).


    :param stoc_proc: the stochastic process of the thermal environment
    :param L: the coupling operator
    """

    def __init__(self, stoc_proc: sp.StocProc, L: DynamicMatrix):
        #: the coupling operator
        self.L: DynamicMatrix = L

        #: the thermal stochastic process
        self.stoc_proc: sp.StocProc = stoc_proc

        self.new_process(np.zeros(self.get_num_y()))  # type: ignore
        # TODO: (Valentin) typing in stocproc

    def new_process(self, z: np.ndarray):
        """Generate a new realization of the thermal stochastic process from
        the complex gaussian random numbers ``z``."""
        self.stoc_proc.new_process(z)

    def get_num_y(self):
        """
        :returns: the number of samples required to generate a new realization
                  of the underlying stochastic process

                  See :any:`new_process`.
        """
        return self.stoc_proc.get_num_y()

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        mat = self.stoc_proc(t).conjugate() * self.L(t)
        return mat + np.transpose(mat.conj(), axes=(0, 2, 1))

    def __bfkey__(self):
        return self.__getstate__()

    def __getstate__(self):
        return self.L, self.stoc_proc

    def __setstate__(self, state):
        self.L, self.stoc_proc = state

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.stoc_proc)}, {repr(self.L)})"


class HOPSActor(ABC):
    r"""An abstract base class for integrating the HOPS equations.

    The methods that have to be overwritten are providing the
    :math:`K,B,C` matrices, as well as the stochastic processes
    shifts if necessary.  The right hand side of the hops equation is
    implemented in :any:`generic_rhs` that can be called in the
    implementation of :any:`rhs`.

    ..  math::

        \dot{Ψ} = OΨ + [K(t) Ψ^⊺]^⊺
             + ∑_{n=1}^{N_b}[B^{(n)}(t) (M^{(n)}_\mathrm{up} Ψ)^⊺]^⊺
             + ∑_{n=1}^{N_b}[C^{(n)}(t) (M^{(n)}_\mathrm{down} Ψ)^⊺]^⊺

    The arguments passed to the class initializer concern the general
    HOPS structure and should not be changed by subclasses.  Rather
    the whole configuration for each implementation should be
    contained in :any:`hops.core.hierarchy_parameters.HIParams`.

    ..  note::

        The ``__init__`` method is not supposed to be overwritten.
        Use ``__post_init__`` instead.

    The ``__init__`` method of this class pre-computes some
    often-used quantities and saves them as instance variables
    without underscores.  Instance variables beginning with an
    underscore are not supposed to be used in subclasses.  Subclasses
    may pre-compute their own helper quantities in their
    ``__post_init__`` method.

    The implementations of this class can be (but don't have to be)
    to be used as a ``ray`` actor (see :any:`ray.init`).  The actors
    are instantiated and controlled by :any:`HOPSSupervisor`.

    :param t_points: The time points at which the solutions is to be
        evaluated.

    :param params: The central HOPS configuration.
    :param result_filter: A function that takes the total integration
        result and filters it, so that no superflous data is returned.

        See :any:`hops.core.hierarchy_data.ResultFilter`.

    :param stream_result_filter: A function that takes the total integration
        result and filters it, so that no superflous data is returned for streaming.
        If :any:`None`, no streaming data is will be returned.

        See :any:`hops.core.hierarchy_data.ResultFilter`.
    """

    def __init__(
        self,
        t_points: np.ndarray,
        params: hip.HIParams,
        result_filter: hid.ResultFilter,
        stream_result_filter: Optional[hid.ResultFilter] = None,
    ):
        self._t_points = t_points
        """The time axis."""

        #######################################################################
        #                          Aliases/Shortcuts                          #
        #######################################################################

        self.params = params
        """The central HOPS configuration."""

        self.result_filter = result_filter
        """A function that takes the total integration result and
        filters it, so that no superfluous data is returned.
        """

        self.stream_result_filter = stream_result_filter
        """A function that takes the total integration result and
        filters it, so that no superfluous data is returned for streaming.
        """

        self.number_bcf_terms = self.params.SysP.number_bcf_terms
        """The number of BCF terms per bath."""

        self.dim_sys = self.params.SysP.dim_sys
        """The system dimension."""

        self.eta = self.params.Eta
        """The driving stochastic processes."""

        self.num_baths = self.params.SysP.num_baths
        """The number of baths."""

        self.total_number_bcf_terms = self.params.SysP.total_number_bcf_terms
        """The total number of BCF terms."""

        self.L = DynamicMatrixList(self.params.SysP.L)
        r"""The coupling operators."""

        #######################################################################
        #                          Derived Quantities                         #
        #######################################################################

        self.indices: IdxDict = params.indices
        """The hierachy indices that are being computed."""

        self._o_vec = get_O_vector(w=self.params.SysP.w, indices=self.indices)
        """The :math:`O` vector."""

        self._m_ups = [
            get_M_up(
                i,
                g,
                self.indices,
            )
            for (i, g) in zip(
                range(self.params.SysP.num_baths),
                self.params.SysP.g,
            )
        ]
        r"""The :math:`M^{(n)}_{\mathrm{up}}` matrices."""

        self._m_downs = [
            get_M_down(i, g, self.indices)
            for (i, g) in zip(
                range(self.params.SysP.num_baths),
                self.params.SysP.g,
            )
        ]
        r"""The :math:`M^{(n)}_{\mathrm{down}}` matrices."""

        self.num_hier = len(self.indices)
        """The number of hierarchy states."""

        log.info(f"Using {self.num_hier} hierarchy states.")

        self.thermal_hamiltonians = [
            ThermNoiseHamiltonian(stoc_proc=eta, L=coupling)
            for (eta, coupling) in zip(self.params.EtaTherm, self.L.list)
            if eta is not None
        ]
        """The thermal hamiltonians (if any)."""

        self.H_sys: DynamicMatrix = self.params.SysP.H_sys + sum(
            self.thermal_hamiltonians
        )

        """The system hamiltonian with the thermal noise included (if necessary).

        It will be updated for each trajectory.
        """

        self.L_dagger: DynamicMatrixList = DynamicMatrixList(
            [op.dag for op in self.L.list]
        )
        """The adjoint of the coupling operators."""

        self.minus_L_dagger: DynamicMatrixList = DynamicMatrixList(
            [-1 * op.dag for op in self.L.list]
        )
        """The adjoint of the coupling operators times minus one."""

        self.__post_init__()

    def initial_state(self) -> np.ndarray:
        """
        :returns: The initial state of the flattened HOPS vector.
        """

        res = np.zeros(self.num_hier * self.dim_sys, dtype=np.complex128)
        res[: self.dim_sys] = self.params.SysP.psi0
        return res

    def __post_init__(self):
        """
        A method that is called after :any:`__init__` supposed to be
        overwritten by implementations.
        """
        pass

    def update_random_numbers(self, sub_seed: int):
        """
        Update the random numbers for the stochastic process
        generation with seed ``sub_seed``.
        """
        np.random.seed(sub_seed)
        np.random.rand(self.params.HiP.rand_skip)

        i = 0
        for proc in self.params.EtaTherm:
            if not proc:
                continue

            num = proc.get_num_y() * 2  # type: ignore
            stoc_temp_z = uni_to_gauss(np.random.rand(num))  # type: ignore
            self.thermal_hamiltonians[i].new_process(stoc_temp_z)  # type: ignore
            i += 1

        # import ipdb

        # ipdb.set_trace()
        for proc in self.params.Eta:
            proc.new_process(uni_to_gauss(np.random.rand(proc.get_num_y() * 2)))  # type: ignore

    @abstractmethod
    def K(
        self,
        t: float,
        psi: np.ndarray,
        eta_stoc: np.ndarray,
        eta_det: Optional[np.ndarray],
    ) -> np.ndarray:
        """The :math:`K` matrix.  A function takes (among other
        things) the stochastic process and returns a matrix that acts
        on the system dimensions of :math:`Ψ`.  See the module
        docstring.

        :param t: the time
        :param psi: the integration vector in the shape ``(number of
            indices, system dimension)``
        :param eta_stoc: the current value of the driving stochastic
            processes (the complex conjugate to be precise)
        :param eta_det: the stochastic process shifts as one complex
            number per bath
        """

        pass

    @abstractmethod
    def Bs(
        self,
        t: float,
        psi: np.ndarray,
    ) -> np.ndarray:
        r"""The :math:`B^{(n)}` matrices.  A function that returns a
        matrix that acts on the system dimensions of :math:`Ψ` in
        conjunction with :math:`M^{(n)}_{\mathrm{up}}`.  See the
        module docstring.

        :param t: the time
        :param psi: the integration vector in the shape ``(number of
            indices, system dimension)``
        """

        pass

    @abstractmethod
    def Cs(
        self,
        t: float,
        psi: np.ndarray,
    ) -> np.ndarray:
        r"""The :math:`C^{(n)}` matrices.  A function that returns a
        matrix that acts on the system dimensions of :math:`Ψ` in
        conjunction with :math:`M^{(n)}_{\mathrm{down}}`.  See the
        module docstring.

        :param t: the time
        :param psi: the integration vector in the shape ``(number of
            indices, system dimension)``
        """

        pass

    def generic_rhs(
        self,
        t: float,
        V_psi: np.ndarray,
        eta_det: Optional[np.ndarray],
    ):
        r"""The concrete implementation of the HOPS differential
        equation right hand side.

        ..  math::

            \dot{Ψ} = OΨ + [K(t) Ψ^⊺]^⊺
               + ∑_{n=1}^{N_b}[B^{(n)}(t) (M^{(n)}_\mathrm{up} Ψ)^⊺]^⊺
               Q+ ∑_{n=1}^{N_b}[C^{(n)}(t) (M^{(n)}_\mathrm{down} Ψ)^⊺]^⊺

        :param t: Time point.
        :param V_psi: The flattened hops vector.
        :param eta_det: The shift part of the stochastic processses.

            There is one shift term per term in the BCF expansion.
            Added together they form the shift term of the respective
            stochastic process.
        """

        psi = V_psi.reshape((self.num_hier, self.dim_sys))

        # this is the result vector
        ddt_psi = np.empty_like(V_psi)

        # create a view on the result vector ...
        ddt_psi_view = ddt_psi.view()

        # ... lets us change the shape while operating on the
        # same block of memory
        ddt_psi_view.shape = (self.num_hier, self.dim_sys)

        eta_stoc = np.fromiter(
            (np.conj(proc(t)) for proc in self.params.Eta),
            dtype=np.complex128,
            count=self.num_baths,
        )

        # filling the data here ...
        K_mat = self.K(t, psi, eta_stoc, eta_det)

        ddt_psi_view[:, :] = (
            self._o_vec * psi
            + psi.dot(K_mat.T)
            + sum(
                M_up.dot(psi).dot(B.T) for M_up, B in zip(self._m_ups, self.Bs(t, psi))
            )
            + sum(
                M_down.dot(psi).dot(C.T)
                for M_down, C in zip(self._m_downs, self.Cs(t, psi))
            )
        )
        # ... leads to the correct linear alignment for the result vector

        return ddt_psi

    @abstractmethod
    def rhs(
        self,
        t: float,
        V_psi: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        The righthand side of the HOPS equations.  The pure hierarchy
        equations are implemented in :any:`generic_rhs` which may be
        called in this method.
        """

        pass

    def call_solver(self):
        """Call the solver and return the trajectory.

        Can be overwritten to customize the solver.
        """

        solver_return = scipy.integrate.solve_ivp(
            self.rhs,
            (0, self._t_points[-1]),
            self.initial_state(),
            t_eval=self._t_points,
            **self.params.IntP.solver_args,
        )

        if solver_return.status < 0:
            raise RuntimeError(solver_return.message)

        return solver_return.y.T

    @ray.method(num_returns=1)
    def integrate(
        self, seed: int, id: int
    ) -> tuple[
        int,
        int,
        tuple[bool, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
        Optional[tuple[bool, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]],
    ]:
        """Integrate the trajectory number ``id`` with the seed
        ``seed``.

        :returns: The ``seed``, ``id`` and integration result that is
                  the return value of
                  :any:`hops.core.hierarchy_data.ResultFilter` and
                  optionally a filtered result for streaming.
        """

        log.debug(f"Integrating trajectory {id} with seed {seed}.")
        self.update_random_numbers(seed)

        solver_result = self.call_solver()
        result = self.result_filter(solver_result)
        streaming_result = (
            self.stream_result_filter(solver_result)
            if self.stream_result_filter is not None
            else None
        )

        return (seed, id, result, streaming_result)

    ###########################################################################
    #                                Utilities                                #
    ###########################################################################

    def psi_zero(self, V_psi: np.ndarray):
        """
        Extracts and returns the zeroth hops state a current HOPS
        integration vector ``V_psi``.
        """

        return V_psi[: self.dim_sys]

    def coupling_expectation(self, t: float, psi_zero: np.ndarray) -> np.ndarray:
        r"""
        Calculate the expectation values of :math:`L^{†,(n)}` in the
        current HOPS state.

        ..  math::

            \langle L^{†,(n)}\rangle =
            \frac{\langle ψ|L^{†,(n)}|ψ\rangle}{\langle ψ|ψ\rangle}

        :param t: The time.
        :param psi_zero: The zeroth order HOPS state.
        """

        psi_conj = psi_zero.conj()
        return np.dot(self.L_dagger(t).dot(psi_zero), psi_conj) / np.dot(
            psi_conj, psi_zero
        )


class LinearHOPSActor(HOPSActor):
    """
    The implementation for the linear HOPS equations.
    """

    def K(
        self,
        t: float,
        psi: np.ndarray,
        eta_stoc: np.ndarray,
        eta_det: Optional[np.ndarray],
    ) -> np.ndarray:
        r"""The :math:`K` matrix for the linear case.

        ..  math::

            -\mathrm{i} H_\mathrm{sys} + ∑_{n=1}^{N_b} η^{\ast,(n)}_t L^{(n)}

        For the arguments see
        :any:`hops.core.integration.HOPSActor.K`.
        """

        del psi, eta_det

        minus_i_H_t = -1j * (self.H_sys(t))
        return minus_i_H_t + sum(eta * L(t) for (eta, L) in zip(eta_stoc, self.L.list))

    def Bs(
        self,
        t: float,
        psi: np.ndarray,
    ) -> np.ndarray:
        """
        The :math:`B^{(n)}` matrices in the linear case.  Has
        dimensions ``(number of baths, system dimension, system
        dimension)``.

        ..  math::

            -L^{†,(n)}

        For the arguments see
        :any:`hops.core.integration.HOPSActor.Bs`.
        """

        del psi

        return self.minus_L_dagger(t)

    def Cs(
        self,
        t: float,
        psi: np.ndarray,
    ) -> np.ndarray:
        """
        The :math:`C^{(n)}` matrices in the linear case.  Has
        dimensions ``(number of baths, system dimension, system
        dimension)``.

        ..  math::

            L^{(n)}

        For the arguments see
        :any:`hops.core.integration.HOPSActor.Cs`.
        """

        del psi

        return self.L(t)

    def rhs(
        self,
        t: float,
        V_psi: np.ndarray,
    ):
        """
        The RHS for the linear HOPS method.

        See :any:`hops.core.integration.HOPSActor.rhs`.
        """

        return self.generic_rhs(t, V_psi, None)


class NonLinearHOPSActor(HOPSActor):
    """
    The implementation for the non-linear HOPS equations.
    """

    def __post_init__(self):
        self.g_ast = [
            np.conj(g_el) for g_el in self.params.SysP.g
        ]  # this is not scaled, occurs in eta_det_non_lin
        r"""
        The complex conjugate of the factors in the BCF expansion. :math:`G_i^{(n),\ast}`
        """

        cumulative_bcf_terms = np.cumsum(np.array(self.number_bcf_terms))
        self.ranges: np.ndarray = np.array(
            list(
                zip(
                    [0, *cumulative_bcf_terms],
                    [*cumulative_bcf_terms, sum(self.number_bcf_terms)],
                )
            )
        )[:-1]
        """
        The index ranges for the individual :math:`η^{(n)}_λ` that
        have been merged together into a single array.
        """

        self.omega_ast = [np.conj(w_el) for w_el in self.params.SysP.w]
        """The conjugate of the BCF expansion exponents :math:`W^{(n)}_i`."""

        self.omega_ast_conc = np.concatenate(self.omega_ast)
        """All elements of :any:`omega_ast` concatenated."""

        self.auto_normalize = self.params.HiP.auto_normalize
        """
        Whether to add a term to the RHS that assures normalization of
        the zeroth hierarchy hops vector.
        """

    def initial_state(self) -> np.ndarray:
        """
        This overwrites the
        :any:`hops.core.integration.HOPSActor.initial_state` to
        include the initial state of the stochastic process shifts.
        """
        return np.concatenate(
            (
                super().initial_state(),
                np.zeros(
                    shape=(self.total_number_bcf_terms,),
                    dtype=np.complex128,
                ),
            ),
        )

    def K(
        self,
        t: float,
        psi: np.ndarray,
        eta_stoc: np.ndarray,
        eta_det: Optional[np.ndarray],
    ) -> np.ndarray:
        r"""The :math:`K` matrix for the nonlinear case.

        ..  math::

            -\mathrm{i} H_\mathrm{sys}
            + ∑_{n=1}^{N_b} \tilde{η}^{\ast,(n)}_t L^{(n)},

        where :math:`\tilde{η}^{(n)}=η^{(n)} + ∑_{μ=1}^{N^{(n)}}
        η^{(n)}_{\mathrm{shift},μ}`.

        The shifts :math:`∑_{μ=1}^N η^{(n)}_{\mathrm{shift},μ}` can
        be integrated in closed form along with the HOPS vector.

        For the arguments see
        :any:`hops.core.integration.HOPSActor.K`.
        """

        del psi
        assert eta_det is not None

        minus_i_H_t = -1j * self.H_sys(t)

        return minus_i_H_t + ((eta_stoc + eta_det)[:, None, None] * self.L(t)).sum(
            axis=0
        )

    def Bs(
        self,
        t: float,
        psi: np.ndarray,
    ) -> np.ndarray:
        r"""
        The :math:`B^{(n)}` matrices in the nonlinear case.

        ..  math::

            -(L^{†,(n)} - \langle L^{†,(n)}\rangle),

        where :math:`\langle L^{†,(n)}\rangle = \frac{\langle
        ψ|L^{†,(n)}|ψ\rangle}{\langle ψ|ψ\rangle}`.

        For the arguments see
        :any:`hops.core.integration.HOPSActor.Bs`.
        """

        psi_zero = psi[0, :]
        exp_vals = np.zeros(self.minus_L_dagger.shape, dtype=np.complex128)
        for i, exp_val in enumerate(self.coupling_expectation(t, psi_zero)):
            np.fill_diagonal(exp_vals[i], exp_val)

        return self.minus_L_dagger(t) + exp_vals

    def Cs(
        self,
        t: float,
        psi: np.ndarray,
    ) -> np.ndarray:
        """
        The :math:`C^{(n)}` matrices in the nonlinear case.  Has
        dimensions ``(number of baths, system dimension, system
        dimension)``.

        ..  math::

            L^{(n)}

        For the arguments see
        :any:`hops.core.integration.HOPSActor.Cs`.
        """

        del psi

        return self.L(t)

    def ddt_eta_lambda(
        self, t: float, psi_zero: np.ndarray, eta_lambda: np.ndarray
    ) -> np.ndarray:
        r"""
        The RHS of the differential equations for the stochastic process
        shifts :math:`\frac{η^{\ast,(n)}_λ}{G_λ^{\ast,(n)}}`.

        :param t: The time.
        :param psi_zero: The zeroth hierarchy state.
        :param eta_lambda: :math:`\frac{η^{\ast,(n)}_λ}{G_λ^{\ast,(n)}}`,
            the shifts that enter the HOPS diffrential equations
        """

        l_exp_vals: np.ndarray = np.empty(len(eta_lambda), dtype=np.complex128)

        np.concatenate(
            [
                exp_val * np.ones(num, dtype=np.complex128)
                for exp_val, num in zip(
                    self.coupling_expectation(t, psi_zero), self.number_bcf_terms
                )
            ],
            out=l_exp_vals,
        )

        return np.array(l_exp_vals - self.omega_ast_conc * eta_lambda)

    def eta_det(self, eta_lambda: np.ndarray) -> np.ndarray:
        r"""Calculate the stochastic process shift with correct scaling
        for each bath.

        :param eta_lambda:
            :math:`\frac{η^{\ast,(n)}_λ}{G_λ^{\ast,(n)}}`, the shifts
            that enter the HOPS diffrential equations
        """

        return np.fromiter(
            (
                (g_ast * eta_lambda[range[0] : range[1]]).sum()
                for g_ast, range in zip(self.g_ast, self.ranges)
            ),
            dtype=np.complex128,
            count=self.num_baths,
        )

    def rhs(
        self,
        t: float,
        V_psi_eta_lambda: np.ndarray,
    ):
        r"""
        The RHS for the nonlinear HOPS method.

        The ``V_psi_and_eta_lambda`` argument now includes the
        stochastic process shifts as the last :math:`N` elements,
        where :math:`N` is the numeber of terms in the BCF expansion.

        Note that actually the shifts divided by their respective BCF
        coefficients :math:`\frac{η^{\ast,(n)}_λ}{G_λ^{\ast,(n)}}` are
        stored in ``V_psi_and_eta_lambda``.  The correct shift can be
        retrieved through :any:`eta_det`.

        See :any:`hops.core.integration.HOPSActor.rhs`.
        """

        # splitting the HOPS states and the stochastic process shifts
        V_psi = V_psi_eta_lambda[: -self.total_number_bcf_terms]
        eta_lambda = V_psi_eta_lambda[-self.total_number_bcf_terms :]

        eta_det = self.eta_det(eta_lambda)
        psi_zero = self.psi_zero(V_psi)

        ddt_eta_lambda = self.ddt_eta_lambda(t, psi_zero, eta_lambda)
        ddt_psi = self.generic_rhs(
            t,
            V_psi,
            eta_det,
        )

        # include normalization by subtracting the amount of change in the
        # direction of the psi_sys-vector: ddt_psi -> ddt_psi - dn * V_psi
        dn = 0
        if self.auto_normalize:
            sys_norm_squared = np.vdot(psi_zero, psi_zero)
            dn = np.vdot(psi_zero, ddt_psi[: self.dim_sys]).real / sys_norm_squared - (
                1
                - sys_norm_squared  # this 2nd term ensures that norm=1 is a *stable* fixpoint
            )

        # and putting them back togeter
        res = np.empty_like(V_psi_eta_lambda)
        res[: -self.total_number_bcf_terms] = ddt_psi - dn * V_psi
        res[-self.total_number_bcf_terms :] = ddt_eta_lambda

        return res


class HOPSSupervisor:
    r"""A wrapper to set up the integration of the HOPS equations and
    launch ray workers for the integration.  Local integration is
    supported as well.

    The wrapper servers as dispatcher by choosing the correct actor
    (see :any:`HOPSActor`) based on the HOPS configuration
    ``hi_key``.

    :param hi_key: The HOPS configuration.
    :param data_name: The name of the database in which the results
        will be stored.
    :param number_of_samples: The target number of samples that shall
        be computed.
    :param min_sample_index: The smallest sample index to begin with.

    :param data_path: The path under which the database is to reside.
        See :any:`HIMetaData`.

    :param hide_progress: Whether to hide the progress bar.

    :param stream_file: The file path where the streamed results will
        be written to if ``stream_result_type`` is not :any:`None`.
    """

    def __init__(
        self,
        hi_key: hip.HIParams,
        number_of_samples: int,
        data_name: str = "data",
        data_path: str = ".",
        min_sample_index: int = 0,
        hide_progress: bool = False,
        stream_file: Optional[str] = None,
    ):
        self.metadata: hid.HIMetaData = hid.HIMetaData(
            hid_name=data_name, hid_path=data_path
        )
        """A wrapper to access the integration result database.

        See :any:`get_data`.
        """

        self.stream_file = stream_file

        self.min_sample_index: int = min_sample_index
        """The smallest index to compute."""

        self.number_of_samples: int = number_of_samples
        """The total number of samples to compute."""

        self.params = hi_key
        """The central HOPS configuration."""

        self._hide_progress = hide_progress
        """Whether to hide the progress bar."""

        self._normed_average: bool = self.params.HiP.nonlinear
        """Whether the samples must be normalized before averaging."""

        self.actor: Type[HOPSActor]
        """The actor class that implements the actual integration."""

        if not self.params.HiP.nonlinear:
            log.info("Choosing the linear integrator.")
            self.actor = LinearHOPSActor

        else:
            log.info("Choosing the nonlinear integrator.")
            self.actor = NonLinearHOPSActor

    def __repr__(self):
        return f"HI({self.params}, {self.number_of_samples}, {self.min_sample_index})"

    @property
    def seeds(self) -> np.ndarray:
        """
        A list of seeds for all trajectories.
        """

        np.random.seed(self.params.HiP.seed)
        np.random.rand(self.params.HiP.rand_skip)
        return (np.random.rand(self.number_of_samples) * 2**32).astype(np.int64)

    def get_job_args(self, data: hid.HIData) -> list[tuple[int, int]]:
        """
        :returns: A list of argument tuples for
                  :any:`hops.core.integration.HOPSActor.integrate`
                  that corresponds to outstanding jobs.
        """

        seeds = self.seeds
        args = [
            (int(seeds[index]), index)
            for index in range(
                self.min_sample_index,
                self.min_sample_index + self.number_of_samples,
            )
            if not data.has_sample(idx=index)
        ]

        log.info(f"Some {len(args)} trajectories have to be integrated.")
        log.debug(f"Trajectories to be integrated: %s", args)
        return args

    def integrate_single_process(self, clear_pd: bool = False):
        """Integrate the HOPS equations locally with a single process and
        store the results using :any:`HIData`.

        :param clear_pd: If set to :any:`True`, the result database will be cleared
                         prior to the integration.
        """

        with self.get_data_and_maybe_clear(clear_pd) as data:
            t = data.get_time()
            indices = self.get_job_args(data)

            integrator = self.actor(
                t, self.params, data.result_filter, data.stream_result_filter
            )

            for seed, index in tqdm(
                indices,
                disable=self._hide_progress,
                smoothing=0,
                dynamic_ncols=True,
                mininterval=1,
            ):
                (
                    _,
                    _,
                    (incomplete, psi0, aux_states, stoc_proc),
                    stream_data,
                ) = integrator.integrate(seed, index)

                if stream_data:
                    (
                        stream_incomplete,
                        stream_psi0,
                        stream_aux_states,
                        stream_stoc_proc,
                    ) = stream_data
                    assert data.stream_result_type is not None

                    data.stream_samples(
                        idx=index,
                        incomplete=stream_incomplete,
                        psi0=stream_psi0,
                        aux_states=stream_aux_states,
                        stoc_proc=stream_stoc_proc,
                        result_type=data.stream_result_type,
                        rng_seed=seed,
                    )

                data.new_samples(
                    idx=index,
                    incomplete=incomplete,
                    psi0=psi0,
                    aux_states=aux_states,
                    stoc_proc=stoc_proc,
                    result_type=self.params.HiP.result_type,
                    normed=self._normed_average,
                    rng_seed=seed,
                )

    @staticmethod
    @ray.remote
    def integration_task(integrator: HOPSActor, arg: tuple[int, int]):
        """
        A ray task that receives an ``integrator`` on which it calls
        the :any:`HOPSActor.integrate` method with ``arg`` and returns
        the result.
        """

        seed, id = arg
        return integrator.integrate(seed, id)

    def integrate(self, clear_pd: bool = False):
        """Integrate the HOPS equations on a ray cluster (see
        :any:`ray.init`).

        :param clear_pd: If set to :any:`True`, the result database
            will be cleared prior to the integration.
        """

        # Despite their name, we don't instantiate the actor as a
        # `ray actor`. Rather we put one instance into the ray object
        # store and then launch as many tasks as we have trajectories.
        #
        # This allows us to leverage the ray scaling and task
        # distribution.
        #
        # I reached this solution after extensive experimentation late
        # at night :P.
        #
        # -- Valentin Boettcher

        with self.get_data_and_maybe_clear(clear_pd) as data:
            t = data.get_time()

            num_integrators = int(ray.available_resources().get("CPU", 0))

            if num_integrators == 0:
                raise RuntimeError("No cpu available for integration!")

            log.info(f"Using {num_integrators} integrators.")

            indices = self.get_job_args(data)

            integrator = ray.put(
                self.actor(
                    t, self.params, data.result_filter, data.stream_result_filter
                )
            )

            integration = tqdm(
                total=len(indices),
                disable=self._hide_progress,
                smoothing=0,
                dynamic_ncols=True,
                mininterval=1,
            )

            handles = [
                self.integration_task.remote(integrator, index) for index in indices
            ]

            def signal_handler(signals):
                del signals
                nonlocal handles

                integration.close()
                data.close()
                for handle in handles:
                    ray.cancel(handle, force=True)

                handles = []

            while len(handles):
                with signal_delay.sig_delay(
                    [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1],
                    signal_handler,
                ):
                    try:
                        done_id, handles = ray.wait(
                            handles, fetch_local=True, num_returns=1, timeout=10
                        )
                        if not done_id:
                            continue

                        (
                            seed,
                            index,
                            (incomplete, psi0, aux_states, stoc_proc),
                            stream_data,
                        ) = ray.get(done_id[0])

                    except TaskCancelledError:
                        break

                    integration.update()

                    if stream_data:
                        (
                            stream_incomplete,
                            stream_psi0,
                            stream_aux_states,
                            stream_stoc_proc,
                        ) = stream_data
                        assert data.stream_result_type is not None

                        data.stream_samples(
                            idx=index,
                            incomplete=stream_incomplete,
                            psi0=stream_psi0,
                            aux_states=stream_aux_states,
                            stoc_proc=stream_stoc_proc,
                            result_type=data.stream_result_type,
                            rng_seed=seed,
                        )

                    data.new_samples(
                        idx=index,
                        incomplete=incomplete,
                        psi0=psi0,
                        aux_states=aux_states,
                        stoc_proc=stoc_proc,
                        result_type=self.params.HiP.result_type,
                        normed=self._normed_average,
                        rng_seed=seed,
                    )

    def get_data(self, read_only: bool = False, stream: bool = True) -> hid.HIData:
        """
        :returns: The database containing results that correspond to the current configuration
                  (:any:`params`).

        :param read_only: Whether to open the database in read only mode.
        """

        return self.metadata.get_HIData(
            key=self.params,
            read_only=read_only,
            robust=True,
            stream_file=self.stream_file if stream else None,
        )

    def get_data_and_maybe_clear(self, clear: bool = False) -> hid.HIData:
        """
        Like :any:`get_data` but conditionally clears the data and sets the time.

        :param clear: Whether to clear the data.
        """

        with self.get_data(stream=False) as data:
            if clear:
                log.info("Clear HIData contained in {}".format(data.h5File))
                data.clear()

            if not data.time_set:
                data.set_time(self.params.IntP.t)

        return self.get_data()

"""Tests for the :any:`hops.core.integration` module.

Todo
====

 - testing the truncation schemes
"""

from hops.core.integration import (
    HiIdx,
    IdxDict,
    HOPSSupervisor,
    LinearHOPSActor,
    NonLinearHOPSActor,
    ThermNoiseHamiltonian,
    get_M_up,
    get_M_down,
    get_O_vector,
)
from hops.core.hierarchy_data import HIData
from hops.core.hierarchy_parameters import HIParams, HiP, IntP, SysP
import stocproc
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
from hops.core.utility import uni_to_gauss
from hops.util.abstract_truncation_scheme import TruncationScheme_Simplex
from hops.util.dynamic_matrix import ConstantMatrix, DynamicMatrix, DynamicMatrixList
from .fixtures import (
    simple_hops_zero,
    data_file,
    simple_hops_nonzero,
    simple_multi_hops_zero,
    simple_multi_hops_nonzero,
)
from collections.abc import Callable
import logging
import scipy.sparse as spe
import os

stocproc.logging_setup(
    logging.WARNING, logging.WARNING, logging.WARNING, logging.WARNING
)


class TestHiIdx:
    def test_init_zero(self):
        n_list = [2, 3]
        i = HiIdx(n_list)

        assert_array_equal(i[0], np.array([0, 0]))
        assert_array_equal(i[1], np.array([0, 0, 0]))

        assert i.to_bin() == b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

    def test_bin(self):
        n_list = [1, 1, 1]
        i = HiIdx(n_list)
        i[0][0] = 2
        i[1][0] = 3
        i[2][0] = 4

        i[0, 0] = 2
        i[1, 0] = 3
        i[2, 0] = 4

        assert (
            i.to_bin()
            == np.array([2], dtype=i._dtype).tobytes()
            + np.array([3], dtype=i._dtype).tobytes()
            + np.array([4], dtype=i._dtype).tobytes()
        )

    def test_equality(self):
        n_list = [2, 3, 100]
        i = HiIdx(n_list)
        j = HiIdx(n_list)

        assert i == j

        i[0, 0] = 10
        assert i != j

        j[0, 0] = 10
        assert i == j

        with pytest.raises(ValueError):
            i == 1

    def test_init_bytes(self):
        n_list = [2, 3]
        i = HiIdx(n_list)

        i[0, 0] = 10

        j = HiIdx(n_list, i.to_bin())

        assert i == j

    def test_init_list(self):
        i = HiIdx.from_list([[1], [2, 3]])

        assert i._n_list == [1, 2]
        assert i[0, 0] == 1
        assert i[1, 0] == 2
        assert i[1, 1] == 3

    def test_init_copy(self):
        n_list = [2, 3]
        i = HiIdx(n_list)

        i[0, 0] = 10

        j = HiIdx.from_other(i)
        assert i == j

        j[0, 0] = 11

        assert i != j
        assert i[0, 0] != j[0, 0]
        assert i[0, 0] == 10

    def test_from_repr(self):
        n_list = [2, 3]
        i = HiIdx(n_list)

        i[0, 0] = 10

        assert i == eval(i.__repr__())

    def test_use_as_hash(self):
        n_list = [2, 3]
        i = HiIdx(n_list)

        i[0, 0] = 10

        j = HiIdx.from_other(i)
        i[1, 0] = 10

        assert hash(i) == hash(i.to_bin())
        d = {i: 1, j.to_bin(): 2}

        assert len(d) == 2
        assert d[i] == 1
        assert d[i.to_bin()] == 1  # type: ignore
        assert d[j] == 2


class TestIdxDict:
    n_list = [1, 2, 3]

    def test_init(self):
        d = IdxDict(self.n_list)

        assert d.idx_dict == {}
        assert d.N == 3

        d = IdxDict(3)

        assert d.idx_dict == {}
        assert d.N == 1
        assert d.n_list == [3]

    def test_add_idx(self):
        d = IdxDict(self.n_list)

        i = HiIdx(self.n_list)
        i[0, 0] = 10

        assert d.add_new_idx(i)
        assert not d.add_new_idx(i)

        assert d[i] is not None
        assert d[i] == d[i.to_bin()]
        assert len(d) == 1

        i[1, 0] = 10
        d += i

        assert d[i] is not None

        assert i in d
        assert i.to_bin() in d

        assert d[i] == d[i.to_bin()]
        assert len(d) == 2

    def test_simplex(self):
        d = IdxDict(self.n_list)
        d.make(TruncationScheme_Simplex(1))

        assert len(d) == sum(self.n_list) + 1
        assert d[HiIdx(self.n_list)] is not None

        for i, size in enumerate(self.n_list):
            for j in range(size):
                index = HiIdx(self.n_list)
                index[i, j] = 1

                assert index in d

    def test_bin_to_idx(self):
        d = IdxDict(self.n_list)

        i = HiIdx(self.n_list)
        i[0, 0] = 10

        assert d.bin_to_idx(i.to_bin()) == i

    def test_iter(self):
        d = IdxDict(self.n_list)
        d.make(TruncationScheme_Simplex(2))

        for i in d:
            assert isinstance(i, HiIdx)
            assert i in d

        for i in d.binkeys():
            assert isinstance(i, bytes)
            assert i in d

        for i, scalar in d.items():
            assert isinstance(i, HiIdx)
            assert d[i] == scalar

        for i, scalar in d.binitems():
            assert isinstance(i, bytes)
            assert d[i] == scalar

    def test_first_hierarchy_indices(self):
        d = IdxDict([2, 2])
        d.make(TruncationScheme_Simplex(3))

        indices = d.get_first_hierarchy_indices()

        assert indices[0][0] == d[HiIdx.from_list([[1, 0], [0, 0]])]
        assert indices[0][1] == d[HiIdx.from_list([[0, 1], [0, 0]])]
        assert indices[1][0] == d[HiIdx.from_list([[0, 0], [1, 0]])]
        assert indices[1][1] == d[HiIdx.from_list([[0, 0], [0, 1]])]


@pytest.mark.filterwarnings("ignore:Consistency check")
class TestHighLevel:
    """Run a full integration for some parameters do detect major regresssions."""

    def compare_with_stored(self, hops_config: HIParams, tmp_path: str, data_file):
        """A helper that takes a hops config, runs an integration
        and compares it against a stored result."""

        params = hops_config
        hierarchy = HOPSSupervisor(params, 5, "test", tmp_path)

        hierarchy.integrate_single_process()
        with (
            hierarchy.get_data() as data,
            HIData(
                data_file("data.h5", data.hdf5_name),
                True,
                params,
                check_consistency=False,
                overwrite_key=True,
            ) as ref_data,
        ):
            assert data.samples == 5
            assert data.largest_idx == 4

            assert data.samples == ref_data.samples
            assert data.largest_idx == ref_data.largest_idx

            assert_array_almost_equal(data.rho_t_accum.mean, ref_data.rho_t_accum.mean)
            assert_array_almost_equal(
                data.rho_t_accum.sample_variance, ref_data.rho_t_accum.sample_variance
            )

    def test_zero_nonlin(
        self,
        simple_hops_zero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        self.compare_with_stored(simple_hops_zero(), tmp_path, data_file)

    def test_zero_lin(
        self,
        simple_hops_zero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        self.compare_with_stored(
            simple_hops_zero(nonlinear=False),
            tmp_path,
            data_file,
        )

    def test_nonzero_lin(
        self,
        simple_hops_nonzero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        self.compare_with_stored(
            simple_hops_nonzero(nonlinear=False),
            tmp_path,
            data_file,
        )

    def test_nonzero_nonlin(
        self,
        simple_hops_nonzero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        params = simple_hops_nonzero()
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

        params.HiP.auto_normalize = False
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

    def test_multi_zero_lin(
        self,
        simple_multi_hops_zero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        self.compare_with_stored(
            simple_multi_hops_zero(nonlinear=False),
            tmp_path,
            data_file,
        )

    def test_multi_zero_nonlin(
        self,
        simple_multi_hops_zero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        params = simple_multi_hops_zero()
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

        params.HiP.auto_normalize = False
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

    def test_multi_nonzero_lin(
        self,
        simple_multi_hops_nonzero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        self.compare_with_stored(
            simple_multi_hops_nonzero(nonlinear=False),
            tmp_path,
            data_file,
        )

    def test_multi_nonzero_nonlin(
        self,
        simple_multi_hops_nonzero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        params = simple_multi_hops_nonzero()
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

        params.HiP.auto_normalize = False
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

    def test_multi_nonzero_lin_one_dyn(
        self,
        simple_multi_hops_nonzero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        params = simple_multi_hops_nonzero(nonlinear=False, time_dep=(0,))
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

        params.HiP.auto_normalize = False
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

    def test_multi_nonzero_lin_two_dyn(
        self,
        simple_multi_hops_nonzero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        params = simple_multi_hops_nonzero(nonlinear=False, time_dep=(0, 1))
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

        params.HiP.auto_normalize = False
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

    def test_m_nonz_nl_one_dynamic(
        self,
        simple_multi_hops_nonzero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        params = simple_multi_hops_nonzero(nonlinear=True, time_dep=(0,))
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )

    def test_m_nonz_nl_two_dynamic(
        self,
        simple_multi_hops_nonzero: Callable[
            ...,
            HIParams,
        ],
        tmp_path: str,
        data_file: str,
    ):
        params = simple_multi_hops_nonzero(nonlinear=True, time_dep=(0, 1))
        self.compare_with_stored(
            params,
            tmp_path,
            data_file,
        )


def assert_sparse_equal(sparse: spe.spmatrix, dictionary: dict[tuple[int, int], float]):
    """
    Asserts that ``sparse`` is equal to the ``dictionary``
    representation of a sparse matrix.
    """
    assert dict(sparse.todok()) == dictionary


class TestHelpers:
    g = [np.array([1, 2, 3]), np.array([4, 5, 6, 14])]
    w = [np.array([7, 8, 9]), np.array([10, 11, 12, 13])]
    bath_sizes = [len(gg) for gg in g]
    indices = IdxDict(bath_sizes)
    indices.make(TruncationScheme_Simplex(2))

    def test_M_up(self):
        M = get_M_up(0, self.g[0], self.indices)

        M_mock = {
            (0, 1): 1j,
            (1, 2): 1.4142135623730951j,
            (1, 3): 1.4142135623730951j,
            (9, 3): 1j,
            (1, 4): 1.7320508075688772j,
            (16, 4): 1j,
            (22, 5): 1j,
            (27, 6): 1j,
            (31, 7): 1j,
            (34, 8): 1j,
            (0, 9): 1.4142135623730951j,
            (9, 10): 2j,
            (9, 11): 1.7320508075688772j,
            (16, 11): 1.4142135623730951j,
            (22, 12): 1.4142135623730951j,
            (27, 13): 1.4142135623730951j,
            (31, 14): 1.4142135623730951j,
            (34, 15): 1.4142135623730951j,
            (0, 16): 1.7320508075688772j,
            (16, 17): 2.449489742783178j,
            (22, 18): 1.7320508075688772j,
            (27, 19): 1.7320508075688772j,
            (31, 20): 1.7320508075688772j,
            (34, 21): 1.7320508075688772j,
        }

        assert_sparse_equal(M, M_mock)

        # M = get_M_up(0, self.g[0], self.indices, self.g[0])
        # assert_sparse_equal(M, {key: 1 for (key, _) in M_mock.items()})

        M = get_M_up(1, self.g[1], self.indices)

        M_mock = {
            (1, 5): 2j,
            (1, 6): 2.23606797749979j,
            (1, 7): 2.449489742783178j,
            (1, 8): 3.7416573867739413j,
            (9, 12): 2j,
            (9, 13): 2.23606797749979j,
            (9, 14): 2.449489742783178j,
            (9, 15): 3.7416573867739413j,
            (16, 18): 2j,
            (16, 19): 2.23606797749979j,
            (16, 20): 2.449489742783178j,
            (16, 21): 3.7416573867739413j,
            (0, 22): 2j,
            (22, 23): 2.8284271247461903j,
            (22, 24): 2.23606797749979j,
            (27, 24): 2j,
            (22, 25): 2.449489742783178j,
            (31, 25): 2j,
            (22, 26): 3.7416573867739413j,
            (34, 26): 2j,
            (0, 27): 2.23606797749979j,
            (27, 28): 3.1622776601683795j,
            (27, 29): 2.449489742783178j,
            (31, 29): 2.23606797749979j,
            (27, 30): 3.7416573867739413j,
            (34, 30): 2.23606797749979j,
            (0, 31): 2.449489742783178j,
            (31, 32): 3.4641016151377544j,
            (31, 33): 3.7416573867739413j,
            (34, 33): 2.449489742783178j,
            (0, 34): 3.7416573867739413j,
            (34, 35): 5.291502622129181j,
        }

        assert_sparse_equal(M, M_mock)

    def test_M_down(self):
        M = get_M_down(0, self.g[0], self.indices)
        assert_sparse_equal(
            M,
            {
                (1, 0): -1j,
                (9, 0): -1.4142135623730951j,
                (16, 0): -1.7320508075688772j,
                (2, 1): -1.4142135623730951j,
                (3, 1): -1.4142135623730951j,
                (4, 1): -1.7320508075688772j,
                (3, 9): -1j,
                (10, 9): -2j,
                (11, 9): -1.7320508075688772j,
                (4, 16): -1j,
                (11, 16): -1.4142135623730951j,
                (17, 16): -2.449489742783178j,
                (5, 22): -1j,
                (12, 22): -1.4142135623730951j,
                (18, 22): -1.7320508075688772j,
                (6, 27): -1j,
                (13, 27): -1.4142135623730951j,
                (19, 27): -1.7320508075688772j,
                (7, 31): -1j,
                (14, 31): -1.4142135623730951j,
                (20, 31): -1.7320508075688772j,
                (8, 34): -1j,
                (15, 34): -1.4142135623730951j,
                (21, 34): -1.7320508075688772j,
            },
        )

        M = get_M_down(1, self.g[1], self.indices)

        assert_sparse_equal(
            M,
            {
                (22, 0): -2j,
                (27, 0): -2.23606797749979j,
                (31, 0): -2.449489742783178j,
                (34, 0): -3.7416573867739413j,
                (5, 1): -2j,
                (6, 1): -2.23606797749979j,
                (7, 1): -2.449489742783178j,
                (8, 1): -3.7416573867739413j,
                (12, 9): -2j,
                (13, 9): -2.23606797749979j,
                (14, 9): -2.449489742783178j,
                (15, 9): -3.7416573867739413j,
                (18, 16): -2j,
                (19, 16): -2.23606797749979j,
                (20, 16): -2.449489742783178j,
                (21, 16): -3.7416573867739413j,
                (23, 22): -2.8284271247461903j,
                (24, 22): -2.23606797749979j,
                (25, 22): -2.449489742783178j,
                (26, 22): -3.7416573867739413j,
                (24, 27): -2j,
                (28, 27): -3.1622776601683795j,
                (29, 27): -2.449489742783178j,
                (30, 27): -3.7416573867739413j,
                (25, 31): -2j,
                (29, 31): -2.23606797749979j,
                (32, 31): -3.4641016151377544j,
                (33, 31): -3.7416573867739413j,
                (26, 34): -2j,
                (30, 34): -2.23606797749979j,
                (33, 34): -2.449489742783178j,
                (35, 34): -5.291502622129181j,
            },
        )

    def test_get_O_vector(self):
        vec = get_O_vector(self.w, self.indices)

        assert_array_equal(
            vec,
            np.array(
                [
                    [0.0 + 0.0j],
                    [-7.0 + 0.0j],
                    [-14.0 + 0.0j],
                    [-15.0 + 0.0j],
                    [-16.0 + 0.0j],
                    [-17.0 + 0.0j],
                    [-18.0 + 0.0j],
                    [-19.0 + 0.0j],
                    [-20.0 + 0.0j],
                    [-8.0 + 0.0j],
                    [-16.0 + 0.0j],
                    [-17.0 + 0.0j],
                    [-18.0 + 0.0j],
                    [-19.0 + 0.0j],
                    [-20.0 + 0.0j],
                    [-21.0 + 0.0j],
                    [-9.0 + 0.0j],
                    [-18.0 + 0.0j],
                    [-19.0 + 0.0j],
                    [-20.0 + 0.0j],
                    [-21.0 + 0.0j],
                    [-22.0 + 0.0j],
                    [-10.0 + 0.0j],
                    [-20.0 + 0.0j],
                    [-21.0 + 0.0j],
                    [-22.0 + 0.0j],
                    [-23.0 + 0.0j],
                    [-11.0 + 0.0j],
                    [-22.0 + 0.0j],
                    [-23.0 + 0.0j],
                    [-24.0 + 0.0j],
                    [-12.0 + 0.0j],
                    [-24.0 + 0.0j],
                    [-25.0 + 0.0j],
                    [-13.0 + 0.0j],
                    [-26.0 + 0.0j],
                ]
            ),
        )


class FakeStocProc(stocproc.StocProc):
    """A trivial deterministic stochastic process."""

    def __post_init__(self, w: float):
        self.w = w
        self.key = "FAKE", w
        self.set_zs = np.empty(0)

    def new_process(self, zs: np.ndarray):
        self.set_zs = zs

    def __call__(self, t):
        return np.exp(1j * t * self.w) + np.sum(self.set_zs)

    def __getstate__(self):
        return w

    def __setstate__(self, w):
        pass

    def get_num_y(self):
        return 2

    def calc_z(self):
        pass

    def set_scale(self, _):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.w})"


class SinusHamiltonian(DynamicMatrix):
    """A simple dynamic hamiltonian."""

    def __init__(self, w: float):
        self.w = w

    def call(self, t: NDArray[np.float64]):
        return (
            np.array([[1, 0], [0, -1]])
            + np.sin(self.w * t[:, None]) * np.array([[0, 1], [0, -1]])[None, :]
        )

    def __getstate__(self):
        return self.w

    def __repr__(self):
        return f"{self.__class__.__name__}({self.w})"


def assert_array_equal_many_times(first, second):
    for t in np.linspace(0, 100, 1000):
        assert_array_equal(second(t), first(t))


class TestDefs:
    """
    These are some basic tests for the
    :any:`hops.core.integration` module.

    Most of them compare the computations in the module against
    explicit and less efficient, but much clearer computations.
    """

    H_sys = ConstantMatrix(np.array([[1, 0], [-1, 0]])) + SinusHamiltonian(1)
    L = [np.array([[0, 1], [0, 0]]), SinusHamiltonian(2)]
    g = [np.array([1, 2, 3]), np.array([4, 5])]
    w = [np.array([1 + 1j, 2, 3]), np.array([2j + 3, 4j + 5])]
    dim_sys = 2
    num_bcf_terms = [3, 2]
    eta_therm = [FakeStocProc(3), None]
    eta = [FakeStocProc(1), FakeStocProc(2)]
    ψ = np.array([1, 0])
    bcf_scale = [1, 2]

    bath_sizes = [len(gg) for gg in g]
    indices = IdxDict(bath_sizes).make(TruncationScheme_Simplex(2))

    params = HIParams(
        SysP=SysP(
            H_sys=H_sys,
            L=L,
            psi0=ψ,
            g=g,
            w=w,
            T=[0],
            bcf_scale=[1, 2],
        ),
        IntP=IntP(t_steps=(2, 10)),
        HiP=HiP(k_max=2),
        Eta=eta,
        EtaTherm=eta_therm,
    )

    linactor = LinearHOPSActor(np.linspace(0, 2), params, lambda _: (_, None, None))  # type: ignore
    nonlinactor = NonLinearHOPSActor(
        np.linspace(0, 2), params, lambda _: (_, None, None)  # type: ignore
    )

    def test_hops_actor(self):
        # we instantiate the linear actor in lieu
        actor = self.linactor

        assert isinstance(actor.L, DynamicMatrixList)
        assert isinstance(actor.L[0], ConstantMatrix)
        assert isinstance(actor.L[1], DynamicMatrix)

        assert_array_equal(actor.L[0]._matrix, self.L[0])  # type: ignore
        assert_array_equal_many_times(actor.L[1], self.L[1])

        assert actor.L.shape == (2, 2, 2)

        assert (
            len(actor.initial_state()) == len(self.indices) * self.params.SysP.dim_sys
        )

        reference = self.H_sys + ThermNoiseHamiltonian(
            FakeStocProc(3), ConstantMatrix(np.array([[0, 1], [0, 0]]))
        )

        assert_array_equal_many_times(reference, actor.H_sys)

    def test_update_random(self):
        actor = self.linactor

        actor.update_random_numbers(2)
        np.random.seed(2)
        stoc_temp_z = uni_to_gauss(np.random.rand(actor.thermal_hamiltonians[0].stoc_proc.get_num_y() * 2))  # type: ignore

        assert_array_equal(actor.thermal_hamiltonians[0].stoc_proc.set_zs, stoc_temp_z)

        new_proc = FakeStocProc(3)

        new_noise = ThermNoiseHamiltonian(
            new_proc,
            ConstantMatrix(np.array([[0, 1], [0, 0]])),
        )
        new_noise.new_process(stoc_temp_z)
        assert actor.thermal_hamiltonians[0].stoc_proc(100) == new_noise.stoc_proc(100)

        reference = self.H_sys + new_noise

        assert_array_equal_many_times(reference, actor.H_sys)

    def test_nonlin_actor(self):
        assert_array_equal(self.nonlinactor.ranges, np.array([[0, 3], [3, 5]]))

    def test_K_lin(self):
        res = self.linactor.K(
            1, self.ψ, np.array([np.conj(eta(1)) for eta in self.eta]), None
        )

        assert res.shape == self.linactor.L_dagger[0].shape

        res2 = self.linactor.K(1, self.ψ, np.zeros(2), None)
        assert_array_almost_equal(
            np.array(res - res2),
            self.eta[0](1).conj() * self.L[0] + self.eta[1](1).conj() * self.L[1](1),
        )

    def test_Bs_lin(self):
        res = self.linactor.Bs(1, np.array([self.ψ]))
        assert res.shape == (2, 2, 2)
        for result, L in zip(res, self.L):
            if not isinstance(L, np.ndarray):
                L = L(1)

            assert_array_equal(result, -L.conj().T)

    def test_Cs_lin(self):
        res = self.linactor.Cs(1, np.array([self.ψ]))
        assert res.shape == (2, 2, 2)
        assert_array_equal(res, np.array([self.L[0], self.L[1](1)]))

    def test_coupling_expectation(self):
        for i, L in enumerate(self.linactor.L.list):
            assert_array_equal_many_times(
                lambda t: self.nonlinactor.coupling_expectation(t, self.ψ)[i],
                lambda t: self.ψ.dot(L.dag(t).dot(self.ψ)),
            )

    def test_K_non_lin(self):
        res = self.nonlinactor.K(
            1, self.ψ, np.array([np.conj(eta(1)) for eta in self.eta]), np.array([1, 2])
        )
        assert res.shape == self.nonlinactor.L_dagger[0].shape

        res2 = self.nonlinactor.K(1, self.ψ, np.zeros(2), np.array([0, 0]))

        assert_array_almost_equal(
            np.array(res - res2),
            (self.eta[0](1).conj() + 1) * self.L[0]
            + (self.eta[1](1).conj() + 2) * self.L[1](1),
        )

    def test_Bs_nonlin(self):
        res = self.nonlinactor.Bs(1, np.array([self.ψ]))
        assert res.shape == (2, 2, 2)
        for result, L, exp in zip(
            res, self.L, self.nonlinactor.coupling_expectation(1, self.ψ)
        ):
            if not isinstance(L, np.ndarray):
                L = L(1)

            assert_array_equal(result, -1 * L.conj().T + exp * np.eye(2))

    def test_Cs_non_lin(self):
        res = self.nonlinactor.Cs(1, np.array([self.ψ]))
        assert res.shape == (2, 2, 2)
        assert_array_equal(res, np.array([self.L[0], self.L[1](1)]))

    def test_det_non_lin(self):
        res = self.nonlinactor.eta_det(np.array([1, 2, 3, 4, 5]))
        assert_array_equal(
            res,
            np.array(
                [
                    (self.nonlinactor.g_ast[0] * np.array([1, 2, 3])).sum(),
                    (self.nonlinactor.g_ast[1] * np.array([4, 5])).sum(),
                ]
            ),
        )

    def test_ddt_eta_lambda_non_lin(self):
        res = self.nonlinactor.ddt_eta_lambda(1, self.ψ, np.array([1, 2, 3, 4, 5]))

        exp_vals = self.nonlinactor.coupling_expectation(1, self.ψ)
        assert_array_equal(
            np.array(
                np.concatenate([exp_vals[0] * np.ones(3), exp_vals[1] * np.ones(2)])
                - np.concatenate(self.nonlinactor.omega_ast) * np.array([1, 2, 3, 4, 5])
            ),
            res,
        )


def test_thermal_noise_h():
    sp = FakeStocProc(10)
    noise_h = ThermNoiseHamiltonian(sp, ConstantMatrix([[0, 1], [0, 0]]))
    noise_h.new_process(np.array([1, 1]))

    sp_ref = FakeStocProc(10)
    sp_ref.new_process(np.array([1, 1]))

    assert_array_equal_many_times(
        lambda t: sp_ref(t).conjugate() * np.array([[0, 1], [0, 0]])
        + sp_ref(t) * np.array([[0, 1], [0, 0]]).T,
        noise_h,
    )

    L = SinusHamiltonian(15)
    noise_h = ThermNoiseHamiltonian(sp, L)
    noise_h.new_process(np.array([1, 1]))

    assert_array_equal_many_times(
        lambda t: sp_ref(t).conjugate() * L(t) + sp_ref(t) * L(t).T.conj(),
        noise_h,
    )

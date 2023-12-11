"""Tests for the :any:`hops.core.hierarchy_data` module.

They are not very exhaustive and should be enhanced at some point.

Todo
====

 - test normalization
"""

from numpy.core.numeric import array_equal
import pytest
from hops.core import hierarchy_data
from hops.core.hierarchy_data import HIData, HIMetaData
import hops.core.hierarchy_math as hm
import binfootprint as bf
from hops.core.hierarchy_parameters import ResultType
from hops.core.hierarchy_parameters import HIParams, HiP, ResultType
from collections.abc import Callable
import numpy as np
from pathlib import Path
from numpy.testing import assert_array_equal, assert_allclose
from .fixtures import simple_hops_zero


@pytest.fixture
def simple_hi_data(tmp_path: str, simple_hops_zero: Callable[..., HIParams]):
    """Initialize HIData with :any:`simple_hops_zero`."""
    hdf_path = Path(tmp_path) / "test.hdf5"
    simple = simple_hops_zero()
    data = HIData(str(hdf_path), False, simple)

    return data, simple


@pytest.fixture
def fake_trajectory():
    """Returns a factory to generate fake trajectories
    given a :any:`HIData` object."""

    def make_traj(data: HIData):
        shape: list[int] = list(data.stoc_traj[0].shape)  # type: ignore
        if data.aux_states:
            shape[-1] += data.aux_states[0].shape[-1]  # type: ignore
        fake_traj = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        return fake_traj

    return make_traj


class TestHiData:
    def test_init(self, simple_hi_data: tuple[HIData, HIParams]):
        """Test the initialization of HIData."""

        data, simple_hops_zero = simple_hi_data

        assert data.samples == 0
        assert data.largest_idx == 0
        assert not data.time_set
        assert not data.accum_only
        assert Path(data.hdf5_name).exists
        assert data.result_type == simple_hops_zero.HiP.result_type

        assert_array_equal(
            data.rho_t_accum.mean,
            np.zeros(
                (
                    simple_hops_zero.IntP.t_steps,
                    simple_hops_zero.SysP.dim_sys,
                    simple_hops_zero.SysP.dim_sys,
                )
            ),
        )

        assert_array_equal(
            data.rho_t_accum.ensemble_std,
            np.zeros(
                (
                    simple_hops_zero.IntP.t_steps,
                    simple_hops_zero.SysP.dim_sys,
                    simple_hops_zero.SysP.dim_sys,
                )
            ),
        )

        assert data.stoc_traj.shape == (
            hierarchy_data.HIData_default_size_stoc_traj,
            simple_hops_zero.IntP.t_steps,
            simple_hops_zero.SysP.dim_sys,
        )

        assert data.stoc_proc != None
        assert data.stoc_proc.shape == (
            hierarchy_data.HIData_default_size_stoc_traj,
            simple_hops_zero.IntP.t_steps,
            simple_hops_zero.SysP.total_number_bcf_terms,
        )

        assert HIMetaData.get_hashed_key(simple_hops_zero) == data.get_hi_key_hash()

    def test_store_and_retreive(
        self,
        simple_hi_data: tuple[HIData, HIParams],
        fake_trajectory: Callable[[HIData], np.ndarray],
    ):
        """Testing writing and loading some data."""

        data, params = simple_hi_data

        fake_1 = fake_trajectory(data)

        assert data.samples == 0
        data.new_samples(
            0, *data.result_filter(fake_1), params.HiP.result_type, False, 0
        )

        assert data.largest_idx == 0
        assert data.samples == 1

        assert data.has_sample(0)
        assert not data.has_sample(1)

        ψ = fake_1[:, 0 : params.SysP.dim_sys]
        assert_array_equal(data.get_stoc_traj(0), ψ)
        assert_array_equal(data.get_aux_states(0), fake_1[:, params.SysP.dim_sys :])

        var_1 = data.rho_t_accum.sample_variance
        assert_array_equal(var_1, np.zeros_like(var_1))

        with pytest.raises(RuntimeError):
            # inconsistent
            data.new_samples(
                0,
                *data.result_filter(fake_1),
                ResultType.ZEROTH_AND_FIRST_ORDER,
                False,
                0
            )

        ρ_1 = hm.projector_psi_t(ψ, normed=False)
        assert array_equal(data.get_rho_t(), ρ_1)

        fake_2 = fake_trajectory(data)
        data.new_samples(
            2000, *data.result_filter(fake_2), params.HiP.result_type, False, 0
        )

        assert data.largest_idx == 2000
        assert data.samples == 2

        assert data.has_sample(0)
        assert not data.has_sample(1)
        assert not data.has_sample(3)
        assert data.has_sample(2000)

        ψ_2 = fake_2[:, 0 : params.SysP.dim_sys]
        ρ_2 = hm.projector_psi_t(fake_2[:, : params.SysP.dim_sys], normed=False)

        assert_allclose(data.get_rho_t(), (ρ_1 + ρ_2) / 2)
        assert data.rho_t_accum.n == 2

        var_2 = data.rho_t_accum.sample_variance
        assert_allclose(
            var_2,
            (
                np.abs(np.array(ρ_1 - data.get_rho_t())) ** 2
                + np.abs(
                    np.array(hm.projector_psi_t(ψ_2, normed=False) - data.get_rho_t())
                )
                ** 2
            ),  # this is correct, think 2 - 1 = 1
        )

    def test_store_fast(
        self,
        simple_hi_data: tuple[HIData, HIParams],
        fake_trajectory: Callable[[HIData], np.ndarray],
    ):
        """Testing writing data fast."""

        data, params = simple_hi_data
        N = 100
        assert data.samples == 0
        for i in range(N):
            data.new_samples(
                i,
                *data.result_filter(fake_trajectory(data)),
                params.HiP.result_type,
                False,
                0
            )

        assert data.largest_idx == N - 1
        assert data.samples == N

    def test_save_and_load(
        self,
        simple_hops_zero: Callable[..., HIParams],
        tmp_path: str,
        fake_trajectory: Callable[[HIData], np.ndarray],
    ):
        """Test closing and reopening a data store."""
        N = 100
        params = simple_hops_zero()
        trajectories = []

        meta = HIMetaData("test", tmp_path)

        with meta.get_HIData(params) as data:
            for i in range(0, N):
                trajectory = fake_trajectory(data)
                trajectories.append(trajectory)

                data.new_samples(
                    i, *data.result_filter(trajectory), params.HiP.result_type, False, 0
                )

            assert data.samples == N
            assert data.rho_t_accum.n == N

        # this does a lot of things that could go wrong
        with meta.get_HIData(params, True) as data:
            assert data.samples == N
            assert data.rho_t_accum.n == N

            for i in range(0, N):
                assert data.has_sample(i)
                assert_array_equal(
                    trajectories[i][:, : params.SysP.dim_sys],
                    data.get_stoc_traj(i),
                )

            # read only
            with pytest.raises(RuntimeError):
                data.new_samples(
                    2000,
                    *data.result_filter(fake_trajectory(data)),
                    params.HiP.result_type,
                    False,
                    0
                )

    @pytest.mark.filterwarnings("ignore:sample")
    def test_write_twice(
        self,
        simple_hi_data: tuple[HIData, HIParams],
        fake_trajectory: Callable[[HIData], np.ndarray],
    ):
        """Test overwriting a trajectory."""
        data, params = simple_hi_data

        fake_1 = fake_trajectory(data)
        fake_2 = fake_trajectory(data)

        data.new_samples(
            0, *data.result_filter(fake_1), params.HiP.result_type, False, 0
        )
        data.new_samples(
            0, *data.result_filter(fake_2), params.HiP.result_type, False, 0
        )

        assert_array_equal(
            fake_1[:, : params.SysP.dim_sys],
            data.get_stoc_traj(0),
        )

    @pytest.mark.filterwarnings("ignore:Storing")
    def test_write_incomplete(
        self,
        simple_hi_data: tuple[HIData, HIParams],
        fake_trajectory: Callable[[HIData], np.ndarray],
    ):
        """Test storing an incomplete trajectory."""
        data, params = simple_hi_data

        fake_1 = fake_trajectory(data)

        data.new_samples(
            0,
            *data.result_filter(fake_1[: params.IntP.t_steps // 2, :]),
            params.HiP.result_type,
            False,
            0
        )

        assert not data.has_sample(0)

        with pytest.raises(RuntimeError):
            data.get_stoc_traj(0)

        traj = data.get_stoc_traj(0, incomplete=True)
        assert_array_equal(
            fake_1[: params.IntP.t_steps // 2, : params.SysP.dim_sys],
            traj[: params.IntP.t_steps // 2, :],
        )

        assert np.isnan(traj[params.IntP.t_steps // 2 :, :]).all()

    def test_zeroth_order_only(
        self,
        simple_hops_zero: Callable[..., HIParams],
        tmp_path: str,
        fake_trajectory: Callable[[HIData], np.ndarray],
    ):
        """Testing the ``ZEROTH_ORDER_ONLY`` result type."""
        params = simple_hops_zero()
        old_hip = params.HiP
        params.HiP = HiP(
            k_max=old_hip.k_max,
            seed=old_hip.seed,
            nonlinear=old_hip.nonlinear,
            result_type=ResultType.ZEROTH_ORDER_ONLY,
        )

        meta = HIMetaData("test", tmp_path)
        data = meta.get_HIData(params)

        fake_1 = fake_trajectory(data)

        assert data.samples == 0
        data.new_samples(
            0, *data.result_filter(fake_1), params.HiP.result_type, False, 0
        )

        assert data.largest_idx == 0
        assert data.samples == 1
        assert_array_equal(data.get_stoc_traj(0), fake_1)

    def test_accum_only(
        self,
        simple_hops_zero: Callable[..., HIParams],
        tmp_path: str,
        fake_trajectory: Callable[[HIData], np.ndarray],
    ):
        """Testing the ``accum_only`` mode."""

        params = simple_hops_zero()
        old_hip = params.HiP
        params.HiP = HiP(
            k_max=old_hip.k_max,
            seed=old_hip.seed,
            nonlinear=old_hip.nonlinear,
            result_type=ResultType.ZEROTH_ORDER_ONLY,
            accum_only=True,
        )

        meta = HIMetaData("test", tmp_path)
        data = meta.get_HIData(params)

        fake_1 = fake_trajectory(data)

        assert data.samples == 0
        data.new_samples(
            0, *data.result_filter(fake_1), params.HiP.result_type, False, 0
        )

        assert data.largest_idx == 0
        assert data.samples == 1

        assert data.has_sample(0)
        assert not data.has_sample(1)

        # we should be able to get the first trajectory
        ψ = fake_1[:, 0 : params.SysP.dim_sys]
        assert_array_equal(data.get_stoc_traj(0), ψ)

        var_1 = data.rho_t_accum.sample_variance
        assert_array_equal(var_1, np.zeros_like(var_1))

        ρ_1 = hm.projector_psi_t(ψ, normed=False)
        assert array_equal(data.get_rho_t(), ρ_1)

        fake_2 = fake_trajectory(data)
        data.new_samples(
            2000, *data.result_filter(fake_2), params.HiP.result_type, False, 0
        )

        assert data.largest_idx == 2000
        assert data.samples == 2

        assert data.has_sample(0)
        assert not data.has_sample(1)
        assert not data.has_sample(3)
        assert data.has_sample(2000)

        ψ_2 = fake_2[:, 0 : params.SysP.dim_sys]
        ρ_2 = hm.projector_psi_t(fake_2[:, : params.SysP.dim_sys], normed=False)

        assert_allclose(data.get_rho_t(), (ρ_1 + ρ_2) / 2)
        assert data.rho_t_accum.n == 2

        var_2 = data.rho_t_accum.sample_variance
        assert_allclose(
            var_2,
            (
                np.abs(np.array(ρ_1 - data.get_rho_t())) ** 2
                + np.abs(
                    np.array(hm.projector_psi_t(ψ_2, normed=False) - data.get_rho_t())
                )
                ** 2
            ),  # this is correct, think 2 - 1 = 1
        )

        with pytest.raises(ValueError):
            data.get_stoc_traj(2000)

    def test_overwrite_key(
        self,
        simple_hops_zero: Callable[..., HIParams],
        tmp_path: str,
        fake_trajectory: Callable[[HIData], np.ndarray],
    ):
        """Testing overwriting the key."""

        params = simple_hops_zero()
        meta = HIMetaData("test", tmp_path)
        data = meta.get_HIData(params)

        fake_1 = fake_trajectory(data)

        assert data.samples == 0
        data.new_samples(
            0, *data.result_filter(fake_1), params.HiP.result_type, False, 0
        )
        assert data.samples == 1

        data.close()
        params.HiP.rand_skip = 10

        with pytest.raises(RuntimeError):
            HIData(data.hdf5_name, read_only=False, hi_key=params)

        data = HIData(
            data.hdf5_name, read_only=False, hi_key=params, overwrite_key=True
        )
        assert data.largest_idx == 0
        assert data.samples == 1

    def test_get_without_key(
        self,
        simple_hops_zero: Callable[..., HIParams],
        tmp_path: str,
    ):
        """Testing to get the data by path only."""

        params = simple_hops_zero()
        meta = HIMetaData("test", tmp_path)
        data = meta.get_HIData(params)
        file = data.hdf5_name
        data.close()

        data = HIData(file, True)
        assert bf.dump(data.params) == bf.dump(params)

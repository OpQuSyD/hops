import pytest

# from stocproc import StocProc_TanhSinh
# from hops.core import hierarchy_parameters
# import pickle
import numpy as np

# from scipy.special import gamma as gamma_func
# from hops.core.hierarchy_parameters import ResultType
from hops.core.hierarchy_parameters import HIParams, HiP, IntP, ResultType, SysP
import hops.util.bcf as bcf
import stocproc as sp
from typing import Union, Optional

# from collections.abc import Callable
import os
import shutil
from pathlib import Path
from hops.util.dynamic_matrix import DynamicMatrix, SmoothStep


@pytest.fixture(scope="session")
def simple_hops_zero():
    """A simple test HOPS config for zero temperature and a single bath."""

    wc = 5
    s = 1

    int_conf = IntP(t_steps=(1, 10))

    bcf_zero = bcf.OhmicBCF_zeroTemp(s=s, eta=1, w_c=wc)
    g, w = bcf_zero.exponential_coefficients(5)
    sd = bcf.OhmicSD_zeroTemp(s=s, eta=1, w_c=wc)

    η = sp.StocProc_FFT(
        t_max=int_conf.t_max,
        alpha=bcf_zero,
        spectral_density=sd,
        intgr_tol=1e-3,
        intpl_tol=1e-3,
        invalidate_cache=True,
    )

    def factory(nonlinear: bool = True):
        hip = HiP(
            k_max=3,
            seed=1,
            nonlinear=nonlinear,
            result_type=ResultType.ALL,
        )

        sys = SysP(
            H_sys=np.array([[1, 0], [0, -1]]),
            L=[np.array([[0, 1], [1, 0]])],
            psi0=np.array([1, 0]),
            g=[g],
            w=[w],
            T=[0],
            bcf_scale=[0.1],
        )

        params = HIParams(hip, int_conf, sys, [η], [None])

        return params

    return factory


@pytest.fixture(scope="session")
def simple_multi_hops_zero():
    """A simple test HOPS config for zero temperature and two baths."""
    int_conf = IntP(
        t_steps=(1, 10),
    )

    s_and_wc = [(1, 5), (0.3, 2)]
    g = []
    w = []
    η = []

    for s, wc in s_and_wc:
        bcf_zero = bcf.OhmicBCF_zeroTemp(s=s, eta=1, w_c=wc)
        g_item, w_item = bcf_zero.exponential_coefficients(5)
        g.append(g_item)
        w.append(w_item)

        sd = bcf.OhmicSD_zeroTemp(s=s, eta=1, w_c=wc)

        η.append(
            sp.StocProc_FFT(
                t_max=int_conf.t_max,
                alpha=bcf_zero,
                spectral_density=sd,
                intgr_tol=1e-3,
                intpl_tol=1e-3,
                invalidate_cache=True,
            )
        )

    def factory(nonlinear: bool = True):
        hip = HiP(
            k_max=3,
            seed=1,
            nonlinear=nonlinear,
            result_type=ResultType.ALL,
        )

        sys = SysP(
            H_sys=np.array([[1, 0], [0, -1]]),
            L=[np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]])],
            psi0=np.array([1, 0]),
            g=g,
            w=w,
            T=[0],
            bcf_scale=[0.1, 0.2],
        )

        params = HIParams(hip, int_conf, sys, η, [None, None])

        return params

    return factory


@pytest.fixture(scope="session")
def simple_hops_nonzero():
    """A simple test HOPS config for nonzero temperature and a single bath."""
    wc = 5
    s = 1
    T = 1

    int = IntP(
        t_steps=(1, 10),
    )

    bcf_zero = bcf.OhmicBCF_zeroTemp(s=s, eta=1, w_c=wc)
    g, w = bcf_zero.exponential_coefficients(5)
    sd = bcf.OhmicSD_zeroTemp(s=s, eta=1, w_c=wc)

    stoc_temp_corr = bcf.Ohmic_StochasticPotentialCorrelations(
        s=s, eta=1, w_c=wc, beta=1 / T
    )
    stoc_temp_dens = bcf.Ohmic_StochasticPotentialDensity(
        s=s, eta=1, w_c=wc, beta=1 / T
    )

    η = sp.StocProc_FFT(
        t_max=int.t_max,
        alpha=bcf_zero,
        spectral_density=sd,
        intgr_tol=1e-3,
        intpl_tol=1e-3,
        invalidate_cache=True,
    )

    ξ = sp.StocProc_FFT(
        t_max=int.t_max,
        alpha=stoc_temp_corr,
        spectral_density=stoc_temp_dens,
        intgr_tol=1e-3,
        intpl_tol=1e-3,
        negative_frequencies=False,
        invalidate_cache=True,
    )

    def factory(nonlinear: bool = True):
        hip = HiP(
            k_max=3,
            seed=1,
            nonlinear=nonlinear,
            result_type=ResultType.ALL,
        )

        sys = SysP(
            H_sys=np.array([[1, 0], [0, -1]]),
            L=[np.array([[0, 1], [1, 0]])],
            psi0=np.array([1, 0]),
            g=[g],
            w=[w],
            T=[T],
            bcf_scale=[0.1],
        )

        params = HIParams(hip, int, sys, [η], [ξ])

        return params

    return factory


@pytest.fixture(scope="session")
def simple_multi_hops_nonzero():
    """A simple test HOPS config for zero temperature and two baths."""
    int_conf = IntP(t_steps=(1, 10))

    s_and_wc = [(1, 5), (0.3, 1)]
    g = []
    w = []
    η = []
    ξ = []
    T = [1, 30]

    for (s, wc), temp in zip(s_and_wc, T):
        bcf_zero = bcf.OhmicBCF_zeroTemp(s=s, eta=1, w_c=wc)
        sd = bcf.OhmicSD_zeroTemp(s=s, eta=1, w_c=wc)
        g_item, w_item = bcf_zero.exponential_coefficients(5)
        g.append(g_item)
        w.append(w_item)

        η.append(
            sp.StocProc_FFT(
                t_max=int_conf.t_max,
                alpha=bcf_zero,
                spectral_density=sd,
                intgr_tol=1e-3,
                intpl_tol=1e-3,
                invalidate_cache=True,
            )
        )

        stoc_temp_corr = bcf.Ohmic_StochasticPotentialCorrelations(
            s=s, eta=1, w_c=wc, beta=1 / temp
        )
        stoc_temp_dens = bcf.Ohmic_StochasticPotentialDensity(
            s=s, eta=1, w_c=wc, beta=1 / temp
        )

        ξ.append(
            (
                sp.StocProc_FFT(
                    t_max=int_conf.t_max,
                    alpha=stoc_temp_corr,
                    spectral_density=stoc_temp_dens,
                    intgr_tol=1e-3,
                    intpl_tol=1e-3,
                    negative_frequencies=False,
                    invalidate_cache=True,
                )
                if s >= 1
                else sp.StocProc_TanhSinh(
                    t_max=int_conf.t_max,
                    alpha=stoc_temp_corr,
                    spectral_density=stoc_temp_dens,
                    intgr_tol=1e-3,
                    intpl_tol=1e-3,
                    negative_frequencies=False,
                    invalidate_cache=True,
                )
            )
        )

    def factory(nonlinear: bool = True, time_dep: Optional[list[int]] = None):
        hip = HiP(
            k_max=3,
            seed=1,
            nonlinear=nonlinear,
            result_type=ResultType.ALL,
        )

        L: list[Union[np.ndarray, DynamicMatrix]] = [
            np.array([[0, 1], [0, 0]]),
            np.array([[0, 0], [1, 0]]),
        ]

        if time_dep:
            for i in time_dep:
                l_i = L[i]
                assert isinstance(l_i, np.ndarray)
                L[i] = SmoothStep(l_i, 0.1, 0.2, order=i)

        sys = SysP(
            H_sys=np.array([[1, 0], [0, -1]]),
            L=L,
            psi0=np.array([1, 0]),
            g=g,
            w=w,
            T=T,
            bcf_scale=[0.1, 0.2],
        )

        params = HIParams(hip, int_conf, sys, η, ξ)

        return params

    return factory


@pytest.fixture
def data_file(tmp_path: str, request: pytest.FixtureRequest):
    """
    Copies the test data located in ``[module_dir]/data/[test_name]/[name]``
    to a temp path that is returned.

    If ``init_file`` is given and the data file is not present yet, it
    will be initialized with ``init_file``.
    """

    path = tmp_path

    def factory(name: str, init_file: Union[str, None] = None):
        data_path: Path = (
            Path(os.path.dirname(request.module.__file__))
            / "data"
            / request.function.__name__
        )

        if not (data_path / name).exists():
            if init_file:
                data_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(init_file, str(data_path / name))
            else:
                raise RuntimeError(
                    f"Data file '{data_path/name}' does not exist and there is no `init_file` specified."
                )

        tmp_path = os.path.join(path, name)
        shutil.copy(str(data_path / name), tmp_path)

        return tmp_path

    return factory

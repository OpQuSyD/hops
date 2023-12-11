"""Tests for the :any:`hops.core.hierarchy_parameters` module."""

import pytest
from hops.core.hierarchy_parameters import HiP, ResultType, HIParams, SysP, IntP
from hops.util.abstract_truncation_scheme import TruncationScheme_Simplex
from .fixtures import simple_hops_zero, simple_hops_nonzero
from hops.util.bcf_fits import get_ohm_g_w
import stocproc as sp
import hops.util.bcf as bcf
import numpy as np


# Well, this is the only logic in this module :P
def test_hip_init():
    # this should go through
    hip = HiP(
        k_max=3,
        seed=1,
        nonlinear=True,
        terminator=False,
        result_type=ResultType.ALL,
        accum_only=False,
        truncation_scheme=None,
        save_therm_rng_seed=True,
    )

    hip = HiP(
        k_max=3,
        seed=1,
        nonlinear=True,
        terminator=False,
        result_type=ResultType.ALL,
        accum_only=False,
        truncation_scheme=None,
        save_therm_rng_seed=True,
    )

    with pytest.raises(ValueError):
        HiP(
            k_max=3,
            seed=1,
            nonlinear=True,
            terminator=False,
            result_type=ResultType.ALL,
            accum_only=False,
            truncation_scheme=TruncationScheme_Simplex(10),
            save_therm_rng_seed=True,
        )

    HiP(
        k_max=None,
        seed=1,
        nonlinear=True,
        terminator=False,
        result_type=ResultType.ALL,
        accum_only=False,
        truncation_scheme=TruncationScheme_Simplex(10),
        save_therm_rng_seed=True,
    )

    HiP(
        k_max=10,
        seed=1,
        nonlinear=True,
        terminator=False,
        result_type=ResultType.ALL,
        accum_only=False,
        truncation_scheme=None,
        save_therm_rng_seed=True,
    )

    with pytest.raises(ValueError):
        HiP(
            k_max=10,
            seed=1,
            nonlinear=True,
            terminator=False,
            result_type=ResultType.ALL,
            accum_only=None,
            truncation_scheme=None,
            rand_skip=10,
            save_therm_rng_seed=True,
        )


def test_hi_params_eta_scale(simple_hops_zero, simple_hops_nonzero):
    for factory in [simple_hops_zero, simple_hops_nonzero]:
        params: HIParams = factory()

        assert params.Eta[0].scale == params.SysP.bcf_scale[0]


def test_initial_norm():
    wc = 5
    s = 1

    g, w = get_ohm_g_w(5, s, wc)
    int = IntP(t_steps=(1, 10))

    bcf_zero = bcf.OhmicBCF_zeroTemp(s=s, eta=1, w_c=wc)
    sd = bcf.OhmicSD_zeroTemp(s=s, eta=1, w_c=wc)

    η = sp.StocProc_FFT(
        t_max=int.t_max,
        alpha=bcf_zero,
        spectral_density=sd,
        intgr_tol=1e-3,
        intpl_tol=1e-3,
        invalidate_cache=True,
    )

    hip = HiP(
        k_max=3, seed=1, nonlinear=True, result_type=ResultType.ALL, auto_normalize=True
    )

    sys = SysP(
        H_sys=np.array([[1, 0], [0, -1]]),
        L=[np.array([[0, 1], [1, 0]])],
        psi0=np.array([1, 1]),
        g=[g],
        w=[w],
        T=[0],
        bcf_scale=[0.1],
    )

    params = HIParams(hip, int, sys, [η], [None])
    assert np.linalg.norm(params.SysP.psi0) == pytest.approx(1)

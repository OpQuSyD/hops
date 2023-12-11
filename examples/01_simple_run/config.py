from hops.core.hierarchy_parameters import HIParams, HiP, IntP, SysP, ResultType
from hops.util.bcf_fits import get_ohm_g_w
from hops.util.truncation_schemes import BathMemory
import hops.util.bcf
import numpy as np
from stocproc import StocProc_FFT

wc = 5
s = 1

# The BCF fit
bcf_terms = 4
g, w = get_ohm_g_w(bcf_terms, s, wc)

integration = IntP(t_steps=(5, 10), rtol=1e-8, atol=1e-8)
system = SysP(
    H_sys=0.5 * np.array([[1, 0], [0, -1]]),
    L=[0.5 * np.array([[0, 1], [1, 0]])],
    psi0=np.array([1, 0]),
    g=[g],
    w=[w],
    bcf_scale=[1],
    T=[0.0],
)

params = HIParams(
    SysP=system,
    IntP=integration,
    HiP=HiP(
        nonlinear=True,
        result_type=ResultType.ALL,
        truncation_scheme=BathMemory.from_system(
            system, multipliers=(1.0, 1), nonlinear=True
        ),
    ),
    Eta=[
        StocProc_FFT(
            spectral_density=hops.util.bcf.OhmicSD_zeroTemp(
                s,
                1,
                wc,
            ),
            alpha=hops.util.bcf.OhmicBCF_zeroTemp(
                s,
                1,
                wc,
            ),
            t_max=integration.t_max,
            intgr_tol=1e-3,
            intpl_tol=1e-3,
            negative_frequencies=False,
        )
    ],
    EtaTherm=[
        StocProc_FFT(
            spectral_density=hops.util.bcf.Ohmic_StochasticPotentialDensity(
                s, 1, wc, beta=1 / system.__non_key__["T"][0]
            ),
            alpha=hops.util.bcf.Ohmic_StochasticPotentialCorrelations(
                s, 1, wc, beta=1 / system.__non_key__["T"][0]
            ),
            t_max=integration.t_max,
            intgr_tol=1e-3,
            intpl_tol=1e-3,
            negative_frequencies=False,
        )
    ]
    if system.__non_key__["T"][0] > 0
    else [None],
)

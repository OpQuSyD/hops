from hops.core.hierarchy_parameters import HIParams, HiP, IntP, SysP, ResultType
from hops.core.integration import HOPSSupervisor
from hops.util.bcf import LorentzianBCF, LorentzianSD
from stocproc import StocProc_FFT
import ray
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import pytest


@pytest.mark.slow
@pytest.mark.parametrize(
    "nonlinear",
    [True, False],
)
def test_damped_jaynes_cummings(nonlinear: bool, tmp_path: str):
    omega_a = 1.0
    omega_c = 2.0
    kappa = 1.5
    g = 3
    e = 0.25
    t_eval = np.linspace(0, 10, 100)

    N = 50
    dim_a = 2
    j = 1 / 2
    J_ops = qt.operators.jmat(j)
    L = qt.sigmam()
    L_full = qt.tensor(L, qt.identity(N))
    a = qt.tensor(qt.identity(dim_a), qt.destroy(N))
    jx = qt.tensor(J_ops[0], qt.identity(N))
    jy = qt.tensor(J_ops[1], qt.identity(N))
    jz = qt.tensor(J_ops[2], qt.identity(N))
    jm = qt.tensor(qt.sigmam(), qt.identity(N))

    c = np.sqrt(2 * kappa) * a
    H_sys = omega_a * J_ops[2] + e * J_ops[0]
    H = (
        qt.tensor(H_sys, qt.identity(N))
        + omega_c * a.dag() * a
        + g * (a.dag() * L_full + a * L_full.dag())
    )

    psi0 = qt.spin_coherent(1 / 2, 0, 0)
    env_alpha_0 = 0
    Psi0_full = qt.tensor(psi0, qt.coherent(N, env_alpha_0))

    res = qt.mesolve(H, Psi0_full, t_eval, c_ops=[c], progress_bar=True)
    exepec_x_me = [(jx * state).tr() for state in res.states]
    exepec_y_me = [(jy * state).tr() for state in res.states]
    exepec_z_me = [(jz * state).tr() for state in res.states]

    ray.init()

    s_x = 0.5 * np.array([[0, 1], [1, 0]], dtype="complex")
    s_y = 0.5 * np.array([[0, -1j], [1j, 0]])
    s_z = 0.5 * np.array([[1, 0], [0, -1]], dtype="complex")

    bcf = LorentzianBCF(g**2 / kappa, kappa, omega_c)
    system = SysP(
        H_sys=omega_a * s_z + e * s_x,
        L=[np.array([[0, 0], [1, 0]])],
        psi0=np.array([1, 0]),
        g=[np.asarray([g**2])],
        w=[np.asarray([kappa + omega_c * 1j])],
        bcf_scale=[1],
        T=[0.0],
    )
    integration = IntP(t=t_eval)
    hierarchy = HiP(
        k_max=5, result_type=ResultType.ALL, auto_normalize=True, nonlinear=nonlinear
    )
    Eta = [
        StocProc_FFT(
            spectral_density=LorentzianSD(g**2 / kappa, kappa, omega_c),
            alpha=bcf,
            t_max=integration.t_max,
            negative_frequencies=True,
            seed=12,
            intgr_tol=1e-3,
            intpl_tol=1e-3,
            invalidate_cache=True,
        )
    ]

    params = HIParams(
        SysP=system, IntP=integration, HiP=hierarchy, Eta=Eta, EtaTherm=[None]
    )

    sv = HOPSSupervisor(params, 8000, data_path=tmp_path)
    sv.integrate()

    def expect(o, r):
        return np.einsum("ij,tji->t", o, r)

    with sv.get_data(read_only=True) as data:
        rho_tij = data.get_rho_t()
        x = expect(s_x, rho_tij).real
        y = expect(s_y, rho_tij).real
        z = expect(s_z, rho_tij).real

    plt.plot(t_eval, np.real(exepec_x_me))
    plt.plot(t_eval, np.real(exepec_y_me))
    plt.plot(t_eval, np.real(exepec_z_me))
    plt.plot(t_eval, z)
    plt.plot(t_eval, y)
    plt.plot(t_eval, x)
    plt.savefig(f"{tmp_path}/plot.svg")

    np.testing.assert_allclose(z, exepec_z_me, atol=1e-2)
    np.testing.assert_allclose(x, exepec_x_me, atol=1e-2)
    np.testing.assert_allclose(y, exepec_y_me, atol=1e-2)

    ray.shutdown()

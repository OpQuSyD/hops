"""
A simple example to run a Spin Boson Model with HOPS comparing
the linear and non-linear HOPS. A plot of the results is saved
in the same directory as this file as a .png file.
"""
import pathlib
import ray
import numpy as np
import matplotlib.pyplot as plt
from stocproc.samplers import StocProc_FFT
from hops.core.integration import HOPSSupervisor
import hops.core.hierarchy_parameters as hops_params
from hops.util.bcf import OhmicSD_zeroTemp, OhmicBCF_zeroTemp


def simulation(num_samples:int = 100,
               non_linear: bool = True):
    # Defining time points
    final_time = 1
    num_timesteps = 100
    t_steps = (final_time, num_timesteps)
    # Defining parameters
    hierarchy_parameters = hops_params.HiP(k_max=3,
                                           seed=12345,
                                           nonlinear=non_linear,
                                           accum_only=False,
                                           save_therm_rng_seed=True,
                                           auto_normalize=True
                                           )
    integration_parameters = hops_params.IntP(t_steps=t_steps)
    pauliz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    paulix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    psi0 = np.array([1, 0], dtype=np.complex128)
    wc = 5
    s = 1
    ohmic_sd = OhmicSD_zeroTemp(s, 1, wc)
    ohmic_bcf = OhmicBCF_zeroTemp(s, 1, wc)
    n = 4
    g , w = ohmic_bcf.exponential_coefficients(n)
    scale = [1.0]
    system_parameters = hops_params.SysP(pauliz,
                                         [paulix],
                                         psi0,
                                         [g], [w], scale,
                                         description="A rather randomly chosen Spin Boson Model")
    ## The Docs of StocProc seem to be outdated
    eta = StocProc_FFT(ohmic_sd,
                       integration_parameters.t_max,
                       ohmic_bcf,
                       intgr_tol=1e-3,
                       intpl_tol=1e-3,
                       )
    params = hops_params.HIParams(HiP=hierarchy_parameters,
                                  IntP=integration_parameters,
                                  SysP=system_parameters,
                                  Eta=[eta],
                                  EtaTherm=[None])
    supervisor = HOPSSupervisor(params,num_samples)
    ray.init()
    supervisor.integrate(clear_pd=True)
    ray.shutdown()
    data = supervisor.get_data(stream=False)
    time = data.get_time()
    rho = data.get_rho_t()
    expectation_values = np.tensordot(rho, pauliz, axes=([1,2],[0,1]))
    print("Maximum Imaginary Part: ", np.max(np.imag(expectation_values)))
    return time, np.real(expectation_values)

def plot_results(time,
                 expectation_values_lin,
                 expectation_values_nonlin):
    _, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for num_samples, exp_val in expectation_values_lin.items():
        axs[0].plot(time, exp_val, label=f"Samples: {num_samples}")
    for num_samples, exp_val in expectation_values_nonlin.items():
        axs[1].plot(time, exp_val, label=f"Samples: {num_samples}")
    axs[0].set_title("Linear")
    axs[1].set_title("Non-Linear")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Time")
    axs[0].set_ylabel("Expectation Value <Z>")
    plt.legend()
    plt.savefig(str(pathlib.Path(__file__).parent.resolve()) + "/spin_boson.png")

def main():
    num_samples_list = [1, 10, 100, 500, 1000]
    expectation_values_nonlin = {}
    expectation_values_lin = {}
    time = None
    for num_samples in num_samples_list:
        _, exp_val = simulation(num_samples=num_samples,
                                   non_linear=False)
        expectation_values_lin[num_samples] = exp_val
        time, exp_val = simulation(num_samples=num_samples,
                                   non_linear=True)
        expectation_values_nonlin[num_samples] = exp_val
    plot_results(time,
                 expectation_values_lin,
                 expectation_values_nonlin)

if __name__ == "__main__":
    main()

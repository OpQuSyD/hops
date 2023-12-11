Tutorial
========

Below we have collected some typical usecases for HOPS and how they can be
implemented with the present code. These tutorials are intended to get you started
with using the HOPS code!

The Simplest Case
-----------------
Let's go ahead with the simple case of the driven, damped Jaynes-Cummings Model,
that is a single qubit coupled to a lossy mode.
There the evolution of the atom alone is identical to the evolution of an atom
coupled to a bath with a Lorentian spectral density. Therefore the correlation
function decays exponentially

.. math::

    \alpha(t-s) = Ge^{-W|t-s|}.

The system itself shall evolve according to the system Hamiltonian

.. math::

    H_{\mathrm{sys}} = \frac{\omega}{2}\sigma^z + \epsilon\sigma^x.

To start we import all necessary modules::

   from hops.core.hierarchy_parameters import HIParams, HiP, IntP, SysP
   from hops.core.integration import HOPSSupervisor
   from hops.util.bcf import LorentzianBCF, LorentzianSD
   from stocproc import StocProc_FFT
   import ray
   import numpy as np
   import matplotlib.pyplot as plt

and define our parameters::

    omega_a = 1.
    omega_c = 2.
    kappa = 15
    g = 1.
    e = .25

To convey all the parameters of the HOPS algorithm we create :ref:`hierarchy parameters`
for the quantum system (physics of the problem), the integration routine and 
the hierarchy. Additionally we create the stochastic process Eta. We can use
the StocProc package to create the noise with exponentially decaying correlation::

    system = SysP(
        H_sys=0.5 * omega_a * np.array([[1, 0], [0, -1]]) + e*np.array([[0, 1], [1, 0]]),
        L=[np.array([[0, 0], [1, 0]])],
        psi0=np.array([1, 0]),
        g=[np.asarray([g**2])],
        w=[np.asarray([kappa+omega_c*1j])],
        bcf_scale=[1.],
        T=[0.0],
    )
    integration = IntP(t_steps=(10, 100))
    hierarchy=HiP(k_max=10)
    Eta=[
        StocProc_FFT(
                spectral_density=LorentzianSD(g**2/kappa, kappa, omega_c),
                alpha=LorentzianBCF(g**2/kappa, kappa, omega_c),
                t_max=integration.t_max,
                negative_frequencies=True
            )
    ]

If we want a calculation at finite temperature we would need to additionally
specify the thermal stochastic process Eta_Therm.
Finally we convey all these information to the HOPS algorithm in a single params
object::

    params = HIParams(SysP=system, IntP=integration, HiP=hierarchy, Eta=Eta, EtaTherm=[None])

Now the time has finally come to run our simulation! We first initiallize the
Ray service on our machine - by default HOPS will then use all available cores.
For more options see `here <https://docs.ray.io/en/latest/configure.html>`_.
Then we create the object that starts the simulation by handing over the
parameters and the number of trajectories that we want to simulate (here 10)
and start the integration::

    ray.init()
    sv = HOPSSupervisor(params, 10)
    sv.integrate()

For debugging purposes it might sometimes be usefull to not parallelize the 
execution. This can be done by using the ``integrate_single_process`` method::

    sv = HOPSSupervisor(params, 10)
    sv.integrate_single_process()

The results of the computation are stored in a file. To analyse the results
we first define::

    def expect(o, r):
        return np.einsum("ij,tji->t", o, r)

    s_x = np.array([[0,1],[1,0]], dtype="complex")
    s_y = np.array([[0, -1j], [1j, 0]])
    s_z = np.array([[1,0], [0,-1]], dtype="complex")

and then read of the output data like this::

    with sv.get_data(read_only=True) as data:
        rho_tij = data.get_rho_t()
        x = expect(s_x, rho_tij).real
        y = expect(s_y, rho_tij).real
        z = expect(s_z, rho_tij).real
        t = list(data.time)

        plt.plot(t, x)
        plt.plot(t, y)
        plt.plot(t, z)

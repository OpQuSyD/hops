"""Utilities to get fit data for bath correlation functions."""

from ..data.path import get_path
import numpy as np
import pickle
from scipy.special import gamma as gamma_func


def get_ohm_g_w(n: int, s: float, wc: float, ws: float = 0, scaled: bool = True):
    r"""
    :returns: The BCF expansion parameters :math:`G_μ` and
              :math:`W_μ` for a (sub)ohminc bath with parameter ``s``
              and ``wc`` (:math:`ω_c`) and ``n`` terms.

        The BCF comes from a bcf of the form

        ..  math::

            α(τ) = η / (1 + i ω_c τ)^{s+1} \cdot e^{-i ω_s τ}

    :param n: The number of BCF terms.
    :param s: The :math:`s` parameter.
    :param wc: The :math:`ω_c` parameter.
    :param ws: The :math:`ω_s` parameter.
    :param scaled: Whether the :math:`G_i` are being scaled with an
        :math:`η` to match the conventions of
        ``hops.util.bcf.OhmicBCF_zeroTemp``.  Otherwise they're
        fitted to :math:`η=1`.
    """

    __BCF_FIT_good_data_path = get_path() + "/good_fit_data_abs_brute_force"
    with open(__BCF_FIT_good_data_path, "rb") as f:
        good_fit_data_abs = pickle.load(f)

    _, g_tilde, w_tilde = good_fit_data_abs[(n, s)]
    g = (
        1 / np.pi * gamma_func(s + 1) * wc ** (s + 1) * np.asarray(g_tilde)
        if scaled
        else np.asarray(g_tilde)
    )
    w = wc * np.asarray(w_tilde) + 1j * ws

    return g, w

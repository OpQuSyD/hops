"""Some mathimatical utilities."""

import numpy as np
from typing import Optional


def projector_psi_t(psi_t: np.ndarray, normed: bool = False) -> np.ndarray:
    r"""
    Calculate the time dependent projector from a time dependent state.

    .. math::

       |ψ\rangle \rightarrow |ψ\rangle\langle ψ|

    :param psi_t: The time dependent state with shape ``(time steps, system dimension)``.
    :param normed: Whether to normalize the projector.
    """

    psi_t_col = np.expand_dims(psi_t, axis=2)
    psi_t_row = np.expand_dims(psi_t, axis=1)

    if normed:
        N = np.sum(np.conj(psi_t) * psi_t, axis=1).reshape(psi_t.shape[0], 1, 1)
        return (psi_t_col * np.conj(psi_t_row)) / N
    else:
        return psi_t_col * np.conj(psi_t_row)


def norm_psi_t(psi_t: np.ndarray, keepdims: Optional[bool] = None) -> np.ndarray:
    """Normalize a time deppendent state.

    :param psi_t: The time dependent state with shape ``(time steps, system dimension)``.
    :param keepdims: See :any:`numpy.sum`.
    """
    return np.sqrt(np.sum(np.abs(psi_t) ** 2, axis=(1,), keepdims=keepdims))  # type: ignore


def operator_norm(L: np.ndarray) -> float:
    r"""Calculates the operator norm of the Hilbert space operator
    ``L``.

    It is defined as :math:`\sup{\frac{||L x||}{||x||}}_{x\in H}`.
    In the finite dimensional case this is just the square root of
    the largest eigenvalue of :math:`L^† L`.
    """

    single = len(L.shape) == 2
    L_dag = np.transpose(L, axes=None if single else (0, 2, 1))
    prods = L_dag @ L
    return np.sqrt(
        np.max(np.abs(np.linalg.eigvals(prods)), axis=(0 if single else 1))
    ).real

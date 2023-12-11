"""Miscelaneous utilities for the HOPS and open systems."""

from ..core.hierarchy_data import HIData

from typing import Optional, TypeVar
import numpy as np
from numpy.typing import NDArray
import scipy


def _square_mat(mat: NDArray) -> NDArray:
    """
    :returns: The matrix ``mat`` multiplied with itself.
    """

    return mat @ mat


def logm2(mat: NDArray) -> NDArray:
    """
    :returns: The base two logarithm of ``mat``.
    """

    return scipy.linalg.logm(mat) / np.log(2)


def entropy(data: HIData) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the von-Neumann entropy and its standard deviation of
    the time series in ``data``.

    :returns: The von-Neumann entropy for every time and its deviation.
    """

    entropy = np.fromiter(
        (-np.trace(ρ @ (logm2(ρ))).real for ρ in data.rho_t_accum.mean),
        dtype=np.float64,
        count=data.rho_t_accum.mean.shape[0],
    )

    Δ_entropy = np.fromiter(
        (
            np.sqrt(np.trace(_square_mat(Δρ @ (logm2(ρ) + 1))).real)
            for ρ, Δρ in zip(data.rho_t_accum.mean, data.rho_t_accum.ensemble_std)
        ),
        dtype=np.float64,
        count=data.rho_t_accum.mean.shape[0],
    )

    return entropy, Δ_entropy


def relative_entropy_single(
    ρ: NDArray[np.complex128], σ: NDArray[np.complex128]
) -> float:
    """
    :returns: The entropy of ``ρ`` relative to ``σ``.
    """

    return np.trace(ρ @ (logm2(ρ) - logm2(σ))).real


def _get_reference_state(
    data: HIData,
    final_index: Optional[int] = None,
    relative_to: Optional[NDArray[np.complex128]] = None,
) -> NDArray[np.complex128]:
    """
    :returns: The state at the ``final_index`` or ``relative to``.  If
              neither is given, ``final_index=-1`` is assumed.
    """

    if final_index is not None and relative_to is not None:
        raise ValueError("Provide either `final_index` or `relative_to`.")

    ρ_ref = relative_to
    if relative_to is None:
        if final_index is None:
            final_index = -1
        ρ_ref = data.rho_t_accum.mean[final_index]

    return ρ_ref


def relative_entropy(
    data: HIData,
    final_index: Optional[int] = None,
    relative_to: Optional[NDArray[np.complex128]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the entropy (and its std deviation) of the state
    contained in ``data`` **relative** to the state at ``final_index``
    **or** to ``relative_to``.

    :param data: The data instance containing the state time series.
    :param final_index: The time-index of the state that the entropy
        is calculated relative to.
    :param relative_to: The state that the entropy is calculated
        relative to. Defaults to ``-1``.

        May be provided instead of ``final_index``.
    """

    ρ_ref = _get_reference_state(data, final_index, relative_to)

    rel_entropy = np.fromiter(
        (relative_entropy_single(ρ, ρ_ref) for ρ in data.rho_t_accum.mean),
        dtype=np.float64,
        count=data.rho_t_accum.mean.shape[0],
    )

    Δ_rel_entropy = np.fromiter(
        (
            np.sqrt(np.trace(_square_mat(Δρ @ (logm2(ρ) + 1 - logm2(ρ_ref))).real))
            for ρ, Δρ in zip(data.rho_t_accum.mean, data.rho_t_accum.ensemble_std)
        ),
        dtype=np.float64,
        count=data.rho_t_accum.mean.shape[0],
    )

    return rel_entropy, Δ_rel_entropy


def trace_distance(
    data: HIData,
    final_index: Optional[int] = None,
    relative_to: Optional[NDArray[np.complex128]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the trace distance (and its std deviation) of the state
    contained in ``data`` to the state at ``final_index``
    **or** to ``relative_to``.

    :param data: The data instance containing the state time series.
    :param final_index: The time-index of the state that the distance
        is calculated to.
    :param final_index: The state that the distance is
        relative to. Defaults to ``-1``.

        May be provided instead of ``final_index``.
    """

    ρ_ref = _get_reference_state(data, final_index, relative_to)

    ρ_rel = data.rho_t_accum.mean - ρ_ref
    trace_dist = np.einsum("tij,tji->t", ρ_rel, ρ_rel)
    trace_dist_deviation = np.einsum(
        "tij,tji->t", data.rho_t_accum.ensemble_std, data.rho_t_accum.ensemble_std
    )
    return np.sqrt(trace_dist), np.sqrt(trace_dist_deviation)

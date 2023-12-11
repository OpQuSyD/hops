"""Functionality for storing HOPS results."""

# Note: the `type: ignore` comments are used when interfacing with the
# all-too polymorphic h5py and NOWHERE ELSE

import hashlib
import logging
import os
import pathlib
import pickle
import shutil
import time
import signal
import warnings
import functools
from typing import Optional, TypeVar, Union, Any
from collections.abc import Iterator, Callable
from h5py._hl.dataset import Dataset

import h5py
import numpy as np
import binfootprint as bf


from .hierarchy_parameters import HIParams, ResultType
from . import signal_delay
from . import hierarchy_math as hm
from . import utility as ut

log = logging.getLogger(__name__)
T = TypeVar("T")
ResultFilter = Callable[
    [np.ndarray],
    tuple[bool, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
]
"""A function that takes the total integration result and filters it,
so that no superflous data is returned.

See :any:`filter_psi_all`.

:returns: A :any:`bool` that signifies whether the sample is
          incomplete, the stochastic trajectory, the auxiliary states (if any)
          and the stochastic process shift (if it is to be stored).

          If :any:`None` is returned, there is no sample to be stored.
"""

HIData_default_size_stoc_traj = 10
"""The default chunk size for storing the trajectories."""

HIData_default_size_rho_t_accum_part = 10


class WelfordAggregator:
    """A helper class to calculate means and variances on datasets incremenally
    (online) in a numerically stable fashion.

    See also the `Wikipedia article <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm>`_.

    The datasets are supposed to managed externally.
    Setting the ``n`` parameter to the correct value upon initialization
    is **your** responsibility.

    :param mean: a dataset to store the mean value
    :param m_2: a dataset to store the welford auxiliary value
        for the variance
    :param n: the number of samples
    """

    def __init__(self, mean: h5py.Dataset, m_2: h5py.Dataset, n: int):
        #: the sample count
        self.n: int = n

        #: the dataset to store the mean
        self.dataset: h5py.Dataset = mean

        #: the dataset to store the auxilliary value
        self._m_2: h5py.Dataset = m_2

    def update(self, new_value: np.ndarray):
        """Add the value ``new_value`` to the aggregator and compute the new
        mean and M2 auxilliary value."""

        self.n += 1
        delta: np.ndarray = new_value - self.dataset[:]  # type: ignore

        self.dataset[:] += delta / self.n  # type: ignore

        delta2 = new_value - self.dataset[:]  # type: ignore
        self._m_2[:] += np.abs(delta) * np.abs(delta2)  # type: ignore

        # Note: bloody h5py...

    def reset(self):
        """Reset the aggregator by setting everything to zero."""

        self.dataset[:] = 0j
        self._m_2[:] = 0
        self.n = 0

    @property
    def mean(self) -> np.ndarray:
        """The mean value of the aggregated quantity."""

        return self.dataset[:]  # type: ignore

    @property
    def sample_variance(self) -> np.ndarray:
        """The sample variance of the aggregated quantity.

        This is the nonbiased empirical variance (think (n-1)).
        """
        if self.n < 2:
            return np.zeros_like(self.mean)

        return self._m_2[:] / (self.n - 1)  # type: ignore

    @property
    def ensemble_variance(self) -> np.ndarray:
        """The ensemble variance of the aggregated quantity.

        This is simply the :any:`sample_variance` divided by :any:`n`.
        """

        if self.n == 0:
            return np.zeros_like(self.mean)

        return self.sample_variance / self.n

    @property
    def ensemble_std(self) -> np.ndarray:
        """The ensemble standard deviation.

        The square root of :any:`ensemble_variance`.
        """

        return np.sqrt(self.ensemble_variance)


def filter_psi_all(
    size_sys: int,
    size_aux_states: int,
    size_t: int,
    first_hier_indices: Optional[np.ndarray],
    result_type: ResultType,
    psi_all: np.ndarray,
) -> tuple[bool, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filter the full integration result ``psi_all`` into a
    :any:`bool` that signifies whether the sample is incomplete,
    the stochastic trajectory, the auxiliary states and the
    stochastic process shift according to :any:`result_type`.

    If the sample is shorter than :any:`size_t` it will be
    classified as incomplete.
    """

    incomplete = False
    _tn, _traj_shape = psi_all.shape

    if _tn < size_t:
        incomplete = True

        psi_all_new = np.empty(shape=(size_t, _traj_shape), dtype=np.complex128)

        psi_all_new[:] = np.nan + 1j * np.nan
        psi_all_new[0:_tn, :] = psi_all
        psi_all = psi_all_new

    c = psi_all.shape[1]

    psi0 = None
    aux_states = None
    stoc_proc = None

    if result_type == ResultType.ZEROTH_ORDER_ONLY:
        psi0 = psi_all[:, :size_sys]

    elif result_type == ResultType.ZEROTH_ORDER_AND_ETA_LAMBDA:
        psi0 = psi_all[:, :size_sys]
        if c > size_sys and stoc_proc:  # the linear HI has no stoc_proc data
            stoc_proc = psi_all[:, size_sys:]

    elif result_type == ResultType.ZEROTH_AND_FIRST_ORDER:
        psi0 = psi_all[:, :size_sys]
        aux_states = psi_all[:, first_hier_indices].reshape(
            psi_all.shape[0], size_aux_states
        )

        # TODO: implement returning the stocproc
        # if c > _i:  # the linear HI has no stoc_proc data
        #     self.stoc_proc[idx] = psi_all[:, _i:]
    else:  # type -> ALL
        psi0 = psi_all[:, :size_sys]
        _i = size_aux_states + size_sys

        aux_states = psi_all[:, size_sys:_i]
        if c > _i:  # the linear HI has no stoc_proc data
            stoc_proc = psi_all[:, _i:]

    return incomplete, np.array(psi0), aux_states, stoc_proc


class HIData:
    """
    Implements the storage of HOPS integration results.

    :param hdf5_name: The filename for the storage.
    :param read_only: Whether to open the file in read-only mode.
    :param hi_key: The information uniquely determining the problem.
    :param hi_key_bin: The binary representation of `hi_key`.
    :param hi_key_hash: The hash of `hi_key`.
    :param check_consistency: Do try to check if the HDF5 file
        contains data that was generated using the settings from
        ``hi_key``.

    :param overwrite_key: Overwrite the ``hi_key`` stored in the HDF5
        file if it is present.

        Does nothing if ``read_only`` is :any:`True`.

    :param robust: Backup the file and create a new one if it fails to
        open.

    :param stream_file: The file path where the streamed results will
        be written to if
        ``hops.core.hierarchy_parameters.HiP.stream_result_type``
        is not :any:`None`.

        For the last two parameters see ``binfootprint``.  If they are
        not given, they will be calculated on the fly.
    """

    ###########################################################################
    #                              Initialization                             #
    ###########################################################################

    def __init__(
        self,
        hdf5_name: str,
        read_only: bool,
        hi_key: Optional[HIParams] = None,
        hi_key_bin: Optional[bytes] = None,
        hi_key_bin_hash: Optional[str] = None,
        check_consistency: bool = True,
        overwrite_key: bool = False,
        robust: bool = True,
        stream_file: Optional[str] = None,
    ):
        self.params: HIParams
        """The HOPS configuration used to generate the data in this database."""

        if not hi_key:
            with h5py.File(hdf5_name, "r", libver="latest") as h5File:
                try:
                    self.params = pickle.loads(bytearray(h5File["pickled_hi_key"][:]))  # type: ignore

                except Exception as e:
                    log.error(
                        f"Could not load the HOPS configuration from {hdf5_name}."
                    )
                    raise e
        else:
            self.params = hi_key

        if hi_key_bin is None:
            hi_key_bin = bytes(bf.dump(self.params))  # type: ignore
            # TODO: (Valentin) binfootprint typing

        if hi_key_bin_hash is None:
            hi_key_bin_hash = hashlib.md5(hi_key_bin).hexdigest()

        #: file name of the data storage
        self.hdf5_name: str = hdf5_name

        #: system dimension
        self.size_sys: int = self.params.SysP.dim_sys

        #: number of time steps
        self.size_t: int = self.params.IntP.t.size

        #: whether to enable the accum_only mode
        self.accum_only: Optional[bool] = self.params.HiP.accum_only

        #: the result type mean to be stored
        self.result_type: ResultType = self.params.HiP.result_type

        #: the result type meant to be streamed
        self.stream_result_type: Optional[ResultType] = self.params.HiP.__non_key__.get(
            "stream_result_type", None
        )

        #: the fifo in which the result will be streamed
        self.stream_file: Optional[str] = (
            os.path.abspath(stream_file) if stream_file else None
        )

        #: whether to save the seed used for the generation of the
        #: thermal stochastic process
        self.save_therm_rng_seed: bool = self.params.HiP.save_therm_rng_seed

        #######################################################################
        #  The following attributes will be initialized in auxiliary methods. #
        #######################################################################

        self.first_hier_indices: np.ndarray
        """The indices of the first hierarchy states with shape
        ``(system dimension * number of BCF terms,)``.

        The indices are flattened out, so that ``psi_all[:, self.first_hier_indices]``
        gives the first hierarchy states concatenated together where ``psi_all`` is
        the result vector of a hops integration.
        """

        self.size_aux_states: int
        """Size of the auxilliary state vector.

        If it is zero, the auxiliary states won't be saved.
        """

        self.sp_num_bcf_terms: int
        """
        The total number of bcf terms.  Used interanlly for storage of
        the stochastic process.  Is set to zero if the process is not
        stored.

        See also
        :any:`hops.core.hierarchy_parameters.SysP.total_number_bcf_terms`.
        """

        #: the underlying storage
        self.h5File: h5py.File

        self.stoc_traj: h5py.Dataset
        """The stochastic trajectories without auxiliary states.

        Has the shape ``(samples, time steps, system dimension)``.
        """

        self._rho_t_accum: h5py.Dataset
        """The storage for the mean value of the Welford aggregator
        for the density matrix. See :any:`rho_t_accum`.
        """

        self._rho_t_accum_m2: h5py.Dataset
        """An intermediate value for storing the M2 value for the Welford
        aggregator (see :any:`WelfordAggregator`).
        """

        self.rho_t_accum_part: h5py.Dataset
        """Essentially :any:`rho_t_accum` matrix after ``2^n``
        samples have been received.

        Has the shape ``(samples, time steps, system dimension, system dimension)``.
        Can be used to observe the convergence of the density matrix.

        Best accessed with :meth:`get_rho_t_part`.
        """

        self.rho_t_accum_part_tracker: h5py.Dataset
        """A dataset indexed by the sample number that contains
        a boolean value that tells whether the sample ``2^n`` in
        question has been added to :any:`rho_t_accum_part`.

        This information is best accessed by :meth:`has_rho_t_accum_part`
        Has the shape ``(samples,)``.
        """

        self.tracker: h5py.Dataset
        """A dataset indexed by the sample number that contains a boolean value that
        tells whether the sample in question has been processed yet`.

        This information is best accessed by :meth:`has_sample`.
        Has the shape ``(samples,)``.
        """

        self.rng_seed: Optional[h5py.Dataset] = None
        """The seeds used for the generation of the thermal stochastic processes.
        Has the shape ``(samples,)``.

        May be :any:`None` if :any:`save_therm_rng_seed` is :any:`False`.
        """

        self.aux_states: Optional[h5py.Dataset] = None
        """The auxiliary states of the first hierarchy if the :any:`result_type` is
        :any:`ResultType.ZEROTH_AND_FIRST_ORDER` or all auxilliary states if
        the result type is :any:`ResultType.ALL`. :any:`None` otherwise.

        Has the shape ``(samples, time steps, size of auxiliary state)``.

        Best accessed through :any:`get_aux_states`.
        """

        self.stoc_proc: Optional[h5py.Dataset] = None
        """The shifts of the driving stochastic processes concatenated.

        Has the shape ``(samples, time steps, number of BCF terms)``.

        May be :any:`None` if :any:`sp_num_bcf_terms` is zero.
        """

        self.time: h5py.Dataset
        """The time points on which the trajectories and everything else are
        given.

        Has the shape ``(time steps,)``.
        Best accessed through :meth:`get_time`.

        Has an attribute "time_set" (``self.time.attrs["time_set"]``) that signifies whether
        :any:`time` has been set to a meaningful value. This property is best accessed through
        :any:`time_set`.
        """

        self._init_bcf_terms_and_aux_states()
        self._open_file(read_only, hi_key_bin, hi_key_bin_hash, overwrite_key, robust)

        if check_consistency:
            self._check_consistency(hi_key_bin, hi_key_bin_hash)
        else:
            warnings.warn("Consistency check bypassed.")

        self._load_file_contents()

        self._idx_cnt = len(self.tracker)  # type: ignore
        self._idx_rho_t_accum_part_tracker_cnt = len(self.rho_t_accum_part_tracker)  # type: ignore

        self.rho_t_accum: WelfordAggregator = WelfordAggregator(
            self._rho_t_accum, self._rho_t_accum_m2, self.samples
        )
        """The sum of the density matrices obtained
        from the the individual trajectories.

        Gets updated with every new trajectory.

        Has the shape ``(time step, system dimension, system dimension)``.

        May be rebuilt with :meth:`rewrite_rho_t`.
        """

        if self.stream_file and self.stream_result_type:
            log.info(f"Creating the streaming fifo at: {self.stream_file}")

            if not os.path.exists(self.stream_file):
                os.mkfifo(self.stream_file)

            self._stream_fifo = open(self.stream_file, "wb")
            """
            A named pipe where results are being streamed to for
            online processing.
            """

        else:
            self._stream_fifo = None

    def _init_bcf_terms_and_aux_states(self):
        if (
            self.params.HiP.result_type
            in [ResultType.ALL, ResultType.ZEROTH_AND_FIRST_ORDER]
        ) and self.accum_only:
            raise ValueError(
                "ResultType.ALL or ResultType.ZEROTH_AND_FIRST_ORDER and accum_only are incompatible"
            )

        if self.params.HiP.result_type == ResultType.ZEROTH_ORDER_ONLY:
            size_aux_state = 0
            num_bcf_terms = 0

        elif self.params.HiP.result_type == ResultType.ZEROTH_ORDER_AND_ETA_LAMBDA:
            size_aux_state = 0
            num_bcf_terms = self.params.SysP.total_number_bcf_terms

        elif self.params.HiP.result_type in [
            ResultType.ALL,
            ResultType.ZEROTH_AND_FIRST_ORDER,
        ]:
            num_bcf_terms = self.params.SysP.total_number_bcf_terms

            idxDict = self.params.indices

            if self.params.HiP.result_type == ResultType.ALL:
                size_aux_state = (idxDict.num_idx() - 1) * self.size_sys
            else:
                size_aux_state = num_bcf_terms * self.size_sys

        else:
            raise RuntimeError(
                "Unknown self.params.HiP.result_type: {}".format(
                    self.params.HiP.result_type
                )
            )

        self.size_aux_states = size_aux_state
        self.sp_num_bcf_terms = num_bcf_terms if self.params.HiP.nonlinear else 0

        # TODO: less copy pasta
        if self.stream_result_type:
            if self.stream_result_type == ResultType.ZEROTH_ORDER_ONLY:
                size_aux_state = 0
                num_bcf_terms = 0

            elif self.stream_result_type == ResultType.ZEROTH_ORDER_AND_ETA_LAMBDA:
                size_aux_state = 0
                num_bcf_terms = self.params.SysP.total_number_bcf_terms

            elif self.stream_result_type in [
                ResultType.ALL,
                ResultType.ZEROTH_AND_FIRST_ORDER,
            ]:
                num_bcf_terms = self.params.SysP.total_number_bcf_terms

                idxDict = self.params.indices

                if self.params.HiP.result_type == ResultType.ALL:
                    size_aux_state = (idxDict.num_idx() - 1) * self.size_sys
                else:
                    size_aux_state = num_bcf_terms * self.size_sys

            else:
                raise RuntimeError(
                    "Unknown self.params.HiP.result_type: {}".format(
                        self.params.HiP.result_type
                    )
                )

        self.stream_size_aux_states = size_aux_state
        self.stream_sp_num_bcf_terms = num_bcf_terms if self.params.HiP.nonlinear else 0

        idxDict = self.params.indices
        idx_raw = np.concatenate(idxDict.get_first_hierarchy_indices())

        self.first_hier_indices: np.ndarray = np.concatenate(
            [
                np.arange(index * self.size_sys, index * self.size_sys + self.size_sys)
                for index in idx_raw
            ]
        )

    def _open_file(
        self,
        read_only: bool,
        hi_key_bin: bytes,
        hi_key_bin_hash: str,
        overwrite_key: bool,
        backup_if_error: bool = False,
    ):
        if ut.file_does_not_exists_or_is_empty(self.hdf5_name):
            self._init_file(hi_key_bin, hi_key_bin_hash)
        else:
            if not read_only:
                try:
                    p = test_file_version(self.hdf5_name)
                    if p:
                        warnings.warn(
                            "can not check version! process list {} has access to hdf5 file {}".format(
                                p, self.hdf5_name
                            )
                        )

                except Exception as e:
                    if not backup_if_error:
                        raise

                    warnings.warn(
                        "test_file_version FAILED with exception {}".format(e)
                    )
                    warnings.warn("hdf5_name {}".format(self.hdf5_name))

                    backup_name = str(self.hdf5_name) + "backup_" + str(time.time())
                    warnings.warn(
                        f"Moving {self.hdf5_name} to {backup_name} and starting fresh."
                    )

                    shutil.move(
                        self.hdf5_name, self.hdf5_name + "backup_" + str(time.time())
                    )
                    self._init_file(hi_key_bin, hi_key_bin_hash)

        if read_only:
            self.h5File = h5py.File(self.hdf5_name, "r", swmr=True, libver="latest")

        else:
            try:
                self.h5File = h5py.File(self.hdf5_name, "r+", libver="latest")
                if overwrite_key:
                    self._write_hi_key(self.h5File, hi_key_bin, hi_key_bin_hash)

            except OSError:
                print("FAILED to open h5 file '{}'".format(self.hdf5_name))
                raise
            try:
                self.h5File.swmr_mode = True
            except:
                print(f"Can't open {self.hdf5_name} in swmr mode.")
                raise

    def _check_consistency(self, hi_key_bin: bytes, hi_key_bin_hash: str):
        if hi_key_bin_hash != self.h5File.attrs["hi_key_bin_hash"]:  # consistency check
            log.info(f"binkeyhi_key_bin_hash_hash passed {hi_key_bin_hash}")
            log.info(f"hi_key_bin_hash from h5 {self.h5File.attrs['hi_key_bin_hash']}")
            self.h5File.close()
            raise RuntimeError(
                "cannot open file '{}' (hi_key_bin_hash mismatch)".format(
                    str(self.hdf5_name)
                )
            )

        hkb = self.h5File["hi_key_bin"]
        if isinstance(hkb, Dataset):
            hkb = bytearray(np.array(hkb[:]))
            if hi_key_bin != hkb:  # consistency check
                self.h5File.close()
                raise RuntimeError(
                    "cannot open file '{}' (hi_key_bin mismatch)".format(
                        str(self.hdf5_name)
                    )
                )

    def _get_with_type_check(self, key: str) -> h5py.Dataset:
        """Tries to get the dataset with key ``key`` from :any:`h5File`.

        Raises a :any:`RuntimeError` if the object found under ``key``
        is not a :any:`h5py.Dataset`.
        """
        tmp = self.h5File[key]

        if not isinstance(tmp, h5py.Dataset):
            raise RuntimeError(
                f'Expected thing under key "{key}" to be a Dataset but got {type(tmp)}.'
            )

        return tmp

    def _load_file_contents(self):
        try:
            self.stoc_traj = self._get_with_type_check("/stoc_traj")
            self._rho_t_accum = self._get_with_type_check("rho_t_accum")
            self._rho_t_accum_m2 = self._get_with_type_check("rho_t_accum_m2")
            self.rho_t_accum_part = self._get_with_type_check("/rho_t_accum_part")
            self.rho_t_accum_part_tracker = self._get_with_type_check(
                "/rho_t_accum_part_tracker"
            )

            self.tracker = self._get_with_type_check("/tracker")
            if self.save_therm_rng_seed:
                self.rng_seed = self._get_with_type_check("/rng_seed")

            if self.size_aux_states != 0:
                self.aux_states = self._get_with_type_check("/aux_states")
            else:
                self.aux_states = None
                if "aux_states" in self.h5File:
                    raise TypeError(
                        "HIData with aux_states=0 finds h5 file with /aux_states"
                    )
            if self.sp_num_bcf_terms != 0:
                self.stoc_proc = self._get_with_type_check("/stoc_proc")
            else:
                self.stoc_proc = None
                if "stoc_proc" in self.h5File:
                    raise TypeError(
                        "HIData init FAILED: num_bcf_terms=0 but h5 file {} has /stoc_proc".format(
                            self.hdf5_name
                        )
                    )

            self.time = self._get_with_type_check("/time")

        except KeyError:
            print("KeyError in hdf5 file '{}'".format(self.hdf5_name))
            raise

    def _write_hi_key(
        self,
        h5File: h5py.File,
        hi_key_bin: bytes,
        hi_key_bin_hash: str,
    ):
        """Write the hierarchy configuration to the HDF5 file,
        overwriting the configuration if it is already present.
        """

        if hi_key_bin_hash is None:
            raise RuntimeError(
                "hi_key_bin_hash must not be none when setting up the hdf5 file for HIData"
            )
        h5File.attrs["hi_key_bin_hash"] = hi_key_bin_hash

        if hi_key_bin is None:
            raise RuntimeError(
                "hi_key_bin must not be none when setting up the hdf5 file for HIData"
            )
        hi_key_bin = bytearray(hi_key_bin)

        if "hi_key_bin" in h5File:
            del h5File["hi_key_bin"]

        data = h5File.create_dataset("hi_key_bin", (len(hi_key_bin),), dtype=np.uint8)
        data[:] = np.asarray(hi_key_bin, dtype=np.uint8)

        pickled_hi_key = bytearray(pickle.dumps(self.params))

        if "pickled_hi_key" in h5File:
            del h5File["pickled_hi_key"]

        data = h5File.create_dataset(
            "pickled_hi_key", (len(pickled_hi_key),), dtype=np.uint8
        )
        data[:] = np.asarray(pickled_hi_key, dtype=np.uint8)

    def _init_file(self, hi_key_bin: bytes, hi_key_bin_hash: str):
        """Initialize the HDF5 file by setting its attributes and
        creating the datasets if required.
        """

        with h5py.File(self.hdf5_name, "w", libver="latest") as h5File:
            self._write_hi_key(h5File, hi_key_bin, hi_key_bin_hash)

            if not self.accum_only:
                size_stoc_traj = HIData_default_size_stoc_traj
            else:
                # need at least one stoch traj to show k_max convergence
                size_stoc_traj = 1

            # size_stoc_traj may be overwritten to account for accum_only
            h5File.create_dataset(
                "stoc_traj",
                (size_stoc_traj, self.size_t, self.size_sys),
                dtype=np.complex128,
                maxshape=(None, self.size_t, self.size_sys),
                chunks=(1, self.size_t, self.size_sys),
            )

            h5File.create_dataset(
                "rho_t_accum",
                (self.size_t, self.size_sys, self.size_sys),
                dtype=np.complex128,
            )

            h5File.create_dataset(
                "rho_t_accum_m2",
                (self.size_t, self.size_sys, self.size_sys),
                dtype=np.float128,
            )

            h5File.create_dataset(
                "rho_t_accum_part",
                (
                    HIData_default_size_rho_t_accum_part,
                    self.size_t,
                    self.size_sys,
                    self.size_sys,
                ),
                dtype=np.complex128,
                maxshape=(None, self.size_t, self.size_sys, self.size_sys),
                chunks=(1, self.size_t, self.size_sys, self.size_sys),
            )

            h5File.create_dataset(
                "rho_t_accum_part_tracker",
                data=HIData_default_size_rho_t_accum_part * [False],
                dtype="bool",
                maxshape=(None,),
            )

            h5File.create_dataset(
                "tracker",
                data=HIData_default_size_stoc_traj * [False],
                dtype="bool",
                maxshape=(None,),
            )

            if self.size_aux_states != 0:
                # size_stoc_traj may be overwritten to account for accum_only
                h5File.create_dataset(
                    "aux_states",
                    (size_stoc_traj, self.size_t, self.size_aux_states),
                    dtype=np.complex128,
                    maxshape=(None, self.size_t, self.size_aux_states),
                    chunks=(1, self.size_t, self.size_aux_states),
                )

            if self.sp_num_bcf_terms != 0:
                # size_stoc_traj may be overwritten to account for accum_only
                h5File.create_dataset(
                    "stoc_proc",
                    (size_stoc_traj, self.size_t, self.sp_num_bcf_terms),
                    dtype=np.complex128,
                    maxshape=(None, self.size_t, self.sp_num_bcf_terms),
                    chunks=(1, self.size_t, self.sp_num_bcf_terms),
                )

            if self.save_therm_rng_seed:
                h5File.create_dataset(
                    "rng_seed",
                    (size_stoc_traj,),
                    dtype=np.int64,
                    maxshape=(None,),
                    chunks=(1,),
                )

            h5File.create_dataset("time", (self.size_t,), dtype=np.float64)

    ###########################################################################
    #                              Functionality                              #
    ###########################################################################

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        self.close()

    def close(self):
        """Close all resources owned."""

        if hasattr(self, "h5File"):  # maybe hasn't been initialized yet
            self.h5File.close()

        if (
            hasattr(self, "_stream_fifo")
            and self._stream_fifo is not None
            and self.stream_file is not None
            and not self._stream_fifo.closed
        ):
            self._stream_fifo.close()

    def has_sample(self, idx: int) -> bool:
        """Returns whether the sample with the index ``idx`` has been
        processed."""
        if idx < self._idx_cnt:
            return bool(self.tracker[idx])
        else:
            return False

    @property
    def largest_idx(self) -> int:
        """The largest sample index yet received."""
        return self.h5File.attrs.get("largest_idx", 0)

    def _set_largest_idx(self, idx: int):
        """Set the largest sample index yet received."""

        self.h5File.attrs["largest_idx"] = idx

    @property
    def samples(self) -> int:
        """The number of samples processed."""
        return self.h5File.attrs.get("samples", 0)

    def _set_samples(self, samples: int):
        """Set the number of samples processed."""

        self.h5File.attrs["samples"] = samples

    @property
    def time_set(self) -> bool:
        """Tells whether :any:`time` has been set to a meaningful value."""

        return self.time.attrs.get("time_set", False)

    def _set_time_set(self, time_set: bool):
        """Sets :any:`time_set`."""
        if not self.time:
            raise RuntimeError(
                "Trying to set attribute of HIData.time before it has been initialized!"
            )

        self.time.attrs["time_set"] = time_set

    def _resize(self, size: int):
        """Resize the first dimension of all sample tracking datasets to
        ``size``."""

        self.tracker.resize(size=(size,))
        if not self.accum_only:
            self.stoc_traj.resize(size=(size, self.size_t, self.size_sys))

            if self.aux_states is not None:
                self.aux_states.resize(size=(size, self.size_t, self.size_aux_states))
            if self.stoc_proc is not None:
                self.stoc_proc.resize(size=(size, self.size_t, self.sp_num_bcf_terms))

            if self.save_therm_rng_seed and self.rng_seed:
                self.rng_seed.resize(size=(size,))

        self._idx_cnt = size

    def _inc_size(self, idx: int):
        """Resize the first dimension off all sample tracking datasets to fit
        the index ``idx``."""
        if self._idx_cnt <= idx:
            new_idx_cnt = 2 * self._idx_cnt
            while new_idx_cnt <= idx:
                new_idx_cnt *= 2

            self._resize(new_idx_cnt)

    def _resize_rho_t_accum_part(self, size: int):
        """Resize :any:`rho_t_accum_part` and :any:`rho_t_accum_part_tracker`
        to ``size``."""

        self.rho_t_accum_part_tracker.resize(size=(size,))
        self.rho_t_accum_part.resize(
            size=(size, self.size_t, self.size_sys, self.size_sys)
        )
        self._idx_rho_t_accum_part_tracker_cnt = size

    def _inc_rho_t_accum_part_size(self, n: int):
        """Resize :any:`rho_t_accum_part` and :any:`rho_t_accum_part_tracker`
        to fit at least ``n`` samples."""

        if self._idx_rho_t_accum_part_tracker_cnt <= n:
            new_idx_cnt = (
                self._idx_rho_t_accum_part_tracker_cnt
                + HIData_default_size_rho_t_accum_part
            )
            while new_idx_cnt <= n:
                new_idx_cnt += HIData_default_size_rho_t_accum_part

            self._resize_rho_t_accum_part(new_idx_cnt)

    def has_rho_t_accum_part(self, n) -> bool:
        """Returns whether the sample number ``n`` has been added to
        :any:`rho_t_accum_part`."""

        if n < self._idx_rho_t_accum_part_tracker_cnt:
            return bool(self.rho_t_accum_part_tracker[n])
        else:
            return False

    def set_time(self, time, force=False) -> bool:
        """Set the time points stored in :any:`time` to ``time``.

        If ``force`` is :any:`True` the time will be set even if it
        already has a value.

        :returns: Whether the time has been set.
        """

        if (not self.time_set) or force:
            self.time[:] = time
            self._set_time_set(True)

            return True

        return False

    def get_time(self) -> np.ndarray:
        """Get the time points of the simulation if they have been set.

        This can be checked with :any:`time_set`.
        """

        if not self.time_set:
            raise RuntimeError("Can't get time, time has not been set yet.")
        return self.time[:]  # type: ignore

    @property
    def result_filter(self) -> ResultFilter:
        """
        A :any:`ResultFilter` that is compatible with the instance.
        """

        return functools.partial(
            filter_psi_all,
            self.size_sys,
            self.size_aux_states,
            self.size_t,
            self.first_hier_indices
            if self.result_type == ResultType.ZEROTH_AND_FIRST_ORDER
            else None,
            self.result_type,
        )

    @property
    def stream_result_filter(self) -> Optional[ResultFilter]:
        """
        A :any:`ResultFilter` that is compatible with the instance for
        streaming results.
        """

        if self.stream_result_type is None or self._stream_fifo is None:
            return None

        return functools.partial(
            filter_psi_all,
            self.size_sys,
            self.stream_size_aux_states,
            self.size_t,
            self.first_hier_indices
            if self.stream_result_type == ResultType.ZEROTH_AND_FIRST_ORDER
            else None,
            self.stream_result_type,
        )

    def new_samples(
        self,
        idx: int,
        incomplete: bool,
        psi0: np.ndarray,
        aux_states: Optional[np.ndarray],
        stoc_proc: Optional[np.ndarray],
        result_type: ResultType,
        normed: bool,
        rng_seed: Optional[int] = None,
    ):
        """Stores a new stochastic trajectory and add it to the
        accumulation.

        .. note::

            It will not be added to the accumulation
            in this case.


        :param idx: the index of the trajectory
        :param incomplete: whether the sample is incomplete

            Incomplete samples aren't accumulated into the density
            matrix.  :any:`has_sample` will return :any:`False` if
            queried with ``idx`` in this case.

        :param psi0: the zeroth order stochastic trajectory ``(time,
            system dimension)``
        :param aux_states: the auxiliary states that are to be saved
            (if any)
        :param stoc_proc: the stochastic process shift (if it is to be
            stored)
        :param result_type: the result type of the trajectory

            .. note::

                This argument is used as a consitency check and an error
                is raised if it differs from :any:`result_type`, which has been
                set from ``hi_key`` in ``__init__``.

        :param normed: whether the trajectory has to be normalized

            This is the case if it is not normalized by hand or comes
            from the linear method.

        :param rng_seed: the seed of the random number generator used
            to generate the thermal stochasitc procsess.

            .. note::

                This is only used if :class:`HIParams.HiP.save_therm_rng_seed`
                was true when the instance of :any:`HIData` was initialized.
        """

        if self.result_type != result_type:
            raise RuntimeError(
                f"""
                Wrong result type received.
                Wanted {self.result_type} and got {result_type}.
                """
            )

        if np.isnan(np.sum(psi0)):
            warnings.warn(f"Trajectory {idx} contains NaN.")

        if incomplete:
            warnings.warn("Storing incomplete data.")

        with signal_delay.sig_delay(
            [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1],
            lambda _: self.h5File.flush(),
        ):
            self._inc_size(idx)

            # if the sample is incomplete, we store it for now, but won't
            # add it to the accumulation

            if self.tracker[idx]:
                warnings.warn("sample {} has already been added!")
                return

            if (not self.accum_only) or (idx == 0):
                self.stoc_traj[idx] = psi0

                if aux_states is not None:
                    if self.aux_states is None:
                        raise RuntimeError(
                            "Can't write the auxiliary states because the dataset doesn't exist."
                        )

                    self.aux_states[idx] = aux_states

                if stoc_proc is not None and self.stoc_proc is not None:
                    self.stoc_proc[idx] = stoc_proc

                if self.save_therm_rng_seed and self.rng_seed:
                    self.rng_seed[idx] = rng_seed

            n = ut.is_int_power(self.samples + 1, b=2)
            if n is not None:
                self._inc_rho_t_accum_part_size(n)

            if not incomplete:
                self.rho_t_accum.update(hm.projector_psi_t(psi0, normed=normed))

                self._set_samples(self.rho_t_accum.n)

            if n is not None:
                self.rho_t_accum_part_tracker[n] = not incomplete

                if not incomplete:
                    self.rho_t_accum_part[n] = self.get_rho_t()

            self.tracker[idx] = not incomplete
            self._set_largest_idx(max(int(self.largest_idx), idx))

    def stream_samples(
        self,
        idx: int,
        incomplete: bool,
        psi0: np.ndarray,
        aux_states: Optional[np.ndarray],
        stoc_proc: Optional[np.ndarray],
        result_type: ResultType,
        rng_seed: Optional[int] = None,
    ):
        """Stream a new sample into :any:`stream_file`.

        See :any:`new_samples` for the arguments.
        """

        if self.stream_result_type is None or self._stream_fifo is None:
            return

        assert self._stream_fifo is not None

        if self.stream_result_type != result_type:
            raise RuntimeError(
                f"""
                Wrong result type received.
                Wanted {self.stream_result_type} and got {result_type}.
                """
            )

        if incomplete:
            log.info(f"Not streaming result {idx} because it is incomplete.")
            return

        with signal_delay.sig_delay(
            [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1],
            lambda _: self._stream_fifo.close(),
        ):
            pickle.dump(
                (idx, psi0, aux_states, stoc_proc, result_type, rng_seed),
                self._stream_fifo,
            )

    def get_rho_t(self, res: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the density matrix of the system or loads it into ``res``
        which is then returned.

        You may also want to check out :any:`rho_t_accum` for access to
        the variance of the matrix elments.

        Has the shape ``(time step, system dimension, system
        dimension)``.

        :param res: Load result in the given numpy array, defaults to None
        :return: The density matrix of the system rho_tij
        """

        if res is None:
            res = np.empty(
                shape=(self.size_t, self.size_sys, self.size_sys), dtype=np.complex128
            )

        self.rho_t_accum.dataset.read_direct(dest=res)
        return res

    def get_rho_t_part(self, n: int) -> np.ndarray:
        """Returns the density matrix of the system after ``n`` samples have
        been processed.

        Has the shape ``(time step, system dimension, system
        dimension)``.
        """
        if self.has_rho_t_accum_part(n):
            return np.array(self.rho_t_accum_part[n])
        else:
            raise RuntimeError(
                "rho_t_accum_part with index {} has not been crunched yet".format(n)
            )

    def no_samples(self) -> bool:
        """Returns :any:`True` if there are no samples stored and :any:`False`
        otherwise."""

        return self.samples == 0

    def clear(self):
        """Clear all data.

        This should be equivalent to deleting the underlying HDF5 file.
        """

        self._resize(HIData_default_size_stoc_traj)
        self._resize_rho_t_accum_part(HIData_default_size_rho_t_accum_part)
        self._set_time_set(False)
        self.rho_t_accum.reset()
        self._set_samples(0)
        self.tracker[:] = False
        self.rho_t_accum_part_tracker[:] = False
        self._set_largest_idx(0)

    def get_therm_rng_seed(self, idx: int) -> int:
        """Get the RNG seed of the thermal stochastic process for index
        ``idx``."""

        if not self.save_therm_rng_seed or not self.rng_seed:
            raise RuntimeError("Thermal rng seeds haven't been saved.")

        if not self.has_sample(idx):
            raise RuntimeError(f"Haven't received sample {idx} yet.")

        return int(self.rng_seed[idx])  # type: ignore

    def get_stoc_traj(
        self, idx: int, incomplete: bool = False, normed: bool = False
    ) -> np.ndarray:
        """Get the stochastic trajectory (0th hierarchy depth) for the index
        ``idx``.

        Has the shape ``(time, dim H_sys)``.

        :param idx: The index of the returned trajectory
        :param incomplete: If :any:`True` potentially incomplete trajectories
            are returned
        :param normed: Whether the trajectory state should be normalized
        :return: The stochastic trajectory psi_ti
        """

        if idx > 0 and self.accum_only:
            raise ValueError(
                "Accumulation only mode is activated (`accum_only` is True). No samples beyond index zero being are stored."
            )

        if self.has_sample(idx) or incomplete:
            psi_z_t: np.ndarray = self.stoc_traj[idx]  # type: ignore
            if normed:
                n_t = hm.norm_psi_t(psi_z_t)
                psi_z_t /= n_t.reshape(-1, 1)
            return psi_z_t  # type: ignore
        else:
            raise RuntimeError(f"Haven't received sample {idx} yet.")

    def get_aux_states(self, idx: int, normed: bool = False) -> np.ndarray:
        """Get the auxiliary states for index ``idx``. See :any:`aux_states`.

        :param idx: Index of the returned trajectory.
        :param normed: Whether the trajectory should by normalized with the
            zeroth hierarchy, defaults to False.
        :return: The stochastic trajectory with auxiliary states psi_ti.
        """
        if self.has_sample(idx):
            if not self.aux_states:
                raise RuntimeError(
                    f"This instance of {__class__} isn't configured to store the auxiliary states."
                )

            states = self.aux_states[idx]
            if normed:
                assert self.stoc_traj

                psi_z_t: np.ndarray = self.stoc_traj[idx]  # type: ignore
                n_t = hm.norm_psi_t(psi_z_t)
                states /= n_t.reshape(-1, 1)

            return states  # type: ignore
        else:
            raise RuntimeError(f"Haven't received sample {idx} yet.")

    def get_rng_seed(self, idx: int) -> np.ndarray:
        """Get the RNG seed for the thermal process used to compute the
        trajectory with the index ``idx``."""

        if self.save_therm_rng_seed and self.rng_seed:
            return self.rng_seed[idx]  # type: ignore

        raise RuntimeError("The thermal trajectory seeds have not been saved.")

    def get_sub_rho_t(
        self, idx_low: int, idx_high: int, normed: bool, overwrite=False
    ) -> tuple[np.ndarray, int]:
        """Returns the system denisty matrix computed using the trajectories with the
        indices ``idx_low`` through ``idx_high``, optionally normalized if ``normed`` is
        :any:`True` and the number count that went into calculating the density matrix.

        The result is cached. The cache will be ignored and overwritten if ``overwrite`` is
        :any:`True`.

        :todo: use a welford aggregator as well!
        """

        name = "{}_{}".format(int(idx_low), int(idx_high))
        if overwrite and name in self.h5File:
            del self.h5File[name]

        if not name in self.h5File:
            smp: int = 0
            rho_t_accum = np.zeros(
                shape=(self.size_t, self.size_sys, self.size_sys), dtype=np.complex128
            )
            for i in range(idx_low, idx_high):
                if self.has_sample(i):
                    smp += 1
                    rho_t_accum += hm.projector_psi_t(self.stoc_traj[i], normed=normed)  # type: ignore
            rho_t = rho_t_accum / smp
            h5_data = self.h5File.create_dataset(
                name,
                shape=(self.size_t, self.size_sys, self.size_sys),
                dtype=np.complex128,
            )
            h5_data[:] = rho_t
            h5_data.attrs["smp"] = smp
        else:
            rho_t = np.empty(
                shape=(self.size_t, self.size_sys, self.size_sys), dtype=np.complex128
            )
            self._get_with_type_check(name).read_direct(dest=rho_t)
            smp = self._get_with_type_check(name).attrs["smp"]  # type: ignore

        return rho_t, smp

    def rewrite_rho_t(self, normed: bool):
        """Recompute the system density matrix and optionally normalize it if ``normed``
        :any:`True`.

        .. note::

            This doesn't work if :any:`accum_only` is :any:`True`.
            An error will be raised in this case.
        """

        if self.accum_only:
            raise ValueError(
                "Accumulation only mode is enabled wich means that there are no samples stored."
            )

        self.rho_t_accum.reset()
        with signal_delay.sig_delay(
            [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1]
        ):
            for i in range(self.largest_idx + 1):
                if self.has_sample(i):
                    self.rho_t_accum.update(
                        hm.projector_psi_t(self.stoc_traj[i], normed=normed)  # type: ignore
                    )

            self._set_samples(self.rho_t_accum.n)

    def invalidate_samples(self, slc: Union[slice, np.ndarray, int]) -> None:
        """Invalidate the samples indexed by ``slc``.

        This removes them from the tracker, so that they will be
        recomputed.  The state will also be recomputed using
        :any:`rewrite_rho_t`.
        """

        sample_range = self.tracker[slc]
        for i, _ in enumerate(sample_range):
            sample_range[i] = False

        self.rewrite_rho_t(self.get_hi_key().HiP.nonlinear)

    def get_hi_key(self) -> HIParams:
        """Returns the configuration that was used to initialize the
        instance."""

        pickled_hi_key = self._get_with_type_check("pickled_hi_key")
        return pickle.loads(bytearray(pickled_hi_key[:]))  # type: ignore

    def get_hi_key_hash(self) -> str:
        """Returns the hash of the configuration that was used to initialize
        the instance."""

        return str(self.h5File.attrs["hi_key_bin_hash"])

    def valid_sample_iterator(self, iterator: Iterator[T]) -> Iterator[T]:
        """
        Takes an ``iterator`` that yields a sequence of items related to
        the sequence of samples and yields them if the sample is
        actually present in the data.
        """

        for i, item in enumerate(iterator):
            if (
                self.has_sample(i)
                and not np.isnan(np.sum(self.stoc_traj[i]))
                and not np.isnan(np.sum(self.aux_states[i]))
            ):
                yield item


class HIMetaData:
    """A helper object to object to create :any:`HIData` instances by
    specifying a name and path.

    :param hid_name: the name of the database to be opened or created
    :param hid_path: the path under which the database is or will be located
    """

    def __init__(self, hid_name: str, hid_path: str):
        self.name = hid_name
        self.hid_path = hid_path
        self.path = pathlib.Path(hid_path) / self.name

        if self.path.exists():
            if not self.path.is_dir:
                raise NotADirectoryError(
                    "the path '{}' exists but is not a directory".format(self.path)
                )
        else:
            self.path.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def hash_bin_key(binkey: bytes) -> str:
        """Get the hash of the binary key (being the binary digest of a
        :any:`HIParams` object, see ``binfootprint``).

        See also :any:`get_hashed_key`.
        """

        return hashlib.md5(binkey).hexdigest()

    @staticmethod
    def get_hashed_key(key: HIParams) -> str:
        """Get the hash of a :any:`HIParams` object, see
        ``binfootprint``."""

        return HIMetaData.hash_bin_key(bf.dump(key))  # type: ignore
        # TODO: (Valentin) binfootprint typing

    def get_HIData_fname(
        self, key: HIParams, ret_bin_and_hash: bool = False
    ) -> Union[str, tuple[str, bytes, str]]:
        """Get the filename of the HDF5 file for the ``key``.

        Optionally return the binary representation and hash of ``key``
        as well.
        """

        # TODO: (Valentin) move bin/hash stuff to binfootprint

        if key is None:
            raise ValueError("key must not be None!")
        bin_key: bytes = bf.dump(key)  # type: ignore
        # TODO: (Valentin) typing in binfootprint
        hashed_key = self.hash_bin_key(bin_key)
        sub_dir = "_" + hashed_key[0]
        sub_p = self.path / sub_dir
        sub_p.mkdir(exist_ok=True)

        idx = 0
        while True:
            idx += 1
            hdf5_name = (
                sub_dir + "/" + self.name + "_" + hashed_key + "_" + str(idx) + ".h5"
            )
            hdf5_p = self.path / hdf5_name
            if not hdf5_p.exists():
                # no file with proposed name
                break

            if hdf5_p.stat().st_size == 0:
                # proposed file is empty
                break

            try:
                with h5py.File(
                    str(self.path / hdf5_name), "r", swmr=True, libver="latest"
                ) as h5File:
                    hkb = h5File["hi_key_bin"]
                    hkb = bytearray(hkb[:])  # type: ignore
                    if hkb == bin_key:  # this is the file we are looking for!
                        break
            except Exception:
                print(
                    "hi_key_bin could not be read from file", str(self.path / hdf5_name)
                )
                raise

        if not ret_bin_and_hash:
            return hdf5_name
        else:
            return hdf5_name, bin_key, hashed_key

    def get_HIData(
        self,
        key: HIParams,
        read_only: bool = False,
        overwrite_key: bool = False,
        robust: bool = True,
        stream_file: Optional[str] = None,
    ) -> HIData:
        """Returns a :any:`HIData` instance initialized with ``key`` and an
        auto-generated file name.

        For the arguments see :any:`hops.core.hierarchy_data.HIData`.
        """

        hdf5_name, bin_key, hashed_key = self.get_HIData_fname(
            key, ret_bin_and_hash=True
        )

        assert isinstance(bin_key, bytes)
        return HIData(
            str(self.path / hdf5_name),
            read_only=read_only,
            hi_key=key,
            hi_key_bin=bin_key,
            hi_key_bin_hash=hashed_key,
            overwrite_key=overwrite_key,
            robust=robust,
            stream_file=stream_file,
        )


def test_file_version(hdf5_name: str):
    """Test whether the HDF5 file has the correct version and upgrade it if
    required.

    :returns: a list of processes accessing the file or ``False`` if nothing needs to be done.
    """

    p = ut.get_processes_accessing_file(hdf5_name)
    if len(p) > 0:
        # another process accesses the file, assume that the file has allready the new format,
        # since that other process has already changed it
        return p

    with h5py.File(hdf5_name, "r+", libver="latest") as h5File:
        # print("test file, open", hdf5_name, "'r+")
        try:
            # print("test file, try to set swmr_mode True")
            h5File.swmr_mode = True
            # print("test file,set swmr_mode=True on file", hdf5_name)
            # print("test file,", h5File)
        except ValueError as e:
            print("got Value Error with msg '{}'".format(e))
            h5File.close()
            change_file_version_to_latest(hdf5_name)
        except:
            raise
    return False


def change_file_version_to_latest(h5fname: str):
    """Upgrate the HDF5 file under ``h5fname`` to the latest HDF5 version."""

    log.info("change file version to 'latest'")

    pid_list = ut.get_processes_accessing_file(h5fname)
    if len(pid_list) > 0:
        raise RuntimeError(
            "can not change file version! the following processes have access to the file: {}".format(
                pid_list
            )
        )

    rand_fname = ut.get_rand_file_name()
    with h5py.File(rand_fname, "w", libver="latest") as f_new:
        with h5py.File(h5fname, "r") as f_old:
            for i in f_old:
                f_old.copy(
                    i,
                    f_new["/"],
                    shallow=False,
                    expand_soft=False,
                    expand_external=False,
                    expand_refs=False,
                    without_attrs=False,
                )
            for k, v in f_old.attrs.items():
                f_new.attrs[k] = v
    print("updated h5 file {} to latest version".format(os.path.abspath(h5fname)))

    shutil.move(h5fname, h5fname + rand_fname + ".old")
    shutil.move(rand_fname, h5fname)
    os.remove(h5fname + rand_fname + ".old")

    assert not os.path.exists(rand_fname)


def migrate_wrongly_named(params: list[HIParams], names: list[str], meta: HIMetaData):
    """
    Migrate the hierarchy data files in ``names`` so that they are
    consistent with the configurations in ``params`` and ``HIMetaData``.
    """

    for param, name in zip(params, names):
        d = HIData(
            name, True, hi_key=param, overwrite_key=True, check_consistency=False
        )
        d.close()

        new_path = str(meta.path / meta.get_HIData_fname(param))
        print(f"Moving `{name}` to `{new_path}`.")
        shutil.move(name, new_path)

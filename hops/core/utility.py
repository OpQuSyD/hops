"""Miscellaneous functionality used in the other modules."""
import numpy as np
import os
import h5py
from typing import Optional
from collections.abc import Callable
from functools import wraps
import subprocess
import logging
import time
import stocproc as sp

log = logging.getLogger(__name__)


def is_int_power(x: float, b: int = 2) -> Optional[int]:
    """Returns ``n`` if ``x`` is ``b ** n`` and None otherwise."""
    n_float = np.log(x) / np.log(b)
    n = int(n_float)
    if b**n == x:
        return n
    else:
        return None


def file_does_not_exists_or_is_empty(fname: str) -> bool:
    """Returns :any:`True` if the file under ``fname`` doesn't exist
    or is empty.
    """
    if not os.path.exists(fname):
        return True
    else:
        if os.path.getsize(fname) == 0:
            return True
        else:
            with h5py.File(fname, "r", swmr=True, libver="latest") as test_f:
                if len(test_f.keys()) == 0:
                    return True
    return False


def get_rand_file_name(l: int = 12, must_not_exist: bool = True) -> str:
    """Generate a random file name of length ``l`` and that ``must_not_exist``."""
    CHAR_SET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    n = len(CHAR_SET)
    while True:
        fname = ""
        for _ in range(l):
            fname += CHAR_SET[np.random.randint(0, n)]
        if (not os.path.exists(fname)) or (not must_not_exist):
            return fname


def get_processes_accessing_file(fname: str) -> list[int]:
    """Get a list of pids of procceses accessing the file ``fname``."""

    cmd = 'timeout 5s lsof "{}"'.format(fname)
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
        encoding="utf8",
    )
    if r.stderr != "":
        log.debug("command '{}' stderr:\n{}".format(cmd, r.stderr))

    if r.returncode == 0:
        # success
        out = r.stdout.split("\n")
        head = out[0].split()
        idx_PID = head.index("PID")
        pid_list = []
        for l in out[1:]:
            l = l.split()
            if len(l) == 0:
                continue

            pid_list.append(int(l[idx_PID]))
        return pid_list
    else:
        # failure, also happens when no process was found
        if r.stdout == "":
            log.debug(
                "lsof has non-zero return code and empty stdout -> assume not process has access to file"
            )

    return []


def uni_to_gauss(x: np.ndarray) -> np.ndarray:
    """Take ``2 * N = len(x)`` uniformly random samples ``x`` and return ``N``
    independent complex gaussian distributed samples."""

    n = len(x) // 2
    phi = x[:n] * 2 * np.pi
    r = np.sqrt(-np.log(x[n:]))

    return r * np.exp(1j * phi)


def time_it(f: Callable) -> Callable:
    """Wraps ``f`` to print its execution time."""

    @wraps(f)
    def wrapper(*args, **kwds):
        t0 = time.perf_counter()
        res = f(*args, **kwds)
        t1 = time.perf_counter()
        print("{}: {:.2e}s".format(f.__name__, t1 - t0))
        return res

    return wrapper


class ZeroProcess(sp.StocProc):
    """A trivial stochastic process that is always zero."""

    def __post_init__(self):
        self.key = "ZERO"
        pass

    def new_process(self, _: np.ndarray):
        pass

    def __call__(self, _):
        return 0

    def __getstate__(self):
        return 0

    def __setstate__(self, _):
        pass

    def get_num_y(self):
        return 0

    def calc_z(self):
        pass

    def set_scale(self, _):
        pass


class TerminalEvent(Exception):
    """
    An exception to signal that an unknown terminal event was
    encountered during integration.
    """

    def __init__(self, result):
        self.result = result

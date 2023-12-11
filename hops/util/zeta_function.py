"""A working (but maybe slow) version of the lerch transcendent.

Taken either from ``arb`` or ``mpmath`` (slow).
"""

import ctypes
import ctypes.util
import logging
import numpy as np
import mpmath


class _complex_double(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    @classmethod
    def from_complex(cls, c: complex):
        return cls(c.real, c.imag)

    def __complex__(self):
        return complex(self.real, self.imag)


try:
    libarb_path = ctypes.util.find_library("arb")
    libarb = ctypes.CDLL(libarb_path)

    _arb_c_lerch_phi = libarb.arb_fpwrap_cdouble_lerch_phi

    def _arb_lerch_phi(z: complex, s: complex, a: complex):
        """The lerch transcendent :math:`ϕ(z, s, a)`."""
        args = [_complex_double.from_complex(complex(arg)) for arg in (z, s, a)]
        cy = _complex_double()

        if _arb_c_lerch_phi(ctypes.byref(cy), *args, 0):
            raise ValueError(f"unable to evaluate function accurately at {z,s,a}")

        return complex(cy)

    lerch_phi = _arb_lerch_phi

except Exception as e:
    logging.info(
        "Using mpmath for the zeta (lerch phi) function. "
        + "Install a recent version of ``arb`` for more performance."
    )

    logging.debug("The error was:", e)

    def _lerch_phi(z, s, a):
        """The lerch transcendent :math:`ϕ(z, s, a)`."""
        return np.complex128(mpmath.lerchphi(z, s, a))

    lerch_phi = _lerch_phi

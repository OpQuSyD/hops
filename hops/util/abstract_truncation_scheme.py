"""
A truncation scheme is a callable that is being called on a :any:`HiIdx`
and returns :any:`True` if the auxilliary state with this index will be integrated.
Otherwise, :any:`False` will be returned.

Truncation schemes can be combined via ``&`` and ``|``.
"""

from abc import ABC, abstractmethod
from hops.core.hi_idx import HiIdx


class TruncationScheme(ABC):
    """The abstract truncation scheme class.

    Prescribes a call operator and defines ``&`` and ``|``
    for truncation schemes.
    """

    def __and__(self, other: "TruncationScheme") -> "TruncationScheme":
        return _TruncationScheme_AND(self, other)

    def __or__(self, other: "TruncationScheme") -> "TruncationScheme":
        return _TruncationScheme_OR(self, other)

    @abstractmethod
    def __call__(self, hi_idx: HiIdx) -> bool:
        pass

    @abstractmethod
    def __bfkey__(self):
        pass


class _TruncationScheme_AND(TruncationScheme):
    def __init__(self, ts1: TruncationScheme, ts2: TruncationScheme):
        self.ts1: TruncationScheme = ts1
        self.ts2: TruncationScheme = ts2

    def __call__(self, hi_idx: HiIdx) -> bool:
        return self.ts1(hi_idx) and self.ts2(hi_idx)

    def __bfkey__(self):
        return ["and", self.ts1, self.ts2]


class _TruncationScheme_OR(TruncationScheme):
    def __init__(self, ts1, ts2):
        self.ts1 = ts1
        self.ts2 = ts2

    def __call__(self, hi_idx: HiIdx) -> bool:
        return self.ts1(hi_idx) or self.ts2(hi_idx)

    def __bfkey__(self):
        return ["or", self.ts1, self.ts2]


class TruncationScheme_Simplex(TruncationScheme):
    """A simplex truncation scheme where the sum of all elements of the multi index
    must not exceed ``kmax``.

    :param kmax: The simplex parameter.
    """

    def __init__(self, kmax: int):
        self.kmax: int = kmax
        """The simplex parameter."""

    def __call__(self, hi_idx: HiIdx) -> bool:
        k = 0
        for i, j in hi_idx.indices():
            k += hi_idx[i, j]

        return k <= self.kmax

    def __bfkey__(self):
        return self.kmax

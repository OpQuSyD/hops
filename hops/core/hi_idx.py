"""A multi index class for use in HOPS."""

from functools import reduce
import numpy as np
from numpy.typing import NDArray
import copy
from typing import Optional, Union
from collections.abc import Iterator


class HiIdx:
    r"""
    An object that represents a list of vectors of multi indices to label
    a specific auxiliary state.

    Instances can be used as dictionary keys. Note that using :any:`to_bin` as key
    instead is equivalent.

    .. code-block:: python

       n_list = [2, 3]
       i = HiIdx(n_list)

       i[0, 0] = 10

       j = HiIdx.from_other(i)
       i[1, 0] = 10

       d = {i: 1, j: 2}

       assert len(d) == 2
       assert d[i] == i
       assert d[i.to_bin()] == i
       assert d[j] == j

    Be aware that the hash function on this type does not take into account
    the ``n_list`` parameter. The same is true, when testing equality to a
    :any:`bytes` object.

    There is one vector of multi indices :math:`\vec{k}_i` per environment.
    The length of each :math:`k_i` is equal to size of the environment,
    i.e. the number of terms in the particular BCF expansion.

    The access / modification can be done directly on this data object or
    the individual mutilindices can be accessed via the ``[]`` operator.

    The constructor creates the zero index vector if ``other_bin`` is not given
    and the index vector corresponding to ``other_bin`` otherwise.


    :param n_list: the size of each environment or just an integer if there is
        only one environment
    :param other_bin: binary representation of HiIdx to copy
    """

    __slots__ = ["_n_list", "_sum_n_list", "_data", "N"]

    #: the :any:`numpy.dtype` for the indices
    _dtype = np.int16

    @classmethod
    def from_other(cls, other: "HiIdx") -> "HiIdx":
        """Create a copy of ``other``."""

        new = copy.copy(other)
        new._data = copy.deepcopy(other._data)

        return new

    @classmethod
    def from_list(cls, other: list[list[int]]) -> "HiIdx":
        """Create a copy of ``other``."""

        n_list = [len(l) for l in other]
        new = cls(n_list)

        new._data = np.array(
            reduce(lambda lst, elem: lst + elem, other, []), dtype=cls._dtype
        )

        return new

    def __getitem__(self, index: Union[int, tuple[int, int]]) -> np.ndarray:
        if isinstance(index, int):
            sum_n = self._sum_n_list[index]
            return self._data[sum_n : sum_n + self._n_list[index]]

        if isinstance(index, tuple) and len(index) == 2:
            bath, i = index
            return self._data[self._sum_n_list[bath] + i]

        raise IndexError("Either index bath or bath and bcf term.")

    def __setitem__(
        self,
        index: Union[int, tuple[int, int]],
        value,
    ):
        if isinstance(index, int):
            sum_n = self._sum_n_list[index]
            self._data[sum_n : sum_n + self._n_list[index]] = value

        if isinstance(index, tuple) and len(index) == 2:
            bath, i = index
            self._data[self._sum_n_list[bath] + i] = value

    def __init__(
        self,
        n_list: Union[list[int], int],
        other_bin: Optional[bytes] = None,
    ):
        #: hierarchy depth for each term in the bcf expansion
        self._n_list: list[int]

        if isinstance(n_list, int):
            #: number of environments
            self.N = 1
            self._n_list = [n_list]
        else:
            self.N = len(n_list)
            self._n_list = n_list

        #: the start indices for the multi-index of each bath
        self._sum_n_list: np.ndarray = np.empty(len(self._n_list), dtype=int)
        self._sum_n_list[0] = 0
        np.cumsum(self._n_list[:-1], out=self._sum_n_list[1:])

        #: the index vector
        self._data: np.ndarray

        if other_bin:
            self._data = np.frombuffer(other_bin, dtype=self._dtype)

        else:
            self._data = np.zeros(shape=(sum(self._n_list),), dtype=self._dtype)

    def __hash__(self):
        return self.to_bin().__hash__()

    @property
    def n_list(self):
        """The hierarchy depth for each term in the bcf expansion."""
        return self._n_list

    def get_all_k_np_array(self) -> np.ndarray:
        """
        :returns: all multiindices concatenated as a single list if ints
        """

        return np.copy(self._data)

    @property
    def depth(self) -> int:
        """The largest value of the multi index."""
        return int(np.max(self._data))

    def to_bin(self) -> bytes:
        """
        :returns: the binary representation of all indices concatenated
        """

        return self._data.tobytes()

    def __repr__(self):
        return f"HiIdx({self.n_list}, {self.to_bin()})"

    def indices(self) -> Iterator[tuple[int, int]]:
        """A generator that yields all index coordinates ``(i,j)`` that
        are accessible. The of the index can then be accessed via ``..[i,j]``.
        """

        for i in range(self.N):
            for j in range(self.n_list[i]):
                yield (i, j)

    def to_string(self):
        """:returns: a human-readable representation of the index"""
        s = ""
        sum_n = 0

        for n in self._n_list:
            next_sum_n = sum_n + n
            s += str(self._data[sum_n:next_sum_n])
            sum_n = next_sum_n

        return s

    def __str__(self):
        return self.to_string()

    def __eq__(self, other: "HiIdx"):
        if isinstance(other, bytes):
            return self.to_bin() == other

        if not isinstance(other, HiIdx):
            raise ValueError(f"Can't compare HiIdx with {other.__class__}.")

        return self.to_bin() == other.to_bin() and self._n_list == other._n_list

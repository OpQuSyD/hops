"""Mulit Index handling for use in HOPS by flattening into one dimension via a lookup table."""

import numpy as np
from typing import Union
from collections.abc import Iterator
from _collections_abc import dict_items
from ..util.abstract_truncation_scheme import TruncationScheme, TruncationScheme_Simplex
from .hi_idx import HiIdx

IdxDictKeyType = Union[bytes, HiIdx]
"""The type of the key that retreives an item from the index
lookup table."""

IdxDictType = dict[bytes, int]
"""The type of the index lookup table used by :any:`IdxDict`."""


class IdxDict:
    r"""Create and manage the lookup table (dictionary) for the internal HOPS-vector-
    index.

    Conceptually the stochastic vector has a vector index :math:`\vec{k}` where each component
    corresponds to a term in the BCF expansion and labels the hierarchy depth.

    Because of the irregular shape of :math:`\vec{k}`, it is advantageous to flatten the
    vector index to scalar index. This is done by this class.

    This object can be indexed by :any:`bytes` or :any:`HiIdx` and supports iteration.
    The iterator returns :any:`HiIdx` objects .
    If an iteration through the binary representation of the idx dict is desired you can
    use the :any:`binkeys` method. See also :any:`items` and :any:`binitems`.

    :param n_list: The size :math:`K_i` for each environment.

        For a single enviornment, a single integer can also be given.
    """

    def __init__(self, n_list: Union[list[int], int]):
        self.idx_dict: IdxDictType = {}
        """The index lookup table. Maps binary indices to integers.

        See also :any:`HiIdx`.
        """

        #: hierarchy depth for each term in the bcf expansion
        self.n_list: list[int]

        if isinstance(n_list, int):
            #: number of terms in the bcf expansion
            self.N = 1

            self.n_list = [n_list]
        else:
            self.N = len(n_list)
            self.n_list = n_list

    def _clear(self):
        """Clear the index lookup table."""

        self.idx_dict = {}

    def __iadd__(self, hi_idx: HiIdx):
        self.add_new_idx(hi_idx)

        return self

    def __iter__(self) -> Iterator[HiIdx]:
        for bin in self.idx_dict.__iter__():
            yield self.bin_to_idx(bin)

    def binkeys(self) -> Iterator[bytes]:
        """An iterator that yields the binary keys of the lookup table."""

        return self.idx_dict.__iter__()

    def items(self) -> Iterator[tuple[HiIdx, int]]:
        """An iterator that yields pairs of :any:`HiIdx` and their associated
        scalar index."""

        for key, val in self.idx_dict.items():
            yield self.bin_to_idx(key), val

    def binitems(self) -> "dict_items[bytes, int]":
        """An iterator that yields pairs of the binary representation of the index as
        :any:`bytes` and their associated scalar index."""

        return self.idx_dict.items()

    def add_new_idx(self, hi_idx: HiIdx) -> bool:
        """Add a new index to dictionary. The assigned HOPS-vector index is the length of the
        previous index dictionary. You can use ``+=`` for the same purpose.

        If the index vector is already contained, nothing happens and :any:`False` is returned.

        :param hi_idx: the index vector to add
        :returns: :any:`True` if ``hi_idx`` was actually added and :any:`False` otherwise
        """

        hi_idx_bin = hi_idx.to_bin()
        if hi_idx_bin not in self.idx_dict:
            self.idx_dict[hi_idx_bin] = len(self.idx_dict)
            return True

        return False

    def __getitem__(self, hi_idx: IdxDictKeyType):
        return self.get_idx(hi_idx)

    def __contains__(self, item: IdxDictKeyType):
        return item in self.idx_dict

    def get_idx(self, hi_idx: IdxDictKeyType):
        """Returns the HOPS-vector index for a given index ``hi_idx`` which
        can be both a binary index as well as an instance of :any:`HiIdx`.

        .. note::

            You can also use the ``[]`` operator, which is defined to call
            this function.
        """

        return self.idx_dict[hi_idx]  # type: ignore
        # because of the __hash__ and __eq__ implementation in
        # HiIdx, we can index with a HiIdx

    def __len__(self):
        return self.num_idx()

    def _comb_gen(self, hi_idx: HiIdx, truncation_scheme: TruncationScheme):
        """A hepler method to create indices according to a trunction scheme."""
        for i, j in hi_idx.indices():
            new_hi_idx = HiIdx.from_other(hi_idx)
            new_hi_idx[i, j] += 1
            if truncation_scheme(new_hi_idx):
                if self.add_new_idx(new_hi_idx):
                    self._comb_gen(new_hi_idx, truncation_scheme)

    def _fill_holes(self) -> bool:
        """
        A helper that creates indeces so that for every index the
        respective indices one level up do exist.
        """

        added = False
        for hi_idx, _ in list(self.items()):
            for i, size in enumerate(self.n_list):
                for j in range(size):
                    if hi_idx[i, j] == 0:
                        continue

                    hi_idx_to = HiIdx.from_other(hi_idx)
                    hi_idx_to[i, j] -= 1

                    if self.add_new_idx(hi_idx_to):
                        added = True

        return (not added) or self._fill_holes()

    def make(self, truncation_scheme: TruncationScheme) -> "IdxDict":
        """Generate a set of index vectors using a :any:`TruncationScheme`.

        :param truncation_scheme: the truncation scheme to follow
        """

        self._clear()

        hi_idx_0 = HiIdx(n_list=self.n_list)
        self.add_new_idx(hi_idx_0)

        self._comb_gen(hi_idx_0, truncation_scheme)
        self._fill_holes()

        return self

    def make_simplex(self, kmax: int) -> "IdxDict":
        """Generate a set of index vectors using the simplex truncation scheme.

        :param kmax: the simplex parameter
        """

        return self.make(TruncationScheme_Simplex(kmax))

    def num_idx(self) -> int:
        """
        :returns: the number of indices contained in the lookup table.
        """

        return len(self.idx_dict)

    def print_all_idx(self):
        """Prints a representation of the lookup table that associates HOPS index
        vectors with a scalar index."""
        for c in self.idx_dict.keys():
            print(
                HiIdx(n_list=self.n_list, other_bin=c).to_string(),
                " ⟶ ",
                self.idx_dict[c],
            )

    def get_first_hierarchy_indices(self) -> list[np.ndarray]:
        """Returns the scalar indices of the first hirarchy states
        for each environment."""

        indices: list[np.ndarray] = []

        for i, num_terms in enumerate(self.n_list):
            hi_idx = HiIdx(n_list=self.n_list)
            curr_indices = np.empty(num_terms, dtype=int)
            for j in range(num_terms):
                hi_idx[i, j] = 1
                if j > 0:
                    hi_idx[i, j - 1] = 0

                curr_indices[j] = self.idx_dict[hi_idx.to_bin()]

            indices.append(curr_indices)

        return indices

    def bin_to_idx(self, bin: bytes) -> HiIdx:
        """Converts the binary representation ``bin`` to an index."""

        return HiIdx(self.n_list, bin)

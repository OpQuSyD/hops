"""
A simple wrapper and utility functionality around (potentially) time
dependent matrices.  These matrices can be used in algebraic python
expressions (summation, scalar/matrix multiplication) just as usual
numpy arrays.  The implementation is not perforance oriented.
Furthermore, differentiation is implemented for some building blocks.

The abstract interface is defined by :any:`DynamicMatrix`.  The
simplest implementation of that interface is a :any:`ConstantMatrix`,
which isn't time dependent at all and redefines much of the
functionality of :any:`DynamicMatrix` for efficiency.

The :any:`DynamicMatrixList` is a very basic wrapper around a list of
:any:`DynamicMatrix` elements that is used internally to evaluate them
all at once and put the result into an array of appropriate shape.

:any:`ScalarTimeDependence` provides an interface for matrices with a
time dependent scalar pre-factor such as :any:`SmoothStep` and
:any:`Harmonic`.

:any:`Piecewise` supports defining piecewise time dependencies (and
differentiating them), :any:`ScaleTime` scales the time dependence of
a matrix, :any:`Shift` shifts it and :any:`Periodic` makes it periodic.

.. code-block:: python

    A = ConstantMatrix([[1, 2], [3, 2]])
    B = SmoothStep([[1, 0], [1, 1]], 0, 1)
    C = Harmonic([[1, 1], [1, 1]], 1)

    D = .5 * (A @ B) + C
    E = D.derivative()

The motivation behind this design is to provide a unified interface
for time dependent matrix coefficients in HOPS, which is otherwise
agnostic of the time dependence.  Also the machinery behind the
configuration mechanism needs unique keys ``__bfkey__`` to hash the
configuration, ruling out simple functions.
"""
from __future__ import annotations  # postponed evaluation

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal, Union, Optional
from ..core.hierarchy_math import operator_norm
import numpy as np
from numpy.typing import ArrayLike, NDArray
import re
from numbers import Number
import scipy.optimize
import scipy.special
from numpy.polynomial.polynomial import Polynomial

MatrixType = np.ndarray
"""The type of matrices aceepted by the algorithm."""


class DynamicMatrix(ABC):
    """
    An abstract dynamic matrix.  Subtypes must implement the
    ``call`` method, but are otherwise arbitrary.

    Addition, Substraction, (Array and Scalar) Multiplication and
    Matrix Multiplication are implemented, as well as taking the
    hermitian conjugate.
    """

    def __call__(self, t: Union[float, ArrayLike]) -> MatrixType:
        """The value of the matrix at time point ``t``

        Calls :any:`call` but removes the extra time dimension if
        ``t`` is scalar.

        :returns: An array of the shape ``(time points, self.shape)``
                  if ``t`` is an array and otherwise an array of the
                  same shape as :any:`self.shape`.
        """

        is_scalar = np.isscalar(t)
        t_eval = np.reshape(t, 1) if is_scalar else np.asarray(t)

        result = self.call(t_eval)

        return result[0] if is_scalar else result

    @abstractmethod
    def call(self, t: NDArray[np.float64]) -> MatrixType:
        """The value of the matrix at time points ``t``.

        :param t: The time points ``t`` at which the matrix is
                  evaluated.

        :returns: The matrix at times ``t`` as an array of the shape
                  ``(time, ... matrix shape ...)``.
        """

        ...

    @property
    def initial_value(self) -> MatrixType:
        """The initial value of the matrix."""

        return self.__call__(0)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The shape of the matrix.
        """

        return self.initial_value.shape

    def derivative(self) -> DynamicMatrix:
        """The time derivative of the dynamic matrix."""

        return NotImplemented

    @abstractmethod
    def __getstate__(self) -> dict[str, Any]:
        """All the parameters that define the instance.

        Should be so that ``__init__(**parameters)`` works."""

        pass

    def __add__(
        self, other: Union[DynamicMatrix, Literal[0], np.ndarray]
    ) -> DynamicMatrix:
        if isinstance(other, Number) and other == 0:
            return self

        if isinstance(other, np.ndarray):
            return self + ConstantMatrix(other)

        assert isinstance(other, DynamicMatrix)
        return DynamicMatrixSum(self, other)

    __radd__ = __add__

    def __sub__(
        self, other: Union[DynamicMatrix, Literal[0], np.ndarray]
    ) -> DynamicMatrix:
        if isinstance(other, Number) and other == 0:
            return self

        if isinstance(other, np.ndarray):
            return self - ConstantMatrix(other)

        assert isinstance(other, DynamicMatrix)
        return DynamicMatrixDifference(self, other)

    def __mul__(self, other: Union[DynamicMatrix, ArrayLike]) -> DynamicMatrix:
        if isinstance(other, self.__class__):
            return DynamicMatrixProduct(self, other)

        assert not isinstance(other, DynamicMatrix)
        return ScaledDynamicMatrix(self, other)

    __rmul__ = __mul__

    def __matmul__(self, other: DynamicMatrix) -> DynamicMatrix:
        return DynamicMatrixMatrixProduct(self, other)

    @property
    def dag(self) -> "DynamicMatrixDagger":
        return DynamicMatrixDagger(self)

    def operator_norm(self, t: ArrayLike) -> float:
        """
        :returns: The operator norm of the matrix at time(s) ``t``.
        """

        return operator_norm(self.__call__(t))

    def max_operator_norm(self, t_max: float = np.inf, width: float = 1) -> float:
        """
        :returns: The maximum of the operator norm between ``0`` and
                  ``t_max`` as tuple ``time, value``.  The maximum is
                  first located using the brute force method and then
                  refined in an area ``width`` around the initial guess.
        """

        fun = lambda tp: -self.operator_norm(tp)

        max_t = scipy.optimize.brute(fun, [(0, t_max)])[0]
        maximum = -scipy.optimize.minimize_scalar(
            fun, bounds=(max_t - 1, max_t + 1), method="bounded"
        ).fun

        return maximum

    def __setstate__(self, parameters: dict[str, Any]):
        """
        Restore the state from the return value of
        :any:`__getstate__`.
        """

        self.__init__(**parameters)

    def __eq__(self, other: DynamicMatrix) -> bool:
        if self.__class__ != other.__class__:
            return False

        this_state = self.__getstate__()
        other_state = other.__getstate__()

        if set(this_state.keys()) != set(other_state.keys()):
            return False

        for key, val in this_state.items():
            other_val = other_state[key]
            same = val == other_val

            if isinstance(val, np.ndarray):
                if not same.all():
                    return False

            elif not same:
                return False

        return True

    def __repr__(self) -> str:
        rep = (
            f"{self.__class__.__name__}("
            + ", ".join(
                f"{key}={repr(val)}" for key, val in self.__getstate__().items()
            )
            + ")"
        )

        return re.sub(r"\s+", " ", rep)


class DynamicMatrixSum(DynamicMatrix):
    """The sum of two dynamic matrices."""

    def __init__(self, left: DynamicMatrix, right: DynamicMatrix):
        self._left = left
        self._right = right

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return self._left(t) + self._right(t)

    def derivative(self) -> DynamicMatrix:
        return self._left.derivative() + self._right.derivative()

    def __getstate__(self):
        return dict(left=self._left, right=self._right)

    def __repr__(self):
        return f"({self._left.__repr__()} + {self._right.__repr__()})"


class DynamicMatrixDifference(DynamicMatrix):
    """The difference of two dynamic matrices."""

    def __init__(self, left: DynamicMatrix, right: DynamicMatrix):
        self._left = left
        self._right = right

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return self._left(t) - self._right(t)

    def derivative(self) -> DynamicMatrix:
        return self._left.derivative() - self._right.derivative()

    def __getstate__(self):
        return dict(left=self._left, right=self._right)

    def __repr__(self):
        return f"({self._left.__repr__()} - {self._right.__repr__()})"


class DynamicMatrixProduct(DynamicMatrix):
    """The product of two dynamic matrices."""

    def __init__(self, left: DynamicMatrix, right: DynamicMatrix):
        self._left = left
        self._right = right

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return self._left(t) * self._right(t)

    def derivative(self) -> DynamicMatrix:
        return (
            self._left.derivative() * self._right
            + self._left * self._right.derivative()
        )

    def __getstate__(self):
        return dict(left=self._left, right=self._right)

    def __repr__(self):
        return f"({self._left.__repr__()} * {self._right.__repr__()})"


class DynamicMatrixMatrixProduct(DynamicMatrix):
    """The matrix product of two dynamic matrices.

    Implemented with :any:`numpy.matmul`.
    """

    def __init__(self, left: DynamicMatrix, right: DynamicMatrix):
        self._left = left
        self._right = right

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return self._left(t) @ self._right(t)

    def derivative(self) -> DynamicMatrix:
        return (
            self._left.derivative() @ self._right
            + self._left @ self._right.derivative()
        )

    def __getstate__(self):
        return dict(left=self._left, right=self._right)

    def __repr__(self):
        return f"({self._left.__repr__()} @ {self._right.__repr__()})"


class ScaledDynamicMatrix(DynamicMatrix):
    """The dynamic matrix scaled by a factor."""

    def __init__(self, left: DynamicMatrix, right: ArrayLike):
        self._left = left
        self._right = np.asarray(right)

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return self._left(t) * self._right

    def derivative(self) -> DynamicMatrix:
        return self._left.derivative() * self._right

    def __getstate__(self):
        return dict(left=self._left, right=self._right)

    def __repr__(self):
        return f"({self._right.__repr__()} * {self._left.__repr__()})"


class DynamicMatrixDagger(DynamicMatrix):
    """The hermitian conjugate of a matrix."""

    def __init__(self, matrix: DynamicMatrix):
        self._matrix = matrix
        self._shape = matrix.shape

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return np.transpose(self._matrix(t), axes=(0, 2, 1)).conj()

    def derivative(self) -> DynamicMatrix:
        return self.__class__(self._matrix.derivative())

    def __getstate__(self):
        return dict(matrix=self._matrix)

    def __repr__(self):
        return f"({self._matrix.__repr__()}).dag"


class DynamicMatrixList:
    """
    An immutable list of dynamic matrices (of the same shape).

    It is callable itself and returns a numpy array where the first
    index corresponds to the list elements.  The shape
    is inferred from the first element.

    :param lst: The list of matrices.
    """

    __slots__ = ["_lst", "_shape"]

    def __init__(self, lst: list[DynamicMatrix]):
        self._lst = lst
        self._shape = (len(self._lst), *self._lst[0].shape)

        for mat in self._lst:
            if mat.shape != self._shape[1:]:
                breakpoint()
                raise ValueError("All matrices must have the same shape!")

    def derivative(self) -> DynamicMatrixList:
        return self.__class__([m.derivative() for m in self._lst])

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the first element."""
        return self._shape

    @property
    def list(self) -> list[DynamicMatrix]:
        """The underlying list of :any:`DynamicMatrix` elements."""

        return self._lst

    @property
    def len(self) -> int:
        """The number of matrices in the list."""
        return self.shape[0]

    def __call__(self, t: Union[float, NDArray[np.float64]]) -> MatrixType:
        """
        Evaluates all the constituen matrices at the time point(s)
        ``t`` and returns a compound array.
        """

        length, *dims = self._shape

        out = np.empty(
            self._shape if np.isscalar(t) else (length, len(t), *dims),  # type: ignore
            dtype=np.complex128,
        )

        for i, mat in enumerate(self._lst):
            out[i] = mat(t)

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self._lst.__repr__()})"

    def __getitem__(self, i: int) -> DynamicMatrix:
        return self._lst[i]

    def __bfkey__(self):
        return self._lst


###########################################################################
#                             Implementations                             #
###########################################################################


class ConstantMatrix(DynamicMatrix):
    """A constant matrix.

    This implementation is admittedly trivial, but overloads the
    arithmetic for performance.

    :param matrix: The matrix to be wrapped.
    """

    def __init__(self, matrix: Union[ArrayLike, list[list]]):
        self._matrix = np.asarray(matrix)

    def __call__(self, t: ArrayLike) -> MatrixType:
        if np.isscalar(t):
            return self._matrix

        return self.call(np.asarray(t))

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return np.repeat(self._matrix[None, ...], len(t), axis=0)

    @property
    def initial_value(self) -> MatrixType:
        """The initial value of the matrix."""

        return self._matrix

    def derivative(self) -> DynamicMatrix:
        return self.__class__(np.zeros_like(self._matrix))

    def __getstate__(self):
        return dict(matrix=self._matrix)

    def __add__(self, other: Union[DynamicMatrix, Literal[0]]) -> DynamicMatrix:
        if isinstance(other, Number) and other == 0:
            return self

        if isinstance(other, self.__class__):
            return ConstantMatrix(self._matrix + other._matrix)

        assert not isinstance(other, Number)
        return super().__add__(other)

    __radd__ = __add__

    def __sub__(self, other: DynamicMatrix) -> DynamicMatrix:
        if isinstance(other, self.__class__):
            return ConstantMatrix(self._matrix + (-1 * other._matrix))

        return super().__sub__(other)

    def __mul__(self, other: Union[DynamicMatrix, ArrayLike]) -> DynamicMatrix:
        if isinstance(other, self.__class__):
            return ConstantMatrix(self._matrix * other._matrix)

        try:
            array = np.asarray(other)
            return ConstantMatrix(array * self._matrix)
        except:
            return super().__mul__(other)

    __rmul__ = __mul__

    def __matmul__(self, other: DynamicMatrix) -> DynamicMatrix:
        if isinstance(other, self.__class__):
            return ConstantMatrix(self._matrix @ other._matrix)

        return super().__matmul__(other)

    @property
    def dagger(self) -> ConstantMatrix:
        return ConstantMatrix(self._matrix.conj().T)

    def __repr__(self):
        matrep = re.sub(r"\s+", " ", self._matrix.__repr__().replace("\n", ""))
        return f"{self.__class__.__name__}({matrep})"

    def max_operator_norm(self, t_max: float = np.inf) -> float:
        del t_max
        return self.operator_norm(0)

    def __eq__(self, other: DynamicMatrix):
        if isinstance(other, self.__class__):
            return (self._matrix == other._matrix).all()

        return super().__eq__(other)


class ScalarTimeDependence(DynamicMatrix, ABC):
    """A dynamic matrix whose time dependence is a scalar factor.

    :param matrix: The constant matrix.
    """

    def __init__(self, matrix: Union[ArrayLike, list[list]]):
        self._matrix = np.asarray(matrix)
        self._dims = tuple(range(1, len(self._matrix.shape) + 1))

    @abstractmethod
    def factor(self, t: NDArray[np.float64]) -> NDArray:
        """The scalar factor."""

        pass

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """All the parameters that define the instance.

        Should be so that ``__init__(matrix, **parameters)`` works.
        """

        pass

    def call(self, t: NDArray[np.float64]):
        factors = self.factor(t)
        factors = np.expand_dims(factors, axis=self._dims)
        return factors * self._matrix[None, :]

    def __getstate__(self):
        return dict(matrix=self._matrix, **self.parameters())


class SmoothStep(ScalarTimeDependence):
    """
    A scalar-factor time dependence in the form of a smooth step
    function (see `Wikipedia`_).  The factor is zero before
    ``t_initial`` and one after ``t_final``.  Between those two points
    a polynomial interpolation is performed so that the derivative of
    order ``order`` vanishes at the boundary.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Smoothstep

    :param t_initial: The initial time for the step.  The smooth step
        is zero before this time.
    :param t_final: The final time for the step.  The smooth step is
        one after this time.
    :param order: The order of the interpolating hermite polynomial.
        The derivative of order ``order`` of the smooth step will be
        zero at the initial and final time.
    :param deriv: The nth-derivative.
    """

    def __init__(
        self,
        matrix: Union[ArrayLike, list[list]],
        t_initial: float,
        t_final: float,
        order: int = 2,
        deriv: Optional[int] = None,
    ):
        super().__init__(matrix)
        self.t_initial = t_initial
        self.t_final = t_final
        self.order = order

        self._time_period = self.t_final - self.t_initial

        coeffs = [0] * (self.order + 1) + [
            scipy.special.comb(self.order + n, n)
            * scipy.special.comb(2 * self.order + 1, self.order - n)
            * (-1) ** n
            for n in range(0, self.order + 1)
        ]

        self._poly = Polynomial(
            coeffs, domain=[self.t_initial, self.t_final], window=[0, 1]
        )

        self.deriv = deriv
        self._poly = self._poly.deriv(self.deriv) if self.deriv else self._poly

    def parameters(self):
        params = dict(t_initial=self.t_initial, t_final=self.t_final, order=self.order)
        if self.deriv:
            params["deriv"] = self.deriv

        return params

    def factor(self, t: NDArray[np.float64]):
        result = np.empty_like(t)

        smaller = t <= self.t_initial
        larger = t >= self.t_final
        between = ~(larger | smaller)

        result[smaller] = 0
        result[larger] = 0 if self.deriv else 1
        result[between] = self._poly(t[between])

        return result

    def derivative(self):
        params = self.parameters()
        current_deriv = self.deriv or 0
        params["deriv"] = current_deriv + 1

        return self.__class__(self._matrix, **params)

    def max_operator_norm(self, t_max: float = np.inf) -> float:
        """
        :returns: The maximum of the operator norm between ``0`` and
                  ``t_max`` as tuple ``time, value``.
        """

        return self.operator_norm(t_max)


class Harmonic(ScalarTimeDependence):
    """
    A scalar-factor time dependence in the form of a sine function
    with frequency ``ω`` and phase ``φ``.

    .. math::

       sin(ω t + φ)

    :param matrix: The constant matrix coefficient.
    :param ω: The modulation frequency.
    :param φ: The modulation phase.
    """

    def __init__(
        self,
        matrix: Union[ArrayLike, list[list]],
        ω: float,
        φ: float = 0,
    ):
        super().__init__(matrix)
        self.ω = ω
        self.φ = φ

    def parameters(self):
        return dict(ω=self.ω, φ=self.φ)

    def derivative(self):
        return self.ω * self.__class__(self._matrix, self.ω, self.φ + np.pi / 2)

    def factor(self, t: NDArray[np.float64]):
        return np.sin((self.ω * t) % (2 * np.pi) + self.φ)


class Periodic(DynamicMatrix):
    """
    A wrapper around a :any:`DynamicMatrix` that repeats its time
    dependence within a certain period.

    :param matrix: The matrix to wrap.
    :param period: The period of repetion.
    """

    def __init__(self, matrix: DynamicMatrix, period: float = 1):
        self._matrix = matrix
        self._period = period

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return self._matrix.call(t % self._period)

    def derivative(self):
        return self.__class__(self._matrix.derivative(), self._period)

    def __getstate__(self):
        return dict(matrix=self._matrix, period=self._period)


class Shift(DynamicMatrix):
    """
    A wrapper around a :any:`DynamicMatrix` that shifts the time
    dependence.

    :param matrix: The matrix to wrap.
    :param delta: The shift.
    """

    def __init__(self, matrix: DynamicMatrix, delta: float = 1):
        self._matrix = matrix
        self._delta = delta

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return self._matrix.call(t - self._delta)

    def derivative(self):
        return self.__class__(self._matrix.derivative(), self._delta)

    def __getstate__(self):
        return dict(matrix=self._matrix, delta=self._delta)


class ScaleTime(DynamicMatrix):
    """
    A wrapper around a :any:`DynamicMatrix` that scales the time
    dependence.

    :param matrix: The matrix to wrap.
    :param gamma: The scale.
    """

    def __init__(self, matrix: DynamicMatrix, gamma: float = 1):
        self._matrix = matrix
        self._gamma = gamma

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        return self._matrix.call(t * self._gamma)

    def derivative(self):
        return self.__class__(self._gamma * self._matrix.derivative(), self._gamma)

    def __getstate__(self):
        return dict(matrix=self._matrix, gamma=self._gamma)


class Piecewise(DynamicMatrix):
    """
    A wrapper around a collection of :any:`DynamicMatrix` that glues
    them together in a time-piece like manner.

    :param matrix: The matrix to wrap.
    :param time_points: The time points at which to switch matrices.
        The first is the initial time.
    """

    def __init__(self, matrices: Iterable[DynamicMatrix], time_points: Iterable[float]):
        self._matrices = list(matrices)
        self._time_points = np.array(time_points)

        assert np.all(
            self._time_points[:-1] <= self._time_points[1:]
        ), "The time points should be sorted in ascending order."

        first_shape = self._matrices[0].shape

        for mat in self._matrices[1:]:
            assert (
                first_shape == mat.shape
            ), "The shapes of all the matrices should be the same."

        self._shape = first_shape

    @property
    def shape(self) -> tuple[int]:
        return self._shape

    def call(self, t: NDArray[np.float64]) -> MatrixType:
        result = np.zeros((len(t), *self._shape), dtype=np.complex128)
        for mat, begin, end in zip(
            self._matrices, self._time_points[:-1], self._time_points[1:]
        ):
            indices = np.where(np.logical_and(begin <= t, t < end))
            result[indices] = mat.call(t[indices])

        return result

    def derivative(self):
        return self.__class__(
            [m.derivative() for m in self._matrices], self._time_points
        )

    def __getstate__(self):
        return dict(matrices=self._matrices, time_points=self._time_points)

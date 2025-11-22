from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

from .exceptions import (
    InvalidMatrixError,
    DimensionError,
    SingularMatrixError,
    NonSquareMatrixError,
)

Number = Union[int, float, complex]


class Matrix:
    """
    Basic Matrix class for educational linear algebra operations.

    - Stored internally as a list of lists of numbers.
    - Supports:
        * Addition, subtraction
        * Scalar multiplication
        * Matrix multiplication
        * Transpose
        * Determinant (via Gaussian elimination)
        * Inverse (via Gauss–Jordan elimination)
    """

    def __init__(self, data: Sequence[Sequence[Number]]) -> None:
        if not data:
            raise InvalidMatrixError("Matrix data cannot be empty.")

        # Convert to list of lists and validate rectangular shape
        rows: List[List[Number]] = [list(row) for row in data]

        row_lengths = {len(row) for row in rows}
        if len(row_lengths) != 1 or 0 in row_lengths:
            raise InvalidMatrixError("All rows must have the same non-zero length.")

        self._data = rows
        self._rows = len(rows)
        self._cols = len(rows[0])



    @property
    def shape(self) -> Tuple[int, int]:
        """Return (rows, cols)."""
        return self._rows, self._cols

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def copy(self) -> "Matrix":
        """Return a shallow copy of the matrix."""
        return Matrix([row[:] for row in self._data])

    def to_list(self) -> List[List[Number]]:
        """Return a deep copy of the underlying list-of-lists."""
        return [row[:] for row in self._data]



    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Matrix":
        if rows <= 0 or cols <= 0:
            raise InvalidMatrixError("Matrix dimensions must be positive.")
        return cls([[0 for _ in range(cols)] for _ in range(rows)])

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        """Return an n x n identity matrix."""
        if n <= 0:
            raise InvalidMatrixError("Size of identity matrix must be positive.")
        data = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            data[i][i] = 1
        return cls(data)

    @classmethod
    def from_flat(cls, flat: Iterable[Number], rows: int, cols: int) -> "Matrix":
        """Create a matrix from a flat iterable of length rows * cols."""
        flat_list = list(flat)
        if len(flat_list) != rows * cols:
            raise InvalidMatrixError(
                f"Expected {rows * cols} elements, got {len(flat_list)}."
            )
        data = [
            flat_list[i * cols : (i + 1) * cols]  # noqa: E203
            for i in range(rows)
        ]
        return cls(data)



    def __repr__(self) -> str:
        return f"Matrix({self._data!r})"

    def __str__(self) -> str:
        return "\n".join("[" + "  ".join(f"{val!r}" for val in row) + "]" for row in self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        if self.shape != other.shape:
            return False
        return self._data == other._data

 

    def __getitem__(self, idx: int) -> List[Number]:
        """Return a row; supports m[i][j] indexing."""
        return self._data[idx]

  

    def _check_same_shape(self, other: "Matrix") -> None:
        if self.shape != other.shape:
            raise DimensionError(
                f"Incompatible shapes {self.shape} and {other.shape} "
                "for elementwise operation."
            )

    def __add__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            return NotImplemented
        self._check_same_shape(other)
        data = [
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self._data, other._data)
        ]
        return Matrix(data)

    def __sub__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            return NotImplemented
        self._check_same_shape(other)
        data = [
            [a - b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self._data, other._data)
        ]
        return Matrix(data)

    def __neg__(self) -> "Matrix":
        return Matrix([[-val for val in row] for row in self._data])

    def __mul__(self, other: Union["Matrix", Number]) -> "Matrix":
        """
        If 'other' is a Matrix: matrix multiplication.
        If 'other' is a scalar: scalar multiplication.
        """
        if isinstance(other, Matrix):
            return self._matmul(other)
        if isinstance(other, (int, float, complex)):
            data = [[other * val for val in row] for row in self._data]
            return Matrix(data)
        return NotImplemented

    def __rmul__(self, other: Number) -> "Matrix":
        # Scalar multiplication from the left.
        if isinstance(other, (int, float, complex)):
            return self * other
        return NotImplemented

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """Support the @ operator for matrix multiplication."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._matmul(other)

    def _matmul(self, other: "Matrix") -> "Matrix":
        if self.cols != other.rows:
            raise DimensionError(
                f"Incompatible shapes {self.shape} and {other.shape} "
                "for matrix multiplication."
            )
        result_data = []
        # Precompute columns of 'other' for speed
        other_cols = [
            [other._data[i][j] for i in range(other.rows)]  # column j
            for j in range(other.cols)
        ]
        for i in range(self.rows):
            result_row: List[Number] = []
            for j in range(other.cols):
                s: Number = 0
                row = self._data[i]
                col = other_cols[j]
                for a, b in zip(row, col):
                    s += a * b
                result_row.append(s)
            result_data.append(result_row)
        return Matrix(result_data)


    @property
    def T(self) -> "Matrix":
        """Shorthand for transpose."""
        return self.transpose()

    def transpose(self) -> "Matrix":
        data = [
            [self._data[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ]
        return Matrix(data)

    def _ensure_square(self) -> None:
        if self.rows != self.cols:
            raise NonSquareMatrixError(
                f"Operation requires a square matrix, got shape {self.shape}."
            )

    def determinant(self) -> Number:
        """
        Compute determinant using Gaussian elimination to upper triangular form.

        Complexity: O(n^3). Mutates a local copy of the data.
        """
        self._ensure_square()
        n = self.rows
        # Work on a copy
        a: List[List[float]] = [
            [float(x) for x in row] for row in self._data
        ]

        det: float = 1.0
        sign: int = 1

        for i in range(n):
            # Find pivot in column i at or below row i
            pivot_row = i
            while pivot_row < n and abs(a[pivot_row][i]) < 1e-12:
                pivot_row += 1

            if pivot_row == n:
                # Column is all zeros => determinant is zero
                return 0.0

            if pivot_row != i:
                # Swap rows
                a[i], a[pivot_row] = a[pivot_row], a[i]
                sign *= -1

            pivot = a[i][i]
            det *= pivot

            # Eliminate below
            for j in range(i + 1, n):
                factor = a[j][i] / pivot
                row_j = a[j]
                row_i = a[i]
                for k in range(i, n):
                    row_j[k] -= factor * row_i[k]

        return sign * det

    def inverse(self) -> "Matrix":
        """
        Compute the inverse using Gauss–Jordan elimination.

        Raises SingularMatrixError if the matrix is singular.
        """
        self._ensure_square()
        n = self.rows

        # Augment with identity: [A | I]
        a: List[List[float]] = [
            [float(x) for x in row] + [1.0 if i == j else 0.0 for j in range(n)]
            for i, row in enumerate(self._data)
        ]

        # Perform Gauss–Jordan elimination
        for i in range(n):
            # Find pivot
            pivot_row = i
            while pivot_row < n and abs(a[pivot_row][i]) < 1e-12:
                pivot_row += 1

            if pivot_row == n:
                raise SingularMatrixError("Matrix is singular and cannot be inverted.")

            # Swap to put pivot on the diagonal
            if pivot_row != i:
                a[i], a[pivot_row] = a[pivot_row], a[i]

            # Normalize pivot row
            pivot = a[i][i]
            if abs(pivot) < 1e-12:
                raise SingularMatrixError("Matrix is singular and cannot be inverted.")

            inv_pivot = 1.0 / pivot
            a[i] = [val * inv_pivot for val in a[i]]

            # Eliminate other rows
            for r in range(n):
                if r == i:
                    continue
                factor = a[r][i]
                if abs(factor) < 1e-15:
                    continue
                row_r = a[r]
                row_i = a[i]
                for c in range(2 * n):
                    row_r[c] -= factor * row_i[c]

        # Extract the right half as the inverse
        inv_data = [row[n:] for row in a]
        return Matrix(inv_data)

    

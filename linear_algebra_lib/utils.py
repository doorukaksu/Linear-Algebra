"""
Utility functions for common matrix constructions and checks.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Union

from .matrix import Matrix
from .exceptions import InvalidMatrixError

Number = Union[int, float, complex]


def is_square(matrix: Matrix) -> bool:
    """Return True if the matrix is square."""
    rows, cols = matrix.shape
    return rows == cols


def diagonal(values: Iterable[Number]) -> Matrix:
    """
    Construct a diagonal matrix from an iterable of values.
    """
    vals = list(values)
    n = len(vals)
    data: List[List[Number]] = [[0 for _ in range(n)] for _ in range(n)]
    for i, v in enumerate(vals):
        data[i][i] = v
    return Matrix(data)


def from_rows(*rows: Iterable[Number]) -> Matrix:
    """
    Convenience wrapper: Matrix.from_rows([row1], [row2], ...)
    """
    return Matrix(list(rows))


def validate_vector(vec: Iterable[Number]) -> Tuple[List[Number], int]:
    """
    Validate a vector (1D iterable) and return (list(vec), length).
    Useful for dot products or building column/row matrices.
    """
    v_list = list(vec)
    if not v_list:
        raise InvalidMatrixError("Vector cannot be empty.")
    return v_list, len(v_list)

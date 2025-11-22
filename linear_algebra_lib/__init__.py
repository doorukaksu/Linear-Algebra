from .exceptions import (
    LinearAlgebraError,
    InvalidMatrixError,
    DimensionError,
    SingularMatrixError,
    NonSquareMatrixError,
)

from .matrix import Matrix
from . import utils

__all__ = [
    "Matrix",
    "LinearAlgebraError",
    "InvalidMatrixError",
    "DimensionError",
    "SingularMatrixError",
    "NonSquareMatrixError",
    "utils",
]

__version__ = "0.1.0"

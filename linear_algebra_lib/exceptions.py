class LinearAlgebraError(Exception):
    """Base exception for all linear algebra library errors."""


class InvalidMatrixError(LinearAlgebraError):
    """Raised when the provided data cannot form a valid matrix."""


class DimensionError(LinearAlgebraError):
    """Raised when matrix dimensions are incompatible for an operation."""


class NonSquareMatrixError(LinearAlgebraError):
    """Raised when an operation requires a square matrix but receives a non-square one."""


class SingularMatrixError(LinearAlgebraError):
    """Raised when attempting to invert a singular (non-invertible) matrix."""

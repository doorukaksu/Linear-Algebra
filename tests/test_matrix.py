# tests/test_matrix.py

import unittest

from linear_algebra_lib import (
    Matrix,
    DimensionError,
    SingularMatrixError,
    NonSquareMatrixError,
)


class TestMatrixBasics(unittest.TestCase):
    def test_shape_and_construction(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.shape, (2, 2))
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)

    def test_invalid_construction(self):
        with self.assertRaises(Exception):
            Matrix([])

        with self.assertRaises(Exception):
            Matrix([[1, 2], [3]])  # non-rectangular

    def test_addition(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        c = a + b
        self.assertEqual(c, Matrix([[6, 8], [10, 12]]))

    def test_add_dimension_mismatch(self):
        a = Matrix([[1, 2]])
        b = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(DimensionError):
            _ = a + b

    def test_scalar_multiplication(self):
        m = Matrix([[1, -2], [3, 0]])
        self.assertEqual(2 * m, Matrix([[2, -4], [6, 0]]))
        self.assertEqual(m * 3, Matrix([[3, -6], [9, 0]]))

    def test_matrix_multiplication(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])
        b = Matrix([[7, 8], [9, 10], [11, 12]])
        c = a @ b
        self.assertEqual(
            c,
            Matrix([[58, 64], [139, 154]]),
        )

    def test_matmul_dimension_mismatch(self):
        a = Matrix([[1, 2]])
        b = Matrix([[1, 2]])
        with self.assertRaises(DimensionError):
            _ = a @ b

    def test_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        t = m.T
        self.assertEqual(t, Matrix([[1, 4], [2, 5], [3, 6]]))


class TestDeterminantAndInverse(unittest.TestCase):
    def test_determinant_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        det = m.determinant()
        self.assertAlmostEqual(det, -2.0)

    def test_determinant_3x3(self):
        m = Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        # Known determinant = -306
        self.assertAlmostEqual(m.determinant(), -306.0)

    def test_determinant_non_square(self):
        m = Matrix([[1, 2, 3]])
        with self.assertRaises(NonSquareMatrixError):
            _ = m.determinant()

    def test_inverse_2x2(self):
        m = Matrix([[4, 7], [2, 6]])
        inv = m.inverse()

        # m * inv should be identity (within numerical tolerance)
        ident = m @ inv
        self.assertEqual(ident.shape, (2, 2))

        # Check approximate identity
        expected = Matrix.identity(2).to_list()
        actual = ident.to_list()
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(actual[i][j], expected[i][j], places=6)

    def test_inverse_singular(self):
        m = Matrix([[1, 2], [2, 4]])  # determinant = 0
        with self.assertRaises(SingularMatrixError):
            _ = m.inverse()

    def test_inverse_non_square(self):
        m = Matrix([[1, 2, 3]])
        with self.assertRaises(NonSquareMatrixError):
            _ = m.inverse()


if __name__ == "__main__":
    unittest.main()

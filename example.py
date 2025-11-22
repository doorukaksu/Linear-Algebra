from linear_algebra_lib import Matrix

A = Matrix([[1,2],
            [3,4]])

B = Matrix([[5,6],
            [7,8]])

print("A+B:\n", A+B)
print("A@B:\n", A@B)
print("det(A):", A.determinant())
print("inv(A):\n", A.inverse())

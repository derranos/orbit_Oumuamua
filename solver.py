import numpy as np  # Still using numpy for basic array operations


def determinant_3x3(matrix):
    """Calculates the determinant of a 3x3 numpy array."""
    a, b, c = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    d, e, f = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    g, h, i = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return det


def solve_3x3_cramer(A, b):

    if A.shape != (3, 3):
        raise ValueError("Matrix A must be 3x3")
    if len(b) != 3:
        raise ValueError("Vector b must have length 3")

    det_A = determinant_3x3(A)

    if abs(det_A) < 1e-15:
        raise ValueError("Matrix A is singular.")

    A_x = A.copy()
    A_x[:, 0] = b
    det_Ax = determinant_3x3(A_x)

    A_y = A.copy()
    A_y[:, 1] = b
    det_Ay = determinant_3x3(A_y)

    A_z = A.copy()
    A_z[:, 2] = b
    det_Az = determinant_3x3(A_z)


    x = det_Ax / det_A
    y = det_Ay / det_A
    z = det_Az / det_A

    return np.array([x, y, z])

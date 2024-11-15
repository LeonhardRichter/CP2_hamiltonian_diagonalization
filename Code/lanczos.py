# Author: Leonhard Richter
# Python script defining the lanczos method for tridiagonalizing a self-adjoing matrix in sparse csr format.

from scipy import linalg
from scipy import sparse
from scipy.sparse import csr_array as csr

import numpy as np
from numpy.typing import ArrayLike


norm = linalg.norm
# we will use sparse.csr, sparse.random_array, np.random.default_rng


def sparse_adjoint(A: csr):
    return A.transpose().conj()


# def is_self_adjoint(A: csr):
#     n, m = A.shape
#     assert n == m, "matrix is not quadratic"
#     # check only the non-zero entries
#     # if they are unequal, a True will be written to the sparse result array
#     # hence, if not self adjoint, the number of datapoints (nnz) is greater than 0
#     return (sparse_adjoint(A) != A).nnz == 0


def is_self_adjoint(A: csr | ArrayLike) -> bool:
    n, m = A.shape
    assert n == m, "matrix is not quadratic"
    if type(A) is csr:
        # check only the non-zero entries
        # if they are unequal, a True will be written to the sparse result array
        # hence, if not self adjoint, the number of datapoints (nnz) is greater than 0
        return (sparse_adjoint(A) != A).nnz == 0
    if type(A) is ArrayLike:
        return np.all(sparse_adjoint(A) == A)


def csr_random_self_adjoint(n: int, d: float = 0.01) -> csr:
    A = sparse.random_array(
        (n, n),
        density=d,
        format="csr",
        dtype=np.complex64,
        random_state=np.random.default_rng(),
    )
    A = csr(1 / 2 * (sparse_adjoint(A) + A))
    assert is_self_adjoint(
        A
    ), "something went wrong: result is not self adjoint"
    return A


def lanczos_tridiag(
    A: csr, v: np.ndarray, max_dim: int, epsilon: float
) -> ArrayLike:
    """
    Funciton of the lanczos method for tridiagonalizing a self-adjoit matrix.
    The implementation follows
        Koch, Erik (2015) ‘The Lanczos Method’, in Pavarini, Eva et al. (eds) Many-body physics: from Kondo to Hubbard: lecture notes of the Autumn School on Correlated Electrons 2015: at Forschungszentrum Jülich, 21-25 September 2015. Jülich: Forschungszentrum Jülich (Schriften des Forschungszentrums Jülich. Reihe Modeling and Simulation, Band 5). Available at: https://www.cond-mat.de/events/correl15/manuscripts/koch.pdf.
    """
    n, m = A.shape
    assert n == m, "matrix is not quadratic"
    assert is_self_adjoint(A), "matrix is not self-adjoint"

    # initialize two arrays a and b for storing the result
    a = np.zeros(max_dim, dtype=np.float64)
    b = np.zeros(max_dim, dtype=np.float64)

    # initialize v and w
    v = v / norm(v)
    w = A @ v

    # first iteration
    a[0] = np.real(v @ w)
    w = w - a[0] * v  # axpy(-a[0],v,w)
    b[1] = norm(w)
    for i in range(1, max_dim):
        if abs(b[i]) < epsilon:
            print(f"invariant subspace encountered at i = {i}")
            return
        w = w / b[i]  # w = v_i = \tilde{v_i}/b_i
        v = -b[i] * v  # v = -b_i v_{i-1}
        v, w = w, v  # v = v_i, w = -b_i v_{i-1}
        # at first, we leave -a_i v_i out, to compute a_i easily without matrix vector product in the next step
        w = w + A @ v  # w = H v_i - b_i v_{i-1}
        a[i] = np.real(
            v @ w
        )  # a_i = <v_i,w> = <v_i, Hv_i> - b_i <v_i, v_{i-1}>
        w = w - a[i] * v  # w = H v_i - b_i v_{i-1} - a_i v_i = \tilde{v_{i+1}}
        if i + 1 < max_dim:
            b[i + 1] = norm(w)  # b_{i+1} = ||\tilde{v_{i+1}}||
    return np.diag(a) + np.diag(b[1::], 1) + np.diag(b[1::], -1)

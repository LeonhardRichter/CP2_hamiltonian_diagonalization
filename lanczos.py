# Author: Leonhard Richter, 2024/2025
# Python script defining the lanczos method for tridiagonalizing a self-adjoing matrix in sparse csr format.

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg, sparse
from scipy.sparse import csr_array as csr

from typing import Union, Callable

from tqdm import tqdm

norm = linalg.norm
rng = np.random.default_rng()
# we will use sparse.csr, sparse.random_array, np.random.default_rng


def adjoint(A: Union[csr, ArrayLike]):
    return A.transpose().conj()


def is_self_adjoint(A: Union[csr, ArrayLike]) -> bool:
    n, m = A.shape
    if n != m:
        print("Matrix is not quadratic")
        return False
    if type(A) is csr:
        # check only the non-zero entries
        # if they are unequal, a True will be written to the sparse result array
        # hence, if not self adjoint, the number of datapoints (nnz) is greater than 0
        return (adjoint(A) != A).nnz == 0
    if type(A) is ArrayLike:
        return np.all(adjoint(A) == A)


def csr_random_self_adjoint(n: int, d: float = 0.01) -> csr:
    A = sparse.random_array(
        (n, n),
        density=d,
        format="csr",
        dtype=np.complex128,
        random_state=np.random.default_rng(),
    )
    A = csr(1 / 2 * (adjoint(A) + A))
    assert is_self_adjoint(
        A
    ), "something went wrong: result is not self adjoint"
    return A


def lanczos(
    A: csr, v: np.ndarray, dim: int, epsilon: float = 0.001
) -> np.ndarray:
    """
    Funciton of the lanczos method for tridiagonalizing a self-adjoit matrix.
    The implementation follows
        Koch, Erik (2015) ‘The Lanczos Method’, in Pavarini, Eva et al. (eds) Many-body physics: from Kondo to Hubbard: lecture notes of the Autumn School on Correlated Electrons 2015: at Forschungszentrum Jülich, 21-25 September 2015. Jülich: Forschungszentrum Jülich (Schriften des Forschungszentrums Jülich. Reihe Modeling and Simulation, Band 5). Available at: https://www.cond-mat.de/events/correl15/manuscripts/koch.pdf.
    """
    n, m = A.shape
    assert n == m, "matrix is not quadratic"
    assert is_self_adjoint(A), "matrix is not self-adjoint"
    assert n == len(v), "Shapes of A and v do not match"

    # initialize two arrays a and b for storing the result
    a = np.zeros(dim, dtype=np.float64)
    b = np.zeros(dim - 1, dtype=np.float64)
    # initialize an array for storing the Lanczos basis as row vectors
    basis = np.zeros((dim, n), dtype=np.complex128)

    # initialize v and w
    v = v / norm(v)
    w = A @ v
    basis[0] = v
    # first iteration
    a[0] = np.real(v @ w)
    w = w - a[0] * v  # axpy(-a[0],v,w)
    b[0] = norm(w)
    for i in range(1, dim):
        if abs(b[i - 1]) < epsilon:
            print(f"invariant subspace encountered at i = {i}")
            return np.diag(a) + np.diag(b, 1) + np.diag(b, -1), basis
        w = w / b[i - 1]  # w = v_i = \tilde{v_i}/b_i
        basis[i] = w
        v = -b[i - 1] * v  # v = -b_i v_{i-1}
        v, w = w, v  # v = v_i, w = -b_i v_{i-1}
        # at first, we leave -a_i v_i out, to compute a_i easily without matrix vector product in the next step
        w = w + A @ v  # w = H v_i - b_i v_{i-1}
        a[i] = np.real(
            v @ w
        )  # a_i = <v_i,w> = <v_i, Hv_i> - b_i <v_i, v_{i-1}>
        w = w - a[i] * v  # w = H v_i - b_i v_{i-1} - a_i v_i = \tilde{v_{i+1}}
        b_new = norm(w)
        if i + 1 < dim:
            b[i] = b_new  # b_{i+1} = ||\tilde{v_{i+1}}||
    basis[-1] = w / b_new
    # return matrix (dense) in the Lanczos basis, and the matrix (dense) of Lanczos basis vectors as rows
    return np.diag(a) + np.diag(b, 1) + np.diag(b, -1), basis

    # # build the matrix in this basis (again in csr)
    # indices = np.arange(max_dim)
    # diag_indices = (indices, indices)
    # diag = csr((a, diag_indices), shape=(max_dim, max_dim))
    # upper_diag_indices = (indices[0:-1], indices[0:-1] + 1)
    # upper_diag = csr((b, upper_diag_indices), shape=(max_dim, max_dim))
    # lower_diag_indices = (indices[0:-1] + 1, indices[0:-1])
    # lower_diag = csr((b, lower_diag_indices), shape=(max_dim, max_dim))
    # return diag + lower_diag + upper_diag


# def tridiag_to_diag(A: Union[ArrayLike, csr], copy: bool = True):
#     """
#     [[WIP]] function to diagonalize a real, symmetric, tridiagonal matrix, resulting from the Lanczos algorithm.
#     """
#     # we diagonalize from top to bottom by row manipulations
#     if copy:
#         A = A.copy()

#     n = A.shape[0] - 1
#     for i in range(0, n):
#         # goal: diagonalize the i-th row
#         Ai1i1 = A[[i + 1], [i + 1]]
#         if Ai1i1 != 0:
#             c = A[[i], [i + 1]] / Ai1i1
#             A[[i], [i]] = A[[i], [i]] - c * A[[i + 1], [i]]
#             # the entry right to the diagonal is zero now:
#             # A[[i], [i+1]] = A[[i],[i+1]] - c*A[[i+1],[i+1]] = 0
#             A[[i], [i + 1]] = 0
#         else:
#             print(f"zero diagonal encountered in row {i+1}, aborting")
#             print(A.toarray().round(2))
#             break
#         if i > 0:
#             if A[[i - 1], [i - 1]] != 0:
#                 A[[i, i - 1]] = 0
#             else:
#                 print(f"zero diagonal encountered in row {i-1}, aborting")
#                 print(A.toarray().round(2))
#                 break

#     # diagonalize the last row by the diagonal row above
#     if A[[n - 1], [n - 1]] != 0:
#         A[[n], [n - 1]] = 0
#     return A


def lanczos_evo(
    H,
    v,
    dim: int,
    T: float,
    dt: float = 0.01,
    observables: tuple[
        Callable[[Union[ArrayLike, csr], Union[ArrayLike, csr]], np.complex128]
    ] = tuple(),
    lanczos_epsilon: float = 0.001,
    save_states: bool = False,
    return_final: bool = True,
    progress_bar: bool = True,
) -> tuple[list[float], list[np.ndarray], list[list[np.complex128]]]:
    """
    Compute the evolution of a given initial state under a given Hamiltonian for some time approximated on the Krylow space of given dimension.
    This is done by approximating the Hamiltonian on the Krylow space in tri-diagonal form and evaluating the matrix exponential with scipy expm for some short time.
    The result is applied to the initial state to obtain the initial state for the next iteration.
    For the next iteration, the Hamiltonian is again approximated using Lanczos alogithm, but now with respect to the evolved state.
    \n
    Returns `t`, `v`, `exp` \n
    `t` : list of times \n
    `v` : list of states corresponding to times (empty if `save_states` and `return_final` are False) \n
    `obs`: list of lists for each element of `observables`. Each list contains expectation values at each time in `t` \n
    """

    n, m = H.shape
    assert n == m, "matrix is not quadratic"
    assert is_self_adjoint(H), "matrix is not self-adjoint"
    assert len(v) == n

    tt = [
        0,
    ]

    vt = [
        v,
    ]

    exp = list()

    for A in observables:
        exp.append(
            [
                A(v),
            ]
        )
    if observables == list():
        exp = [
            list(),
        ]

    def update(v_new, t, vt, tt):
        tt.append(t)
        for i, A in enumerate(observables):
            exp[i].append(A(v_new))
        if save_states:
            vt.append(v_new)
        if not save_states:
            vt[-1] = v_new

    def step(dt, v):
        H_approx, basis = lanczos(A=H, v=v, dim=dim, epsilon=lanczos_epsilon)
        U_approx = linalg.expm(-1j * dt * H_approx)
        v_approx = np.zeros(len(basis), dtype=np.complex128)
        v_approx[0] = (
            1  # the first basis vector in Lanczos basis is v, should be the same as basis @ v
        )
        v_approx_new = U_approx @ v_approx  # in krylov basis
        # multiply each basis vector with the parameter of the approximated vector and sum over all rows
        return np.sum(v_approx_new[:, np.newaxis] * basis, axis=0)

    # mainloop
    with tqdm(total=T // dt + 1, disable=not progress_bar) as pbar:
        t = dt
        while t < T:
            v_new = step(dt, vt[-1])
            t += dt
            update(v_new, t, vt, tt)
            pbar.update(1)
        t_rest = T - t
        if t_rest != 0:
            v_final = step(t_rest, vt[-1])
            t += t_rest
            update(v_final, t, vt, tt)
            pbar.update(1)
    if return_final:
        return tt, vt, exp
    if not return_final:
        return tt, list(), exp


if __name__ == "__main__":
    H = csr_random_self_adjoint(100)
    A = csr_random_self_adjoint(100)
    B = csr_random_self_adjoint(100)
    v = rng.random(100) + 1j * rng.random(100)

    K = 10

    H_approx, basis = lanczos(H, v, K)

    print(np.round(H_approx, 2))
    print(len(basis))

    tt, vt, exp = lanczos_evo(
        H, v, dim=K, T=0.1, dt=0.01, observables=[A, B], save_states=True
    )
    print(len(tt))
    print(len(vt))
    print("distance final to initial")
    print(np.linalg.norm(vt[-1] - v))
    print(np.real(np.round(exp, 3)))
    print(np.isclose(tt[-1], 0.1))

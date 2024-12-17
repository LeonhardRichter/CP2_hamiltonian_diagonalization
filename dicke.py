import numpy as np
from fractions import Fraction
from scipy.sparse import csr_array as csr
from scipy import sparse, linalg


def j_list_gen(N) -> tuple[int | Fraction]:
    jj = list()
    j = Fraction(N, 2)
    if j.is_integer():
        j = int(j)

    while j >= 0:
        jj.append(j)
        j -= 1

    return tuple(jj)


def m_list_gen(j: int | Fraction) -> tuple[int | Fraction]:
    return tuple(j - _ for _ in range(int(2 * j) + 1))


def dicke_dim(N: int):
    return sum([len([m for m in m_list_gen(j)]) for j in j_list_gen(N)])


def dicke_basis(N: int) -> tuple[tuple[int | Fraction]]:
    """
    Gives a tuple of 2-tuples.
    Each 2-touple containts the j and m of one Dicke state.
    The states are ordered in decreasing order first in j then in m.

    Examples:
    --------
        >>> dicke_basis(3)
        ((Fraction(3, 2), Fraction(3, 2)),
        (Fraction(3, 2), Fraction(1, 2)),
        (Fraction(3, 2), Fraction(-1, 2)),
        (Fraction(3, 2), Fraction(-3, 2)),
        (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(-1, 2)))
    """
    jj = j_list_gen(N)
    mm = m_list_gen
    ordered_basis = tuple((j, m) for j in jj for m in mm(j))
    return ordered_basis


def dicke_dim_bak(N: int) -> int:
    return len(dicke_basis(N))


X = csr(([1, 1], ([0, 1], [1, 0])), dtype=np.complex128)
Y = csr(([-1j, 1j], ([0, 1], [1, 0])), dtype=np.complex128)
Z = csr(([1, -1], ([0, 1], [0, 1])), dtype=np.complex128)


def Sz(N: int) -> csr:
    basis = dicke_basis(N)
    dim = dicke_dim(N)

    row_ind = []
    col_ind = []
    data = []

    for i, (_, m) in enumerate(basis):
        row_ind.append(i)
        col_ind.append(i)
        data.append(np.float64(m))
    A = csr((data, (row_ind, col_ind)), shape=(dim, dim))
    A.eliminate_zeros()
    return A


def Sp(N: int) -> csr:
    basis = dicke_basis(N)
    dim = dicke_dim(N)

    row_ind = []
    col_ind = []
    data = []

    for i, (j, m) in enumerate(basis):
        if i != 0:
            row_ind.append(i)  # recall to decreasing ordering
            col_ind.append(i - 1)
            data.append(np.sqrt(np.float64((j - m) * (j + m + 1))))
    A = csr((data, (row_ind, col_ind)), shape=(dim, dim))
    A.eliminate_zeros()
    return A


def Sm(N: int) -> csr:
    basis = dicke_basis(N)
    dim = dicke_dim(N)

    row_ind = []
    col_ind = []
    data = []

    for i, (j, m) in enumerate(basis):
        if i != dim - 1:
            row_ind.append(i)
            col_ind.append(i + 1)  # due to decreasing ordering
            data.append(np.sqrt(np.float64((j + m) * (j - m + 1))))
    A = csr((data, (row_ind, col_ind)), shape=(dim, dim))
    A.eliminate_zeros()
    return A


# attention: for the bosonic part we use an increasing ordering such that when
# extending the cut off stuff gets appended not prepented


def Ad(n_max: int) -> csr:
    """
    `n_max` is the maximal number of photons in the cut-off Hilbert space.
    The dimension of this subspace is `n_max +1` due to the vacuum spase with index `0`
    The `(n_max +1)`-th component gets mapped to zero.
    """
    row_ind = []
    col_ind = []
    data = []

    for n in range(n_max):
        row_ind.append(n)
        col_ind.append(n + 1)
        data.append(np.sqrt(n + 1))
    A = csr((data, (row_ind, col_ind)), shape=(n_max + 1, n_max + 1))
    return A


def An(n_max: int) -> csr:
    """`n_max` is the maximal number of photons in the cut-off Hilbert space.
    The dimension of this subspace is `n_max +1` due to the vacuum spase with index `0`
    """
    row_ind = []
    col_ind = []
    data = []

    for n in range(1, n_max + 1):
        row_ind.append(n)
        col_ind.append(n - 1)
        data.append(np.sqrt(n))

    A = csr((data, (row_ind, col_ind)), shape=(n_max + 1, n_max + 1))
    return A


def num(n_max: int) -> csr:
    row_ind = []
    col_ind = []
    data = []

    for n in range(n_max + 1):
        row_ind.append(n)
        col_ind.append(n)
        data.append(n)

    A = csr((data, (row_ind, col_ind)), shape=(n_max + 1, n_max + 1))
    return A


def H_D(
    N: int,
    n_max: int,
    coupling: float = 1.0,
    frequency: float = 1.0,
    energy_gap=None,
) -> csr:
    if energy_gap is None:
        energy_gap = frequency
    n_id = sparse.eye_array(n_max + 1)
    N_id = sparse.eye_array(dicke_dim(N))
    return (
        energy_gap / 2 * sparse.kron(Sz(N), n_id)
        + frequency * sparse.kron(N_id, num(n_max))
        + coupling * sparse.kron(Sp(N) + Sm(N), Ad(n_max) + An(n_max))
    )


if __name__ == "__main__":
    print(">>>j_list_gen(5)")
    print(j_list_gen(5))
    print(">>>j_list_gen(6)")
    print(j_list_gen(6))

    print(">>>m_list_gen(Fraction(5, 2))")
    print(m_list_gen(Fraction(5, 2)))
    print(">>>m_list_gen(3)")
    print(m_list_gen(3))

    print(">>>dicke_basis(3)")
    print(dicke_basis(3))
    print(">>>dicke_dim(10) == dicke_dim_bak(10)")
    print(dicke_dim(10) == dicke_dim_bak(10))

    print(">>>X")
    print(X.toarray())
    print(">>>Y")
    print(Y.toarray())
    print(">>>Z")
    print(Z.toarray())

    print(">>> check SZ(3)")
    print(
        np.all(
            Sz(3).toarray()
            == np.array(
                [
                    [1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                ]
            )
        )
    )
    print(">>> check SZ(4)")
    print(
        np.all(
            Sz(4).toarray()
            == np.array(
                [
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
    )
    print(">>>np.round(Sp(3).toarray(),3)")
    print(np.round(Sp(3).toarray(), 3))
    print(">>>np.round(Sp(1).toarray(),3)")
    print(np.round(Sp(1).toarray(), 3))

    print(">>>np.round(Sm(3).toarray(),3)")
    print(np.round(Sm(3).toarray(), 3))
    print(">>>np.round(Sm(1).toarray(),3)")
    print(np.round(Sm(1).toarray(), 3))

    print(">>>np.round(Ad(5).toarray(),3)")
    print(np.round(Ad(5).toarray(), 3))

    print(">>>np.round(An(5).toarray(),3)")
    print(np.round(An(5).toarray(), 3))

    print(">>>np.round(num(5).toarray(),3)")
    print(np.round(num(5).toarray(), 3))

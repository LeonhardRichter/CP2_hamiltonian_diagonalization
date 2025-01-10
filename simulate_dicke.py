import pickle
from ast import literal_eval
from datetime import datetime
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from tqdm import tqdm
import timeit

from dicke import (
    dicke_dim,
    dicke_excited,
    dicke_hamiltonian,
    dicke_state,
    dicke_superradiant,
    num,
)
from lanczos import lanczos_evo


def n_max_default(N) -> int:
    return 2 * N


lanczos_iterations_default = 10


def time_step_default(T) -> float:
    0.001
    # return T / 1000


def sim_dicke(
    N: int,
    T: float,
    n: int = 0,
    spin_state_name: str | tuple[int | Fraction] = "excited",
    freq: float = 1.0,
    coup: float = 1.0,
    time_step: float | None = None,
    n_max: int | None = None,
    lanczos_iterations: int | None = None,
    progress_bar: bool = True,
):
    if time_step is None:
        time_step = time_step_default(T)

    if n_max is None:
        n_max = n_max_default(N)

    if lanczos_iterations is None:
        lanczos_iterations = lanczos_iterations_default

    H = dicke_hamiltonian(N, n_max, coupling=coup, frequency=freq)
    photon_state = np.zeros(n_max + 1)
    photon_state[n] = 1
    print(spin_state_name)
    if spin_state_name == "excited":
        spin_state = dicke_excited(N)

    elif spin_state_name == "superradiant":
        spin_state = dicke_superradiant(N)

    elif type(spin_state_name) is tuple and len(spin_state_name) == 2:
        spin_state = dicke_state(spin_state_name)

    assert spin_state.shape[0] == dicke_dim(
        N
    ), "Dimension of given spin state does not match the number of Dicke atoms"

    state = np.kron(spin_state, photon_state)

    id_o_num = sparse.kron(sparse.eye_array(dicke_dim(N)), num(n_max))
    start = timeit.default_timer()
    tt, vt, et = lanczos_evo(
        H,
        state,
        observables=(id_o_num,),
        dim=lanczos_iterations + 1,
        T=T,
        dt=time_step,
        return_final=False,
        save_states=False,
        progress_bar=progress_bar,
    )
    stop = timeit.default_timer()
    return tt, vt, et, stop - start


if __name__ == "__main__":
    skip_selection = (
        input("Skip parameter selection and run immidietly? [YES/no]: ").strip()
        or "yes"
    )
    if skip_selection.lower() == "no":
        min_N = int(input("min_N [2]: ").strip() or "2")
        max_N = int(input("max_N [20]: ").strip() or "20")
        step_N = int(input("step_N [1]: ").strip() or "1")

        spin_state_name = (
            input("spin_state [EXCITED, superradiant, (j,m)]: ").strip()
            or "excited"
        ).lower()
        if spin_state_name not in {"excited", "superradiant"}:
            spin_state_name = literal_eval(spin_state_name)
            assert (
                type(spin_state_name) is tuple and len(spin_state_name) == 2
            ), "unsuported input type"

        T = float(input("T [6.0]: ").strip() or "6.0")

        now = datetime.now()
        path = (
            input(
                "path/to/data.pickle [./dicke_sim_d_m_Y-H_M_S_spin-state_N-min_N-max_N_T-T.pickle]:"
            ).strip()
            or f"dicke_sim_{now.strftime("%d_%m_%Y-%H_%M_%S")}_{spin_state_name}_N-{min_N}-{max_N}_T-{T}.pickle"
        )
        show_progress_bars = (
            input("show progress bars [yes/NO]: ").strip() or "no"
        )
    else:
        min_N = 2
        max_N = 20
        step_N = 1
        spin_state_name = "excited"
        T = 6.0
        now = datetime.now()
        path = f"dicke_sim_{now.strftime("%d_%m_%Y-%H_%M_%S")}_excited_N-{min_N}-{max_N}_T-{T}.pickle"
        show_progress_bars = "no"

    print(f"""Running simulation with following parameters:
            N_range: range({min_N},{max_N},{step_N})
            T: {T}
            show_progress_bars: {show_progress_bars}
          and saving gathered data to 
            path: {path}
          """)
    if skip_selection.lower() == "no":
        proceed = input("Proceed? [YES,no]: ")
        if proceed.lower() == "no":
            exit()

    NN = range(min_N, max_N + 1)

    data = list()
    for N in tqdm(NN[::-1]):
        tt, vt, et, duration = sim_dicke(
            N=N,
            T=T,
            spin_state_name=spin_state_name,
            progress_bar={"yes": True, "no": False}.get(
                show_progress_bars, False
            ),
        )
        data.append(
            {
                "N": N,
                "spin_state": spin_state_name,
                "duration": duration,
                "tt": tt,
                "vt": vt,
                "et": et,
            }
        )

    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

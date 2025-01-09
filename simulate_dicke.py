import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import sparse

from dicke import (
    dicke_dim,
    dicke_excited,
    dicke_hamiltonian,
    dicke_superradiant,
    num,
)
from lanczos import lanczos_evo
import pickle
from datetime import datetime
from tqdm import tqdm


def n_max(N):
    return 2 * N


lanczos_iterations = 10


def time_step(T):
    return T / 1000


def sim_dicke(
    N: int,
    T: float,
    n: int = 0,
    freq: float = 1.0,
    coup: float = 1.0,
    progress_bar: bool = True,
):
    H = dicke_hamiltonian(N, n_max(N), coupling=coup, frequency=freq)
    photon_state = np.zeros(n_max(N) + 1)
    photon_state[n] = 1
    spin_state = dicke_excited(N)
    state = np.kron(spin_state, photon_state)

    id_o_num = sparse.kron(sparse.eye_array(dicke_dim(N)), num(n_max(N)))

    tt, vt, et = lanczos_evo(
        H,
        state,
        observables=(id_o_num,),
        dim=lanczos_iterations + 1,
        T=T,
        dt=time_step(T),
        return_final=False,
        save_states=False,
        progress_bar=progress_bar,
    )
    return tt, vt, et


if __name__ == "__main__":
    skip_selection = (
        input("Skip parameter selection and run immidietly? [YES/no]: ").strip()
        or "yes"
    )
    if skip_selection.lower() == "no":
        min_N = int(input("min_N [2]: ").strip() or "2")
        max_N = int(input("max_N [20]: ").strip() or "20")
        step_N = int(input("step_N [1]: ").strip() or "1")

        T = float(input("T [6.0]: ").strip() or "6.0")

        now = datetime.now()
        path = (
            input(
                "path/to/data.pickle [./dicke_sim_d_m_Y-H_M_S_N-min_N-max_N_T-T.pickle]:"
            ).strip()
            or f"dicke_sim_{now.strftime("%d_%m_%Y-%H_%M_%S")}_N-{min_N}-{max_N}_T-{T}.pickle"
        )
        show_progress_bars = (
            input("show progress bars [yes/NO]: ").strip() or "no"
        )
    else:
        min_N = 2
        max_N = 20
        step_N = 1
        T = 6.0
        now = datetime.now()
        path = f"dicke_sim_{now.strftime("%d_%m_%Y-%H_%M_%S")}_N-{min_N}-{max_N}_T-{T}.pickle"
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
    for N in tqdm(NN):
        tt, vt, et = sim_dicke(
            N,
            T,
            progress_bar={"yes": True, "no": False}.get(
                show_progress_bars, False
            ),
        )
        data.append({"N": N, "tt": tt, "vt": vt, "et": et})

    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

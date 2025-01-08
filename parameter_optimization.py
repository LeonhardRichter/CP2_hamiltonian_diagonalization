# script for determining suitable parameters for which the Lanczos algorithm enables to compute the time evolution efficiently.
# parameters are the bosonic cut off number, the number of Lanczos iterations and the size of the time-step for computing the evoultion

# Author: Leonhard Richter
# Date: 05.01.2025


import pickle
import timeit

import numpy as np

from dicke import dicke_excited, dicke_hamiltonian, fock
from lanczos import lanczos_evo as evo
import datetime

N = 10
coupling = 0.1
frequency = 1

cut_off_range = [3 * N, 5 / 2 * N, 2 * N, 3 / 2 * N, N, 1 / 2 * N]
cut_off_range = [int(x) for x in cut_off_range]

iterations_range = [100, 80, 60, 40, 20, 10, 8, 6, 4, 2, 1]
time_step_range = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]

data = dict()

path = input("path/to/data.pickle:")

now = datetime.datetime.now()

if path == "":
    path = f"./parameter_optimization{now.strftime("%d_%m_%Y-%H_%M_%S")}.pickle"


def test_parameters(parameters: tuple):
    cut_off, iterations, time_step = parameters

    H = dicke_hamiltonian(
        N=N, n_max=cut_off, coupling=coupling, frequency=frequency
    )
    v = np.kron(dicke_excited(N), fock(0, cut_off))

    start = timeit.default_timer()
    tt, vt, obs = evo(H, v, dim=iterations + 1, T=1.0, dt=time_step)
    stop = timeit.default_timer()
    duration = stop - start
    return vt, duration


if True:
    data["test cut_off"] = dict()
    iterations = iterations_range[0]
    time_step = time_step_range[0]

    for cut_off in cut_off_range:
        parameters = (cut_off, iterations, time_step)
        data["test cut_off"][parameters] = dict()

        vt, duration = test_parameters(parameters)

        data["test cut_off"][parameters]["final vector"] = vt
        data["test cut_off"][parameters]["duration"] = duration

with open("parameters_optimization_data.pickle", "wb") as file:
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

if True:
    cut_off = cut_off_range[0]
    time_step = time_step_range[0]
    data["test iterations"] = dict()

    for iterations in iterations_range:
        parameters = (cut_off, iterations, time_step)
        data["test iterations"][parameters] = dict()

        vt, duration = test_parameters(parameters)

        data["test iterations"][parameters]["final vector"] = vt
        data["test iterations"][parameters]["duration"] = duration

with open("parameters_optimization_data.pickle", "wb") as file:
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

if True:
    data["test time_step"] = dict()
    cut_off = cut_off_range[0]
    iterations = iterations_range[0]

    for time_step in time_step_range:
        parameters = (cut_off, iterations, time_step)
        data["test time_step"][parameters] = dict()

        vt, duration = test_parameters(parameters)

        data["test time_step"][parameters]["final vector"] = vt
        data["test time_step"][parameters]["duration"] = duration

with open("parameters_optimization_data.pickle", "wb") as file:
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

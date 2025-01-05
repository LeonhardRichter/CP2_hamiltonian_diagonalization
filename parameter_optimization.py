# script for determining suitable parameters for which the Lanczos algorithm enables to compute the time evolution efficiently.
# parameters are the bosonic cut off number, the number of Lanczos iterations and the size of the time-step for computing the evoultion

# Author: Leonhard Richter
# Date: 05.01.2025


import numpy as np

from lanczos import lanczos_evo as evo

from dicke import dicke_hamiltonian, dicke_excited, fock

N = 10
coupling = 0.1
frequency = 1

cut_off_0 = 3 * N
iterations = 100
time_step = 0.0001

H = dicke_hamiltonian(
    N=N, n_max=cut_off_0, coupling=coupling, frequency=frequency
)
v = np.kron(dicke_excited(N), fock(0, cut_off_0))

t, v, obs = evo(H, v, dim=iterations + 1, T=1.0, dt=time_step)

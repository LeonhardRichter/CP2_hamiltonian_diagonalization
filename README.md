# CP2_hamiltonian_diagonalization

Repository for implementations of the Lanczos algorithm approximating hermitian matrices by a lower-dimensional tri-diagonalized one.
The main implementations of the algorithm are made in `lanczos.py`.
There, the main algorithm as well as its application to time evolution operators are implemented.

The algorithm is then used to simulate the evolution photon number expectation value for the Dicke model with the fully excited and superradient initial states in `simulate_dicke.py`.
Defintions regarding the Dicke model are made in `dicke.py` and gathered data can be found in sequential form in `data/`.
Just `pickle.load` the files to obtain the raw data.
The analysis of the obtained data is done in 'analysis_photon_count.py'.

This project is part of the course CP2: Advanced projects in computational physics at FAU in the winter term of 2024/2025.

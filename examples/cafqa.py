import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, "../")))

from clapton.clapton import claptonize
from clapton.ansatzes import *
import numpy as np

# define Hamiltonian, e.g. 3q Heisenberg model with random coefficients
paulis = ["XXI", "IXX", "YYI", "IYY", "ZZI", "IZZ"]
coeffs = np.random.random(len(paulis))

from clapton.depolarization import GateGeneralDepolarizationModel

# let's add a noise model where we specify global 1q and 2q gate errors
nm = GateGeneralDepolarizationModel(p1=0.005, p2=0.02)

vqe_pcirc = circular_ansatz_mirrored(N=len(paulis[0]), reps=2, fix_2q=True)
vqe_pcirc.add_depolarization_model(nm)

# we can perform nCAFQA by using the main optimization function "claptonize"
# now with the noisy circuit
# this is slower, as the noisy circuit needs to be sampled 
ks_best, energy_noisy, energy_noiseless = claptonize(
    paulis,
    coeffs,
    vqe_pcirc,
    n_proc=4,           # total number of processes in parallel
    n_starts=4,         # number of random genetic algorithm starts in parallel
    n_rounds=1,         # number of budget rounds, if None it will terminate itself
    callback=print,     # callback for internal parameter (#iteration, energies, ks) processing
    budget=20           # budget per genetic algorithm instance
)

# differrence
print(f"Difference between Noisy and Noiseless calculation : {np.abs(energy_noisy-energy_noiseless)}")

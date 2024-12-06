import sys
import os
import numpy as np
import stim 

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, "../")))

from clapton.clapton import claptonize
from clapton.ansatzes import *
from clapton.depolarization import GateGeneralDepolarizationModel

np.random.seed(0)

# Define Hamiltonian, e.g. 3q Heisenberg model with random coefficients
paulis = ["XXI", "IXX", "YYI", "IYY", "ZZI", "IZZ"]
coeffs = np.random.random(len(paulis))

def circuit_to_tableau(circuit: stim.Circuit) -> stim.Tableau:
    s = stim.TableauSimulator()
    s.do_circuit(circuit)
    return s.current_inverse_tableau() ** -1

# nm = GateGeneralDepolarizationModel(p1=0.005, p2=0.02)
nm = None
pauli_twirl = True

assert not pauli_twirl or nm is not None, "Depolarization model must be defined if Pauli Twirling is applied"

if pauli_twirl:
    init_ansatz = circular_ansatz_mirrored(N=len(paulis[0]), reps=1, fix_2q=True)
    #Pauli Twirl the circuit
    vqe_pcirc = init_ansatz.add_pauli_twirl()
    pauli_twirl_list = [init_ansatz.add_pauli_twirl() for _ in range(100)]
    vqe_pcirc.add_pauli_twirl_list(pauli_twirl_list)

    #Ensure Twirled Circuits are logically equal to Original Ansatz
    for i, circuit in enumerate(pauli_twirl_list):
        assert circuit_to_tableau(vqe_pcirc.stim_circuit()) == circuit_to_tableau(circuit.stim_circuit()), \
            f"Circuit Mismatch at index {i}"

    vqe_pcirc.add_depolarization_model(nm)
    pauli_twirl_list = [circuit.add_depolarization_model(nm) for circuit in pauli_twirl_list]
else:
    vqe_pcirc = circular_ansatz_mirrored(N=len(paulis[0]), reps=1, fix_2q=True)
    vqe_pcirc.add_depolarization_model(nm)

# Perform nCAFQA using the main optimization function "claptonize" with the noisy circuit
ks_best, energy_noisy, energy_noiseless = claptonize(
    paulis,
    coeffs,
    vqe_pcirc,
    n_proc=4,           # Total number of processes in parallel
    n_starts=4,         # Number of random genetic algorithm starts in parallel
    n_rounds=1,         # Number of budget rounds, if None it will terminate itself
    callback=print,     # Callback for internal parameter (#iteration, energies, ks) processing
    budget=20,          # Budget per genetic algorithm instance
)

# Print the energies
print(f"Energy Noisy: {energy_noisy}")
print(f"Energy Noiseless: {energy_noiseless}")

# Difference
print(f"Difference between Noisy and Noiseless calculation: {np.abs(energy_noisy - energy_noiseless)}")

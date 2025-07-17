import numpy as np


def n_to_dits(n, dims):
    """
    n = ... + x_3 a_2 a_1 a_0 + x_2 a_1 a_0 + x_1 a_0 + x_0
    dims = [..., a_3, a_2, a_1, a_0]
    x = [..., x_3, x_2, x_1, x_0], 0 <= x_i < a_i
    return x
    """
    assert n <= np.prod(dims) - 1, "n cannot be represented in this basis"
    x = np.zeros(len(dims), dtype=int)
    i = 0
    while n > 0:
        x[-1-i] = n % dims[-1-i]
        n = n // dims[-1-i]
        i += 1
    return x

def dits_to_n(x, dims):
    """
    n = ... + x_3 a_2 a_1 a_0 + x_2 a_1 a_0 + x_1 a_0 + x_0
    dims = [..., a_3, a_2, a_1, a_0]
    x = [..., x_3, x_2, x_1, x_0], 0 <= x_i < a_i
    return n
    """
    assert len(x) <= len(dims), "x and dims do not have compatible dimensions"
    if len(x) < len(dims):
        x = np.concatenate(np.zeros(len(dims) - len(x)), x)
    return np.cumprod(dims[-1:0:-1]) @ x[-2::-1] + x[-1]


class Results:
    def __init__(self, name):
        self.name = name
        
        self.hamiltonian_strings = []
        self.paulis = []
        self.pauli_coeffs = []
        self.bond_lengths = []
        
        self.times = []
        self.ks_bests = []
        self.energy_bests = []
        
    def add_result(self, hamiltonian_string, pauli, pauli_coeff, time, ks_best, energy_best):
        self.hamiltonian_strings += [hamiltonian_string]
        self.paulis += [pauli]
        self.pauli_coeffs += [pauli_coeff]
        self.bond_lengths += [float(hamiltonian_string.split(" ")[-1])]
        
        self.times += [time]
        self.ks_bests += [ks_best]
        self.energy_bests += [energy_best]
        
    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"  Hamiltonian: {self.hamiltonian_strings}\n"
            f"  Paulis: {self.paulis}\n"
            f"  Pauli Coefficients: {self.pauli_coeffs}\n"
            f"  Optimization History:\n"
            f"    Time: {self.times}\n"
            f"    Best ks: {self.ks_bests if self.ks_bests else 'N/A'}\n"
            f"    Best Energy: {self.energy_bests}" if self.energy_bests else "    Best Energy: N/A"
        )
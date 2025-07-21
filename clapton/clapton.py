import numpy as np
import pygad
import multiprocessing as mp
from clapton.clifford import ParametrizedCliffordCircuit
from clapton.evaluation import transform_paulis, get_energy, weighted_relative_pauli_weight
from clapton.utils import n_to_dits
from clapton.mp_helpers import SignalHandler

from gurobipy import Model, GRB, quicksum
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler, CmaEsSampler, QMCSampler, GPSampler
from optuna.pruners import MedianPruner
import logging

optuna.logging.set_verbosity(logging.WARNING)  # or logging.ERROR to suppress even more

import concurrent
import random

import random
from typing import List, Tuple, Optional

import copy

### Clapton
def loss_func(
        x: list[int], 
        paulis: list[str], 
        coeffs: list[float], 
        vqe_pcirc: ParametrizedCliffordCircuit, 
        trans_pcirc: ParametrizedCliffordCircuit | None = None, 
        alpha: float | None = None, 
        return_sublosses: bool = False, 
        **energy_kwargs
    ):
    if trans_pcirc is None:
        vqe_pcirc.assign(x)
        vqe_pcirc.snapshot()
        vqe_pcirc.snapshot_noiseless()
        energy = get_energy(
                    vqe_pcirc, 
                    paulis, 
                    coeffs, 
                    **energy_kwargs
                    )
        energy_noiseless = get_energy(
                            vqe_pcirc, 
                            paulis, 
                            coeffs, 
                            get_noiseless=True,
                            **energy_kwargs
                            )
        pauli_weight_loss = 0.
        loss = energy + energy_noiseless
    else:
        trans_circ = trans_pcirc.assign(x).stim_circuit()
        paulis_trans, signs = transform_paulis(trans_circ, paulis)
        coeffs_trans = np.multiply(signs, coeffs)
        # assume vqe_pcirc has stim circuit snapshot with all 0 parameters
        energy = get_energy(
                    vqe_pcirc, 
                    paulis_trans, 
                    coeffs_trans, 
                    **energy_kwargs
                    )
        energy_noiseless = get_energy(
                            vqe_pcirc, 
                            paulis_trans, 
                            coeffs_trans, 
                            get_noiseless=True,
                            **energy_kwargs
                            )
        if alpha is not None:
            pauli_weight_loss = alpha * weighted_relative_pauli_weight(paulis_trans, np.abs(coeffs))
        else:
            pauli_weight_loss = 0.
        loss = energy + energy_noiseless + pauli_weight_loss
    if return_sublosses:
        return loss, energy, energy_noiseless, pauli_weight_loss
    else:
        return loss
    

def eval_xs_terms(
        xs: list[list[int]], 
        paulis: list[str], 
        coeffs: list[float],
        vqe_pcirc: ParametrizedCliffordCircuit,
        trans_pcirc: ParametrizedCliffordCircuit | None = None,
        p_start_idx: int = 0, 
        p_end_idx: int | None = None, 
        result_queue = None,
        result_id: int | None = None,
        **loss_kwargs
    ):
    S = len(xs)
    P = len(paulis)
    if p_end_idx is None:
        p_end_idx = P - 1
    idx1 = p_start_idx
    idx2 = P - 1
    partial_losses = []
    for s in range(S-1):
        partial_losses.append(loss_func(
            xs[s], 
            paulis[idx1:idx2+1], 
            coeffs[idx1:idx2+1], 
            vqe_pcirc,
            trans_pcirc,
            **loss_kwargs
            ))
        idx1 = 0
    idx2 = p_end_idx
    partial_losses.append(loss_func(
            xs[S-1], 
            paulis[idx1:idx2+1], 
            coeffs[idx1:idx2+1], 
            vqe_pcirc,
            trans_pcirc,
            **loss_kwargs
            ))
    if result_queue is None:
        return partial_losses
    else:
        result_queue.put((result_id, partial_losses))


def handle_out_data(
        x: list[int], 
        losses: list[float], 
        out_data: list | None = None, 
        callback = None
    ):
        if out_data is not None:
            out_data[0] += 1
            out_data[1] = losses
            out_data[2] = x
            if callback is not None:
                callback(out_data)


def eval_xs_terms_mp(
        xs: list[list[int]], 
        paulis: list[str], 
        coeffs: list[float],
        vqe_pcirc: ParametrizedCliffordCircuit,
        trans_pcirc: ParametrizedCliffordCircuit | None = None,
        n_proc: int = 1, 
        out_data: list | None = None, 
        callback = None, 
        **loss_kwargs
    ):
    S = len(xs)
    P = len(paulis)
    SP = S * P
    ntasks_per_P = int(np.ceil(SP / n_proc))
    sp_start_idc = [n_to_dits(c*ntasks_per_P, [S, P]) for c in range(n_proc)] # c is process / core idx
    sp_end_idc = [n_to_dits((c+1)*ntasks_per_P-1, [S, P]) for c in range(n_proc-1)]
    sp_end_idc.append(np.array([S-1, P-1], dtype=int))
    processes = []
    result_queue = mp.Manager().Queue()
    loss_kwargs["return_sublosses"] = True
    # start n_proc - 1 other subprocesses
    for i in range(1, n_proc):
        process = mp.Process(
            target=eval_xs_terms,
            args=(
                xs[sp_start_idc[i][0]:sp_end_idc[i][0]+1],
                paulis,
                coeffs,
                vqe_pcirc,
                trans_pcirc,
                sp_start_idc[i][1],
                sp_end_idc[i][1],
                result_queue,
                i
            ),
            kwargs=loss_kwargs
        )
        processes.append(process)
        process.start()

    partial_losses = eval_xs_terms(
        xs[sp_start_idc[0][0]:sp_end_idc[0][0]+1],
        paulis,
        coeffs,
        vqe_pcirc,
        trans_pcirc,
        sp_start_idc[0][1],
        sp_end_idc[0][1],
        **loss_kwargs
    )
    losses = np.zeros((S, len(partial_losses[0])))
    ss = range(sp_start_idc[0][0], sp_end_idc[0][0]+1)
    for s_idx, s in enumerate(ss):
        losses[s] += partial_losses[s_idx]

    # Block until others finished
    for process in processes:
        process.join()

    while not result_queue.empty():
        item = result_queue.get()
        i = item[0]
        partial_losses = item[1]
        ss = range(sp_start_idc[i][0], sp_end_idc[i][0]+1)
        for s_idx, s in enumerate(ss):
            losses[s] += partial_losses[s_idx]
    best_idx = np.argmin(losses[:,0])
    handle_out_data(xs[best_idx], losses[best_idx], out_data, callback)
    return losses[:,0]


def claptonize(
        paulis: list[str],
        coeffs: list[float],
        vqe_pcirc: ParametrizedCliffordCircuit,
        trans_pcirc: ParametrizedCliffordCircuit | None = None,
        n_proc: int = 10,
        n_starts: int = 10,
        n_rounds: int | None =None,
        n_retry_rounds: int = 0,
        return_n_rounds: bool = False,
        mix_best_pop_frac: float = 0.2,
        **optimizer_and_loss_kwargs
    ):
    sig_handler = SignalHandler()

    assert vqe_pcirc.num_physical_qubits == len(paulis[0])
    if trans_pcirc is not None:
        assert trans_pcirc.num_physical_qubits == len(paulis[0])
        # take snapshot for more efficient sim in cost function (is initialized to params all 0)
        vqe_pcirc.snapshot()
        vqe_pcirc.snapshot_noiseless()
    
    n_proc = n_proc // n_starts
    if n_proc == 0:
        n_proc = 1
    initial_populations = [None] * n_starts
    out_data = [-1, [np.inf]*3, None]
    optimizer_and_loss_kwargs["n_proc"] = n_proc
    optimizer_and_loss_kwargs["return_best_pop_frac"] = mix_best_pop_frac
    optimizer_and_loss_kwargs["out_data"] = out_data
    
    r_idx = 0
    r_idx_last_change = 0
    last_best_energy_ideal = np.inf
    while True:
        print(f"STARTING ROUND {r_idx}\n\n")
        # start parallelization
        master_processes = []
        master_queue = mp.Manager().Queue()
        # start n_starts - 1 other master processes
        for m in range(1, n_starts):
            optimizer_and_loss_kwargs["initial_population"] = initial_populations[m]
            master_process = mp.Process(
                                target=genetic_algorithm, 
                                args=(
                                    paulis,
                                    coeffs,
                                    vqe_pcirc,
                                    trans_pcirc,
                                    master_queue,
                                    m
                                ),
                                kwargs=optimizer_and_loss_kwargs)
            master_processes.append(master_process)
            master_process.start()

        # this is also a master process
        optimizer_and_loss_kwargs["initial_population"] = initial_populations[0]
        xs, losses = genetic_algorithm(
            paulis,
            coeffs,
            vqe_pcirc,
            trans_pcirc,
            **optimizer_and_loss_kwargs
        )
        best_count = len(xs)

        # wait until others are finished
        for master_process in master_processes:
            master_process.join()
        # fetch others
        while not master_queue.empty():
            item = master_queue.get()
            xs = np.vstack((xs, item[1]))
            losses = np.concatenate((losses, item[2]))
        num_xs = xs.shape[0]
        assert num_xs == n_starts * best_count
        
        # create new initial populations for next round
        rand_shuffled_idc = np.random.choice(range(num_xs), size=num_xs, replace=False)
        for i in range(n_starts):
            idc = rand_shuffled_idc[i*best_count:(i+1)*best_count]
            initial_populations[i] = xs[idc]

        best_idx = np.argmin(losses)
        x_best = xs[best_idx]

        _, energy_noisy, energy_ideal, _ = loss_func(
                                                x_best, 
                                                paulis, 
                                                coeffs, 
                                                vqe_pcirc, 
                                                trans_pcirc,
                                                alpha=optimizer_and_loss_kwargs.get("alpha"),
                                                return_sublosses=True
                                                )

        if n_rounds is None:
            if energy_ideal < last_best_energy_ideal:
                r_idx_last_change = r_idx
                last_best_energy_ideal = energy_ideal
                r_idx += 1
            else:
                if r_idx == r_idx_last_change + 1 + n_retry_rounds:
                    # no change within n_retry_rounds
                    r_idx += 1
                    break
                else:
                    r_idx += 1
        else:
            r_idx += 1
            if r_idx == n_rounds:
                break
    
    sig_handler.restore_handlers()
    if return_n_rounds:
        return list(x_best), energy_noisy, energy_ideal, r_idx
    else:
        return list(x_best), energy_noisy, energy_ideal


### Solvers
def genetic_algorithm(
        paulis: list[str], 
        coeffs: list[str],
        vqe_pcirc: ParametrizedCliffordCircuit,
        trans_pcirc: ParametrizedCliffordCircuit | None,
        master_queue = None,
        master_id: int | None = None,
        n_proc: int = 1,
        out_data: list | None = None,
        callback = None,
        budget: int = 100,
        population_size: int = 100,
        return_best_pop_frac: int = 0.2,
        initial_population: np.ndarray = None,
        init_no_2qb: bool = True,
        keep_elitism: bool = None,
        num_parents_mating: int = None,
        parent_selection_type: str = "tournament", #"sss"
        keep_parents: int = -1,
        crossover_type: str = "single_point",
        crossover_probability: float = 0.9,
        mutation_type: str = "adaptive",
        mutation_probability: tuple[float, float] =(0.25, 0.01), #(0.25, 0.05)  
        **loss_kwargs
    ):
    print(f"started GA at id {master_id} with {n_proc} procs\n")
    if trans_pcirc is None:
        gene_space = vqe_pcirc.parameter_space()
        idc_param_2qb = vqe_pcirc.idc_param_2qb()
    else:
        gene_space = trans_pcirc.parameter_space()
        idc_param_2qb = trans_pcirc.idc_param_2qb()
    num_params = len(gene_space)
    num_generations = budget
    num_genes = num_params
    if keep_elitism is None:
        keep_elitism = population_size // 10
    if num_parents_mating is None:
        num_parents_mating = 2 * population_size // 10
    best_count = int(population_size * return_best_pop_frac)

    def fitness_func(ga_instance, solutions, solutions_idc):
        return -eval_xs_terms_mp(
            solutions, 
            paulis,
            coeffs,
            vqe_pcirc,
            trans_pcirc,
            n_proc,
            out_data,
            callback,
            **loss_kwargs
            )
    ga_instance = pygad.GA(
                    num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitness_func,
                    sol_per_pop=population_size,
                    num_genes=num_genes,
                    parent_selection_type=parent_selection_type,
                    keep_parents=keep_parents,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    gene_space=gene_space,
                    gene_type=[int]*num_params,
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    keep_elitism=keep_elitism,
                    fitness_batch_size=population_size
                    )
    if initial_population is not None:
        initial_population = np.asarray(initial_population)
        assert len(initial_population.shape) == 2
        assert initial_population.shape[1] == num_params
        num_fixed_pops = initial_population.shape[0]
        ga_instance.initial_population[:num_fixed_pops] = initial_population[:population_size]
        ga_instance.population[:num_fixed_pops] = initial_population[:population_size].copy()
    else:
        if init_no_2qb:
            ga_instance.initial_population[:,idc_param_2qb] = 0 
            ga_instance.population[:,idc_param_2qb] = 0 
    
    ga_instance.run()
    last_losses = -ga_instance.last_generation_fitness
    best_idc = np.argsort(last_losses)[:best_count]
    best_losses = last_losses[best_idc]
    best_xs = ga_instance.population[best_idc,:]

    if master_queue is None:
        return best_xs, best_losses
    else:
        master_queue.put((master_id, best_xs, best_losses))
        
def claptonize_opt(
        paulis: list[str],
        coeffs: list[float],
        vqe_pcirc: ParametrizedCliffordCircuit,
        trans_pcirc: ParametrizedCliffordCircuit | None = None,
        **optimizer_and_loss_kwargs
    ):
    assert vqe_pcirc.num_physical_qubits == len(paulis[0])
    if trans_pcirc is not None:
        assert trans_pcirc.num_physical_qubits == len(paulis[0])
        # take snapshot for more efficient sim in cost function (is initialized to params all 0)
        vqe_pcirc.snapshot()
        vqe_pcirc.snapshot_noiseless()
        
    if trans_pcirc is None:
        gene_space = vqe_pcirc.parameter_space()
        idc_param_2qb = vqe_pcirc.idc_param_2qb()
    else:
        gene_space = trans_pcirc.parameter_space()
        idc_param_2qb = trans_pcirc.idc_param_2qb()
    
    def objective(trial):
    # Assume gene_space is a list of lists or ranges, e.g. [[0,1,2], [0,1], [0,1,2,3], ...]
        x_sol = [
            trial.suggest_int(f"x{i}", min(xs_), max(xs_))
            for i, xs_ in enumerate(gene_space)
        ]

        energy = loss_func(
            x_sol, paulis, coeffs, vqe_pcirc,
            trans_pcirc, alpha=optimizer_and_loss_kwargs.get("alpha"),
            return_sublosses=False
        )
        return energy

    min_trials = optimizer_and_loss_kwargs.get("min_trials", 100)
    early_stopping_trials = optimizer_and_loss_kwargs.get("early_stopping_trials", 100)
    sampler = optimizer_and_loss_kwargs.get("sampler", TPESampler)
    n_proc = optimizer_and_loss_kwargs.get("n_proc", 1)

    study = run_study_with_custom_stopping(objective, min_trials, early_stopping_trials, sampler, "minimize", n_proc)
    
    x_best = [study.best_params[f"x{i}"] for i in range(len(gene_space))]

    _, energy_noisy, energy_ideal, _ = loss_func(
                                            x_best, 
                                            paulis, 
                                            coeffs, 
                                            vqe_pcirc, 
                                            trans_pcirc,
                                            alpha=optimizer_and_loss_kwargs.get("alpha"),
                                            return_sublosses=True
                                            )

    return x_best, energy_noisy, energy_ideal


def run_study_with_custom_stopping(objective, min_trials=100, early_stopping_trials=100, sampler=TPESampler, direction="minimize", n_proc=20):
    sampler_ = sampler(seed=42)
    study = optuna.create_study(direction=direction, sampler=sampler_)
    print(f"Sampler is {study.sampler.__class__.__name__}")
    
    best_value = None
    no_improvement_counter = 0

    for i in range(1000000000000000):  # max limit
        study.optimize(objective, n_trials=1, catch=(Exception,), n_jobs=n_proc)

        trial_value = study.best_value
        if best_value is None or trial_value < best_value:
            best_value = trial_value
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Check custom stopping condition
        if i + 1 >= min_trials and no_improvement_counter >= early_stopping_trials:
            # print(f"Early stopping triggered: no improvement in last {early_stopping_trials} trials")
            break

    return study




from scipy.optimize import minimize
import numpy as np

def claptonize_opt_scipy(
    paulis: list[str],
    coeffs: list[float],
    vqe_pcirc: ParametrizedCliffordCircuit,
    trans_pcirc: ParametrizedCliffordCircuit | None = None,
    **optimizer_and_loss_kwargs
):
    assert vqe_pcirc.num_physical_qubits == len(paulis[0])
    if trans_pcirc is not None:
        assert trans_pcirc.num_physical_qubits == len(paulis[0])
        vqe_pcirc.snapshot()
        vqe_pcirc.snapshot_noiseless()

    # Parameter space and mapping
    if trans_pcirc is None:
        gene_space = vqe_pcirc.parameter_space()
        idc_param_2qb = vqe_pcirc.idc_param_2qb()
    else:
        gene_space = trans_pcirc.parameter_space()
        idc_param_2qb = trans_pcirc.idc_param_2qb()

    # Discrete search space -> convert to bounds and index mapping
    bounds = [(min(xs_), max(xs_)) for xs_ in gene_space]
    
    def objective(x):
        # Round to nearest valid discrete value per gene
        x_rounded = [int(round(xi)) for xi in x]
        # Clip to ensure within discrete value ranges
        x_clipped = [min(max(xi, min(gene_space[i])), max(gene_space[i])) for i, xi in enumerate(x_rounded)]
        
        energy = loss_func(
            x_clipped, paulis, coeffs, vqe_pcirc,
            trans_pcirc, alpha=optimizer_and_loss_kwargs.get("alpha"),
            return_sublosses=False
        )
        return energy

    # Initial guess: midpoint of each gene space
    # x0 = [np.mean(xs_) for xs_ in gene_space]
    x0 = [1 for xs_ in gene_space]
    
    # Optimization using Nelder-Mead
    res = minimize(
        objective,
        x0=x0,
        # method='Nelder-Mead',
        method=optimizer_and_loss_kwargs.get("optimizer", 'L-BFGS-B'),
        options={
            'maxiter': optimizer_and_loss_kwargs.get("maxiter", 100000),
            'xatol': 1e-3,
            'fatol': 1e-4,
            'disp': True
        },
        bounds=bounds,
    )

    # Final best solution
    x_best = [int(round(xi)) for xi in res.x]
    print(x_best)
    x_best = [min(max(xi, min(gene_space[i])), max(gene_space[i])) for i, xi in enumerate(x_best)]
    print(x_best)

    _, energy_noisy, energy_ideal, _ = loss_func(
        x_best,
        paulis,
        coeffs,
        vqe_pcirc,
        trans_pcirc,
        alpha=optimizer_and_loss_kwargs.get("alpha"),
        return_sublosses=True
    )

    return x_best, energy_noisy, energy_ideal


import nevergrad as ng

def claptonize_opt_ng(
    paulis: list[str],
    coeffs: list[float],
    vqe_pcirc: ParametrizedCliffordCircuit,
    trans_pcirc: ParametrizedCliffordCircuit | None = None,
    **optimizer_and_loss_kwargs
):
    assert vqe_pcirc.num_physical_qubits == len(paulis[0])
    if trans_pcirc is not None:
        assert trans_pcirc.num_physical_qubits == len(paulis[0])
        vqe_pcirc.snapshot()
        vqe_pcirc.snapshot_noiseless()

    # Get gene space
    gene_space = (
        trans_pcirc.parameter_space() if trans_pcirc else vqe_pcirc.parameter_space()
    )
    idc_param_2qb = (
        trans_pcirc.idc_param_2qb() if trans_pcirc else vqe_pcirc.idc_param_2qb()
    )

    # Correct way: unpack each discrete var
    param_space = ng.p.Instrumentation(
        *[ng.p.Choice(xs_) for xs_ in gene_space]
    )
    
    # print(param_space)

    def objective(*x):
        energy = loss_func(
            list(x), paulis, coeffs, vqe_pcirc,
            trans_pcirc, alpha=optimizer_and_loss_kwargs.get("alpha"),
            return_sublosses=False
        )
        return energy

    optimizer_cls = optimizer_and_loss_kwargs.get("ng_optimizer", ng.optimizers.NelderMead)
    budget = optimizer_and_loss_kwargs.get("budget", 200)

    worker_count = optimizer_and_loss_kwargs.get("n_proc", 16)
    print(f"Using {worker_count} workers for optimization")
    optimizer = optimizer_cls(parametrization=param_space, budget=budget, num_workers=worker_count)
    with concurrent.futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(objective, executor=executor, batch_mode=True)
    x_best = list(recommendation.value)[0]
    
    print(x_best)

    _, energy_noisy, energy_ideal, _ = loss_func(
        x_best, paulis, coeffs, vqe_pcirc, trans_pcirc,
        alpha=optimizer_and_loss_kwargs.get("alpha"),
        return_sublosses=True
    )

    return x_best, energy_noisy, energy_ideal

    # with concurrent.futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    #     futures = {}
    #     for _ in range(optimizer.budget):
    #         candidate = optimizer.ask()
    #         print(candidate.value[0])
    #         future = executor.submit(objective, candidate.value[0])
    #         futures[future] = candidate

    #     for future in concurrent.futures.as_completed(futures):
    #         result = future.result()
    #         optimizer.tell(futures[future], result)
    # recommendation = optimizer.provide_recommendation()
    
    
    
from typing import List, Tuple

def pauli_support(pauli_string: str) -> list[int]:
    """Return indices of non-identity terms in a Pauli string."""
    return [i for i, p in enumerate(pauli_string) if p != 'I']


def commutes(p1: str, p2: str) -> bool:
    """Check if two Pauli strings commute (up to a global phase)."""
    # Commutation rule: count number of positions with anti-commuting terms
    anti_commute_pairs = {('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y')}
    count = 0
    for a, b in zip(p1, p2):
        if (a, b) in anti_commute_pairs:
            count += 1
    return count % 2 == 0

def group_commuting_paulis(paulis: List[str], coeffs: List[float]) -> List[List[Tuple[str, float]]]:
    """Group Pauli strings into sets of mutually commuting terms."""
    from collections import deque

    unused = deque(zip(paulis, coeffs))
    groups = []

    while unused:
        base_p, base_c = unused.popleft()
        group = [(base_p, base_c)]
        to_remove = []

        for i in range(len(unused)):
            pi, ci = unused[i]
            if all(commutes(pi, pj) for pj, _ in group):
                group.append((pi, ci))
                to_remove.append(i)

        # Remove in reverse order to avoid reindexing
        for i in reversed(to_remove):
            del unused[i]

        groups.append(group)

    return groups






import nevergrad as ng
import concurrent.futures
from collections import defaultdict

def claptonize_opt_structured(
    paulis: list[str],
    coeffs: list[float],
    vqe_pcirc: ParametrizedCliffordCircuit,
    trans_pcirc: ParametrizedCliffordCircuit | None = None,
    **optimizer_and_loss_kwargs
):
    assert vqe_pcirc.num_physical_qubits == len(paulis[0])
    if trans_pcirc is not None:
        assert trans_pcirc.num_physical_qubits == len(paulis[0])
        vqe_pcirc.snapshot()
        vqe_pcirc.snapshot_noiseless()

    # Structural insight 1: Group Pauli terms by commutativity or support
    pauli_groups = group_commuting_paulis(paulis, coeffs)  # [[(pauli, coeff), ...], ...]

    # Structural insight 2: Prioritize parameters affecting high-weight terms
    qubit_weights = defaultdict(float)
    for p, c in zip(paulis, coeffs):
        for q in pauli_support(p):  # Return indices where p[q] ≠ 'I'
            qubit_weights[q] += abs(c)

    # Get gene space and param-to-qubit mapping
    gene_space = trans_pcirc.parameter_space() if trans_pcirc else vqe_pcirc.parameter_space()
    idc_param_2qb = trans_pcirc.idc_param_2qb() if trans_pcirc else vqe_pcirc.idc_param_2qb()
    
    print(idc_param_2qb)

    # Sort parameters to prioritize high-weight qubits
    def param_priority(param_index):
        # qubits = idc_param_2qb.get(param_index, [])
        qubits = idc_param_2qb[param_index]
        return -sum(qubit_weights[q] for q in qubits)

    sorted_indices = sorted(range(len(gene_space)), key=param_priority)
    gene_space = [gene_space[i] for i in sorted_indices]
    reverse_map = [sorted_indices.index(i) for i in range(len(sorted_indices))]

    param_space = ng.p.Instrumentation(*[ng.p.Choice(xs_) for xs_ in gene_space])

    def objective(*x_raw):
        x = [x_raw[0][i] for i in reverse_map]
        energy = loss_func(
            x, paulis, coeffs, vqe_pcirc,
            trans_pcirc, alpha=optimizer_and_loss_kwargs.get("alpha"),
            return_sublosses=False
        )
        return energy

    optimizer_cls = optimizer_and_loss_kwargs.get("ng_optimizer", ng.optimizers.NGOpt)
    budget = optimizer_and_loss_kwargs.get("budget", 200)
    worker_count = optimizer_and_loss_kwargs.get("n_proc", 16)
    print(f"Using {worker_count} workers with Hamiltonian structure guidance")

    optimizer = optimizer_cls(parametrization=param_space, budget=budget, num_workers=worker_count)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        recommendation = optimizer.minimize(objective, executor=executor, batch_mode=True)

    x_best_sorted = list(recommendation.value)[0]
    x_best = [x_best_sorted[i] for i in reverse_map]

    _, energy_noisy, energy_ideal, _ = loss_func(
        x_best, paulis, coeffs, vqe_pcirc, trans_pcirc,
        alpha=optimizer_and_loss_kwargs.get("alpha"),
        return_sublosses=True
    )

    return x_best, energy_noisy, energy_ideal

# # Ensure loss_func is imported or defined in your module
# def claptonize_opt_tableau_search(
#     paulis: List[str],
#     coeffs: List[float],
#     vqe_pcirc: ParametrizedCliffordCircuit,
#     trans_pcirc: Optional[ParametrizedCliffordCircuit] = None,
#     alpha: Optional[float] = None,
#     max_iters: int = 100,
#     random_seed: int = 0,
#     verbose: bool = False
# ) -> Tuple[List[int], float, float]:
#     """
#     Greedy local search in the space of stabilizer tableaus by modifying one Clifford generator at a time.

#     Args:
#         paulis: List of Pauli strings (e.g., ['XZI', 'YYI', ...]).
#         coeffs: Corresponding coefficients for each Pauli term.
#         vqe_pcirc: The parameterized Clifford circuit for VQE.
#         trans_pcirc: Optional transformed circuit (for noise mitigation).
#         alpha: Regularization or penalty weight (if used in loss_func).
#         max_iters: Maximum number of local-search iterations.
#         random_seed: Seed for reproducible initialization.
#         verbose: If True, prints progress at each improvement.

#     Returns:
#         best_params: Parameter vector for the optimal tableau.
#         energy_noisy: Expectation value of H under noisy simulation.
#         energy_ideal: Expectation value under ideal (noiseless) simulation.
#     """
#     # Take snapshots if using a transformed circuit
#     if trans_pcirc is not None:
#         vqe_pcirc.snapshot()
#         trans_pcirc.snapshot()
#     else:
#         vqe_pcirc.snapshot()

#     # Initialize RNG
#     rng = random.Random(random_seed)

#     # Gene space: list of lists of possible generator indices
#     gene_space = trans_pcirc.parameter_space() if trans_pcirc else vqe_pcirc.parameter_space()

#     # Random initial parameter vector
#     best_params = [rng.choice(choices) for choices in gene_space]

#     # Evaluate initial energy
#     best_energy = loss_func(
#         best_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
#         alpha=alpha, return_sublosses=False
#     )
    
#     final_params = copy.deepcopy(best_params)
#     final_energy = copy.deepcopy(best_energy)
    
#     it = 0
#     while it < max_iters:
#     # Greedy local search: try changing one parameter at a time
#     # for it in range(max_iters):
#         improved = False
#         # Explore neighbors in parameter space (i.e., tableau neighbors)
#         for idx, choices in enumerate(gene_space):
#             current = best_params[idx]
#             for choice in choices:
#                 if choice == current:
#                     continue
#                 candidate = best_params.copy()
#                 candidate[idx] = choice
#                 energy = loss_func(
#                     candidate, paulis, coeffs, vqe_pcirc, trans_pcirc,
#                     alpha=alpha, return_sublosses=False
#                 )
#                 if energy < best_energy:
#                     best_energy = energy
#                     best_params = candidate
#                     improved = True
#                     if verbose:
#                         print(f"Iter {it}: param {idx} -> {choice}, energy = {energy:.6f}")
#                     break  # accept first-improvement
#             if improved:
#                 break
#         if not improved:
#             # if verbose:
#             #     print(f"No improvement at iteration {it}, stopping search.")
#             # break
#             if best_energy < final_energy:
#                 final_params = copy.deepcopy(best_params)
#                 final_energy = copy.deepcopy(best_energy)
            
#             best_params = [rng.choice(choices) for choices in gene_space]
#             best_energy = loss_func(
#                 best_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
#                 alpha=alpha, return_sublosses=False
#             )
            
#         it += 1

#     # Final evaluation with sublosses
#     _, energy_noisy, energy_ideal, _ = loss_func(
#         final_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
#         alpha=alpha, return_sublosses=True
#     )

#     # Return optimal tableau parameters and energies
#     return best_params, energy_noisy, energy_ideal


import copy
import random
import math
from typing import List, Optional, Tuple

def claptonize_opt_tableau_search_____(
    paulis: List[str],
    coeffs: List[float],
    vqe_pcirc: ParametrizedCliffordCircuit,
    trans_pcirc: Optional[ParametrizedCliffordCircuit] = None,
    alpha: Optional[float] = None,
    max_iters: int = 100,
    random_seed: int = 0,
    verbose: bool = False
) -> Tuple[List[int], float, float]:
    """
    Greedy local search with unique random restarts in the space of stabilizer tableaus.
    """
    # --- snapshot circuits ---
    if trans_pcirc is not None:
        vqe_pcirc.snapshot()
        trans_pcirc.snapshot()
    else:
        vqe_pcirc.snapshot()

    # --- prepare RNG and gene space ---
    rng = random.Random(random_seed)
    gene_space = trans_pcirc.parameter_space() if trans_pcirc else vqe_pcirc.parameter_space()
    # how many total initial vectors are possible?
    total_combinations = math.prod(len(choices) for choices in gene_space)

    # --- visited set of tuples ---
    visited = set()

    def sample_new(max_tries: int = 1000) -> Optional[List[int]]:
        """Try up to max_tries to sample a fresh initial vector."""
        for _ in range(max_tries):
            candidate = tuple(rng.choice(choices) for choices in gene_space)
            if candidate not in visited:
                visited.add(candidate)
                return list(candidate)
        return None  # gave up

    # --- first random initialization ---
    best_params = sample_new()
    if best_params is None:
        raise RuntimeError("Cannot sample initial params—search space too small or max_tries exceeded.")
    best_energy = loss_func(best_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
                            alpha=alpha, return_sublosses=False)

    # track the best over all restarts
    final_params = copy.deepcopy(best_params)
    final_energy = best_energy

    it = 0
    while it < max_iters:
        is_visited = False
        improved = False

        # greedy local search around best_params
        for idx, choices in enumerate(gene_space):
            current = best_params[idx]
            for choice in choices:
                if choice == current:
                    continue
                candidate = best_params.copy()
                candidate[idx] = choice
                if tuple(candidate) in visited:
                    is_visited = True
                    continue
                visited.add(tuple(candidate))
                energy = loss_func(candidate, paulis, coeffs, vqe_pcirc, trans_pcirc,
                                   alpha=alpha, return_sublosses=False)
                if energy < best_energy:
                    best_energy = energy
                    best_params = candidate
                    improved = True
                    if verbose:
                        print(f"Iter {it}: param[{idx}] → {choice}, energy = {energy:.6f}")
                    break
            # if improved:
            #     break
            if is_visited or improved:
                break

        if not improved:
            # record global best
            if best_energy < final_energy:
                final_energy = best_energy
                final_params = copy.deepcopy(best_params)

            # if we’ve tried every possible init, stop
            if len(visited) >= total_combinations:
                if verbose:
                    print("All unique restarts exhausted; terminating early.")
                break

            # get a fresh random restart
            new_init = sample_new()
            if new_init is None:
                if verbose:
                    print("Failed to sample fresh init; terminating early.")
                break
            best_params = new_init
            best_energy = loss_func(best_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
                                    alpha=alpha, return_sublosses=False)

        it += 1

    # final evaluation (with sublosses) on the best overall parameters
    _, energy_noisy, energy_ideal, _ = loss_func(
        final_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
        alpha=alpha, return_sublosses=True
    )

    return final_params, energy_noisy, energy_ideal


import copy
import random
import math
from typing import List, Optional, Tuple

def claptonize_opt_tableau_search(
    paulis: List[str],
    coeffs: List[float],
    vqe_pcirc: ParametrizedCliffordCircuit,
    trans_pcirc: Optional[ParametrizedCliffordCircuit] = None,
    alpha: Optional[float] = None,
    max_iters: int = 100,
    random_seed: int = 0,
    verbose: bool = False
) -> Tuple[List[int], float, float]:
    """
    Greedy local search with unique random restarts in the space of stabilizer tableaus.
    """
    # --- snapshot circuits ---
    if trans_pcirc is not None:
        vqe_pcirc.snapshot()
        trans_pcirc.snapshot()
    else:
        vqe_pcirc.snapshot()

    # --- prepare RNG and gene space ---
    rng = random.Random(random_seed)
    gene_space = trans_pcirc.parameter_space() if trans_pcirc else vqe_pcirc.parameter_space()
    # how many total initial vectors are possible?
    total_combinations = math.prod(len(choices) for choices in gene_space)

    # --- visited set of tuples ---
    visited = set()

    def sample_new(max_tries: int = 1000) -> Optional[List[int]]:
        """Try up to max_tries to sample a fresh initial vector."""
        for _ in range(max_tries):
            candidate = tuple(rng.choice(choices) for choices in gene_space)
            if candidate not in visited:
                visited.add(candidate)
                return list(candidate)
        return None  # gave up

    # --- first random initialization ---
    best_params = sample_new()
    if best_params is None:
        raise RuntimeError("Cannot sample initial params—search space too small or max_tries exceeded.")
    best_energy = loss_func(best_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
                            alpha=alpha, return_sublosses=False)

    # track the best over all restarts
    final_params = copy.deepcopy(best_params)
    final_energy = best_energy

    it = 0
    while it < max_iters:
        improved = False

        # greedy local search around best_params
        for idx, choices in enumerate(gene_space):
            current = best_params[idx]
            for choice in choices:
                if choice == current:
                    continue
                candidate = best_params.copy()
                candidate[idx] = choice
                energy = loss_func(candidate, paulis, coeffs, vqe_pcirc, trans_pcirc,
                                   alpha=alpha, return_sublosses=False)
                if energy < best_energy:
                    best_energy = energy
                    best_params = candidate
                    improved = True
                    if verbose:
                        print(f"Iter {it}: param[{idx}] → {choice}, energy = {energy:.6f}")
                    break
            if improved:
                break

        if not improved:
            # record global best
            if best_energy < final_energy:
                final_energy = best_energy
                final_params = copy.deepcopy(best_params)

            # if we’ve tried every possible init, stop
            if len(visited) >= total_combinations:
                if verbose:
                    print("All unique restarts exhausted; terminating early.")
                break

            # get a fresh random restart
            new_init = sample_new()
            if new_init is None:
                if verbose:
                    print("Failed to sample fresh init; terminating early.")
                break
            best_params = new_init
            best_energy = loss_func(best_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
                                    alpha=alpha, return_sublosses=False)

        it += 1

    # final evaluation (with sublosses) on the best overall parameters
    _, energy_noisy, energy_ideal, _ = loss_func(
        final_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
        alpha=alpha, return_sublosses=True
    )

    return final_params, energy_noisy, energy_ideal



import random
import copy
import math
from typing import List, Optional, Tuple

def claptonize_opt_tableau_search_vns(
    paulis: List[str],
    coeffs: List[float],
    vqe_pcirc: ParametrizedCliffordCircuit,
    trans_pcirc: Optional[ParametrizedCliffordCircuit] = None,
    alpha: Optional[float] = None,
    max_iters: int = 100,
    random_seed: int = 0,
    verbose: bool = False
) -> Tuple[List[int], float, float]:
    """
    Variable Neighborhood Search in the space of stabilizer tableaus.
    Returns (best_params, energy_noisy, energy_ideal).
    """
    # --- snapshot circuits ---
    if trans_pcirc is not None:
        vqe_pcirc.snapshot()
        trans_pcirc.snapshot()
    else:
        vqe_pcirc.snapshot()

    rng = random.Random(random_seed)
    gene_space = trans_pcirc.parameter_space() if trans_pcirc else vqe_pcirc.parameter_space()
    num_params = len(gene_space)

    # --- helper: sample a random initial solution ---
    def sample_initial() -> List[int]:
        return [rng.choice(choices) for choices in gene_space]

    # --- greedy one‐parameter local descent ---
    def local_search(init: List[int]) -> Tuple[List[int], float]:
        current = init[:]
        current_energy = loss_func(current, paulis, coeffs, vqe_pcirc, trans_pcirc,
                                   alpha=alpha, return_sublosses=False)
        improved = True
        while improved:
            improved = False
            for i, choices in enumerate(gene_space):
                base = current[i]
                for choice in choices:
                    if choice == base:
                        continue
                    trial = current.copy()
                    trial[i] = choice
                    e = loss_func(trial, paulis, coeffs, vqe_pcirc, trans_pcirc,
                                  alpha=alpha, return_sublosses=False)
                    if e < current_energy:
                        current_energy = e
                        current = trial
                        improved = True
                        if verbose:
                            print(f"  ↓ local: idx={i} → {choice}, energy={e:.6f}")
                        break
                if improved:
                    break
        return current, current_energy

    # --- shake: random move in k‐neighborhood ---
    def shake(sol: List[int], k: int) -> List[int]:
        idxs = rng.sample(range(num_params), k)
        nbr = sol[:]
        for i in idxs:
            choices = gene_space[i]
            # pick a different random choice
            nbr[i] = rng.choice([c for c in choices if c != nbr[i]])
        return nbr

    # set up neighborhood sizes: 1,2,... up to half the params (or full)
    max_k = min(num_params, 3)  # you can tune max_k
    neighborhood_sizes = list(range(1, max_k + 1))

    # --- initialize ---
    best_params = sample_initial()
    best_params, best_energy = local_search(best_params)
    if verbose:
        print(f"Init best energy = {best_energy:.6f}")
    final_params, final_energy = best_params, best_energy

    it = 0
    while it < max_iters:
        improved = False

        for k in neighborhood_sizes:
            # shaking
            candidate = shake(best_params, k)
            if verbose:
                print(f"Iter {it}, shake k={k}: starting local search from {candidate}")
            # local descent
            candidate, candidate_energy = local_search(candidate)

            if candidate_energy < best_energy:
                if verbose:
                    print(f"  ↑ improved! k={k}, energy {best_energy:.6f} → {candidate_energy:.6f}")
                best_params, best_energy = candidate, candidate_energy
                final_params, final_energy = best_params, best_energy
                improved = True
                break  # restart at k=1

        if not improved:
            # no improvement across all neighborhoods → fresh random restart
            if verbose:
                print(f"Iter {it}: no improvement in any neighborhood, random restart")
            fresh = sample_initial()
            best_params, best_energy = local_search(fresh)
            if verbose:
                print(f"  restart best energy = {best_energy:.6f}")

        it += 1

    # final evaluation with sublosses
    _, energy_noisy, energy_ideal, _ = loss_func(
        final_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
        alpha=alpha, return_sublosses=True
    )
    return final_params, energy_noisy, energy_ideal


import random
import copy
import math
from typing import List, Optional, Tuple

def claptonize_opt_tableau_search_lns(
    paulis: List[str],
    coeffs: List[float],
    vqe_pcirc: ParametrizedCliffordCircuit,
    trans_pcirc: Optional[ParametrizedCliffordCircuit] = None,
    alpha: Optional[float] = None,
    max_iters: int = 100,
    destroy_size: Optional[int] = None,
    destroy_fraction: float = 0.2,
    random_seed: int = 0,
    verbose: bool = False
) -> Tuple[List[int], float, float]:
    """
    Large-Neighborhood Search in the space of stabilizer tableaus:
      1. Start from a random solution.
      2. Repeat for max_iters:
         a) Destroy: randomly reassign a subset of parameters.
         b) Repair: run greedy local search to a local optimum.
         c) Accept if improved.
      3. Return best-found params and both noisy/ideal energies.
    """
    # --- snapshot circuits ---
    if trans_pcirc is not None:
        vqe_pcirc.snapshot()
        trans_pcirc.snapshot()
    else:
        vqe_pcirc.snapshot()

    # --- prepare RNG and gene space ---
    rng = random.Random(random_seed)
    gene_space = trans_pcirc.parameter_space() if trans_pcirc else vqe_pcirc.parameter_space()
    num_vars = len(gene_space)

    # --- helper: greedy local search from a given start ---
    def greedy_search(start_params: List[int], start_energy: float) -> Tuple[List[int], float]:
        params = start_params.copy()
        energy = start_energy
        improved = True

        while improved:
            improved = False
            enumerator = list(enumerate(gene_space))
            random.shuffle(enumerator)
            
            for i, choices in enumerator:
                current = params[i]
                for choice in choices:
                    if choice == current:
                        continue
                    candidate = params.copy()
                    candidate[i] = choice
                    e = loss_func(candidate, paulis, coeffs, vqe_pcirc,
                                  trans_pcirc, alpha=alpha, return_sublosses=False)
                    if e < energy:
                        energy = e
                        params = candidate
                        improved = True
                        if verbose:
                            print(f"  Greedy → idx {i} = {choice}, energy {e:.6f}")
                        break
                if improved:
                    break
        return params, energy

    # --- initialize with one random sample ---
    best_params = [rng.choice(choices) for choices in gene_space]
    best_energy = loss_func(best_params, paulis, coeffs, vqe_pcirc,
                            trans_pcirc, alpha=alpha, return_sublosses=False)

    # --- determine destroy block size ---
    if destroy_size is None:
        # fraction of total variables, at least 1
        k = max(1, int(num_vars * destroy_fraction))
    else:
        k = max(1, min(num_vars, destroy_size))

    # --- main LNS loop ---
    for it in range(1, max_iters + 1):
        # a) Destroy: pick k random positions to re-randomize
        destroy_idxs = rng.sample(range(num_vars), k)
        candidate = best_params.copy()
        for idx in destroy_idxs:
            choices = gene_space[idx]
            # ensure a change
            new_val = rng.choice([c for c in choices if c != candidate[idx]])
            candidate[idx] = new_val

        if verbose:
            print(f"Iter {it:3d}: destroyed {destroy_idxs}, start greedy repair")

        # b) Repair: greedy local search
        repaired_params, repaired_energy = greedy_search(
            candidate,
            loss_func(candidate, paulis, coeffs, vqe_pcirc,
                      trans_pcirc, alpha=alpha, return_sublosses=False)
        )

        # c) Accept if improved
        if repaired_energy < best_energy:
            best_params, best_energy = repaired_params, repaired_energy
            if verbose:
                print(f"→ New best at iter {it}: energy = {best_energy:.6f}")

    # --- final eval with sublosses ---
    _, energy_noisy, energy_ideal, _ = loss_func(
        best_params, paulis, coeffs, vqe_pcirc, trans_pcirc,
        alpha=alpha, return_sublosses=True
    )
    return best_params, energy_noisy, energy_ideal

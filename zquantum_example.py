import numpy as np
from qequlacs.simulator import QulacsSimulator
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.cost_function import AnsatzBasedCostFunction
from zquantum.core.graph import load_graph, generate_graph_from_specs
from zquantum.core.interfaces.optimizer import optimization_result
from zquantum.optimizers.scipy_optimizer import ScipyOptimizer
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.problems.maxcut import get_maxcut_hamiltonian
from zquantum.qaoa.problems.stable_set import get_stable_set_hamiltonian
from qequlacs import QulacsSimulator
import time
import os

# https://arxiv.org/pdf/1812.04170.pdf


def load_graph_from_file(graph_size, graph_id):
    return load_graph(
        os.path.join(
            "graphs", f"size_{graph_size}", f"graph_{graph_size}_{graph_id}.json"
        )
    )


def get_problem_hamiltonian(graph, problem="maxcut"):
    if problem == "maxcut":
        return get_maxcut_hamiltonian(graph)
    if problem == "stable_set":
        return get_stable_set_hamiltonian(graph)


def get_graph(size_of_graph, graph_id, graph_specs=None):
    if graph_specs is None:
        return load_graph_from_file(size_of_graph, graph_id)
    graph_specs["num_nodes"] = size_of_graph
    return generate_graph_from_specs(graph_specs)


def find_appropriate_params(cost_hamiltonian, ansatz, initial_params, mode):

    backend = QulacsSimulator()
    optimizer = ScipyOptimizer(method="L-BFGS-B")
    estimation_method = calculate_exact_expectation_values
    ####################
    # Modification suggestions
    ####################
    # from qeqiskit import QiskitSimulator()
    # backend = QiskitSimulator()

    # from zquantum.optimizers.basin_hopping import BasinHoppingOptimizer
    # optimizer = BasinHoppingOptimizer()
    # from zquantum.optimizers.cma_es_optimizer import CMAESOptimizer
    # optimizer = CMAESOptimizer(sigma_0=0.8)

    # from zquantum.qaoa.estimators import CvarEstimator
    # estimation_method = CvarEstimator(alpha=0.8)

    if mode == "low":
        cost_function = AnsatzBasedCostFunction(
            target_operator=-cost_hamiltonian,
            ansatz=ansatz,
            backend=backend,
            estimation_method=estimation_method,
        )
    else:
        cost_function = AnsatzBasedCostFunction(
            target_operator=cost_hamiltonian,
            ansatz=ansatz,
            backend=backend,
            estimation_method=estimation_method,
        )

    if mode in {"random", "evaluate"}:
        opt_results = optimization_result(
            opt_value=cost_function(initial_params),
            opt_params=initial_params,
        )
    else:
        opt_results = optimizer.minimize(cost_function, initial_params)

    if mode == "low":
        opt_results.opt_value = -opt_results.opt_value
    return opt_results


def main():
    ####################
    # 1. Simulation parameters
    ####################
    size_of_graph = 10
    size_of_big_graph = 20
    number_of_graphs = 25
    min_layers = 2
    max_layers = 7
    modes = ["high", "random", "low"]
    problem = "maxcut"
    # problem = "stable_set"
    graph_specs = None
    # graph_specs = {"type_graph": "regular", "degree": 3}
    # graph_specs = {"type_graph": "regular", "degree": 4}
    # graph_specs = {"type_graph": "erdos_renyi", "probability": 0.6}
    # graph_specs = {"type_graph": "erdos_renyi", "probability": 0.8}

    start_time = time.time()
    ####################
    # 2. Layers loop
    ####################
    for number_of_layers in range(min_layers, max_layers + 1):
        ####################
        # 3. Problem definition
        ####################
        print("Number of layers:", number_of_layers)
        graph_id = 0
        graph = get_graph(size_of_graph, graph_id, graph_specs)
        cost_hamiltonian = get_problem_hamiltonian(graph, problem)
        ansatz = QAOAFarhiAnsatz(
            number_of_layers=number_of_layers, cost_hamiltonian=cost_hamiltonian
        )

        ####################
        # 4. Finding parameters
        ####################
        selected_params = {}
        for mode in modes:
            number_of_params = ansatz.number_of_params
            initial_params = np.random.uniform(-np.pi, np.pi, number_of_params)
            opt_results = find_appropriate_params(
                cost_hamiltonian, ansatz, initial_params, mode
            )
            selected_params[mode] = opt_results.opt_params

        ####################
        # 5. Evaluating parameters
        ####################
        for evaluation_size in [size_of_graph, size_of_big_graph]:
            print("Graph size: ", evaluation_size)
            values = {mode: [] for mode in modes}
            for graph_id in range(number_of_graphs):
                ####################
                # 5.1 Problem definition
                ####################

                graph = load_graph_from_file(evaluation_size, graph_id)
                cost_hamiltonian = get_problem_hamiltonian(graph, problem)
                ansatz = QAOAFarhiAnsatz(
                    number_of_layers=number_of_layers, cost_hamiltonian=cost_hamiltonian
                )
                ####################
                # 5.2 Execution
                ####################
                for mode in modes:
                    params = selected_params[mode]
                    opt_results = find_appropriate_params(
                        cost_hamiltonian, ansatz, params, "evaluate"
                    )
                    values[mode].append(opt_results.opt_value)

            ####################
            # 6. Data presentation
            ####################
            for mode in modes:
                print(mode, np.mean(values[mode]), np.std(values[mode]))

    print("Total time:", time.time() - start_time)


if __name__ == "__main__":
    main()
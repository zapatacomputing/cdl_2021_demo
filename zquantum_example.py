import numpy as np
from qequlacs.simulator import QulacsSimulator
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.cost_function import AnsatzBasedCostFunction
from zquantum.core.graph import generate_random_regular_graph, save_graph, load_graph
from zquantum.core.interfaces.optimizer import optimization_result
from zquantum.core.serialization import save_optimization_results
from zquantum.optimizers.scipy_optimizer import ScipyOptimizer
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.problems.maxcut import get_maxcut_hamiltonian
from qequlacs import QulacsSimulator
import time
from datetime import datetime
import os

# https://arxiv.org/pdf/1812.04170.pdf


def load_graph_from_file(graph_size, graph_id):
    return load_graph(
        os.path.join(
            "graphs", f"size_{graph_size}", f"graph_{graph_size}_{graph_id}.json"
        )
    )


def main():
    size_of_graph = 6
    number_of_graphs = 25
    number_of_optimization_runs = 10
    min_layers = 2
    max_layers = 7
    start_time = time.time()
    dir_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    mode = "good"
    mode = "random"
    # mode = "bad"
    os.mkdir(dir_name)

    for graph_id in range(number_of_graphs):
        print("Graph:", graph_id)
        # graph = generate_random_regular_graph(size_of_graph, 3)
        # save_graph(graph, os.path.join(dir_name, f"graph_{graph_id}.json"))
        graph = load_graph_from_file(size_of_graph, graph_id)
        cost_hamiltonian = get_maxcut_hamiltonian(graph)
        for number_of_layers in range(min_layers, max_layers + 1):
            print("Number of layers:", number_of_layers)
            ansatz = QAOAFarhiAnsatz(
                number_of_layers=number_of_layers, cost_hamiltonian=cost_hamiltonian
            )
            number_of_params = number_of_layers * 2

            for opt_run_id in range(number_of_optimization_runs):
                backend = QulacsSimulator()
                optimizer = ScipyOptimizer(method="L-BFGS-B")
                initial_params = np.random.uniform(-np.pi, np.pi, number_of_params)
                if mode in {"good", "random"}:
                    cost_function = AnsatzBasedCostFunction(
                        target_operator=cost_hamiltonian,
                        ansatz=ansatz,
                        backend=backend,
                        estimation_method=calculate_exact_expectation_values,
                    )
                elif mode == "bad":
                    cost_function = AnsatzBasedCostFunction(
                        target_operator=-cost_hamiltonian,
                        ansatz=ansatz,
                        backend=backend,
                        estimation_method=calculate_exact_expectation_values,
                    )
                if mode == "random":
                    opt_results = optimization_result(
                        opt_value=cost_function(initial_params),
                        opt_params=initial_params,
                    )
                else:
                    print("Optimization:", opt_run_id)
                    opt_results = optimizer.minimize(cost_function, initial_params)
                file_name = f"opt_results_graph_{graph_id}_layers_{number_of_layers}_run_{opt_run_id}_{mode}.json"
                save_optimization_results(
                    opt_results, os.path.join(dir_name, file_name)
                )
    print("Total time:", time.time() - start_time)


if __name__ == "__main__":
    main()
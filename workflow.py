import logging
import os
from typing import List, Optional, Dict
import networkx as nx

import numpy as np
import pandas as pd
from openfermion import QubitOperator
import qe.sdk.v1 as qe
from qequlacs import QulacsSimulator
from zquantum.core.interfaces.optimizer import Optimizer, OptimizeResult

from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.cost_function import AnsatzBasedCostFunction
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.graph import (
    generate_graph_from_specs,
    load_graph,
)
from zquantum.core.interfaces.backend import QuantumBackend
from zquantum.core.interfaces.optimizer import (
    Optimizer,
    optimization_result,
)
from zquantum.optimizers.scipy_optimizer import ScipyOptimizer
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.problems.maxcut import get_maxcut_hamiltonian


def load_graph_from_file(graph_size, graph_id):
    return load_graph(
        os.path.join(
            "graphs", f"size_{graph_size}", f"graph_{graph_size}_{graph_id}.json"
        )
    )


def get_problem_hamiltonian(graph: nx.graph) -> QubitOperator:
    # from zquantum.qaoa.problems.graph_partition import get_graph_partition_hamiltonian
    # return get_graph_partition_hamiltonian(graph)
    # from zquantum.qaoa.problems.stable_set import get_stable_set_hamiltonian
    # return get_stable_set_hamiltonian(graph)
    return get_maxcut_hamiltonian(graph)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def get_graph_step(
    size_of_graph: int, graph_id: int, graph_specs: Optional[Dict]
) -> nx.Graph:
    return get_graph(size_of_graph, graph_id, graph_specs)


def get_graph(
    size_of_graph: int, graph_id: int, graph_specs: Optional[Dict]
) -> nx.Graph:
    if graph_specs is None:
        return load_graph_from_file(size_of_graph, graph_id)
    graph_specs["num_nodes"] = size_of_graph
    return generate_graph_from_specs(graph_specs)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def get_problem_hamiltonian_step(graph: nx.Graph) -> QubitOperator:
    return get_problem_hamiltonian(graph)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def get_ansatz_step(number_of_layers: int, cost_hamiltonian: QubitOperator) -> Ansatz:
    return get_ansatz(
        number_of_layers=number_of_layers, cost_hamiltonian=cost_hamiltonian
    )


def get_ansatz(number_of_layers: int, cost_hamiltonian: QubitOperator) -> Ansatz:
    return QAOAFarhiAnsatz(
        number_of_layers=number_of_layers, cost_hamiltonian=cost_hamiltonian
    )


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def generate_random_parameters(
    min_value: float, max_value: float, number_of_parameters: int
) -> np.ndarray:
    return np.random.uniform(min_value, max_value, number_of_parameters)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def get_params_from_results(results: OptimizeResult) -> np.ndarray:
    return results.opt_params


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def find_appropriate_params_step(
    cost_hamiltonian: QubitOperator,
    ansatz: Ansatz,
    initial_params: np.ndarray,
    mode: str,
) -> OptimizeResult:
    return find_appropriate_params(cost_hamiltonian, ansatz, initial_params, mode)


def find_appropriate_params(
    cost_hamiltonian: QubitOperator,
    ansatz: Ansatz,
    initial_params: np.ndarray,
    mode: str,
) -> OptimizeResult:
    backend = QulacsSimulator()
    optimizer = ScipyOptimizer(method="L-BFGS-B")

    if mode == "low":
        cost_function = AnsatzBasedCostFunction(
            target_operator=-cost_hamiltonian,
            ansatz=ansatz,
            backend=backend,
            estimation_method=calculate_exact_expectation_values,
        )
    else:
        cost_function = AnsatzBasedCostFunction(
            target_operator=cost_hamiltonian,
            ansatz=ansatz,
            backend=backend,
            estimation_method=calculate_exact_expectation_values,
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


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def evaluate_params(
    graph_size: int,
    number_of_graphs: int,
    graph_specs: Dict,
    number_of_layers: int,
    selected_results: OptimizeResult,
) -> Dict:
    results = []
    for graph_id in range(number_of_graphs):
        ####################
        # 5.1 Problem definition
        ####################

        graph = get_graph(graph_size, graph_id, graph_specs)
        cost_hamiltonian = get_problem_hamiltonian(graph)
        ansatz = get_ansatz(
            number_of_layers=number_of_layers, cost_hamiltonian=cost_hamiltonian
        )
        ####################
        # 5.2 Execution
        ####################
        params = selected_results.opt_params
        opt_results = find_appropriate_params(
            cost_hamiltonian, ansatz, params, "evaluate"
        )
        results.append(opt_results.opt_value)
    return results


@qe.workflow(
    name="qaoa-concentration",
    import_defs=[
        qe.Z.Quantum.Core(branch_name="dev"),
        qe.Z.Quantum.Optimizers(branch_name="dev"),
        qe.Z.Quantum.Qaoa(branch_name="master"),
        qe.QE.Qulacs(branch_name="dev"),
        qe.GitImportDefinition.get_current_repo_and_branch(),
    ],
)
def qaoa_concentration_workflow(
    size_of_graph: int,
    size_of_big_graph: int,
    number_of_graphs: int,
    min_layers: int,
    max_layers: int,
    modes: List[str],
    graph_specs: Optional[Dict],
) -> List[qe.StepDefinition]:

    ####################
    # 2. Layers loop
    ####################
    all_results = []
    for number_of_layers in range(min_layers, max_layers + 1):
        ####################
        # 3. Problem definition
        ####################
        graph_id = 0
        graph = get_graph_step(size_of_graph, graph_id, graph_specs)
        cost_hamiltonian = get_problem_hamiltonian_step(graph)
        ansatz = get_ansatz_step(
            number_of_layers=number_of_layers, cost_hamiltonian=cost_hamiltonian
        )

        ####################
        # 4. Finding parameters
        ####################
        for mode in modes:
            number_of_params = 2 * number_of_layers
            initial_params = generate_random_parameters(-np.pi, np.pi, number_of_params)
            opt_results = find_appropriate_params_step(
                cost_hamiltonian, ansatz, initial_params, mode
            )
            ####################
            # 5. Evaluating parameters
            ####################
            for evaluation_size in [size_of_graph, size_of_big_graph]:
                evaluation_results = evaluate_params(
                    evaluation_size,
                    number_of_graphs,
                    graph_specs,
                    number_of_layers,
                    opt_results,
                )
                all_results.append(evaluation_results)

    return all_results


if __name__ == "__main__":
    ####################
    # 1. Simulation parameters
    ####################
    size_of_graph = 10
    size_of_big_graph = 20
    number_of_graphs = 25
    min_layers = 2
    max_layers = 7
    modes = ["high", "random", "low"]
    # graph_specs = {"type_graph": "regular", "degree": 3}
    # graph_specs = {"type_graph": "regular", "degree": 4}
    graph_specs = {"type_graph": "erdos_renyi", "probability": 0.6}
    # graph_specs = {"type_graph": "erdos_renyi", "probability": 0.8}

    wf: qe.WorkflowDefinition = qaoa_concentration_workflow(
        size_of_graph=size_of_graph,
        size_of_big_graph=size_of_big_graph,
        number_of_graphs=number_of_graphs,
        min_layers=min_layers,
        max_layers=max_layers,
        modes=modes,
        graph_specs=graph_specs,
    )

    # result = wf.local_run(log_level=logging.INFO)
    # import pdb

    # pdb.set_trace()
    wf.validate()
    # wf.print_workflow()
    wf.submit()
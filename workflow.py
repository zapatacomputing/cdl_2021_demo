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
from zquantum.qaoa.problems.maxcut import (
    get_maxcut_hamiltonian as _get_maxcut_hamiltonian,
)


def load_graph_from_file(graph_size, graph_id):
    return load_graph(
        os.path.join(
            "graphs", f"size_{graph_size}", f"graph_{graph_size}_{graph_id}.json"
        )
    )


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
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
def get_maxcut_hamiltonian(graph: nx.Graph) -> QubitOperator:
    return _get_maxcut_hamiltonian(graph)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
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
def analyze_evaluations(
    number_of_layers: int, evaluation_results: List  # Dict
) -> pd.DataFrame:
    df = pd.DataFrame(columns=["n_layers", "mode", "mean", "std"])
    for mode, results_list in evaluation_results.items():
        values = [result.opt_value for result in results_list]
        df = df.append(
            {
                "n_layers": number_of_layers,
                "mode": mode,
                "mean": np.mean(values),
                "std": np.std(values),
            }
        )
    return df


@qe.workflow(
    name="qaoa-opt",
    import_defs=[
        qe.Z.Quantum.Core(branch_name="dev"),
        qe.Z.Quantum.Optimizers(branch_name="dev"),
        qe.Z.Quantum.Qaoa(branch_name="dev"),
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
        graph = get_graph(size_of_graph, graph_id, graph_specs)
        cost_hamiltonian = get_maxcut_hamiltonian(graph)
        ansatz = get_ansatz(
            number_of_layers=number_of_layers, cost_hamiltonian=cost_hamiltonian
        )

        ####################
        # 4. Finding parameters
        ####################
        selected_results = []
        for mode in modes:
            number_of_params = 2 * number_of_layers
            initial_params = generate_random_parameters(-np.pi, np.pi, number_of_params)
            opt_results = find_appropriate_params(
                cost_hamiltonian, ansatz, initial_params, mode
            )
            selected_results.append(opt_results)
        ####################
        # 5. Evaluating parameters
        ####################
        for evaluation_size in [size_of_graph, size_of_big_graph]:
            # execution_results = {mode: [] for mode in modes}
            execution_results = []
            for graph_id in range(number_of_graphs):
                ####################
                # 5.1 Problem definition
                ####################

                graph = get_graph(evaluation_size, graph_id, graph_specs)
                cost_hamiltonian = get_maxcut_hamiltonian(graph)
                ansatz = get_ansatz(
                    number_of_layers=number_of_layers, cost_hamiltonian=cost_hamiltonian
                )
                ####################
                # 5.2 Execution
                ####################
                for mode, results in zip(modes, selected_results):
                    params = get_params_from_results(results)
                    opt_results = find_appropriate_params(
                        cost_hamiltonian, ansatz, params, "evaluate"
                    )
                    # execution_results[mode].append(opt_results)
                    all_results.append(opt_results)

            # ####################
            # # 6. Data presentation
            # ####################
            analysis_results = analyze_evaluations(number_of_layers, execution_results)
            # all_results.append(analysis_results)
    return all_results


if __name__ == "__main__":
    ####################
    # 1. Simulation parameters
    ####################
    size_of_graph = 10
    size_of_big_graph = 20
    number_of_graphs = 1
    min_layers = 2
    max_layers = 2
    modes = ["high", "random", "low"]
    graph_specs = {"type_graph": "regular", "degree": 3}
    # graph_specs = None

    wf: qe.WorkflowDefinition = qaoa_concentration_workflow(
        size_of_graph=size_of_graph,
        size_of_big_graph=size_of_big_graph,
        number_of_graphs=number_of_graphs,
        min_layers=min_layers,
        max_layers=max_layers,
        modes=modes,
        graph_specs=graph_specs,
    )

    result = wf.local_run(log_level=logging.INFO)
    # import pdb

    # pdb.set_trace()
    # wf.validate()
    # wf.submit()
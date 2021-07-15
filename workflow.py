import logging
from typing import Callable, List, Optional
import numpy as np
from openfermion.ops.operators.qubit_operator import QubitOperator
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.backend import QuantumBackend
import qe.sdk.v1 as qe
from qequlacs.simulator import QulacsSimulator
from zquantum.core.interfaces.optimizer import Optimizer, OptimizeResult
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.cost_function import AnsatzBasedCostFunction
from zquantum.core.graph import generate_random_regular_graph
from zquantum.optimizers.scipy_optimizer import ScipyOptimizer
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.problems.maxcut import (
    get_maxcut_hamiltonian as _get_maxcut_hamiltonian,
)
from qequlacs import QulacsSimulator
import networkx as nx
from openfermion import QubitOperator


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def generate_graph(size: int, degree: int) -> nx.Graph:
    return generate_random_regular_graph(size, degree)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def get_maxcut_hamiltonian(graph: nx.Graph) -> QubitOperator:
    # return QubitOperator("2*[Z0 Z1]")
    return _get_maxcut_hamiltonian(graph)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def get_ansatz(n_layers: int, cost_hamiltonian: QubitOperator) -> Ansatz:
    return QAOAFarhiAnsatz(number_of_layers=n_layers, cost_hamiltonian=cost_hamiltonian)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def generate_random_initial_parameters(
    min_value: float, max_value: float, number_of_parameters: int
) -> np.ndarray:
    return np.random.uniform(min_value, max_value, number_of_parameters)


@qe.step(
    resource_def=qe.ResourceDefinition(cpu="1000m", memory="2Gi", disk="10Gi"),
)
def optimize_circuit(
    ansatz: QAOAFarhiAnsatz,
    cost_hamiltonian: QubitOperator,
    backend: QuantumBackend,
    optimizer: Optimizer,
    initial_parameters: np.ndarray,
    estimation_method: Callable,
) -> OptimizeResult:

    cost_function = AnsatzBasedCostFunction(
        target_operator=cost_hamiltonian,
        ansatz=ansatz,
        backend=backend,
        estimation_method=estimation_method,
    )
    return optimizer.minimize(cost_function, initial_parameters)


@qe.workflow(
    name="qaoa-opt",
    import_defs=[
        qe.Z.Quantum.Core(),
        qe.Z.Quantum.Optimizers(),
        qe.Z.Quantum.Qaoa(),
        qe.QE.Qulacs(),
        qe.GitImportDefinition.get_current_repo_and_branch(),
    ],
)
def qaoa_workflow(
    size_of_graph: int,
    number_of_graphs: int,
    number_of_optimization_runs: int,
    backend: QuantumBackend,
    optimizer: Optimizer,
    seed: Optional[int] = None,
) -> List[qe.StepDefinition]:
    assert size_of_graph % 2 == 0
    if seed is not None:
        np.random.seed(seed)

    all_runs = []
    for graph_id in range(number_of_graphs):
        graph = generate_graph(size_of_graph, 3)
        cost_hamiltonian = get_maxcut_hamiltonian(graph)
        ansatz = get_ansatz(2, cost_hamiltonian)
        number_of_params = 4
        all_runs += [graph, cost_hamiltonian, ansatz]
        for _ in range(number_of_optimization_runs):
            all_runs.append(
                optimize_circuit(
                    ansatz,
                    cost_hamiltonian,
                    backend,
                    optimizer,
                    generate_random_initial_parameters(
                        -np.pi / 2, np.pi / 2, number_of_params
                    ),
                    calculate_exact_expectation_values,
                )
            )
    return all_runs


if __name__ == "__main__":
    rng_seed = 9
    # graph = generate_random_regular_graph(4, 3)
    # hamiltonian = _get_maxcut_hamiltonian(graph)
    wf: qe.WorkflowDefinition = qaoa_workflow(
        size_of_graph=4,
        number_of_graphs=2,
        number_of_optimization_runs=2,
        backend=QulacsSimulator(),
        optimizer=ScipyOptimizer(method="L-BFGS-B"),
        seed=rng_seed,
    )

    # result = wf.local_run(log_level=logging.INFO)
    # import pdb

    # pdb.set_trace()
    wf.validate()
    # wf.print_workflow()
    wf.submit()